"""Main ingestion pipeline entrypoint.

Reads the curated corpus, extracts text, extracts claims + verbatim evidence,
embeds them, and writes ONE source-of-truth JSON dataset to data/claims.json.

Run from the repo root:
    python scripts/ingest.py

A future uploaded PDF enters the same path: it is saved under papers/ and
processed by the identical extract_text -> extract_claims -> embed flow.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "scripts" / "corpus.json"
OUTPUT = ROOT / "data" / "claims.json"
PUBLIC_OUTPUT = ROOT / "public" / "data" / "claims.json"
BATCH_SIZE = 10


def _load_env(path: Path) -> None:
    """Load GROQ_API_KEY (and other keys) from .env.local if present.

    Small dependency-free loader so ingestion reads the same env file the
    Next.js runtime uses. Never prints the value.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env(ROOT / ".env.local")

sys.path.insert(0, str(ROOT / "scripts"))

from extract_claims import (
    EXTRACT_SYSTEM_RETRY,
    _REJECTED,
    BatchExtractionError,
    extract_claims_from_block,
    extract_claims_from_blocks_batch,
)  # noqa: E402
from embed import embed_texts  # noqa: E402
from extract_text import extract_text  # noqa: E402

# Constrained prompt used to retry blocks Groq flagged with json_validate_failed.
_RETRY_SYSTEM = EXTRACT_SYSTEM_RETRY

# Total ingestion retries (transport + unusable-output), reported in STATS.
_RETRIES = 0


def main() -> None:
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    embedding_model = corpus.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")

    papers = corpus["papers"]
    documents = []
    for paper in papers:
        src = ROOT / paper["source_file"]
        if not src.exists():
            raise FileNotFoundError(f"Source file missing: {src}")
        documents.append((paper, extract_text(src)))

    # Collect all claim candidates first, then embed in one batch.
    raw_claims = []  # (paper, block, claim)
    stats = {
        "blocks_total": 0,
        "blocks_with_claims": 0,
        "blocks_zero_claims": 0,
        "blocks_permanent_failure": [],
        "retries": 0,
        "accepted": 0,
        "rejected": 0,
        "proposed": 0,
    }
    for paper, doc in documents:
        blocks = doc.blocks
        for start in range(0, len(blocks), BATCH_SIZE):
            batch = blocks[start:start + BATCH_SIZE]
            stats["blocks_total"] += len(batch)
            try:
                batched = extract_claims_from_blocks_batch(batch, _llm)
            except Exception:
                # A difficult batch must not abort the run. Record and continue.
                for block in batch:
                    stats["blocks_permanent_failure"].append(
                        f"{paper['paper_id']}:{block.section}:{block.page}"
                    )
                continue
            block_hits: set[int] = set()
            for bid, claim in batched:
                block = batch[bid]
                raw_claims.append((paper, block, claim))
                block_hits.add(bid)
            stats["blocks_with_claims"] += len(block_hits)
            stats["blocks_zero_claims"] += len(batch) - len(block_hits)
            stats["accepted"] += len(batched)
            stats["rejected"] = _REJECTED
            stats["retries"] = _RETRIES

    stats["proposed"] = stats["accepted"] + stats["rejected"]

    texts_to_embed = [c.claim_text for (_, _, c) in raw_claims]
    print(f"Extracted {len(texts_to_embed)} claims; embedding...")
    embeddings = embed_texts(texts_to_embed)

    claims_out = []
    for (paper, block, claim), embedding in zip(raw_claims, embeddings):
        claims_out.append(
            {
                "claim_id": f"{paper['paper_id']}-{len(claims_out) + 1:03d}",
                "paper_id": paper["paper_id"],
                "paper_title": paper["title"],
                "section": block.section,
                "page": block.page,
                "claim_text": claim.claim_text,
                "evidence_text": claim.evidence_text,
                "source_location": _location(block),
                "embedding": embedding,
                "relations": [],
            }
        )

    dataset = {
        "version": 1,
        "embedding_model": embedding_model,
        "papers": papers,
        "claims": claims_out,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {len(claims_out)} claims to {OUTPUT}")
    print(f"STATS blocks_total={stats['blocks_total']}")
    print(f"STATS blocks_with_claims={stats['blocks_with_claims']}")
    print(f"STATS blocks_zero_claims={stats['blocks_zero_claims']}")
    print(f"STATS blocks_permanent_failure={stats['blocks_permanent_failure']}")
    print(f"STATS retries={stats['retries']}")
    print(f"STATS proposed={stats['proposed']}")
    print(f"STATS accepted={stats['accepted']}")
    print(f"STATS rejected={stats['rejected']}")


def _location(block) -> str:
    if block.page is not None:
        return f"Section {block.section}, p.{block.page}"
    return f"Section {block.section}"


def _is_unusable(content: str) -> bool:
    """True when the model returned no parseable claim blocks at all."""
    return "CLAIM:" not in content and "NO_CLAIMS" not in content


def _llm(system: str, prompt: str) -> str:
    """LLM client used during ingestion (Groq, plain text, no JSON mode).

    Returns the raw model text; extract_claims_from_block parses it with a
    deterministic delimiter parser. No response_format is requested so the model
    is not rejected for JSON-mode failures. Keeps transport retries (429/5xx)
    and retries a block with a stricter prompt if the output is unusable.
    """
    import re
    import time

    import httpx

    global _RETRIES

    api_key = os.environ["GROQ_API_KEY"]
    url = "https://api.groq.com/openai/v1/chat/completions"
    # Ingestion uses a cheap, high-quota model (llama-3.1-8b-instant) rather than
    # the reasoning-heavy runtime model, so the daily budget is not exhausted.
    model = os.environ.get("GROQ_INGEST_MODEL", "llama-3.1-8b-instant")

    retry_system = system
    # Distinguish transport/rate-limit retries (429/5xx) from unusable-output
    # retries. Rate-limit backoff can exceed the generic attempt budget, so give
    # 429 its own allowance that honors the server's retry-after.
    rate_attempts = 8
    output_attempts = 2
    attempt = 0
    while rate_attempts > 0 or output_attempts > 0:
        attempt += 1
        if attempt > 1:
            # Cap backoff so a long outage doesn't hang the run.
            time.sleep(min(2 ** (attempt - 1), 30))
        # Pace requests to respect the per-minute token budget (free tier ~6k/min).
        time.sleep(2.0)
        res = httpx.post(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": retry_system},
                    {"role": "user", "content": prompt},
                ],
                # gpt-oss models consume reasoning tokens that can crowd out the
                # final content; give explicit headroom and low reasoning effort
                # so claim output is not truncated/emptied on larger batches.
                "max_tokens": 4000,
                "reasoning_effort": "low",
            },
            timeout=120,
        )
        if res.status_code == 429:
            # Sleep the retry-after the server specified, then retry. Rate-limit
            # exhaustion can need several seconds, so budget generously.
            body = res.text
            m = re.search(r"try again in ([\d.]+)s", body)
            wait = float(m.group(1)) if m else 2 ** attempt
            time.sleep(min(wait, 60))
            _RETRIES += 1
            rate_attempts -= 1
            continue
        if res.status_code >= 500:
            _RETRIES += 1
            rate_attempts -= 1
            continue
        if res.status_code != 200:
            raise RuntimeError(f"Groq HTTP {res.status_code}: {res.text[:500]}")
        content = res.json()["choices"][0]["message"]["content"]
        if _is_unusable(content):
            # Stricter prompt retry on the same block, then give up for this block.
            retry_system = _RETRY_SYSTEM
            _RETRIES += 1
            output_attempts -= 1
            continue
        return content
    raise ValueError("Ingestion LLM produced unusable output for a block")


def _sha256_bytes(data: bytes) -> str:
    """Stable content hash of the original uploaded PDF bytes (duplicate key)."""
    return hashlib.sha256(data).hexdigest()


def _atomic_write(obj, path: Path) -> None:
    """Write JSON atomically so a failed run never leaves a partial dataset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _publish_dataset() -> None:
    """Publish the canonical dataset to the runtime public asset."""
    if not OUTPUT.exists():
        return
    PUBLIC_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC_OUTPUT.write_bytes(OUTPUT.read_bytes())


def _extract_batch_with_recovery(
    start: int,
    batch: list,
    call_llm,
    notify=None,
) -> tuple[list, bool]:
    """Extract claims from a batch with TARGETED recovery.

    Fast path: one call over the whole batch (unchanged normal throughput). On
    failure, retry ONLY that same batch once (stricter-prompt retry is handled
    inside call_llm), then split ONLY the failed batch into two halves, recursing
    only into failed halves, then fall back to individual blocks. Successful
    work is never reprocessed. Returns ([{global_block_index, RawClaim}], all_ok).
    """
    def norm(res):
        # Local batch indices -> global block indices (start offset).
        return [(start + bid, c) for (bid, c) in res]

    if len(batch) == 1:
        # Last resort: individual-block fallback.
        if notify:
            notify("block")
        try:
            return norm(extract_claims_from_blocks_batch(batch, call_llm)), True
        except Exception:
            return [], False

    # Fast path: whole batch.
    try:
        return norm(extract_claims_from_blocks_batch(batch, call_llm)), True
    except Exception:
        pass

    # Retry the same batch once.
    if notify:
        notify("retry")
    try:
        return norm(extract_claims_from_blocks_batch(batch, call_llm)), True
    except Exception:
        pass

    # Split ONLY this failed batch into two halves; recurse into each half.
    if notify:
        notify("split")
    mid = (len(batch) + 1) // 2
    left_claims, left_ok = _extract_batch_with_recovery(start, batch[:mid], call_llm, notify)
    right_claims, right_ok = _extract_batch_with_recovery(start + mid, batch[mid:], call_llm, notify)
    return left_claims + right_claims, (left_ok and right_ok)


def _detect_paper_title(pdf_path: Path, blocks) -> str:
    """Determine a human-readable paper title without an extra LLM call.

    Preference order:
      1. PDF metadata title (if present and not obviously unusable).
      2. First meaningful line of the first extracted block (title header).
      3. Sanitized filename stem as a last-resort fallback.
    """
    title: str | None = None

    # 1. PDF metadata title.
    if pdf_path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(pdf_path))
            meta = reader.metadata or {}
            meta_title = (meta.get("/Title") or "").strip()
            # Reject auto-generated/placeholder metadata titles.
            if meta_title and not re.match(r"^(untitled|document|scanned)\b", meta_title, re.I):
                title = meta_title
        except Exception:
            title = None

    # 2. First meaningful line of the first block.
    if not title:
        for blk in blocks:
            for line in blk.text.splitlines():
                line = line.strip()
                if line and len(line) >= 3:
                    title = line
                    break
            if title:
                break

    # 3. Sanitized filename stem fallback.
    if not title:
        title = pdf_path.stem

    # Sanitize for safe storage/display but keep it human-readable.
    title = re.sub(r"\s+", " ", title).strip()
    title = title.strip(":;.,- ")
    return title[:300]


def ingest_uploaded_paper(
    pdf_path: Path,
    title: str | None = None,
    progress=None,
) -> dict:
    """Incrementally add ONE uploaded PDF to the existing corpus.

    Reuses extract_text -> batched claim extraction -> verbatim validation ->
    embed_texts, then atomically merges the new paper's claims into the canonical
    data/claims.json without regenerating the pre-fed papers. Returns a status
    dict. Never writes a partial corpus: the merge happens only after all
    extraction + validation + embedding succeed.
    """
    if progress:
        progress("uploaded")

    if not pdf_path.exists():
        return {"status": "failed", "reason": "Uploaded file is missing.", "claims_added": 0}

    raw_bytes = pdf_path.read_bytes()
    file_hash = _sha256_bytes(raw_bytes)

    # Load the canonical dataset (or start fresh if none yet).
    if OUTPUT.exists():
        dataset = json.loads(OUTPUT.read_text(encoding="utf-8"))
    else:
        dataset = {"version": 1, "embedding_model": "sentence-transformers/all-MiniLM-L6-v2", "papers": [], "claims": []}

    # Duplicate detection by SHA-256 of the original PDF bytes.
    for p in dataset.get("papers", []):
        if p.get("source_hash") == file_hash:
            return {"status": "already_exists", "paper_id": p.get("paper_id"), "claims_added": 0}

    if progress:
        progress("extracting_text")
    try:
        doc = extract_text(pdf_path)
    except Exception as e:
        return {"status": "failed", "reason": f"PDF extraction failed: {e}", "claims_added": 0}
    blocks = doc.blocks
    if not blocks:
        return {"status": "failed", "reason": "No extractable text found.", "claims_added": 0}

    # Batched claim extraction with TARGETED per-batch recovery.
    raw_claims: list[tuple[int, object]] = []  # (global_block_index, RawClaim)
    num_batches = (len(blocks) + BATCH_SIZE - 1) // BATCH_SIZE
    all_ok = True
    recovered = [False]

    def _notify(kind, b):
        if kind != "retry":
            recovered[0] = True
        if not progress:
            return
        if kind == "retry":
            progress("retrying_batch", batch=b, total_batches=num_batches)
        elif kind == "split":
            progress("recovering", batch=b, total_batches=num_batches)
        elif kind == "block":
            progress("recovering_blocks", batch=b, total_batches=num_batches)

    for b in range(num_batches):
        start = b * BATCH_SIZE
        batch = blocks[start:start + BATCH_SIZE]
        if progress:
            progress("extracting_claims", batch=b, total_batches=num_batches)
        claims, ok = _extract_batch_with_recovery(start, batch, _llm, lambda k: _notify(k, b))
        if not ok:
            all_ok = False
            break
        raw_claims.extend(claims)

    # All-or-nothing: if any block could not be processed after all recovery,
    # discard EVERYTHING for this paper and leave the corpus untouched.
    if not all_ok:
        if progress:
            progress("failed", reason="Paper could not be completely processed. Nothing was added to the corpus.")
        return {
            "status": "failed",
            "reason": "Paper could not be completely processed. Nothing was added to the corpus.",
            "claims_added": 0,
        }

    if progress:
        progress("embedding")
    texts = [c.claim_text for (_, c) in raw_claims]
    embeddings = embed_texts(texts) if texts else []

    if progress:
        progress("merging")
    paper_id = f"user-{file_hash[:10]}"
    paper_title = _detect_paper_title(pdf_path, blocks) or (title or pdf_path.stem or "Untitled").strip()
    existing_count = len(dataset.get("claims", []))
    new_claims = []
    for n, ((bid, claim), embedding) in enumerate(zip(raw_claims, embeddings), start=1):
        block = blocks[bid]
        new_claims.append({
            "claim_id": f"{paper_id}-{n:03d}",
            "paper_id": paper_id,
            "paper_title": paper_title,
            "section": block.section,
            "page": block.page,
            "claim_text": claim.claim_text,
            "evidence_text": claim.evidence_text,
            "source_location": _location(block),
            "embedding": embedding,
            "relations": [],
        })

    # Atomic merge: only commit after everything succeeded.
    merged = {
        **dataset,
        "papers": dataset.get("papers", []) + [{
            "paper_id": paper_id,
            "title": paper_title,
            "authors": [],
            "focus": [],
            "source_file": str(pdf_path),
            "source_hash": file_hash,
        }],
        "claims": dataset.get("claims", []) + new_claims,
    }
    _atomic_write(merged, OUTPUT)
    _publish_dataset()
    if progress:
        progress("completed", claims_added=len(new_claims))
    return {
        "status": "completed",
        "paper_id": paper_id,
        "paper_title": paper_title,
        "claims_added": len(new_claims),
        "recovered": recovered[0],
    }


if __name__ == "__main__":
    main()
