"""Claim + evidence extraction from document blocks.

Each extracted claim must carry a verbatim evidence passage that actually
appears in the source text. The LLM proposes claims/evidence; we validate that
each evidence passage is a real substring of the source block before accepting
it, which guards against hallucinated paper content during ingestion.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from extract_text import TextBlock  # type: ignore  # scripts/ is on sys.path when run via ingest.py

EXTRACT_SYSTEM = """You extract atomic factual/research claims from research paper
text. For each claim you must also provide the EXACT verbatim passage from the
given text that supports it.

Output ONLY the following delimiter format, repeating one block per claim:

CLAIM:
<claim text>
EVIDENCE:
<exact contiguous evidence substring>
END_CLAIM

Rules:
- Extract only claims that are actually stated or implied by the given text.
- Do not add outside knowledge or invent metrics, results, or citations.
- Do not use JSON, markdown fences, numbering, or any commentary.
- Preserve mathematical notation and Unicode exactly where possible.
- The evidence_text MUST be an exact contiguous substring of the provided source
  text (copy it exactly, including punctuation). Do not paraphrase it.
- Keep each claim atomic: one factual statement per claim.
- Emit at most 5 claims per source block.
- If no useful claims exist, return only: NO_CLAIMS"""

# Stricter prompt used when a block's first extraction attempt is unusable.
EXTRACT_SYSTEM_RETRY = """Re-read the source text and extract claims again, strictly.

Return ONLY the delimiter format, nothing else:

CLAIM:
<claim text>
EVIDENCE:
<exact contiguous evidence substring>
END_CLAIM

Hard rules:
- No JSON, no markdown, no commentary, no extra text, no numbering.
- Preserve mathematical notation and Unicode exactly where possible.
- evidence_text MUST be an exact contiguous substring of the source text (copy it
  exactly, including punctuation). Do not paraphrase.
- At most 5 claims.
- If no useful claims exist, return only: NO_CLAIMS"""


@dataclass
class RawClaim:
    claim_text: str
    evidence_text: str


class BatchExtractionError(Exception):
    """Raised when a batched extraction response is structurally invalid.

    Indicates a batch-level failure (bad/missing/duplicate/out-of-range BLOCK_ID
    or malformed CLAIM/EVIDENCE structure). Callers retry the whole batch; they
    must NOT partially accept a structurally malformed batch.
    """


def _normalize(s: str) -> str:
    return " ".join(s.split())


# Count of claims rejected by the verbatim-evidence validator (module-global
# so ingest.py can report it without changing the parser's return contract).
_REJECTED = 0


def _parse_delimiter_claims(content: str) -> list[dict]:
    """Deterministically parse CLAIM:/EVIDENCE:/END_CLAIM blocks into claim dicts.

    Not LLM-based. Returns a list of {"claim_text", "evidence_text"} for each
    well-formed block; malformed blocks are skipped.
    """
    out: list[dict] = []
    blocks = re.split(r"(?m)^END_CLAIM\s*$", content)
    for blk in blocks:
        if "CLAIM:" not in blk or "EVIDENCE:" not in blk:
            continue
        _, _, after_claim = blk.partition("CLAIM:")
        claim_part, _, evidence_part = after_claim.partition("EVIDENCE:")
        claim_text = claim_part.strip()
        evidence_text = evidence_part.strip()
        if not claim_text or not evidence_text:
            continue
        out.append({"claim_text": claim_text, "evidence_text": evidence_text})
    return out


BATCH_EXTRACT_SYSTEM = """You extract atomic factual/research claims from research paper
text. Multiple source blocks are provided, each identified by a BLOCK_ID.

Output ONLY the following delimiter format, repeating one group per claim.
Every claim group MUST begin with a BLOCK_ID line:

BLOCK_ID: <id>
CLAIM:
<claim text>
EVIDENCE:
<exact contiguous evidence substring>
END_CLAIM

Example:
BLOCK_ID: 2
CLAIM:
The system achieves a mean reduction of 27.3%.
EVIDENCE:
achieves a mean token reduction of27.3%
END_CLAIM

Rules:
- EVERY claim group MUST start with "BLOCK_ID: <id>" — never omit it. The <id>
  is the id of the source block the claim comes from (an integer from the input).
- Extract only claims actually stated or implied in that block.
- Do not add outside knowledge or invent metrics, results, or citations.
- Do not use JSON, markdown fences, or commentary.
- Preserve mathematical notation and Unicode exactly where possible.
- evidence_text MUST be an exact contiguous substring of the block with that
  BLOCK_ID (copy it exactly, including punctuation). Do not paraphrase.
- Keep each claim atomic: one factual statement per claim.
- Emit at most 5 claims per block.
- A block with no useful claims may simply have no entries for it.
- Every claim MUST include a valid BLOCK_ID."""


def extract_claims_from_block(block: TextBlock, call_llm) -> list[RawClaim]:
    """Extract claims from one block, keeping only verbatim-supported evidence."""
    global _REJECTED
    prompt = (
        f"SOURCE TEXT:\n\"\"\"\n{block.text}\n\"\"\"\n\n"
        "Extract the claims and verbatim evidence as specified."
    )
    content = call_llm(EXTRACT_SYSTEM, prompt)
    normalized_source = _normalize(block.text)

    claims: list[RawClaim] = []
    for item in _parse_delimiter_claims(content):
        claim_text = (item.get("claim_text") or "").strip()
        evidence_text = (item.get("evidence_text") or "").strip()
        if not claim_text or not evidence_text:
            continue
        # Verbatim check: the normalized evidence must appear in the source.
        if _normalize(evidence_text) not in normalized_source:
            _REJECTED += 1
            continue
        claims.append(RawClaim(claim_text=claim_text, evidence_text=evidence_text))
    return claims


def _parse_batched(content: str, blocks: list[TextBlock]) -> list[tuple[int, RawClaim]]:
    """Parse a batched extraction response into (block_index, RawClaim) pairs.

    Structural problems (missing/invalid/duplicate/out-of-range BLOCK_ID, or
    malformed CLAIM/EVIDENCE) raise BatchExtractionError so the caller can retry
    the whole batch. Individual claims whose evidence fails the normalized
    substring check against their OWN source block are rejected per-claim.
    Never fuzzy-match a block.
    """
    global _REJECTED
    claims: list[tuple[int, RawClaim]] = []
    for grp in re.split(r"(?m)^END_CLAIM\s*$", content):
        has_claim = "CLAIM:" in grp
        has_evidence = "EVIDENCE:" in grp
        if not has_claim and not has_evidence and "BLOCK_ID:" not in grp:
            # Non-claim filler between groups; ignore.
            continue
        if not (has_claim and has_evidence and "BLOCK_ID:" in grp):
            raise BatchExtractionError("missing BLOCK_ID/CLAIM/EVIDENCE in a claim group")
        # Exactly one BLOCK_ID per claim group.
        if grp.count("BLOCK_ID:") != 1:
            raise BatchExtractionError("duplicate/missing BLOCK_ID in a claim group")
        _, _, rest = grp.partition("BLOCK_ID:")
        id_line, _, after_id = rest.partition("\n")
        try:
            bid = int(id_line.strip())
        except ValueError:
            raise BatchExtractionError(f"invalid BLOCK_ID: {id_line.strip()!r}")
        if bid < 0 or bid >= len(blocks):
            raise BatchExtractionError(f"out-of-range BLOCK_ID: {bid}")
        _, _, after_claim = after_id.partition("CLAIM:")
        claim_part, _, evidence_part = after_claim.partition("EVIDENCE:")
        claim_text = claim_part.strip()
        evidence_text = evidence_part.strip()
        if not claim_text or not evidence_text:
            raise BatchExtractionError("empty CLAIM or EVIDENCE")
        # Per-block verbatim check: only against THIS block's text, never the
        # concatenated batch. Evidence from another block is rejected here.
        if _normalize(evidence_text) not in _normalize(blocks[bid].text):
            _REJECTED += 1
            continue
        claims.append((bid, RawClaim(claim_text=claim_text, evidence_text=evidence_text)))
    return claims


def extract_claims_from_blocks_batch(
    blocks: list[TextBlock], call_llm
) -> list[tuple[int, RawClaim]]:
    """Extract claims from a fixed-size batch of blocks (block-tagged).

    Returns (block_index, RawClaim) so callers can recover section/page from the
    original TextBlock. Raises BatchExtractionError on structural failure.
    """
    prompt_parts = []
    for i, block in enumerate(blocks):
        prompt_parts.append(f"BLOCK_ID: {i}\n\"\"\"\n{block.text}\n\"\"\"")
    prompt = (
        "SOURCE BLOCKS (each labeled with its BLOCK_ID):\n"
        + "\n\n".join(prompt_parts)
        + "\n\nExtract claims per block, tagging each with its BLOCK_ID, as specified."
    )
    content = call_llm(BATCH_EXTRACT_SYSTEM, prompt)
    return _parse_batched(content, blocks)


def _default_llm_call(system: str, prompt: str) -> str:
    """Default LLM client (Groq chat completions, plain text).

    Used only when ingest.py does not supply its own _llm. Returns the raw model
    text; the delimiter parser in extract_claims_from_block does the parsing.
    """
    import httpx

    api_key = os.environ["GROQ_API_KEY"]
    url = "https://api.groq.com/openai/v1/chat/completions"
    model = os.environ.get("GROQ_INGEST_MODEL", "llama-3.1-8b-instant")
    for attempt in range(2):
        res = httpx.post(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=120,
        )
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]
    raise ValueError("Ingestion LLM call failed")
