"""Local/DEV upload worker: incrementally ingest ONE uploaded PDF into the corpus.

Usage:
    python scripts/ingest_upload.py <path-to-pdf> [title]

Reads GROQ_API_KEY from .env.local, processes the PDF through the existing
pipeline (batched claim extraction, verbatim validation, MiniLM embeddings),
and atomically merges the new paper's claims into data/claims.json. Exits 0 on
success, 2 on duplicate, 1 on failure.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ingest import OUTPUT, ingest_uploaded_paper  # noqa: E402


def _main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest_upload.py <pdf> [title]", file=sys.stderr)
        return 1
    pdf = Path(sys.argv[1])
    title = sys.argv[2] if len(sys.argv) > 2 else None
    if not pdf.is_file():
        print(f"error: file not found: {pdf}", file=sys.stderr)
        return 1

    status_file = ROOT / "data" / ".upload_status.json"

    def progress(state: str, **kwargs) -> None:
        status_file.parent.mkdir(parents=True, exist_ok=True)
        status_file.write_text(
            json.dumps({"state": state, **kwargs}), encoding="utf-8"
        )
        if kwargs:
            extra = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            print(f"  [{state}] {extra}", flush=True)
        else:
            print(f"  [{state}]", flush=True)

    result = ingest_uploaded_paper(pdf, title=title, progress=progress)
    print(json.dumps(result))
    # Normalize the terminal result so `state` is always present alongside
    # `status` (progress uses `state`; the final result uses `status`).
    result = {**result, "state": result.get("status")}
    status_file.write_text(json.dumps(result), encoding="utf-8")
    # Release the single-upload lock so the next upload can proceed.
    lock = ROOT / "data" / ".upload_lock"
    if lock.exists():
        try:
            lock.unlink()
        except OSError:
            pass
    if result.get("status") == "completed":
        print(f"Paper added successfully. Claims added: {result.get('claims_added')}")
        return 0
    if result.get("status") == "already_exists":
        print(f"already_exists: paper {result.get('paper_id')}")
        return 2
    print(f"Paper ingestion failed. Reason: {result.get('reason')}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(_main())
