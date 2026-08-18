"""Cross-runtime embedding consistency test (Python side).

Companion test for scripts/embed.py (the ingestion embedding implementation).
Embeds a fixed set of representative sentences and writes the normalized
vectors to data/embed_test_python.json so the Transformers.js side can embed
the same sentences and compare (lib/embed_consistency_js.mjs).

Usage:
    python scripts/embed_consistency_py.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from embed import embed_texts, EMBEDDING_MODEL  # noqa: E402

SENTENCES = [
    "ObjectGraph stores entities and relations as a native graph for agentic AI.",
    "Phase-Scheduled Multi-Agent Systems reduce token cost through coordinated scheduling.",
    "Agents require persistent memory to operate effectively across tasks.",
    "Retrieval-augmented generation improves answer accuracy using source evidence.",
    "Token efficiency is critical for scalable multi-agent coordination.",
]


def main() -> None:
    vectors = embed_texts(SENTENCES)
    out = {
        "model": EMBEDDING_MODEL,
        "dim": len(vectors[0]),
        "sentences": SENTENCES,
        "vectors": vectors,
    }
    dest = ROOT / "data" / "embed_test_python.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {len(vectors)} vectors (dim={len(vectors[0])}) to {dest}")


if __name__ == "__main__":
    main()
