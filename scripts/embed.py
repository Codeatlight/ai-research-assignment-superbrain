"""Embedding generation for the ingestion pipeline.

Local, offline, deterministic embeddings via sentence-transformers using the
all-MiniLM-L6-v2 model (384-dim, mean pooling + L2 normalization). The PyTorch
checkpoint here (sentence-transformers/all-MiniLM-L6-v2) shares the SAME weights
as the ONNX checkpoint used at runtime by the browser Transformers.js
implementation (Xenova/all-MiniLM-L6-v2, lib/embed.ts), so claim vectors computed
here are comparable to query vectors computed in the browser.

Model weights are cached under scripts/.cache on first run (a one-time download).
"""
from __future__ import annotations

from pathlib import Path

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_CACHE_DIR = Path(__file__).resolve().parent / ".cache"

_model = None


def _get_model():
    """Lazily load the sentence-transformers model once per process."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        # cache_folder keeps weights local/reproducible across runs
        _model = SentenceTransformer(
            EMBEDDING_MODEL,
            cache_folder=str(_CACHE_DIR),
        )
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts into normalized 384-dim vectors.

    SentenceTransformer defaults to mean pooling and L2-normalizes embeddings
    when normalize_embeddings=True, matching the browser implementation.
    """
    model = _get_model()
    if not texts:
        return []
    vectors = model.encode(texts, normalize_embeddings=True)
    return [list(map(float, v)) for v in vectors]
