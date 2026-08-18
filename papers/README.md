# Corpus source files

This directory holds the raw text of the curated research corpus. The ingestion
pipeline (`scripts/ingest.py`) reads these files and produces the single
source-of-truth dataset at `data/claims.json`.

Expected files (per `scripts/corpus.json`):

- `objectgraph.txt` — ObjectGraph: From Document Injection to Knowledge Traversal
- `psmas.txt` — Phase-Scheduled Multi-Agent Systems for Token-Efficient Coordination

## Adding a paper

Drop a `.txt`/`.md` (or PDF with a text layer) here and add a matching entry to
`scripts/corpus.json`, then run `python scripts/ingest.py`. A future uploaded PDF
enters the same pipeline.

## Important

Only include real, locally-available source text. Never fabricate paper content;
the ingestion script keeps only evidence passages that appear verbatim in the
source text.
