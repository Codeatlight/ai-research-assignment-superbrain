# Research Claim & Evidence Tool

A claim-and-evidence-grounded research assistant: explore a curated corpus of
technical AI papers, ask a question, and inspect whether each claim in the
generated answer is supported by its source evidence.

Pipeline:
`question → retrieve relevant claims + evidence → generate answer → extract
answer claims → assess each claim against source evidence → show evidence and
support status`.

## Statuses

- **Supported by evidence**
- **Partially supported**
- **Not supported by retrieved evidence**
- **Insufficient evidence** (no retrieved source addresses the claim)

A claim is never treated as verified merely because an LLM generated it; the
assessment always points back to verbatim source evidence.

## Architecture

- **Offline ingestion** (`scripts/`): `papers/` → text extraction → claim +
  verbatim-evidence extraction (validated as substrings of the source) →
  embeddings → `data/claims.json` (single source of truth).
- **Runtime** (Next.js, Vercel-compatible): `/api/ask` runs the sequential
  pipeline. No vector DB — brute-force cosine similarity over the small corpus.
  `/api/library` serves paper metadata.

## Quick start

```bash
npm install
cp .env.example .env.local   # add OPENAI_API_KEY
npm run ingest               # builds data/claims.json from papers/
npm run build                # copies dataset to public/ and builds
npm run dev
```

Open http://localhost:3000, ask a question, expand a claim to inspect evidence.

## Adding a paper

Drop the source text (`.txt`/`.md`, or PDF with text layer) into `papers/`,
add an entry to `scripts/corpus.json`, then `npm run ingest`. The same pipeline
is used for future uploaded PDFs.

## Evaluation

`npm run eval` sends the manual questions in `eval/eval_questions.json` to the
running dev server and prints per-question metrics (claim counts, status
distribution, expected-claim recall).
