import type { Claim, Dataset } from './types';

/**
 * Retrieval helpers. Query embedding is NOT here — it lives in lib/embed.ts
 * (Transformers.js, Xenova/all-MiniLM-L6-v2), used client-side in the browser
 * and by tests. This module only computes similarity and selects top claims.
 */

/** Compute cosine similarity between two equal-length vectors. */
export function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let magA = 0;
  let magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  if (magA === 0 || magB === 0) return 0;
  return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}

export interface RetrievedClaim {
  claim: Claim;
  score: number;
}

/**
 * Minimum cosine-similarity for a claim to be considered relevant to a query.
 * Configurable via the RETRIEVAL_THRESHOLD env var. Defaults to 0, which
 * restores the original behavior (no relevance filter — all top-k returned).
 * A measured corpus-specific value should be set from the similarity analysis
 * in eval/threshold_analysis.mjs once approved; do not hardcode an unvalidated
 * value here.
 */
export const RETRIEVAL_THRESHOLD: number =
  process.env.RETRIEVAL_THRESHOLD !== undefined
    ? Number(process.env.RETRIEVAL_THRESHOLD)
    : 0;

/**
 * Retrieve the top-k claims most similar to the query embedding,
 * using brute-force cosine similarity over the small corpus.
 * When `paperId` is provided, restrict retrieval to that paper's claims
 * (paper-specific queries). A null paperId preserves corpus-wide retrieval.
 */
export function retrieveClaims(
  dataset: Dataset,
  queryEmbedding: number[],
  topK = 10,
  threshold = RETRIEVAL_THRESHOLD,
  paperId?: string | null,
): RetrievedClaim[] {
  const pool = paperId
    ? dataset.claims.filter((c) => c.paper_id === paperId)
    : dataset.claims;
  const scored: RetrievedClaim[] = pool.map((claim) => ({
    claim,
    score: cosineSimilarity(queryEmbedding, claim.embedding),
  }));
  scored.sort((a, b) => b.score - a.score);
  // Drop claims below the relevance threshold so unrelated questions do not
  // receive a fixed top-k of weakly similar claims.
  return scored.filter((r) => r.score >= threshold).slice(0, topK);
}
