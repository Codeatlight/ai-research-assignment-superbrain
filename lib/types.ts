// Shared type definitions for the claim-and-evidence research tool.

/** Single source of truth for the embedding model used by ingestion + runtime. */
export const EMBEDDING_MODEL = 'Xenova/all-MiniLM-L6-v2';
export const EMBEDDING_DIM = 384;


/** A single curated research paper in the corpus. */
export interface Paper {
  paper_id: string;
  title: string;
  authors: string[];
  /** Curated topic tags, drawn from the product's focus areas. */
  focus: string[];
  source_file: string;
  /** Number of claims belonging to this paper (library display). */
  claimCount?: number;
}

/** A single extracted claim with its verbatim supporting evidence. */
export interface Claim {
  claim_id: string;
  paper_id: string;
  paper_title: string;
  section: string;
  /** 1-based page number when the source text preserves pages. */
  page: number | null;
  claim_text: string;
  /** Verbatim passage from the source supporting the claim. */
  evidence_text: string;
  source_location: string;
  embedding: number[];
  /** Reserved for future relationship support; empty in v1. */
  relations: ClaimRelation[];
}

export interface ClaimRelation {
  type: 'supports' | 'contradicts' | 'extends' | 'related_to';
  target_claim_id: string;
}

/** The static dataset: single source of truth produced by scripts/ingest.py. */
export interface Dataset {
  version: number;
  embedding_model: string;
  papers: Paper[];
  claims: Claim[];
}

/** The four evidence-assessment states. */
export type SupportStatus =
  | 'supported'
  | 'partially_supported'
  | 'not_supported'
  | 'insufficient_evidence';

/** A claim extracted from a generated answer, with its assessment. */
export interface AnswerClaim {
  claim_text: string;
  status: SupportStatus | 'unverified';
  /** Short human-readable explanation of the assigned status. */
  reason: string;
  /** The claim/evidence records that support (or fail to support) it. */
  evidence: EvidenceReference[];
}

export interface EvidenceReference {
  claim_id: string;
  paper_title: string;
  section: string;
  page: number | null;
  evidence_text: string;
  source_location: string;
}

export interface AskResponse {
  answer: string;
  claims: AnswerClaim[];
}
