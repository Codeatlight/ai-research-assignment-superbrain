import type { AnswerClaim, EvidenceReference, SupportStatus } from './types';

export const SUPPORT_LABELS: Record<SupportStatus, string> = {
  supported: 'Supported by evidence',
  partially_supported: 'Partially supported',
  not_supported: 'Not supported by retrieved evidence',
  insufficient_evidence: 'Insufficient evidence',
};

export interface EvidenceContext {
  claim_id: string;
  paper_title: string;
  section: string;
  page: number | null;
  evidence_text: string;
  source_location: string;
}

/**
 * Adversarial assessment: given a generated answer claim and the retrieved
 * source evidence, decide whether the source actually supports the claim.
 *
 * The model must quote matching source text or explicitly say none exists.
 * A claim is never auto-accepted just because an LLM generated it.
 */
/** A single verification result for one answer claim within a batch. */
export interface BatchVerifyResult {
  claim_index: number;
  status: SupportStatus;
  reason: string;
  evidence_ids: string[];
}

const VALID_STATUSES: SupportStatus[] = [
  'supported',
  'partially_supported',
  'not_supported',
  'insufficient_evidence',
];

/** Build the batch verification prompt: all claims indexed, evidence once. */
const BATCH_ASSESS_PROMPT = (
  claims: { claim_text: string }[],
  evidence: EvidenceContext[],
): string => {
  const claimList = claims
    .map((c, i) => `${i}: "${c.claim_text}"`)
    .join('\n');
  const evidenceBlock = evidence
    .map(
      (e) =>
        `[${e.claim_id}] (${e.paper_title}, ${e.source_location})\n${e.evidence_text}`,
    )
    .join('\n\n');
  return `You are a strict research-verification reviewer.

You must decide how well the SOURCE EVIDENCE below supports EACH claim in a
generated answer. Evaluate every claim INDEPENDENTLY. You are NOT allowed to
use general knowledge or to accept a claim simply because it appears in an
answer. Do not infer unsupported facts.

CLAIMS TO VERIFY (indexed):
${claimList}

RETRIEVED SOURCE EVIDENCE (available once, for all claims):
${evidenceBlock || '(No evidence was retrieved.)'}

For EACH claim decide the status:
- "supported": the evidence directly and explicitly supports the claim.
- "partially_supported": the evidence supports part of the claim but not all of it.
- "not_supported": the evidence is present but contradicts the claim or says the opposite.
- "insufficient_evidence": no retrieved evidence substantially addresses the claim.

Rules:
- Return EXACTLY one verification object per claim, preserving the original claim_index.
- Do not omit any claim; do not invent extra entries.
- "evidence_ids" must contain ONLY ids present in the RETRIEVED SOURCE EVIDENCE that support/relate to THAT specific claim. Empty array when none apply.
- Return valid JSON only:
{"verifications": [{"claim_index": 0, "status": "...", "reason": "...", "evidence_ids": ["..."]}, ...]}`;
};

/** Strict validation of a batched verification response. Returns null if valid. */
export function validateBatchResults(
  data: unknown,
  expectedCount: number,
): string | null {
  if (typeof data !== 'object' || data === null) return 'verifications missing';
  const o = data as Record<string, unknown>;
  const list = o.verifications;
  if (!Array.isArray(list)) return 'verifications is not an array';
  if (list.length !== expectedCount) {
    return `expected ${expectedCount} verification entries, got ${list.length}`;
  }
  const seen = new Set<number>();
  const validIds = new Set<string>();
  for (const item of list) {
    if (typeof item !== 'object' || item === null) return 'entry not an object';
    const v = item as Record<string, unknown>;
    const idx = v.claim_index;
    if (typeof idx !== 'number' || !Number.isInteger(idx)) return 'claim_index not an integer';
    if (idx < 0 || idx >= expectedCount) return `claim_index out of range: ${idx}`;
    if (seen.has(idx)) return `duplicate claim_index: ${idx}`;
    seen.add(idx);
    if (typeof v.status !== 'string' || !VALID_STATUSES.includes(v.status as SupportStatus)) {
      return `invalid status for claim ${idx}`;
    }
    if (typeof v.reason !== 'string' || v.reason.trim().length === 0) {
      return `empty reason for claim ${idx}`;
    }
    if (!Array.isArray(v.evidence_ids)) return `evidence_ids not array for claim ${idx}`;
    for (const id of v.evidence_ids) {
      if (typeof id !== 'string') return `non-string evidence_id for claim ${idx}`;
      validIds.add(id);
    }
  }
  // validIds is only used to confirm ids are strings; membership is checked at
  // attach time against the refById map (unknown ids must fail, not silently pass).
  return null;
}

/**
 * Run the assessment for a list of answer claims against the given evidence.
 * Injected with an LLM callback so tests can stub it. On any failure the
 * status falls back to "unverified" and the reason reports the failure.
 */
const ASSESS_SYSTEM =
  'You are a strict research-verification reviewer that returns JSON.';

/** Result of a batched verification: either the mapped claims or an error string. */
export type BatchOutcome =
  | { ok: true; claims: AnswerClaim[] }
  | { ok: false; error: string };

export async function assessClaims(
  answerClaims: { claim_text: string }[],
  evidence: EvidenceContext[],
  callLLM: (
    system: string,
    user: string,
  ) => Promise<{ verifications: BatchVerifyResult[] }>,
): Promise<AnswerClaim[]> {
  // Map for fast id -> EvidenceReference lookup.
  const refById = new Map<string, EvidenceReference>(
    evidence.map((e) => [e.claim_id, {
      claim_id: e.claim_id,
      paper_title: e.paper_title,
      section: e.section,
      page: e.page,
      evidence_text: e.evidence_text,
      source_location: e.source_location,
    }]),
  );
  // Lenient secondary lookup keyed on the id suffix. The verifier LLM can drop
  // a paper prefix (e.g. returns "957bb3ec38-081" for "user-957bb3ec38-081").
  // We resolve by the trailing "-NNN" segment so a mangled prefix still maps to
  // the correct evidence record instead of failing the whole batch.
  const bySuffix = new Map<string, EvidenceReference>();
  for (const e of evidence) {
    const dash = e.claim_id.lastIndexOf('-');
    const suffix = dash >= 0 ? e.claim_id.slice(dash + 1) : e.claim_id;
    if (!bySuffix.has(suffix)) bySuffix.set(suffix, refById.get(e.claim_id)!);
  }
  const resolveId = (id: string): EvidenceReference | undefined => {
    // 1) exact claim_id match
    if (refById.has(id)) return refById.get(id);
    // 2) if the id already looks like a bare numeric suffix, use bySuffix directly
    const givenTail = id.includes('-') ? id.slice(id.lastIndexOf('-') + 1) : id;
    return bySuffix.get(id) ?? bySuffix.get(givenTail);
  };

  const attempt = async (): Promise<BatchOutcome> => {
    const res = await callLLM(
      ASSESS_SYSTEM,
      BATCH_ASSESS_PROMPT(answerClaims, evidence),
    );
    const validationError = validateBatchResults(res, answerClaims.length);
    if (validationError) return { ok: false, error: `Batch validation failed: ${validationError}` };

    // Build an ordered result per claim_index.
    const byIndex = new Map<number, BatchVerifyResult>(
      res.verifications.map((v) => [v.claim_index, v]),
    );
    const claims: AnswerClaim[] = [];
    for (let i = 0; i < answerClaims.length; i++) {
      const v = byIndex.get(i)!;
      // Resolve evidence ids against the retrieved set, tolerating a dropped
      // paper prefix via bySuffix. A genuinely unresolvable id is ignored (the
      // claim then has no evidence -> insufficient_evidence), rather than
      // failing the entire batch because of one mangled id.
      const relevant: EvidenceReference[] = v.evidence_ids
        .map((id) => resolveId(id))
        .filter((r): r is EvidenceReference => !!r);
      // A claim with no relevant evidence is an explicit insufficient state,
      // never a pile of unrelated retrieved passages.
      const finalStatus: SupportStatus =
        relevant.length === 0 ? 'insufficient_evidence' : v.status;
      claims.push({
        claim_text: answerClaims[i].claim_text,
        status: finalStatus,
        reason: v.reason,
        evidence: relevant,
      });
    }
    return { ok: true, claims };
  };

  try {
    const outcome = await attempt();
    if (outcome.ok) return outcome.claims;
    // Validation failure: treat the whole batch as failed -> all unverified.
    return answerClaims.map((ac) => ({
      claim_text: ac.claim_text,
      status: 'unverified' as const,
      reason: outcome.error,
      evidence: [],
    }));
  } catch (err) {
    return answerClaims.map((ac) => ({
      claim_text: ac.claim_text,
      status: 'unverified' as const,
      reason: `Verification could not be completed: ${err instanceof Error ? err.message : String(err)}`,
      evidence: [],
    }));
  }
}
