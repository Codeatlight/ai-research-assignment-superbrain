import { NextRequest, NextResponse } from 'next/server';
import type { Dataset, SupportStatus } from '@/lib/types';
import { retrieveClaims, RETRIEVAL_THRESHOLD } from '@/lib/retrieval';
import { assessClaims, type BatchVerifyResult, type EvidenceContext } from '@/lib/assess';
import { chatJSON } from '@/lib/llm';
// Import the corpus at build time so it is bundled into the serverless function,
// deterministically available on Vercel regardless of fs tracing / process.cwd().
import claimsData from '@/data/claims.json';

// Server-side singleton (the imported JSON is already a build-time constant).
const datasetCache: Dataset = claimsData as Dataset;

function loadDataset(): Dataset {
  return datasetCache;
}

interface AnswerLLMResult {
  answer: string;
  answer_claims: string[];
}

function isAnswerLLMResult(d: unknown): d is AnswerLLMResult {
  if (typeof d !== 'object' || d === null) return false;
  const o = d as Record<string, unknown>;
  return (
    typeof o.answer === 'string' &&
    Array.isArray(o.answer_claims) &&
    o.answer_claims.length >= 1 &&
    o.answer_claims.every((c) => typeof c === 'string' && c.trim().length > 0)
  );
}

const ANSWER_SYSTEM =
  'You are a research assistant grounded strictly in the provided claims and their source evidence. Return JSON.';

// Distinctive tokens used to detect which paper a question is about, so a
// paper-specific query is retrieved only from that paper instead of pulling in
// semantically-similar claims from other papers. Falls back to corpus-wide
// when the question names no paper.
const PAPER_KEYWORDS: Record<string, string[]> = {
  objectgraph: ['objectgraph', 'object graph'],
  psmas: ['psmas', 'phase-scheduled', 'phase scheduled', 'token-efficient coordination'],
  'user-957bb3ec38': ['retrieval-augmented generation', 'retrieval augmented generation', 'rag', 'openai assistant api'],
};

function detectPaperFromQuestion(question: string): string | null {
  const q = question.toLowerCase();
  for (const [paperId, keys] of Object.entries(PAPER_KEYWORDS)) {
    if (keys.some((k) => q.includes(k))) return paperId;
  }
  return null;
}

// Answer generation grounded ONLY in retrieved claims + evidence.
function buildAnswerPrompt(
  question: string,
  retrieved: ReturnType<typeof retrieveClaims>,
): string {
  const evidenceBlock = retrieved
    .map(
      (r) =>
        `- Claim: ${r.claim.claim_text}\n  Source: ${r.claim.paper_title}, ${r.claim.source_location}\n  Evidence: ${r.claim.evidence_text}`,
    )
    .join('\n');
  return `Answer the question using ONLY the evidence below. If the evidence does
not answer the question, say so clearly.

EVIDENCE:
${evidenceBlock}

QUESTION:
"""${question}"""

Return JSON only: {"answer": "your grounded answer", "answer_claims": ["atomic claim 1", "atomic claim 2"]}`;
}

function isBatchVerifyResult(d: unknown): d is { verifications: BatchVerifyResult[] } {
  if (typeof d !== 'object' || d === null) return false;
  const o = d as Record<string, unknown>;
  if (!Array.isArray(o.verifications)) return false;
  const statuses: SupportStatus[] = [
    'supported',
    'partially_supported',
    'not_supported',
    'insufficient_evidence',
  ];
  return o.verifications.every((v) => {
    if (typeof v !== 'object' || v === null) return false;
    const e = v as Record<string, unknown>;
    return (
      typeof e.claim_index === 'number' &&
      statuses.includes(e.status as SupportStatus) &&
      typeof e.reason === 'string' &&
      Array.isArray(e.evidence_ids) &&
      e.evidence_ids.every((id) => typeof id === 'string')
    );
  });
}

export async function POST(req: NextRequest) {
  console.log('[ask] ask route started');
  try {
    const body = await req.json();
    console.log('[ask] body parsed');
    const { question, queryEmbedding, paper_id } = body as {
      question?: string;
      queryEmbedding?: number[];
      paper_id?: string;
    };
    if (!question || typeof question !== 'string') {
      console.log('[ask] validation failed: question');
      return NextResponse.json({ error: 'question is required' }, { status: 400 });
    }
    if (!queryEmbedding || !Array.isArray(queryEmbedding) || queryEmbedding.length === 0) {
      console.log('[ask] validation failed: queryEmbedding');
      return NextResponse.json(
        { error: 'queryEmbedding is required' },
        { status: 400 },
      );
    }
    console.log('[ask] validation passed. embedding length:', queryEmbedding.length);

    const dataset = loadDataset();
    console.log('[ask] corpus loaded. number of claims:', dataset?.claims?.length);

    const resolvedPaperId = paper_id || detectPaperFromQuestion(question);
    console.log('[ask] resolvedPaperId:', resolvedPaperId);

    const retrieved = retrieveClaims(dataset, queryEmbedding, 10, RETRIEVAL_THRESHOLD, resolvedPaperId);
    console.log('[ask] retrieval completed. retrieved count:', retrieved.length);

    if (retrieved.length === 0) {
      console.log('[ask] returning insufficient evidence');
      return NextResponse.json({
        answer: 'No retrieved evidence in the corpus is relevant enough to this question to support a grounded answer.',
        claims: [],
      });
    }

    const evidenceContext: EvidenceContext[] = retrieved.map((r) => ({
      claim_id: r.claim.claim_id,
      paper_title: r.claim.paper_title,
      section: r.claim.section,
      page: r.claim.page,
      evidence_text: r.claim.evidence_text,
      source_location: r.claim.source_location,
    }));
    console.log('[ask] context built. GROQ_API_KEY present:', !!process.env.GROQ_API_KEY, 'GROQ_MODEL name:', process.env.GROQ_MODEL || 'default(llama-3.3-70b-versatile)');
    console.log('[ask] calling Groq for answer');

    const { answer, answer_claims } = await chatJSON<AnswerLLMResult>(
      ANSWER_SYSTEM,
      buildAnswerPrompt(question, retrieved),
      isAnswerLLMResult,
    );
    console.log('[ask] Groq response received for answer');

    const verifyLLM = async (system: string, user: string) => {
      console.log('[ask] calling Groq for verification');
      const res = await chatJSON<{ verifications: BatchVerifyResult[] }>(
        system,
        user,
        isBatchVerifyResult,
      );
      console.log('[ask] Groq response received for verification');
      return res;
    };

    const assessed = await assessClaims(
      answer_claims.map((text) => ({ claim_text: text })),
      evidenceContext,
      verifyLLM,
    );
    console.log('[ask] assessment completed');

    return NextResponse.json({ answer, claims: assessed });
  } catch (err) {
    console.error('[ask] error caught:', err instanceof Error ? err.stack || err.message : String(err));
    console.error('[ask] error type:', typeof err, err ? (err as any).constructor?.name : 'null');
    console.error('[ask] GROQ_API_KEY present:', !!process.env.GROQ_API_KEY);
    return NextResponse.json(
      { error: err instanceof Error ? err.message : String(err) },
      { status: 500 },
    );
  }
}
