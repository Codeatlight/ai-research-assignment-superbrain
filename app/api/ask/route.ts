import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import type { Dataset, SupportStatus } from '@/lib/types';
import { retrieveClaims, RETRIEVAL_THRESHOLD } from '@/lib/retrieval';
import { assessClaims, type BatchVerifyResult, type EvidenceContext } from '@/lib/assess';
import { chatJSON } from '@/lib/llm';

// Server-side, cached dataset loaded once from the shipped public asset. The
// client never sends the dataset; the server reads it directly.
let datasetCache: Dataset | null = null;

function loadDataset(): Dataset {
  if (!datasetCache) {
    const raw = fs.readFileSync(
      path.join(process.cwd(), 'public', 'data', 'claims.json'),
      'utf-8',
    );
    datasetCache = JSON.parse(raw) as Dataset;
  }
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
  try {
    const body = await req.json();
    const { question, queryEmbedding, paper_id } = body as {
      question?: string;
      queryEmbedding?: number[];
      paper_id?: string;
    };
    if (!question || typeof question !== 'string') {
      return NextResponse.json({ error: 'question is required' }, { status: 400 });
    }
    if (!queryEmbedding || !Array.isArray(queryEmbedding) || queryEmbedding.length === 0) {
      return NextResponse.json(
        { error: 'queryEmbedding is required' },
        { status: 400 },
      );
    }

    const dataset = loadDataset();
    // Paper-specific queries retrieve only from the named paper; a question
    // that names no paper keeps corpus-wide retrieval (cross-paper allowed).
    const resolvedPaperId = paper_id || detectPaperFromQuestion(question);
    const retrieved = retrieveClaims(dataset, queryEmbedding, 10, RETRIEVAL_THRESHOLD, resolvedPaperId);

    // No claim clears the relevance threshold: return an explicit
    // insufficient-evidence response instead of answering from weak context.
    if (retrieved.length === 0) {
      return NextResponse.json({
        answer: 'No retrieved evidence in the corpus is relevant enough to this question to support a grounded answer.',
        claims: [],
      });
    }

    // Only relevant claims/evidence go to the model; source refs are preserved.
    const evidenceContext: EvidenceContext[] = retrieved.map((r) => ({
      claim_id: r.claim.claim_id,
      paper_title: r.claim.paper_title,
      section: r.claim.section,
      page: r.claim.page,
      evidence_text: r.claim.evidence_text,
      source_location: r.claim.source_location,
    }));

    const { answer, answer_claims } = await chatJSON<AnswerLLMResult>(
      ANSWER_SYSTEM,
      buildAnswerPrompt(question, retrieved),
      isAnswerLLMResult,
    );

    const verifyLLM = (system: string, user: string) =>
      chatJSON<{ verifications: BatchVerifyResult[] }>(
        system,
        user,
        isBatchVerifyResult,
      );

    const assessed = await assessClaims(
      answer_claims.map((text) => ({ claim_text: text })),
      evidenceContext,
      verifyLLM,
    );

    return NextResponse.json({ answer, claims: assessed });
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : String(err) },
      { status: 500 },
    );
  }
}
