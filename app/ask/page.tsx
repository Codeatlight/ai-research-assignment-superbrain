'use client';

import { Suspense, useEffect, useState } from 'react';
import { useSearchParams } from 'next/navigation';
import Link from 'next/link';
import type { AskResponse, Dataset } from '@/lib/types';
import { SUPPORT_LABELS } from '@/lib/assess';
import { embedTexts } from '@/lib/embed';

function AskForm() {
  const params = useSearchParams();
  const [query, setQuery] = useState('');
  const [submitted, setSubmitted] = useState('');
  const [result, setResult] = useState<AskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [openClaim, setOpenClaim] = useState<number | null>(null);

  useEffect(() => {
    const q = params.get('q');
    if (q) {
      setQuery(q);
      run(q);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Map a raw API/LLM error to a concise, user-friendly message. The full
  // technical detail is still logged to the console for debugging.
  function friendlyError(err: unknown): string {
    const msg = err instanceof Error ? err.message : String(err);
    const text = msg.toLowerCase();
    if (text.includes('429') || text.includes('rate limit') || text.includes('tokens per minute')) {
      return 'Unable to generate an answer right now. The language model service has temporarily reached its usage limit. Please try again later.';
    }
    if (text.includes('fetch failed') || text.includes('network')) {
      return 'Could not reach the answer service. Please check your connection and try again.';
    }
    if (text.includes('insufficient_evidence') || text.includes('no retrieved evidence')) {
      return 'No evidence in the research corpus was relevant enough to answer this question.';
    }
    return 'Unable to generate an answer right now. Please try again later.';
  }

  async function run(q: string) {
    if (loading) return; // prevent duplicate submissions
    setLoading(true);
    setError(null);
    setResult(null);
    setOpenClaim(null);
    try {
      // Embed the query client-side with the same MiniLM model used during
      // ingestion. Only the question + query embedding are sent to the server;
      // the server loads and caches the dataset itself.
      const queryEmbedding = await embedTexts([q]).then((vecs) => vecs[0]);

      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, queryEmbedding }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
      setResult(data);
    } catch (e) {
      // Log the raw detail for debugging, but show only a friendly message.
      console.error('Ask error:', e);
      setError(friendlyError(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="container">
      <Link href="/" style={{ fontSize: '0.9rem' }}>
        ← Research library
      </Link>

      <form
        className="card"
        style={{ margin: '1.5rem 0', display: 'flex', gap: '0.75rem' }}
        onSubmit={(e) => {
          e.preventDefault();
          if (query.trim()) {
            setSubmitted(query.trim());
            run(query.trim());
          }
        }}
      >
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question across the research corpus…"
          style={{
            flex: 1,
            padding: '0.75rem',
            borderRadius: 8,
            border: '1px solid #333a48',
            background: '#0f1115',
            color: '#e6e8ee',
            fontSize: '1rem',
          }}
        />
        <button
          type="submit"
          disabled={loading}
          style={{
            padding: '0.75rem 1.25rem',
            borderRadius: 8,
            border: 'none',
            background: '#3b82f6',
            color: '#fff',
            fontSize: '1rem',
            cursor: loading ? 'default' : 'pointer',
            opacity: loading ? 0.6 : 1,
          }}
        >
          {loading ? 'Researching…' : 'Ask'}
        </button>
      </form>

      {error && (
        <div className="card" style={{ borderColor: '#7a2027', marginBottom: '1.5rem' }}>
          <p style={{ color: '#f07b7b', margin: 0 }}>{error}</p>
          <p className="muted" style={{ margin: '0.5rem 0 0' }}>
            The answer was not generated rather than returning a partial result.
          </p>
        </div>
      )}

      {loading && <p className="muted">Researching the corpus, generating an answer, and verifying each claim…</p>}

      {result && (
        <>
          <div className="card" style={{ marginBottom: '1.5rem' }}>
            <h3 style={{ marginTop: 0 }}>Answer</h3>
            <p style={{ whiteSpace: 'pre-wrap', margin: 0 }}>{result.answer}</p>
          </div>

          <h3 style={{ margin: '1.5rem 0 1rem' }}>Claims in this answer</h3>
          {result.claims.map((claim, i) => {
            const statusClass = `status-${claim.status}`;
            return (
              <div key={i} className="card" style={{ marginBottom: '1rem' }}>
                <div
                  style={{ display: 'flex', justifyContent: 'space-between', gap: '1rem', alignItems: 'flex-start' }}
                >
                  <p style={{ margin: 0, flex: 1 }}>{claim.claim_text}</p>
                  <button
                    onClick={() => setOpenClaim(openClaim === i ? null : i)}
                    className={`badge ${statusClass}`}
                    style={{ border: 'none', cursor: 'pointer', whiteSpace: 'nowrap' }}
                    title="Inspect evidence"
                  >
                    {claim.status === 'unverified'
                      ? 'Verification unavailable'
                      : SUPPORT_LABELS[claim.status as keyof typeof SUPPORT_LABELS] ?? claim.status}
                  </button>
                </div>
                {claim.status === 'unverified' && (
                  <p className="muted" style={{ margin: '0.5rem 0 0', fontSize: '0.85rem' }}>
                    Verification could not be completed for this claim — this is not a judgment
                    that the claim is unsupported.
                  </p>
                )}

                {openClaim === i && (
                  <div style={{ marginTop: '1rem', borderTop: '1px solid #262b36', paddingTop: '1rem' }}>
                    <p style={{ margin: '0 0 0.75rem' }}>
                      <strong>Why:</strong> {claim.reason}
                    </p>
                    <h4 className="muted" style={{ margin: '1rem 0 0.5rem', fontSize: '0.85rem' }}>
                      SUPPORTING EVIDENCE
                    </h4>
                    {claim.evidence.length === 0 ? (
                      <p className="muted">No source evidence was attached to this claim.</p>
                    ) : (
                      claim.evidence.map((ev, j) => (
                        <div key={j} className="card" style={{ marginBottom: '0.75rem', background: '#13161d' }}>
                          <p className="muted" style={{ margin: '0 0 0.4rem', fontSize: '0.85rem' }}>
                            {ev.paper_title} — {ev.source_location}
                          </p>
                          <blockquote style={{ margin: 0, paddingLeft: '0.75rem', borderLeft: '3px solid #3b82f6' }}>
                            {ev.evidence_text}
                          </blockquote>
                        </div>
                      ))
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </>
      )}

      {submitted && !result && !loading && !error && (
        <p className="muted">No result.</p>
      )}
    </div>
  );
}

export default function AskPage() {
  return (
    <Suspense fallback={<div className="container">Loading…</div>}>
      <AskForm />
    </Suspense>
  );
}
