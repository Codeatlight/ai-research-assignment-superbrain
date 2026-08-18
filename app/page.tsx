'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import type { Paper } from '@/lib/types';

interface Library {
  papers: Paper[];
  claimCount: number;
}

export default function Home() {
  const [library, setLibrary] = useState<Library | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  async function refreshLibrary() {
    try {
      const r = await fetch('/api/library');
      if (r.ok) setLibrary(await r.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  function formatStatus(s: any): string {
    const st = s.state || s.status;
    switch (st) {
      case 'uploaded': return 'Uploaded — starting processing…';
      case 'extracting_text': return 'Extracting text…';
      case 'extracting_claims': return `Extracting claims — Batch ${(s.batch ?? 0) + 1}/${s.total_batches ?? '?'}`;
      case 'retrying_batch': return `Batch ${(s.batch ?? 0) + 1} needs recovery — retrying…`;
      case 'recovering': return `Recovering batch ${(s.batch ?? 0) + 1} — processing sub-batches…`;
      case 'recovering_blocks': return 'Recovering difficult blocks…';
      case 'embedding': return 'Generating embeddings…';
      case 'merging': return 'Adding paper to corpus…';
      case 'completed':
      case 'completed_uploaded': return `Paper added successfully — ${s.claims_added ?? 0} claims added.`;
      case 'already_exists': return 'This paper is already in the corpus.';
      case 'failed': return `Paper could not be completely processed. Nothing was added to the corpus.`;
      default: return s.reason || 'Processing…';
    }
  }

  async function handleUpload(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const form = e.currentTarget;
    const input = form.elements.namedItem('pdf') as HTMLInputElement;
    const file = input?.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadStatus('Uploading…');
    const fd = new FormData();
    fd.append('file', file);
    try {
      const res = await fetch('/api/upload', { method: 'POST', body: fd });
      const data = await res.json();
      if (!res.ok) {
        setUploadStatus(data.error || 'Upload failed');
        setUploading(false);
        return;
      }
      setUploadStatus('Uploaded — starting processing…');
      // Track whether we have observed a terminal state so stale poll
      // iterations cannot overwrite the final success/failure message.
      const TERMINAL = ['completed', 'already_exists', 'failed'];
      let terminal = false;
      // Poll the status endpoint until terminal state.
      for (let i = 0; i < 240; i++) {
        await new Promise((r) => setTimeout(r, 1000));
        try {
          const sr = await fetch('/api/upload-status');
          const s = await sr.json();
          const st = s.state || s.status;
          if (st && !terminal) setUploadStatus(formatStatus(s));
          if (st && TERMINAL.includes(st)) {
            terminal = true;
            setUploadStatus(formatStatus(s));
            break;
          }
        } catch {
          // transient poll error; keep polling (never replace a known terminal state)
        }
      }
      // ONE final unconditional read so a just-completed job can never leave the
      // UI stuck on an intermediate message such as "Batch 9/13".
      if (!terminal) {
        try {
          const sr = await fetch('/api/upload-status');
          const s = await sr.json();
          const st = s.state || s.status;
          if (st && TERMINAL.includes(st)) {
            terminal = true;
            setUploadStatus(formatStatus(s));
          }
        } catch {
          // ignore: corpus refresh below still runs
        }
      }
      // Refresh the searchable dataset after completion/already_exists.
      setLibrary(null);
      await refreshLibrary();
    } catch (err) {
      setUploadStatus(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
    }
  }

  useEffect(() => {
    fetch('/api/library')
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then(setLibrary)
      .catch((e) => setError(e.message));
  }, []);

  return (
    <div className="container">
      <header style={{ marginBottom: '2rem' }}>
        <h1 style={{ marginBottom: '0.25rem' }}>Research Claim &amp; Evidence</h1>
        <p className="muted" style={{ margin: 0 }}>
          Explore research claims, ask questions, and inspect whether each answer
          claim is supported by its source evidence.
        </p>
      </header>

      <form
        className="card"
        style={{ marginBottom: '2rem', display: 'flex', gap: '0.75rem' }}
        onSubmit={(e) => {
          e.preventDefault();
          if (query.trim()) {
            window.location.href = `/ask?q=${encodeURIComponent(query.trim())}`;
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
          style={{
            padding: '0.75rem 1.25rem',
            borderRadius: 8,
            border: 'none',
            background: '#3b82f6',
            color: '#fff',
            fontSize: '1rem',
            cursor: 'pointer',
          }}
        >
          Ask
        </button>
      </form>

      <form
        className="card"
        style={{ marginBottom: '2rem', display: 'flex', gap: '0.75rem', alignItems: 'center' }}
        onSubmit={handleUpload}
      >
        <input type="file" name="pdf" accept="application/pdf" style={{ flex: 1 }} />
        <button
          type="submit"
          disabled={uploading}
          style={{
            padding: '0.6rem 1rem',
            borderRadius: 8,
            border: 'none',
            background: '#3b82f6',
            color: '#fff',
            cursor: uploading ? 'default' : 'pointer',
            opacity: uploading ? 0.6 : 1,
          }}
        >
          {uploading ? 'Uploading…' : 'Upload PDF'}
        </button>
        {uploadStatus && <p className="muted" style={{ margin: 0 }}>{uploadStatus}</p>}
      </form>

      <h2 className="muted" style={{ fontSize: '0.9rem', letterSpacing: 1 }}>
        RESEARCH LIBRARY
      </h2>
      {error && <p style={{ color: '#f07b7b' }}>Failed to load library: {error}</p>}
      {!library && !error && <p className="muted">Loading library…</p>}
      {library &&
        library.papers.map((paper) => (
          <div key={paper.paper_id} className="card" style={{ marginBottom: '1rem' }}>
            <h3 style={{ marginTop: 0, marginBottom: '0.5rem' }}>{paper.title}</h3>
            <div style={{ marginBottom: '0.5rem' }}>
              {paper.focus.map((tag) => (
                <span key={tag} className="badge">
                  {tag}
                </span>
              ))}
            </div>
            {paper.authors.length > 0 && (
              <p className="muted" style={{ margin: '0 0 0.5rem' }}>
                {paper.authors.join(', ')}
              </p>
            )}
            <p className="muted" style={{ margin: 0 }}>
              {paper.claimCount} claims
            </p>
          </div>
        ))}
    </div>
  );
}
