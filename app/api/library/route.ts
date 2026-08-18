import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import type { Dataset, Paper } from '@/lib/types';

const DATASET_PATH = path.join(process.cwd(), 'public', 'data', 'claims.json');

export interface PaperWithClaims extends Paper {
  claimCount: number;
}

let cache: { papers: PaperWithClaims[]; claimCount: number } | null = null;

function load() {
  if (!cache) {
    const raw = fs.readFileSync(DATASET_PATH, 'utf-8');
    const dataset = JSON.parse(raw) as Dataset;
    // Per-paper claim counts grouped by paper_id.
    const counts = new Map<string, number>();
    for (const c of dataset.claims) {
      counts.set(c.paper_id, (counts.get(c.paper_id) ?? 0) + 1);
    }
    const papers = dataset.papers.map((p) => ({
      ...p,
      claimCount: counts.get(p.paper_id) ?? 0,
    }));
    cache = { papers, claimCount: dataset.claims.length };
  }
  return cache;
}

export async function GET() {
  try {
    const { papers, claimCount } = load();
    return NextResponse.json({ papers, claimCount });
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : String(err) },
      { status: 500 },
    );
  }
}
