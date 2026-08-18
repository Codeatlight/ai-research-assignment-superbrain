import { NextRequest, NextResponse } from 'next/server';
import { existsSync, readFileSync } from 'fs';
import { join } from 'path';

// LOCAL/DEV status endpoint. Reads the worker's progress from
// data/.upload_status.json and returns ONLY sanitized status fields (no API
// keys, stack traces, raw Groq responses, or internal paths).
export async function GET(req: NextRequest) {
  const statusFile = join(process.cwd(), 'data', '.upload_status.json');
  if (!existsSync(statusFile)) {
    return NextResponse.json({ state: 'none' });
  }
  let raw: any = {};
  try {
    raw = JSON.parse(readFileSync(statusFile, 'utf-8'));
  } catch {
    return NextResponse.json({ state: 'unknown' });
  }
  // Whitelist only fields the UI needs.
  const sanitized: Record<string, any> = {};
  for (const k of [
    'state',
    'job_id',
    'batch',
    'total_batches',
    'claims_added',
    'paper_title',
    'reason',
    'status',
  ]) {
    if (raw[k] !== undefined) sanitized[k] = raw[k];
  }
  return NextResponse.json(sanitized);
}
