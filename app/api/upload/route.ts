import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import { writeFileSync, mkdirSync, existsSync, readFileSync } from 'fs';
import { join } from 'path';

// LOCAL/DEV-only upload route. Accepts a PDF, saves it to a temp location,
// starts the local ingestion worker (scripts/ingest_upload.py) in the background,
// and returns a job_id immediately WITHOUT blocking for the ~3-4 min ingestion.
// The UI polls /api/upload-status to read progress from data/.upload_status.json.
// This does NOT persist on Vercel serverless (ephemeral filesystem).

const LOCK = join(process.cwd(), 'data', '.upload_lock');
const STATUS_FILE = join(process.cwd(), 'data', '.upload_status.json');

function readStatus(): any {
  try {
    return existsSync(STATUS_FILE) ? JSON.parse(readFileSync(STATUS_FILE, 'utf-8')) : {};
  } catch {
    return {};
  }
}

export async function POST(req: NextRequest) {
  try {
    // Single-user/local-dev concurrency guard: reject a second upload while one runs.
    if (existsSync(LOCK)) {
      return NextResponse.json(
        { error: 'An upload is already being processed. Please wait for it to finish.' },
        { status: 409 },
      );
    }

    const form = await req.formData();
    const file = form.get('file');
    if (!(file instanceof File)) {
      return NextResponse.json({ error: 'A PDF file is required.' }, { status: 400 });
    }
    if (file.type !== 'application/pdf' && !file.name.toLowerCase().endsWith('.pdf')) {
      return NextResponse.json({ error: 'File must be a PDF.' }, { status: 400 });
    }
    if (file.size > 30 * 1024 * 1024) {
      return NextResponse.json({ error: 'File too large (max 30 MB).' }, { status: 400 });
    }
    const buf = Buffer.from(await file.arrayBuffer());
    if (buf.length === 0) {
      return NextResponse.json({ error: 'File is empty.' }, { status: 400 });
    }

    const tmpDir = join(process.cwd(), 'data', 'uploads');
    mkdirSync(tmpDir, { recursive: true });
    const jobId = `${Date.now()}`;
    const safeName = file.name.replace(/[^a-zA-Z0-9._-]/g, '_');
    const pdfPath = join(tmpDir, `${jobId}-${safeName}`);
    writeFileSync(pdfPath, buf);

    // Set the lock so a second upload is rejected while this one runs.
    writeFileSync(LOCK, jobId);
    // Initialize status to 'uploaded'.
    writeFileSync(STATUS_FILE, JSON.stringify({ state: 'uploaded', job_id: jobId }));

    const worker = join(process.cwd(), 'scripts', 'ingest_upload.py');
    const pythonCmd = process.env.PYTHON ?? 'python';

    // Fire-and-forget: spawn the worker detached, do NOT await it.
    const child = spawn(pythonCmd, [worker, pdfPath], {
      cwd: process.cwd(),
      detached: true,
      stdio: 'ignore',
    });
    child.unref();

    return NextResponse.json({ job_id: jobId, status: 'accepted' });
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : String(err) },
      { status: 500 },
    );
  }
}

