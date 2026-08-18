// Cross-runtime embedding consistency test (Transformers.js / Node side).
// Companion to scripts/embed_consistency_py.py. Reads the vectors the Python
// ingestion embedder wrote to data/embed_test_python.json, embeds the same
// sentences here with @huggingface/transformers (Xenova/all-MiniLM-L6-v2, mean
// pooling + L2 normalization) and reports per-sentence cosine similarity plus a
// unit-norm check (pooling/normalization equivalence).
//
//   python scripts/embed_consistency_py.py   # first
//   node scripts/embed_consistency_js.mjs
import { readFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { pipeline } from '@huggingface/transformers';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const ref = JSON.parse(
  readFileSync(resolve(root, 'data', 'embed_test_python.json'), 'utf-8'),
);
const { sentences, vectors: pyVectors, dim: pyDim, model } = ref;

const EMBEDDING_MODEL = 'Xenova/all-MiniLM-L6-v2';

function norm(v) {
  return Math.sqrt(v.reduce((s, x) => s + x * x, 0));
}
function cosine(a, b) {
  let dot = 0, ma = 0, mb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    ma += a[i] * a[i];
    mb += b[i] * b[i];
  }
  if (ma === 0 || mb === 0) return 0;
  return dot / (Math.sqrt(ma) * Math.sqrt(mb));
}

async function main() {
  if (pyDim !== 384) throw new Error(`Expected 384-dim from Python, got ${pyDim}`);
  const extractor = await pipeline('feature-extraction', EMBEDDING_MODEL);
  const jsVectors = (await extractor(sentences, { pooling: 'mean', normalize: true })).tolist();

  console.log(`Model: JS=${EMBEDDING_MODEL} Python=${model}`);
  console.log(`JS dim=${jsVectors[0].length} Python dim=${pyDim}`);
  console.log('--- per-sentence cosine similarity (JS vs Python) ---');
  let total = 0;
  sentences.forEach((s, i) => {
    const sim = cosine(jsVectors[i], pyVectors[i]);
    total += sim;
    console.log(
      `${sim.toFixed(5)}  |Py|=${norm(pyVectors[i]).toFixed(4)}  |JS|=${norm(jsVectors[i]).toFixed(4)}  :: ${s.slice(0, 60)}`,
    );
  });
  const mean = total / sentences.length;
  console.log(`--- mean cosine similarity: ${mean.toFixed(5)} ---`);
  const ok = mean >= 0.98 && jsVectors.every((v) => Math.abs(norm(v) - 1) < 1e-3);
  console.log(ok ? 'PASS: cross-runtime embedding consistency acceptable' : 'FAIL');
  process.exit(ok ? 0 : 1);
}

main().catch((e) => {
  console.error('Error:', e.message);
  process.exit(1);
});
