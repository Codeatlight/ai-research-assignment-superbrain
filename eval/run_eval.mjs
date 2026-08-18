// Minimal manual evaluation harness.
// Sends each question in eval_questions.json to the /api/ask endpoint and
// prints per-question metrics (answer, number of claims, status distribution).
// Run: node eval/run_eval.mjs  (dev server must be running on :3000)

const BASE = process.env.BASE_URL || 'http://localhost:3000';
const questions = JSON.parse(
  await import('fs').then((fs) => fs.readFileSync('eval/eval_questions.json', 'utf-8')),
);

// Embed the question with the same MiniLM model used at runtime/ingestion.
const { pipeline } = await import('@huggingface/transformers');
const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
async function embed(text) {
  const out = await extractor([text], { pooling: 'mean', normalize: true });
  return out.tolist()[0];
}

const statusCounts = {};
let totalClaims = 0;

for (const { question, expected_claims } of questions) {
  const queryEmbedding = await embed(question);
  const res = await fetch(`${BASE}/api/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, queryEmbedding }),
  });
  const data = await res.json();
  if (!res.ok) {
    console.log(`\nQ: ${question}\n  ERROR: ${data.error}`);
    continue;
  }
  console.log(`\nQ: ${question}`);
  console.log(`  Answer: ${data.answer.slice(0, 200)}`);
  console.log(`  Claims: ${data.claims.length}`);
  for (const c of data.claims) {
    statusCounts[c.status] = (statusCounts[c.status] || 0) + 1;
    totalClaims++;
  }
  // Expected-claim recall (manual, informational).
  const text = data.answer.toLowerCase();
  for (const ec of expected_claims) {
    console.log(`  Expected present: ${text.includes(ec.toLowerCase()) ? 'yes' : 'NO'}`);
  }
}

console.log('\n=== Summary ===');
console.log('Total answer claims:', totalClaims);
console.log('Status distribution:', statusCounts);
