// Browser embedding via Transformers.js using the SAME weights as the Python
// ingestion pipeline (scripts/embed.py): all-MiniLM-L6-v2, 384-dim, mean
// pooling + L2 normalization. Query vectors computed here are comparable to
// stored claim vectors at query time.
//
// The transformers module is imported dynamically so it is never evaluated
// during server-side prerendering (it constructs WASM URLs that fail in Node).
export const EMBEDDING_MODEL = 'Xenova/all-MiniLM-L6-v2';
export const EMBEDDING_DIM = 384;

type FeatureExtractor = (
  texts: string[],
  opts: { pooling: 'mean'; normalize: boolean },
) => Promise<{ tolist(): number[][] }>;

let extractorPromise: Promise<FeatureExtractor> | null = null;

async function getExtractor(): Promise<FeatureExtractor> {
  if (!extractorPromise) {
    const { pipeline } = await import('@huggingface/transformers');
    extractorPromise = pipeline('feature-extraction', EMBEDDING_MODEL) as Promise<
      FeatureExtractor
    >;
  }
  return extractorPromise;
}

export async function embedTexts(texts: string[]): Promise<number[][]> {
  if (texts.length === 0) return [];
  const extractor = await getExtractor();
  // pooling 'mean' + normalize true matches sentence-transformers defaults.
  const output = await extractor(texts, { pooling: 'mean', normalize: true });
  return output.tolist();
}

