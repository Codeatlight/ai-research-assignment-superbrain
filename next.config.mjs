import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const transformersDir = resolve(
  __dirname,
  'node_modules/@huggingface/transformers/dist',
);

/** @type {import('next').NextConfig} */
const nextConfig = {
  // Dataset is shipped as a static public asset, not bundled into the
  // serverless function, so Vercel function size stays small.
  output: 'standalone',
  // Ensure the corpus is present in the serverless function bundle. API routes
  // read public/data/claims.json via fs.readFileSync at runtime; without this,
  // Vercel's output file tracing can omit it and /api/ask fails with a 500
  // before any Groq request is made.
  outputFileTracingIncludes: {
    '/api/**/*': ['./public/data/claims.json'],
  },
  webpack: (config) => {
    // Force the browser (WASM) build of @huggingface/transformers so webpack
    // does not resolve the "node" condition and bundle onnxruntime-node's
    // native .node binaries into the client bundle.
    config.resolve.alias['@huggingface/transformers'] = resolve(
      transformersDir,
      'transformers.web.js',
    );
    return config;
  }
};

export default nextConfig;
