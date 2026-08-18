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
