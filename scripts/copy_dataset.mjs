// Copies the single source-of-truth dataset (data/claims.json) into the
// public asset folder (public/data/claims.json) so the Next.js app and
// serverless routes can read it. Run automatically before build/dev via
// package.json. Never edit public/data/claims.json by hand.
import { copyFileSync, existsSync, mkdirSync } from 'fs';
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const src = resolve(root, 'data', 'claims.json');
const dest = resolve(root, 'public', 'data', 'claims.json');

if (!existsSync(src)) {
  console.warn(
    'WARNING: data/claims.json not found. Run `npm run ingest` first. Building without a dataset.',
  );
} else {
  mkdirSync(dirname(dest), { recursive: true });
  copyFileSync(src, dest);
  console.log(`Copied dataset to ${dest}`);
}
