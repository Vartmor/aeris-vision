import { readdir, stat } from 'node:fs/promises';
import { extname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../dist/', import.meta.url));
const formats = new Set(['.css', '.html', '.ico', '.jpeg', '.jpg', '.js', '.mjs', '.png', '.svg', '.txt', '.webp', '.woff', '.woff2']);
const privateFolders = new Set(['data', 'outputs', 'checkpoints', 'node_modules', 'src', 'branding']);
const approvedText = new Set(['licenses/lato.txt', 'licenses/noto-serif.txt', 'licenses/noto-sans-mono.txt']);
const entries = await readdir(root, { withFileTypes: true, recursive: true });
for (const entry of entries) {
  const path = join(entry.parentPath, entry.name);
  const relative = path.slice(root.length).split('\\').join('/');
  if (entry.isSymbolicLink() || entry.name.startsWith('.') || (entry.isDirectory() && privateFolders.has(entry.name))) {
    throw new Error(`Yayın çıktısında beklenmeyen yol: ${relative}`);
  }
  if (!entry.isDirectory() && (
    !entry.isFile() || !formats.has(extname(entry.name))
    || (entry.name.endsWith('.txt') && !approvedText.has(relative))
  )) {
    throw new Error(`Yayın çıktısında beklenmeyen dosya biçimi: ${relative}`);
  }
}
for (const route of ['index.html', 'gunluk/index.html', 'galeri/index.html']) {
  if (!(await stat(join(root, route))).isFile()) throw new Error(`Yayın sayfası eksik: ${route}`);
}
console.log(`Yayına hazır: ${entries.filter((entry) => entry.isFile()).length} statik dosya.`);
