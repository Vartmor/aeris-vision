import { readFile, writeFile } from 'node:fs/promises';
import { journalFigures } from '../src/content/journal.ts';

const readme = new URL('../../README.md', import.meta.url);
const start = '<!-- journal-gallery:start -->';
const end = '<!-- journal-gallery:end -->';
const escape = (text) => text.replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);

const cards = journalFigures.slice(0, 4).map(({ entry, figure }) => {
  const image = `site/public/media/gunluk/${encodeURIComponent(figure.file)}`;
  const link = `https://vartmor.github.io/aeris-vision/galeri/?yazi=${entry.number}`;
  return `<td width="50%" valign="top">
  <a href="${image}"><img src="${image}" alt="${escape(figure.alt)}" width="480"></a>
  <br><a href="${link}">${entry.number}. ${escape(entry.title)}</a>
  <br>${escape(figure.caption)}
</td>`;
});

const rows = [];
for (let index = 0; index < cards.length; index += 2) {
  rows.push(`<tr>\n${cards.slice(index, index + 2).join('\n')}\n</tr>`);
}
const block = [start, '<table>', ...rows, '</table>', end].join('\n');
const previous = await readFile(readme, 'utf8');
const from = previous.indexOf(start);
const to = previous.indexOf(end, from + start.length);
if (from < 0 || to < 0) throw new Error('README dosyasında galeri işaretleri eksik');
const next = previous.slice(0, from) + block + previous.slice(to + end.length);

if (process.argv.includes('--check')) {
  if (next !== previous) {
    console.error('README galerisi güncel değil. site dizininde npm run readme:gallery çalıştır.');
    process.exitCode = 1;
  }
} else {
  await writeFile(readme, next);
}
