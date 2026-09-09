// 图谱侧栏改读无正文的站点目录，论文笔记链接不再依赖正文扫描（编号 3）
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

const source = readFileSync(resolve(__dirname, '../docs/graph.html'), 'utf8');
const start = source.indexOf('    const NOTEBOOK_SITE_PREFIXES = [');
const end = source.indexOf('    /* ── 悬停高亮 ── */', start);
assert.ok(start >= 0 && end > start, 'sidebar link helpers are located');

const context = vm.createContext({ Set, String, Array });
vm.runInContext(source.slice(start, end), context);

const notebook = (name) =>
  `https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/${name}/${name}.html`;

test('a catalog item without a body still yields its notebook links', () => {
  const links = context.collectPaperNotebookLinks({
    title: 'Demo',
    paper_notebook_links: [{ url: notebook('Demo'), label: 'Demo 笔记' }],
    source_links: [{ url: notebook('Other'), label: 'Other' }, { url: 'https://arxiv.org/abs/1', label: 'arXiv' }],
  });
  assert.deepEqual([...links.map((l) => l.url)], [notebook('Demo'), notebook('Other')]);
  assert.equal(links[0].label, 'Demo 笔记');
});

test('duplicates across the two fields collapse and non-notebook links are dropped', () => {
  const links = context.collectPaperNotebookLinks({
    title: 'Demo',
    paper_notebook_links: [{ url: `${notebook('Demo')}#section`, label: 'Demo' }],
    source_links: [{ url: notebook('Demo'), label: 'Demo again' }, 'https://github.com/x/y'],
  });
  assert.deepEqual([...links.map((l) => l.url)], [notebook('Demo')]);
});

test('an item with no link fields renders no notebook section', () => {
  assert.equal(context.collectPaperNotebookLinks({ title: 'Demo' }).length, 0);
  assert.equal(context.collectPaperNotebookLinks(null).length, 0);
});

test('the graph page loads the body-free catalog instead of the full index', () => {
  assert.ok(source.includes("fetch('exports/site-catalog-v1.json')"), 'sidebar metadata comes from the catalog');
  assert.ok(!source.includes("fetch('exports/index-v1.json')"), 'the body-carrying export is no longer requested');
  assert.ok(source.includes('d.pages && d.pages.detail_pages'), 'catalog pages are mapped into sidebar items');
  assert.ok(!/\.content_markdown/.test(source), 'no code path reads a page body any more');
});
