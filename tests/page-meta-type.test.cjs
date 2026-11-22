const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const { JSDOM } = require('jsdom');

const read = file => readFileSync(resolve(__dirname, '../docs', file), 'utf8');
const scripts = ['wiki-type-labels.js', 'main.js'].map(read);
const contentUrl = `exports/page-content/${'a'.repeat(64)}.json`;

async function loadPage(t, kind, { type, path }, detailExists = true) {
  const id = kind === 'roadmap' ? 'roadmap-demo' : 'demo';
  const dom = new JSDOM(read(`${kind}.html`), {
    url: `https://example.test/${kind}.html?id=${id}`,
    runScripts: 'outside-only',
    pretendToBeVisual: true,
  });
  const { window } = dom;
  t.after(() => window.close());
  window.matchMedia = () => ({ matches: false });

  const updated = '2026-11-22';
  const catalog = {
    version: 'v1', content_mode: 'per-page-v1', pages: {
      detail_pages: detailExists ? { [id]: { title: 'Test page', type, path, updated, content_url: contentUrl } } : {},
      roadmap_pages: kind === 'roadmap' ? { [id]: { title: 'Test roadmap', path, updated } } : {},
      page_aliases: {},
    },
  };
  window.fetch = async (url) => {
    if (url === 'exports/site-catalog-v1.json') return { ok: true, json: async () => catalog };
    if (url === contentUrl) return { ok: true, json: async () => ({ version: 'v1', id, content_markdown: '' }) };
    return { ok: false, status: 503 };
  };
  for (const script of scripts) window.eval(script);
  await new Promise(setImmediate);
  return window.document;
}

test('detail metadata displays fine-grained and fallback types without the graph', async (t) => {
  const cases = [
    ['wiki_page', 'wiki/concepts/demo.md', '概念 (Concept)'],
    ['wiki_page', 'wiki/methods/demo.md', '方法 (Method)'],
    ['wiki_page', 'wiki/tasks/demo.md', '任务 (Task)'],
    ['wiki_page', 'wiki/comparisons/demo.md', '对比 (Comparison)'],
    ['wiki_page', 'wiki/formalizations/demo.md', '形式化 (Formalization)'],
    ['wiki_page', 'wiki/queries/demo.md', '查询 (Query)'],
    ['wiki_page', 'wiki/overview/demo.md', '总览 (Overview)'],
    ['entity_page', 'wiki/entities/demo.md', '实体 (Entity)'],
    ['reference_page', 'references/papers/demo.md', '参考 (Reference)'],
    ['tech_map_node', 'tech-map/overview.md', '技术节点 (Tech Node)'],
    ['wiki_page', undefined, '知识页 (Wiki)'],
  ];
  for (const [type, path, label] of cases) {
    const document = await loadPage(t, 'detail', { type, path });
    const row = document.getElementById('detailMetaType');
    assert.equal(row.hidden, false, label);
    assert.equal(row.querySelector('.detail-meta-badge').textContent, label);
    assert.equal(row.previousElementSibling.id, 'detailMetaUpdated');
    assert.equal(row.previousElementSibling.hidden, false);
    assert.equal(row.nextElementSibling.id, 'detailMetaCommunity');
  }
});

test('roadmap metadata displays its type even without a counterpart in the detail catalog', async (t) => {
  const document = await loadPage(t, 'roadmap', { path: 'roadmap/demo.md' }, false);
  const row = document.getElementById('roadmapMetaType');
  assert.equal(row.hidden, false);
  assert.equal(row.querySelector('.detail-meta-badge').textContent, '路线 (Roadmap)');
  assert.equal(row.previousElementSibling.id, 'roadmapMetaUpdated');
  assert.equal(row.previousElementSibling.hidden, false);
  assert.equal(row.nextElementSibling.id, 'roadmapMetaCommunity');
});

test('a missing detail page has no type badge', async (t) => {
  const document = await loadPage(t, 'detail', {}, false);
  assert.equal(document.querySelector('#detailMeta .detail-meta-type'), null);
  assert.equal(document.getElementById('detailEmptyState').hidden, false);
});
