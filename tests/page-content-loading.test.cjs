const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');
const source = readFileSync(resolve(__dirname, '../docs/main.js'), 'utf8');
const catalogUrl = 'exports/site-catalog-v1.json';
const bodyUrl = `exports/page-content/${'a'.repeat(64)}.json`;
const routeUrl = `exports/page-content/${'b'.repeat(64)}.json`;
const catalog = () => ({ version: 'v1', content_mode: 'per-page-v1', pages: {
  detail_pages: {
    'entity-demo': { id: 'entity-demo', path: 'wiki/entities/demo.md', summary: 'preview', content_url: bodyUrl },
    'roadmap-motion-control': { id: 'roadmap-motion-control', type: 'roadmap_page', content_url: routeUrl },
  },
  roadmap_pages: { 'roadmap-motion-control': { stages: ['intro'] } },
  page_aliases: { old: 'entity-demo' },
} });
const body = (id = 'entity-demo') => ({ version: 'v1', id, content_markdown: '# Article\n\n[Link](../demo.md)' });
const response = (data, status = 200) => ({ ok: status === 200, status, json: async () => data });
const tick = () => new Promise(setImmediate);

function harness(fetcher, runtime = false) {
  const calls = [], timers = new Map(), renders = [];
  let timerId = 0;
  const button = { disabled: false, addEventListener(_event, fn) { this.click = fn; } };
  const summary = { innerHTML: '', querySelector: () => button };
  const elements = { detailTitle: {}, detailSummary: summary, detailContent: { innerHTML: '' } };
  const context = vm.createContext({
    Map, URLSearchParams, AbortController,
    window: {
      location: { search: '?id=entity-demo', hash: '' },
      setTimeout: (fn) => { timers.set(++timerId, fn); return timerId; },
      clearTimeout: (id) => timers.delete(id),
    },
    fetch: async (url, options) => { calls.push(url); return fetcher(url, options, calls); },
    document: { getElementById: (id) => elements[id] || null },
    removeLoadingState() {}, escapeHtml: (s) => s,
    renderDetailPage: (data) => renders.push(data),
  });
  vm.runInContext(source.slice(source.indexOf('  function resolveDetailPage('), source.indexOf('  var DETAIL_MINI_TABLEAU10')), context);
  const start = source.indexOf('  var pageDataRequests =');
  const end = source.indexOf(runtime ? '  if (homeStatsRoot) {' : '  function handlePageDataError(', start);
  vm.runInContext(source.slice(start, end), context);
  return { context, calls, timers, renders, button, summary, load: context.loadSitePageData };
}
const goodFetch = async (url) => response(url === catalogUrl ? catalog() : body(url === routeUrl ? 'roadmap-motion-control' : 'entity-demo'));

test('detail loads one body, shares concurrent requests and leaves previews metadata-only', async () => {
  const h = harness(goodFetch);
  const [a, b] = await Promise.all([h.load('detail', 'entity-demo'), h.load('detail', 'entity-demo')]);
  assert.deepEqual(h.calls, [catalogUrl, bodyUrl]);
  assert.equal(a.pages.detail_pages['entity-demo'].content_markdown, body().content_markdown);
  assert.equal(b.pages.detail_pages['entity-demo'].content_markdown, body().content_markdown);
  assert.equal(h.context.loadedPageCatalog.pages.detail_pages['entity-demo'].content_markdown, undefined);
  assert.equal(a.pages.detail_pages['roadmap-motion-control'].content_markdown, undefined);
  assert.equal(a.pages.detail_pages['entity-demo'].summary, 'preview');
  assert.equal(h.timers.size, 0);
});

test('module, preview, unknown IDs and detail-to-roadmap redirects need no body', async () => {
  const h = harness(goodFetch);
  await h.load('catalog', 'control');
  for (const id of ['../invalid', 'constructor', '__proto__']) await h.load('detail', id);
  await h.load('roadmap', 'missing');
  await h.load('detail', 'roadmap-motion-control');
  assert.deepEqual(h.calls, [catalogUrl]);
});

test('aliases and legacy entity IDs resolve to the same canonical body', async () => {
  const h = harness(goodFetch);
  for (const id of ['old', 'wiki-entities-demo', 'entity-demo']) {
    const data = await h.load('detail', id);
    assert.equal(data.pages.detail_pages['entity-demo'].content_markdown, body().content_markdown);
  }
  assert.deepEqual(h.calls, [catalogUrl, bodyUrl]);
});

test('legacy roadmap loads its own body and preserves stages', async () => {
  const h = harness(goodFetch);
  const data = await h.load('roadmap', 'roadmap-route-a-motion-control');
  assert.deepEqual(h.calls, [catalogUrl, routeUrl]);
  assert.equal(data.pages.detail_pages['roadmap-motion-control'].content_markdown, body().content_markdown);
  assert.deepEqual(data.pages.roadmap_pages['roadmap-motion-control'].stages, ['intro']);
});

test('HTTP, malformed JSON and mismatched body failures do not poison retries', async () => {
  for (const failed of [response(null, 503), { ok: true, json: async () => { throw new SyntaxError('bad JSON'); } }, response(body('wrong-id'))]) {
    let fail = true;
    const h = harness(async (url) => url === catalogUrl ? response(catalog()) : fail ? failed : goodFetch(url));
    await assert.rejects(h.load('detail', 'entity-demo'));
    fail = false;
    await h.load('detail', 'entity-demo');
    assert.deepEqual(h.calls, [catalogUrl, bodyUrl, bodyUrl]);
    assert.equal(h.timers.size, 0);
  }
});

test('404 after a deployment refreshes the catalog and uses its new hash', async () => {
  const freshUrl = `exports/page-content/${'c'.repeat(64)}.json`;
  let catalogs = 0;
  const h = harness(async (url) => {
    if (url === catalogUrl) {
      const data = catalog();
      if (++catalogs > 1) data.pages.detail_pages['entity-demo'].content_url = freshUrl;
      return response(data);
    }
    return url === bodyUrl ? response(null, 404) : response(body());
  });
  await h.load('detail', 'entity-demo');
  assert.deepEqual(h.calls, [catalogUrl, bodyUrl, catalogUrl, freshUrl]);
});

test('a persistent 404 stops after one refresh and remains retryable', async () => {
  const h = harness(async (url) => url === catalogUrl ? response(catalog()) : response(null, 404));
  await assert.rejects(h.load('detail', 'entity-demo'));
  assert.deepEqual(h.calls, [catalogUrl, bodyUrl, catalogUrl, bodyUrl]);
});

test('timeout aborts pending body and a later request recovers', async () => {
  let hang = true;
  const h = harness(async (url, { signal }) => {
    if (url === catalogUrl || !hang) return goodFetch(url);
    return new Promise((_resolve, reject) => signal.addEventListener('abort', () => reject(new Error('aborted'))));
  });
  const loading = h.load('detail', 'entity-demo');
  const rejected = assert.rejects(loading, /aborted/);
  await tick();
  [...h.timers.values()].forEach((fn) => fn());
  await rejected;
  hang = false;
  await h.load('detail', 'entity-demo');
  assert.equal(h.timers.size, 0);
});

test('rejects external and traversal body URLs without requesting them', async () => {
  for (const url of ['https://example.org/body.json', 'exports/page-content/../../secret.json']) {
    const h = harness(async () => { const data = catalog(); data.pages.detail_pages['entity-demo'].content_url = url; return response(data); });
    await assert.rejects(h.load('detail', 'entity-demo'), /Invalid content URL/);
    assert.deepEqual(h.calls, [catalogUrl]);
  }
});

test('real page startup offers retry/source and renders after the retry button is clicked', async () => {
  let fail = true;
  const h = harness(async (url) => url !== catalogUrl && fail ? response(null, 503) : goodFetch(url), true);
  await tick();
  assert.equal(h.renders.length, 0);
  assert.match(h.summary.innerHTML, /打开原文/);
  assert.match(h.summary.innerHTML, /wiki\/entities\/demo.md/);
  fail = false;
  h.button.click({ currentTarget: h.button });
  await tick();
  assert.equal(h.renders.length, 1);
  assert.equal(h.renders[0].pages.detail_pages['entity-demo'].content_markdown, body().content_markdown);
});

function installRouting(context) {
  vm.runInContext(source.slice(source.indexOf('  function detailHref('), source.indexOf('  function latestNodeHref(')), context);
  vm.runInContext(source.slice(source.indexOf('  function moduleHref('), source.indexOf('  function unescapeMarkdownEscapes(')), context);
}

test('metadata-only catalog resolves Markdown deep links with anchors', async () => {
  const h = harness(goodFetch);
  const data = await h.load('catalog', '');
  installRouting(h.context);
  const routes = h.context.buildMarkdownRouteIndex(data);
  assert.equal(h.context.resolveInternalMarkdownHref('../entities/demo.md#section', 'wiki/concepts/current.md', routes), 'detail.html?id=entity-demo#section');
  assert.deepEqual(h.calls, [catalogUrl]);
});

test('actual detail and roadmap renderers retain alias redirects and fragments', async () => {
  const h = harness(goodFetch);
  const data = await h.load('catalog', '');
  installRouting(h.context);
  const redirects = [];
  h.context.window.location.replace = (href) => redirects.push(href);
  h.context.window.location.hash = '#section';
  vm.runInContext(source.slice(source.indexOf('  function renderDetailPage('), source.indexOf('  function renderModulePage(')), h.context);
  vm.runInContext(source.slice(source.indexOf('  function renderRoadmapPage('), source.indexOf('  function renderRoadmapMarkdownBody(')), h.context);
  h.context.window.location.search = '?id=old';
  h.context.renderDetailPage(data);
  h.context.window.location.search = '?id=roadmap-motion-control';
  h.context.renderDetailPage(data);
  h.context.window.location.search = '?id=roadmap-route-a-motion-control';
  h.context.renderRoadmapPage(data);
  assert.deepEqual(redirects, ['detail.html?id=entity-demo#section', 'roadmap.html?id=roadmap-motion-control#section', 'roadmap.html?id=roadmap-motion-control#section']);
  assert.deepEqual(h.calls, [catalogUrl]);
});

test('actual inline preview renders title, summary and destination without fetching a body', async () => {
  const h = harness(goodFetch);
  const data = await h.load('catalog', '');
  installRouting(h.context);
  vm.runInContext(readFileSync(resolve(__dirname, '../docs/graph-tooltip.js'), 'utf8'), h.context);
  vm.runInContext(source.slice(source.indexOf('  var ROADMAP_KMAP_PATH_TYPE'), source.indexOf('  function collectDepthBranchRoadmaps(')), h.context);
  vm.runInContext(source.slice(source.indexOf('  function formatGraphTooltipSummary('), source.indexOf('  function buildGraphNodeTooltipHtml(')), h.context);
  vm.runInContext(source.slice(source.indexOf('  var detailLinkBridge ='), source.indexOf('  function collectInlineLinkPreviewRoots(')), h.context);
  const html = h.context.buildDetailInlineLinkTooltipHtml('entity-demo', data.pages.detail_pages['entity-demo']);
  assert.match(html, /preview/);
  assert.match(html, /entity-demo/);
  assert.match(html, /detail.html\?id=entity-demo/);
  assert.deepEqual(h.calls, [catalogUrl]);
});
