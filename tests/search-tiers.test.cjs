// 两层搜索索引：聚焦只加载轻量索引；全文索引只在点「搜索全文」后加载并复用（编号 9）
const assert = require('node:assert/strict');
const { test } = require('node:test');
const { searchHarness, flush, response } = require('./helpers/search-harness.cjs');

const lite = {
  meta: { version: 'v1', tier: 'lite', N: 2, full_index_gzip_bytes: 27 * 1048576 },
  docs: [
    { id: 'ppo', title: 'PPO', path: 'wiki/ppo.md', summary: '策略优化', tags: ['rl'] },
    { id: 'slam', title: 'SLAM', path: 'wiki/slam.md', summary: '定位建图', tags: [] },
  ],
};
// 全文索引写 df 而不是 idf；"airtime" 只出现在 ppo 正文里
const full = {
  meta: { version: 'v1', tier: 'full', N: 2, avgdl: 3, k1: 1.5, b: 0.75 },
  df: { ppo: 1, slam: 1, airtime: 1 },
  docs: [
    { id: 'ppo', title: 'PPO', path: 'wiki/ppo.md', summary: '策略优化', tags: ['rl'], dl: 3, tokens: { ppo: 1, airtime: 2 } },
    { id: 'slam', title: 'SLAM', path: 'wiki/slam.md', summary: '定位建图', tags: [], dl: 2, tokens: { slam: 1 } },
  ],
};

function tiered() {
  const calls = [];
  const h = searchHarness(
    async () => { calls.push('lite'); return response(lite); },
    undefined,
    { fetchFull: async () => { calls.push('full'); return response(full); } },
  );
  return { h, calls };
}
const clickFullText = (h) => h.searchResults.handlers.click({ target: { closest: (selector) => selector === '.search-full-text' } });

test('focus and a title hit use only the lite index and offer full-text search', async () => {
  const { h, calls } = tiered();
  h.searchInput.handlers.focus();
  await flush();
  assert.deepEqual(calls, ['lite']);
  h.searchInput.value = 'PPO'; h.triggerSearch();
  await flush();
  assert.match(h.searchResults.innerHTML, /detail\.html\?id=ppo/);
  assert.match(h.searchResults.innerHTML, /search-full-text/, 'results carry a 搜索全文 button');
  assert.match(h.searchResults.innerHTML, /约 27 MB/, 'the download size comes from the lite meta');
  assert.deepEqual(calls, ['lite'], 'no full index download without the user asking');
});

test('a body-only term finds nothing in the lite tier until the user asks for full text', async () => {
  const { h, calls } = tiered();
  h.searchInput.value = 'airtime'; h.triggerSearch();
  await flush();
  assert.match(h.searchResults.innerHTML, /未找到/);
  assert.match(h.searchResults.innerHTML, /search-full-text/);
  assert.deepEqual(calls, ['lite']);
  clickFullText(h);
  await flush(); await flush();
  assert.deepEqual(calls, ['lite', 'full'], 'the full index is fetched once on demand');
  assert.match(h.searchResults.innerHTML, /detail\.html\?id=ppo/, 'BM25 over body tokens with df-derived idf finds the page');
  assert.doesNotMatch(h.searchResults.innerHTML, /search-full-text/, 'no further upsell once full text is active');
  h.searchInput.value = 'SLAM'; h.triggerSearch();
  await flush();
  assert.match(h.searchResults.innerHTML, /detail\.html\?id=slam/);
  assert.deepEqual(calls, ['lite', 'full'], 'later searches reuse the loaded full index');
});

test('a failed full-text download is reported with a retry and does not poison the next attempt', async () => {
  let fail = true;
  const calls = [];
  const h = searchHarness(
    async () => response(lite),
    undefined,
    // 该 harness 的 response() 恒为 ok，失败响应需手写
    { fetchFull: async () => { calls.push('full'); if (fail) return { ok: false, status: 503 }; return response(full); } },
  );
  h.searchInput.value = 'airtime'; h.triggerSearch();
  await flush();
  clickFullText(h);
  await flush(); await flush();
  assert.match(h.searchResults.innerHTML, /全文索引暂时无法加载/);
  assert.match(h.searchResults.innerHTML, /search-full-text/);
  fail = false;
  clickFullText(h);
  await flush(); await flush();
  assert.deepEqual(calls, ['full', 'full']);
  assert.match(h.searchResults.innerHTML, /detail\.html\?id=ppo/);
});

test('legacy full indexes that still carry idf keep scoring', async () => {
  const legacy = { meta: { version: 'v1', avgdl: 3, N: 2 }, idf: { airtime: 1.2 }, docs: full.docs };
  const h = searchHarness(async () => response(legacy));
  h.searchInput.value = 'airtime'; h.triggerSearch();
  await flush();
  assert.match(h.searchResults.innerHTML, /detail\.html\?id=ppo/);
  assert.doesNotMatch(h.searchResults.innerHTML, /search-full-text/, 'an index without tier=lite is treated as full');
});
