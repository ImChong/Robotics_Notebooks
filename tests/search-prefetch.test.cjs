// 搜索索引只在搜索意图出现时加载：空闲时间本身不触发下载（编号 4）
const assert = require('node:assert/strict');
const { test } = require('node:test');
const { searchHarness, indexData, flush, response } = require('./helpers/search-harness.cjs');

function countingIndex() {
  const calls = [];
  return {
    calls,
    fetchIndex: async () => {
      calls.push('search-index.json');
      return response(indexData);
    },
  };
}

test('idle time and timers alone never download the search index', async () => {
  const { calls, fetchIndex } = countingIndex();
  const context = searchHarness(fetchIndex);
  context.runIdle();
  context.expire(1200);
  context.expire(2500);
  await flush();
  assert.deepEqual(calls, []);
});

test('focusing the search box loads the index once and reuses it', async () => {
  const { calls, fetchIndex } = countingIndex();
  const context = searchHarness(fetchIndex);
  context.searchInput.handlers.focus();
  await flush();
  assert.equal(calls.length, 1);
  context.searchInput.handlers.focus();
  await flush();
  assert.equal(calls.length, 1);
});

test('the #wiki-search deep link still prefetches on arrival', async () => {
  const { calls, fetchIndex } = countingIndex();
  searchHarness(fetchIndex, undefined, { hash: '#wiki-search' });
  await flush();
  assert.equal(calls.length, 1);
});
