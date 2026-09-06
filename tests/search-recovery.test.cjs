const assert = require('node:assert/strict');
const { test } = require('node:test');
const { searchHarness, indexData, deferred, flush, response } = require('./helpers/search-harness.cjs');

test('search shares in-flight loading and retries after network / HTTP / JSON failures', async () => {
  for (const failure of [
    () => Promise.reject(new Error('offline')),
    () => Promise.resolve({ ok: false, status: 503 }),
    () => Promise.resolve({ ok: true, json: async () => { throw new Error('invalid JSON'); } }),
  ]) {
    let calls = 0;
    const pending = deferred();
    const h = searchHarness(() => ++calls === 1 ? failure() : pending.promise);
    await assert.rejects(h.ensureSearchIndex());
    const first = h.ensureSearchIndex();
    assert.equal(first, h.ensureSearchIndex());
    pending.resolve(response(indexData));
    assert.equal(await first, indexData);
    assert.equal(await h.ensureSearchIndex(), indexData);
    assert.equal(calls, 2);
  }
});

test('search timeout aborts loading and permits a new attempt', async () => {
  let calls = 0;
  const h = searchHarness(({ signal }) => ++calls === 1
    ? new Promise((_, reject) => signal.addEventListener('abort', () => reject(new Error('timeout'))))
    : Promise.resolve(response(indexData)));
  const rejected = assert.rejects(h.ensureSearchIndex(), /timeout/);
  h.expire(30000);
  await rejected;
  assert.equal(await h.ensureSearchIndex(), indexData);
});

test('community timeout does not block search permanently and can retry', async () => {
  let attempts = 0;
  const h = searchHarness(async () => response(indexData), ({ signal }) => ++attempts === 1
    ? new Promise((_, reject) => signal.addEventListener('abort', () => reject(new Error('timeout'))))
    : Promise.resolve(response({ nodes: [{ id: 'wiki/ppo.md', community: 'control' }] })));
  h.searchInput.value = 'PPO'; h.triggerSearch();
  await flush();
  h.expire(30000);
  await flush();
  assert.match(h.searchResults.innerHTML, /detail\.html\?id=ppo/);
  assert.equal((await h.ensureCommunityByPath()).get('wiki/ppo.md'), 'control');
});

test('visible retry button recovers without a page reload', async () => {
  let calls = 0;
  const h = searchHarness(async () => {
    if (++calls === 1) throw new Error('offline');
    return response(indexData);
  });
  h.searchInput.value = 'PPO'; h.triggerSearch();
  await flush();
  assert.match(h.searchResults.innerHTML, /重试搜索/);
  assert.doesNotMatch(h.searchResults.innerHTML, /python3/);
  h.searchResults.handlers.click({ target: { closest: (selector) => selector === '.search-retry' } });
  await flush();
  assert.match(h.searchResults.innerHTML, /detail\.html\?id=ppo/);
  assert.equal(calls, 2);
});
