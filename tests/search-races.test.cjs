const assert = require('node:assert/strict');
const { test } = require('node:test');
const { searchHarness, indexData, deferred, flush, response } = require('./helpers/search-harness.cjs');

for (const outcome of ['resolve', 'reject']) {
  test(`clearing input prevents a pending ${outcome} from repopulating results`, async () => {
    const pending = deferred();
    const h = searchHarness(() => pending.promise);
    h.searchInput.value = 'PPO'; h.triggerSearch();
    h.searchInput.value = ''; h.searchInput.handlers.input();
    pending[outcome](outcome === 'resolve' ? response(indexData) : new Error('offline'));
    await flush();
    assert.equal(h.searchResults.innerHTML, '');
    assert.equal(h.window.__miniGraphPendingQuery, null);
  });
}

test('a new keystroke invalidates old results before the debounce expires', async () => {
  const pending = deferred();
  const h = searchHarness(() => pending.promise);
  h.searchInput.value = 'PPO'; h.triggerSearch();
  h.searchInput.value = 'SLAM'; h.searchInput.handlers.input();
  pending.resolve(response(indexData));
  await flush();
  assert.doesNotMatch(h.searchResults.innerHTML, /detail\.html\?id=ppo/);
  h.expire(120); await flush();
  assert.match(h.searchResults.innerHTML, /detail\.html\?id=slam/);
  assert.doesNotMatch(h.searchResults.innerHTML, /detail\.html\?id=ppo/);
});

test('Escape clears pending searches and a delayed debounce', async () => {
  const pending = deferred();
  const h = searchHarness(() => pending.promise);
  h.searchInput.value = 'PPO'; h.triggerSearch();
  h.searchInput.handlers.input();
  h.searchInput.handlers.keydown({ key: 'Escape' });
  pending.resolve(response(indexData));
  h.expire(120); await flush();
  assert.equal(h.searchResults.innerHTML, '');
  assert.equal(h.searchInput.value, '');
});

test('changing community while loading renders only the latest filter', async () => {
  const pending = deferred();
  const h = searchHarness(() => pending.promise, async () => response({ nodes: [
    { id: 'wiki/ppo.md', community: 'control' }, { id: 'wiki/slam.md', community: 'perception' },
  ] }));
  h.communityFilter.value = 'control'; h.triggerSearch();
  h.communityFilter.value = 'perception'; h.communityFilter.handlers.change();
  pending.resolve(response(indexData));
  await flush();
  assert.match(h.searchResults.innerHTML, /detail\.html\?id=slam/);
  assert.doesNotMatch(h.searchResults.innerHTML, /detail\.html\?id=ppo/);
});
