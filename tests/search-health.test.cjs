const assert = require('node:assert/strict');
const { test } = require('node:test');
const { searchHarness, indexData, flush, response } = require('./helpers/search-harness.cjs');

test('result cards show the health bar of the matching link-graph node', async () => {
  const h = searchHarness(async () => response(indexData), async () => response({ nodes: [
    { id: 'wiki/ppo.md', health_score: 2, health_gaps: 4 },
  ] }));
  h.searchInput.value = 'PPO'; h.triggerSearch();
  await flush();
  assert.match(h.searchResults.innerHTML, /<span class="node-health" data-score="2"><\/span>/);
});

test('result cards without a graph health score render no bar', async () => {
  const h = searchHarness(async () => response(indexData));
  h.searchInput.value = 'SLAM'; h.triggerSearch();
  await flush();
  assert.match(h.searchResults.innerHTML, /detail\.html\?id=slam/);
  assert.doesNotMatch(h.searchResults.innerHTML, /node-health/);
});
