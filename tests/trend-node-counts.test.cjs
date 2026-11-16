const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

const html = readFileSync(resolve(__dirname, '../docs/star-history.html'), 'utf8');
const loaders = html.slice(html.indexOf('      function getJSON('), html.indexOf('      var touch ='));
const plain = (value) => JSON.parse(JSON.stringify(value));
const modes = ['community', 'type', 'opensource', 'health'];
const viewIds = ['communities', 'type-nodes', 'opensource-nodes', 'health-nodes'];
const graph = {
  communities: [
    { id: 'vla', label: '视觉动作（VLA） 社区', size: 3 },
    { id: 'control', label: '控制（Control） 社区', size: 2 },
    { id: 'empty', label: '空社区', size: 0 },
  ],
  nodes: [
    { community: 'vla', community_secondary: 'control', type: 'entity', has_repo: true, health_score: 3 },
    { community: 'vla', type: 'entity', has_repo: true, health_score: 2 },
    { community: 'vla', type: 'concept', has_repo: false, health_score: 1 },
    { community: 'control', type: 'concept', has_repo: true, health_score: 3 },
    { community: 'control', type: 'entity' },
  ],
};

function setup(data = graph) {
  const requests = [];
  let theme = 'dark';
  const context = vm.createContext({
    document: { documentElement: { getAttribute: () => theme } },
    fetch: async (url) => {
      requests.push(url);
      const result = {
        'exports/star-history.json': { counts: [] },
        'exports/wiki-activity.json': { days: [] },
        'exports/link-graph.json': data,
      };
      assert.ok(Object.hasOwn(result, url), url);
      return { ok: true, json: async () => result[url] };
    },
  });
  context.window = context;
  for (const file of ['graph-tooltip.js', 'wiki-type-labels.js']) {
    vm.runInContext(readFileSync(resolve(__dirname, '../docs', file), 'utf8'), context);
  }
  vm.runInContext(loaders, context);
  return { context, requests, theme: (value) => { theme = value; } };
}

const totals = (result) => result.cats.map((_, i) =>
  result.series.reduce((sum, series) => sum + series.values[i], 0));

test('all 16 row/stack combinations preserve each row total, order and community links', async () => {
  const { context, requests } = setup();
  const expected = [[3, 2], [3, 2], [3, 2], [2, 1, 1, 1]];
  for (const [i, rowMode] of modes.entries()) {
    const view = context.VIEWS.find((view) => view.id === viewIds[i]);
    assert.equal(view.rowMode, rowMode);
    let previous;
    for (const stack of modes) {
      context.nodeStackMode = stack;
      const result = plain(await view.load());
      assert.deepEqual(totals(result), expected[i], `${rowMode}/${stack}`);
      assert.equal(totals(result).reduce((a, b) => a + b, 0), graph.nodes.length);
      if (previous) assert.deepEqual(result.cats, previous.cats);
      previous = result;
      if (rowMode === 'community') {
        assert.deepEqual(result.cats, ['视觉动作（VLA）', '控制（Control）']);
        assert.deepEqual(result.links, ['hubs.html?community=vla', 'hubs.html?community=control']);
      }
      assert.equal(result.showLegend, rowMode !== stack);
    }
  }
  assert.equal(requests.filter((url) => url === 'exports/link-graph.json').length, 1);
  const health = plain(context.buildNodeCounts(graph, 'health', 'community', 'dark'));
  assert.match(health.total, /平均 1\.80 分/);
  assert.match(plain(context.buildNodeCounts(graph, 'opensource', 'type')).total, /开源 3 \/ 5 节点（60\.0%）/);
});

test('cross classifications use primary communities, repo flags and bounded health scores', () => {
  const { context } = setup();
  const counts = (row, stack) => plain(context.buildNodeCounts(graph, row, stack, 'dark'));
  assert.deepEqual(counts('community', 'type').series.map((s) => s.values), [[2, 1], [1, 1]]);
  assert.deepEqual(counts('community', 'opensource').series.map((s) => s.values), [[2, 1], [1, 1]]);
  assert.deepEqual(counts('type', 'health').series.map((s) => s.values), [[1, 1], [1, 0], [0, 1], [1, 0]]);
  assert.deepEqual(counts('opensource', 'health').series.map((s) => s.values), [[2, 0], [1, 0], [0, 1], [0, 1]]);
  assert.deepEqual(counts('health', 'opensource').series.map((s) => s.values), [[2, 1, 0, 0], [0, 0, 1, 1]]);
});

test('missing metadata and empty graphs never drop nodes or produce invalid totals', () => {
  const { context } = setup();
  const data = { communities: graph.communities, nodes: [
    ...graph.nodes, {}, { community: 'unknown', type: 'custom-type', health_score: 9 },
    { community: 'control', health_score: -1 },
  ] };
  for (const row of modes) {
    for (const stack of modes) {
      const result = plain(context.buildNodeCounts(data, row, stack, 'dark'));
      assert.equal(totals(result).reduce((a, b) => a + b, 0), data.nodes.length, `${row}/${stack}`);
      const empty = plain(context.buildNodeCounts({ nodes: [], communities: [] }, row, stack));
      assert.equal(totals(empty).reduce((a, b) => a + b, 0), 0);
      assert.doesNotMatch(empty.total, /NaN|Infinity/);
    }
  }
  const community = plain(context.buildNodeCounts(data, 'community', 'type'));
  assert.equal(community.cats.at(-1), '未分类');
  assert.equal(community.links.at(-1), null);
});

test('same-dimension bars keep per-row colors and health colors update with the theme', async () => {
  const { context, requests, theme } = setup();
  const view = context.VIEWS.find((view) => view.id === 'health-nodes');
  context.nodeStackMode = 'health';
  assert.deepEqual(plain(await view.load()).series[0].color, ['#4ade80', '#facc15', '#fb923c', '#f87171']);
  theme('light');
  assert.deepEqual(plain(await view.load()).series[0].color, ['#16a34a', '#ca8a04', '#ea580c', '#ef4444']);
  context.nodeStackMode = 'community';
  assert.deepEqual(plain(await view.load()).series.map((s) => s.name), ['视觉动作', '控制']);
  assert.equal(requests.filter((url) => url === 'exports/link-graph.json').length, 1);
});
