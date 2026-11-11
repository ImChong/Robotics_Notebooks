const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

const page = readFileSync(resolve(__dirname, '../docs/star-history.html'), 'utf8');
const tooltip = readFileSync(resolve(__dirname, '../docs/graph-tooltip.js'), 'utf8');
const typeLabels = readFileSync(resolve(__dirname, '../docs/wiki-type-labels.js'), 'utf8');
const loaders = page.slice(page.indexOf('      function getJSON('), page.indexOf('      var touch ='));

const graph = {
  nodes: [
    { id: 'wiki/a.md', community: 'vla', community_secondary: 'control', type: 'concept', has_repo: true, health_score: 3 },
    { id: 'wiki/b.md', community: 'control', type: 'method', has_repo: false, health_score: 2 },
    { id: 'wiki/c.md', community: 'vla', type: 'concept', has_repo: true, health_score: 1 },
    { id: 'wiki/d.md', community: 'control', type: 'entity' },
    { id: 'wiki/undated.md', community: 'archive', type: 'query', health_score: 3 },
    { id: 'wiki/stale.md', community: 'removed-community', type: 'custom-type', has_repo: true, health_score: 5 },
    { id: 'wiki/no-community.md', health_score: -1 },
  ],
  communities: [
    { id: 'control', label: '控制（Control） 社区', size: 2 },
    { id: 'community-other', label: '其他社区', size: 0 },
    { id: 'vla', label: '视觉动作（VLA） 社区', size: 3 },
    { id: 'archive', label: '归档社区', size: 1 },
  ],
};

function loadViews(days) {
  const requests = [];
  let theme = 'dark';
  const context = vm.createContext({
    document: { documentElement: { getAttribute: () => theme } },
    fetch: async (url) => {
      requests.push(url);
      const data = {
        'exports/star-history.json': { counts: [] },
        'exports/wiki-activity.json': { days },
        'exports/link-graph.json': graph,
      };
      assert.ok(Object.hasOwn(data, url), url);
      return { ok: true, json: async () => data[url] };
    },
  });
  context.window = context;
  vm.runInContext(tooltip, context);
  vm.runInContext(typeLabels, context);
  vm.runInContext(loaders, context);
  return { context, requests, setTheme: (value) => { theme = value; },
    view: (id) => context.VIEWS.find((view) => view.id === id) };
}

const plain = (data) => JSON.parse(JSON.stringify(data));
const added = (path) => ({ action: 'added', path });

test('both trend views share the current primary-community grouping and reconcile with the previous totals', async () => {
  const days = [
    { date: '2026-02-27', nodes: [added('wiki/a.md'), added('wiki/b.md')] },
    { date: '2026-02-28', nodes: [{ path: 'wiki/b.md', action: 'maintained' }] },
    { date: '2026-03-01', nodes: [added('wiki/c.md')] },
    { date: '2026-03-03', nodes: [added('wiki/d.md')] },
  ];
  const { context, requests, view } = loadViews(days);
  await view('stars').load();
  assert.ok(!requests.includes('exports/link-graph.json'));
  const [cumulative, daily] = await Promise.all([view('nodes').load(), view('nodes-daily').load()]);
  assert.equal(cumulative, daily);
  assert.equal(requests.filter((url) => url === 'exports/link-graph.json').length, 1);
  const result = plain(daily);
  assert.deepEqual(result.x, ['2026-02-27', '2026-02-28', '2026-03-01', '2026-03-02', '2026-03-03']);
  const oldDaily = context.dailyFromDates(days.flatMap(({ date, nodes }) =>
    nodes.filter((node) => node.action === 'added').map(() => date)));
  const previous = new Map(oldDaily.x.map((date, i) => [date, oldDaily.y[i]]));
  const colors = context.RNGraphTooltip.buildCommunityColorMap(graph.communities);
  assert.deepEqual(result.series.map(({ id, name, color }) => [id, name, color]), [
    ['vla', '视觉动作', colors.vla], ['control', '控制', colors.control], ['archive', '归档', colors.archive],
  ]);
  assert.deepEqual(result.series.map((series) => series.daily), [
    [1, 0, 1, 0, 0], [1, 0, 0, 0, 1], [0, 0, 0, 0, 0],
  ]);
  let runningTotal = 0;
  for (const [i, date] of result.x.entries()) {
    const oldTotal = previous.get(date) || 0;
    runningTotal += oldTotal;
    assert.equal(result.series.reduce((sum, series) => sum + series.daily[i], 0), oldTotal, date);
    assert.equal(result.series.reduce((sum, series) => sum + series.cumulative[i], 0), runningTotal, date);
    assert.equal(result.daily[i], oldTotal, date);
    assert.equal(result.cumulative[i], runningTotal, date);
  }
});

test('duplicate paths keep the first date and unmatched or missing communities still count', async () => {
  const missingPath = { action: 'added', detail_id: 'no-path' };
  const { view } = loadViews([
    { date: '2026-03-03', nodes: [added('wiki/a.md'), missingPath, added('wiki/no-community.md')] },
    { date: '2026-03-01', nodes: [added('wiki/a.md'), added('wiki/a.md'), missingPath, added('wiki/missing.md')] },
    { date: '2026-03-02', nodes: [added('wiki/stale.md')] },
  ]);
  const result = plain(await view('nodes-daily').load());
  assert.deepEqual(result.daily, [3, 1, 1]);
  assert.deepEqual(result.cumulative, [3, 4, 5]);
  assert.deepEqual(result.series.find(({ id }) => id === 'vla').daily, [1, 0, 0]);
  const fallback = result.series.at(-1);
  assert.equal(fallback.name, '未分类');
  assert.deepEqual(fallback.daily, [2, 1, 1]);
  assert.deepEqual(fallback.cumulative, [2, 3, 4]);
});

test('an empty activity feed has no dates or community layers', async () => {
  const { view } = loadViews([]);
  assert.deepEqual(plain(await view('nodes').load()), { x: [], series: [], daily: [], cumulative: [] });
});

test('every stacking mode preserves the same deduplicated daily and cumulative totals', () => {
  const { context } = loadViews([]);
  const entries = [
    { date: '2026-03-03', path: 'wiki/a.md' },
    ...['a', 'b', 'c'].map((id) => ({ date: '2026-03-01', path: `wiki/${id}.md` })),
    ...['d', 'stale', 'no-community', 'missing'].map((id) => ({ date: '2026-03-03', path: `wiki/${id}.md` })),
  ];
  for (const mode of Object.keys(context.NODE_STACK_MODES)) {
    const result = plain(context.buildNodeTrends(entries, graph, mode, 'dark'));
    assert.deepEqual(result.daily, [3, 0, 4], mode);
    assert.deepEqual(result.cumulative, [3, 3, 7], mode);
    for (const [i, daily] of result.daily.entries()) {
      assert.equal(result.series.reduce((sum, s) => sum + s.daily[i], 0), daily, mode);
      assert.equal(result.series.reduce((sum, s) => sum + s.cumulative[i], 0), result.cumulative[i], mode);
    }
    assert.ok(result.series.find((s) => s.name === '未分类'), mode);
    assert.deepEqual(plain(context.buildNodeTrends([], graph, mode)), { x: [], series: [], daily: [], cumulative: [] });
  }
});

test('type, repository and health groups use the current graph metadata and classification colors', () => {
  const { context } = loadViews([]);
  const entries = ['a', 'b', 'c', 'd', 'stale', 'no-community'].map((id) => ({ date: '2026-03-01', path: `wiki/${id}.md` }));
  const types = plain(context.buildNodeTrends(entries, graph, 'type')).series;
  assert.equal(types.find((s) => s.id === 'concept').cumulative[0], 2);
  assert.equal(types.find((s) => s.id === '').name, '知识页');
  assert.equal(types.find((s) => s.id === 'custom-type').color, context.RNGraphTooltip.GRAPH_NODE_TYPE_COLOR['']);
  for (const series of types) {
    assert.equal(series.color, context.RNGraphTooltip.GRAPH_NODE_TYPE_COLOR[series.id] || context.RNGraphTooltip.GRAPH_NODE_TYPE_COLOR['']);
  }
  const repos = plain(context.buildNodeTrends(entries, graph, 'opensource')).series;
  assert.deepEqual(repos.map((s) => [s.name, s.color, s.cumulative[0]]), [
    ['开源', '#2dd4bf', 3], ['未开源', '#fb7185', 3],
  ]);
  const health = plain(context.buildNodeTrends(entries, graph, 'health', 'dark')).series;
  assert.deepEqual(health.map((s) => [s.name, s.color, s.cumulative[0]]), [
    ['3 分', '#4ade80', 2], ['2 分', '#facc15', 1], ['1 分', '#fb923c', 1], ['0 分', '#f87171', 2],
  ]);
  const light = plain(context.buildNodeTrends(entries, graph, 'health', 'light')).series;
  assert.deepEqual(light.map((s) => s.color), ['#16a34a', '#ca8a04', '#ea580c', '#ef4444']);
  const single = plain(context.buildNodeTrends(entries.slice(0, 1), graph, 'health')).series;
  assert.deepEqual(single.map((s) => s.daily[0]), [1, 0, 0, 0]);
});

test('switching mode or theme reuses data requests and cached series with matching health colors', async () => {
  const { context, requests, setTheme, view } = loadViews([{ date: '2026-03-01', nodes: [added('wiki/a.md')] }]);
  const results = new Map();
  for (const mode of Object.keys(context.NODE_STACK_MODES)) {
    context.nodeStackMode = mode;
    results.set(mode, await view('nodes').load());
    assert.equal(await view('nodes-daily').load(), results.get(mode));
  }
  setTheme('light');
  const light = await view('nodes').load();
  assert.equal(light.series[0].color, '#16a34a');
  assert.notEqual(light, results.get('health'));
  setTheme('dark');
  assert.equal(await view('nodes').load(), results.get('health'));
  assert.equal(requests.filter((url) => url === 'exports/link-graph.json').length, 1);
  assert.equal(requests.filter((url) => url === 'exports/wiki-activity.json').length, 1);
});
