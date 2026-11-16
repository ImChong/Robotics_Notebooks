const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

const html = readFileSync(resolve(__dirname, '../docs/star-history.html'), 'utf8');
const depthFilters = readFileSync(resolve(__dirname, '../docs/depth-filters.js'), 'utf8');
const tooltip = readFileSync(resolve(__dirname, '../docs/graph-tooltip.js'), 'utf8');
const typeLabels = readFileSync(resolve(__dirname, '../docs/wiki-type-labels.js'), 'utf8');
const loaders = html.slice(html.indexOf('      function getJSON('), html.indexOf('      var touch ='));

const graph = {
  nodes: [
    { id: 'wiki/entities/a.md', detail_id: 'a', community: 'control', type: 'entity' },
    { id: 'wiki/concepts/b.md', detail_id: 'b', community: 'vla', type: 'concept' },
  ],
  communities: [
    { id: 'control', label: '控制（Control）社区', size: 2 },
    { id: 'vla', label: '视觉动作（VLA）社区', size: 1 },
  ],
};

async function loadRoutes(routeHealth, mode = 'community') {
  const requests = [];
  const context = vm.createContext({
    document: { documentElement: { getAttribute: () => 'dark' } },
    fetch: async (url) => {
      requests.push(url);
      const data = {
        'exports/star-history.json': { counts: [] },
        'exports/wiki-activity.json': { days: [] },
        'exports/graph-stats.json': { route_health: routeHealth },
        'exports/link-graph.json': graph,
      };
      assert.ok(Object.hasOwn(data, url), url);
      return { ok: true, json: async () => data[url] };
    },
  });
  context.window = context;
  vm.runInContext(tooltip, context);
  vm.runInContext(typeLabels, context);
  vm.runInContext(depthFilters, context);
  vm.runInContext(loaders, context);
  context.nodeStackMode = mode;
  const view = context.VIEWS.find((item) => item.id === 'depth-nodes');
  return {
    result: JSON.parse(JSON.stringify(await view.load())),
    views: context.VIEWS, depth: context.RNDepthFilters, requests,
  };
}

function row(detail_id, label, counts, communities, types, opensource) {
  return { detail_id, label, counts, communities, types, opensource };
}

const rows = [
  row('roadmap-depth-wam', 'WAM', [1, 0, 0, 0], { control: 1 }, { entity: 1 }, { open: 0, closed: 1 }),
  row('roadmap-depth-vla', 'VLA', [0, 2, 0, 0], { vla: 2 }, { concept: 2 }, { open: 1, closed: 1 }),
  row('roadmap-depth-pcb-design', 'PCB', [0, 0, 0, 3], { control: 3 }, { entity: 3 }, { open: 3, closed: 0 }),
  row('roadmap-unknown-route', '未知路线', [3, 0, 0, 0], { control: 3 }, { entity: 3 }, { open: 0, closed: 3 }),
  row('roadmap-motion-control', '主路线', [0, 1, 1, 0], { control: 1, vla: 1 },
    { entity: 1, concept: 1 }, { open: 1, closed: 1 }),
  row('roadmap-depth-rsi', 'RSI', [0, 0, 3, 1], { control: 2, vla: 2 },
    { entity: 3, concept: 1 }, { open: 2, closed: 2 }),
];

test('merged route chart uses historical roadmap order and keeps old health breakdown', async () => {
  const { result, depth, views } = await loadRoutes(rows, 'health');
  const keys = ['motion-control', 'pcb-design', 'rsi', 'vla', 'wam'];
  const expectedLinks = keys.map((key) =>
    'roadmap.html?id=roadmap-' + (key === 'motion-control' ? key : 'depth-' + key)
  );
  assert.deepEqual(result.links, [...expectedLinks, 'roadmap.html?id=roadmap-unknown-route']);
  assert.deepEqual(
    result.cats.map((name) => name.split(' · 均 ')[0]),
    [...keys.map((key) => depth.DEPTH_META[key].label), '未知路线']
  );
  assert.deepEqual(result.series.map((item) => item.values), [
    [0, 3, 1, 0, 0, 0], // 3 分
    [1, 0, 3, 0, 0, 0], // 2 分
    [1, 0, 0, 2, 0, 0], // 1 分
    [0, 0, 0, 0, 1, 3], // 0 分
  ]);
  assert.equal(views.filter((item) => item.id === 'route-health').length, 0);
  assert.match(result.total, /路线内去重/);
});

test('four stack modes each sum to the same collected route node count', async () => {
  const expectedTotals = [2, 3, 4, 2, 1, 3];
  for (const mode of ['community', 'type', 'opensource', 'health']) {
    const { result, requests } = await loadRoutes(rows, mode);
    const totals = result.cats.map((_, i) =>
      result.series.reduce((sum, series) => sum + series.values[i], 0)
    );
    assert.deepEqual(totals, expectedTotals, mode);
    assert.match(result.total, /合计 15 节点次/);
    assert.equal(requests.filter((url) => url === 'exports/graph-stats.json').length, 1);
    assert.equal(requests.filter((url) => url === 'exports/link-graph.json').length, 1);
    if (mode === 'health') assert.match(result.cats[0], /均 1\.50/);
  }
});

test('stale graph-stats group data cannot silently show incorrect route counts', async () => {
  const old = rows.map(({ detail_id, label, counts }) => ({ detail_id, label, counts }));
  await assert.rejects(loadRoutes(old, 'community'), /路线分组数据未生成/);
});
