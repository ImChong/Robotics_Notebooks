const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

const source = readFileSync(resolve(__dirname, '../docs/star-history.html'), 'utf8');
const tooltip = readFileSync(resolve(__dirname, '../docs/graph-tooltip.js'), 'utf8');
const loaders = source.slice(source.indexOf('      function getJSON('), source.indexOf('      var touch ='));

function loadViews(graph, companies) {
  const requests = [];
  const context = vm.createContext({
    fetch: async (url) => {
      requests.push(url);
      const data = {
        'exports/star-history.json': { counts: [] },
        'exports/wiki-activity.json': { days: [] },
        'exports/link-graph.json': graph,
        'company-roadmaps.json': { companies },
      };
      assert.ok(Object.hasOwn(data, url), url);
      return { ok: true, json: async () => data[url] };
    },
  });
  context.window = context;
  vm.runInContext(tooltip, context);
  vm.runInContext(loaders, context);
  return { requests, views: context.VIEWS };
}

const graph = {
  nodes: [
    { id: 'wiki/entities/a.md', detail_id: 'entity-a', community: 'vla' },
    { id: 'wiki/entities/b.md', detail_id: 'entity-b', community: 'control' },
  ],
  communities: [
    { id: 'vla', label: '视觉动作（VLA）社区', size: 10 },
    { id: 'control', label: '控制（Control）社区', size: 5 },
  ],
};

test('company counts join detail IDs, deduplicate within a company, and retain shared nodes', async () => {
  const { views } = loadViews(graph, [
    { name: '公司乙', nodes: [{ id: 'entity-a' }, { id: 'entity-a' }, { id: 'entity-b' }, { id: 'missing' }] },
    { name: '公司甲', nodes: [{ id: 'entity-a' }] },
    { name: '空公司', nodes: [] },
  ]);
  const result = JSON.parse(JSON.stringify(await views.find((v) => v.id === 'company-nodes').load()));
  assert.deepEqual(result.cats, ['公司乙', '公司甲', '空公司']);
  assert.deepEqual(result.series.map((s) => s.values), [[1, 1, 0], [1, 0, 0]]);
  assert.ok(result.series.every((s) => s.name && /^#[0-9a-f]{6}$/i.test(s.color)));
  assert.match(result.total, /3 家公司，合计 3 节点次/);
});

test('company data loads on demand and repeated views share the graph and company requests', async () => {
  const { requests, views } = loadViews(graph, []);
  assert.ok(!requests.includes('company-roadmaps.json'));
  assert.ok(!requests.includes('exports/link-graph.json'));
  await views.find((v) => v.id === 'communities').load();
  const view = views.find((v) => v.id === 'company-nodes');
  const [result] = await Promise.all([view.load(), view.load()]);
  assert.equal(requests.filter((url) => url === 'exports/link-graph.json').length, 1);
  assert.equal(requests.filter((url) => url === 'company-roadmaps.json').length, 1);
  assert.equal(result.cats.length, 0);
  assert.equal(result.series.length, 0);
  assert.match(result.total, /0 家公司，合计 0 节点次/);
});

test('company timeline groups dated nodes by month and skips undated nodes and empty companies', async () => {
  const { requests, views } = loadViews(graph, [
    { name: '公司乙', nodes: [
      { date: '2026-03', title: 'B2', track: 'VLA' },
      { date: '', title: '无日期', track: '仿真' },
      { date: '2025-01', title: 'B1', track: '世界模型' },
      { date: '2026-03', title: 'B3', track: '全身控制' },
    ] },
    { name: '公司甲', nodes: [{ date: '2024-10', title: 'A1', track: 'VLA' }] },
    { name: '空公司', nodes: [{ date: '', title: '无日期', track: '数据' }] },
  ]);
  const result = JSON.parse(JSON.stringify(await views.find((v) => v.id === 'company-timeline').load()));
  assert.ok(!requests.includes('exports/link-graph.json'));
  assert.deepEqual(result.rows, [
    { name: '公司乙', points: [
      { date: '2025-01', items: ['B1（世界模型）'] },
      { date: '2026-03', items: ['B2（VLA）', 'B3（全身控制）'] },
    ] },
    { name: '公司甲', points: [{ date: '2024-10', items: ['A1（VLA）'] }] },
  ]);
  assert.equal(result.first, '2024-10');
  assert.equal(result.last, '2026-03');
  assert.match(result.total, /2 家公司，4 个已注明日期的节点（2024-10 → 2026-03）；另有 2 个未注明日期的节点未绘制/);
});
