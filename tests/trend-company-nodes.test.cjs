const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

const source = readFileSync(resolve(__dirname, '../docs/star-history.html'), 'utf8');
const tooltip = readFileSync(resolve(__dirname, '../docs/graph-tooltip.js'), 'utf8');
const loaders = source.slice(source.indexOf('      function getJSON('), source.indexOf('      var touch ='));

function loadViews(graph, companies, labs = []) {
  const requests = [];
  const context = vm.createContext({
    fetch: async (url) => {
      requests.push(url);
      const data = {
        'exports/star-history.json': { counts: [] },
        'exports/wiki-activity.json': { days: [] },
        'exports/link-graph.json': graph,
        'company-roadmaps.json': { companies },
        'lab-roadmaps.json': { labs },
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
    { key: 'yi', name: '公司乙', nodes: [
      { date: '2026-03', title: 'B2', track: 'VLA' },
      { date: '', title: '无日期', track: '仿真' },
      { date: '2025-01', title: 'B1', track: '世界模型' },
      { date: '2026-03', title: 'B3', track: '全身控制' },
    ] },
    { key: 'jia', name: '公司甲', nodes: [{ date: '2024-10', title: 'A1', track: 'VLA' }] },
    { key: 'kong', name: '空公司', nodes: [{ date: '', title: '无日期', track: '数据' }] },
  ]);
  const result = JSON.parse(JSON.stringify(await views.find((v) => v.id === 'company-timeline').load()));
  assert.ok(!requests.includes('exports/link-graph.json'));
  assert.deepEqual(result.rows.map(({ key, name, href, points }) => ({ key, name, href, points })), [
    { key: 'yi', name: '公司乙', href: 'company.html?id=yi', points: [
      { date: '2025-01', items: ['B1（世界模型）'] },
      { date: '2026-03', items: ['B2（VLA）', 'B3（全身控制）'] },
    ] },
    { key: 'jia', name: '公司甲', href: 'company.html?id=jia', points: [{ date: '2024-10', items: ['A1（VLA）'] }] },
  ]);
  assert.equal(result.first, '2024-10');
  assert.equal(result.last, '2026-03');
  assert.match(result.total, /2 家公司，4 个已注明日期的节点（2024-10 → 2026-03）；另有 2 个未注明日期的节点未绘制/);
});

test('lab timeline links rows back to lab pages and loads only lab data', async () => {
  const { requests, views } = loadViews(graph, [], [
    { key: 'eth-rsl', name: '实验室甲', founded_year: 2015, nodes: [
      { date: '2016-10', title: 'L1', track: '硬件' },
      { date: '', title: '无日期', track: '工具' },
    ] },
  ]);
  const result = JSON.parse(JSON.stringify(await views.find((v) => v.id === 'lab-timeline').load()));
  assert.ok(!requests.includes('company-roadmaps.json'));
  assert.deepEqual(result.rows.map((r) => [r.kind, r.href]), [['lab', 'lab.html?id=eth-rsl']]);
  assert.match(result.total, /^1 个实验室，1 个已注明日期的节点（2016-10 → 2016-10）；另有 1 个未注明日期的节点未绘制/);
});

test('combined timeline interleaves companies and labs by founding year, companies first within a year', async () => {
  const { views } = loadViews(graph, [
    { key: 'old', name: '老公司', founded_year: 1993, nodes: [{ date: '2020-10', title: 'C1', track: '仿真' }] },
    { key: 'new', name: '新公司', founded_year: 2022, nodes: [{ date: '2025-01', title: 'C2', track: 'VLA' }] },
  ], [
    { key: 'lab-a', name: '实验室甲', founded_year: 2015, nodes: [{ date: '2016-10', title: 'L1', track: '硬件' }] },
    { key: 'lab-b', name: '实验室乙', founded_year: 2022, nodes: [{ date: '2026-01', title: 'L2', track: '全身' }] },
  ]);
  const result = JSON.parse(JSON.stringify(await views.find((v) => v.id === 'org-timeline').load()));
  assert.deepEqual(result.rows.map((r) => [r.name, r.kind]), [
    ['老公司', 'company'], ['实验室甲', 'lab'], ['新公司', 'company'], ['实验室乙', 'lab'],
  ]);
  assert.equal(result.first, '2016-10');
  assert.equal(result.last, '2026-01');
  assert.match(result.total, /^2 家公司、2 个实验室，4 个已注明日期的节点/);
});
