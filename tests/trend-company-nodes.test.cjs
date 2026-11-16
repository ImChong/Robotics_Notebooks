const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

const source = readFileSync(resolve(__dirname, '../docs/star-history.html'), 'utf8');
const tooltip = readFileSync(resolve(__dirname, '../docs/graph-tooltip.js'), 'utf8');
const depthFilters = readFileSync(resolve(__dirname, '../docs/depth-filters.js'), 'utf8');
const typeLabels = readFileSync(resolve(__dirname, '../docs/wiki-type-labels.js'), 'utf8');
const loaders = source.slice(source.indexOf('      function getJSON('), source.indexOf('      var touch ='));

function loadViews(graph, companies, labs = [], routeHealth = []) {
  const requests = [];
  let theme = 'dark';
  const context = vm.createContext({
    document: { documentElement: { getAttribute: () => theme } },
    fetch: async (url) => {
      requests.push(url);
      const data = {
        'exports/star-history.json': { counts: [] },
        'exports/wiki-activity.json': { days: [] },
        'exports/link-graph.json': graph,
        'exports/graph-stats.json': { route_health: routeHealth },
        'company-roadmaps.json': { companies },
        'lab-roadmaps.json': { labs },
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
  return { requests, views: context.VIEWS, context, setTheme: (value) => { theme = value; } };
}

const graph = {
  nodes: [
    { id: 'wiki/entities/a.md', detail_id: 'entity-a', community: 'vla', type: 'entity', has_repo: true, health_score: 3 },
    { id: 'wiki/entities/b.md', detail_id: 'entity-b', community: 'control', type: 'concept', health_score: 1 },
  ],
  communities: [
    { id: 'vla', label: '视觉动作（VLA）社区', size: 10 },
    { id: 'control', label: '控制（Control）社区', size: 5 },
  ],
};

test('company counts join detail IDs, deduplicate within a company, and retain shared nodes', async () => {
  const { views } = loadViews(graph, [
    { key: 'yi', name: '公司乙', nodes: [{ id: 'entity-a' }, { id: 'entity-a' }, { id: 'entity-b' }, { id: 'missing' }] },
    { key: 'jia', name: '公司甲', nodes: [{ id: 'entity-a' }] },
    { key: 'kong', name: '空公司', nodes: [] },
  ]);
  const result = JSON.parse(JSON.stringify(await views.find((v) => v.id === 'company-nodes').load()));
  assert.deepEqual(result.cats, ['公司乙', '公司甲', '空公司']);
  assert.deepEqual(result.links, ['company.html?id=yi', 'company.html?id=jia', 'company.html?id=kong']);
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

test('lab node counts link each bar back to its lab page', async () => {
  const { requests, views } = loadViews(graph, [], [
    { key: 'eth-rsl', name: '实验室甲', founded_year: 2015, nodes: [{ id: 'entity-a' }, { id: 'entity-b' }] },
  ]);
  const result = JSON.parse(JSON.stringify(await views.find((v) => v.id === 'lab-nodes').load()));
  assert.ok(!requests.includes('company-roadmaps.json'));
  assert.deepEqual(result.cats, ['实验室甲']);
  assert.deepEqual(result.links, ['lab.html?id=eth-rsl']);
  assert.match(result.total, /^1 个实验室，合计 2 节点次/);
});

test('combined node counts interleave companies and labs by founding year and mark each kind', async () => {
  const { views } = loadViews(graph, [
    { key: 'old', name: '老公司', founded_year: 1993, nodes: [{ id: 'entity-a' }] },
    { key: 'new', name: '新公司', founded_year: 2022, nodes: [{ id: 'entity-b' }] },
  ], [
    { key: 'lab-a', name: '实验室甲', founded_year: 2015, nodes: [{ id: 'entity-a' }] },
    { key: 'lab-b', name: '实验室乙', founded_year: 2022, nodes: [] },
  ]);
  const result = JSON.parse(JSON.stringify(await views.find((v) => v.id === 'org-nodes').load()));
  assert.deepEqual(result.cats, ['老公司', '实验室甲', '新公司', '实验室乙']);
  assert.deepEqual(result.kinds, ['company', 'lab', 'company', 'lab']);
  assert.deepEqual(result.links, ['company.html?id=old', 'lab.html?id=lab-a', 'company.html?id=new', 'lab.html?id=lab-b']);
  assert.equal(result.marked, true);
  assert.match(result.total, /^2 家公司、2 个实验室，合计 3 节点次（.*名称前 ● 为公司、◆ 为实验室）$/);
});

test('all 12 institution/stack combinations preserve deduplication, row order, links and totals', async () => {
  const { context, views, requests } = loadViews(graph, [
    { key: 'old', name: '老公司', founded_year: 1993, nodes: [
      { id: 'entity-a' }, { id: 'entity-a' }, { id: 'entity-b' }, { id: 'missing' },
    ] },
    { key: 'new', name: '新公司', founded_year: 2022, nodes: [{ id: 'entity-b' }] },
    { key: 'empty', name: '空公司', founded_year: 2023, nodes: [] },
  ], [
    { key: 'lab-a', name: '实验室甲', founded_year: 2015, nodes: [
      { id: 'entity-a' }, { id: 'entity-b' }, { id: 'entity-a' },
    ] },
    { key: 'lab-b', name: '实验室乙', founded_year: 2022, nodes: [{ id: 'entity-a' }] },
    { key: 'lab-empty', name: '空实验室', founded_year: 2024, nodes: [] },
  ]);
  const modes = ['community', 'type', 'opensource', 'health'];
  for (const [id, expected] of [
    ['company-nodes', [2, 1, 0]], ['lab-nodes', [2, 1, 0]], ['org-nodes', [2, 2, 1, 1, 0, 0]],
  ]) {
    const view = views.find((v) => v.id === id);
    assert.equal(view.orgNodes, true);
    let previous;
    for (const mode of modes) {
      context.nodeStackMode = mode;
      const result = JSON.parse(JSON.stringify(await view.load()));
      assert.deepEqual(result.cats.map((_, i) => result.series.reduce((sum, s) => sum + s.values[i], 0)), expected, id + '/' + mode);
      assert.equal(result.sum, expected.reduce((a, b) => a + b, 0));
      if (previous) {
        for (const key of ['cats', 'links', 'kinds', 'marked', 'total']) assert.deepEqual(result[key], previous[key], key);
      }
      previous = result;
      if (id === 'company-nodes') {
        assert.deepEqual(result.series.map((s) => s.values), mode === 'health'
          ? [[1, 0, 0], [0, 0, 0], [1, 1, 0], [0, 0, 0]] : [[1, 0, 0], [1, 1, 0]]);
      }
    }
  }
  for (const url of ['exports/link-graph.json', 'company-roadmaps.json', 'lab-roadmaps.json']) {
    assert.equal(requests.filter((request) => request === url).length, 1, url);
  }
});

test('institution stacks retain unclassified nodes, zero buckets and current theme colors', async () => {
  const data = { communities: graph.communities, nodes: [
    ...graph.nodes,
    { id: 'wiki/c.md', detail_id: 'c', community: 'unknown', health_score: -1 },
    { id: 'wiki/d.md', detail_id: 'd', health_score: 9 },
  ] };
  const { context, views, setTheme } = loadViews(data, [{ key: 'test', name: '公司', nodes: [
    { id: 'entity-a' }, { id: 'c' }, { id: 'c' }, { id: 'd' }, { id: 'missing' },
  ] }]);
  const view = views.find((v) => v.id === 'company-nodes');
  const load = async () => JSON.parse(JSON.stringify(await view.load()));
  let result = await load();
  assert.deepEqual(result.series.map((s) => [s.name, s.values]), [['视觉动作', [1]], ['未分类', [2]]]);
  context.nodeStackMode = 'type';
  result = await load();
  assert.deepEqual(result.series.map((s) => s.values).sort(), [[1], [2]]);
  context.nodeStackMode = 'opensource';
  assert.deepEqual((await load()).series.map((s) => [s.name, s.values]), [['开源', [1]], ['未开源', [2]]]);
  context.nodeStackMode = 'health';
  const dark = await load();
  assert.deepEqual(dark.series.map((s) => s.values), [[2], [0], [0], [1]]);
  assert.deepEqual(dark.series.map((s) => s.color), ['#4ade80', '#facc15', '#fb923c', '#f87171']);
  setTheme('light');
  const light = await load();
  assert.deepEqual(light.series.map((s) => s.values), dark.series.map((s) => s.values));
  assert.deepEqual(light.series.map((s) => s.color), ['#16a34a', '#ca8a04', '#ea580c', '#ef4444']);
});

test('institution loaders capture the requested stack before their shared requests resolve', async () => {
  const { context, views } = loadViews(graph, [{ key: 'test', name: '公司', nodes: [{ id: 'entity-a' }] }]);
  const view = views.find((v) => v.id === 'company-nodes');
  context.nodeStackMode = 'opensource';
  const openRequest = view.load();
  context.nodeStackMode = 'health';
  const healthRequest = view.load();
  assert.deepEqual(JSON.parse(JSON.stringify((await openRequest).series.map((s) => s.name))), ['开源', '未开源']);
  assert.deepEqual(JSON.parse(JSON.stringify((await healthRequest).series.map((s) => s.name))), ['3 分', '2 分', '1 分', '0 分']);
});

test('route collected node counts link each route to its roadmap page', async () => {
  const routeHealth = [
    { detail_id: 'roadmap-depth-vla', label: 'VLA', counts: [0, 0, 1, 0],
      communities: { vla: 1 }, types: { entity: 1 }, opensource: { open: 0, closed: 1 } },
    { detail_id: 'roadmap-motion-control', label: '主路线', counts: [0, 1, 0, 1],
      communities: { vla: 1, control: 1 }, types: { entity: 2 }, opensource: { open: 1, closed: 1 } },
  ];
  const { views } = loadViews(graph, [], [], routeHealth);
  const result = JSON.parse(JSON.stringify(await views.find((v) => v.id === 'depth-nodes').load()));
  assert.deepEqual(result.links, [
    'roadmap.html?id=roadmap-motion-control',
    'roadmap.html?id=roadmap-depth-vla',
  ]);
  assert.equal(result.links.length, result.cats.length);
  assert.deepEqual(
    result.cats.map((_, i) => result.series.reduce((sum, series) => sum + series.values[i], 0)),
    [2, 1]
  );
});

test('community node counts link each community to its hubs ranking', async () => {
  const { views } = loadViews(graph, []);
  const result = JSON.parse(JSON.stringify(await views.find((v) => v.id === 'communities').load()));
  assert.deepEqual(result.cats, ['视觉动作（VLA）', '控制（Control）']);
  assert.deepEqual(result.links, ['hubs.html?community=vla', 'hubs.html?community=control']);
});
