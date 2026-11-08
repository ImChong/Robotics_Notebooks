const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

const html = readFileSync(resolve(__dirname, '../docs/star-history.html'), 'utf8');
const depthFilters = readFileSync(resolve(__dirname, '../docs/depth-filters.js'), 'utf8');
const loaders = html.slice(html.indexOf('      function getJSON('), html.indexOf('      var touch ='));

async function loadRouteHealth(routeHealth) {
  const context = vm.createContext({
    document: { documentElement: { getAttribute: () => 'dark' } },
    fetch: async (url) => {
      const data = {
        'exports/star-history.json': { counts: [] },
        'exports/wiki-activity.json': { days: [] },
        'exports/graph-stats.json': { route_health: routeHealth },
      };
      assert.ok(Object.hasOwn(data, url), url);
      return { ok: true, json: async () => data[url] };
    },
  });
  context.window = context;
  vm.runInContext(depthFilters, context);
  vm.runInContext(loaders, context);
  const view = context.VIEWS.find((item) => item.id === 'route-health');
  return { result: JSON.parse(JSON.stringify(await view.load())), depth: context.RNDepthFilters };
}

test('route health follows the historical depth order, not the average health score', async () => {
  // Intentionally shuffled, with scores that would produce a different average-score order.
  const { result, depth } = await loadRouteHealth([
    { detail_id: 'roadmap-depth-wam', label: 'WAM', counts: [1, 0, 0, 0] },
    { detail_id: 'roadmap-depth-vla', label: 'VLA', counts: [0, 2, 0, 0] },
    { detail_id: 'roadmap-depth-pcb-design', label: 'PCB', counts: [0, 0, 0, 3] },
    { detail_id: 'roadmap-unknown-route', label: '未知路线', counts: [3, 0, 0, 0] },
    { detail_id: 'roadmap-motion-control', label: '主路线', counts: [0, 1, 1, 0] },
    { detail_id: 'roadmap-depth-rsi', label: 'RSI', counts: [0, 0, 3, 1] },
  ]);

  const keys = ['motion-control', 'pcb-design', 'rsi', 'vla', 'wam'];
  const expectedLinks = keys.map((key) =>
    'roadmap.html?id=roadmap-' + (key === 'motion-control' ? key : 'depth-' + key)
  );
  assert.deepEqual(result.links, [...expectedLinks, 'roadmap.html?id=roadmap-unknown-route']);
  assert.deepEqual(
    result.cats.map((name) => name.split(' · 均 ')[0]),
    [...keys.map((key) => depth.DEPTH_META[key].label), '未知路线']
  );
  // Order must be applied consistently to all four stacked score series.
  assert.deepEqual(result.series.map((item) => item.values), [
    [0, 3, 1, 0, 0, 0], // 3 分
    [1, 0, 3, 0, 0, 0], // 2 分
    [1, 0, 0, 2, 0, 0], // 1 分
    [0, 0, 0, 0, 1, 3], // 0 分
  ]);
  assert.match(result.total, /历史顺序排列（主路线置顶）/);
});
