// 首页最近更新只用 home-stats 的新增摘要，凑不满 5 条才回填活动全集（编号 36）
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

const source = readFileSync(resolve(__dirname, '../docs/main.js'), 'utf8');
const helperStart = source.indexOf('  function homeLatestItemsOf(');
const helperEnd = source.indexOf('  function renderUpdatesItemSuffix(');
const blockStart = source.indexOf('  if (homeStatsRoot) {');
const blockEnd = source.indexOf('  // 完整互链榜单页：独立拉取', blockStart);
assert.ok(helperStart >= 0 && helperEnd > helperStart, 'home latest helpers are located');
assert.ok(blockStart >= 0 && blockEnd > blockStart, 'home stats bootstrap is located');

const statsUrl = 'exports/home-stats.json';
const activityUrl = 'exports/wiki-activity.json';
const added = (n) => Array.from({ length: n }, (_, i) => ({ detail_id: `node-${i}`, action: 'added' }));
const activity = { days: [{ date: '2026-09-09', nodes: added(5).map((n) => ({ ...n, detail_id: `fill-${n.detail_id}` })) }] };
const flush = () => new Promise(setImmediate);

function harness(stats, options = {}) {
  const calls = [], renders = [];
  const deferred = [];
  const mount = {
    attrs: options.compact === false ? {} : { 'data-compact': '' },
    hasAttribute(name) { return Object.prototype.hasOwnProperty.call(this.attrs, name); },
    classList: { remove() {} },
  };
  const context = vm.createContext({
    Promise, console: { warn() {} },
    homeStatsRoot: {},
    document: { getElementById: (id) => (id === 'homeLatestWikiModule' ? mount : null) },
    window: { setTimeout: (fn) => { deferred.push(fn); return deferred.length; } },
    fetch: async (url) => {
      calls.push(url);
      return { ok: true, status: 200, json: async () => (url === statsUrl ? stats : activity) };
    },
    initHeroStatCountUp() {}, renderHomeStats() {}, renderHotTopics() {}, renderHomeHubs() {},
    renderLatestWikiNode: (_stats, wikiActivity) => renders.push(wikiActivity),
  });
  vm.runInContext(source.slice(helperStart, helperEnd), context);
  vm.runInContext(source.slice(blockStart, blockEnd), context);
  return { calls, renders, runDeferred: () => { for (const fn of deferred.splice(0)) fn(); } };
}

test('the compact homepage list renders without the full activity export', async () => {
  const h = harness({ latest_wiki_nodes: added(5) });
  await flush();
  h.runDeferred();
  await flush();
  assert.deepEqual(h.calls, [statsUrl]);
  assert.deepEqual(h.renders, [null]);
});

test('a short compact list backfills from the activity export and re-renders', async () => {
  const h = harness({ latest_wiki_nodes: added(2) });
  await flush();
  h.runDeferred();
  await flush();
  assert.deepEqual(h.calls, [statsUrl, activityUrl]);
  assert.equal(h.renders.length, 2);
  assert.equal(h.renders[0], null);
  assert.equal(h.renders[1], activity);
});

test('the full change-log timeline still loads the activity export up front', async () => {
  const h = harness({ latest_wiki_nodes: added(5) }, { compact: false });
  await flush();
  h.runDeferred();
  await flush();
  assert.deepEqual(h.calls, [statsUrl, activityUrl]);
  assert.deepEqual(h.renders, [activity]);
});
