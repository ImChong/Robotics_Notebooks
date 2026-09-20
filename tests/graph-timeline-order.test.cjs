// 时序动画必须按 git 首次加入日从最老放到最新，不能用会塌成同一天的 recency。
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

const source = readFileSync(resolve(__dirname, '../docs/graph.html'), 'utf8');
const start = source.indexOf('    // 时序时钟：默认 git 首次加入日（added）');
const end = source.indexOf('    // 同步进度文本 + 进度条 + 顶部计数', start);
assert.ok(start >= 0 && end > start, 'timeline sort helpers are located');

function loadSort(recencyIncludeMaintained) {
  const context = vm.createContext({ recencyIncludeMaintained: !!recencyIncludeMaintained });
  vm.runInContext(source.slice(start, end), context);
  return context;
}

function node(id, addedTs, activityTs) {
  return {
    id,
    _addedTs: addedTs,
    _activityTs: activityTs,
    _addedDate: addedTs == null ? null : 'd',
    _activityDate: activityTs == null ? null : 'd',
  };
}

test('timeline order is oldest added date first, newest last', () => {
  const ctx = loadSort(false);
  const ordered = ctx.sortNodesByTimeline([
    node('wiki/new.md', Date.parse('2026-09-20'), Date.parse('2026-09-20')),
    node('wiki/old.md', Date.parse('2026-04-17'), Date.parse('2026-08-01')),
    node('wiki/mid.md', Date.parse('2026-06-11'), Date.parse('2026-09-01')),
  ]);
  assert.deepEqual(ordered.map((n) => n.id), [
    'wiki/old.md',
    'wiki/mid.md',
    'wiki/new.md',
  ]);
});

test('same added day keeps a stable id tie-break and does not reverse to newest-first', () => {
  const ctx = loadSort(false);
  const day = Date.parse('2026-08-10');
  const ordered = ctx.sortNodesByTimeline([
    node('wiki/tasks/z.md', day, day),
    node('wiki/concepts/a.md', day, day),
    node('wiki/entities/m.md', day, day),
  ]);
  assert.deepEqual(ordered.map((n) => n.id), [
    'wiki/concepts/a.md',
    'wiki/entities/m.md',
    'wiki/tasks/z.md',
  ]);
});

test('nodes missing an added date sort last so they do not seed the oldest cluster', () => {
  const ctx = loadSort(false);
  const ordered = ctx.sortNodesByTimeline([
    node('wiki/undated.md', null, Date.parse('2026-09-20')),
    node('wiki/old.md', Date.parse('2026-04-17'), Date.parse('2026-04-17')),
    node('wiki/new.md', Date.parse('2026-09-20'), Date.parse('2026-09-20')),
  ]);
  assert.deepEqual(ordered.map((n) => n.id), [
    'wiki/old.md',
    'wiki/new.md',
    'wiki/undated.md',
  ]);
});

test('maintenance toggle uses last-touch dates but still plays oldest to newest', () => {
  const ctx = loadSort(true);
  const ordered = ctx.sortNodesByTimeline([
    node('wiki/recently-touched-old.md', Date.parse('2026-04-17'), Date.parse('2026-09-18')),
    node('wiki/stale.md', Date.parse('2026-05-01'), Date.parse('2026-06-01')),
    node('wiki/brand-new.md', Date.parse('2026-09-20'), Date.parse('2026-09-20')),
  ]);
  assert.deepEqual(ordered.map((n) => n.id), [
    'wiki/stale.md',
    'wiki/recently-touched-old.md',
    'wiki/brand-new.md',
  ]);
});

test('graph.html no longer sorts the timeline by frontmatter/mtime recency', () => {
  assert.ok(!/function getNodeRecency\(d\)/.test(source));
  assert.match(source, /function sortNodesByTimeline\(list\)/);
  assert.match(source, /return ta - tb;/);
  assert.doesNotMatch(source, /getNodeRecency\(a\)\.localeCompare/);
});
