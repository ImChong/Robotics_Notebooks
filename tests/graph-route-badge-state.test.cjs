// 路线序号徽标状态：总览全同色；步进后当前 / 已过 / 未到三态，顺序来自 route_order 下标。
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

const source = readFileSync(resolve(__dirname, '../docs/graph.html'), 'utf8');
const start = source.indexOf("    // null | 'all' | 'done' | 'current' | 'todo'");
const end = source.indexOf('    function updateRouteStepper()', start);
assert.ok(start >= 0 && end > start, 'route badge helpers are located');

function load(order, step) {
  const routeIndexById = new Map(order.map((id, i) => [id, i + 1]));
  const context = vm.createContext({ routeIndexById, routeStep: step });
  vm.runInContext(source.slice(start, end), context);
  return context;
}

test('overview marks every route node "all" and others null', () => {
  const ctx = load(['hub', 'a', 'b'], -1);
  assert.equal(ctx.routeBadgeStateOf({ id: 'hub' }), 'all');
  assert.equal(ctx.routeBadgeStateOf({ id: 'b' }), 'all');
  assert.equal(ctx.routeBadgeStateOf({ id: 'zzz' }), null);
  const b = ctx.routeBadgeOf({ id: 'b' });
  assert.equal(b.index, 3);
  assert.equal(b.state, 'all');
  assert.equal(ctx.routeBadgeOf({ id: 'zzz' }), null);
});

test('stepping splits route into done / current / todo by prose order', () => {
  const ctx = load(['hub', 'a', 'b', 'c'], 2);
  assert.equal(ctx.routeBadgeStateOf({ id: 'hub' }), 'done');
  assert.equal(ctx.routeBadgeStateOf({ id: 'a' }), 'done');
  assert.equal(ctx.routeBadgeStateOf({ id: 'b' }), 'current');
  assert.equal(ctx.routeBadgeStateOf({ id: 'c' }), 'todo');
});
