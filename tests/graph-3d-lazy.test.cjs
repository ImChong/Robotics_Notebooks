// 完整图谱页 2D 浏览不加载 3D 库，进入 3D 时才注入（编号 9）
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

const source = readFileSync(resolve(__dirname, '../docs/graph.html'), 'utf8');
const start = source.indexOf('    /* ── 3D 库懒加载（编号 9）──');
const end = source.indexOf('    function setSpatialViewMode(', start);
assert.ok(start >= 0 && end > start, 'lazy 3D loader is located');

const BUNDLE = 'vendor/3d-force-graph.min.js';
const flush = () => new Promise(setImmediate);

function harness(options = {}) {
  const appended = [], switches = [], busy = [];
  let available = false;
  const head = {
    appendChild(script) {
      appended.push(script.src);
      script.parentNode = head;
      setImmediate(() => {
        if (options.failLoad && appended.length === 1) { script.onerror(); return; }
        available = true;
        script.onload();
      });
    },
    removeChild() {},
  };
  const button = {
    disabled: false, attributes: {},
    setAttribute(name, value) { this.attributes[name] = value; busy.push(true); },
    removeAttribute(name) { delete this.attributes[name]; busy.push(false); },
  };
  const context = vm.createContext({
    Promise, console: { warn() {} },
    window: { RNGraph3D: { isAvailable: () => available } },
    document: { createElement: () => ({ src: '', onload: null, onerror: null, parentNode: null }), head },
    viewMode3dBtn: button,
    setSpatialViewMode: (mode, opts) => switches.push([mode, opts]),
  });
  vm.runInContext(source.slice(start, end), context);
  return { context, appended, switches, button, busy };
}

test('staying in 2D never downloads the 3D bundle', async () => {
  const h = harness();
  await h.context.requestSpatialViewMode('2d');
  await flush();
  assert.deepEqual(h.appended, []);
  assert.equal(h.switches[0][0], '2d');
});

test('entering 3D loads the bundle once and then switches', async () => {
  const h = harness();
  const pending = h.context.requestSpatialViewMode('3d', { force: true });
  assert.deepEqual(h.appended, [BUNDLE]);
  assert.equal(h.button.disabled, true, 'the 3D button is busy while the bundle loads');
  await pending;
  assert.equal(h.button.disabled, false);
  assert.deepEqual(h.switches, [['3d', { force: true }]]);
  await h.context.requestSpatialViewMode('3d');
  assert.equal(h.appended.length, 1, 'an already loaded bundle is not fetched again');
  assert.equal(h.switches.length, 2);
});

test('a failed bundle load reaches the existing 2D fallback and can be retried', async () => {
  const h = harness({ failLoad: true });
  await h.context.requestSpatialViewMode('3d');
  assert.deepEqual(h.appended, [BUNDLE]);
  assert.equal(h.button.disabled, false, 'the button is usable again after a failure');
  assert.deepEqual(h.switches, [['3d', undefined]], '交给 setSpatialViewMode 既有的失败恢复路径');
  await h.context.requestSpatialViewMode('3d');
  assert.equal(h.appended.length, 2, 'the failed load is not cached');
});

test('the graph page ships no unconditional 3D bundle tag', () => {
  assert.ok(!/<script[^>]+3d-force-graph\.min\.js/.test(source), 'graph.html must not preload the 3D bundle');
});
