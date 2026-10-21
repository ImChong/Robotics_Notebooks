const assert = require('node:assert/strict');
const { test } = require('node:test');
const { readFileSync } = require('node:fs');
const { JSDOM } = require('jsdom');
const source = readFileSync(require('node:path').join(__dirname, '../docs/graph-route-trace.js'), 'utf8');

function setup(reduced = false) {
  const dom = new JSDOM('<svg></svg>', { runScripts: 'outside-only' });
  const w = dom.window, frames = new Map();
  let seq = 0;
  const motion = { matches: reduced, addEventListener() {}, removeEventListener() {} };
  w.matchMedia = () => motion;
  w.requestAnimationFrame = fn => { frames.set(++seq, fn); return seq; };
  w.cancelAnimationFrame = id => frames.delete(id);
  w.eval(source);
  const svg = w.document.querySelector('svg');
  const coords = { hub: { x: 0, y: 0 }, z: { x: 100, y: 0 }, a: { x: 100, y: 100 } };
  const hidden = new Set();
  let active = true;
  const trace = w.RNGraphRouteTrace.create({ svg, project: id => coords[id],
    isVisible: id => !hidden.has(id), isActive: () => active });
  const step = now => { const batch = [...frames.values()]; frames.clear(); batch.forEach(fn => fn(now)); };
  return { trace, svg, coords, hidden, frames, step, motion, helpers: w.RNGraphRouteTrace,
    path: () => svg.querySelector('.route-trace-core').getAttribute('d'),
    head: () => ({ x: +svg.querySelector('.route-trace-head').getAttribute('cx'),
      y: +svg.querySelector('.route-trace-head').getAttribute('cy') }),
    setActive: on => { active = on; } };
}

test('straight line grows through nodes as one connected path with a moving light', () => {
  const s = setup(); s.trace.setRoute(['hub', 'z', 'a']);
  s.step(0); s.step(700);
  const early = s.head();
  assert.ok(early.x > 0 && early.x < 100 && early.y === 0);
  assert.doesNotMatch(s.path(), /C/, 'segments are straight');
  s.step(2100);
  assert.ok(s.head().y > 0 && s.head().y < 100);
  assert.equal((s.path().match(/M/g) || []).length, 1, 'no restart at each node');
  assert.ok(s.svg.querySelector('.route-trace-tail').getAttribute('d').length > 0);
  assert.equal(s.svg.querySelector('.route-trace-flow').getAttribute('d'), s.path());
  s.step(2800);
  assert.equal(s.trace.getState().progress, 1);
  assert.equal(s.frames.size, 0, 'native CSS handles subtle flow after growth');
  assert.deepEqual(s.head(), { x: 100, y: 100 });
  const before = s.path(); s.coords.z.x = 200; s.trace.render();
  assert.notEqual(s.path(), before, 'line follows changed layout or projection');
});

test('arc length maps halfway to halfway distance across unequal edges', () => {
  const s = setup();
  const p = [{ x: 0, y: 0 }, { x: 10, y: 0 }, { x: 110, y: 0 }];
  const g = s.helpers.geometry([[p[0], p[1], true], [p[1], p[2], true]]);
  const halfway = s.helpers.slice(g, 0, g.total / 2);
  assert.ok(Math.abs(halfway.tip.x - 55) < 0.3, 'speed must not jump at a short edge');
  assert.equal((s.helpers.slice(g, 0, g.total).d.match(/M/g) || []).length, 1);
});

test('branches connect lit nodes missing from the prose order', () => {
  const s = setup(true);
  const adjacency = new Map([['hub', new Set(['x'])], ['x', new Set(['hub', 'y'])], ['y', new Set(['x'])]]);
  const lit = new Set(['hub', 'z', 'x', 'y', 'lone']);
  const community = id => (id === 'lone' ? 'c1' : id === 'z' ? 'c1' : 'c0');
  const out = s.helpers.branches(['hub', 'z'], lit, adjacency, community);
  assert.deepEqual(JSON.parse(JSON.stringify(out)), [['hub', 'x'], ['x', 'y'], ['z', 'lone']]);
  s.coords.x = { x: -100, y: 0 }; s.coords.y = { x: -100, y: -100 }; s.coords.lone = { x: 200, y: 0 };
  s.trace.setRoute(['hub', 'z'], out);
  assert.equal((s.path().match(/M/g) || []).length, 4, 'chain plus three branch starts');
  assert.ok(s.path().includes('-100.00,-100.00'));
  assert.deepEqual(s.head(), { x: 200, y: 0 });
});

test('rapid switches cancel previous RAF, repeat replays, all clears', () => {
  const s = setup(); s.trace.setRoute(['hub', 'z']); s.step(0); s.step(600);
  s.trace.setRoute(['hub', 'a']); assert.equal(s.frames.size, 1);
  s.step(1000); s.step(3800);
  assert.deepEqual(s.head(), { x: 100, y: 100 });
  s.trace.setRoute(['hub', 'a']); assert.equal(s.trace.getState().progress, 0);
  s.trace.setRoute([]); assert.equal(s.svg.style.display, 'none'); assert.equal(s.frames.size, 0);
});

test('hidden nodes break the line without joining across the gap', () => {
  const s = setup(true); s.trace.setRoute(['hub', 'z', 'a']);
  s.hidden.add('z'); s.trace.render(); assert.equal(s.path(), '');
  const q = [{ x: 0, y: 0 }, { x: 10, y: 0 }, null, { x: 50, y: 0 }, { x: 60, y: 0 }];
  const g = s.helpers.geometry(q.slice(1).map((b, i) => [q[i], b, true]));
  assert.equal((s.helpers.slice(g, 0, g.total).d.match(/M/g) || []).length, 2);
  s.setActive(false); s.trace.render(); assert.equal(s.svg.style.display, 'none');
  s.setActive(true); s.hidden.clear(); s.trace.render(); assert.ok(s.path());
});

test('reduced motion immediately shows static line and disables shimmer', () => {
  const s = setup(true); s.trace.setRoute(['hub', 'z', 'a']);
  assert.equal(s.trace.getState().progress, 1); assert.equal(s.frames.size, 0);
  assert.ok(s.svg.classList.contains('route-trace-reduced'));
  assert.equal(s.svg.querySelector('.route-trace-tail').getAttribute('d'), '');
  s.trace.destroy(); assert.equal(s.svg.children.length, 0);
});
