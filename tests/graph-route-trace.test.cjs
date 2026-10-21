const assert = require('node:assert/strict');
const { test } = require('node:test');
const { readFileSync } = require('node:fs');
const { JSDOM } = require('jsdom');
const source = readFileSync(require('node:path').join(__dirname, '../docs/graph-route-trace.js'), 'utf8');

function setup(reduced = false) {
  const dom = new JSDOM('<svg></svg>', { runScripts: 'outside-only' });
  const w = dom.window;
  const frames = new Map();
  let seq = 0;
  const motion = { matches: reduced, addEventListener() {}, removeEventListener() {} };
  w.matchMedia = () => motion;
  w.requestAnimationFrame = fn => { frames.set(++seq, fn); return seq; };
  w.cancelAnimationFrame = id => frames.delete(id);
  w.eval(source);
  const svg = w.document.querySelector('svg');
  const coords = { hub: { x: 0, y: 0 }, z: { x: 10, y: 0 }, a: { x: 10, y: 10 } };
  const hidden = new Set();
  let active = true;
  const trace = w.RNGraphRouteTrace.create({ svg, project: id => coords[id],
    isVisible: id => !hidden.has(id), isActive: () => active });
  const step = now => { const batch = [...frames.values()]; frames.clear(); batch.forEach(fn => fn(now)); };
  return { trace, svg, coords, hidden, frames, step, motion, setActive: on => { active = on; } };
}

test('grows from hub through the supplied order, completes and releases RAF', () => {
  const s = setup();
  s.trace.setRoute(['hub', 'z', 'a']);
  s.step(0); s.step(600);
  assert.equal(s.svg.querySelector('path').getAttribute('d'), 'M0,0L5,0');
  s.step(1800);
  assert.equal(s.svg.querySelector('path').getAttribute('d'), 'M0,0L10,0M10,0L10,5');
  s.step(2400);
  assert.equal(s.trace.getState().progress, 1);
  assert.equal(s.frames.size, 0);
  s.coords.z.x = 20;
  s.trace.render();
  assert.equal(s.svg.querySelector('path').getAttribute('d'), 'M0,0L20,0M20,0L10,10');
});

test('rapid switches cancel the old animation, repeated selection replays, all clears', () => {
  const s = setup();
  s.trace.setRoute(['hub', 'z']); s.step(0); s.step(600);
  s.trace.setRoute(['hub', 'a']);
  assert.equal(s.frames.size, 1);
  s.step(1000); s.step(3400);
  assert.equal(s.svg.querySelector('path').getAttribute('d'), 'M0,0L10,10');
  s.trace.setRoute(['hub', 'a']);
  assert.equal(s.trace.getState().progress, 0);
  s.trace.setRoute([]);
  assert.equal(s.svg.style.display, 'none');
  assert.equal(s.frames.size, 0);
});

test('hidden middle nodes leave gaps and timeline can hide the overlay', () => {
  const s = setup(true);
  s.trace.setRoute(['hub', 'z', 'a']);
  s.hidden.add('z'); s.trace.render();
  assert.equal(s.svg.querySelector('path').getAttribute('d'), '');
  s.setActive(false); s.trace.render();
  assert.equal(s.svg.style.display, 'none');
  s.setActive(true); s.hidden.clear(); s.trace.render();
  assert.equal(s.svg.querySelector('path').getAttribute('d'), 'M0,0L10,0M10,0L10,10');
});

test('reduced motion draws the complete route with no animation task', () => {
  const s = setup(true);
  s.trace.setRoute(['hub', 'z', 'a']);
  assert.equal(s.trace.getState().progress, 1);
  assert.equal(s.frames.size, 0);
  s.trace.destroy();
  assert.equal(s.svg.children.length, 0);
});
