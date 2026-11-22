const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');
const { JSDOM } = require('jsdom');

const readDoc = (file) => readFileSync(resolve(__dirname, '../docs', file), 'utf8');

test('graph.html showTooltip keeps .hidden until rAF sets left/top and stays idempotent on same-node moves', async () => {
  const html = readDoc('graph.html');
  const scriptStart = html.indexOf("    const tooltip = document.getElementById('graph-tooltip');");
  const start = html.indexOf('    function tooltipSummary(raw) {', scriptStart);
  const end = html.indexOf('    // 移动端 tooltip 内链接点击：先跳转', start);
  assert.ok(scriptStart > 0 && start > scriptStart && end > start, 'tooltip section found in graph.html');

  const dom = new JSDOM('<!doctype html><html><body><div id="graph-tooltip" class="hidden" role="tooltip"></div></body></html>', {
    url: 'https://example.test/graph.html',
  });
  const w = dom.window;
  try {
    const tooltipEl = w.document.getElementById('graph-tooltip');
    Object.defineProperty(tooltipEl, 'offsetWidth', { configurable: true, get: () => 220 });
    Object.defineProperty(tooltipEl, 'offsetHeight', { configurable: true, get: () => 140 });

    let rafSeq = 0;
    const rafQueue = new Map();
    const statesObserved = [];
    let renderCalls = 0;

    const context = vm.createContext({
      window: Object.assign(w, {
        innerWidth: 1280,
        innerHeight: 800,
        RNGraphTooltip: {
          formatTooltipSummary: (s) => s || '',
          buildNodeTooltipHtml: (opts) => {
            renderCalls += 1;
            return `<div class="tt-title">${opts.title}</div>`;
          },
          renderMath: () => {},
        },
      }),
      document: w.document,
      tooltip: tooltipEl,
      requestAnimationFrame: (cb) => {
        const id = ++rafSeq;
        rafQueue.set(id, cb);
        return id;
      },
      cancelAnimationFrame: (id) => rafQueue.delete(id),
      ensureSummaries: () => {},
      pageUrlForNode: (d) => `detail.html?id=${d.id}`,
      escapeHtml: (s) => String(s),
      pageLinkLabelForNode: () => '打开详情页 →',
      TYPE_COLOR: {},
      metaBadgeColor: () => '#60a5fa',
      communityLabelMap: new Map(),
      communityColorMap: new Map(),
      COMMUNITY_COLORS: ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac'],
      pinnedNode: null,
      sidebarNode: null,
      TYPE_FALLBACK: '',
    });

    vm.runInContext(html.slice(start, end), context);
    context.isTouchInteraction = () => false;

    const recordSnapshot = (label) => {
      statesObserved.push({
        label,
        hidden: tooltipEl.classList.contains('hidden'),
        left: tooltipEl.style.left,
        top: tooltipEl.style.top,
      });
    };

    const nodeA = { id: 'wiki/a.md', label: 'Node A', _info: { title: 'Node A', summary: 'Summary A' } };
    const nodeB = { id: 'wiki/b.md', label: 'Node B', _info: { title: 'Node B', summary: 'Summary B' } };

    // Missing coords must never reveal tooltip at (0, 0)
    context.showTooltip(null, nodeA, false);
    assert.equal(tooltipEl.classList.contains('hidden'), true);

    // Simulate 3D hover entering node A: before rAF fires, tooltip MUST remain hidden (no top-left flash)
    context.showTooltip({ clientX: 400, clientY: 300 }, nodeA, false);
    recordSnapshot('before-first-raf');
    assert.equal(tooltipEl.classList.contains('hidden'), true, 'tooltip stays hidden until positioned');

    // Run rAF: left/top are set and .hidden is removed together
    const firstCallbacks = Array.from(rafQueue.entries());
    rafQueue.clear();
    for (const [, cb] of firstCallbacks) cb();
    recordSnapshot('after-first-raf');
    assert.equal(tooltipEl.classList.contains('hidden'), false);
    assert.equal(tooltipEl.style.left, '414px');
    assert.equal(tooltipEl.style.top, '290px');
    assert.equal(renderCalls, 1);

    // Moving within same node in 3D must NOT clear left/top or re-render HTML
    context.showTooltip({ clientX: 410, clientY: 305 }, nodeA, false);
    recordSnapshot('same-node-move-before-raf');
    assert.equal(tooltipEl.style.left, '414px', 'existing left kept before rAF updates it');
    assert.equal(tooltipEl.style.top, '290px', 'existing top kept before rAF updates it');
    assert.equal(renderCalls, 1, 'same-node move does not rebuild tooltip HTML');

    for (const [, cb] of Array.from(rafQueue.entries())) cb();
    rafQueue.clear();
    assert.equal(tooltipEl.style.left, '424px');
    assert.equal(tooltipEl.style.top, '295px');

    // Transitioning directly from node A to node B keeps valid position instead of clearing to (0, 0)
    context.showTooltip({ clientX: 430, clientY: 320 }, nodeB, false);
    assert.equal(tooltipEl.style.left, '424px', 'no clearing left to empty string on node change');
    assert.equal(tooltipEl.style.top, '295px', 'no clearing top to empty string on node change');
    for (const [, cb] of Array.from(rafQueue.entries())) cb();
    rafQueue.clear();
    assert.equal(tooltipEl.style.left, '444px');
    assert.equal(tooltipEl.style.top, '310px');
    assert.equal(renderCalls, 2);

    // Switching from pinned back to hover stays hidden until rAF writes left/top
    context.showTooltip(null, nodeA, true);
    assert.equal(tooltipEl.classList.contains('tt-pinned'), true);
    assert.equal(tooltipEl.style.left, '');
    context.showTooltip({ clientX: 500, clientY: 250 }, nodeB, false);
    assert.equal(tooltipEl.classList.contains('hidden'), true, 'pinned->hover hides until left/top applied');
    for (const [, cb] of Array.from(rafQueue.entries())) cb();
    rafQueue.clear();
    assert.equal(tooltipEl.classList.contains('hidden'), false);
    assert.equal(tooltipEl.style.left, '514px');
    assert.equal(tooltipEl.style.top, '240px');
  } finally {
    w.close();
  }
});

test('graph-3d.js initializes lastPointer to null and falls back to node screen coords when pointer has not moved', () => {
  const src = readDoc('graph-3d.js');
  assert.match(src, /var lastPointer = \{ clientX: null, clientY: null \};/);
  assert.ok(src.includes('graph.graph2ScreenCoords(node.x, node.y, node.z != null ? node.z : 0)'));
});

test('main.js setupGraphHoverTooltip positions before removing .hidden and ignores missing pointer coords', () => {
  const mainJs = readDoc('main.js');
  const start = mainJs.indexOf('  function setupGraphHoverTooltip(tooltipEl) {');
  const end = mainJs.indexOf('  function buildPathToDetailIdIndex(', start);
  assert.ok(start > 0 && end > start);

  const dom = new JSDOM('<!doctype html><html><body><div id="tip" class="graph-hover-tooltip hidden"></div></body></html>', {
    url: 'https://example.test/detail.html',
  });
  const w = dom.window;
  try {
    const tipEl = w.document.getElementById('tip');
    Object.defineProperty(tipEl, 'offsetWidth', { configurable: true, get: () => 200 });
    Object.defineProperty(tipEl, 'offsetHeight', { configurable: true, get: () => 120 });

    const seenBeforeVisible = [];
    const origRemove = tipEl.classList.remove.bind(tipEl.classList);
    tipEl.classList.remove = (...args) => {
      if (args.includes('hidden')) {
        seenBeforeVisible.push({ left: tipEl.style.left, top: tipEl.style.top });
      }
      return origRemove(...args);
    };

    const ctx = vm.createContext({
      window: Object.assign(w, { innerWidth: 1280, innerHeight: 800, matchMedia: () => ({ matches: false }) }),
      document: w.document,
      renderDetailMath: () => {},
      Number,
      Math,
    });
    vm.runInContext(mainJs.slice(start, end), ctx);
    const tip = ctx.setupGraphHoverTooltip(tipEl);

    // Missing coordinates must keep tooltip hidden
    tip.show(null, { id: 'a' }, '<div>A</div>', false);
    assert.equal(tipEl.classList.contains('hidden'), true);

    // Valid hover sets left/top BEFORE removing .hidden
    tip.show({ clientX: 300, clientY: 200 }, { id: 'a' }, '<div>A</div>', false);
    assert.equal(tipEl.classList.contains('hidden'), false);
    assert.deepEqual(seenBeforeVisible, [{ left: '314px', top: '190px' }]);

    // Near top/left corner after flip clamps to >= 12px instead of negative off-screen
    tip.show({ clientX: 5, clientY: 5 }, { id: 'b' }, '<div>B</div>', false);
    assert.equal(tipEl.style.left, '19px');
    assert.equal(tipEl.style.top, '12px');
  } finally {
    w.close();
  }
});

test('detail mini-map 2D node drag reheats simulation and rebounds smoothly on release', async (t) => {
  const dom = new JSDOM(readDoc('detail.html'), {
    url: 'https://example.test/detail.html?id=demo',
    runScripts: 'outside-only',
  });
  const w = dom.window;
  t.after(() => w.close());
  w.matchMedia = () => ({ matches: false });

  const current = 'wiki/concepts/current.md';
  const neighborA = 'wiki/concepts/a.md';
  const neighborB = 'wiki/concepts/b.md';
  const graph = {
    nodes: [current, neighborA, neighborB].map((id) => ({ id, label: id, type: 'concept' })),
    edges: [{ source: current, target: neighborA }, { source: current, target: neighborB }],
    communities: [],
  };

  Object.assign(w, {
    ensureLinkGraphData: async () => graph,
    buildPathToDetailIdIndex: () => ({}),
    setupGraphHoverTooltip: () => ({ bindBlankDismiss() {}, bindOutsideDismiss() {}, hide() {}, move() {}, getPinned: () => null }),
    detailLinkBridge: {},
    detailBridgeHighlightBody() {},
  });
  // Provide viewport geometry & SVG bounding rect for D3 pointer/drag in JSDOM
  const vp = w.document.getElementById('detailMiniMapViewport');
  Object.defineProperty(vp, 'clientWidth', { configurable: true, get: () => 700 });
  Object.defineProperty(vp, 'clientHeight', { configurable: true, get: () => 360 });
  const svgEl = w.document.getElementById('detailMiniMapSvg');
  svgEl.getBoundingClientRect = () => ({ left: 0, top: 0, width: 700, height: 360, right: 700, bottom: 360 });
  const origCreateElNS = w.document.createElementNS.bind(w.document);
  w.document.createElementNS = (ns, tag) => {
    const el = origCreateElNS(ns, tag);
    el.getBoundingClientRect = () => ({ left: 0, top: 0, width: 700, height: 360, right: 700, bottom: 360 });
    return el;
  };

  for (const file of ['vendor/d3.min.js', 'graph-node-size.js']) w.eval(readDoc(file));
  const main = readDoc('main.js');
  w.eval(main.slice(main.indexOf('  function renderDetailMiniMap('), main.indexOf('  function renderDetailPage(')));
  w.renderDetailMiniMap({ path: current }, {});
  await new Promise(setImmediate);

  const neighborEl = w.document.querySelector('#detailMiniMapSvg g.mini-node');
  assert.ok(neighborEl, 'rendered neighbor node');
  const d = neighborEl.__data__;
  const restX = d.x;
  const restY = d.y;

  // Simulate pointer drag + release on the SVG node using D3 drag events
  const cam = w.d3.zoomTransform(svgEl);
  const startScreenX = cam.applyX(restX);
  const startScreenY = cam.applyY(restY);
  neighborEl.dispatchEvent(new w.MouseEvent('mousedown', { bubbles: true, cancelable: true, view: w, button: 0, clientX: startScreenX, clientY: startScreenY }));
  w.dispatchEvent(new w.MouseEvent('mousemove', { bubbles: true, cancelable: true, view: w, clientX: startScreenX + 120, clientY: startScreenY + 60 }));
  assert.ok(Math.hypot(d.x - restX, d.y - restY) > 40, 'node moved during 2D drag');
  w.dispatchEvent(new w.MouseEvent('mouseup', { bubbles: true, cancelable: true, view: w, clientX: startScreenX + 120, clientY: startScreenY + 60 }));

  const releasedDist = Math.hypot(d.x - restX, d.y - restY);
  assert.equal(d.fx, null, 'fx cleared on drag end to allow rebound');
  assert.equal(d.fy, null, 'fy cleared on drag end to allow rebound');

  // Let d3 simulation timer tick and rebound the node
  await new Promise((r) => setTimeout(r, 1200));
  const afterDist = Math.hypot(d.x - restX, d.y - restY);
  assert.ok(afterDist < releasedDist * 0.45, `node rebounded cleanly (released=${releasedDist.toFixed(1)}, after=${afterDist.toFixed(1)})`);
});

test('mini-graph.js showTooltip positions before removing .hidden and skips DOM rebuild on same-node hover', () => {
  const miniJs = readDoc('mini-graph.js');
  const start = miniJs.indexOf('  var pinnedNode = null;');
  const end = miniJs.indexOf('  var TABLEAU10 =', start);
  assert.ok(start > 0 && end > start);

  const dom = new JSDOM('<!doctype html><html><body><div id="mini-graph-tooltip" class="graph-hover-tooltip hidden"></div></body></html>', {
    url: 'https://example.test/index.html',
  });
  const w = dom.window;
  try {
    const tt = w.document.getElementById('mini-graph-tooltip');
    Object.defineProperty(tt, 'offsetWidth', { configurable: true, get: () => 200 });
    Object.defineProperty(tt, 'offsetHeight', { configurable: true, get: () => 120 });

    const seenBeforeReveal = [];
    const origRemove = tt.classList.remove.bind(tt.classList);
    tt.classList.remove = (...args) => {
      if (args.includes('hidden')) {
        seenBeforeReveal.push({ left: tt.style.left, top: tt.style.top });
      }
      return origRemove(...args);
    };

    let buildCount = 0;
    const ctx = vm.createContext({
      window: Object.assign(w, {
        innerWidth: 1280,
        innerHeight: 800,
        RNGraphTooltip: {
          formatTooltipSummary: (s) => s || '',
          buildNodeTooltipHtml: (opts) => {
            buildCount += 1;
            return `<div class="tt-title">${opts.title}</div>`;
          },
          renderMath: () => {},
        },
      }),
      document: w.document,
      tooltip: tt,
      isMobile: false,
      Number,
      Math,
    });
    vm.runInContext(miniJs.slice(start, end), ctx);

    const node = { id: 'wiki/a.md', label: 'Node A', summary: 'S' };
    ctx.showTooltip(null, node, () => '#60a5fa', {}, false);
    assert.equal(tt.classList.contains('hidden'), true, 'unpositioned hover stays hidden');

    ctx.showTooltip({ clientX: 250, clientY: 180 }, node, () => '#60a5fa', {}, false);
    assert.equal(tt.classList.contains('hidden'), false);
    assert.deepEqual(seenBeforeReveal, [{ left: '264px', top: '170px' }]);
    assert.equal(buildCount, 1);

    ctx.showTooltip({ clientX: 260, clientY: 190 }, node, () => '#60a5fa', {}, false);
    assert.equal(buildCount, 1, 'same-node hover move reuses rendered tooltip HTML');
    assert.equal(tt.style.left, '274px');
    assert.equal(tt.style.top, '180px');
  } finally {
    w.close();
  }
});
