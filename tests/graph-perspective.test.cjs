const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

const perspectiveApi = require('../docs/graph-perspective.js');
const graphHtml = readFileSync(resolve(__dirname, '../docs/graph.html'), 'utf8');
const swSource = readFileSync(resolve(__dirname, '../docs/sw.js'), 'utf8');

function button() {
  return {
    disabled: false,
    attributes: {},
    classes: new Set(),
    classList: {
      toggle(name, on) {
        if (on) this._set.add(name);
        else this._set.delete(name);
      },
      contains(name) {
        return this._set.has(name);
      },
    },
    setAttribute(name, value) { this.attributes[name] = value; },
    removeAttribute(name) { delete this.attributes[name]; },
    init() { this.classList._set = this.classes; return this; },
  }.init();
}

test('tier defaults and clamping preserve valid ranges and recover from malformed values', () => {
  assert.deepEqual(perspectiveApi.getDefaultsForTier('full'), {
    perspective: 1150,
    rotateX: 34,
    rotateY: -6,
    depth: 60,
  });
  assert.deepEqual(perspectiveApi.getDefaultsForTier('lite'), {
    perspective: 1200,
    rotateX: 26,
    rotateY: 0,
    depth: 40,
  });
  assert.deepEqual(perspectiveApi.getDefaultsForTier('reduced'), {
    perspective: 1350,
    rotateX: 16,
    rotateY: 0,
    depth: 25,
  });

  const clampedFull = perspectiveApi.clampParams({
    perspective: 9999,
    rotateX: 80,
    rotateY: -99,
    depth: -20,
  }, 'full');
  assert.deepEqual(clampedFull, {
    perspective: 2000,
    rotateX: 58,
    rotateY: -28,
    depth: 0,
  });

  const clampedLite = perspectiveApi.clampParams({
    perspective: '1320',
    rotateX: 56,
    rotateY: '12',
    depth: 120,
  }, 'lite');
  assert.deepEqual(clampedLite, {
    perspective: 1320,
    rotateX: 50,
    rotateY: 12,
    depth: 100,
  });

  const recovered = perspectiveApi.clampParams({
    perspective: NaN,
    rotateX: null,
    rotateY: 'invalid',
    depth: Infinity,
  }, 'full');
  assert.deepEqual(recovered, perspectiveApi.getDefaultsForTier('full'));
});

test('resolveCapabilityTier handles unsupported, reduced motion, lite mobile/low-end, and full desktop', () => {
  assert.equal(perspectiveApi.resolveCapabilityTier({
    supports3D: false,
    reducedMotion: true,
    coarsePointer: true,
  }), 'unsupported');

  assert.equal(perspectiveApi.resolveCapabilityTier({
    supports3D: true,
    reducedMotion: true,
    coarsePointer: false,
    hardwareConcurrency: 16,
    deviceMemory: 32,
  }), 'reduced');

  assert.equal(perspectiveApi.resolveCapabilityTier({
    supports3D: true,
    reducedMotion: false,
    coarsePointer: true,
    hardwareConcurrency: 8,
    deviceMemory: 8,
  }), 'lite');

  assert.equal(perspectiveApi.resolveCapabilityTier({
    supports3D: true,
    reducedMotion: false,
    coarsePointer: false,
    hardwareConcurrency: 4,
    deviceMemory: 8,
  }), 'lite');

  assert.equal(perspectiveApi.resolveCapabilityTier({
    supports3D: true,
    reducedMotion: false,
    coarsePointer: false,
    hardwareConcurrency: 8,
    deviceMemory: 4,
  }), 'lite');

  assert.equal(perspectiveApi.resolveCapabilityTier({
    supports3D: true,
    reducedMotion: false,
    coarsePointer: false,
    hardwareConcurrency: 8,
    deviceMemory: 16,
    dpr: 3,
    nodeCount: 7800,
  }), 'lite');

  assert.equal(perspectiveApi.resolveCapabilityTier({
    supports3D: true,
    reducedMotion: false,
    coarsePointer: false,
    hardwareConcurrency: 8,
    deviceMemory: 16,
    dpr: 2,
    nodeCount: 7800,
  }), 'full');
});

test('projectSvgToClient and unprojectClientToSvg round-trip accurately across tilt configurations', () => {
  const configs = [
    { params: { perspective: 1150, rotateX: 0, rotateY: 0, depth: 0 }, w: 1200, h: 800, left: 0, top: 0, tier: 'full' },
    { params: { perspective: 1150, rotateX: 34, rotateY: -6, depth: 60 }, w: 1440, h: 796, left: 0, top: 104, tier: 'full' },
    { params: { perspective: 900, rotateX: 58, rotateY: 28, depth: 100 }, w: 1280, h: 720, left: 20, top: 80, tier: 'full' },
    { params: { perspective: 700, rotateX: 58, rotateY: -28, depth: 100 }, w: 1100, h: 700, left: 0, top: 104, tier: 'full' },
    { params: { perspective: 1200, rotateX: 26, rotateY: 0, depth: 40 }, w: 390, h: 740, left: 0, top: 104, tier: 'lite' },
    { params: { perspective: 1350, rotateX: 16, rotateY: 0, depth: 25 }, w: 1440, h: 800, left: 0, top: 104, tier: 'reduced' },
  ];
  const samplePoints = [
    [0.12, 0.15],
    [0.5, 0.5],
    [0.85, 0.2],
    [0.2, 0.85],
    [0.88, 0.88],
  ];

  for (const cfg of configs) {
    const metrics = perspectiveApi.computeSceneMetrics(cfg.params, cfg.w, cfg.h, cfg.tier);
    const stageRect = { left: cfg.left, top: cfg.top, width: cfg.w, height: cfg.h };
    for (const [fx, fy] of samplePoints) {
      const u = cfg.w * fx;
      const v = cfg.h * fy;
      const client = perspectiveApi.projectSvgToClient(u, v, metrics, stageRect);
      const back = perspectiveApi.unprojectClientToSvg(client[0], client[1], metrics, stageRect);
      const err = Math.hypot(back[0] - u, back[1] - v);
      assert.ok(err < 1e-6, `round-trip drift ${err} too large for ${JSON.stringify(cfg.params)}`);
    }
  }
});

test('unprojectClientToSvg remains finite above the vanishing horizon', () => {
  const metrics = perspectiveApi.computeSceneMetrics({
    perspective: 700,
    rotateX: 58,
    rotateY: 28,
    depth: 100,
  }, 1200, 800, 'full');
  const stageRect = { left: 0, top: 104, width: 1200, height: 800 };
  for (const [sx, sy] of [[-5000, -10000], [600, -5000], [5000, -8000]]) {
    const uv = perspectiveApi.unprojectClientToSvg(sx, sy, metrics, stageRect);
    assert.ok(Number.isFinite(uv[0]) && Number.isFinite(uv[1]), 'must stay finite near/past horizon');
  }
});

test('installSvgCoordinateAdapter transparently maps SVG and world coordinates only while active', () => {
  let active = false;
  const nativeSvgMarker = { tag: 'native-svg' };
  const nativeNodeMarker = { tag: 'native-node' };
  const svgEl = {
    createSVGPoint() {
      return { x: 0, y: 0, matrixTransform: (m) => ({ x: this.x + 1, y: this.y + 1, marker: m }) };
    },
    getScreenCTM() {
      return { inverse: () => nativeSvgMarker };
    },
  };
  const gNodesEl = {
    getScreenCTM() {
      return { inverse: () => nativeNodeMarker };
    },
  };
  const adapter = perspectiveApi.installSvgCoordinateAdapter({
    svgEl,
    gNodesEl,
    isAdapterActive: () => active,
    getZoomTransform: () => ({ x: 100, y: 50, k: 2 }),
    unprojectClientPoint: (cx, cy) => [cx * 0.5 + 10, cy * 0.5 + 20],
  });

  // Inactive: native delegation
  const pInactive = svgEl.createSVGPoint();
  pInactive.x = 40;
  pInactive.y = 60;
  assert.equal(svgEl.getScreenCTM().inverse(), nativeSvgMarker);
  assert.equal(gNodesEl.getScreenCTM().inverse(), nativeNodeMarker);
  assert.equal(pInactive.matrixTransform(svgEl.getScreenCTM().inverse()).marker, nativeSvgMarker);

  // Active: SVG + world unprojection
  active = true;
  const pSvg = svgEl.createSVGPoint();
  pSvg.x = 200;
  pSvg.y = 400;
  const outSvg = pSvg.matrixTransform(svgEl.getScreenCTM().inverse());
  assert.deepEqual([outSvg.x, outSvg.y], [110, 220]);

  const pWorld = svgEl.createSVGPoint();
  pWorld.x = 200;
  pWorld.y = 400;
  const outWorld = pWorld.matrixTransform(gNodesEl.getScreenCTM().inverse());
  assert.deepEqual([outWorld.x, outWorld.y], [(110 - 100) / 2, (220 - 50) / 2]);

  // Real/unknown matrix still delegates to native point
  const unknown = pWorld.matrixTransform(nativeSvgMarker);
  assert.equal(unknown.marker, nativeSvgMarker);

  adapter.uninstall();
  active = true;
  assert.equal(svgEl.getScreenCTM().inverse(), nativeSvgMarker);
  assert.equal(gNodesEl.getScreenCTM().inverse(), nativeNodeMarker);
});

test('graph.html supports 3-way mutual exclusion across 2D, CSS perspective, and WebGL 3D without lazy-loading 3D in perspective mode', () => {
  const controlsStart = graphHtml.indexOf('    function is3DView()');
  const controlsEnd = graphHtml.indexOf('    function colorModeClusterKey');
  const lazyStart = graphHtml.indexOf('    /* ── 3D 库懒加载（编号 9）──');
  const transitionStart = graphHtml.indexOf('    function setSpatialViewMode(');
  const transitionEnd = graphHtml.indexOf('    function syncTimeline3DState(');
  assert.ok(controlsStart >= 0 && controlsEnd > controlsStart);
  assert.ok(lazyStart >= 0 && transitionStart > lazyStart && transitionEnd > transitionStart);

  const storage = new Map();
  const appended = [];
  const perspectiveCalls = [];
  let svgDisplay = 'block';
  let paused = 0;
  let resumed = 0;
  let forceUpdates = 0;

  const view3d = {
    show() {},
    hide() {},
    syncFilters() {},
    syncRouteBadges() {},
    clearSidebarHighlight() {},
  };

  const context = vm.createContext({
    Promise,
    console: { warn() {} },
    spatialViewMode: '2d',
    timelineAnimating: false,
    teardownTimelineMode() { context.timelineAnimating = false; },
    window: { RNGraph3D: { isAvailable: () => false } },
    document: {
      createElement: () => ({ src: '', onload: null, onerror: null, parentNode: null }),
      head: { appendChild(s) { appended.push(s.src); } },
    },
    showGraphNotice() {},
    graph3dView: null,
    graphCanvas3d: { hidden: true },
    graph3dHoverNode: null,
    perspectiveStageEl: { hidden: false },
    viewMode2dBtn: button(),
    viewModePerspectiveBtn: button(),
    viewMode3dBtn: button(),
    checkCssPerspectiveEl: { checked: false },
    perspectiveControlsEl: { hidden: true },
    graphHint2dEl: { hidden: false },
    graphHintPerspectiveEl: { hidden: true },
    graphHint3dEl: { hidden: true },
    isMobile: false,
    pinnedNode: null,
    sidebarNode: null,
    applyNodeNeighborHighlight() {},
    SPATIAL_VIEW_KEY: 'rn-graph-spatial-view',
    localStorage: { setItem: (k, v) => storage.set(k, v) },
    svg: { style: (_, v) => { svgDisplay = v; } },
    edgeCanvas: { style: { display: 'block' } },
    pauseSimulation2D: () => { paused += 1; },
    resumeSimulation2D: () => { resumed += 1; },
    hideTooltip() {},
    syncGraphDomFromSimulation() {},
    fitToScreen() {},
    sync2DNodeLabels() {},
    syncTimeline3DState() {},
    syncMagneticCheckbox() {},
    updateForces: () => { forceUpdates += 1; },
    ensureGraph3DView: () => { context.graph3dView = view3d; },
    syncPerspectivePresentation: (on) => { perspectiveCalls.push(!!on); return true; },
  });

  vm.runInContext(graphHtml.slice(controlsStart, controlsEnd), context);
  vm.runInContext(graphHtml.slice(lazyStart, transitionStart), context);
  vm.runInContext(graphHtml.slice(transitionStart, transitionEnd), context);

  // 1) 2D -> CSS perspective: no 3D bundle download, no force restart, perspective enabled
  context.requestSpatialViewMode('perspective');
  assert.equal(context.spatialViewMode, 'perspective');
  assert.deepEqual(appended, []);
  assert.deepEqual(perspectiveCalls, [true]);
  assert.equal(context.viewModePerspectiveBtn.attributes['aria-pressed'], 'true');
  assert.equal(context.viewMode2dBtn.attributes['aria-pressed'], 'false');
  assert.equal(context.viewMode3dBtn.attributes['aria-pressed'], 'false');
  assert.equal(context.checkCssPerspectiveEl.checked, true);
  assert.equal(context.perspectiveControlsEl.hidden, false);
  assert.equal(context.graphHint2dEl.hidden, true);
  assert.equal(context.graphHintPerspectiveEl.hidden, false);
  assert.equal(context.graphHint3dEl.hidden, true);
  assert.equal(storage.get('rn-graph-spatial-view'), 'perspective');
  assert.equal(paused, 0);
  assert.equal(resumed, 0);
  assert.equal(forceUpdates, 0);
  assert.equal(svgDisplay, 'block');

  // 2) CSS perspective -> 3D (available): disables CSS perspective, hides stage & SVG
  context.window.RNGraph3D.isAvailable = () => true;
  context.requestSpatialViewMode('3d');
  assert.equal(context.spatialViewMode, '3d');
  assert.deepEqual(perspectiveCalls, [true, false]);
  assert.equal(context.perspectiveStageEl.hidden, true);
  assert.equal(svgDisplay, 'none');
  assert.equal(context.edgeCanvas.style.display, 'none');
  assert.equal(context.graphCanvas3d.hidden, false);
  assert.equal(context.checkCssPerspectiveEl.checked, false);
  assert.equal(paused, 1);

  // 3) 3D -> CSS perspective: restores stage/SVG & enables CSS perspective
  context.requestSpatialViewMode('perspective');
  assert.equal(context.spatialViewMode, 'perspective');
  assert.deepEqual(perspectiveCalls, [true, false, true]);
  assert.equal(context.perspectiveStageEl.hidden, false);
  assert.equal(svgDisplay, 'block');
  assert.equal(context.edgeCanvas.style.display, 'block');
  assert.equal(context.graphCanvas3d.hidden, true);
  assert.equal(resumed, 1);

  // 4) CSS perspective -> 2D: disables CSS perspective cleanly
  context.requestSpatialViewMode('2d');
  assert.equal(context.spatialViewMode, '2d');
  assert.deepEqual(perspectiveCalls, [true, false, true, false]);
  assert.equal(context.viewMode2dBtn.attributes['aria-pressed'], 'true');
  assert.equal(context.checkCssPerspectiveEl.checked, false);
  assert.equal(context.perspectiveControlsEl.hidden, true);
  assert.equal(context.graphHint2dEl.hidden, false);
  assert.equal(context.graphHintPerspectiveEl.hidden, true);
});

test('graph.html DOM layering, CSS @supports fallback, theme token derivation, and SW precache stay intact', () => {
  // Stage wraps edge-canvas + graph-canvas together inside #graph-perspective-plane
  const stageIdx = graphHtml.indexOf('<div id="graph-perspective-stage">');
  const planeIdx = graphHtml.indexOf('<div id="graph-perspective-plane">', stageIdx);
  const edgeIdx = graphHtml.indexOf('<canvas id="graph-edge-canvas"', planeIdx);
  const svgIdx = graphHtml.indexOf('<svg id="graph-canvas"></svg>', edgeIdx);
  const canvas3dIdx = graphHtml.indexOf('<div id="graph-canvas-3d"', svgIdx);
  const tooltipIdx = graphHtml.indexOf('<div id="graph-tooltip"', canvas3dIdx);
  const physicsIdx = graphHtml.indexOf('<div id="physics-panel"', tooltipIdx);
  const filterIdx = graphHtml.indexOf('<div id="filter-panel"', physicsIdx);
  assert.ok(stageIdx > 0 && planeIdx > stageIdx && edgeIdx > planeIdx && svgIdx > edgeIdx);
  assert.ok(canvas3dIdx > svgIdx && tooltipIdx > canvas3dIdx && physicsIdx > tooltipIdx && filterIdx > physicsIdx);

  // Check stage closes before #graph-canvas-3d (so fixed overlays are outside transformed layers)
  const stageSlice = graphHtml.slice(stageIdx, canvas3dIdx);
  assert.ok(!stageSlice.includes('id="graph-tooltip"'));
  assert.ok(!stageSlice.includes('id="physics-panel"'));
  assert.ok(!stageSlice.includes('id="filter-panel"'));

  // Default spatial mode stays '2d'
  assert.match(graphHtml, /return saved === '3d' \|\| saved === 'perspective' \? saved : '2d'|if \(saved === '3d' \|\| saved === 'perspective'\) return saved;\s+return '2d'/);

  // @supports + prefers-reduced-motion + theme tokens derived from style.css
  assert.ok(graphHtml.includes('@supports (perspective: 1px) and (transform-style: preserve-3d)'));
  assert.ok(graphHtml.includes('@media (prefers-reduced-motion: reduce)'));
  assert.ok(graphHtml.includes('--persp-stage-bg:'));
  assert.ok(graphHtml.includes('var(--accent)'));
  assert.ok(graphHtml.includes('var(--bg)'));
  assert.ok(graphHtml.includes('<script src="graph-perspective.js"></script>'));

  // Service Worker precache list includes graph-perspective.js without altering CACHE_NAME
  assert.ok(swSource.includes("'graph-perspective.js'"));
});
