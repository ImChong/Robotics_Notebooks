const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');
const { JSDOM } = require('jsdom');

const perspectiveApi = require('../docs/graph-perspective.js');
const graphHtml = readFileSync(resolve(__dirname, '../docs/graph.html'), 'utf8');
const styleCss = readFileSync(resolve(__dirname, '../docs/style.css'), 'utf8');
const indexHtml = readFileSync(resolve(__dirname, '../docs/index.html'), 'utf8');
const miniGraphJs = readFileSync(resolve(__dirname, '../docs/mini-graph.js'), 'utf8');
const swSource = readFileSync(resolve(__dirname, '../docs/sw.js'), 'utf8');

function button() {
  return {
    attributes: {},
    classList: { toggle() {}, add() {}, remove() {} },
    setAttribute(name, value) { this.attributes[name] = value; },
  };
}

function makeMockCanvasEnv() {
  const stats = { drawCalls: 0, programsCreated: 0, buffersUploaded: 0, labelFillCalls: 0 };
  const glProto = {
    VERTEX_SHADER: 0x8b31,
    FRAGMENT_SHADER: 0x8b30,
    COMPILE_STATUS: 0x8b81,
    LINK_STATUS: 0x8b82,
    ARRAY_BUFFER: 0x8892,
    ELEMENT_ARRAY_BUFFER: 0x8893,
    STATIC_DRAW: 0x88e4,
    FLOAT: 0x1406,
    UNSIGNED_SHORT: 0x1403,
    TRIANGLES: 0x0004,
    LINES: 0x0001,
    POINTS: 0x0000,
    DEPTH_TEST: 0x0b71,
    BLEND: 0x0be2,
    SRC_ALPHA: 0x0302,
    ONE_MINUS_SRC_ALPHA: 0x0303,
    COLOR_BUFFER_BIT: 0x4000,
    DEPTH_BUFFER_BIT: 0x0100,
    LESS: 0x0201,
    LEQUAL: 0x0203,
    CULL_FACE: 0x0b44,
    BACK: 0x0405,
    getExtension() { return null; },
    getParameter() { return 0; },
    createShader() { return {}; },
    shaderSource() {},
    compileShader() {},
    getShaderParameter() { return true; },
    getShaderInfoLog() { return ''; },
    deleteShader() {},
    createProgram() { stats.programsCreated += 1; return {}; },
    attachShader() {},
    linkProgram() {},
    getProgramParameter() { return true; },
    getProgramInfoLog() { return ''; },
    deleteProgram() {},
    useProgram() {},
    createBuffer() { return {}; },
    bindBuffer() {},
    bufferData() { stats.buffersUploaded += 1; },
    deleteBuffer() {},
    getAttribLocation(_p, name) {
      if (name === 'aXZ' || name === 'aPos' || name === 'aCenter') return 0;
      if (name === 'aH' || name === 'aQuad' || name === 'aSize' || name === 'aCol') return 1;
      if (name === 'aGrad' || name === 'aInstColor') return 2;
      return 3;
    },
    getUniformLocation() { return {}; },
    enableVertexAttribArray() {},
    disableVertexAttribArray() {},
    vertexAttribPointer() {},
    uniform1f() {},
    uniform1i() {},
    uniform2f() {},
    uniform3f() {},
    uniform3fv() {},
    uniform4f() {},
    uniformMatrix4fv() {},
    viewport() {},
    clearColor() {},
    clear() {},
    enable() {},
    disable() {},
    blendFunc() {},
    blendFuncSeparate() {},
    blendEquation() {},
    depthMask() {},
    depthFunc() {},
    cullFace() {},
    drawArrays() { stats.drawCalls += 1; },
    drawElements() { stats.drawCalls += 1; },
  };
  const gl = new Proxy(glProto, {
    get(target, prop) {
      if (prop in target) return target[prop];
      return () => {};
    },
  });
  const ctx2d = new Proxy({
    setTransform() {},
    clearRect() {},
    measureText(t) { return { width: String(t || '').length * 6 }; },
    fillText() { stats.labelFillCalls += 1; },
    roundRect() {},
  }, {
    get(target, prop) {
      if (prop in target) return target[prop];
      return () => {};
    },
  });
  return { gl, ctx2d, stats };
}

test('tier defaults and clamping preserve terrain bounds and recover from malformed values', () => {
  assert.deepEqual(perspectiveApi.getDefaultsForTier('full'), {
    height: 100,
    contours: 16,
    pitch: 36,
    yaw: -18,
  });
  assert.deepEqual(perspectiveApi.getDefaultsForTier('lite'), {
    height: 88,
    contours: 12,
    pitch: 32,
    yaw: -12,
  });
  assert.deepEqual(perspectiveApi.getDefaultsForTier('reduced'), {
    height: 86,
    contours: 14,
    pitch: 38,
    yaw: -14,
  });

  const clampedFull = perspectiveApi.clampParams({
    height: 999,
    contours: 80,
    pitch: 99,
    yaw: -360,
  }, 'full');
  assert.deepEqual(clampedFull, {
    height: 160,
    contours: 28,
    pitch: 78,
    yaw: -180,
  });

  const clampedLite = perspectiveApi.clampParams({
    height: '150',
    contours: 3,
    pitch: '5',
    yaw: '140',
  }, 'lite');
  assert.deepEqual(clampedLite, {
    height: 128,
    contours: 6,
    pitch: 14,
    yaw: 140,
  });

  const recovered = perspectiveApi.clampParams({
    height: NaN,
    contours: null,
    pitch: 'bad',
    yaw: Infinity,
  }, 'full');
  assert.deepEqual(recovered, perspectiveApi.getDefaultsForTier('full'));
});

test('resolveCapabilityTier handles unsupported WebGL, reduced motion, lite mobile/low-end, and full desktop', () => {
  assert.equal(perspectiveApi.resolveCapabilityTier({
    supportsWebGL: false,
    reducedMotion: true,
    coarsePointer: true,
  }), 'unsupported');

  assert.equal(perspectiveApi.resolveCapabilityTier({
    supportsWebGL: true,
    reducedMotion: true,
    coarsePointer: false,
    hardwareConcurrency: 16,
    deviceMemory: 32,
  }), 'reduced');

  assert.equal(perspectiveApi.resolveCapabilityTier({
    supportsWebGL: true,
    reducedMotion: false,
    coarsePointer: true,
    hardwareConcurrency: 8,
    deviceMemory: 8,
  }), 'lite');

  assert.equal(perspectiveApi.resolveCapabilityTier({
    supportsWebGL: true,
    reducedMotion: false,
    coarsePointer: false,
    hardwareConcurrency: 4,
    deviceMemory: 8,
  }), 'lite');

  assert.equal(perspectiveApi.resolveCapabilityTier({
    supportsWebGL: true,
    reducedMotion: false,
    coarsePointer: false,
    hardwareConcurrency: 8,
    deviceMemory: 4,
  }), 'lite');

  assert.equal(perspectiveApi.resolveCapabilityTier({
    supportsWebGL: true,
    reducedMotion: false,
    coarsePointer: false,
    hardwareConcurrency: 8,
    deviceMemory: 8,
    dpr: 3,
    nodeCount: 5000,
  }), 'lite');

  assert.equal(perspectiveApi.resolveCapabilityTier({
    supportsWebGL: true,
    reducedMotion: false,
    coarsePointer: false,
    hardwareConcurrency: 8,
    deviceMemory: 8,
    dpr: 2,
    nodeCount: 8200,
  }), 'full');
});

test('computeNodeDegrees and computeTerrainField elevate high-degree hubs and dissolve perimeter boundaries', () => {
  const nodes = [
    { id: 'hub', x: 100, y: 100 },
    { id: 'n1', x: 120, y: 110 },
    { id: 'n2', x: 80, y: 110 },
    { id: 'n3', x: 110, y: 80 },
    { id: 'n4', x: 90, y: 80 },
    { id: 'leaf', x: 280, y: 260 },
  ];
  const edges = [
    { source: 'hub', target: 'n1' },
    { source: 'hub', target: 'n2' },
    { source: 'hub', target: 'n3' },
    { source: 'hub', target: 'n4' },
    { source: 'n1', target: 'leaf' },
  ];

  const degreeInfo = perspectiveApi.computeNodeDegrees(nodes, edges);
  assert.equal(degreeInfo.counts.hub, 4);
  assert.equal(degreeInfo.counts.leaf, 1);
  assert.equal(degreeInfo.maxDegree, 4);

  const field = perspectiveApi.computeTerrainField(nodes, degreeInfo, { gridSize: 48, extent: 1.35 });
  const hubMeta = field.nodeMeta.hub;
  const leafMeta = field.nodeMeta.leaf;
  assert.ok(hubMeta && leafMeta);
  assert.ok(hubMeta.degNorm > leafMeta.degNorm);
  assert.ok(hubMeta.surfaceH > leafMeta.surfaceH, 'high-degree hub must produce higher terrain peak than low-degree leaf');
  assert.ok(hubMeta.elevH >= hubMeta.surfaceH, 'floating marker sits above terrain surface on vertical stem');

  const sampledHub = perspectiveApi.sampleTerrainHeight(field, hubMeta.wx, hubMeta.wz);
  assert.ok(Math.abs(sampledHub - hubMeta.surfaceH) < 0.08);

  // Corner/border vertices must attenuate smoothly toward 0 (borderless mist dissolution, no box ridge)
  const cornerH = field.heights[0];
  assert.ok(cornerH <= 0.05, `expected near-zero perimeter terrain height, got ${cornerH}`);
});

test('camera projection places world origin inside viewport across oblique and top-down terrain views', () => {
  const oblique = perspectiveApi.buildCameraMatrices({ pitch: 36, yaw: -18 }, { distance: 1.72 }, 16 / 9);
  const topDown = perspectiveApi.buildCameraMatrices(perspectiveApi.TOP_VIEW_PARAMS, { distance: 1.72 }, 16 / 9);
  assert.ok(topDown.eye[1] > oblique.eye[1], 'top-down pitch places camera higher along +Y');

  const pCenter = perspectiveApi.projectWorldToScreen(0, 0.2, 0, oblique.viewProj, 1280, 720);
  assert.ok(pCenter);
  assert.ok(pCenter[0] > 400 && pCenter[0] < 880);
  assert.ok(pCenter[1] > 100 && pCenter[1] < 500);

  // Screen ray -> horizontal plane unprojection & world->graph coordinate round-trip
  const worldSample = [0.32, 0.24, -0.21];
  const scr = perspectiveApi.projectWorldToScreen(worldSample[0], worldSample[1], worldSample[2], oblique.viewProj, 1280, 720);
  const unproj = perspectiveApi.unprojectScreenToPlaneY(scr[0], scr[1], worldSample[1], oblique, 1280, 720);
  assert.ok(unproj, 'unprojection onto node horizontal plane succeeds');
  assert.ok(Math.abs(unproj[0] - worldSample[0]) < 1e-4);
  assert.ok(Math.abs(unproj[2] - worldSample[2]) < 1e-4);

  const center = { x: 400, y: 240, halfSpan: 180, coreSpan: 0.78 };
  const gPt = perspectiveApi.worldToGraphCoords(0.39, -0.195, center);
  assert.ok(Math.abs(gPt.x - 490) < 1e-3);
  assert.ok(Math.abs(gPt.y - 195) < 1e-3);

  // Far outer nodes map linearly without rectangular box compression and round-trip cleanly
  const spreadNodes = [];
  for (let i = 0; i < 40; i++) {
    const a = (i / 40) * Math.PI * 2;
    spreadNodes.push({ id: `c${i}`, x: Math.cos(a) * 100, y: Math.sin(a) * 100 });
  }
  spreadNodes.push({ id: 'outer1', x: 260, y: 30 }, { id: 'outer2', x: 420, y: 30 });
  const spreadField = perspectiveApi.computeTerrainField(spreadNodes, perspectiveApi.computeNodeDegrees(spreadNodes, []), { gridSize: 48 });
  const m1 = spreadField.nodeMeta.outer1;
  const m2 = spreadField.nodeMeta.outer2;
  assert.ok(m2.wx > m1.wx * 1.45, 'far outer nodes spread linearly outward without being squished against a rectangular wall');
  const rtOuter = perspectiveApi.worldToGraphCoords(m2.wx, m2.wz, spreadField.center);
  assert.ok(Math.abs(rtOuter.x - 420) < 1e-3 && Math.abs(rtOuter.y - 30) < 1e-3, 'far outer node round-trips accurately');

  // 3D terrain sphere overlap solver separates clustered spheres around hubs monotonically without oscillation
  const overlapItems = [
    { node: { id: 'hub' }, wx: 0, wz: 0, wySurf: 0.1, wy: 0.2, degNorm: 1, baseRadius: 0.028 },
    { node: { id: 'a' }, wx: 0.002, wz: 0.001, wySurf: 0.1, wy: 0.15, degNorm: 0.2, baseRadius: 0.006 },
    { node: { id: 'b' }, wx: -0.001, wz: 0.002, wySurf: 0.1, wy: 0.15, degNorm: 0.2, baseRadius: 0.006 },
    { node: { id: 'c' }, wx: 0, wz: 0, wySurf: 0.1, wy: 0.15, degNorm: 0.2, baseRadius: 0.006 },
  ];
  perspectiveApi.resolveSphereOverlaps3D(overlapItems, spreadField, 0.35, { gap: 0.002, maxPasses: 14 });
  for (let i = 0; i < overlapItems.length; i++) {
    for (let j = i + 1; j < overlapItems.length; j++) {
      const d = Math.hypot(overlapItems[i].wx - overlapItems[j].wx, overlapItems[i].wz - overlapItems[j].wz);
      assert.ok(d >= overlapItems[i].baseRadius + overlapItems[j].baseRadius + 0.0015, `expected spheres ${i}/${j} separated, got ${d}`);
    }
  }
  // Dense 25-node crowded cluster must converge monotonically (step magnitude strictly decays, zero self-oscillation)
  const denseCluster = [];
  for (let i = 0; i < 25; i++) {
    denseCluster.push({
      node: { id: `d${i}` },
      wx: ((i % 5) - 2) * 0.002,
      wz: (Math.floor(i / 5) - 2) * 0.002,
      wySurf: 0.1,
      wy: 0.15,
      degNorm: 0.2,
      baseRadius: 0.004,
    });
  }
  const passMoves = [];
  for (let p = 0; p < 8; p++) {
    const prev = denseCluster.map((it) => [it.wx, it.wz]);
    perspectiveApi.resolveSphereOverlaps3D(denseCluster, null, 0, { gap: 0.0008, maxPasses: 1 });
    let maxMv = 0;
    for (let i = 0; i < denseCluster.length; i++) {
      const mv = Math.hypot(denseCluster[i].wx - prev[i][0], denseCluster[i].wz - prev[i][1]);
      if (mv > maxMv) maxMv = mv;
    }
    passMoves.push(maxMv);
  }
  assert.ok(passMoves[passMoves.length - 1] < passMoves[0] * 0.35, `dense cluster must damp monotonically without oscillating: ${passMoves.map((v) => v.toFixed(5)).join(', ')}`);

  // Entry force-simulation bloom starts gathered inward, expands organically, and settles to exact equilibrium at 1.0
  const startPos = perspectiveApi.evalForceMotionXZ(0.6, 0.3, 0.0, 0.026);
  const midPos = perspectiveApi.evalForceMotionXZ(0.6, 0.3, 0.55, 0.026);
  const endPos = perspectiveApi.evalForceMotionXZ(0.6, 0.3, 1.0, 0.026);
  assert.ok(Math.hypot(startPos[0], startPos[1]) < Math.hypot(0.6, 0.3) * 0.6, 'force bloom starts gathered inward');
  assert.ok(Math.hypot(midPos[0], midPos[1]) > Math.hypot(startPos[0], startPos[1]), 'force bloom expands outward');
  assert.deepEqual(endPos, [0.6, 0.3], 'force bloom settles to exact original coordinates at settle=1');
});

test('graph.html supports 3-way mutual exclusion across 2D, 3D Terrain, and WebGL 3D without lazy-loading 3D bundle', () => {
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
    labelsSettled2D: true,
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
    perspectiveStageEl: { hidden: true },
    viewMode2dBtn: button(),
    viewModePerspectiveBtn: button(),
    viewMode3dBtn: button(),
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

  // 1) 2D -> 3D 地形 (perspective): no 3D-force-graph download, pauses 2D ticks, reveals WebGL terrain stage
  context.requestSpatialViewMode('perspective');
  assert.equal(context.spatialViewMode, 'perspective');
  assert.deepEqual(appended, []);
  assert.deepEqual(perspectiveCalls, [true]);
  assert.equal(context.perspectiveStageEl.hidden, false);
  assert.equal(context.viewModePerspectiveBtn.attributes['aria-pressed'], 'true');
  assert.equal(context.viewMode2dBtn.attributes['aria-pressed'], 'false');
  assert.equal(context.viewMode3dBtn.attributes['aria-pressed'], 'false');
  assert.equal(context.perspectiveControlsEl.hidden, false);
  assert.equal(context.graphHint2dEl.hidden, true);
  assert.equal(context.graphHintPerspectiveEl.hidden, false);
  assert.equal(context.graphHint3dEl.hidden, true);
  assert.equal(storage.get('rn-graph-spatial-view'), 'perspective');
  assert.equal(paused, 1);
  assert.equal(resumed, 0);
  assert.equal(forceUpdates, 0);

  // 2) 3D 地形 -> 3D 立体: disables terrain stage & hides 2D layers
  context.window.RNGraph3D.isAvailable = () => true;
  context.requestSpatialViewMode('3d');
  assert.equal(context.spatialViewMode, '3d');
  assert.deepEqual(perspectiveCalls, [true, false]);
  assert.equal(context.perspectiveStageEl.hidden, true);
  assert.equal(svgDisplay, 'none');
  assert.equal(context.edgeCanvas.style.display, 'none');
  assert.equal(context.graphCanvas3d.hidden, false);
  assert.equal(context.perspectiveControlsEl.hidden, true);

  // 3) 3D 立体 -> 2D: cleanly returns to 2D
  context.requestSpatialViewMode('2d');
  assert.equal(context.spatialViewMode, '2d');
  assert.equal(context.graphCanvas3d.hidden, true);
  assert.equal(context.perspectiveStageEl.hidden, true);
  assert.equal(svgDisplay, 'block');

  // 4) Unsupported WebGL terrain restores 2D gracefully
  context.syncPerspectivePresentation = () => false;
  context.requestSpatialViewMode('perspective');
  assert.equal(context.spatialViewMode, '2d');
  assert.equal(context.perspectiveStageEl.hidden, true);
  assert.equal(storage.get('rn-graph-spatial-view'), '2d');
});

test('graph.html and style.css define borderless WebGL terrain stage, light/dark horizon tokens, and terrain controls', () => {
  assert.ok(!graphHtml.includes('graph-perspective-frame'), 'must not reintroduce rectangular floor border box');
  assert.ok(!styleCss.includes('.graph-perspective-frame'), 'must not style a rectangular frame border in 3D view');
  assert.ok(!graphHtml.includes('id="check-css-perspective"'), 'redundant 3D terrain checkbox removed in favor of view mode buttons');

  // Sphere sizing follows 3D force-graph sqrt(degree) hierarchy across preview and full graph
  const rPreviewLow = perspectiveApi.computeSphereRadius3D(10, 3000, true);
  const rPreviewMid = perspectiveApi.computeSphereRadius3D(500, 3000, true);
  const rPreviewHigh = perspectiveApi.computeSphereRadius3D(3000, 3000, true);
  assert.ok(rPreviewHigh > rPreviewMid && rPreviewMid > rPreviewLow, 'preview sphere radius scales strictly with connection count');
  assert.ok(rPreviewHigh / rPreviewLow > 3.5, 'preview sphere size range matches 3D force-graph 4.5x scale');

  const rFullLow = perspectiveApi.computeSphereRadius3D(1, 3000, false);
  const rFullMid = perspectiveApi.computeSphereRadius3D(300, 3000, false);
  const rFullHigh = perspectiveApi.computeSphereRadius3D(3000, 3000, false);
  assert.ok(rFullHigh > rFullMid && rFullMid > rFullLow, 'full-graph sphere radius scales strictly with connection count');
  assert.ok(rFullHigh / rFullLow > 6.5, 'full-graph sphere size hierarchy matches RNGraphNodeSize');

  assert.ok(graphHtml.includes('<div id="graph-perspective-stage" class="graph-perspective-stage"'));
  assert.ok(graphHtml.includes('id="sl-persp-height"'));
  assert.ok(graphHtml.includes('id="sl-persp-contours"'));
  assert.ok(graphHtml.includes('id="sl-persp-pitch"'));
  assert.ok(graphHtml.includes('id="sl-persp-yaw"'));
  assert.ok(graphHtml.includes('#graph-wrap.is-perspective-mode #graph-canvas'));
  assert.ok(graphHtml.includes('<script src="graph-perspective.js"></script>'));

  assert.ok(styleCss.includes('--persp-stage-bg:'));
  assert.ok(styleCss.includes('--persp-horizon-edge: var(--bg);'));
  assert.ok(styleCss.includes('--persp-horizon-edge: var(--bg-alt);'));
  assert.ok(styleCss.includes('.graph-perspective-stage::before'), 'horizon atmospheric reflection layer styled');
  assert.ok(styleCss.includes('--persp-horizon-glow'), 'horizon glow fades out cleanly in top-down pitch');
  assert.ok(styleCss.includes('.graph-perspective-stage::after'));
  assert.ok(styleCss.includes('#mini-graph-svg line.mini-edge-hover'), 'homepage preview 2D hover edge highlight styled');
  assert.ok(styleCss.includes('.mini-graph-node.mini-node-hover-dim'), 'homepage preview 2D hover dim style present');
  assert.ok(styleCss.includes('#mini-graph-wrap.is-perspective-mode'));
  assert.ok(styleCss.includes('border: 1px solid var(--border);'), 'homepage 3D preview retains visible viewport border');
  const perspectiveSrc = readFileSync(resolve(__dirname, '../docs/graph-perspective.js'), 'utf8');
  assert.ok(!perspectiveSrc.includes('buildPolyhedronLibrary'), '3D terrain nodes must be round spheres, not faceted polyhedra');
  assert.ok(perspectiveSrc.includes('uCamRight') && perspectiveSrc.includes('if (r2 > 1.0) discard;'), 'hubs render as analytical round spheres');
  assert.ok(perspectiveSrc.includes('uPointScale') && perspectiveSrc.includes('if (pn.hovered) radius *= 1.10;'), 'point sprites share unified world radius with subtle 1.10x hover swell');
  assert.ok(perspectiveSrc.includes('uForceSettle') && perspectiveSrc.includes('applyForceMotion'), '3D terrain entry includes GPU force-simulation bloom motion');
  assert.ok(perspectiveSrc.includes('vMeshRadial') && perspectiveSrc.includes('riseVis'), 'top-down entry fades terrain edge by unscaled mesh radial with zero bounded square grid floor');
  assert.ok(perspectiveSrc.includes('stepReboundAnim') && perspectiveSrc.includes('rebounding: !!reboundAnim'), '3D node drag includes spring-force rebound simulation');
  assert.ok(graphHtml.includes('perspectiveSimActive') && graphHtml.includes('reseed: true'), '3D terrain layout refresh randomizes nodes and restarts force simulation');
  assert.ok(graphHtml.includes('getRouteBadges:') && perspectiveSrc.includes('function focusNode('), '3D terrain supports route step-by-step focus and numbered route badges');
  assert.ok(graphHtml.includes('--depth-banner-chat-shift') && graphHtml.includes('syncDepthBannerChatOffset'), 'route stepper banner shifts cleanly to avoid open chat panel');
  assert.ok(graphHtml.includes('function resolveNodeOverlaps2D('), 'graph.html includes 2D spatial-hash non-overlap solver');
  const graph3dJs = readFileSync(resolve(__dirname, '../docs/graph-3d.js'), 'utf8');
  assert.ok(graph3dJs.includes('function create3DCollisionForce(') && graph3dJs.includes('function resolveNodeOverlaps3D('), 'graph-3d.js includes 3D sphere collision force and settle overlap solver');
  assert.ok(miniGraphJs.includes('d3.drag()') && miniGraphJs.includes('nodeDrag2d'), 'homepage mini-graph preview binds 2D node mouse dragging');
  assert.ok(graphHtml.includes('getCommunityLabels:') && perspectiveSrc.includes('communityLabelCount'), '3D terrain renders floating community cluster capsules');
  assert.ok(perspectiveSrc.includes('uHeightScale') && perspectiveSrc.includes('uGlintPhase') && perspectiveSrc.includes('startCameraTransition'), 'top-down to oblique entry swoop and horizon glint present');
  assert.ok(styleCss.includes('[data-theme="light"]'));
  assert.ok(styleCss.includes('var(--accent)'));
  assert.ok(swSource.includes("'graph-perspective.js'"));
});

test('createController isolates persist:false defaults and accepts degree-terrain parameters', () => {
  const prevStorage = global.localStorage;
  const store = new Map([[perspectiveApi.STORAGE_KEY, JSON.stringify({ height: 40, contours: 8, pitch: 76, yaw: 0 })]]);
  global.localStorage = {
    getItem(k) { return store.has(k) ? store.get(k) : null; },
    setItem(k, v) { store.set(k, String(v)); },
  };
  try {
    const ctrl = perspectiveApi.createController({
      persist: false,
      defaultParamsByTier: {
        lite: { height: 96, contours: 12, pitch: 32, yaw: -12 },
        unsupported: { height: 112, contours: 14, pitch: 35, yaw: -16 },
      },
      nodeCount: 50,
    });
    assert.deepEqual(ctrl.getParams(), { height: 112, contours: 14, pitch: 35, yaw: -16 });
    ctrl.setParam('height', 140);
    ctrl.resetDefaults();
    assert.deepEqual(ctrl.getParams(), { height: 112, contours: 14, pitch: 35, yaw: -16 });
    // Graph page stored params untouched
    assert.equal(JSON.parse(store.get(perspectiveApi.STORAGE_KEY)).pitch, 76);
    ctrl.destroy();
  } finally {
    global.localStorage = prevStorage;
  }
});

test('homepage mini-graph preview renders WebGL 3D terrain on 3D toggle with search highlight linkage', async () => {
  assert.ok(!indexHtml.includes('id="mini-graph-3d"'), 'obsolete WebGL #mini-graph-3d container removed');
  assert.ok(!miniGraphJs.includes('3d-force-graph'), 'mini-graph.js must not reference 3d-force-graph');
  assert.ok(!miniGraphJs.includes('ForceGraph3D'), 'mini-graph.js must not invoke ForceGraph3D');
  assert.ok(indexHtml.includes('<script src="graph-perspective.js" defer></script>'));

  const dom = new JSDOM(indexHtml, { url: 'https://example.test/', runScripts: 'outside-only' });
  const w = dom.window;
  const mock = makeMockCanvasEnv();
  const origGetContext = w.HTMLCanvasElement.prototype.getContext;
  w.HTMLCanvasElement.prototype.getContext = function (type) {
    if (type === 'webgl2' || type === 'webgl' || type === 'experimental-webgl') return mock.gl;
    if (type === '2d') return mock.ctx2d;
    return origGetContext ? origGetContext.call(this, type) : null;
  };
  w.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });

  const syntheticGraph = {
    communities: [{ id: 'c-rl', label: '强化学习' }, { id: 'c-ctrl', label: '运动控制' }],
    nodes: [
      { id: 'n1', label: 'PPO', type: 'method', community: 'c-rl', summary: 'Policy gradient' },
      { id: 'n2', label: 'MPC', type: 'method', community: 'c-ctrl', summary: 'Model predictive control' },
      { id: 'n3', label: 'Sim2Real', type: 'concept', community: 'c-rl', summary: 'Transfer' },
    ],
    edges: [
      { source: 'n1', target: 'n2' },
      { source: 'n1', target: 'n3' },
    ],
  };
  w.RNLinkGraph = { ensure: () => Promise.resolve(syntheticGraph) };
  w.RNGraphTooltip = {
    getCommunityColors: () => ['#5b9cf6', '#34d399'],
    buildCommunityColorMap: () => ({ 'c-rl': '#5b9cf6', 'c-ctrl': '#34d399' }),
    formatTooltipSummary: s => s || '',
    buildNodeTooltipHtml: o => '<div>' + o.title + '</div>',
    bindBlankDismiss() {},
    bindOutsideDismiss() {},
  };

  w.eval(readFileSync(resolve(__dirname, '../docs/vendor/d3.min.js'), 'utf8'));
  w.eval(readFileSync(resolve(__dirname, '../docs/graph-node-size.js'), 'utf8'));
  w.eval(readFileSync(resolve(__dirname, '../docs/graph-perspective.js'), 'utf8'));
  w.eval(miniGraphJs);

  await new Promise((r) => setTimeout(r, 80));

  const wrap = w.document.getElementById('mini-graph-wrap');
  const stage = w.document.getElementById('mini-graph-perspective-stage');
  const svg = w.document.getElementById('mini-graph-svg');
  const btn2d = w.document.getElementById('miniGraphMode2d');
  const btn3d = w.document.getElementById('miniGraphMode3d');
  const expand = w.document.getElementById('mini-graph-expand');

  assert.ok(!stage.contains(svg), 'mini-graph SVG sits cleanly alongside WebGL terrain stage');
  assert.equal(stage.hidden, true);
  assert.equal(w.RNMiniGraph.getMode(), '2d');
  assert.equal(wrap.classList.contains('is-perspective-mode'), false);
  assert.equal(btn2d.getAttribute('aria-pressed'), 'true');
  assert.equal(btn3d.getAttribute('aria-pressed'), 'false');
  assert.equal(expand.getAttribute('href'), 'graph.html');

  // 2D hover highlights connected edges + neighbor nodes and dims unrelated ones, then restores cleanly
  const nodeGs = Array.from(w.document.querySelectorAll('#mini-graph-svg .mini-graph-node'));
  const mpcNodeG = nodeGs.find((el) => el.textContent.includes('MPC'));
  assert.ok(mpcNodeG, 'found MPC node in 2D mini-graph');
  mpcNodeG.dispatchEvent(new w.MouseEvent('mouseenter', { bubbles: false, clientX: 200, clientY: 200 }));
  const hoverEdges = w.document.querySelectorAll('#mini-graph-svg line.mini-edge-hover');
  const dimEdges = w.document.querySelectorAll('#mini-graph-svg line.mini-edge-hover-dim');
  assert.equal(hoverEdges.length, 1, 'MPC has 1 connected edge highlighted on hover');
  assert.equal(dimEdges.length, 1, 'unconnected edge dimmed on hover');
  assert.equal(hoverEdges[0].getAttribute('stroke'), '#34d399', 'highlighted edge adopts hovered node color');
  assert.equal(hoverEdges[0].getAttribute('stroke-width'), '2');
  assert.equal(w.document.querySelectorAll('#mini-graph-svg .mini-graph-node.mini-node-hover').length, 1);
  assert.equal(w.document.querySelectorAll('#mini-graph-svg .mini-graph-node.mini-node-hover-neighbor').length, 1);
  assert.equal(w.document.querySelectorAll('#mini-graph-svg .mini-graph-node.mini-node-hover-dim').length, 1);

  mpcNodeG.dispatchEvent(new w.MouseEvent('mouseleave', { bubbles: false }));
  assert.equal(w.document.querySelectorAll('#mini-graph-svg line.mini-edge-hover').length, 0);
  assert.equal(w.document.querySelectorAll('#mini-graph-svg line.mini-edge-hover-dim').length, 0);
  assert.equal(w.document.querySelectorAll('#mini-graph-svg .mini-graph-node.mini-node-hover-dim').length, 0);

  // Switch preview 3D on -> WebGL terrain draws heightfield + contour lines + spheres + labels
  btn3d.click();
  await new Promise((r) => setTimeout(r, 40));
  assert.equal(w.RNMiniGraph.getMode(), '3d');
  assert.equal(wrap.classList.contains('is-perspective-mode'), true);
  assert.equal(stage.hidden, false);
  assert.ok(mock.stats.programsCreated >= 4, 'compiled WebGL terrain/edge/poly/point programs');
  assert.ok(mock.stats.drawCalls >= 2, 'drew WebGL terrain meshes');
  assert.ok(stage.querySelector('.graph-terrain-gl'));
  assert.ok(stage.querySelector('.graph-terrain-labels'));
  assert.equal(btn2d.getAttribute('aria-pressed'), 'false');
  assert.equal(btn3d.getAttribute('aria-pressed'), 'true');
  assert.equal(expand.getAttribute('href'), 'graph.html?view=perspective');

  // 3D hover picks node, highlights connected 3D arcs, and clears cleanly on pointerleave
  stage.getBoundingClientRect = () => ({ left: 0, top: 0, width: 700, height: 480 });
  const targets3d = w.RNMiniGraph.getPerspectiveTargets(5);
  assert.ok(targets3d.length > 0, '3D preview exposes projected screen targets');
  const t0 = targets3d[0];
  stage.dispatchEvent(new w.MouseEvent('pointermove', { bubbles: true, clientX: t0.x, clientY: t0.y }));
  await new Promise((r) => setTimeout(r, 30));
  const hoverState3d = w.RNMiniGraph.getPerspectiveState();
  assert.equal(hoverState3d.hoveredNodeId, t0.id);
  assert.ok(hoverState3d.fieldStats && hoverState3d.fieldStats.hiEdgeCount >= 1, '3D hover builds highlighted connected arcs');
  stage.dispatchEvent(new w.MouseEvent('pointerleave', { bubbles: false }));
  await new Promise((r) => setTimeout(r, 30));
  const leftState3d = w.RNMiniGraph.getPerspectiveState();
  assert.equal(leftState3d.hoveredNodeId, null);
  assert.equal(leftState3d.fieldStats.hiEdgeCount, 0, 'pointerleave clears 3D highlighted arcs');

  // 3D node drag moves node (x, y) coordinates without orbiting camera, rebounds on release, and syncs back to 2D SVG
  w.requestAnimationFrame = (cb) => w.setTimeout(() => cb(w.performance.now()), 16);
  w.cancelAnimationFrame = (id) => w.clearTimeout(id);
  const draggedNodeEl = Array.from(w.document.querySelectorAll('#mini-graph-svg .mini-graph-node'))
    .find((el) => el.textContent.includes(t0.label.slice(0, 3)));
  const transformBefore = draggedNodeEl ? draggedNodeEl.getAttribute('transform') : '';
  const yawBefore = leftState3d.params.yaw;
  const pitchBefore = leftState3d.params.pitch;
  stage.dispatchEvent(new w.MouseEvent('pointerdown', { bubbles: true, button: 0, clientX: t0.x, clientY: t0.y }));
  stage.dispatchEvent(new w.MouseEvent('pointermove', { bubbles: true, button: 0, clientX: t0.x + 85, clientY: t0.y + 42 }));
  await new Promise((r) => setTimeout(r, 30));
  const midDragState = w.RNMiniGraph.getPerspectiveState();
  assert.equal(midDragState.draggingNodeId, t0.id, 'dragging a 3D node enters node-drag mode');
  assert.equal(midDragState.params.yaw, yawBefore, 'dragging a node must not orbit camera yaw');
  assert.equal(midDragState.params.pitch, pitchBefore, 'dragging a node must not orbit camera pitch');
  stage.dispatchEvent(new w.MouseEvent('pointerup', { bubbles: true, button: 0, clientX: t0.x + 85, clientY: t0.y + 42 }));
  const rightAfterDrop = w.RNMiniGraph.getPerspectiveState();
  assert.equal(rightAfterDrop.draggingNodeId, null);
  assert.equal(rightAfterDrop.rebounding, true, 'releasing dragged 3D node triggers elastic force rebound');
  await new Promise((r) => setTimeout(r, 40));
  const transformAfter = draggedNodeEl ? draggedNodeEl.getAttribute('transform') : '';
  assert.notEqual(transformAfter, transformBefore, 'dragging node in 3D updates its underlying (x, y) synced to 2D');
  delete w.requestAnimationFrame;
  delete w.cancelAnimationFrame;

  // Dragging blank terrain still orbits camera normally
  stage.dispatchEvent(new w.MouseEvent('pointerdown', { bubbles: true, button: 0, clientX: 18, clientY: 18 }));
  stage.dispatchEvent(new w.MouseEvent('pointermove', { bubbles: true, button: 0, clientX: 98, clientY: 54 }));
  stage.dispatchEvent(new w.MouseEvent('pointerup', { bubbles: true, button: 0, clientX: 98, clientY: 54 }));
  const afterOrbit = w.RNMiniGraph.getPerspectiveState();
  assert.notEqual(afterOrbit.params.yaw, yawBefore, 'dragging blank space still orbits camera yaw');

  const drawsBeforeHighlight = mock.stats.drawCalls;
  w.RNMiniGraph.highlight(['n1'], 'PPO');
  await new Promise((r) => setTimeout(r, 40));
  const hitCount = w.document.querySelectorAll('#mini-graph-svg .mini-graph-node.mini-node-hit').length;
  assert.equal(hitCount, 1);
  assert.ok(mock.stats.drawCalls > drawsBeforeHighlight, 'highlight re-renders 3D terrain peak emphasis');

  // Switch back to 2D -> hides 3D terrain stage and restores 2D SVG
  btn2d.click();
  assert.equal(w.RNMiniGraph.getMode(), '2d');
  assert.equal(wrap.classList.contains('is-perspective-mode'), false);
  assert.equal(stage.hidden, true);
  assert.equal(btn2d.getAttribute('aria-pressed'), 'true');
  assert.equal(btn3d.getAttribute('aria-pressed'), 'false');
  assert.equal(expand.getAttribute('href'), 'graph.html');
});
