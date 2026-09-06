const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

function button() {
  return {
    attributes: {}, classList: { toggle() {} },
    setAttribute(name, value) { this.attributes[name] = value; },
  };
}
function harness(failure) {
  const storage = new Map(), alerts = [];
  let resumed = 0, paused = 0, display = 'block';
  const view = {
    show() { if (failure === 'show') throw new Error('WebGL disabled'); },
    hide() { if (failure === 'cleanup') throw new Error('partial renderer'); },
    syncFilters() { if (failure === 'cleanup') throw new Error('filter failure'); },
  };
  const context = vm.createContext({
    console: { warn() {} }, spatialViewMode: '2d', timelineAnimating: false,
    window: { RNGraph3D: { isAvailable: () => failure !== 'library' }, alert: (s) => alerts.push(s) },
    graph3dView: null, graphCanvas3d: { hidden: true }, graph3dHoverNode: null,
    viewMode2dBtn: button(), viewMode3dBtn: button(), graphHint2dEl: {}, graphHint3dEl: {},
    isMobile: false, pinnedNode: null, SPATIAL_VIEW_KEY: 'view',
    localStorage: { setItem: (k, v) => storage.set(k, v) },
    svg: { style: (_, value) => { display = value; } },
    pauseSimulation2D: () => { paused++; }, resumeSimulation2D: () => { resumed++; },
    hideTooltip() {}, syncGraphDomFromSimulation() {}, fitToScreen() {}, sync2DNodeLabels() {},
    syncTimeline3DState() {}, syncMagneticCheckbox() {}, updateForces() {},
    ensureGraph3DView: () => { context.graph3dView = failure === 'create' ? null : view; },
  });
  const source = readFileSync(resolve(__dirname, '../docs/graph.html'), 'utf8');
  const controls = source.indexOf('    function is3DView()');
  const controlsEnd = source.indexOf('    function colorModeClusterKey');
  const transition = source.indexOf('    function setSpatialViewMode(');
  const transitionEnd = source.indexOf('    function syncTimeline3DState(');
  assert.ok(controls >= 0 && controlsEnd > controls && transition >= 0 && transitionEnd > transition);
  // Run production transition and button/hint rendering; mock only external renderer/DOM APIs.
  vm.runInContext(source.slice(controls, controlsEnd), context);
  vm.runInContext(source.slice(transition, transitionEnd), context);
  return { context, storage, alerts, display: () => display, resumed: () => resumed, paused: () => paused };
}

for (const failure of ['library', 'create', 'show', 'cleanup']) {
  test(`3D ${failure} failure restores 2D visibility, controls and persisted state`, () => {
    const h = harness(failure);
    h.context.setSpatialViewMode('3d');
    assert.equal(h.context.spatialViewMode, '2d');
    assert.equal(h.display(), 'block');
    assert.equal(h.context.graphCanvas3d.hidden, true);
    assert.equal(h.context.viewMode2dBtn.attributes['aria-pressed'], 'true');
    assert.equal(h.context.viewMode3dBtn.attributes['aria-pressed'], 'false');
    assert.equal(h.context.graphHint2dEl.hidden, false);
    assert.equal(h.storage.get('view'), '2d');
    assert.equal(h.resumed(), 1);
    assert.match(h.alerts[0], /已恢复 2D/);
    h.context.setSpatialViewMode('3d');
    assert.equal(h.resumed(), 2, 'retry also leaves a usable 2D view');
  });
}

test('successful 3D entry and return to 2D preserve normal behavior', () => {
  const h = harness();
  h.context.setSpatialViewMode('3d');
  assert.equal(h.context.spatialViewMode, '3d');
  assert.equal(h.display(), 'none');
  assert.equal(h.context.viewMode3dBtn.attributes['aria-pressed'], 'true');
  assert.equal(h.storage.get('view'), '3d');
  assert.equal(h.paused(), 1);
  h.context.setSpatialViewMode('2d');
  assert.equal(h.display(), 'block');
  assert.equal(h.storage.get('view'), '2d');
  assert.equal(h.resumed(), 1);
  assert.deepEqual(h.alerts, []);
});

test('failed automatic 3D restore clears the saved 3D preference', () => {
  const h = harness('show');
  h.context.spatialViewMode = '3d';
  h.context.setSpatialViewMode('3d', { force: true });
  assert.equal(h.storage.get('view'), '2d');
  assert.equal(h.display(), 'block');
});
