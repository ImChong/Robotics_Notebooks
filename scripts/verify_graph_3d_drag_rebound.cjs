// Verify 3D drag rebound AFTER initial force engine cool-down.
// Usage: node scripts/verify_graph_3d_drag_rebound.cjs [baseUrl] [outDir]
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');

const baseUrl = process.argv[2] || 'http://127.0.0.1:8765/graph.html?view=3d';
const outDir = path.resolve(process.argv[3] || path.join(__dirname, '..', '.cursor-artifacts', 'screenshots'));

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

(async () => {
  fs.mkdirSync(outDir, { recursive: true });
  const exe = process.env.PUPPETEER_EXECUTABLE_PATH
    || (fs.existsSync('/usr/local/bin/google-chrome') ? '/usr/local/bin/google-chrome' : 'google-chrome');

  const browser = await puppeteer.launch({
    executablePath: exe,
    headless: 'new',
    protocolTimeout: 180000,
    args: [
      '--no-sandbox',
      '--disable-dev-shm-usage',
      '--window-size=1280,800',
      '--use-gl=angle',
      '--use-angle=swiftshader',
      '--enable-unsafe-swiftshader',
    ],
    defaultViewport: { width: 1280, height: 800 },
  });

  const report = { ok: false, steps: [], errors: [] };

  try {
    const page = await browser.newPage();
    page.on('pageerror', (e) => report.errors.push(String(e)));

    await page.evaluateOnNewDocument(() => {
      window.__fg3dInstances = [];
      const hook = () => {
        if (!window.ForceGraph3D || window.__fgHooked) return;
        window.__fgHooked = true;
        const Orig = window.ForceGraph3D;
        function Wrapped(cfg) {
          const factory = Orig(cfg);
          return function (el) {
            const inst = factory(el);
            window.__fg3dInstances.push(inst);
            return inst;
          };
        }
        Object.assign(Wrapped, Orig);
        window.ForceGraph3D = Wrapped;
      };
      const t = setInterval(() => { hook(); if (window.__fgHooked) clearInterval(t); }, 5);
    });

    await page.goto(baseUrl, { waitUntil: 'domcontentloaded', timeout: 120000 });
    await page.waitForFunction(() => {
      const el = document.getElementById('graph-loading');
      return !el || el.hidden || el.classList.contains('is-hidden');
    }, { timeout: 120000 });

    // Wait for initial layout to cool (positions stable).
    await sleep(7000);
    await page.waitForFunction(() => {
      const g = window.__fg3dInstances && window.__fg3dInstances[window.__fg3dInstances.length - 1];
      if (!g) return false;
      const n = g.graphData().nodes[0];
      if (!n) return false;
      const key = n.x.toFixed(2) + ':' + n.y.toFixed(2) + ':' + n.z.toFixed(2);
      if (window.__coolKey === key) window.__coolN = (window.__coolN || 0) + 1;
      else { window.__coolKey = key; window.__coolN = 0; }
      return window.__coolN > 20;
    }, { timeout: 45000, polling: 100 });

    const result = await page.evaluate(async () => {
      const g = window.__fg3dInstances[window.__fg3dInstances.length - 1];
      async function frames(n) {
        for (let i = 0; i < n; i++) await new Promise((r) => requestAnimationFrame(r));
      }
      function pick() {
        return g.graphData().nodes.reduce((a, b) => ((a.val || 0) >= (b.val || 0) ? a : b));
      }

      const hasDrag = typeof g.onNodeDrag === 'function';
      const hasDragEnd = typeof g.onNodeDragEnd === 'function';
      const dragCb = hasDrag ? g.onNodeDrag() : null;
      const dragEndCb = hasDragEnd ? g.onNodeDragEnd() : null;

      // Simulate library drag + our registered callbacks (as DragControls would).
      const target = pick();
      const before = { x: target.x, y: target.y, z: target.z, fxUndef: target.fx === undefined };
      target.__initialFixedPos = { fx: target.fx, fy: target.fy, fz: target.fz };
      target.__initialPos = { x: target.x, y: target.y, z: target.z };
      target.fx = target.x = before.x + 90;
      target.fy = target.y = before.y - 70;
      target.fz = target.z = before.z + 50;
      target.__dragged = true;

      // Library drag frame: alphaTarget(0.3).resetCountdown() — wrapper has no
      // d3AlphaTarget / resetCountdown; callbacks perform the needed reheat.
      if (typeof dragCb === 'function') dragCb(target, { x: 90, y: -70, z: 50 });
      await frames(15);
      const mid = { x: target.x, y: target.y, z: target.z, fx: target.fx };

      // Library dragend: restore fx from __initialFixedPos, then onNodeDragEnd, then alphaTarget(0)
      const l = target.__initialFixedPos;
      ['x', 'y', 'z'].forEach((axis) => {
        const key = 'f' + axis;
        if (void 0 === l[key]) delete target[key];
      });
      delete target.__initialFixedPos;
      delete target.__initialPos;
      const translate = { x: before.x - target.x, y: before.y - target.y, z: before.z - target.z };
      delete target.__dragged;
      if (typeof dragEndCb === 'function') dragEndCb(target, translate);

      await frames(120);
      const after = {
        x: target.x, y: target.y, z: target.z,
        hasOwnFx: Object.prototype.hasOwnProperty.call(target, 'fx'),
        fx: target.fx,
      };
      const distFromPinned = Math.hypot(after.x - mid.x, after.y - mid.y, after.z - mid.z);
      return {
        before,
        mid,
        after,
        distFromPinned,
        released: !after.hasOwnFx && after.fx === undefined,
        dragCbType: typeof dragCb,
        dragEndCbType: typeof dragEndCb,
        nodeCount: g.graphData().nodes.length,
        fxNull: g.graphData().nodes.filter((n) => n.fx === null).length,
        alphaMin: g.d3AlphaMin(),
      };
    });

    report.steps.push(result);
    report.metrics = {
      distFromPinned: result.distFromPinned,
      released: result.released,
      callbacksWired: result.dragCbType === 'function' && result.dragEndCbType === 'function',
    };
    // After cool-down, released node must move away from the pinned drop point.
    report.ok = report.metrics.callbacksWired
      && report.metrics.released
      && report.metrics.distFromPinned > 15
      && result.fxNull === 0;

    await page.screenshot({
      path: path.join(outDir, 'graph-3d-drag-rebound-after-cooldown.png'),
      type: 'png',
    });
    fs.writeFileSync(
      path.join(outDir, 'graph-3d-drag-rebound-verify.json'),
      JSON.stringify(report, null, 2)
    );
    console.log(JSON.stringify(report, null, 2));
    if (!report.ok) process.exitCode = 1;
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
