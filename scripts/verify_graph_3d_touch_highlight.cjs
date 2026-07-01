// Verify 3D selected-node highlight survives touch orbit drag (mobile viewport).
// Usage: node scripts/verify_graph_3d_touch_highlight.cjs [baseUrl] [outDir]
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');

const baseUrl = process.argv[2] || 'http://127.0.0.1:8765/graph.html';
const outDir = path.resolve(process.argv[3] || path.join(__dirname, '..', '.cursor-artifacts', 'screenshots'));
const focusId = 'wiki/concepts/sim2real.md';
const chromeArgs = [
  '--no-sandbox',
  '--disable-dev-shm-usage',
  '--window-size=390,844',
  '--use-gl=angle',
  '--use-angle=swiftshader',
  '--enable-unsafe-swiftshader',
];

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function urlWith(params) {
  const u = new URL(baseUrl);
  Object.entries(params).forEach(([k, v]) => u.searchParams.set(k, v));
  return u.toString();
}

async function sampleLinkScales(page) {
  return page.evaluate(() => {
    const g = window.__fg3dInstances && window.__fg3dInstances[window.__fg3dInstances.length - 1];
    if (!g || !g.scene) return { error: 'no ForceGraph3D instance' };
    const scene = g.scene();
    const scales = [];
    scene.traverse((o) => {
      if (o.isMesh && o.geometry?.type?.includes('Cylinder')) {
        scales.push({ x: o.scale.x, y: o.scale.y });
      }
    });
    return {
      cylinders: scales.length,
      thick12: scales.filter((s) => s.x >= 11.9 && s.x <= 12.1).length,
      maxX: scales.reduce((m, s) => Math.max(m, s.x), 0),
      isMobile: window.matchMedia('(hover: none) and (pointer: coarse)').matches,
    };
  });
}

(async () => {
  fs.mkdirSync(outDir, { recursive: true });
  const exe = process.env.PUPPETEER_EXECUTABLE_PATH
    || (fs.existsSync('/usr/local/bin/google-chrome') ? '/usr/local/bin/google-chrome' : 'google-chrome');

  const browser = await puppeteer.launch({
    executablePath: exe,
    headless: 'new',
    args: chromeArgs,
    defaultViewport: { width: 390, height: 844, deviceScaleFactor: 2, isMobile: true, hasTouch: true },
  });

  const report = { cases: [], ok: false };

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 2, isMobile: true, hasTouch: true });

    await page.evaluateOnNewDocument(() => {
      const orig = window.matchMedia.bind(window);
      window.matchMedia = (q) => {
        if (q === '(hover: none) and (pointer: coarse)') {
          return { matches: true, media: q, addListener() {}, removeListener() {}, addEventListener() {}, removeEventListener() {}, dispatchEvent() { return true; } };
        }
        return orig(q);
      };
    });

    await page.evaluateOnNewDocument(() => {
      window.__fg3dInstances = [];
      const hook = () => {
        if (!window.ForceGraph3D || window.__fgHooked) return;
        window.__fgHooked = true;
        const Orig = window.ForceGraph3D;
        function Wrapped(el) {
          const inst = Orig(el);
          window.__fg3dInstances.push(inst);
          return inst;
        }
        Wrapped.prototype = Orig.prototype;
        Object.assign(Wrapped, Orig);
        window.ForceGraph3D = Wrapped;
      };
      const t = setInterval(() => { hook(); if (window.__fgHooked) clearInterval(t); }, 5);
    });

    // Case 1: mobile viewport + focus selection (persistent sidebarNodeId highlight)
    await page.goto(urlWith({ view: '3d', focus: focusId }), { waitUntil: 'domcontentloaded', timeout: 120000 });
    await page.waitForFunction(() => {
      const el = document.getElementById('graph-loading');
      return !el || el.hidden || el.classList.contains('is-hidden');
    }, { timeout: 90000 });
    await sleep(8000);
    await page.evaluate(() => {
      const el = document.getElementById('graph-loading');
      if (el) { el.hidden = true; el.classList.add('is-hidden'); el.style.display = 'none'; }
    });

    const before = await sampleLinkScales(page);
    before.label = 'before-orbit';
    before.pass = before.thick12 > 0 && before.maxX >= 11.9;
    report.cases.push(before);

    const canvas = await page.$('#graph-canvas-3d canvas');
    const box = await canvas.boundingBox();
    const cx = box.x + box.width * 0.52;
    const cy = box.y + box.height * 0.48;

    await page.touchscreen.touchStart(cx, cy);
    await sleep(60);
    await page.touchscreen.touchMove(cx + 110, cy - 70);
    await sleep(60);
    await page.touchscreen.touchMove(cx - 80, cy + 90);
    await sleep(60);
    await page.touchscreen.touchEnd();
    await sleep(2000);

    const after = await sampleLinkScales(page);
    after.label = 'after-orbit';
    after.pass = after.thick12 > 0 && after.maxX >= 11.9;
    report.cases.push(after);

    await page.screenshot({ path: path.join(outDir, 'graph-3d-touch-highlight-after-rotate.png') });

    report.ok = report.cases.every((c) => c.pass);
    const outJson = path.join(outDir, 'graph-3d-touch-highlight-verify.json');
    fs.writeFileSync(outJson, JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report, null, 2));
    if (!report.ok) process.exitCode = 1;
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
