// Verify PR #867: selected/hovered 3D links thicken via linkPositionUpdate (mesh x/y scale).
// Also records a short MP4 comparing sidebar closed vs open.
// Usage: node scripts/verify_graph_3d_link_thickness.cjs [baseUrl] [outDir]
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const baseUrl = process.argv[2] || 'http://127.0.0.1:8765/graph.html';
const outDir = path.resolve(process.argv[3] || path.join(__dirname, '..', '.cursor-artifacts', 'screenshots'));
const focusId = 'wiki/concepts/sim2real.md';
const chromeArgs = [
  '--no-sandbox',
  '--disable-dev-shm-usage',
  '--window-size=1440,900',
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

async function sampleLinkScales(page, label) {
  return page.evaluate((label) => {
    const g = window.__fg3dInstances && window.__fg3dInstances[window.__fg3dInstances.length - 1];
    if (!g || !g.scene) return { label, error: 'no ForceGraph3D instance' };
    const scene = g.scene();
    const scales = [];
    scene.traverse((o) => {
      if (o.isMesh && o.geometry?.type?.includes('Cylinder')) {
        scales.push({ x: o.scale.x, y: o.scale.y, z: o.scale.z });
      }
    });
    return {
      label,
      cylinders: scales.length,
      thick35: scales.filter((s) => s.x >= 3.4 && s.x <= 3.6).length,
      thick3: scales.filter((s) => s.x >= 2.9 && s.x <= 3.1).length,
      unit: scales.filter((s) => Math.abs(s.x - 1) < 0.05).length,
      maxX: scales.reduce((m, s) => Math.max(m, s.x), 0),
      sidebarOpen: document.getElementById('sidebar')?.classList.contains('open'),
    };
  }, label);
}

async function hideLoading(page) {
  await page.evaluate(() => {
    const el = document.getElementById('graph-loading');
    if (el) {
      el.hidden = true;
      el.classList.add('is-hidden');
      el.style.display = 'none';
    }
  });
}

async function captureFrames(page, dir, count, delayMs) {
  fs.mkdirSync(dir, { recursive: true });
  const files = [];
  for (let i = 0; i < count; i++) {
    const file = path.join(dir, `frame_${String(i).padStart(3, '0')}.png`);
    await page.screenshot({ path: file });
    files.push(file);
    if (i + 1 < count) await sleep(delayMs);
  }
  return files;
}

function framesToMp4(frameDir, outMp4, fps) {
  const pattern = path.join(frameDir, 'frame_%03d.png');
  execSync(
  `ffmpeg -y -framerate ${fps} -i "${pattern}" -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p "${outMp4}"`,
    { stdio: 'pipe' },
  );
}

(async () => {
  fs.mkdirSync(outDir, { recursive: true });
  const exe = process.env.PUPPETEER_EXECUTABLE_PATH
    || (fs.existsSync('/usr/local/bin/google-chrome') ? '/usr/local/bin/google-chrome' : 'google-chrome');

  const browser = await puppeteer.launch({
    executablePath: exe,
    headless: 'new',
    args: chromeArgs,
    defaultViewport: { width: 1440, height: 900, deviceScaleFactor: 1 },
  });

  const report = { cases: [], ok: false };

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1 });

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

    // Case 1: 3D baseline (no sidebar highlight)
    await page.goto(urlWith({ view: '3d' }), { waitUntil: 'domcontentloaded', timeout: 120000 });
    await page.waitForFunction(() => {
      const el = document.getElementById('graph-loading');
      return !el || el.hidden || el.classList.contains('is-hidden');
    }, { timeout: 90000 });
    await sleep(6000);
    await hideLoading(page);
    const baseline = await sampleLinkScales(page, 'baseline-no-sidebar');
    baseline.pass = baseline.maxX <= 1.1 && baseline.thick35 === 0;
    report.cases.push(baseline);
    await page.screenshot({ path: path.join(outDir, 'graph-3d-link-baseline.png') });

    const framesDir = path.join(outDir, 'graph-3d-link-frames');
    if (fs.existsSync(framesDir)) {
      for (const f of fs.readdirSync(framesDir)) fs.unlinkSync(path.join(framesDir, f));
    }
    await captureFrames(page, framesDir, 4, 500);

    // Case 2: open sidebar via ?focus= (selected node links should scale x/y to 3.5)
    await page.goto(urlWith({ view: '3d', focus: focusId }), { waitUntil: 'domcontentloaded', timeout: 120000 });
    await page.waitForFunction(() => {
      const el = document.getElementById('graph-loading');
      return !el || el.hidden || el.classList.contains('is-hidden');
    }, { timeout: 90000 });
    await sleep(7000);
    await hideLoading(page);
    const focused = await sampleLinkScales(page, 'sidebar-focus');
    focused.pass = focused.thick35 > 0 && focused.unit > 0 && focused.maxX >= 3.4;
    report.cases.push(focused);
    await page.screenshot({ path: path.join(outDir, 'graph-3d-link-focused.png') });
    await captureFrames(page, framesDir, 8, 500);

    // Case 3: close sidebar — scales should return to 1
    await page.click('#sb-close');
    await sleep(2500);
    const cleared = await sampleLinkScales(page, 'sidebar-closed');
    cleared.pass = cleared.maxX <= 1.1 && cleared.thick35 === 0;
    report.cases.push(cleared);
    await captureFrames(page, framesDir, 4, 500);

    const mp4 = path.join(outDir, 'graph-3d-link-thickness-verify.mp4');
    framesToMp4(framesDir, mp4, 2);
    report.video = mp4;
    report.focusId = focusId;
    report.ok = report.cases.every((c) => c.pass);

    const outJson = path.join(outDir, 'graph-3d-link-thickness-verify.json');
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
