// Verify graph.html mobile fit-to-screen keeps all nodes inside viewport.
// Usage: node scripts/screenshot_graph_mobile_fit_verify.cjs [baseUrl] [outDir]
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');

(async () => {
  const baseUrl = process.argv[2] || 'http://127.0.0.1:8765/graph.html';
  const outDir = path.resolve(process.argv[3] || path.join(__dirname, '..', '.cursor-artifacts', 'screenshots'));
  fs.mkdirSync(outDir, { recursive: true });

  const exe = process.env.PUPPETEER_EXECUTABLE_PATH
    || (fs.existsSync('/opt/pw-browsers/chromium-1194/chrome-linux/chrome')
      ? '/opt/pw-browsers/chromium-1194/chrome-linux/chrome'
      : (fs.existsSync('/usr/local/bin/google-chrome') ? '/usr/local/bin/google-chrome' : 'google-chrome'));

  const d3Local = path.resolve(__dirname, '..', 'node_modules', 'd3', 'dist', 'd3.min.js');
  const d3Body = fs.existsSync(d3Local) ? fs.readFileSync(d3Local) : null;

  const browser = await puppeteer.launch({
    executablePath: exe,
    headless: 'new',
    args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--window-size=390,844'],
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 2, isMobile: true, hasTouch: true });

    if (d3Body) {
      await page.setRequestInterception(true);
      page.on('request', (req) => {
        const u = req.url();
        if (u.includes('cdn.jsdelivr.net/npm/d3')) {
          req.respond({ status: 200, contentType: 'application/javascript', body: d3Body });
        } else {
          req.continue();
        }
      });
    }

    await page.goto(baseUrl, { waitUntil: 'domcontentloaded', timeout: 45000 });
    await page.waitForFunction(() => {
      const count = document.getElementById('graph-node-count');
      return count && count.textContent && !count.textContent.includes('加载中');
    }, { timeout: 90000 });
    await new Promise((r) => setTimeout(r, 3000));

    const loadedStats = await page.evaluate(() => {
      const svg = document.getElementById('graph-canvas');
      const wrap = document.getElementById('graph-wrap');
      const circles = Array.from(document.querySelectorAll('#graph-canvas .node-circle'));
      if (!svg || !wrap || !circles.length) return { ok: false, reason: 'missing graph elements' };

      const root = svg.querySelector('g');
      const tr = root?.getAttribute('transform') || '';
      const m = tr.match(/translate\(([-\d.]+),([-\d.]+)\)\s*scale\(([-\d.]+)\)/);
      if (!m) return { ok: false, reason: 'no zoom transform' };
      const tx = +m[1];
      const ty = +m[2];
      const k = +m[3];
      const W = wrap.clientWidth;
      const H = wrap.clientHeight;

      const pts = circles.map((c) => {
        const gtr = c.closest('g')?.getAttribute('transform') || '';
        const gm = gtr.match(/translate\(([-\d.]+),([-\d.]+)\)/);
        if (!gm) return null;
        const gx = +gm[1];
        const gy = +gm[2];
        const r = +(c.getAttribute('r') || 0);
        return {
          x: tx + k * gx,
          y: ty + k * gy,
          r: k * r,
        };
      }).filter(Boolean);

      const margin = 6;
      let outside = 0;
      for (const p of pts) {
        if (p.x - p.r < -margin || p.x + p.r > W + margin || p.y - p.r < -margin || p.y + p.r > H + margin) {
          outside += 1;
        }
      }

      return {
        ok: outside === 0,
        nodeCount: pts.length,
        outside,
        viewport: { W, H },
        zoom: { tx, ty, k },
      };
    });

    await page.click('#fit-to-screen');
    await new Promise((r) => setTimeout(r, 900));

    const afterFitStats = await page.evaluate(() => {
      const svg = document.getElementById('graph-canvas');
      const wrap = document.getElementById('graph-wrap');
      const circles = Array.from(document.querySelectorAll('#graph-canvas .node-circle'));
      const root = svg.querySelector('g');
      const tr = root?.getAttribute('transform') || '';
      const m = tr.match(/translate\(([-\d.]+),([-\d.]+)\)\s*scale\(([-\d.]+)\)/);
      const tx = +m[1];
      const ty = +m[2];
      const k = +m[3];
      const W = wrap.clientWidth;
      const H = wrap.clientHeight;
      const pts = circles.map((c) => {
        const gtr = c.closest('g')?.getAttribute('transform') || '';
        const gm = gtr.match(/translate\(([-\d.]+),([-\d.]+)\)/);
        if (!gm) return null;
        const gx = +gm[1];
        const gy = +gm[2];
        const r = +(c.getAttribute('r') || 0);
        return { x: tx + k * gx, y: ty + k * gy, r: k * r };
      }).filter(Boolean);
      const margin = 6;
      let outside = 0;
      for (const p of pts) {
        if (p.x - p.r < -margin || p.x + p.r > W + margin || p.y - p.r < -margin || p.y + p.r > H + margin) outside += 1;
      }
      return { ok: outside === 0, outside, nodeCount: pts.length, zoom: { k } };
    });

    const shotPath = path.join(outDir, 'graph-mobile-fit-screen.png');
    await page.screenshot({ path: shotPath, fullPage: false });

    const report = { loaded: loadedStats, afterFit: afterFitStats, screenshot: shotPath };
    fs.writeFileSync(path.join(outDir, 'graph-mobile-fit-verify.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report, null, 2));

    if (!loadedStats.ok || !afterFitStats.ok) process.exitCode = 1;
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
