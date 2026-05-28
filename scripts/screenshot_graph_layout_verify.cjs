// Verify graph.html layout: nodes centered after load and after clicking blank canvas.
// Usage: node scripts/screenshot_graph_layout_verify.cjs [baseUrl] [outDir]
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');

(async () => {
  const baseUrl = process.argv[2] || 'http://127.0.0.1:8765/graph.html';
  const outDir = path.resolve(process.argv[3] || path.join(__dirname, '..', '.cursor-artifacts', 'screenshots'));
  fs.mkdirSync(outDir, { recursive: true });

  const exe = process.env.PUPPETEER_EXECUTABLE_PATH
    || (fs.existsSync('/usr/local/bin/google-chrome') ? '/usr/local/bin/google-chrome' : 'google-chrome');

  const d3Local = path.resolve(__dirname, '..', 'node_modules', 'd3', 'dist', 'd3.min.js');
  const d3Body = fs.existsSync(d3Local) ? fs.readFileSync(d3Local) : null;

  const browser = await puppeteer.launch({
    executablePath: exe,
    headless: 'new',
    args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--window-size=1440,900'],
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1 });

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
      const loading = document.getElementById('graph-loading');
      const count = document.getElementById('graph-node-count');
      const loadingHidden = !loading || loading.style.display === 'none';
      const countReady = count && count.textContent && !count.textContent.includes('加载中');
      return loadingHidden && countReady;
    }, { timeout: 90000 });
    await new Promise((r) => setTimeout(r, 2500));

    const loadedPath = path.join(outDir, 'graph-layout-loaded.png');
    await page.screenshot({ path: loadedPath, fullPage: false });

    const loadedStats = await page.evaluate(() => {
      const circles = Array.from(document.querySelectorAll('#graph-canvas .node-circle'));
      if (!circles.length) return { ok: false, reason: 'no nodes rendered' };
      const pts = circles.map((c) => {
        const tr = c.closest('g')?.getAttribute('transform') || '';
        const m = tr.match(/translate\(([-\d.]+),([-\d.]+)\)/);
        return m ? { x: +m[1], y: +m[2] } : null;
      }).filter(Boolean);
      const xs = pts.map((p) => p.x);
      const ys = pts.map((p) => p.y);
      const cx = xs.reduce((a, b) => a + b, 0) / xs.length;
      const cy = ys.reduce((a, b) => a + b, 0) / ys.length;
      const svg = document.getElementById('graph-canvas');
      const w = svg?.clientWidth || 1440;
      const h = svg?.clientHeight || 900;
      const cornerCluster = xs.filter((x) => x < w * 0.15).length / xs.length > 0.8
        && ys.filter((y) => y < h * 0.15).length / ys.length > 0.8;
      return {
        ok: !cornerCluster,
        nodeCount: pts.length,
        centroid: { x: Math.round(cx), y: Math.round(cy) },
        viewport: { w, h },
        cornerCluster,
      };
    });

    // Click blank canvas (not a node) to trigger closeSidebar / simulation restart
    await page.mouse.click(720, 450);
    await new Promise((r) => setTimeout(r, 2000));

    const afterClickPath = path.join(outDir, 'graph-layout-after-blank-click.png');
    await page.screenshot({ path: afterClickPath, fullPage: false });

    const afterStats = await page.evaluate(() => {
      const circles = Array.from(document.querySelectorAll('#graph-canvas .node-circle'));
      const pts = circles.map((c) => {
        const tr = c.closest('g')?.getAttribute('transform') || '';
        const m = tr.match(/translate\(([-\d.]+),([-\d.]+)\)/);
        return m ? { x: +m[1], y: +m[2] } : null;
      }).filter(Boolean);
      const xs = pts.map((p) => p.x);
      const ys = pts.map((p) => p.y);
      const cx = xs.reduce((a, b) => a + b, 0) / xs.length;
      const cy = ys.reduce((a, b) => a + b, 0) / ys.length;
      const svg = document.getElementById('graph-canvas');
      const w = svg?.clientWidth || 1440;
      const h = svg?.clientHeight || 900;
      const cornerBR = xs.filter((x) => x > w * 0.85).length / xs.length > 0.8
        && ys.filter((y) => y > h * 0.85).length / ys.length > 0.8;
      return {
        ok: !cornerBR,
        centroid: { x: Math.round(cx), y: Math.round(cy) },
        cornerBR,
      };
    });

    const report = {
      loaded: loadedStats,
      afterClick: afterStats,
      screenshots: [loadedPath, afterClickPath],
    };
    fs.writeFileSync(path.join(outDir, 'graph-layout-verify.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report, null, 2));

    if (!loadedStats.ok || !afterStats.ok) {
      process.exitCode = 1;
    }
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
