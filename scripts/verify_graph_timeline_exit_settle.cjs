// Regression: 时序动画 → 退出 → 点击空白处后，2D 力模拟必须冷却，不能自振。
// Usage: node scripts/verify_graph_timeline_exit_settle.cjs [baseUrl] [outDir]
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');

(async () => {
  const baseUrl = process.argv[2] || 'http://127.0.0.1:8765/graph.html';
  const outDir = path.resolve(
    process.argv[3] || path.join(__dirname, '..', '.cursor-artifacts', 'screenshots')
  );
  fs.mkdirSync(outDir, { recursive: true });

  const browser = await puppeteer.launch({
    executablePath: process.env.PUPPETEER_EXECUTABLE_PATH
      || (fs.existsSync('/usr/local/bin/google-chrome') ? '/usr/local/bin/google-chrome' : 'google-chrome'),
    headless: 'new',
    args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--window-size=1440,900'],
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1440, height: 900 });
    await page.setCacheEnabled(false);
    await page.goto(baseUrl, { waitUntil: 'domcontentloaded', timeout: 45000 });
    await page.waitForFunction(() => {
      const loading = document.getElementById('graph-loading');
      const count = document.getElementById('graph-node-count');
      const loadingHidden = !loading || loading.hidden || loading.classList.contains('is-hidden')
        || window.getComputedStyle(loading).display === 'none';
      return loadingHidden && count && count.textContent && !count.textContent.includes('加载中');
    }, { timeout: 120000 });

    // Wait for initial layout settle
    await new Promise((r) => setTimeout(r, 2500));

    await page.click('#physics-toggle');
    await page.waitForSelector('#physics-panel:not([hidden])');
    await page.waitForSelector('#timeline-animate');

    // Enter timeline animation briefly
    await page.click('#timeline-animate');
    await page.waitForFunction(() => {
      const btn = document.getElementById('timeline-animate');
      return btn && btn.classList.contains('is-timeline-active');
    }, { timeout: 10000 });
    await new Promise((r) => setTimeout(r, 1200));

    // Exit timeline
    await page.click('#timeline-animate');
    await page.waitForFunction(() => {
      const dbg = window.__RN_GRAPH2D_DEBUG__;
      return dbg && !dbg.timelineAnimating() && dbg.alphaTarget() === 0;
    }, { timeout: 10000 });

    const afterExit = await page.evaluate(() => ({
      alpha: window.__RN_GRAPH2D_DEBUG__.alpha(),
      alphaTarget: window.__RN_GRAPH2D_DEBUG__.alphaTarget(),
      timelineAnimating: window.__RN_GRAPH2D_DEBUG__.timelineAnimating(),
    }));
    if (afterExit.alphaTarget !== 0) {
      throw new Error('alphaTarget must be 0 after exit, got ' + afterExit.alphaTarget);
    }

    const exitPath = path.join(outDir, 'graph-timeline-exit.png');
    await page.screenshot({ path: exitPath, fullPage: false });

    // Close physics panel so blank click lands on the graph canvas itself
    await page.evaluate(() => {
      const panel = document.getElementById('physics-panel');
      if (panel) panel.hidden = true;
    });

    // Click blank area on SVG (closeSidebar; should NOT reheat if sidebar was closed)
    const canvas = await page.$('#graph-canvas');
    const box = await canvas.boundingBox();
    await page.mouse.click(box.x + 40, box.y + 40);

    // After blank click with sidebar closed: sim must stay frozen (no stealth reheat)
    const samples = await page.evaluate(async () => {
      function stats() {
        const gs = [...document.querySelectorAll('#graph-canvas .nodes g.node-g')];
        let sumV = 0;
        let n = 0;
        gs.forEach((g) => {
          const d = g.__data__;
          if (!d) return;
          sumV += Math.hypot(d.vx || 0, d.vy || 0);
          n += 1;
        });
        return {
          meanSpeed: n ? sumV / n : null,
          alpha: window.__RN_GRAPH2D_DEBUG__.alpha(),
          alphaTarget: window.__RN_GRAPH2D_DEBUG__.alphaTarget(),
        };
      }
      const out = [];
      const t0 = performance.now();
      for (let i = 0; i < 20; i++) {
        out.push({ tMs: Math.round(performance.now() - t0), ...stats() });
        await new Promise((r) => setTimeout(r, 100));
      }
      return out;
    });

    const settlePath = path.join(outDir, 'graph-timeline-exit-blank-settled.png');
    await page.screenshot({ path: settlePath, fullPage: false });

    const alphas = samples.map((s) => s.alpha || 0);
    const alphaDrift = Math.max(...alphas) - Math.min(...alphas);
    const anyHotTarget = samples.some((s) => (s.alphaTarget || 0) > 1e-6);
    const lateMeanSpeed = samples.slice(-5).reduce((a, s) => a + (s.meanSpeed || 0), 0) / 5;

    const report = {
      afterExit,
      alphaDrift,
      lateMeanSpeed,
      anyHotTarget,
      samplesTail: samples.slice(-5),
      exitPath,
      settlePath,
    };
    console.log(JSON.stringify(report, null, 2));

    if (anyHotTarget) {
      throw new Error('alphaTarget stayed hot after blank click — self-oscillation risk');
    }
    // Sidebar was closed: blank click must not restart the force sim
    if (alphaDrift > 1e-6) {
      throw new Error('alpha drifted after blank click (unexpected reheat): ' + alphaDrift);
    }
    if (lateMeanSpeed > 0.05) {
      throw new Error('nodes still moving after exit+blank (should be frozen): ' + lateMeanSpeed);
    }

    console.log('OK: timeline exit + blank click stays settled (no self-oscillation)');
  } finally {
    await browser.close();
  }
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
