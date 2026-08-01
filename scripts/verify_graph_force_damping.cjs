// Measure 2D force-sim kinetic energy decay on graph.html and screenshot settle phases.
// Usage: node scripts/verify_graph_force_damping.cjs [baseUrl] [outDir]
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');

(async () => {
  const baseUrl = process.argv[2] || 'http://127.0.0.1:8765/graph.html';
  const outDir = path.resolve(
    process.argv[3] || path.join(__dirname, '..', '.cursor-artifacts', 'screenshots')
  );
  fs.mkdirSync(outDir, { recursive: true });

  const exe = process.env.PUPPETEER_EXECUTABLE_PATH
    || (fs.existsSync('/usr/local/bin/google-chrome') ? '/usr/local/bin/google-chrome' : 'google-chrome');

  const browser = await puppeteer.launch({
    executablePath: exe,
    headless: 'new',
    args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--window-size=1440,900'],
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1 });
    await page.goto(baseUrl, { waitUntil: 'domcontentloaded', timeout: 45000 });

    await page.waitForFunction(() => {
      const loading = document.getElementById('graph-loading');
      const count = document.getElementById('graph-node-count');
      const loadingHidden = !loading || loading.hidden
        || loading.style.display === 'none'
        || window.getComputedStyle(loading).display === 'none';
      const countReady = count && count.textContent && !count.textContent.includes('加载中');
      return loadingHidden && countReady;
    }, { timeout: 90000 });

    // Sample mean |v| from live node transforms via MutationObserver-like polling of d3 nodes.
    // graph.html keeps simulation in closure; read velocities from DOM-bound datum via __data__.
    const earlyPath = path.join(outDir, 'graph-force-damping-early.png');
    await new Promise((r) => setTimeout(r, 400));
    await page.screenshot({ path: earlyPath, fullPage: false });

    const samples = await page.evaluate(async () => {
      function meanSpeed() {
        const gs = document.querySelectorAll('#graph-canvas .nodes g.node-g');
        let sum = 0;
        let n = 0;
        gs.forEach((g) => {
          const d = g.__data__;
          if (!d || d.vx == null || d.vy == null) return;
          sum += Math.hypot(d.vx, d.vy);
          n += 1;
        });
        return n ? sum / n : null;
      }
      function alphaGuess() {
        // No direct handle; infer from residual speed trend.
        return null;
      }
      const out = [];
      const t0 = performance.now();
      for (let i = 0; i < 24; i++) {
        out.push({
          tMs: Math.round(performance.now() - t0),
          meanSpeed: meanSpeed(),
          alpha: alphaGuess(),
        });
        await new Promise((r) => setTimeout(r, 120));
      }
      return out;
    });

    const midPath = path.join(outDir, 'graph-force-damping-mid.png');
    await page.screenshot({ path: midPath, fullPage: false });

    // Wait for settle, then restart once to exercise phyllotaxis seed path.
    await new Promise((r) => setTimeout(r, 2000));
    const settledPath = path.join(outDir, 'graph-force-damping-settled.png');
    await page.screenshot({ path: settledPath, fullPage: false });

    // 「刷新布局」在参数浮窗内，默认 hidden。
    await page.click('#physics-toggle');
    await page.waitForSelector('#physics-panel:not([hidden])', { timeout: 5000 });
    await page.click('#restart-simulation');
    await new Promise((r) => setTimeout(r, 500));
    const restartEarlyPath = path.join(outDir, 'graph-force-damping-restart-early.png');
    await page.screenshot({ path: restartEarlyPath, fullPage: false });

    const restartSamples = await page.evaluate(async () => {
      function meanSpeed() {
        const gs = document.querySelectorAll('#graph-canvas .nodes g.node-g');
        let sum = 0;
        let n = 0;
        gs.forEach((g) => {
          const d = g.__data__;
          if (!d || d.vx == null || d.vy == null) return;
          sum += Math.hypot(d.vx, d.vy);
          n += 1;
        });
        return n ? sum / n : null;
      }
      const out = [];
      const t0 = performance.now();
      for (let i = 0; i < 20; i++) {
        out.push({ tMs: Math.round(performance.now() - t0), meanSpeed: meanSpeed() });
        await new Promise((r) => setTimeout(r, 120));
      }
      return out;
    });

    await new Promise((r) => setTimeout(r, 1800));
    const restartSettledPath = path.join(outDir, 'graph-force-damping-restart-settled.png');
    await page.screenshot({ path: restartSettledPath, fullPage: false });

    const report = {
      ok: true,
      samples,
      restartSamples,
      thresholds: {
        // After ~2.5s sampling window, mean |v| should be small (not still ringing hard).
        lateMeanSpeedMax: 5.0,
      },
      lateMeanSpeed: samples.length ? samples[samples.length - 1].meanSpeed : null,
      restartLateMeanSpeed: restartSamples.length
        ? restartSamples[restartSamples.length - 1].meanSpeed
        : null,
      screenshots: {
        early: earlyPath,
        mid: midPath,
        settled: settledPath,
        restartEarly: restartEarlyPath,
        restartSettled: restartSettledPath,
      },
    };
    const late = report.lateMeanSpeed;
    const rlate = report.restartLateMeanSpeed;
    report.ok = (late != null && late < report.thresholds.lateMeanSpeedMax)
      && (rlate != null && rlate < report.thresholds.lateMeanSpeedMax);

    const reportPath = path.join(outDir, 'graph-force-damping-report.json');
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report, null, 2));
    if (!report.ok) process.exitCode = 2;
  } finally {
    await browser.close();
  }
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
