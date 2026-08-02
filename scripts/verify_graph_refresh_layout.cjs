// Verify 2D graph settle via 「刷新布局」ONLY — never touches timeline animation.
// Usage: node scripts/verify_graph_refresh_layout.cjs [baseUrl] [outDir]
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

    await new Promise((r) => setTimeout(r, 2000));
    await page.click('#physics-toggle');
    await page.waitForSelector('#physics-panel:not([hidden])');

    const btnText = await page.$eval('#restart-simulation', (el) => el.textContent.trim());
    if (!btnText.includes('刷新布局')) {
      throw new Error('Expected 刷新布局 button, got: ' + btnText);
    }

    async function sampleAfterRefresh() {
      await page.click('#restart-simulation');
      // sync warmup 20 tick 很快结束；稍等一帧再开始采可见落稳阶段
      await new Promise((r) => setTimeout(r, 50));
      return page.evaluate(async () => {
        function stats() {
          const gs = [...document.querySelectorAll('#graph-canvas .nodes g.node-g')];
          let sumV = 0;
          let sumR = 0;
          let cx = 0;
          let cy = 0;
          const pts = [];
          gs.forEach((g) => {
            const d = g.__data__;
            if (!d || d.x == null) return;
            pts.push(d);
            cx += d.x;
            cy += d.y;
          });
          const n = pts.length;
          if (!n) return null;
          cx /= n;
          cy /= n;
          pts.forEach((d) => {
            sumV += Math.hypot(d.vx || 0, d.vy || 0);
            sumR += Math.hypot(d.x - cx, d.y - cy);
          });
          return { meanSpeed: sumV / n, meanR: sumR / n };
        }
        const out = [];
        const t0 = performance.now();
        for (let i = 0; i < 28; i++) {
          out.push({ tMs: Math.round(performance.now() - t0), ...stats() });
          await new Promise((r) => setTimeout(r, 100));
        }
        return out;
      });
    }

    function analyze(series) {
      // 平滑后再数半径方向翻转，避免采样噪声虚高「回弹次数」
      const rs = series.map((s) => s.meanR);
      const sm = rs.map((_, i) => {
        let s = 0;
        let n = 0;
        for (let j = i - 2; j <= i + 2; j++) {
          if (j >= 0 && j < rs.length) { s += rs[j]; n += 1; }
        }
        return s / n;
      });
      let flips = 0;
      let prev = 0;
      for (let i = 1; i < sm.length; i++) {
        const d = sm[i] - sm[i - 1];
        if (Math.abs(d) < 2.5) continue;
        const sign = d > 0 ? 1 : -1;
        if (prev && sign !== prev) {
          flips += 1;
          prev = sign;
        } else if (!prev) prev = sign;
      }
      return {
        flips,
        firstV: +series[0].meanSpeed.toFixed(2),
        lateV: +series.at(-1).meanSpeed.toFixed(2),
        maxV: +Math.max(...series.map((s) => s.meanSpeed)).toFixed(2),
      };
    }

    const s1 = await sampleAfterRefresh();
    await page.screenshot({
      path: path.join(outDir, 'graph-refresh-layout-verify-1.png'),
      fullPage: false,
    });
    await new Promise((r) => setTimeout(r, 800));
    const s2 = await sampleAfterRefresh();
    await page.screenshot({
      path: path.join(outDir, 'graph-refresh-layout-verify-2.png'),
      fullPage: false,
    });

    const a1 = analyze(s1);
    const a2 = analyze(s2);
    const report = {
      method: 'refresh-layout-only (#restart-simulation)',
      neverTouchedTimeline: true,
      button: btnText,
      refresh1: a1,
      refresh2: a2,
      // 有活力 + 最多一次半径回弹 + 数秒内落稳
      ok: a1.maxV > 15 && a1.flips <= 1 && a1.lateV < 6
        && a2.maxV > 15 && a2.flips <= 1 && a2.lateV < 6,
    };
    fs.writeFileSync(path.join(outDir, 'graph-refresh-layout-report.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report, null, 2));
    if (!report.ok) process.exitCode = 2;
  } finally {
    await browser.close();
  }
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
