// Verify graph.html shell tracks visual viewport height (Chrome toolbar show/hide).
// Usage: node scripts/verify_graph_mobile_viewport_gap.cjs [baseUrl] [outDir]
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');

(async () => {
  const baseUrl = process.argv[2] || 'http://127.0.0.1:8765/graph.html';
  const outDir = path.resolve(process.argv[3] || path.join(__dirname, '..', '.cursor-artifacts', 'screenshots'));
  fs.mkdirSync(outDir, { recursive: true });

  const exe = process.env.PUPPETEER_EXECUTABLE_PATH
    || (fs.existsSync('/usr/local/bin/google-chrome')
      ? '/usr/local/bin/google-chrome'
      : (fs.existsSync('/opt/pw-browsers/chromium-1194/chrome-linux/chrome')
        ? '/opt/pw-browsers/chromium-1194/chrome-linux/chrome'
        : 'google-chrome'));

  const browser = await puppeteer.launch({
    executablePath: exe,
    headless: 'new',
    args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--window-size=390,844'],
  });

  try {
    const page = await browser.newPage();
    // Simulate Chrome with bottom toolbar visible (shorter visual viewport)
    await page.setViewport({ width: 390, height: 700, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
    await page.goto(baseUrl, { waitUntil: 'domcontentloaded', timeout: 45000 });
    await page.waitForFunction(() => {
      const count = document.getElementById('graph-node-count');
      return count && count.textContent && !count.textContent.includes('加载中');
    }, { timeout: 90000 });
    await new Promise((r) => setTimeout(r, 1500));

    const measure = () => page.evaluate(() => {
      const wrap = document.getElementById('graph-wrap');
      const rect = wrap.getBoundingClientRect();
      const appVh = getComputedStyle(document.documentElement).getPropertyValue('--app-vh').trim();
      const vvH = window.visualViewport ? window.visualViewport.height : window.innerHeight;
      const gap = Math.round(window.innerHeight - rect.bottom);
      return {
        wrapTop: Math.round(rect.top),
        wrapBottom: Math.round(rect.bottom),
        wrapHeight: Math.round(rect.height),
        innerHeight: window.innerHeight,
        vvHeight: Math.round(vvH),
        appVh,
        gapBottom: gap,
        ok: Math.abs(gap) <= 2,
      };
    });

    const before = await measure();
    await page.screenshot({
      path: path.join(outDir, 'graph-mobile-viewport-toolbar-visible.png'),
      fullPage: false,
    });

    // Simulate Chrome bottom toolbar hiding → taller visual viewport
    await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
    await page.waitForFunction(() => window.innerHeight >= 840, { timeout: 5000 });
    // Real Chrome may fire resize; Puppeteer often does not. User scenario is「点到 graph view」—
    // pointerdown handler re-syncs shell to the new innerHeight.
    await page.evaluate(() => {
      window.dispatchEvent(new Event('resize'));
      window.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, clientX: 200, clientY: 400 }));
    });
    // Wait until shell catches up (dvh and/or JS --app-vh + double-rAF)
    await page.waitForFunction(() => {
      const wrap = document.getElementById('graph-wrap');
      if (!wrap) return false;
      const gap = Math.abs(window.innerHeight - wrap.getBoundingClientRect().bottom);
      return gap <= 2;
    }, { timeout: 5000 });
    await new Promise((r) => setTimeout(r, 400));

    const after = await measure();
    await page.screenshot({
      path: path.join(outDir, 'graph-mobile-viewport-toolbar-hidden.png'),
      fullPage: false,
    });

    const grew = after.wrapHeight > before.wrapHeight;
    const report = { before, after, grew, pass: before.ok && after.ok && grew };
    const reportPath = path.join(outDir, 'graph-mobile-viewport-gap-report.json');
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report, null, 2));
    if (!report.pass) {
      console.error('FAIL: graph shell did not track viewport height');
      process.exitCode = 1;
    } else {
      console.log('PASS: graph shell tracks viewport; gapBottom≈0 before/after');
    }
  } finally {
    await browser.close();
  }
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
