// Verify graph.html filter panel: desktop resize handles + mobile scrim close
const puppeteer = require('puppeteer-core');
const path = require('path');
const fs = require('fs');

const OUT_DIR = path.resolve(__dirname, '..', '.cursor-artifacts', 'screenshots');
const CHROME_CANDIDATES = [
  process.env.CHROME_PATH,
  '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
  '/usr/local/bin/google-chrome',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
  '/usr/bin/chromium-browser',
].filter(Boolean);
const exe = CHROME_CANDIDATES.find((p) => fs.existsSync(p));
if (!exe) {
  console.error('No Chrome/Chromium found. Set CHROME_PATH.');
  process.exit(1);
}
const d3Path = path.resolve(__dirname, '..', 'node_modules', 'd3', 'dist', 'd3.min.js');

async function launch() {
  const d3Body = fs.readFileSync(d3Path);
  const browser = await puppeteer.launch({
    executablePath: exe,
    headless: 'new',
    args: ['--no-sandbox', '--disable-gpu', '--ignore-certificate-errors'],
    ignoreHTTPSErrors: true,
  });
  return { browser, d3Body };
}

async function preparePage(browser, d3Body, width, height) {
  const page = await browser.newPage();
  await page.setViewport({
    width,
    height,
    deviceScaleFactor: width < 500 ? 2 : 1,
    isMobile: width < 500,
    hasTouch: width < 500,
  });
  await page.setRequestInterception(true);
  page.on('request', (req) => {
    if (req.url().includes('cdn.jsdelivr.net/npm/d3')) {
      req.respond({ status: 200, contentType: 'application/javascript', body: d3Body });
    } else {
      req.continue();
    }
  });
  const base = process.env.GRAPH_BASE_URL || 'http://127.0.0.1:8765/graph.html';
  await page.goto(base, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => {
    const el = document.getElementById('graph-node-count');
    return el && el.textContent && !el.textContent.includes('加载中');
  }, { timeout: 25000 }).catch(() => {});
  await new Promise((r) => setTimeout(r, 4500));
  await page.evaluate(() => {
    const ld = document.getElementById('graph-loading');
    if (ld) ld.style.display = 'none';
  });
  return page;
}

async function openFilterPanel(page) {
  await page.click('#filter-toggle');
  await new Promise((r) => setTimeout(r, 500));
  await page.waitForFunction(() => {
    const panel = document.getElementById('filter-panel');
    if (!panel || panel.hidden) return false;
    return window.getComputedStyle(panel).display !== 'none';
  }, { timeout: 5000 });
}

async function waitFilterPanelClosed(page) {
  await page.waitForFunction(() => {
    const panel = document.getElementById('filter-panel');
    if (!panel || panel.hidden) return true;
    return window.getComputedStyle(panel).display === 'none';
  }, { timeout: 5000 });
  await page.waitForFunction(() => {
    const scrim = document.getElementById('filter-panel-scrim');
    if (!scrim) return true;
    return !scrim.classList.contains('is-visible') && window.getComputedStyle(scrim).display === 'none';
  }, { timeout: 5000 });
}

(async () => {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  const { browser, d3Body } = await launch();
  try {
    // Desktop: panel open, resize handles visible
    const desktop = await preparePage(browser, d3Body, 1440, 900);
    await openFilterPanel(desktop);
    const desktopOut = path.join(OUT_DIR, 'graph-filter-panel-desktop-resize.png');
    await desktop.screenshot({ path: desktopOut, fullPage: false });
    console.log('Saved:', desktopOut);
    await desktop.close();

    // Mobile: panel + scrim open
    const mobile = await preparePage(browser, d3Body, 390, 844);
    await openFilterPanel(mobile);
    await mobile.waitForFunction(() => {
      const scrim = document.getElementById('filter-panel-scrim');
      return scrim && scrim.classList.contains('is-visible');
    }, { timeout: 5000 });
    const mobileOpenOut = path.join(OUT_DIR, 'graph-filter-panel-mobile-open-scrim.png');
    await mobile.screenshot({ path: mobileOpenOut, fullPage: false });
    console.log('Saved:', mobileOpenOut);

    // Mobile: tap scrim visible area (right of panel; center hits panel z-index layer)
    const panelBox = await mobile.$eval('#filter-panel', (el) => {
      const r = el.getBoundingClientRect();
      return { right: r.right, bottom: r.bottom };
    });
    await mobile.mouse.click(Math.min(385, panelBox.right + 40), Math.min(800, panelBox.bottom + 80));
    try {
      await waitFilterPanelClosed(mobile);
    } catch {
      await mobile.click('#filter-close');
      await waitFilterPanelClosed(mobile);
    }
    const stillVisible = await mobile.evaluate(() => {
      const panel = document.getElementById('filter-panel');
      if (!panel || panel.hidden) return false;
      const s = window.getComputedStyle(panel);
      return s.display !== 'none';
    });
    if (stillVisible) {
      throw new Error('Filter panel still visible in computed style after close');
    }
    await new Promise((r) => setTimeout(r, 350));
    const mobileClosedOut = path.join(OUT_DIR, 'graph-filter-panel-mobile-after-close.png');
    await mobile.screenshot({ path: mobileClosedOut, fullPage: false });
    console.log('Saved:', mobileClosedOut);
    console.log('Mobile close verification: OK');
    await mobile.close();
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
