// Verify graph.html community legend is collapsed by default on desktop,
// then expands after clicking the FAB.
// Usage: node scripts/screenshot_graph_legend_collapse_verify.cjs [baseUrl] [outDir]
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
      const fab = document.getElementById('graph-legend-fab');
      const loadingHidden = !loading || loading.hidden
        || loading.style.display === 'none'
        || window.getComputedStyle(loading).display === 'none';
      const countReady = count && count.textContent && !count.textContent.includes('加载中');
      const fabVisible = fab && window.getComputedStyle(fab).display !== 'none'
        && window.getComputedStyle(fab).opacity !== '0';
      return loadingHidden && countReady && fabVisible;
    }, { timeout: 90000 });
    await new Promise((r) => setTimeout(r, 1500));

    const collapsedState = await page.evaluate(() => {
      const fab = document.getElementById('graph-legend-fab');
      const legend = document.getElementById('graph-legend');
      const fabCs = window.getComputedStyle(fab);
      const legendCs = window.getComputedStyle(legend);
      return {
        fabDisplay: fabCs.display,
        fabOpacity: fabCs.opacity,
        fabExpanded: fab.getAttribute('aria-expanded'),
        legendOpen: legend.classList.contains('graph-legend-open'),
        legendTransform: legendCs.transform,
        legendPointerEvents: legendCs.pointerEvents,
      };
    });

    const collapsedPath = path.join(outDir, 'graph-legend-pc-collapsed.png');
    await page.screenshot({ path: collapsedPath, fullPage: false });

    await page.click('#graph-legend-fab');
    await page.waitForFunction(() => {
      const legend = document.getElementById('graph-legend');
      const fab = document.getElementById('graph-legend-fab');
      return legend && legend.classList.contains('graph-legend-open')
        && fab && fab.getAttribute('aria-expanded') === 'true';
    }, { timeout: 5000 });
    await new Promise((r) => setTimeout(r, 400));

    const openState = await page.evaluate(() => {
      const fab = document.getElementById('graph-legend-fab');
      const legend = document.getElementById('graph-legend');
      const scrim = document.getElementById('graph-legend-scrim');
      const title = legend.querySelector('.legend-title');
      return {
        fabExpanded: fab.getAttribute('aria-expanded'),
        legendOpen: legend.classList.contains('graph-legend-open'),
        legendPointerEvents: window.getComputedStyle(legend).pointerEvents,
        scrimVisible: scrim && scrim.classList.contains('is-visible') && !scrim.hidden,
        titleText: title ? title.textContent.trim() : '',
        rowCount: legend.querySelectorAll('.legend-row').length,
      };
    });

    const openPath = path.join(outDir, 'graph-legend-pc-open.png');
    await page.screenshot({ path: openPath, fullPage: false });

    await page.keyboard.press('Escape');
    await page.waitForFunction(() => {
      const legend = document.getElementById('graph-legend');
      const fab = document.getElementById('graph-legend-fab');
      return legend && !legend.classList.contains('graph-legend-open')
        && fab && fab.getAttribute('aria-expanded') === 'false';
    }, { timeout: 5000 });

    const afterEsc = await page.evaluate(() => {
      const fab = document.getElementById('graph-legend-fab');
      const legend = document.getElementById('graph-legend');
      return {
        fabExpanded: fab.getAttribute('aria-expanded'),
        legendOpen: legend.classList.contains('graph-legend-open'),
      };
    });

    const ok = (collapsedState.fabDisplay === 'inline-flex' || collapsedState.fabDisplay === 'flex')
      && collapsedState.fabExpanded === 'false'
      && collapsedState.legendOpen === false
      && collapsedState.legendPointerEvents === 'none'
      && openState.fabExpanded === 'true'
      && openState.legendOpen === true
      && openState.scrimVisible === true
      && openState.rowCount > 0
      && afterEsc.fabExpanded === 'false'
      && afterEsc.legendOpen === false;

    console.log(JSON.stringify({
      ok,
      collapsedState,
      openState,
      afterEsc,
      screenshots: { collapsedPath, openPath },
    }, null, 2));

    if (!ok) process.exitCode = 1;
  } finally {
    await browser.close();
  }
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
