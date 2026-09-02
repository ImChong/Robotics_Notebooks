// Verify graph filter panel: mode tabs + collapsible dimension options with selected count
const puppeteer = require('puppeteer-core');
const path = require('path');
const fs = require('fs');

const OUT_DIR = path.resolve(__dirname, '..', '.cursor-artifacts', 'screenshots');
const ART_DIR = '/opt/cursor/artifacts/screenshots';
const CHROME_CANDIDATES = [
  process.env.CHROME_PATH,
  '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
  '/usr/local/bin/google-chrome',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
].filter(Boolean);
const exe = CHROME_CANDIDATES.find((p) => fs.existsSync(p));
if (!exe) {
  console.error('No Chrome/Chromium found. Set CHROME_PATH.');
  process.exit(1);
}
const d3Body = fs.readFileSync(path.resolve(__dirname, '..', 'node_modules', 'd3', 'dist', 'd3.min.js'));

function copyToArtifacts(src, name) {
  fs.mkdirSync(ART_DIR, { recursive: true });
  const dest = path.join(ART_DIR, name);
  fs.copyFileSync(src, dest);
  return dest;
}

(async () => {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  const browser = await puppeteer.launch({
    executablePath: exe,
    headless: 'new',
    args: ['--no-sandbox', '--disable-gpu', '--ignore-certificate-errors'],
    ignoreHTTPSErrors: true,
  });
  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1 });
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

    await page.click('#filter-toggle');
    await new Promise((r) => setTimeout(r, 500));

    const chromeState = await page.evaluate(() => {
      const tabs = {
        type: !!document.getElementById('filter-mode-type'),
        community: !!document.getElementById('filter-mode-community'),
        opensource: !!document.getElementById('filter-mode-opensource'),
        health: !!document.getElementById('filter-mode-health'),
        communityActive: document.getElementById('filter-mode-community')?.classList.contains('active'),
        order: Array.from(document.querySelectorAll('#filter-mode-tabs .chip')).map((el) => el.id),
      };
      const sections = [
        'filter-dimension-section',
        'filter-depth-section',
        'filter-institution-section',
      ].map((id) => {
        const el = document.getElementById(id);
        return { id, exists: !!el, open: !!(el && el.open) };
      });
      return { tabs, sections };
    });
    console.log('UI state:', JSON.stringify(chromeState, null, 2));
    if (!chromeState.tabs.type || !chromeState.tabs.community || !chromeState.tabs.opensource || !chromeState.tabs.health) {
      throw new Error('Mode tabs missing');
    }
    const expectedOrder = [
      'filter-mode-type',
      'filter-mode-community',
      'filter-mode-opensource',
      'filter-mode-health',
    ];
    if (JSON.stringify(chromeState.tabs.order) !== JSON.stringify(expectedOrder)) {
      throw new Error('Mode tab order mismatch: ' + JSON.stringify(chromeState.tabs.order));
    }
    if (!chromeState.tabs.communityActive) {
      throw new Error('Community mode tab should be active by default');
    }
    if (chromeState.sections.some((s) => !s.exists)) {
      throw new Error('Missing collapsible sections');
    }
    const openSections = chromeState.sections.filter((s) => s.open);
    if (openSections.length !== 1 || openSections[0].id !== 'filter-dimension-section') {
      throw new Error(
        'Default accordion should open only filter-dimension-section, got: ' +
          JSON.stringify(openSections)
      );
    }

    const collapsedOut = path.join(OUT_DIR, 'graph-filter-collapse-default.png');
    await page.screenshot({ path: collapsedOut, fullPage: false });
    copyToArtifacts(collapsedOut, 'graph-filter-collapse-default.png');
    console.log('Saved:', collapsedOut);

    // Dimension section already open by default; select 2 communities, switch to depth
    // (accordion keeps exactly one open), then verify count remains on dimension summary
    await page.evaluate(() => {
      const list = document.getElementById('filter-panel-body');
      const boxes = Array.from((list && list.querySelectorAll('input[type="checkbox"]')) || []).slice(0, 2);
      boxes.forEach((cb) => {
        if (!cb.checked) {
          cb.checked = true;
          cb.dispatchEvent(new Event('change', { bubbles: true }));
        }
      });
    });
    await page.click('#filter-depth-section > summary');
    await new Promise((r) => setTimeout(r, 300));

    let summary = await page.evaluate(() => ({
      label: (document.getElementById('filter-dimension-label') || {}).textContent || '',
      current: (document.getElementById('filter-dimension-current') || {}).textContent || '',
      badge: (document.getElementById('filter-count') || {}).textContent || '',
      subtitle: (document.getElementById('filter-subtitle') || {}).textContent || '',
      dimOpen: !!document.getElementById('filter-dimension-section')?.open,
      depthOpen: !!document.getElementById('filter-depth-section')?.open,
    }));
    console.log('After community select + depth open:', summary);
    if (summary.label !== '按社区') throw new Error('Label should be 按社区: ' + summary.label);
    if (!summary.current.includes('2')) throw new Error('Missing community count: ' + summary.current);
    if (summary.dimOpen || !summary.depthOpen) {
      throw new Error('Accordion should switch open pane to depth');
    }

    const countedOut = path.join(OUT_DIR, 'graph-filter-collapse-counts.png');
    await page.screenshot({ path: countedOut, fullPage: false });
    copyToArtifacts(countedOut, 'graph-filter-collapse-counts.png');
    console.log('Saved:', countedOut);

    // Switch to type via mode button — clears selection (三选一)
    await page.click('#filter-mode-type');
    await new Promise((r) => setTimeout(r, 400));
    summary = await page.evaluate(() => ({
      typeActive: document.getElementById('filter-mode-type')?.classList.contains('active'),
      communityActive: document.getElementById('filter-mode-community')?.classList.contains('active'),
      label: (document.getElementById('filter-dimension-label') || {}).textContent || '',
      current: (document.getElementById('filter-dimension-current') || {}).textContent || '',
      badgeDisplay: document.getElementById('filter-count')
        ? getComputedStyle(document.getElementById('filter-count')).display
        : '',
      subtitle: (document.getElementById('filter-subtitle') || {}).textContent || '',
    }));
    console.log('After switch to type tab:', summary);
    if (!summary.typeActive || summary.communityActive) {
      throw new Error('Type tab should be exclusively active');
    }
    if (summary.label !== '按类型') throw new Error('Label should switch to 按类型');
    if (!summary.current.includes('全部')) throw new Error('Selection should clear on mode switch');
    if (!summary.subtitle.includes('按类型')) throw new Error('Subtitle should mention 按类型');

    // Select 1 type, open section for screenshot
    await page.evaluate(() => {
      const sec = document.getElementById('filter-dimension-section');
      if (sec) sec.open = true;
    });
    await new Promise((r) => setTimeout(r, 200));
    await page.evaluate(() => {
      const list = document.getElementById('filter-panel-body');
      const cb = list && list.querySelector('input[type="checkbox"]');
      if (cb && !cb.checked) {
        cb.checked = true;
        cb.dispatchEvent(new Event('change', { bubbles: true }));
      }
    });
    await new Promise((r) => setTimeout(r, 300));
    summary = await page.evaluate(() => ({
      current: (document.getElementById('filter-dimension-current') || {}).textContent || '',
      badge: (document.getElementById('filter-count') || {}).textContent || '',
    }));
    if (!summary.current.includes('1')) throw new Error('Type count missing: ' + summary.current);

    const openOut = path.join(OUT_DIR, 'graph-filter-collapse-community-open.png');
    await page.screenshot({ path: openOut, fullPage: false });
    copyToArtifacts(openOut, 'graph-filter-collapse-type-open.png');
    // keep legacy filename for PR embeds
    copyToArtifacts(openOut, 'graph-filter-collapse-community-open.png');
    console.log('Saved:', openOut);

    console.log('OK: mode tabs + collapsible options verified');
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
