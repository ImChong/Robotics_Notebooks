// Verify graph filter Top N sections are collapsible <details>, default closed,
// summary shows current value (全部 / N), and the two are mutually exclusive
// (only one open at a time; both may be closed). Independent of the 三区 accordion.
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
const d3Candidates = [
  path.resolve(__dirname, '..', 'node_modules', 'd3', 'dist', 'd3.min.js'),
  path.resolve(__dirname, '..', 'docs', 'vendor', 'd3.min.js'),
];
const d3Path = d3Candidates.find((p) => fs.existsSync(p));
if (!d3Path) {
  console.error('No d3.min.js found.');
  process.exit(1);
}
const d3Body = fs.readFileSync(d3Path);

function copyToArtifacts(src, name) {
  try {
    fs.mkdirSync(ART_DIR, { recursive: true });
    const dest = path.join(ART_DIR, name);
    fs.copyFileSync(src, dest);
    return dest;
  } catch (_) {
    return null;
  }
}

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

async function readTopN(page) {
  return page.evaluate(() => {
    const deg = document.getElementById('filter-degree-section');
    const rec = document.getElementById('filter-recency-section');
    const dim = document.getElementById('filter-dimension-section');
    return {
      degIsDetails: deg && deg.tagName === 'DETAILS',
      recIsDetails: rec && rec.tagName === 'DETAILS',
      degOpen: !!(deg && deg.open),
      recOpen: !!(rec && rec.open),
      degSummary: document.getElementById('filter-degree-current')?.textContent.trim() || '',
      recSummary: document.getElementById('filter-recency-current')?.textContent.trim() || '',
      degVal: document.getElementById('val-degree-top')?.textContent.trim() || '',
      recVal: document.getElementById('val-recency-top')?.textContent.trim() || '',
      degHeight: deg ? deg.getBoundingClientRect().height : 0,
      recHeight: rec ? rec.getBoundingClientRect().height : 0,
      degSummaryH: deg?.querySelector('summary')?.getBoundingClientRect().height || 0,
      recSummaryH: rec?.querySelector('summary')?.getBoundingClientRect().height || 0,
      dimOpen: !!(dim && dim.open),
    };
  });
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
    }, { timeout: 60000 });
    await page.waitForFunction(() => {
      const slider = document.getElementById('sl-degree-top');
      return slider && Number(slider.max) > Number(slider.min);
    }, { timeout: 30000 });
    await page.evaluate(() => {
      const ld = document.getElementById('graph-loading');
      if (ld) ld.style.display = 'none';
    });

    await page.click('#filter-toggle');
    await page.waitForFunction(() => {
      const panel = document.getElementById('filter-panel');
      return panel && !panel.hidden;
    }, { timeout: 5000 });
    await new Promise((r) => setTimeout(r, 400));

    let state = await readTopN(page);
    console.log('default', state);
    assert(state.degIsDetails && state.recIsDetails, 'Top N sections should be <details>');
    assert(!state.degOpen && !state.recOpen, 'Top N should default collapsed');
    assert(state.degSummary === '全部' && state.recSummary === '全部', 'summary should show 全部');
    assert(state.degHeight <= state.degSummaryH + 8, `degree collapsed too tall: ${state.degHeight}`);
    assert(state.recHeight <= state.recSummaryH + 8, `recency collapsed too tall: ${state.recHeight}`);
    assert(state.dimOpen, 'dimension accordion default open should be preserved');

    const collapsedOut = path.join(OUT_DIR, 'graph-filter-topn-collapsed.png');
    await page.screenshot({ path: collapsedOut, fullPage: false });
    copyToArtifacts(collapsedOut, 'graph-filter-topn-collapsed.png');
    console.log('Saved:', collapsedOut);

    // Expand 连接数 Top N
    await page.click('#filter-degree-section > summary');
    await new Promise((r) => setTimeout(r, 250));
    state = await readTopN(page);
    assert(state.degOpen && !state.recOpen, 'only degree should open');
    assert(state.degHeight > state.degSummaryH + 20, 'degree open should show slider');
    assert(state.dimOpen, 'opening Top N must not close dimension accordion');

    const degOpenOut = path.join(OUT_DIR, 'graph-filter-topn-degree-open.png');
    await page.screenshot({ path: degOpenOut, fullPage: false });
    copyToArtifacts(degOpenOut, 'graph-filter-topn-degree-open.png');
    console.log('Saved:', degOpenOut);

    // Expand 更新时间 Top N → 连接数应收起（二选一）
    await page.click('#filter-recency-section > summary');
    await new Promise((r) => setTimeout(r, 250));
    state = await readTopN(page);
    assert(!state.degOpen && state.recOpen, 'Top N should be exclusive: only recency open');
    assert(state.dimOpen, 'accordion still open');

    const recOpenOut = path.join(OUT_DIR, 'graph-filter-topn-recency-open.png');
    await page.screenshot({ path: recOpenOut, fullPage: false });
    copyToArtifacts(recOpenOut, 'graph-filter-topn-recency-open.png');
    console.log('Saved:', recOpenOut);

    // Switch back to degree → recency closes
    await page.click('#filter-degree-section > summary');
    await new Promise((r) => setTimeout(r, 250));
    state = await readTopN(page);
    assert(state.degOpen && !state.recOpen, 'switching to degree should close recency');

    // Change slider → summary updates while open
    await page.$eval('#sl-degree-top', (el) => {
      el.value = el.min;
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await new Promise((r) => setTimeout(r, 200));
    state = await readTopN(page);
    const minLabel = await page.$eval('#sl-degree-top', (el) => String(el.min));
    assert(state.degSummary === minLabel, `degree summary should be ${minLabel}, got ${state.degSummary}`);
    assert(state.degVal === minLabel, `degree slider label should be ${minLabel}`);

    const exclusiveOut = path.join(OUT_DIR, 'graph-filter-topn-exclusive.png');
    await page.screenshot({ path: exclusiveOut, fullPage: false });
    copyToArtifacts(exclusiveOut, 'graph-filter-topn-exclusive.png');
    console.log('Saved:', exclusiveOut);

    // Collapse degree — both may be closed
    await page.click('#filter-degree-section > summary');
    await new Promise((r) => setTimeout(r, 200));
    state = await readTopN(page);
    assert(!state.degOpen && !state.recOpen, 'both Top N may collapse');
    assert(state.degSummary === minLabel, 'collapsed summary keeps active value');

    console.log('OK: Top N collapse default + exclusive toggle verified');
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
