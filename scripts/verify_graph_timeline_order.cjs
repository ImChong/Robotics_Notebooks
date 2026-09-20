// Verify graph.html 时序动画：从最老 added 日放到最新，而不是最新放到最老。
// Usage: node scripts/verify_graph_timeline_order.cjs [baseUrl] [outDir]
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');

const OUT_DIR = path.resolve(
  process.argv[3] || path.join(__dirname, '..', '.cursor-artifacts', 'screenshots')
);
const ART_DIR = '/opt/cursor/artifacts';
const CHROME_CANDIDATES = [
  process.env.CHROME_PATH,
  process.env.PUPPETEER_EXECUTABLE_PATH,
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
  console.error('No d3.min.js found in node_modules or docs/vendor.');
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

function isNonDecreasing(dates) {
  for (let i = 1; i < dates.length; i++) {
    if (dates[i] < dates[i - 1]) return false;
  }
  return true;
}

(async () => {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  const browser = await puppeteer.launch({
    executablePath: exe,
    headless: 'new',
    args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--window-size=1440,900'],
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1 });
    await page.setCacheEnabled(false);
    await page.setRequestInterception(true);
    page.on('request', (req) => {
      if (req.url().includes('cdn.jsdelivr.net/npm/d3')) {
        req.respond({ status: 200, contentType: 'application/javascript', body: d3Body });
      } else {
        req.continue();
      }
    });

    const baseUrl = process.argv[2] || process.env.GRAPH_BASE_URL || 'http://127.0.0.1:8765/graph.html';
    await page.goto(baseUrl, { waitUntil: 'domcontentloaded', timeout: 45000 });
    await page.waitForFunction(() => {
      const loading = document.getElementById('graph-loading');
      const count = document.getElementById('graph-node-count');
      const loadingHidden = !loading || loading.hidden || loading.classList.contains('is-hidden')
        || window.getComputedStyle(loading).display === 'none';
      return loadingHidden && count && count.textContent && !count.textContent.includes('加载中');
    }, { timeout: 120000 });

    await page.click('#physics-toggle');
    await page.waitForSelector('#physics-panel:not([hidden])');
    await page.waitForSelector('#timeline-animate');

    const preview = await page.evaluate(() => window.__RN_GRAPH2D_DEBUG__.timelineOrderPreview());
    assert(Array.isArray(preview) && preview.length > 20, 'timeline preview too short: ' + preview.length);

    const dated = preview.filter((n) => n.added);
    assert(dated.length > 20, 'not enough added dates on timeline preview');
    const dates = dated.map((n) => n.added);
    assert(isNonDecreasing(dates), 'timeline preview is not oldest→newest by added date');

    const first = dated[0];
    const last = dated[dated.length - 1];
    assert(first.added < last.added, `expected first ${first.added} < last ${last.added}`);

    const newestFirst = dates[0] > dates[dates.length - 1];
    assert(!newestFirst, 'timeline still plays newest→oldest');

    await page.click('#timeline-animate');
    await page.waitForFunction(() => {
      const dbg = window.__RN_GRAPH2D_DEBUG__;
      return dbg && dbg.timelineAnimating() && dbg.timelineIdx() > 0;
    }, { timeout: 10000 });

    await new Promise((r) => setTimeout(r, 1600));
    await page.click('#timeline-playpause');
    await page.waitForFunction(() => {
      const dbg = window.__RN_GRAPH2D_DEBUG__;
      return dbg && dbg.timelineAnimating() && !dbg.timelinePlaying();
    }, { timeout: 5000 });

    const mid = await page.evaluate(() => {
      const dbg = window.__RN_GRAPH2D_DEBUG__;
      const order = dbg.timelineOrderPreview();
      const idx = dbg.timelineIdx();
      const revealed = order.slice(0, idx);
      const countText = document.getElementById('graph-node-count')?.textContent || '';
      const dateText = document.getElementById('timeline-scrub-date')?.textContent || '';
      return {
        idx,
        total: dbg.timelineTotal(),
        firstAdded: revealed[0] && revealed[0].added,
        lastAdded: revealed[revealed.length - 1] && revealed[revealed.length - 1].added,
        firstId: revealed[0] && revealed[0].id,
        lastId: revealed[revealed.length - 1] && revealed[revealed.length - 1].id,
        countText,
        dateText,
        datesNonDecreasing: revealed.every((n, i, arr) => i === 0 || !n.added || !arr[i - 1].added || n.added >= arr[i - 1].added),
      };
    });

    assert(mid.idx > 0, 'timeline did not reveal any nodes');
    assert(mid.datesNonDecreasing, 'revealed prefix is not oldest→newest');
    assert(mid.firstAdded, 'first revealed node missing added date');
    assert(mid.countText.includes(mid.dateText) || mid.dateText === mid.lastAdded || mid.dateText === '起点',
      'toolbar/scrub date missing: ' + JSON.stringify(mid));

    const startShot = path.join(OUT_DIR, 'graph-timeline-oldest-first.png');
    await page.screenshot({ path: startShot, fullPage: false });
    copyToArtifacts(startShot, 'graph-timeline-oldest-first.png');

    const report = {
      firstId: first.id,
      firstAdded: first.added,
      lastId: last.id,
      lastAdded: last.added,
      mid,
      startShot,
    };
    console.log(JSON.stringify(report, null, 2));
    console.log('OK: timeline plays oldest → newest by added date');
  } finally {
    await browser.close();
  }
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
