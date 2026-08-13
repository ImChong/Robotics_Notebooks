// Verify graph.html「更新时间 Top N」slider: default=all, left=latest subset,
// AND with degree Top N, clear resets both. Screenshots filter panel states.
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

async function preparePage(browser) {
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
    const slider = document.getElementById('sl-recency-top');
    return slider && Number(slider.max) > Number(slider.min);
  }, { timeout: 30000 });
  await page.evaluate(() => {
    const ld = document.getElementById('graph-loading');
    if (ld) ld.style.display = 'none';
  });
  return page;
}

async function openFilter(page) {
  await page.click('#filter-toggle');
  await page.waitForFunction(() => {
    const panel = document.getElementById('filter-panel');
    return panel && !panel.hidden && getComputedStyle(panel).display !== 'none';
  }, { timeout: 5000 });
}

async function readState(page) {
  return page.evaluate(() => {
    const slider = document.getElementById('sl-recency-top');
    const label = document.getElementById('val-recency-top');
    const countEl = document.getElementById('graph-node-count');
    const badge = document.getElementById('filter-count');
    const degSlider = document.getElementById('sl-degree-top');
    const visible = [];
    document.querySelectorAll('#graph-canvas .nodes g').forEach((g) => {
      const op = getComputedStyle(g).opacity;
      const pe = getComputedStyle(g).pointerEvents;
      if (Number(op) > 0.5 && pe !== 'none') {
        const title = g.querySelector('title');
        // fallback: data from __data__ via d3 not available; use circle presence
        visible.push(g.getAttribute('data-id') || g.id || 'node');
      }
    });
    // Prefer the toolbar count text which applyFilters updates
    return {
      min: Number(slider.min),
      max: Number(slider.max),
      value: Number(slider.value),
      label: label ? label.textContent.trim() : '',
      countText: countEl ? countEl.textContent.trim() : '',
      badgeText: badge && badge.style.display !== 'none' ? badge.textContent.trim() : '',
      badgeVisible: !!(badge && badge.style.display !== 'none'),
      degValue: degSlider ? Number(degSlider.value) : null,
      degMax: degSlider ? Number(degSlider.max) : null,
      sectionExists: !!document.getElementById('filter-recency-section'),
      mode: document.getElementById('filter-recency-section')?.getAttribute('data-recency-mode') || '',
    };
  });
}

async function setRecency(page, value) {
  await page.$eval('#sl-recency-top', (el, v) => {
    el.value = String(v);
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
  await new Promise((r) => setTimeout(r, 200));
}

async function setDegree(page, value) {
  await page.$eval('#sl-degree-top', (el, v) => {
    el.value = String(v);
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
  await new Promise((r) => setTimeout(r, 200));
}

function parseVisibleCount(countText) {
  // e.g. "128 / 2072 节点" or "2072 节点"
  const m = countText.match(/(\d+)\s*\/\s*(\d+)/);
  if (m) return { visible: Number(m[1]), total: Number(m[2]) };
  const m2 = countText.match(/(\d+)/);
  if (m2) return { visible: Number(m2[1]), total: Number(m2[1]) };
  return { visible: NaN, total: NaN };
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
    const page = await preparePage(browser);
    await openFilter(page);

    const initial = await readState(page);
    console.log('initial', initial);
    assert(initial.sectionExists, 'filter-recency-section missing');
    assert(initial.mode === 'node', `default recency mode should be node, got ${initial.mode}`);
    assert(initial.value === initial.max, `default should be max/all, got value=${initial.value} max=${initial.max}`);
    assert(initial.label === '全部', `default label should be 全部, got ${initial.label}`);
    assert(!initial.badgeVisible, 'filter badge should be hidden at default');

    const shotAll = path.join(OUT_DIR, 'graph-recency-topn-all.png');
    await page.screenshot({ path: shotAll, fullPage: false });
    copyToArtifacts(shotAll, 'graph-recency-topn-all.png');
    console.log('Saved:', shotAll);

    // Slide to minimum (latest subset)
    await setRecency(page, initial.min);
    const latest = await readState(page);
    console.log('latest', latest);
    assert(latest.value === latest.min, `min value expected ${latest.min}, got ${latest.value}`);
    assert(latest.label === String(latest.min), `label should be ${latest.min}, got ${latest.label}`);
    assert(latest.badgeVisible, 'filter badge should show when recency Top N active');
    const latestCounts = parseVisibleCount(latest.countText);
    assert(latestCounts.visible === latest.min, `visible count should be ${latest.min}, got ${latest.countText}`);
    assert(latestCounts.total === latest.max, `total should be ${latest.max}, got ${latest.countText}`);

    const shotLatest = path.join(OUT_DIR, 'graph-recency-topn-latest.png');
    await page.screenshot({ path: shotLatest, fullPage: false });
    copyToArtifacts(shotLatest, 'graph-recency-topn-latest.png');
    console.log('Saved:', shotLatest);

    // Mid value
    const mid = Math.max(latest.min, Math.round(latest.max * 0.25));
    await setRecency(page, mid);
    const midState = await readState(page);
    const midCounts = parseVisibleCount(midState.countText);
    assert(midCounts.visible === mid, `mid visible should be ${mid}, got ${midState.countText}`);

    // AND with degree Top N — expected size from page-side sets
    const degN = Math.max(initial.min, 20);
    await setDegree(page, degN);
    const expectedAnd = await page.evaluate((recN, degLimit) => {
      // Recompute from node circles' __data__ if present; else trust toolbar only.
      const nodes = [];
      const sel = d3.selectAll('#graph-canvas .node-g');
      if (!sel.empty()) {
        sel.each(function (d) { if (d && d.id) nodes.push(d); });
      }
      if (!nodes.length) return null;
      const byDegree = nodes.slice().sort((a, b) => (b._degree || 0) - (a._degree || 0)).slice(0, degLimit).map((n) => n.id);
      const byRecency = nodes.slice().sort((a, b) => {
        const ta = a._recencyTs != null ? a._recencyTs : -Infinity;
        const tb = b._recencyTs != null ? b._recencyTs : -Infinity;
        if (tb !== ta) return tb - ta;
        return a.id < b.id ? -1 : (a.id > b.id ? 1 : 0);
      }).slice(0, recN).map((n) => n.id);
      const degSet = new Set(byDegree);
      return byRecency.filter((id) => degSet.has(id)).length;
    }, mid, degN);
    const andState = await readState(page);
    const andCounts = parseVisibleCount(andState.countText);
    console.log('and', andState, andCounts, 'expected', expectedAnd);
    assert(andCounts.visible <= Math.min(mid, degN),
      `AND visible ${andCounts.visible} should be <= min(${mid},${degN})`);
    if (expectedAnd != null) {
      assert(andCounts.visible === expectedAnd,
        `AND visible ${andCounts.visible} should equal expected ${expectedAnd}`);
    }

    const shotAnd = path.join(OUT_DIR, 'graph-recency-topn-and-degree.png');
    await page.screenshot({ path: shotAnd, fullPage: false });
    copyToArtifacts(shotAnd, 'graph-recency-topn-and-degree.png');
    console.log('Saved:', shotAnd);

    // Clear resets both
    await page.click('#filter-clear');
    await new Promise((r) => setTimeout(r, 300));
    const cleared = await readState(page);
    console.log('cleared', cleared);
    assert(cleared.value === cleared.max, 'clear should reset recency to max');
    assert(cleared.label === '全部', 'clear label should be 全部');
    assert(cleared.degValue === cleared.degMax, 'clear should reset degree to max');
    assert(!cleared.badgeVisible, 'badge hidden after clear');

    console.log('OK verify_graph_recency_topn');
  } finally {
    await browser.close();
  }
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
