// Verify graph.html「更新时间 Top N」按节点 / 按日期切换：
// default=按节点+全部；节点子集切到按日期 → 整日保留（可见数 >= N）；
// 切回按节点恢复原 N；按日期滑块按天数计数；清除复位。
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

async function expandRecency(page) {
  const open = await page.$eval('#filter-recency-section', (el) => el.open);
  if (!open) {
    await page.click('#filter-recency-section > summary');
    await new Promise((r) => setTimeout(r, 250));
  }
}

async function readState(page) {
  return page.evaluate(() => {
    const slider = document.getElementById('sl-recency-top');
    const label = document.getElementById('val-recency-top');
    const countEl = document.getElementById('graph-node-count');
    const badge = document.getElementById('filter-count');
    const nodeBtn = document.getElementById('recency-mode-node');
    const dateBtn = document.getElementById('recency-mode-date');
    const maintainedBtn = document.getElementById('recency-include-maintained');
    const hint = document.getElementById('filter-recency-hint');
    const axis = document.getElementById('label-recency-top');
    return {
      min: Number(slider.min),
      max: Number(slider.max),
      value: Number(slider.value),
      label: label ? label.textContent.trim() : '',
      summary: document.getElementById('filter-recency-current')?.textContent.trim() || '',
      countText: countEl ? countEl.textContent.trim() : '',
      badgeVisible: !!(badge && badge.style.display !== 'none'),
      mode: document.getElementById('filter-recency-section')?.getAttribute('data-recency-mode') || '',
      kind: document.getElementById('filter-recency-section')?.getAttribute('data-recency-kind') || '',
      nodeActive: !!(nodeBtn && nodeBtn.classList.contains('is-active')),
      dateActive: !!(dateBtn && dateBtn.classList.contains('is-active')),
      dateDisabled: !!(dateBtn && dateBtn.disabled),
      maintainedActive: !!(maintainedBtn && maintainedBtn.classList.contains('is-active')),
      hint: hint ? hint.textContent.trim() : '',
      axis: axis ? axis.textContent.trim() : '',
    };
  });
}

async function setRecency(page, value) {
  await page.$eval('#sl-recency-top', (el, v) => {
    el.value = String(v);
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
  await new Promise((r) => setTimeout(r, 250));
}

async function clickMode(page, mode) {
  const id = mode === 'date' ? '#recency-mode-date' : '#recency-mode-node';
  await page.click(id);
  await new Promise((r) => setTimeout(r, 250));
}

function parseVisibleCount(countText) {
  const m = countText.match(/(\d+)\s*\/\s*(\d+)/);
  if (m) return { visible: Number(m[1]), total: Number(m[2]) };
  const m2 = countText.match(/(\d+)/);
  if (m2) return { visible: Number(m2[1]), total: Number(m2[1]) };
  return { visible: NaN, total: NaN };
}

async function readGraphNodes(page) {
  return page.evaluate(() => {
    const nodes = [];
    const sel = d3.selectAll('#graph-canvas .node-g');
    if (sel.empty()) return nodes;
    sel.each(function (d) {
      if (d && d.id) {
        nodes.push({
          id: d.id,
          recencyDate: d._addedDate || null,
          recencyTs: d._addedTs == null ? null : d._addedTs,
          activityDate: d._activityDate || null,
          activityTs: d._activityTs == null ? null : d._activityTs,
        });
      }
    });
    return nodes;
  });
}

function sortByRecency(nodes) {
  return nodes.slice().sort((a, b) => {
    const ta = a.recencyTs != null ? a.recencyTs : -Infinity;
    const tb = b.recencyTs != null ? b.recencyTs : -Infinity;
    if (tb !== ta) return tb - ta;
    return a.id < b.id ? -1 : (a.id > b.id ? 1 : 0);
  });
}

function uniqueDatesDesc(nodes) {
  return Array.from(new Set(nodes.map((n) => n.recencyDate).filter(Boolean)))
    .sort((a, b) => (a < b ? 1 : a > b ? -1 : 0));
}

function countForDates(nodes, dates) {
  const keep = new Set(dates);
  return nodes.filter((n) => keep.has(n.recencyDate)).length;
}

function countForActivityDates(nodes, dates) {
  const keep = new Set(dates);
  return nodes.filter((n) => keep.has(n.activityDate)).length;
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
    await expandRecency(page);

    const initial = await readState(page);
    console.log('initial', initial);
    assert(initial.mode === 'node' && initial.nodeActive && !initial.dateActive,
      `default should be 按节点, got mode=${initial.mode}`);
    assert(initial.kind === 'added' && !initial.maintainedActive,
      `default should be 仅新增, got kind=${initial.kind}`);
    assert(initial.label === '全部', `default label should be 全部, got ${initial.label}`);
    assert(initial.axis === '最新节点数', `default axis should be 最新节点数, got ${initial.axis}`);
    assert(initial.value === initial.max, 'default slider should be max/all');
    assert(!initial.badgeVisible, 'badge hidden at default');
    assert(!initial.dateDisabled, '按日期 should be enabled when activity dates exist');

    const shotDefault = path.join(OUT_DIR, 'graph-recency-date-mode-default.png');
    await page.screenshot({ path: shotDefault, fullPage: false });
    copyToArtifacts(shotDefault, 'graph-recency-date-mode-default.png');
    console.log('Saved:', shotDefault);

    const nodes = await readGraphNodes(page);
    assert(nodes.length > 0, 'could not read graph node data');
    const dates = uniqueDatesDesc(nodes);
    assert(dates.length > 0, 'no recency dates on nodes');

    await setRecency(page, initial.min);
    const nodeSubset = await readState(page);
    const nodeCounts = parseVisibleCount(nodeSubset.countText);
    console.log('node subset', nodeSubset, nodeCounts);
    assert(nodeSubset.mode === 'node', 'still node mode after sliding');
    assert(nodeCounts.visible === nodeSubset.min,
      `node Top N visible should be ${nodeSubset.min}, got ${nodeSubset.countText}`);

    const topNodes = sortByRecency(nodes).slice(0, nodeSubset.min);
    const datesInTop = uniqueDatesDesc(topNodes);
    const coversAllDates = datesInTop.length >= dates.length;
    const expectedDateExpand = coversAllDates ? nodes.length : countForDates(nodes, datesInTop);
    assert(expectedDateExpand >= nodeSubset.min, 'date expand should keep at least the node subset');

    await clickMode(page, 'date');
    const dateFromNode = await readState(page);
    const dateFromNodeCounts = parseVisibleCount(dateFromNode.countText);
    console.log('date from node', dateFromNode, dateFromNodeCounts, 'expected', expectedDateExpand, 'dates', datesInTop);
    assert(dateFromNode.mode === 'date' && dateFromNode.dateActive,
      `should switch to 按日期, got mode=${dateFromNode.mode}`);
    assert(dateFromNode.axis === '最新天数', `axis should be 最新天数, got ${dateFromNode.axis}`);
    assert(dateFromNode.max === dates.length,
      `date slider max should be unique dates ${dates.length}, got ${dateFromNode.max}`);
    assert(dateFromNode.value === datesInTop.length,
      `date limit should match unique dates in node subset (${datesInTop.length}), got ${dateFromNode.value}`);
    if (coversAllDates) {
      assert(dateFromNode.label === '全部', `full-date expand label should be 全部, got ${dateFromNode.label}`);
    } else {
      assert(dateFromNode.label === `${datesInTop.length} 日`,
        `date label should be ${datesInTop.length} 日, got ${dateFromNode.label}`);
    }
    assert(dateFromNodeCounts.visible === expectedDateExpand,
      `date-expand visible ${dateFromNodeCounts.visible} should equal ${expectedDateExpand}`);
    assert(dateFromNodeCounts.visible >= nodeSubset.min,
      '按日期区分保留 should not drop nodes vs the node subset');
    assert(dateFromNode.hint.includes('新增日'), `date hint should mention 新增日, got ${dateFromNode.hint}`);

    const shotDateExpand = path.join(OUT_DIR, 'graph-recency-date-mode-expand.png');
    await page.screenshot({ path: shotDateExpand, fullPage: false });
    copyToArtifacts(shotDateExpand, 'graph-recency-date-mode-expand.png');
    console.log('Saved:', shotDateExpand);

    await clickMode(page, 'node');
    const backToNode = await readState(page);
    const backCounts = parseVisibleCount(backToNode.countText);
    console.log('back to node', backToNode, backCounts);
    assert(backToNode.mode === 'node', 'switching back should restore 按节点');
    assert(backToNode.value === nodeSubset.min, `node slider should restore ${nodeSubset.min}, got ${backToNode.value}`);
    assert(backCounts.visible === nodeSubset.min,
      `visible should restore to ${nodeSubset.min}, got ${backToNode.countText}`);
    assert(backToNode.axis === '最新节点数', 'axis restored');

    await clickMode(page, 'date');
    await setRecency(page, 1);
    const oneDay = await readState(page);
    const oneDayCounts = parseVisibleCount(oneDay.countText);
    const expectedOneDay = dates.length <= 1 ? nodes.length : countForDates(nodes, dates.slice(0, 1));
    console.log('one day', oneDay, oneDayCounts, 'expected', expectedOneDay);
    assert(oneDay.mode === 'date', 'still date mode after sliding');
    assert(oneDay.value === 1, `date slider min should be 1, got ${oneDay.value}`);
    if (dates.length <= 1) {
      assert(oneDay.label === '全部', `single-date graph label should be 全部, got ${oneDay.label}`);
    } else {
      assert(oneDay.label === '1 日', `label should be 1 日, got ${oneDay.label}`);
      assert(oneDay.badgeVisible, 'badge should show when date Top N active');
    }
    assert(oneDayCounts.visible === expectedOneDay,
      `1-day visible ${oneDayCounts.visible} should equal ${expectedOneDay}`);

    const shotOneDay = path.join(OUT_DIR, 'graph-recency-date-mode-one-day.png');
    await page.screenshot({ path: shotOneDay, fullPage: false });
    copyToArtifacts(shotOneDay, 'graph-recency-date-mode-one-day.png');
    console.log('Saved:', shotOneDay);

    const activityDates = uniqueDatesDesc(nodes.map((n) => ({ recencyDate: n.activityDate })));
    const expectedMaintainedOneDay = activityDates.length <= 1
      ? nodes.length
      : countForActivityDates(nodes, activityDates.slice(0, 1));
    await page.click('#recency-include-maintained');
    await new Promise((r) => setTimeout(r, 250));
    const withMaintained = await readState(page);
    const withMaintainedCounts = parseVisibleCount(withMaintained.countText);
    console.log('with maintained', withMaintained, withMaintainedCounts, 'expected', expectedMaintainedOneDay);
    assert(withMaintained.kind === 'all' && withMaintained.maintainedActive,
      `显示维护更新 should activate, got kind=${withMaintained.kind}`);
    assert(withMaintained.mode === 'date', 'date mode should persist');
    assert(withMaintained.value === 1, `maintained date slider should stay at 1, got ${withMaintained.value}`);
    assert(withMaintainedCounts.visible === expectedMaintainedOneDay,
      `maintained 1-day visible ${withMaintainedCounts.visible} should equal ${expectedMaintainedOneDay}`);
    assert(withMaintained.hint.includes('含维护'), `hint should mention 含维护, got ${withMaintained.hint}`);

    const shotMaintained = path.join(OUT_DIR, 'graph-recency-date-mode-maintained.png');
    await page.screenshot({ path: shotMaintained, fullPage: false });
    copyToArtifacts(shotMaintained, 'graph-recency-date-mode-maintained.png');
    console.log('Saved:', shotMaintained);

    await page.click('#recency-include-maintained');
    await new Promise((r) => setTimeout(r, 250));
    const backAdded = await readState(page);
    const backAddedCounts = parseVisibleCount(backAdded.countText);
    assert(backAdded.kind === 'added' && !backAdded.maintainedActive, 'toggle off should restore 仅新增');
    assert(backAddedCounts.visible === expectedOneDay,
      `toggle off visible ${backAddedCounts.visible} should restore added-only ${expectedOneDay}`);

    await page.click('#filter-clear');
    await new Promise((r) => setTimeout(r, 300));
    await expandRecency(page);
    const cleared = await readState(page);
    console.log('cleared', cleared);
    assert(cleared.mode === 'node', 'clear should reset to 按节点');
    assert(cleared.kind === 'added' && !cleared.maintainedActive, 'clear should reset to 仅新增');
    assert(cleared.value === cleared.max, 'clear should reset recency to max');
    assert(cleared.label === '全部', 'clear label should be 全部');
    assert(cleared.axis === '最新节点数', 'clear restores node axis');
    assert(!cleared.badgeVisible, 'badge hidden after clear');

    console.log('OK verify_graph_recency_date_mode');
  } finally {
    await browser.close();
  }
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
