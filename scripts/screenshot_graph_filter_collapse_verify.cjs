// Verify graph filter panel: type/community/health collapsible sections + selected counts
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

    // 1) Default collapsed: five sections present, all closed
    const collapsedState = await page.evaluate(() => {
      const ids = [
        'filter-type-section',
        'filter-community-section',
        'filter-health-section',
        'filter-depth-section',
        'filter-institution-section',
      ];
      return ids.map((id) => {
        const el = document.getElementById(id);
        return { id, exists: !!el, open: !!(el && el.open) };
      });
    });
    console.log('Collapsed state:', JSON.stringify(collapsedState, null, 2));
    if (collapsedState.some((s) => !s.exists)) {
      throw new Error('Missing filter sections');
    }
    if (collapsedState.some((s) => s.open)) {
      console.warn('Warning: some sections open by default');
    }

    const collapsedOut = path.join(OUT_DIR, 'graph-filter-collapse-default.png');
    await page.screenshot({ path: collapsedOut, fullPage: false });
    copyToArtifacts(collapsedOut, 'graph-filter-collapse-default.png');
    console.log('Saved:', collapsedOut);

    // 2) Open community, select 2 checkboxes, collapse, verify count badge text
    await page.evaluate(() => {
      const sec = document.getElementById('filter-community-section');
      if (sec) sec.open = true;
    });
    await new Promise((r) => setTimeout(r, 300));
    const selected = await page.evaluate(() => {
      const list = document.getElementById('filter-community-list');
      if (!list) return 0;
      const boxes = Array.from(list.querySelectorAll('input[type="checkbox"]')).slice(0, 2);
      boxes.forEach((cb) => {
        if (!cb.checked) {
          cb.checked = true;
          cb.dispatchEvent(new Event('change', { bubbles: true }));
        }
      });
      return boxes.length;
    });
    await new Promise((r) => setTimeout(r, 600));
    await page.evaluate(() => {
      const sec = document.getElementById('filter-community-section');
      if (sec) sec.open = false;
    });
    await new Promise((r) => setTimeout(r, 200));

    const communitySummary = await page.evaluate(() => {
      const cur = document.getElementById('filter-community-current');
      return cur ? cur.textContent.trim() : '';
    });
    console.log('Community summary after select', selected, ':', communitySummary);
    if (!communitySummary.includes(String(selected))) {
      throw new Error('Community collapse header missing selected count: ' + communitySummary);
    }

    // 3) 三选一：切换到类型并勾选 1 项后，社区数量应被清空回「全部」
    await page.evaluate(() => {
      const sec = document.getElementById('filter-type-section');
      if (sec) sec.open = true;
    });
    await new Promise((r) => setTimeout(r, 300));
    const exclusiveState = await page.evaluate(() => {
      const list = document.getElementById('filter-type-list');
      const cb = list && list.querySelector('input[type="checkbox"]');
      if (cb && !cb.checked) {
        cb.checked = true;
        cb.dispatchEvent(new Event('change', { bubbles: true }));
      }
      return {
        typeOpen: !!(document.getElementById('filter-type-section') || {}).open,
        communityOpen: !!(document.getElementById('filter-community-section') || {}).open,
        healthOpen: !!(document.getElementById('filter-health-section') || {}).open,
        type: (document.getElementById('filter-type-current') || {}).textContent || '',
        community: (document.getElementById('filter-community-current') || {}).textContent || '',
        health: (document.getElementById('filter-health-current') || {}).textContent || '',
        badge: (document.getElementById('filter-count') || {}).textContent || '',
        subtitle: (document.getElementById('filter-subtitle') || {}).textContent || '',
      };
    });
    console.log('After switch to type:', exclusiveState);
    if (!exclusiveState.type.includes('1')) {
      throw new Error('Type collapse header missing count: ' + exclusiveState.type);
    }
    if (!exclusiveState.community.includes('全部')) {
      throw new Error('Community should clear on exclusive switch: ' + exclusiveState.community);
    }
    if (!exclusiveState.health.includes('全部')) {
      throw new Error('Health should remain 全部: ' + exclusiveState.health);
    }
    if (exclusiveState.communityOpen || exclusiveState.healthOpen) {
      throw new Error('Other dimension sections should auto-collapse');
    }
    if (!exclusiveState.subtitle.includes('按类型')) {
      throw new Error('Subtitle should switch to 按类型: ' + exclusiveState.subtitle);
    }
    if (exclusiveState.badge !== '1') {
      throw new Error('Filter badge should be 1 after exclusive switch, got: ' + exclusiveState.badge);
    }

    await page.evaluate(() => {
      const sec = document.getElementById('filter-type-section');
      if (sec) sec.open = false;
    });
    await new Promise((r) => setTimeout(r, 200));

    const countedOut = path.join(OUT_DIR, 'graph-filter-collapse-counts.png');
    await page.screenshot({ path: countedOut, fullPage: false });
    copyToArtifacts(countedOut, 'graph-filter-collapse-counts.png');
    console.log('Saved:', countedOut);

    // 4) 再切回社区并勾选 2 项，类型应清空
    await page.evaluate(() => {
      const sec = document.getElementById('filter-community-section');
      if (sec) sec.open = true;
    });
    await new Promise((r) => setTimeout(r, 300));
    await page.evaluate(() => {
      const list = document.getElementById('filter-community-list');
      const boxes = Array.from((list && list.querySelectorAll('input[type="checkbox"]')) || []).slice(0, 2);
      boxes.forEach((cb) => {
        if (!cb.checked) {
          cb.checked = true;
          cb.dispatchEvent(new Event('change', { bubbles: true }));
        }
      });
    });
    await new Promise((r) => setTimeout(r, 400));
    const backToCommunity = await page.evaluate(() => ({
      type: (document.getElementById('filter-type-current') || {}).textContent || '',
      community: (document.getElementById('filter-community-current') || {}).textContent || '',
      typeOpen: !!(document.getElementById('filter-type-section') || {}).open,
      communityOpen: !!(document.getElementById('filter-community-section') || {}).open,
      badge: (document.getElementById('filter-count') || {}).textContent || '',
    }));
    console.log('Back to community:', backToCommunity);
    if (!backToCommunity.community.includes('2')) {
      throw new Error('Community count missing after reselect: ' + backToCommunity.community);
    }
    if (!backToCommunity.type.includes('全部')) {
      throw new Error('Type should clear when switching back to community: ' + backToCommunity.type);
    }
    if (backToCommunity.typeOpen) {
      throw new Error('Type section should be closed while community is active');
    }

    const openOut = path.join(OUT_DIR, 'graph-filter-collapse-community-open.png');
    await page.screenshot({ path: openOut, fullPage: false });
    copyToArtifacts(openOut, 'graph-filter-collapse-community-open.png');
    console.log('Saved:', openOut);

    console.log('OK: filter collapse + exclusive counts verified');
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
