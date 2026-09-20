// Verify 时序动画按钮在各种交互下不会把图谱藏没或卡死。
// Usage: node scripts/verify_graph_timeline_buttons.cjs [baseUrl] [outDir]
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
    fs.copyFileSync(src, path.join(ART_DIR, name));
  } catch (_err) { /* optional */ }
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

function graphIsVisible2D(vis, baselineShown) {
  const minShown = Math.max(50, Math.floor((baselineShown || vis.total || 0) * 0.5));
  return vis
    && vis.svgDisplay !== 'none'
    && vis.canvas3dHidden !== false
    && !vis.timelineAnimating
    && vis.shown >= minShown
    && vis.simNodes >= minShown
    && vis.peNone < vis.total * 0.5
    && !String(vis.countText).includes('时序');
}

(async () => {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  const browser = await puppeteer.launch({
    executablePath: exe,
    headless: 'new',
    args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--window-size=1440,900'],
    protocolTimeout: 180000,
  });

  const results = [];
  const fail = (name, extra) => {
    results.push({ name, ok: false, extra });
    throw new Error(`${name}: ${extra || ''}`);
  };
  const pass = (name, extra) => {
    results.push({ name, ok: true, extra });
    console.log(`PASS ${name}${extra ? ' — ' + extra : ''}`);
  };

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1 });
    await page.setCacheEnabled(false);
    const pageErrors = [];
    page.on('pageerror', (err) => pageErrors.push(err.message));
    await page.setRequestInterception(true);
    page.on('request', (req) => {
      if (req.url().includes('cdn.jsdelivr.net/npm/d3')) {
        req.respond({ status: 200, contentType: 'application/javascript', body: d3Body });
      } else if (req.url().includes('3d-force-graph')) {
        // 本脚本覆盖按钮/显隐，不把 4360 节点 WebGL 初始化跑到卡住主线程。
        req.abort('failed').catch(() => {});
      } else req.continue();
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
    await sleep(1500);

    await page.click('#physics-toggle');
    await page.waitForSelector('#physics-panel:not([hidden])');
    await page.waitForSelector('#timeline-animate');

    const vis = () => page.evaluate(() => window.__RN_GRAPH2D_DEBUG__.visibility());
    const shot = async (name) => {
      const dest = path.join(OUT_DIR, name);
      await page.screenshot({ path: dest, fullPage: false });
      copyToArtifacts(dest, name);
      return dest;
    };

    const baseline = await vis();
    assert(baseline && baseline.total > 20, 'graph did not load nodes');
    assert(graphIsVisible2D(baseline, baseline.shown), 'baseline graph not visible: ' + JSON.stringify(baseline));
    pass('baseline visible', `shown=${baseline.shown}/${baseline.total}`);
    await shot('graph-timeline-buttons-baseline.png');

    async function enterTimeline() {
      const before = await vis();
      if (before.timelineAnimating) return;
      await page.click('#timeline-animate');
      await page.waitForFunction(() => window.__RN_GRAPH2D_DEBUG__.timelineAnimating(), { timeout: 8000 });
    }
    async function exitTimeline() {
      const before = await vis();
      if (!before.timelineAnimating) return;
      await page.click('#timeline-animate');
      await page.waitForFunction(() => !window.__RN_GRAPH2D_DEBUG__.timelineAnimating(), { timeout: 8000 });
      await sleep(250);
    }
    async function expectVisible(name) {
      const v = await vis();
      if (!graphIsVisible2D(v, baseline.shown)) {
        await shot(`graph-timeline-buttons-FAIL-${name.replace(/\s+/g, '-')}.png`);
        fail(name, JSON.stringify(v));
      }
      pass(name, `shown=${v.shown} sim=${v.simNodes}`);
      return v;
    }

    // 1. Enter then immediately exit (often idx≈0, all opacity 0)
    await enterTimeline();
    await exitTimeline();
    await expectVisible('enter then immediate exit');
    await shot('graph-timeline-buttons-after-immediate-exit.png');

    // 2. Enter, let it play, pause, exit
    await enterTimeline();
    await sleep(1200);
    await page.click('#timeline-playpause');
    await page.waitForFunction(() => {
      const d = window.__RN_GRAPH2D_DEBUG__;
      return d.timelineAnimating() && !d.timelinePlaying();
    }, { timeout: 5000 });
    await exitTimeline();
    await expectVisible('pause then exit');

    // 3. Enter, seek to 0 (empty graph), then exit — classic "graph vanished" path
    await enterTimeline();
    await sleep(800);
    await page.$eval('#timeline-progress', (el) => {
      el.value = '0';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await sleep(200);
    const atZero = await vis();
    if (atZero.timelineAnimating && atZero.shown > 20) {
      fail('seek 0 should hide nodes while still in timeline', JSON.stringify(atZero));
    }
    pass('seek 0 hides nodes in-mode', `shown=${atZero.shown}`);
    await shot('graph-timeline-buttons-seek-zero.png');
    await exitTimeline();
    await expectVisible('seek 0 then exit restores graph');
    await shot('graph-timeline-buttons-after-seek-zero-exit.png');

    // 4. Restart (seek 0) then exit
    await enterTimeline();
    await sleep(800);
    await page.click('#timeline-restart');
    await sleep(200);
    await exitTimeline();
    await expectVisible('restart then exit restores graph');

    // 5. Rapid toggle 时序动画 6 times, end outside timeline
    for (let i = 0; i < 6; i++) {
      await page.click('#timeline-animate');
      await sleep(80);
    }
    await page.waitForFunction(() => !window.__RN_GRAPH2D_DEBUG__.timelineAnimating(), { timeout: 8000 });
    await sleep(250);
    await expectVisible('rapid toggle 6x ends with visible graph');
    await shot('graph-timeline-buttons-after-rapid-toggle.png');

    // 6. Enter, 刷新布局 (teardown + reseed)
    await enterTimeline();
    await sleep(600);
    await page.click('#restart-simulation');
    await page.waitForFunction(() => !window.__RN_GRAPH2D_DEBUG__.timelineAnimating(), { timeout: 8000 });
    await sleep(600);
    await expectVisible('refresh layout during timeline restores graph');
    await shot('graph-timeline-buttons-after-refresh.png');

    // 7. Play-to-end wrap: seek near end, press play (seeks 0), then exit
    await enterTimeline();
    const total = await page.evaluate(() => window.__RN_GRAPH2D_DEBUG__.timelineTotal());
    await page.$eval('#timeline-progress', (el, t) => {
      el.value = String(t);
      el.dispatchEvent(new Event('input', { bubbles: true }));
    }, total);
    await sleep(150);
    await page.click('#timeline-playpause');
    await sleep(400);
    await exitTimeline();
    await expectVisible('play-from-end wrap then exit restores graph');

    // 8. Play/pause spam then exit
    await enterTimeline();
    for (let i = 0; i < 8; i++) {
      await page.click('#timeline-playpause');
      await sleep(40);
    }
    await exitTimeline();
    await expectVisible('play/pause spam then exit restores graph');

    // 9. 2D 时序中点 3D：库加载失败也会走回 2D。必须立刻拆除时序且全图可见。
    await enterTimeline();
    await sleep(400);
    page.once('dialog', (d) => d.dismiss().catch(() => {}));
    await page.click('#view-mode-3d');
    await sleep(800);
    const after3dClick = await vis();
    if (after3dClick.timelineAnimating) {
      fail('clicking 3D during timeline must teardown immediately', JSON.stringify(after3dClick));
    }
    if (!graphIsVisible2D(after3dClick, baseline.shown)) {
      await shot('graph-timeline-buttons-FAIL-3d-click.png');
      fail('3D click/fallback left the 2D graph hidden', JSON.stringify(after3dClick));
    }
    pass('3D click during timeline teardowns and keeps 2D visible',
      `shown=${after3dClick.shown} svg=${after3dClick.svgDisplay}`);
    await shot('graph-timeline-buttons-after-3d-click.png');

    await expectVisible('final 2D graph still visible');
    await shot('graph-timeline-buttons-final-2d.png');

    const realErrors = pageErrors.filter((m) => !/t\.name is not a function/.test(m));
    if (realErrors.length) {
      fail('page errors during button interactions', realErrors.join(' | '));
    }
    pass('no unexpected page errors', pageErrors.length ? `ignored ${pageErrors.length} 3D-abort noise` : '');

    const report = { ok: true, baselineShown: baseline.shown, results };
    console.log(JSON.stringify(report, null, 2));
    fs.writeFileSync(path.join(OUT_DIR, 'graph-timeline-buttons-report.json'), JSON.stringify(report, null, 2));
    copyToArtifacts(
      path.join(OUT_DIR, 'graph-timeline-buttons-report.json'),
      'graph_timeline_buttons_report.json'
    );
    console.log('OK: timeline button interactions keep the graph visible');
  } catch (err) {
    console.error(err);
    console.error('partial results', JSON.stringify(results, null, 2));
    process.exitCode = 1;
  } finally {
    await browser.close();
  }
})();
