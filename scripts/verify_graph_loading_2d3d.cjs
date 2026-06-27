// Verify PR #830: loading overlay dismisses on 2D/3D ready; 2D↔3D switch works.
// Usage: node scripts/verify_graph_loading_2d3d.cjs [baseUrl] [outDir] [--headed]
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');

const headed = process.argv.includes('--headed');
const args = process.argv.filter((a) => a !== '--headed');
const baseUrl = args[2] || 'http://127.0.0.1:8765/graph.html';
const outDir = path.resolve(args[3] || path.join(__dirname, '..', '.cursor-artifacts', 'screenshots'));

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function openPhysicsPanel(page) {
  await page.evaluate(() => {
    const panel = document.getElementById('physics-panel');
    if (panel && panel.hidden) {
      document.getElementById('physics-toggle')?.click();
    }
  });
  await sleep(300);
}

async function clickViewMode(page, mode) {
  await openPhysicsPanel(page);
  const sel = mode === '3d' ? '#view-mode-3d' : '#view-mode-2d';
  await page.waitForSelector(sel, { visible: true, timeout: 10000 });
  const btn = await page.$(sel);
  if (!btn) throw new Error(`missing ${sel}`);
  await btn.click();
  await sleep(400);
}

function is3dViewActive(view) {
  return view.has3dCanvas && view.svgDisplay === 'none';
}

function is2dViewActive(view) {
  return view.c3dHidden && view.svgDisplay !== 'none';
}

function loadingState(page) {
  return page.evaluate(() => {
    const el = document.getElementById('graph-loading');
    if (!el) return { exists: false };
    const cs = window.getComputedStyle(el);
    return {
      exists: true,
      hidden: el.hidden,
      display: cs.display,
      opacity: cs.opacity,
      hasIsHidden: el.classList.contains('is-hidden'),
      pct: document.getElementById('graph-loading-pct')?.textContent || '',
      title: document.getElementById('graph-loading-title')?.textContent || '',
      isError: el.classList.contains('is-error'),
    };
  });
}

function viewState(page) {
  return page.evaluate(() => {
    const svg = document.getElementById('graph-canvas');
    const c3d = document.getElementById('graph-canvas-3d');
    const btn2d = document.getElementById('view-mode-2d');
    const btn3d = document.getElementById('view-mode-3d');
    const svgDisplay = svg ? window.getComputedStyle(svg).display : 'none';
    const c3dHidden = c3d ? c3d.hidden : true;
    const nodes2d = document.querySelectorAll('#graph-canvas .node-circle').length;
    return {
      svgDisplay,
      c3dHidden,
      btn2dActive: btn2d?.classList.contains('is-active'),
      btn3dActive: btn3d?.classList.contains('is-active'),
      nodes2d,
      has3dCanvas: !!c3d && !c3dHidden,
    };
  });
}

function isLoadingDismissed(st) {
  return !st.exists || st.hidden || st.display === 'none' || st.hasIsHidden;
}

(async () => {
  fs.mkdirSync(outDir, { recursive: true });
  const exe = process.env.PUPPETEER_EXECUTABLE_PATH
    || (fs.existsSync('/usr/local/bin/google-chrome') ? '/usr/local/bin/google-chrome' : 'google-chrome');
  const d3Local = path.resolve(__dirname, '..', 'node_modules', 'd3', 'dist', 'd3.min.js');
  const d3Body = fs.existsSync(d3Local) ? fs.readFileSync(d3Local) : null;

  const browser = await puppeteer.launch({
    executablePath: exe,
    headless: headed ? false : 'new',
    slowMo: headed ? 80 : 0,
    args: [
      '--no-sandbox',
      '--disable-gpu',
      '--disable-dev-shm-usage',
      '--window-size=1440,900',
      ...(headed ? [] : []),
    ],
    defaultViewport: headed ? null : { width: 1440, height: 900, deviceScaleFactor: 1 },
  });

  const report = { cases: [], ok: true };

  try {
    const page = await browser.newPage();
    if (!headed) {
      await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1 });
    }

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

    // Case 1: 2D initial load — loading visible then dismissed
    await page.goto(baseUrl, { waitUntil: 'domcontentloaded', timeout: 60000 });
    const earlyLoad = await loadingState(page);
    await page.waitForFunction(() => {
      const el = document.getElementById('graph-loading');
      if (!el) return true;
      return el.hidden || el.classList.contains('is-hidden');
    }, { timeout: 90000 });
    await sleep(800);
    const after2dLoad = await loadingState(page);
    const view2d = await viewState(page);
    const simStillRunning = await page.evaluate(() => {
      const circles = document.querySelectorAll('#graph-canvas .node-circle');
      return circles.length > 0;
    });
    const case1 = {
      name: '2D initial load dismisses before force layout settles',
      earlyVisible: earlyLoad.exists && !earlyLoad.hidden,
      loadingDismissed: isLoadingDismissed(after2dLoad),
      view2dActive: view2d.btn2dActive && !view2d.btn3dActive,
      nodesRendered: view2d.nodes2d > 0,
      graphVisibleDuringSim: simStillRunning,
      pct: after2dLoad.pct,
    };
    case1.pass = case1.loadingDismissed && case1.view2dActive && case1.nodesRendered && case1.graphVisibleDuringSim;
    report.cases.push(case1);
    await page.screenshot({ path: path.join(outDir, 'graph-loading-2d-ready.png') });

    // Case 2: switch to 3D — loading stays dismissed, 3D canvas shown
    await clickViewMode(page, '3d');
    await sleep(2500);
    const after3dSwitch = await loadingState(page);
    const view3d = await viewState(page);
    const case2 = {
      name: '2D → 3D switch without loading overlay blocking',
      loadingDismissed: isLoadingDismissed(after3dSwitch),
      view3dActive: view3d.btn3dActive && !view3d.btn2dActive,
      has3dCanvas: view3d.has3dCanvas,
      svgHidden: view3d.svgDisplay === 'none',
    };
    case2.pass = case2.loadingDismissed && is3dViewActive(view3d);
    report.cases.push(case2);
    await page.screenshot({ path: path.join(outDir, 'graph-loading-3d-switched.png') });

    // Case 3: switch back to 2D
    await clickViewMode(page, '2d');
    await sleep(2000);
    const after2dSwitch = await loadingState(page);
    const view2dAgain = await viewState(page);
    const case3 = {
      name: '3D → 2D switch restores 2D graph',
      loadingDismissed: isLoadingDismissed(after2dSwitch),
      view2dActive: view2dAgain.btn2dActive && !view2dAgain.btn3dActive,
      nodesRendered: view2dAgain.nodes2d > 0,
      c3dHidden: view2dAgain.c3dHidden,
    };
    case3.pass = case3.loadingDismissed && is2dViewActive(view2dAgain) && case3.nodesRendered;
    report.cases.push(case3);
    await page.screenshot({ path: path.join(outDir, 'graph-loading-2d-switched-back.png') });

    // Case 4: initial 3D load via ?view=3d
    await page.goto(`${baseUrl}${baseUrl.includes('?') ? '&' : '?'}view=3d`, {
      waitUntil: 'domcontentloaded',
      timeout: 60000,
    });
    await page.waitForFunction(() => {
      const el = document.getElementById('graph-loading');
      if (!el) return true;
      return el.hidden || el.classList.contains('is-hidden');
    }, { timeout: 90000 });
    await sleep(2500);
    const after3dInit = await loadingState(page);
    const view3dInit = await viewState(page);
    const case4 = {
      name: 'Initial ?view=3d dismisses loading and enters 3D',
      loadingDismissed: isLoadingDismissed(after3dInit),
      view3dActive: view3dInit.btn3dActive,
      has3dCanvas: view3dInit.has3dCanvas,
    };
    case4.pass = case4.loadingDismissed && is3dViewActive(view3dInit);
    report.cases.push(case4);
    await page.screenshot({ path: path.join(outDir, 'graph-loading-3d-initial.png') });

    report.ok = report.cases.every((c) => c.pass);
    const outJson = path.join(outDir, 'graph-loading-2d3d-verify.json');
    fs.writeFileSync(outJson, JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report, null, 2));
    if (!report.ok) process.exitCode = 1;
    if (headed) await sleep(1500);
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
