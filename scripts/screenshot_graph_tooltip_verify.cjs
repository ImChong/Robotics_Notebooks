// Capture graph hover tooltips (type badge colored by community; sidebar community = detail button).
// Usage: node scripts/screenshot_graph_tooltip_verify.cjs [basePort] [outDir]
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');

(async () => {
  const port = process.argv[2] || '8765';
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

  async function setupPage(page) {
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
  }

  async function hoverGraphNodeWithCommunity(page, rootSelector, circleSelector) {
    const sel = circleSelector || (rootSelector + ' circle');
    const pt = await page.evaluate((circleSel) => {
      const circles = Array.from(document.querySelectorAll(circleSel));
      for (let i = 0; i < circles.length; i++) {
        const circle = circles[i];
        const rect = circle.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) continue;
        return {
          x: rect.left + rect.width / 2,
          y: rect.top + rect.height / 2,
        };
      }
      return null;
    }, sel);
    if (!pt) throw new Error('no hover target for ' + sel);
    await page.mouse.move(pt.x, pt.y);
    return pt;
  }

  async function waitTooltip(page, tooltipSelector) {
    await page.waitForFunction((sel) => {
      const el = document.querySelector(sel);
      if (!el || el.classList.contains('hidden')) return false;
      return !!el.querySelector('.tt-type');
    }, { timeout: 30000 }, tooltipSelector);
    await new Promise((r) => setTimeout(r, 400));
  }

  async function screenshotTooltip(page, tooltipSelector, outPath) {
    const box = await page.evaluate((sel) => {
      const el = document.querySelector(sel);
      if (!el) return null;
      const r = el.getBoundingClientRect();
      return { x: r.x, y: r.y, width: r.width, height: r.height };
    }, tooltipSelector);
    if (!box || box.width < 10) throw new Error('tooltip box missing: ' + outPath);
    const pad = 12;
    await page.screenshot({
      path: outPath,
      clip: {
        x: Math.max(0, box.x - pad),
        y: Math.max(0, box.y - pad),
        width: Math.min(1440, box.width + pad * 2),
        height: Math.min(900, box.height + pad * 2),
      },
    });
    return outPath;
  }

  const shots = [];

  try {
    // 1) Full graph page
    {
      const page = await browser.newPage();
      await setupPage(page);
      await page.goto(`http://127.0.0.1:${port}/graph.html`, { waitUntil: 'domcontentloaded', timeout: 45000 });
      await page.waitForFunction(() => {
        const count = document.getElementById('graph-node-count');
        const nodes = document.querySelectorAll('#graph-canvas .node-g');
        return count && count.textContent && !count.textContent.includes('加载中') && nodes.length > 0;
      }, { timeout: 90000 });
      await new Promise((r) => setTimeout(r, 4000));
      await hoverGraphNodeWithCommunity(page, '#graph-canvas', '#graph-canvas .node-circle');
      await waitTooltip(page, '#graph-tooltip');
      const out = path.join(outDir, 'graph-tooltip-badges.png');
      await screenshotTooltip(page, '#graph-tooltip', out);
      const meta = await page.evaluate(() => {
        const tip = document.querySelector('#graph-tooltip');
        const type = tip?.querySelector('.tt-type');
        return {
          typeText: type?.textContent?.trim() || '',
          typeBg: type ? getComputedStyle(type).backgroundColor : '',
          hasCommunityBadge: !!tip?.querySelector('.tt-community'),
        };
      });
      shots.push({ page: 'graph.html', out, meta });

      // 1b) graph sidebar badges (PC click opens sidebar)
      const clickPt = await page.evaluate(() => {
        const circle = document.querySelector('#graph-canvas .node-circle');
        if (!circle) return null;
        const rect = circle.getBoundingClientRect();
        return { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 };
      });
      if (!clickPt) throw new Error('graph.html: no node to click for sidebar');
      await page.mouse.click(clickPt.x, clickPt.y);
      await page.waitForFunction(() => {
        const sidebar = document.getElementById('graph-sidebar');
        if (!sidebar || !sidebar.classList.contains('open')) return false;
        const type = sidebar.querySelector('#sb-meta-badges .tt-type');
        const community = sidebar.querySelector('#sb-community .detail-community-badge');
        return !!(type && community);
      }, { timeout: 15000 });
      await new Promise((r) => setTimeout(r, 400));
      const sidebarOut = path.join(outDir, 'graph-sidebar-badges.png');
      const sidebarBox = await page.evaluate(() => {
        const el = document.getElementById('graph-sidebar');
        if (!el) return null;
        const r = el.getBoundingClientRect();
        return { x: r.x, y: r.y, width: r.width, height: Math.min(r.height, 320) };
      });
      if (!sidebarBox) throw new Error('sidebar box missing');
      await page.screenshot({
        path: sidebarOut,
        clip: {
          x: Math.max(0, sidebarBox.x),
          y: Math.max(0, sidebarBox.y),
          width: Math.min(1440, sidebarBox.width),
          height: Math.min(900, sidebarBox.height),
        },
      });
      const sidebarMeta = await page.evaluate(() => {
        const sidebar = document.getElementById('graph-sidebar');
        const type = sidebar?.querySelector('#sb-meta-badges .tt-type');
        const community = sidebar?.querySelector('#sb-community .detail-community-badge');
        return {
          typeText: type?.textContent?.trim() || '',
          typeBg: type ? getComputedStyle(type).backgroundColor : '',
          communityText: community?.textContent?.trim() || '',
          hasMetaCommunityBadge: !!sidebar?.querySelector('#sb-meta-badges .tt-community'),
        };
      });
      shots.push({ page: 'graph.html sidebar', out: sidebarOut, meta: sidebarMeta });
      await page.close();
    }

    // 2) Homepage mini-graph
    {
      const page = await browser.newPage();
      await setupPage(page);
      await page.goto(`http://127.0.0.1:${port}/index.html`, { waitUntil: 'domcontentloaded', timeout: 45000 });
      await page.waitForSelector('#mini-graph-svg .mini-graph-node circle', { timeout: 60000 });
      await new Promise((r) => setTimeout(r, 5000));
      await hoverGraphNodeWithCommunity(page, '#mini-graph-svg', '#mini-graph-svg .mini-graph-node circle');
      await waitTooltip(page, '#mini-graph-tooltip');
      const out = path.join(outDir, 'mini-graph-tooltip-badges.png');
      await screenshotTooltip(page, '#mini-graph-tooltip', out);
      shots.push({ page: 'index.html', out });
      await page.close();
    }

    // 3) Detail page knowledge mini-map
    {
      const page = await browser.newPage();
      await setupPage(page);
      await page.goto(`http://127.0.0.1:${port}/detail.html?id=wiki-concepts-sim2real`, {
        waitUntil: 'domcontentloaded',
        timeout: 45000,
      });
      await page.waitForSelector('#detailMiniMapSvg .mini-node circle', { timeout: 60000 });
      await new Promise((r) => setTimeout(r, 3000));
      await hoverGraphNodeWithCommunity(page, '#detailMiniMapSvg', '#detailMiniMapSvg .mini-node circle');
      await waitTooltip(page, '#detail-mini-map-tooltip');
      const out = path.join(outDir, 'detail-mini-map-tooltip-badges.png');
      await screenshotTooltip(page, '#detail-mini-map-tooltip', out);
      shots.push({ page: 'detail.html', out });
      await page.close();
    }

    const reportPath = path.join(outDir, 'graph-tooltip-verify.json');
    fs.writeFileSync(reportPath, JSON.stringify({ shots }, null, 2));
    console.log(JSON.stringify({ ok: true, shots, reportPath }, null, 2));
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
