// Verify graph.html 2D 边层 Canvas（plan.md G4）：
//  1. 边不再有 SVG 元素，节点/标签与交互仍在 SVG
//  2. 边层 canvas 存在、后备缓冲随视口与像素比上限
//  3. 四种上色模式都画得出东西：默认 / 悬停 / 侧栏聚焦 / 时序动画
//  4. 筛选后力模拟缩成真实子图，边层仍绘制全量（含被筛掉的暗边）
//  5. 主题切换后整层重绘
//  6. 放大后视口裁剪生效（实际绘制的边数下降）
//  7. 2D → 3D → 2D 往返后边层重新绘制
// Usage: node scripts/verify_graph_edge_canvas.cjs [baseUrl] [outDir]
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  const baseUrl = process.argv[2] || 'http://127.0.0.1:8765/graph.html';
  const outDir = path.resolve(process.argv[3] || path.join(__dirname, '..', '.cursor-artifacts', 'screenshots'));
  fs.mkdirSync(outDir, { recursive: true });

  const exe = process.env.PUPPETEER_EXECUTABLE_PATH
    || (fs.existsSync('/opt/pw-browsers/chromium') ? '/opt/pw-browsers/chromium' : 'google-chrome');

  const browser = await puppeteer.launch({
    executablePath: exe,
    headless: 'new',
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--window-size=1440,900'],
    protocolTimeout: 180000,
  });

  const results = [];
  const check = (name, ok, extra) => {
    results.push({ name, ok, extra });
    console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${extra ? ' — ' + extra : ''}`);
  };

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1 });
    const pageErrors = [];
    page.on('pageerror', (err) => pageErrors.push(err.message));

    await page.goto(baseUrl, { waitUntil: 'domcontentloaded', timeout: 60000 });
    await page.waitForFunction(() => {
      const loading = document.getElementById('graph-loading');
      const count = document.getElementById('graph-node-count');
      const loadingHidden = !loading || loading.hidden
        || window.getComputedStyle(loading).display === 'none';
      return loadingHidden && count && count.textContent && !count.textContent.includes('加载中');
    }, { timeout: 180000 });
    await sleep(4000);

    // 边层墨迹：canvas 上 alpha>0 的像素数，作为「确实画出来了」的判据
    const ink = () => page.evaluate(() => {
      const cv = document.getElementById('graph-edge-canvas');
      if (!cv) return -1;
      const { data } = cv.getContext('2d').getImageData(0, 0, cv.width, cv.height);
      let n = 0;
      for (let i = 3; i < data.length; i += 4) if (data[i] > 0) n++;
      return n;
    });
    const dbg = (fn) => page.evaluate(fn);

    // ── 1. DOM 结构 ──
    const structure = await dbg(() => {
      const svg = document.getElementById('graph-canvas');
      return {
        lines: svg.querySelectorAll('.edges line').length,
        nodes: svg.querySelectorAll('.nodes g.node-g').length,
        total: svg.getElementsByTagName('*').length,
        edgeCount: Number((document.getElementById('graph-node-count').textContent
          .match(/·\s*(\d+)\s*边/) || [])[1] || 0),
      };
    });
    check('边不再有 SVG 元素', structure.lines === 0, `.edges line=${structure.lines}`);
    check('节点仍在 SVG', structure.nodes > 0, `node-g=${structure.nodes}`);
    check('SVG 元素总数只剩节点量级', structure.total < structure.edgeCount,
      `svg 元素=${structure.total}，边=${structure.edgeCount}`);

    // ── 2. canvas 后备缓冲 ──
    const canvasInfo = await dbg(() => {
      const cv = document.getElementById('graph-edge-canvas');
      const wrap = document.getElementById('graph-wrap');
      return cv ? {
        w: cv.width, h: cv.height,
        cssW: Math.round(cv.getBoundingClientRect().width),
        cssH: Math.round(cv.getBoundingClientRect().height),
        wrapW: wrap.clientWidth, wrapH: wrap.clientHeight,
        dpr: window.__RN_GRAPH2D_DEBUG__.edgeCanvasSize().dpr,
      } : null;
    });
    check('边层 canvas 存在且覆盖画布', !!canvasInfo
      && Math.abs(canvasInfo.cssW - canvasInfo.wrapW) <= 1
      && Math.abs(canvasInfo.cssH - canvasInfo.wrapH) <= 1, JSON.stringify(canvasInfo));
    check('像素比不超过上限 2', !!canvasInfo && canvasInfo.dpr <= 2, `dpr=${canvasInfo && canvasInfo.dpr}`);

    // ── 3. 默认态 ──
    const inkDefault = await ink();
    const drawnDefault = await dbg(() => window.__RN_GRAPH2D_DEBUG__.edgeDrawnCount());
    check('默认态画出连线', inkDefault > 1000 && drawnDefault > 0,
      `ink=${inkDefault}，drawn=${drawnDefault}`);
    await page.screenshot({ path: path.join(outDir, 'graph-edge-canvas-default.png') });

    // ── 4. 悬停高亮 ──
    const hoverId = await dbg(() => {
      const id = window.__RN_GRAPH2D_DEBUG__.topNodeIds(1)[0];
      for (const el of document.querySelectorAll('.nodes g.node-g')) {
        if (el.__data__ && el.__data__.id === id) {
          const r = el.getBoundingClientRect();
          el.dispatchEvent(new MouseEvent('mouseenter', {
            bubbles: false, clientX: r.x + r.width / 2, clientY: r.y + r.height / 2 }));
          return id;
        }
      }
      return null;
    });
    await sleep(700);
    const inkHover = await ink();
    check('悬停切到高亮上色', !!hoverId && inkHover > 0 && inkHover !== inkDefault,
      `node=${hoverId}，ink=${inkHover}`);
    await page.screenshot({ path: path.join(outDir, 'graph-edge-canvas-hover.png') });
    await page.evaluate((id) => {
      for (const el of document.querySelectorAll('.nodes g.node-g')) {
        if (el.__data__ && el.__data__.id === id) {
          el.dispatchEvent(new MouseEvent('mouseleave', { bubbles: false }));
          return;
        }
      }
    }, hoverId);
    await sleep(900);

    // ── 5. 筛选 Top 300：真实子图 + 边层仍全量绘制 ──
    await page.evaluate(() => {
      const sl = document.getElementById('sl-degree-top');
      sl.value = String(Math.min(300, Number(sl.max)));
      sl.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await sleep(2000);
    const filtered = await dbg(() => ({
      simNodes: window.__RN_GRAPH2D_DEBUG__.simNodeCount(),
      simLinks: window.__RN_GRAPH2D_DEBUG__.simLinkCount(),
      drawn: window.__RN_GRAPH2D_DEBUG__.edgeDrawnCount(),
      lines: document.querySelectorAll('.edges line').length,
    }));
    check('筛选后力模拟缩成子图', filtered.simNodes <= 300 && filtered.simLinks < 35000,
      JSON.stringify(filtered));
    check('筛选后边层仍绘制（含暗边）', filtered.drawn > filtered.simLinks, `drawn=${filtered.drawn}`);
    await page.screenshot({ path: path.join(outDir, 'graph-edge-canvas-filtered.png') });
    await page.evaluate(() => {
      const sl = document.getElementById('sl-degree-top');
      sl.value = sl.max;
      sl.dispatchEvent(new Event('input', { bubbles: true }));
      sl.dispatchEvent(new Event('change', { bubbles: true }));
    });
    await sleep(2500);

    // ── 6. 主题切换重绘 ──
    await page.evaluate(() => document.documentElement.setAttribute('data-theme', 'light'));
    await sleep(900);
    const inkLight = await ink();
    check('浅色主题重绘后仍有连线', inkLight > 1000, `ink=${inkLight}`);
    await page.screenshot({ path: path.join(outDir, 'graph-edge-canvas-light.png') });
    await page.evaluate(() => document.documentElement.setAttribute('data-theme', 'dark'));
    await sleep(900);

    // ── 7. 放大后视口裁剪 ──
    const drawnBeforeZoom = await dbg(() => window.__RN_GRAPH2D_DEBUG__.edgeDrawnCount());
    await page.mouse.move(700, 450);
    await page.mouse.wheel({ deltaY: -900 });
    await sleep(1200);
    const drawnAfterZoom = await dbg(() => window.__RN_GRAPH2D_DEBUG__.edgeDrawnCount());
    check('放大后视口裁剪减少绘制量', drawnAfterZoom < drawnBeforeZoom,
      `${drawnBeforeZoom} → ${drawnAfterZoom}`);
    await page.click('#fit-to-screen');
    await sleep(1500);

    // ── 8. 时序动画只画已显现的边 ──
    await page.click('#physics-toggle');
    await page.waitForSelector('#physics-panel:not([hidden])');
    await page.click('#timeline-animate');
    await sleep(2500);
    const tl = await dbg(() => ({
      animating: window.__RN_GRAPH2D_DEBUG__.timelineAnimating(),
      drawn: window.__RN_GRAPH2D_DEBUG__.edgeDrawnCount(),
    }));
    check('时序动画只画已显现的边', tl.animating && tl.drawn > 0 && tl.drawn < drawnBeforeZoom,
      JSON.stringify(tl));
    await page.screenshot({ path: path.join(outDir, 'graph-edge-canvas-timeline.png') });
    await page.click('#timeline-animate');
    await sleep(2500);
    check('退出时序后边层恢复', (await ink()) > 1000);

    // ── 9. 2D → 3D → 2D ──
    await page.click('#view-mode-3d');
    await page.waitForFunction(() => !!window.__RN_GRAPH3D_VIEW__, { timeout: 60000 });
    await sleep(3000);
    const hiddenIn3d = await dbg(() =>
      getComputedStyle(document.getElementById('graph-edge-canvas')).display);
    check('3D 视图下边层隐藏', hiddenIn3d === 'none', `display=${hiddenIn3d}`);
    await page.click('#view-mode-2d');
    await sleep(3000);
    check('回到 2D 后边层重绘', (await ink()) > 1000);
    await page.screenshot({ path: path.join(outDir, 'graph-edge-canvas-back-2d.png') });

    // 边层相关的页面异常（t.name 为改动前既有问题，不计入）
    const relevant = pageErrors.filter((m) => /edge|canvas|draw|Edge/.test(m));
    check('无边层相关 JS 异常', relevant.length === 0, relevant.join(' | '));
    if (pageErrors.length) console.log('（页面全部异常，含改动前既有项）:', [...new Set(pageErrors)].join(' | '));

    const failed = results.filter((r) => !r.ok);
    console.log(failed.length ? `\n${failed.length} 项失败` : '\n全部通过');
    process.exitCode = failed.length ? 1 : 0;
  } finally {
    await browser.close();
  }
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
