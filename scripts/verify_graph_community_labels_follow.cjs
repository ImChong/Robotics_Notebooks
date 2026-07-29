// 复现/验证「标签跟随」问题：
//  3D：引擎收敛后 onEngineTick 停发，旋转相机时若标签不重投影则冻结（bug）。
//  2D：标签在 gRoot 内，zoom/pan 由组变换带动，应始终贴合聚类质心（结构性免疫，实证确认）。
// Usage: node scripts/verify_graph_community_labels_follow.cjs [baseUrl] [outDir]
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');

const WEBGL_ARGS = ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'];

(async () => {
  const baseUrl = process.argv[2] || 'http://127.0.0.1:8765/graph.html';
  const outDir = path.resolve(process.argv[3] || path.join(__dirname, '..', '.cursor-artifacts', 'screenshots'));
  fs.mkdirSync(outDir, { recursive: true });

  const exe = process.env.PUPPETEER_EXECUTABLE_PATH
    || (fs.existsSync('/usr/local/bin/google-chrome') ? '/usr/local/bin/google-chrome' : 'google-chrome');

  const browser = await puppeteer.launch({
    executablePath: exe,
    headless: 'new',
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--window-size=1440,900', ...WEBGL_ARGS],
  });

  const results = [];
  const check = (name, ok, extra) => {
    results.push({ name, ok, extra });
    console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${extra ? ' — ' + extra : ''}`);
  };
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  async function waitGraphLoaded(page) {
    await page.waitForFunction(() => {
      const el = document.getElementById('graph-loading');
      const c = document.getElementById('graph-node-count');
      const lh = !el || el.hidden || el.classList.contains('is-hidden');
      return lh && c && c.textContent && !c.textContent.includes('加载中');
    }, { timeout: 90000 });
  }

  try {
    /* ════════════ 用例 1：3D 引擎收敛后旋转相机，标签必须跟随 ════════════ */
    const page = await browser.newPage();
    await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1 });
    page.on('pageerror', (e) => console.log('PAGEERROR:', e.message));

    await page.goto(`${baseUrl}?view=3d`, { waitUntil: 'domcontentloaded', timeout: 60000 });
    await waitGraphLoaded(page);
    await page.waitForFunction(() => {
      const c3d = document.getElementById('graph-canvas-3d');
      return c3d && !c3d.hidden && !!c3d.querySelector('canvas');
    }, { timeout: 90000 });

    // 社区标签默认开启；打开参数面板仅便于人工对照，无需再勾选
    await page.click('#physics-toggle');

    const read3dLabels = () => page.evaluate(() => {
      const els = Array.from(document.querySelectorAll('#graph-canvas-3d .graph-3d-community-label'));
      const vis = els.filter((el) => el.style.visibility === 'visible');
      return {
        count: vis.length,
        pos: vis.slice(0, 5).map((el) => ({ text: el.textContent, left: el.style.left, top: el.style.top })),
      };
    });

    // 等引擎真正收敛：连续两次采样标签屏幕坐标完全静止 ⇒ 引擎停转（onEngineTick 不再触发）。
    // （SwiftShader 下 270 tick 收敛可能需 20s+，上限 60s 兜底）
    const parse = (s) => Number(String(s).replace('px', '')) || 0;
    const posKey = (st) => st.pos.map((p) => `${p.left}|${p.top}`).join(';');
    let settled = false;
    for (let i = 0; i < 24 && !settled; i++) {
      const a = posKey(await read3dLabels());
      await sleep(2500);
      const b = posKey(await read3dLabels());
      settled = a.length > 0 && a === b;
      if (!settled && i % 4 === 3) console.log(`  …等待引擎收敛 (${((i + 1) * 2.5).toFixed(0)}s)`);
    }
    check('3D：引擎收敛（标签坐标静止）', settled);

    const before = await read3dLabels();
    check('3D：收敛后 15 个标签可见', before.count === 15, `count=${before.count}`);
    await page.screenshot({ path: path.join(outDir, 'graph-3d-follow-before-rotate.png') });

    // 左键拖拽旋转相机（TrackballControls：左键=旋转）
    await page.mouse.move(720, 450);
    await page.mouse.down();
    await page.mouse.move(1050, 250, { steps: 25 });
    await page.mouse.up();
    await sleep(1200);

    const after = await read3dLabels();
    await page.screenshot({ path: path.join(outDir, 'graph-3d-follow-after-rotate.png') });

    const moved = before.pos.map((b, i) => {
      const a = after.pos[i] || { left: '0', top: '0' };
      return Math.hypot(parse(a.left) - parse(b.left), parse(a.top) - parse(b.top));
    });
    const frozenCount = moved.filter((d) => d < 1).length;
    check('3D：旋转相机后标签跟随重投影（不冻结）', frozenCount === 0,
      `样本位移(px)=${moved.map((d) => d.toFixed(1)).join(',')}`);

    /* ════════════ 用例 2：2D 模拟收敛后缩放/平移，标签仍贴合聚类质心 ════════════ */
    const page2 = await browser.newPage();
    await page2.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1 });
    page2.on('pageerror', (e) => console.log('PAGEERROR(2D):', e.message));
    await page2.goto(baseUrl, { waitUntil: 'domcontentloaded', timeout: 60000 });
    await waitGraphLoaded(page2);
    await sleep(4000);   // 等 2D 力布局收敛；社区标签默认开启
    await page2.click('#physics-toggle');
    await sleep(600);

    const gluedCheck = () => page2.evaluate(() => {
      const svg = document.getElementById('graph-canvas');
      const labels = Array.from(svg.querySelectorAll('g.community-label'));
      // 节点按社区分组求屏幕质心（nodeLabelVisible 不可达，这里取全部节点，与无筛选时标签质心口径一致）
      const byComm = new Map();
      svg.querySelectorAll('g.node-g').forEach((g) => {
        const tr = g.getAttribute('transform') || '';
        const m = tr.match(/translate\(([-\d.e]+),([- \d.e]+)\)/);
        if (!m) return;
        const r = g.getBoundingClientRect();
        const comm = g.__data__ && g.__data__.community;
        if (!comm) return;
        let a = byComm.get(comm);
        if (!a) { a = { sx: 0, sy: 0, n: 0 }; byComm.set(comm, a); }
        a.sx += r.left + r.width / 2; a.sy += r.top + r.height / 2; a.n += 1;
      });
      const misses = [];
      labels.forEach((g) => {
        const text = g.querySelector('text.community-label-text');
        const lr = g.getBoundingClientRect();
        const lx = lr.left + lr.width / 2;
        const ly = lr.top + lr.height / 2;
        let best = null;
        byComm.forEach((a) => {
          const cx = a.sx / a.n;
          const cy = a.sy / a.n;
          const d = Math.hypot(lx - cx, ly - cy);
          if (!best || d < best.d) best = { d, cx, cy };
        });
        // 与最近社区屏幕质心偏差（应≈0：标签即画在该社区质心）
        if (!best || best.d > 3) misses.push({ text: text && text.textContent, d: best && best.d });
      });
      return { labels: labels.length, misses };
    });

    // 收敛后直接校验一次
    let glued = await gluedCheck();
    check('2D：收敛后标签贴合各社区屏幕质心', glued.labels === 15 && glued.misses.length === 0,
      JSON.stringify(glued.misses.slice(0, 3)));

    // 滚轮放大 + 拖拽平移（引擎停转状态下的纯视角操作）
    await page2.mouse.move(720, 450);
    await page2.mouse.wheel({ deltaY: -600 });
    await sleep(500);
    await page2.mouse.move(720, 450);
    await page2.mouse.down();
    await page2.mouse.move(500, 600, { steps: 15 });
    await page2.mouse.up();
    await sleep(500);

    glued = await gluedCheck();
    check('2D：缩放+平移后标签仍贴合各社区屏幕质心', glued.misses.length === 0,
      JSON.stringify(glued.misses.slice(0, 3)));

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
