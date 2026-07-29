// Verify graph.html 3D 视图下的社区胶囊标签：
//  1. 3D 视图中勾选 → HTML overlay 胶囊标签出现在各社区 3D 质心投影位置
//  2. 取消勾选 → 标签隐藏
//  3. 3D 中切到按类型筛选 → 勾选框置灰且标签隐藏；切回按社区恢复
// Usage: node scripts/verify_graph_community_labels_3d.cjs [baseUrl] [outDir]
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');

(async () => {
  const baseUrl = process.argv[2] || 'http://127.0.0.1:8765/graph.html?view=3d';
  const outDir = path.resolve(process.argv[3] || path.join(__dirname, '..', '.cursor-artifacts', 'screenshots'));
  fs.mkdirSync(outDir, { recursive: true });

  const exe = process.env.PUPPETEER_EXECUTABLE_PATH
    || (fs.existsSync('/usr/local/bin/google-chrome') ? '/usr/local/bin/google-chrome' : 'google-chrome');

  // 3D 需要 WebGL：headless 下用 SwiftShader 软件渲染（--disable-gpu 会导致 WebGL 上下文创建失败）
  const browser = await puppeteer.launch({
    executablePath: exe,
    headless: 'new',
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--window-size=1440,900',
      '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
  });

  const results = [];
  const check = (name, ok, extra) => {
    results.push({ name, ok, extra });
    console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${extra ? ' — ' + extra : ''}`);
  };

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1 });
    page.on('pageerror', (err) => console.log('PAGEERROR:', err.message));

    await page.goto(baseUrl, { waitUntil: 'domcontentloaded', timeout: 45000 });
    await page.waitForFunction(() => {
      const loading = document.getElementById('graph-loading');
      const count = document.getElementById('graph-node-count');
      const loadingHidden = !loading || loading.hidden
        || loading.style.display === 'none'
        || window.getComputedStyle(loading).display === 'none';
      const countReady = count && count.textContent && !count.textContent.includes('加载中');
      const canvas3d = document.querySelector('#graph-canvas-3d:not([hidden]) canvas');
      return loadingHidden && countReady && canvas3d;
    }, { timeout: 90000 });
    // 等 3D 力导向布局收敛
    await new Promise((r) => setTimeout(r, 6000));

    const labelState = () => page.evaluate(() => {
      const cb = document.getElementById('check-community-labels');
      const els = Array.from(document.querySelectorAll('#graph-canvas-3d .graph-3d-community-label'));
      const visible = els.filter((el) => el.style.visibility === 'visible' && el.style.opacity === '1');
      return {
        disabled: cb ? cb.disabled : null,
        checked: cb ? cb.checked : null,
        total: els.length,
        visibleCount: visible.length,
        labels: visible.map((el) => el.textContent),
        pillStyleOk: visible.every((el) => {
          const cs = getComputedStyle(el);
          // inline background 读回时被浏览器规范化为 rgb(...) 形式
          return cs.borderRadius === '999px' && /^(#[0-9a-f]{6}|rgb\(\d+, \d+, \d+\))$/i.test(el.style.background || '');
        }),
        inViewport: visible.every((el) => {
          const r = el.getBoundingClientRect();
          return r.left >= 0 && r.top >= 0 && r.right <= innerWidth && r.bottom <= innerHeight;
        }),
        sample: visible.slice(0, 3).map((el) => ({
          text: el.textContent,
          left: el.style.left,
          top: el.style.top,
          background: el.style.background,
          color: el.style.color,
        })),
      };
    });

    // 打开参数面板并勾选
    await page.click('#physics-toggle');
    await page.click('#check-community-labels');
    await page.waitForFunction(() => {
      const els = Array.from(document.querySelectorAll('#graph-canvas-3d .graph-3d-community-label'));
      return els.some((el) => el.style.visibility === 'visible');
    }, { timeout: 20000 });
    await new Promise((r) => setTimeout(r, 800));

    let s = await labelState();
    check('3D 勾选后：胶囊标签全部可见', s.visibleCount === 15, `visible=${s.visibleCount}/15`);
    check('3D 勾选后：标签在视口内', s.inViewport === true);
    check('3D 勾选后：胶囊样式（999px 圆角 + 社区色背景）', s.pillStyleOk === true);
    console.log('  标签示例:', JSON.stringify(s.sample));
    await page.screenshot({ path: path.join(outDir, 'graph-community-labels-3d-on.png') });

    // 取消勾选 → 全部隐藏
    await page.click('#check-community-labels');
    await new Promise((r) => setTimeout(r, 500));
    s = await labelState();
    check('3D 取消勾选：标签全部隐藏', s.visibleCount === 0, `visible=${s.visibleCount}`);

    // 重新勾选，切到按类型 → 置灰 + 隐藏
    await page.click('#check-community-labels');
    await new Promise((r) => setTimeout(r, 400));
    await page.evaluate(() => document.getElementById('filter-mode-type').click());
    await new Promise((r) => setTimeout(r, 600));
    s = await labelState();
    check('3D 按类型筛选：勾选框置灰', s.disabled === true);
    check('3D 按类型筛选：标签全部隐藏', s.visibleCount === 0);

    // 切回按社区 → 恢复可见
    await page.evaluate(() => document.getElementById('filter-mode-community').click());
    await new Promise((r) => setTimeout(r, 600));
    s = await labelState();
    check('3D 切回按社区：勾选框恢复且标签重现', s.disabled === false && s.visibleCount === 15,
      `visible=${s.visibleCount}`);

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
