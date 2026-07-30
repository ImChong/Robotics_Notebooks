// Verify graph.html 3D 视图下的社区胶囊标签：
//  1. 3D 视图默认开启 → HTML overlay 胶囊标签出现在各社区 3D 质心投影位置
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
      const fontSizes = visible.map((el) => parseFloat(el.style.fontSize)).filter((n) => Number.isFinite(n));
      const fontMin = fontSizes.length ? Math.min(...fontSizes) : null;
      const fontMax = fontSizes.length ? Math.max(...fontSizes) : null;
      const parseScale = (transform) => {
        const m = /scale\(([-0-9.]+)\)/.exec(transform || '');
        return m ? parseFloat(m[1]) : 1;
      };
      const scales = visible.map((el) => parseScale(el.style.transform));
      const scaleMin = scales.length ? Math.min(...scales) : null;
      const scaleMax = scales.length ? Math.max(...scales) : null;
      return {
        disabled: cb ? cb.disabled : null,
        checked: cb ? cb.checked : null,
        total: els.length,
        visibleCount: visible.length,
        labels: visible.map((el) => el.textContent),
        fontSizes,
        fontMin,
        fontMax,
        scaleMin,
        scaleMax,
        sampleScale: scales[0] != null ? scales[0] : null,
        pillStyleOk: visible.every((el) => {
          const cs = getComputedStyle(el);
          // inline background 读回时被浏览器规范化为 rgb(...) 形式
          return cs.borderRadius === '999px' && /^(#[0-9a-f]{6}|rgb\(\d+, \d+, \d+\))$/i.test(el.style.background || '');
        }),
        inViewport: visible.every((el) => {
          const r = el.getBoundingClientRect();
          return r.left >= -40 && r.top >= -40 && r.right <= innerWidth + 40 && r.bottom <= innerHeight + 40;
        }),
        inViewportCount: visible.filter((el) => {
          const r = el.getBoundingClientRect();
          return r.left >= -40 && r.top >= -40 && r.right <= innerWidth + 40 && r.bottom <= innerHeight + 40;
        }).length,
        sample: visible.slice(0, 3).map((el) => ({
          text: el.textContent,
          fontSize: el.style.fontSize,
          transform: el.style.transform,
          background: el.style.background,
          color: el.style.color,
        })),
        usesTransform: visible.every((el) => /translate3d\(/.test(el.style.transform || '')),
        usesZoomScale: visible.every((el) => /scale\(/.test(el.style.transform || '')),
      };
    });

    // 打开参数面板；社区标签默认开启，等待 3D overlay 可见
    await page.click('#physics-toggle');
    await page.waitForFunction(() => {
      const els = Array.from(document.querySelectorAll('#graph-canvas-3d .graph-3d-community-label'));
      return els.some((el) => el.style.visibility === 'visible');
    }, { timeout: 20000 });
    // 等首次取景写入 baselineCameraDist；再等到布局收敛后的最终适配（scale≈1）
    await page.waitForFunction(() => {
      const view = window.__RN_GRAPH3D_VIEW__;
      return !!(view && view.getBaselineCameraDist && view.getBaselineCameraDist());
    }, { timeout: 20000 });
    await page.waitForFunction(() => {
      const view = window.__RN_GRAPH3D_VIEW__;
      if (!view || typeof view.getCommunityLabelZoomScale !== 'function') return false;
      const z = view.getCommunityLabelZoomScale();
      return z > 0.85 && z < 1.2;
    }, { timeout: 20000 }).catch(() => {});
    await new Promise((r) => setTimeout(r, 800));

    let s = await labelState();
    const expectedCommunities = await page.evaluate(() => {
      const set = new Set();
      document.querySelectorAll('#graph-legend .legend-row[data-community-id]')
        .forEach((row) => {
          const id = row.getAttribute('data-community-id');
          if (id && id !== 'community-other') set.add(id);
        });
      return set.size || 15;
    });
    check('3D 默认开启：勾选框已勾选', s.checked === true);
    check('3D 默认开启：命名社区胶囊可见（排除其他）', s.visibleCount === expectedCommunities,
      `visible=${s.visibleCount}/${expectedCommunities}`);
    check('3D 默认开启：不含「其他」兜底社区标签',
      s.labels.every((t) => t !== '其他' && !/^其他/.test(t)),
      `labels=${s.labels.join('|')}`);
    check('3D 默认开启：标签在视口内', s.inViewport === true || (s.inViewportCount >= Math.ceil(s.visibleCount * 0.75)),
      `inViewport=${s.inViewport} count=${s.inViewportCount}/${s.visibleCount}`);
    check('3D 默认开启：胶囊样式（999px 圆角 + 社区色背景）', s.pillStyleOk === true);
    check('3D 默认开启：位置用 translate3d（非 left/top）', s.usesTransform === true);
    check('3D 默认开启：transform 含 scale（随相机缩放）', s.usesZoomScale === true,
      `sample=${s.sample && s.sample[0] && s.sample[0].transform}`);
    check('3D 默认开启：字号随社区节点数缩放（3D 专用约 8–16px，小于 2D）',
      s.fontMin != null && s.fontMax != null
        && s.fontMin >= 7.5 && s.fontMax <= 16.5
        && (s.fontMax - s.fontMin) >= 4,
      `min=${s.fontMin} max=${s.fontMax}`);
    console.log('  标签示例:', JSON.stringify(s.sample));
    await page.screenshot({ path: path.join(outDir, 'graph-community-labels-3d-on.png') });

    // 程序化推进相机：胶囊 scale 应随距离变化（比 headless 滚轮更稳）
    const scaleBeforeZoomIn = s.sampleScale;
    const dollyInfo = await page.evaluate(() => {
      const view = window.__RN_GRAPH3D_VIEW__;
      if (!view || typeof view.dollyZoom !== 'function') {
        return { ok: false, reason: 'no-view' };
      }
      const before = {
        scale: view.getCommunityLabelZoomScale && view.getCommunityLabelZoomScale(),
        dist: view.getCameraDistance && view.getCameraDistance(),
        baseline: view.getBaselineCameraDist && view.getBaselineCameraDist(),
      };
      const ok = view.dollyZoom(1.8, 0);
      return {
        ok: !!ok,
        before,
        afterImmediate: {
          scale: view.getCommunityLabelZoomScale && view.getCommunityLabelZoomScale(),
          dist: view.getCameraDistance && view.getCameraDistance(),
        },
      };
    });
    await new Promise((r) => setTimeout(r, 400));
    s = await labelState();
    check('3D 程序化放大：dollyZoom 可用', dollyInfo.ok === true, JSON.stringify(dollyInfo));
    check('3D 程序化放大：社区标签 scale 增大',
      s.sampleScale != null && scaleBeforeZoomIn != null && s.sampleScale > scaleBeforeZoomIn + 0.05,
      `before=${scaleBeforeZoomIn} after=${s.sampleScale} info=${JSON.stringify(dollyInfo)}`);
    await page.screenshot({ path: path.join(outDir, 'graph-community-labels-3d-zoomed-in.png') });

    // 再拉远：相对放大后应变小
    const scaleAfterZoomIn = s.sampleScale;
    await page.evaluate(() => {
      const view = window.__RN_GRAPH3D_VIEW__;
      if (view && view.dollyZoom) view.dollyZoom(1 / 2.2, 0);
    });
    await new Promise((r) => setTimeout(r, 400));
    s = await labelState();
    check('3D 程序化缩小：社区标签 scale 减小',
      s.sampleScale != null && scaleAfterZoomIn != null && s.sampleScale < scaleAfterZoomIn - 0.05,
      `before=${scaleAfterZoomIn} after=${s.sampleScale}`);
    await page.screenshot({ path: path.join(outDir, 'graph-community-labels-3d-zoomed-out.png') });

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
    check('3D 切回按社区：勾选框恢复且标签重现',
      s.disabled === false && s.visibleCount === expectedCommunities,
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
