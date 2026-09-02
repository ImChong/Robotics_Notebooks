// Verify graph.html 「显示社区标签」勾选框行为：
//  1. 按社区模式默认开启 → 各聚类中心出现社区名称标签
//  2. 取消勾选 → 标签消失
//  3. 切到按类型/按健康度/按开源筛选 → 勾选框置灰不可选、标签不显示
//  4. 切回按社区 → 勾选框恢复可选，勾选状态下标签重现
//  5. 勾选具体社区 → 只剩对应社区的标签
// Usage: node scripts/verify_graph_community_labels.cjs [baseUrl] [outDir]
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');

(async () => {
  const baseUrl = process.argv[2] || 'http://127.0.0.1:8765/graph.html';
  const outDir = path.resolve(process.argv[3] || path.join(__dirname, '..', '.cursor-artifacts', 'screenshots'));
  fs.mkdirSync(outDir, { recursive: true });

  const exe = process.env.PUPPETEER_EXECUTABLE_PATH
    || (fs.existsSync('/usr/local/bin/google-chrome') ? '/usr/local/bin/google-chrome' : 'google-chrome');

  const browser = await puppeteer.launch({
    executablePath: exe,
    headless: 'new',
    args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--window-size=1440,900'],
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
      return loadingHidden && countReady;
    }, { timeout: 90000 });
    // 页面默认进 3D；本脚本验证 2D SVG 社区标签，先切回 2D
    await page.evaluate(() => {
      const btn = document.getElementById('view-mode-2d');
      if (btn) btn.click();
    });
    await page.waitForFunction(() => {
      const canvas = document.getElementById('graph-canvas');
      if (!canvas) return false;
      return window.getComputedStyle(canvas).display !== 'none';
    }, { timeout: 15000 }).catch(() => {});
    // 等力导向布局收敛，聚类形状稳定
    await new Promise((r) => setTimeout(r, 4500));

    const labelState = () => page.evaluate(() => {
      const cb = document.getElementById('check-community-labels');
      const toggle = document.getElementById('community-labels-toggle');
      const groups = Array.from(document.querySelectorAll('#graph-canvas g.community-label'));
      const texts = groups.map((g) => g.querySelector('text.community-label-text')).filter(Boolean);
      const pills = groups.map((g) => g.querySelector('rect.community-label-bg')).filter(Boolean);
      const fontSizes = texts.map((t) => parseFloat(t.getAttribute('font-size'))).filter((n) => Number.isFinite(n));
      const fontMin = fontSizes.length ? Math.min(...fontSizes) : null;
      const fontMax = fontSizes.length ? Math.max(...fontSizes) : null;
      return {
        disabled: cb ? cb.disabled : null,
        checked: cb ? cb.checked : null,
        toggleDisabledClass: toggle ? toggle.classList.contains('is-disabled') : null,
        count: groups.length,
        labels: texts.map((t) => t.textContent),
        fontSizes,
        fontMin,
        fontMax,
        pillOk: pills.length === groups.length && pills.every((r) => {
          const h = Number(r.getAttribute('height'));
          const rx = Number(r.getAttribute('rx'));
          const w = Number(r.getAttribute('width'));
          return h > 0 && w > 0 && Math.abs(rx - h / 2) < 0.01;
        }),
        pillFillColored: pills.every((r) => /^#[0-9a-f]{6}$/i.test(r.getAttribute('fill') || '')),
        sample: groups.slice(0, 3).map((g) => ({
          text: g.querySelector('text.community-label-text')?.textContent,
          fontSize: g.querySelector('text.community-label-text')?.getAttribute('font-size'),
          transform: g.getAttribute('transform'),
          fill: g.querySelector('rect.community-label-bg')?.getAttribute('fill'),
          textFill: g.querySelector('text.community-label-text')?.getAttribute('fill'),
        })),
      };
    });

    // 打开参数面板
    await page.click('#physics-toggle');

    // ── 1. 初始状态：按社区模式，默认开启社区标签 ──
    let s = await labelState();
    check('初始：勾选框可用（按社区模式）', s.disabled === false && s.toggleDisabledClass === false);
    check('初始：默认勾选且有社区标签', s.checked === true && s.count >= 3,
      `count=${s.count} labels=${s.labels.join('|')}`);
    check('默认开启：标签为短名（无括号英文/无“社区”后缀）',
      s.labels.every((t) => !t.includes('（') && !t.includes('社区')));
    check('默认开启：标签为胶囊卡片（rect 底 + rx=height/2 + 社区色填充）',
      s.pillOk === true && s.pillFillColored === true);
    check('默认开启：字号随社区节点数缩放（约 8–28px 且差异更明显）',
      s.fontMin != null && s.fontMax != null
        && s.fontMin >= 7.5 && s.fontMax <= 28.5
        && (s.fontMax - s.fontMin) >= 12,
      `min=${s.fontMin} max=${s.fontMax}`);
    check('默认开启：不含「其他」兜底社区标签',
      s.labels.every((t) => t !== '其他' && !/^其他/.test(t)),
      `labels=${s.labels.join('|')}`);
    const expectedCommunities = await page.evaluate(() => {
      const set = new Set();
      document.querySelectorAll('#graph-legend .legend-row[data-community-id]')
        .forEach((row) => {
          const id = row.getAttribute('data-community-id');
          if (id && id !== 'community-other') set.add(id);
        });
      return set.size;
    });
    check('默认开启：标签数与命名社区数一致（排除其他）', s.count === expectedCommunities,
      `labels=${s.count} communities=${expectedCommunities}`);
    console.log('  标签示例:', JSON.stringify(s.sample));
    await page.screenshot({ path: path.join(outDir, 'graph-community-labels-on.png') });

    // ── 2. 取消勾选 → 标签消失 ──
    await page.click('#check-community-labels');
    await new Promise((r) => setTimeout(r, 400));
    s = await labelState();
    check('取消勾选：标签消失', s.checked === false && s.count === 0);

    // 重新勾选，验证模式切换的置灰与恢复
    await page.click('#check-community-labels');
    await new Promise((r) => setTimeout(r, 400));

    // ── 3. 切到按类型筛选 → 勾选框置灰、标签隐藏 ──
    await page.click('#filter-toggle');
    await page.click('#filter-mode-type');
    await new Promise((r) => setTimeout(r, 500));
    s = await labelState();
    check('按类型筛选：勾选框置灰不可选', s.disabled === true && s.toggleDisabledClass === true);
    check('按类型筛选：社区标签不显示', s.count === 0);
    // 全局点击会收起参数面板，重开后裁剪面板区域截灰态
    // （筛选面板展开时坐标点击可能被遮罩拦截，改用 DOM click）
    await page.evaluate(() => document.getElementById('physics-toggle').click());
    await new Promise((r) => setTimeout(r, 300));
    const panelBox = await page.evaluate(() => {
      const el = document.getElementById('physics-panel');
      const r = el.getBoundingClientRect();
      return { x: Math.max(0, r.left - 8), y: Math.max(0, r.top - 8), width: r.width + 16, height: r.height + 16 };
    });
    await page.screenshot({
      path: path.join(outDir, 'graph-community-labels-disabled-type-mode.png'),
      clip: panelBox,
    });

    // ── 3b. 切到按健康度筛选 → 同样置灰 ──
    // （上一步重开参数面板会触发全局点击收起筛选面板，改用 DOM click 触发 tab 切换）
    await page.evaluate(() => document.getElementById('filter-mode-health').click());
    await new Promise((r) => setTimeout(r, 500));
    s = await labelState();
    check('按健康度筛选：勾选框置灰不可选', s.disabled === true && s.toggleDisabledClass === true);

    // ── 3c. 切到按开源筛选 → 同样置灰 ──
    await page.evaluate(() => document.getElementById('filter-mode-opensource').click());
    await new Promise((r) => setTimeout(r, 500));
    s = await labelState();
    check('按开源筛选：勾选框置灰不可选', s.disabled === true && s.toggleDisabledClass === true);
    check('按开源筛选：社区标签不显示', s.count === 0);

    // ── 4. 切回按社区 → 勾选框恢复、勾选状态下标签重现 ──
    await page.evaluate(() => document.getElementById('filter-mode-community').click());
    await new Promise((r) => setTimeout(r, 500));
    s = await labelState();
    check('切回按社区：勾选框恢复可选', s.disabled === false && s.toggleDisabledClass === false);
    check('切回按社区：勾选状态保留且标签重现', s.checked === true && s.count === expectedCommunities,
      `count=${s.count}`);

    // ── 5. 勾选具体命名社区 → 只剩该社区标签 ──
    const firstCommunity = await page.evaluate(() => {
      const cbs = Array.from(document.querySelectorAll('#filter-panel-body input[type="checkbox"]'));
      const cb = cbs.find((el) => el.value && el.value !== 'community-other') || cbs[0];
      if (!cb) return null;
      cb.click();
      return cb.value;
    });
    await new Promise((r) => setTimeout(r, 600));
    s = await labelState();
    check('勾选具体社区：只剩一个社区标签', s.count === 1, `community=${firstCommunity} labels=${s.labels.join('|')}`);
    await page.screenshot({ path: path.join(outDir, 'graph-community-labels-single-community.png') });

    // ── 6. 只勾选「其他」兜底社区 → 无漂浮标签 ──
    await page.evaluate(() => {
      document.querySelectorAll('#filter-panel-body input[type="checkbox"]').forEach((cb) => {
        if (cb.checked) cb.click();
      });
      const other = Array.from(document.querySelectorAll('#filter-panel-body input[type="checkbox"]'))
        .find((el) => el.value === 'community-other');
      if (other && !other.checked) other.click();
    });
    await new Promise((r) => setTimeout(r, 600));
    s = await labelState();
    check('只勾选「其他」：无漂浮社区标签', s.count === 0, `labels=${s.labels.join('|')}`);

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
