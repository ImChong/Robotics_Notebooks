// Verify graph.html 「按开源」着色与筛选：
//  1. 筛选 tab 顺序为 按类型 / 按社区 / 按开源 / 按健康度
//  2. 切到按开源 → 节点只使用 teal / rose 两色；图例与勾选项为已开源 / 未开源
//  3. 只勾选已开源 → 可见节点均 has_repo；只勾选未开源 → 可见节点均无 has_repo
// Usage: node scripts/verify_graph_opensource_filter.cjs [baseUrl] [outDir]
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');

const OPEN = '#2dd4bf';
const CLOSED = '#fb7185';
const ART_DIR = '/opt/cursor/artifacts/screenshots';

const CHROME_CANDIDATES = [
  process.env.PUPPETEER_EXECUTABLE_PATH,
  process.env.CHROME_PATH,
  '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
  '/usr/local/bin/google-chrome',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
].filter(Boolean);

(async () => {
  const baseUrl = process.argv[2] || 'http://127.0.0.1:8765/graph.html';
  const outDir = path.resolve(
    process.argv[3] || path.join(__dirname, '..', '.cursor-artifacts', 'screenshots')
  );
  fs.mkdirSync(outDir, { recursive: true });
  try {
    fs.mkdirSync(ART_DIR, { recursive: true });
  } catch (_) { /* optional */ }

  const exe = CHROME_CANDIDATES.find((p) => fs.existsSync(p));
  if (!exe) {
    console.error('No Chrome/Chromium found. Set CHROME_PATH.');
    process.exit(1);
  }

  const results = [];
  const check = (name, ok, extra) => {
    results.push({ name, ok, extra });
    console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${extra ? ' — ' + extra : ''}`);
  };

  const browser = await puppeteer.launch({
    executablePath: exe,
    headless: 'new',
    args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--window-size=1440,900'],
  });

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

    await page.evaluate(() => {
      const btn = document.getElementById('view-mode-2d');
      if (btn) btn.click();
    });
    await page.waitForFunction(() => {
      const canvas = document.getElementById('graph-canvas');
      return canvas && window.getComputedStyle(canvas).display !== 'none';
    }, { timeout: 15000 }).catch(() => {});
    await new Promise((r) => setTimeout(r, 2500));

    const tabOrder = await page.evaluate(() =>
      Array.from(document.querySelectorAll('#filter-mode-tabs .chip')).map((el) => ({
        id: el.id,
        text: el.textContent.trim(),
      }))
    );
    check(
      '筛选 tab 顺序为 类型/社区/开源/健康度',
      JSON.stringify(tabOrder.map((t) => t.id)) === JSON.stringify([
        'filter-mode-type',
        'filter-mode-community',
        'filter-mode-opensource',
        'filter-mode-health',
      ]),
      JSON.stringify(tabOrder)
    );

    await page.evaluate(() => document.getElementById('filter-toggle').click());
    await page.waitForFunction(() => {
      const panel = document.getElementById('filter-panel');
      return panel && !panel.hidden;
    }, { timeout: 5000 });

    await page.evaluate(() => document.getElementById('filter-mode-opensource').click());
    await new Promise((r) => setTimeout(r, 800));

    const ui = await page.evaluate((openColor, closedColor) => {
      const subtitle = document.getElementById('filter-subtitle')?.textContent || '';
      const legendTitle = document.querySelector('#graph-legend .legend-title')?.textContent || '';
      const legendRows = Array.from(document.querySelectorAll('#graph-legend .legend-row')).map((row) => ({
        key: row.getAttribute('data-opensource'),
        label: row.textContent.trim(),
        color: row.querySelector('.legend-dot')?.style.background || '',
      }));
      const filterItems = Array.from(document.querySelectorAll('#filter-panel-body .filter-item')).map((el) => ({
        value: el.querySelector('input')?.value,
        label: el.querySelector('.filter-item-label')?.textContent?.trim(),
        color: el.querySelector('.filter-dot')?.style.background || '',
        count: el.querySelector('.filter-count-badge')?.textContent?.trim() || '',
      }));
      const fills = Array.from(document.querySelectorAll('#graph-canvas circle.node-circle'))
        .map((el) => (el.getAttribute('fill') || '').toLowerCase());
      const uniqueFills = Array.from(new Set(fills));
      const openFill = openColor.toLowerCase();
      const closedFill = closedColor.toLowerCase();
      const openCount = fills.filter((c) => c === openFill).length;
      const closedCount = fills.filter((c) => c === closedFill).length;
      return {
        subtitle,
        legendTitle,
        legendRows,
        filterItems,
        uniqueFills,
        openCount,
        closedCount,
        totalCircles: fills.length,
        tabActive: document.getElementById('filter-mode-opensource')?.classList.contains('active'),
      };
    }, OPEN, CLOSED);

    check('按开源 tab 高亮', ui.tabActive === true);
    check('副标题为按开源着色与筛选', ui.subtitle.includes('按开源'));
    check('图例标题为按开源着色', ui.legendTitle.includes('按开源'));
    check(
      '图例两行已开源/未开源',
      ui.legendRows.length === 2
        && ui.legendRows[0].key === 'open'
        && ui.legendRows[1].key === 'closed'
        && ui.legendRows[0].label.includes('已开源')
        && ui.legendRows[1].label.includes('未开源'),
      JSON.stringify(ui.legendRows)
    );
    check(
      '筛选勾选项为已开源/未开源且带数量',
      ui.filterItems.length === 2
        && ui.filterItems[0].value === 'open'
        && ui.filterItems[1].value === 'closed'
        && Number(ui.filterItems[0].count) > 0
        && Number(ui.filterItems[1].count) > 0,
      JSON.stringify(ui.filterItems)
    );
    check(
      '节点只使用开源/未开源两色',
      ui.uniqueFills.every((c) => c === OPEN.toLowerCase() || c === CLOSED.toLowerCase())
        && ui.openCount > 0
        && ui.closedCount > 0,
      `unique=${ui.uniqueFills.join(',')} open=${ui.openCount} closed=${ui.closedCount} total=${ui.totalCircles}`
    );

    const panelShot = path.join(outDir, 'graph-opensource-filter-panel.png');
    await page.screenshot({ path: panelShot, fullPage: false });
    try { fs.copyFileSync(panelShot, path.join(ART_DIR, 'graph-opensource-filter-panel.png')); } catch (_) {}

    await page.evaluate(() => {
      const cb = document.querySelector('#filter-panel-body input[value="open"]');
      if (cb && !cb.checked) cb.click();
    });
    await new Promise((r) => setTimeout(r, 600));

    const openOnly = await page.evaluate((openColor) => {
      const nodes = Array.from(document.querySelectorAll('#graph-canvas g.node-g'));
      const visible = nodes.filter((g) => {
        const circle = g.querySelector('circle.node-circle');
        const op = Number(circle && circle.getAttribute('fill-opacity'));
        return Number.isFinite(op) ? op > 0.2 : true;
      });
      const visibleHasRepo = visible.map((g) => {
        const d = g.__data__;
        return !!(d && d.has_repo);
      });
      const fills = visible.map((g) => (g.querySelector('circle.node-circle')?.getAttribute('fill') || '').toLowerCase());
      return {
        visible: visible.length,
        allHasRepo: visibleHasRepo.every(Boolean),
        allOpenColor: fills.every((c) => c === openColor.toLowerCase()),
        hasRepoCount: visibleHasRepo.filter(Boolean).length,
      };
    }, OPEN);
    check(
      '只勾选已开源：可见节点均 has_repo 且为 teal',
      openOnly.visible > 0 && openOnly.allHasRepo && openOnly.allOpenColor,
      JSON.stringify(openOnly)
    );

    const openOnlyShot = path.join(outDir, 'graph-opensource-filter-open-only.png');
    await page.screenshot({ path: openOnlyShot, fullPage: false });
    try { fs.copyFileSync(openOnlyShot, path.join(ART_DIR, 'graph-opensource-filter-open-only.png')); } catch (_) {}

    await page.evaluate(() => {
      const openCb = document.querySelector('#filter-panel-body input[value="open"]');
      const closedCb = document.querySelector('#filter-panel-body input[value="closed"]');
      if (openCb && openCb.checked) openCb.click();
      if (closedCb && !closedCb.checked) closedCb.click();
    });
    await new Promise((r) => setTimeout(r, 600));

    const closedOnly = await page.evaluate((closedColor) => {
      const nodes = Array.from(document.querySelectorAll('#graph-canvas g.node-g'));
      const visible = nodes.filter((g) => {
        const circle = g.querySelector('circle.node-circle');
        const op = Number(circle && circle.getAttribute('fill-opacity'));
        return Number.isFinite(op) ? op > 0.2 : true;
      });
      const visibleClosed = visible.map((g) => {
        const d = g.__data__;
        return !(d && d.has_repo);
      });
      const fills = visible.map((g) => (g.querySelector('circle.node-circle')?.getAttribute('fill') || '').toLowerCase());
      return {
        visible: visible.length,
        allClosed: visibleClosed.every(Boolean),
        allClosedColor: fills.every((c) => c === closedColor.toLowerCase()),
      };
    }, CLOSED);
    check(
      '只勾选未开源：可见节点均无 has_repo 且为 rose',
      closedOnly.visible > 0 && closedOnly.allClosed && closedOnly.allClosedColor,
      JSON.stringify(closedOnly)
    );

    const closedOnlyShot = path.join(outDir, 'graph-opensource-filter-closed-only.png');
    await page.screenshot({ path: closedOnlyShot, fullPage: false });
    try { fs.copyFileSync(closedOnlyShot, path.join(ART_DIR, 'graph-opensource-filter-closed-only.png')); } catch (_) {}

    const failed = results.filter((r) => !r.ok);
    if (failed.length) {
      console.error(`FAILED ${failed.length}/${results.length}`);
      process.exit(1);
    }
    console.log(`All ${results.length} checks passed.`);
  } finally {
    await browser.close();
  }
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
