// 验证：详情页「正文内链 ↔ 关联知识图谱迷你图」浮窗联动。
//
// 断言两个方向：
//   1. 悬停正文内链 → 弹出图谱同款浮窗（标题/摘要/打开详情页），迷你图同一节点点亮（.mini-node-linked）
//   2. 悬停迷你图节点 → 正文中指向该节点的内链点亮（.detail-inline-link-linked）
//
// 前置：仓库根目录先生成站点数据并起静态服务
//   make export graph
//   cd docs && python3 -m http.server 8765
//
// 用法（仓库根目录）：
//   node scripts/verify_detail_inline_link_preview.cjs [baseUrl] [outDir] [pageId]
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');

(async () => {
  const base = process.argv[2] || 'http://127.0.0.1:8765';
  const outDir = process.argv[3] || path.resolve(__dirname, '..', '.cursor-artifacts', 'screenshots');
  const pageId = process.argv[4] || 'wiki-concepts-sim2real';
  fs.mkdirSync(outDir, { recursive: true });
  const candidates = [
    process.env.CHROME_PATH,
    '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
    '/usr/bin/chromium',
    '/usr/bin/google-chrome',
  ].filter(Boolean);
  const exe = candidates.find((p) => fs.existsSync(p));
  if (!exe) throw new Error('No Chrome/Chromium found. Set CHROME_PATH.');

  const browser = await puppeteer.launch({
    executablePath: exe, headless: 'new',
    args: ['--no-sandbox', '--disable-gpu'],
  });
  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1400, height: 1000 });
    const errs = [];
    page.on('pageerror', (e) => errs.push(String(e)));
    await page.goto(base + '/detail.html?id=' + encodeURIComponent(pageId), {
      waitUntil: 'networkidle2', timeout: 60000,
    });
    await page.waitForFunction(
      () => document.querySelectorAll('#detailContent a.detail-inline-link').length > 0,
      { timeout: 30000 }
    );
    await page.waitForFunction(
      () => document.querySelectorAll('#detailMiniMapSvg g.mini-node').length > 0,
      { timeout: 30000 }
    );
    // 等力模拟落稳 + 适配动画结束，节点坐标才可用于鼠标命中
    await new Promise((r) => setTimeout(r, 2500));

    await page.addStyleTag({ content: 'html { scroll-behavior: auto !important; }' });

    // 挑一个「正文内链 + 迷你图节点」都存在的 path，才能验证双向联动
    const target = await page.evaluate(() => {
      const miniIds = new Set(
        [...document.querySelectorAll('#detailMiniMapSvg g.mini-node')].map((g) => g.__data__ && g.__data__.id)
      );
      const link = [...document.querySelectorAll('#detailContent a.detail-inline-link')]
        .find((a) => miniIds.has(a.dataset.wikiPath) && a.getClientRects().length);
      if (!link) return null;
      link.id = 'rn-verify-inline-link';
      // 迷你图在 hero 区，滚动到内链上方留白，保证两者同屏可见
      window.scrollTo(0, link.getBoundingClientRect().top + window.scrollY - 620);
      return { wikiPath: link.dataset.wikiPath, wikiId: link.dataset.wikiId, text: link.textContent.trim() };
    });
    if (!target) throw new Error('未找到同时出现在正文内链与迷你图中的节点');
    await new Promise((r) => setTimeout(r, 400));

    // 跨行内链的 boundingRect 会并成整块，取首个行盒中心才落在文字上
    const linkPoint = await page.evaluate(() => {
      const r = document.getElementById('rn-verify-inline-link').getClientRects()[0];
      return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
    });
    await page.mouse.move(linkPoint.x, linkPoint.y);
    await new Promise((r) => setTimeout(r, 300));
    const bodyHover = await page.evaluate(() => {
      const tip = document.getElementById('detail-inline-link-tooltip');
      return {
        tooltipVisible: !tip.classList.contains('hidden'),
        tooltipTitle: (tip.querySelector('.tt-title') || {}).textContent || '',
        tooltipHasSummary: !!tip.querySelector('.tt-summary'),
        tooltipHasLink: !!tip.querySelector('.tt-link'),
        miniLinked: document.querySelectorAll('#detailMiniMapSvg g.mini-node-linked').length,
      };
    });
    await page.screenshot({ path: path.join(outDir, 'detail-inline-link-hover.png') });

    // 反向：鼠标移出正文，滚回 hero 区（迷你图所在）后悬停同一节点
    await page.mouse.move(5, 5);
    await page.evaluate(() => window.scrollTo(0, 0));
    await new Promise((r) => setTimeout(r, 400));
    const nodeBox = await page.evaluate((wikiPath) => {
      const g = [...document.querySelectorAll('#detailMiniMapSvg g.mini-node')]
        .find((el) => el.__data__ && el.__data__.id === wikiPath);
      if (!g) return null;
      const c = g.querySelector('circle').getBoundingClientRect();
      return { x: c.x + c.width / 2, y: c.y + c.height / 2 };
    }, target.wikiPath);
    if (!nodeBox) throw new Error('迷你图中找不到目标节点');
    await page.mouse.move(nodeBox.x, nodeBox.y);
    await new Promise((r) => setTimeout(r, 300));
    const miniHover = await page.evaluate(() => ({
      bodyLinked: document.querySelectorAll('#detailContent a.detail-inline-link-linked').length,
      miniTooltipVisible: !document.getElementById('detail-mini-map-tooltip').classList.contains('hidden'),
    }));
    await page.screenshot({ path: path.join(outDir, 'detail-inline-link-mini-hover.png') });

    console.log('pageerrors :', errs.length ? errs : 'none');
    console.log('target     :', JSON.stringify(target));
    console.log('BODY→MINI  :', JSON.stringify(bodyHover));
    console.log('MINI→BODY  :', JSON.stringify(miniHover));

    const ok = bodyHover.tooltipVisible && bodyHover.tooltipTitle && bodyHover.tooltipHasLink
      && bodyHover.miniLinked === 1 && miniHover.bodyLinked >= 1 && !errs.length;
    console.log(ok ? 'PASS' : 'FAIL');
    process.exitCode = ok ? 0 : 1;
  } finally {
    await browser.close();
  }
})();
