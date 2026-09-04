// 验证：路线详情页本库超链接悬停浮窗（正文 / 知识地图 / 阶段相关项）。
//
// 断言：
//   1. 正文内链悬停 → 图谱同款浮窗（标题 / 打开详情页）
//   2. 知识地图叶子悬停 → 同样弹出浮窗（若该路线有知识地图）
//
// 前置：仓库根目录先生成站点数据并起静态服务
//   make export graph
//   cd docs && python3 -m http.server 8765
//
// 用法（仓库根目录）：
//   node scripts/verify_roadmap_inline_link_preview.cjs [baseUrl] [outDir] [pageId]
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');

(async () => {
  const base = process.argv[2] || 'http://127.0.0.1:8765';
  const outDir = process.argv[3] || path.resolve(__dirname, '..', '.cursor-artifacts', 'screenshots');
  const pageId = process.argv[4] || 'roadmap-motion-control';
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
    await page.goto(base + '/roadmap.html?id=' + encodeURIComponent(pageId), {
      waitUntil: 'networkidle2', timeout: 60000,
    });
    await page.waitForFunction(
      () => document.querySelectorAll('a.detail-inline-link').length > 0,
      { timeout: 30000 }
    );
    await page.addStyleTag({ content: 'html { scroll-behavior: auto !important; }' });
    const titledLeaves = await page.evaluate(() =>
      [...document.querySelectorAll('a.detail-inline-link[title]')].map((a) => a.textContent.trim())
    );
    if (titledLeaves.length) {
      throw new Error('本库内链仍带原生 title，会与浮窗重叠: ' + titledLeaves.slice(0, 5).join(' | '));
    }

    async function hoverFirst(selector, shotName) {
      const found = await page.evaluate((sel) => {
        const currentId = decodeURIComponent(new URLSearchParams(location.search).get('id') || '');
        const link = [...document.querySelectorAll(sel)].find((a) => {
          if (a.dataset.wikiId === currentId) return false;
          const details = a.closest('details');
          if (details) details.open = true;
          return a.getClientRects().length > 0;
        });
        if (!link) return null;
        link.id = 'rn-verify-inline-link';
        window.scrollTo(0, link.getBoundingClientRect().top + window.scrollY - 220);
        return {
          wikiPath: link.dataset.wikiPath,
          wikiId: link.dataset.wikiId,
          text: link.textContent.trim().replace(/\s+/g, ' '),
        };
      }, selector);
      if (!found) return null;
      await new Promise((r) => setTimeout(r, 300));
      const linkPoint = await page.evaluate(() => {
        const r = document.getElementById('rn-verify-inline-link').getClientRects()[0];
        return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
      });
      await page.mouse.move(linkPoint.x, linkPoint.y);
      await new Promise((r) => setTimeout(r, 300));
      const hover = await page.evaluate(() => {
        const tip = document.getElementById('detail-inline-link-tooltip');
        return {
          tooltipVisible: tip && !tip.classList.contains('hidden'),
          tooltipTitle: ((tip && tip.querySelector('.tt-title')) || {}).textContent || '',
          tooltipHasType: !!(tip && tip.querySelector('.tt-type')),
          tooltipHasLink: !!(tip && tip.querySelector('.tt-link')),
          nativeTitle: document.getElementById('rn-verify-inline-link').getAttribute('title') || '',
        };
      });
      await page.screenshot({ path: path.join(outDir, shotName) });
      await page.mouse.move(5, 5);
      await page.evaluate(() => {
        const marked = document.getElementById('rn-verify-inline-link');
        if (marked) marked.removeAttribute('id');
      });
      return { target: found, hover };
    }

    const body = await hoverFirst('#roadmapContent a.detail-inline-link', 'roadmap-inline-link-body-hover.png');
    if (!body) throw new Error('未找到路线正文本库内链');

    const kmap = await hoverFirst(
      '#roadmapKnowledgeMapTree a.detail-inline-link',
      'roadmap-inline-link-kmap-hover.png'
    );

    console.log('pageerrors :', errs.length ? errs : 'none');
    console.log('BODY       :', JSON.stringify(body));
    console.log('KMAP       :', JSON.stringify(kmap));

    const bodyOk = body.hover.tooltipVisible && body.hover.tooltipTitle && body.hover.tooltipHasLink
      && !body.hover.nativeTitle;
    const kmapOk = !kmap || (kmap.hover.tooltipVisible && kmap.hover.tooltipTitle && kmap.hover.tooltipHasLink
      && !kmap.hover.nativeTitle);
    const ok = bodyOk && kmapOk && !errs.length;
    console.log(ok ? 'PASS' : 'FAIL');
    process.exitCode = ok ? 0 : 1;
  } finally {
    await browser.close();
  }
})();
