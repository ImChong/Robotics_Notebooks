// 验证：首页搜索结果卡片「快速预览」——长摘要默认收起（2 行 line-clamp）+ 可就地展开全文。
//
// 前置：仓库根目录先生成搜索索引并起静态服务
//   python3 scripts/build_search_index.py
//   cd docs && python3 -m http.server 8765
//
// 用法（仓库根目录）：
//   node scripts/verify_search_preview.cjs [baseUrl] [outDir] [query]
//   node scripts/verify_search_preview.cjs http://127.0.0.1:8765 .cursor-artifacts/screenshots slam
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');

(async () => {
  const base = process.argv[2] || 'http://127.0.0.1:8765';
  const outDir = process.argv[3] || path.resolve(__dirname, '..', '.cursor-artifacts', 'screenshots');
  const query = process.argv[4] || 'slam';
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
    await page.setViewport({ width: 1200, height: 1600 });
    const errs = [];
    page.on('pageerror', (e) => errs.push(String(e)));
    await page.goto(base + '/index.html', { waitUntil: 'networkidle2', timeout: 30000 });
    await page.focus('#wikiSearchInput');
    await page.type('#wikiSearchInput', query);
    await page.waitForFunction(
      () => document.querySelectorAll('#wikiSearchResults article.card').length > 0,
      { timeout: 15000 }
    );
    await page.waitForFunction(
      () => document.querySelectorAll('.result-preview-toggle').length > 0,
      { timeout: 15000 }
    );

    const collapsed = await page.evaluate(() => {
      const btn = document.querySelector('.result-preview-toggle');
      const sum = btn.closest('.card').querySelector('.result-summary');
      return {
        btnText: btn.textContent,
        ariaExpanded: btn.getAttribute('aria-expanded'),
        clamped: sum.classList.contains('is-clamped'),
        clientH: sum.clientHeight,
        scrollH: sum.scrollHeight,
        toggles: document.querySelectorAll('.result-preview-toggle').length,
        cards: document.querySelectorAll('#wikiSearchResults article.card').length,
      };
    });
    await page.screenshot({ path: path.join(outDir, 'search-preview-collapsed.png') });

    await page.click('.result-preview-toggle');
    const expanded = await page.evaluate(() => {
      const btn = document.querySelector('.result-preview-toggle');
      const sum = btn.closest('.card').querySelector('.result-summary');
      return {
        btnText: btn.textContent,
        ariaExpanded: btn.getAttribute('aria-expanded'),
        clamped: sum.classList.contains('is-clamped'),
        clientH: sum.clientHeight,
        scrollH: sum.scrollHeight,
      };
    });
    await page.screenshot({ path: path.join(outDir, 'search-preview-expanded.png') });

    console.log('pageerrors:', errs.length ? errs : 'none');
    console.log('COLLAPSED:', JSON.stringify(collapsed));
    console.log('EXPANDED :', JSON.stringify(expanded));

    const ok =
      errs.length === 0 &&
      collapsed.clamped === true &&
      collapsed.btnText === '预览全文' &&
      collapsed.ariaExpanded === 'false' &&
      collapsed.clientH < collapsed.scrollH &&
      expanded.clamped === false &&
      expanded.btnText === '收起' &&
      expanded.ariaExpanded === 'true' &&
      expanded.clientH >= collapsed.scrollH - 2;
    console.log(ok ? 'VERIFY: PASS' : 'VERIFY: FAIL');
    process.exit(ok ? 0 : 1);
  } finally {
    await browser.close();
  }
})().catch((e) => { console.error(e); process.exit(2); });
