// 验证：标题 / 规模数字落到自托管 DM Mono；中文经 unicode-range 回落到 --font-sans。
//
// 前置：仓库根目录生成站点数据并起静态服务
//   make export graph
//   cd docs && python3 -m http.server 8765
//
// 用法（仓库根目录）：
//   node scripts/verify_display_font.cjs [baseUrl] [outDir]
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');

(async () => {
  const base = process.argv[2] || 'http://127.0.0.1:8765';
  const outDir = process.argv[3] || path.resolve(__dirname, '..', '.cursor-artifacts', 'screenshots');
  fs.mkdirSync(outDir, { recursive: true });
  const candidates = [
    process.env.CHROME_PATH,
    '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
    '/usr/bin/chromium',
    '/usr/bin/google-chrome',
    '/usr/bin/google-chrome-stable',
  ].filter(Boolean);
  const exe = candidates.find((p) => fs.existsSync(p));
  if (!exe) throw new Error('No Chrome/Chromium found. Set CHROME_PATH.');

  const browser = await puppeteer.launch({
    executablePath: exe,
    headless: 'new',
    args: ['--no-sandbox', '--disable-gpu'],
  });

  // 实际参与渲染的字体（DevTools Protocol），比读 computed font-family 更可信：
  // 能区分「声明了 DM Mono」与「DM Mono 真的画了这些字」。
  async function usedFonts(page, selector) {
    const client = await page.createCDPSession();
    await client.send('DOM.enable');
    await client.send('CSS.enable');
    const { root } = await client.send('DOM.getDocument');
    const { nodeId } = await client.send('DOM.querySelector', { nodeId: root.nodeId, selector });
    if (!nodeId) throw new Error('selector not found: ' + selector);
    const { fonts } = await client.send('CSS.getPlatformFontsForNode', { nodeId });
    await client.detach();
    return fonts.map((f) => ({ family: f.familyName, glyphs: f.glyphCount }));
  }

  const results = {};
  const errs = [];
  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1280, height: 900 });
    page.on('pageerror', (e) => errs.push(String(e)));
    const failedFonts = [];
    page.on('requestfailed', (r) => {
      if (r.url().includes('/vendor/fonts/')) failedFonts.push(r.url());
    });

    await page.goto(base + '/index.html', { waitUntil: 'networkidle2', timeout: 30000 });
    await page.evaluate(() => document.fonts.ready);

    // 1) @font-face 已加载，且 woff2 请求均成功
    results.loadedFaces = await page.evaluate(() =>
      [...document.fonts]
        .filter((f) => f.family === 'DM Mono')
        .map((f) => ({ weight: f.weight, status: f.status }))
        .sort((a, b) => a.weight.localeCompare(b.weight))
    );
    results.fontRequestFailures = failedFonts;

    // 2) 纯数字的规模数字用 DM Mono 渲染
    results.heroStatNum = await usedFonts(page, '#heroNodeCount');
    // 3) 中文标题不命中 DM Mono（unicode-range 回落）
    results.heroTitle = await usedFonts(page, '.hero-title');
    // 4) 品牌栏中英混排：拉丁走 DM Mono，中文走正文栈
    results.siteTitle = await usedFonts(page, '.site-title');

    await page.screenshot({ path: path.join(outDir, 'home-display-font.png'), fullPage: false });

    // 5) 详情页拉丁标题（论文名）走 DM Mono
    await page.goto(base + '/detail.html?id=wiki-entities-paper-behavior-foundation-model-humanoid', {
      waitUntil: 'networkidle2',
      timeout: 30000,
    });
    await page.evaluate(() => document.fonts.ready);
    await page.waitForFunction(
      () => !/正在加载/.test(document.getElementById('detailTitle').textContent || ''),
      { timeout: 15000 }
    );
    results.detailTitleText = await page.$eval('#detailTitle', (el) => el.textContent.trim());
    results.detailTitle = await usedFonts(page, '#detailTitle');
    await page.screenshot({ path: path.join(outDir, 'detail-display-font.png'), fullPage: false });

    const hasDM = (list) => list.some((f) => /DM Mono/i.test(f.family));
    const checks = {
      // 四个 face 按 unicode-range 各自懒加载：本页只用到拉丁 + 500 字重，
      // 其余保持 unloaded 是预期结果，故只断言「至少有一个 face 真的加载了」。
      facesLoaded:
        results.loadedFaces.length === 4 &&
        results.loadedFaces.some((f) => f.status === 'loaded'),
      noFontRequestFailure: results.fontRequestFailures.length === 0,
      // 规模数字（纯 ASCII 数字）应完全由 DM Mono 渲染
      heroStatNumIsMono: hasDM(results.heroStatNum),
      // 中文大标题不应命中 DM Mono
      heroTitleFallsBack: !hasDM(results.heroTitle),
      // 中英混排的品牌栏两种字体都出现
      siteTitleMixed: hasDM(results.siteTitle) && results.siteTitle.length >= 2,
      // 详情页拉丁标题（论文名）走 DM Mono
      detailTitleIsMono: hasDM(results.detailTitle),
      noPageError: errs.length === 0,
    };

    console.log('FONTS:', JSON.stringify(results, null, 2));
    console.log('pageerrors:', errs.length ? errs : 'none');
    console.log('CHECKS:', JSON.stringify(checks));

    if (Object.values(checks).some((v) => !v)) {
      process.exitCode = 1;
    } else {
      console.log('OK display font (DM Mono headings + CJK fallback)');
    }
  } finally {
    await browser.close();
  }
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
