// 验证：详情页「未找到」与长正文切换时主栏不再因滚动条有无而左右抖动。
//
// 断言：
//   1. found / not-found 主栏宽度一致（同属 detail-content-main）
//   2. found / not-found 主栏 left 差 < 1px（scrollbar-gutter: stable 生效）
//   3. 详情加载过程中 #detail-empty-section 不再以 88px 空 padding 占位
//
// 前置：仓库根目录先生成站点数据并起静态服务
//   make export
//   cd docs && python3 -m http.server 8765
//
// 用法（仓库根目录）：
//   node scripts/verify_detail_empty_column_stable.cjs [baseUrl]
const puppeteer = require('puppeteer-core');
const fs = require('fs');

(async () => {
  const base = process.argv[2] || 'http://127.0.0.1:8765';
  const candidates = [
    process.env.CHROME_PATH,
    '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
    '/usr/bin/chromium',
    '/usr/bin/google-chrome',
  ].filter(Boolean);
  const exe = candidates.find((p) => fs.existsSync(p));
  if (!exe) throw new Error('No Chrome/Chromium found. Set CHROME_PATH.');

  // headed：才能稳定复现经典滚动条（headless 多为 overlay，sb 宽度常为 0）
  const browser = await puppeteer.launch({
    executablePath: exe,
    headless: false,
    args: ['--no-sandbox', '--disable-gpu', '--window-size=1400,900'],
  });

  async function measure(id) {
    const page = await browser.newPage();
    await page.setViewport({ width: 1400, height: 900 });
    await page.goto(base + '/detail.html?id=' + encodeURIComponent(id), {
      waitUntil: 'networkidle0',
      timeout: 60000,
    });
    await page.waitForFunction(() => {
      const t = document.getElementById('detailTitle')?.textContent || '';
      return t && !t.includes('正在加载');
    }, { timeout: 30000 });
    await new Promise((r) => setTimeout(r, 250));
    const m = await page.evaluate(() => {
      const empty = document.querySelector('#detailEmptyState');
      const main = document.querySelector('#detailContentSection .detail-content-main');
      const col = empty && !empty.hidden
        ? document.querySelector('#detail-empty-section .detail-content-main')
        : main;
      const heroC = document.querySelector('.detail-hero .container');
      return {
        title: document.getElementById('detailTitle')?.textContent?.trim() || '',
        sb: window.innerWidth - document.documentElement.clientWidth,
        gutter: getComputedStyle(document.documentElement).scrollbarGutter,
        colLeft: col ? col.getBoundingClientRect().left : null,
        colW: col ? col.getBoundingClientRect().width : null,
        heroLeft: heroC ? heroC.getBoundingClientRect().left : null,
      };
    });
    await page.close();
    return m;
  }

  try {
    const found = await measure('wiki-concepts-sim2real');
    const notfound = await measure('missing-id-xyz-jitter-check');

    // 加载闪动：found 页 domcontentloaded 后立刻采样 empty section 高度
    const flashPage = await browser.newPage();
    await flashPage.setViewport({ width: 1400, height: 900 });
    await flashPage.goto(base + '/detail.html?id=wiki-concepts-sim2real', {
      waitUntil: 'domcontentloaded',
      timeout: 60000,
    });
    const flash = await flashPage.evaluate(async () => {
      const samples = [];
      const t0 = performance.now();
      while (performance.now() - t0 < 1500) {
        const s = document.querySelector('#detail-empty-section');
        samples.push({
          t: Math.round(performance.now() - t0),
          emptyHidden: !!s?.hidden,
          emptyH: s && !s.hidden ? Math.round(s.getBoundingClientRect().height) : 0,
          title: document.getElementById('detailTitle')?.textContent?.trim() || '',
        });
        await new Promise((r) => requestAnimationFrame(r));
      }
      return {
        maxEmptyH: Math.max(...samples.map((x) => x.emptyH)),
        anyVisible: samples.some((x) => !x.emptyHidden && x.emptyH > 0),
      };
    });
    await flashPage.close();

    const widthDelta = Math.abs((found.colW || 0) - (notfound.colW || 0));
    const leftDelta = Math.abs((found.colLeft || 0) - (notfound.colLeft || 0));
    const heroLeftDelta = Math.abs((found.heroLeft || 0) - (notfound.heroLeft || 0));

    const errors = [];
    if (!found.title || found.title.includes('未找到')) {
      errors.push('found page did not resolve: ' + found.title);
    }
    if (!notfound.title.includes('未找到')) {
      errors.push('notfound page title unexpected: ' + notfound.title);
    }
    if (widthDelta > 1) {
      errors.push('main column width mismatch: found=' + found.colW + ' notfound=' + notfound.colW);
    }
    if (leftDelta > 1) {
      errors.push('main column left jitter: delta=' + leftDelta.toFixed(2) + 'px (found left=' + found.colLeft + ', notfound left=' + notfound.colLeft + ', sb found/notfound=' + found.sb + '/' + notfound.sb + ')');
    }
    if (heroLeftDelta > 1) {
      errors.push('hero container left jitter: delta=' + heroLeftDelta.toFixed(2) + 'px');
    }
    if (flash.maxEmptyH > 0 || flash.anyVisible) {
      errors.push('empty section flashed during load: maxH=' + flash.maxEmptyH + ' anyVisible=' + flash.anyVisible);
    }
    if (!String(found.gutter || '').includes('stable')) {
      errors.push('html scrollbar-gutter is not stable: ' + found.gutter);
    }

    const report = {
      found,
      notfound,
      widthDelta,
      leftDelta,
      heroLeftDelta,
      flash,
      ok: errors.length === 0,
      errors,
    };
    console.log(JSON.stringify(report, null, 2));
    if (errors.length) {
      console.error('FAIL:', errors.join('; '));
      process.exit(1);
    }
    console.log('OK: detail empty/found column stable');
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
