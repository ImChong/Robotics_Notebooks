'use strict';
/**
 * 截取详情页「目录导航 + 正文主栏」并排区域，用于验证 TOC 高亮样式。
 *
 * 用法:
 *   node scripts/screenshot_detail_toc_viewport.cjs <pageUrl> <outPng> [viewportWidth] [viewportHeight]
 */
const puppeteer = require('puppeteer-core');

const pageUrl = process.argv[2];
const outPath = process.argv[3];
const viewportWidth = parseInt(process.argv[4] || '1680', 10);
const viewportHeight = parseInt(process.argv[5] || '900', 10);

const executablePath =
  process.env.PUPPETEER_EXECUTABLE_PATH ||
  process.env.CHROME ||
  '/usr/local/bin/google-chrome';

(async function main() {
  if (!pageUrl || !outPath) {
    console.error(
      '用法: node screenshot_detail_toc_viewport.cjs <url> <out.png> [width] [height]'
    );
    process.exit(2);
  }

  const browser = await puppeteer.launch({
    executablePath,
    headless: true,
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu'],
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({
      width: viewportWidth,
      height: viewportHeight,
      deviceScaleFactor: 1,
    });
    await page.goto(pageUrl, { waitUntil: 'networkidle2', timeout: 120000 });
    await page.waitForSelector('#detailTocList:not(.data-loading)', { timeout: 90000 });
    await page.waitForSelector('#detailContent:not(.data-loading)', { timeout: 90000 });
    await page.waitForSelector('.detail-toc-list a.active, .detail-toc-list .toc-entry.active', {
      timeout: 30000,
    });

    const scrollY = await page.evaluate(function () {
      const section = document.getElementById('detailContentSection');
      if (!section) return 0;
      const headerOffset = 96;
      const target = section.getBoundingClientRect().top + window.scrollY - headerOffset;
      const y = Math.max(0, target);
      window.scrollTo(0, y);
      return y;
    });
    await page.waitForFunction(
      function (expected) {
        return Math.abs(window.scrollY - expected) < 2;
      },
      { timeout: 5000 },
      scrollY
    );
    await new Promise(function (resolve) {
      setTimeout(resolve, 300);
    });

    const layoutOk = await page.evaluate(function () {
      const toc = document.getElementById('detailTocSection');
      const main = document.querySelector('.detail-content-main');
      if (!toc || !main) return false;
      const tocTop = toc.getBoundingClientRect().top;
      return window.scrollY > 100 && tocTop > 40 && tocTop < 220;
    });
    if (!layoutOk) {
      throw new Error('screenshot_detail_toc_viewport: TOC 未滚入视口，请检查 viewport 宽度（建议 ≥1680）');
    }

    await page.screenshot({ path: outPath, fullPage: false });
    console.log(outPath);
  } finally {
    await browser.close();
  }
})().catch(function (err) {
  console.error(err);
  process.exit(1);
});
