'use strict';
/**
 * 对本地 detail.html 做视口截图：等待正文与来源区渲染后，将指定 id 滚入视口再截图。
 * 供 screenshot_site_detail.sh 在传入锚点时使用（headless Chrome 的 --virtual-time-budget
 * 无法可靠等待 Mermaid / fetch 完成后再滚动）。
 *
 * 用法:
 *   node scripts/screenshot_detail_anchor_viewport.cjs <pageUrl> <outPng> <viewportHeight> <elementId>
 *
 * 环境变量:
 *   PUPPETEER_EXECUTABLE_PATH — Chromium/Chrome 可执行文件路径（默认 google-chrome）
 */
const fs = require('fs');
const puppeteer = require('puppeteer-core');

const pageUrl = process.argv[2];
const outPath = process.argv[3];
const viewportHeight = parseInt(process.argv[4] || '1000', 10);
const anchorId = process.argv[5] || 'detail-sources';

const executablePath =
  process.env.PUPPETEER_EXECUTABLE_PATH || process.env.CHROME || 'google-chrome';

(async function main() {
  if (!pageUrl || !outPath) {
    console.error('用法: node screenshot_detail_anchor_viewport.cjs <url> <out.png> <height> <elementId>');
    process.exit(2);
  }
  const browser = await puppeteer.launch({
    executablePath,
    headless: true,
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu']
  });
  try {
    const page = await browser.newPage();
    await page.setViewport({
      width: 1280,
      height: viewportHeight,
      deviceScaleFactor: 1
    });
    await page.goto(pageUrl, { waitUntil: 'networkidle2', timeout: 120000 });
    await page.waitForSelector('#detailSourceList article.card', { timeout: 90000 });
    await page.waitForSelector('#' + anchorId, { timeout: 10000 });
    await page.evaluate(function (id) {
      const el = document.getElementById(id);
      if (el) el.scrollIntoView({ block: 'start', behavior: 'instant' });
    }, anchorId);
    await new Promise(function (resolve) {
      setTimeout(resolve, 500);
    });
    await page.screenshot({ path: outPath, type: 'png' });
  } finally {
    await browser.close();
  }
  const bytes = fs.statSync(outPath).size;
  if (bytes < 10000) {
    console.error('screenshot_detail_anchor_viewport: 输出过小', outPath, bytes);
    process.exit(1);
  }
  console.error('screenshot_detail_anchor_viewport: 已生成', outPath, '(' + bytes + ' bytes)');
})().catch(function (err) {
  console.error(err);
  process.exit(1);
});
