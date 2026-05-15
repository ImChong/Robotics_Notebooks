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
    // 骨架屏同样是 article.card，不能用来判断「已写入真实来源」；以去掉 data-loading 为准。
    await page.waitForSelector('#detailSourceList:not(.data-loading)', { timeout: 90000 });
    await page.waitForFunction(
      function () {
        const root = document.getElementById('detailSourceList');
        if (!root) return false;
        return (
          root.querySelector('.detail-source-url') !== null ||
          /暂无来源|无可展示/.test(root.textContent || '')
        );
      },
      { timeout: 90000 }
    );
    await page.waitForSelector('#' + anchorId, { timeout: 10000 });

    async function scrollAnchorIntoView() {
      await page.evaluate(function (id) {
        const el = document.getElementById(id);
        if (el) el.scrollIntoView({ block: 'start', behavior: 'instant' });
      }, anchorId);
    }

    await scrollAnchorIntoView();
    await new Promise(function (resolve) {
      setTimeout(resolve, 600);
    });
    await scrollAnchorIntoView();

    const anchorOk = await page.evaluate(function (id) {
      const el = document.getElementById(id);
      if (!el) return false;
      const r = el.getBoundingClientRect();
      // scrollIntoView(block:start) 在吸顶 header 下常见 top≈200~320，仍算「来源区已进入视口」
      return r.top >= -80 && r.top < 520 && r.bottom > 120;
    }, anchorId);
    if (!anchorOk) {
      throw new Error(
        'screenshot_detail_anchor_viewport: 锚点 #' +
          anchorId +
          ' 未滚入视口上部，请检查页面 id 或等待条件。'
      );
    }

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
