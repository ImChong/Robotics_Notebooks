'use strict';
/**
 * iPhone 视口截图：展开 roadmap 指定 Mermaid 后截取视口或节点。
 * 用法: node scripts/screenshot_roadmap_iphone.cjs <baseUrl> <sourceMatch> <outViewportPng> [outElementPng]
 */
const fs = require('fs');
const { chromium, devices } = require('playwright');

const baseUrl = process.argv[2];
const sourceMatch = process.argv[3];
const outViewport = process.argv[4];
const outElement = process.argv[5] || '';

(async function main() {
  if (!baseUrl || !sourceMatch || !outViewport) {
    console.error('用法: node screenshot_roadmap_iphone.cjs <baseUrl> <sourceMatch> <outViewportPng> [outElementPng]');
    process.exit(2);
  }
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ ...devices['iPhone 14 Pro'] });
  const page = await context.newPage();
  const url = baseUrl.replace(/\/$/, '') + '/roadmap.html?id=roadmap-motion-control&_=' + Date.now();
  await page.goto(url, { waitUntil: 'networkidle', timeout: 120000 });
  await page.waitForSelector('#roadmapContent:not(.data-loading)', { timeout: 90000 });
  await page.evaluate(function (match) {
    var major = Array.from(document.querySelectorAll('details.roadmap-major-section')).find(function (d) {
      return Array.from(d.querySelectorAll('.mermaid')).some(function (el) {
        return (el.getAttribute('data-mermaid-source') || '').indexOf(match) >= 0;
      });
    });
    if (major) major.open = true;
    var selftest = Array.from(document.querySelectorAll('details.selftest-answers')).find(function (d) {
      return (d.innerHTML || '').indexOf(match) >= 0;
    });
    if (selftest) selftest.open = true;
  }, sourceMatch);
  await page.waitForFunction(function (match) {
    return Array.from(document.querySelectorAll('.mermaid')).some(function (el) {
      return (el.getAttribute('data-mermaid-source') || '').indexOf(match) >= 0 && el.querySelector('svg');
    });
  }, sourceMatch, { timeout: 60000 });
  await page.waitForTimeout(2500);
  var locator = page.locator('.mermaid[data-mermaid-source*="' + sourceMatch + '"]');
  await locator.waitFor({ state: 'visible', timeout: 30000 });
  await locator.scrollIntoViewIfNeeded();
  await page.waitForTimeout(600);
  await page.screenshot({ path: outViewport, fullPage: false });
  if (outElement) {
    await locator.screenshot({ path: outElement });
  }
  var diag = await page.evaluate(function (match) {
    var el = Array.from(document.querySelectorAll('.mermaid')).find(function (node) {
      return (node.getAttribute('data-mermaid-source') || '').indexOf(match) >= 0;
    });
    if (!el) return { error: 'not found' };
    var r = el.getBoundingClientRect();
    return {
      scrollY: window.scrollY,
      innerWidth: window.innerWidth,
      innerHeight: window.innerHeight,
      top: r.top,
      bottom: r.bottom,
      width: r.width,
      height: r.height,
      inView: r.top >= 0 && r.bottom <= window.innerHeight,
      text: el.innerText.replace(/\s+/g, ' ').slice(0, 120)
    };
  }, sourceMatch);
  console.error('diag:', JSON.stringify(diag));
  await browser.close();
  [outViewport, outElement].filter(Boolean).forEach(function (p) {
    console.error('wrote', p, fs.statSync(p).size, 'bytes');
  });
})().catch(function (err) {
  console.error(err);
  process.exit(1);
});
