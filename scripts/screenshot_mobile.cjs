// Screenshot graph.html at mobile viewport, cycling topics to show no wrap
const puppeteer = require('puppeteer-core');
const path = require('path');
const fs = require('fs');

(async () => {
  const [, , url, outDir] = process.argv;
  const exe = '/opt/pw-browsers/chromium-1194/chrome-linux/chrome';
  const d3Local = path.resolve(__dirname, '..', 'node_modules', 'd3', 'dist', 'd3.min.js');
  const d3Body = fs.readFileSync(d3Local);

  const browser = await puppeteer.launch({
    executablePath: exe,
    headless: 'new',
    args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--window-size=390,844', '--ignore-certificate-errors'],
    ignoreHTTPSErrors: true,
  });
  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 2 });
    await page.setRequestInterception(true);
    page.on('request', req => {
      if (req.url().includes('cdn.jsdelivr.net/npm/d3')) {
        req.respond({ status: 200, contentType: 'application/javascript', body: d3Body });
      } else { req.continue(); }
    });
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 30000 });
    await page.waitForFunction(() => {
      const el = document.getElementById('graph-node-count');
      return el && el.textContent && !el.textContent.includes('加载中');
    }, { timeout: 25000 }).catch(() => {});
    await new Promise(r => setTimeout(r, 6000));
    await page.evaluate(() => {
      const ld = document.getElementById('graph-loading');
      if (ld) ld.style.display = 'none';
    });

    // screenshot toolbar in default state (no topic chosen)
    await page.screenshot({ path: path.resolve(outDir, 'mobile-toolbar-all.png'), clip: { x:0, y:0, width:390, height:100 } });

    // activate a topic via filter panel, close, screenshot toolbar to verify badge
    await page.click('#filter-toggle');
    await new Promise(r => setTimeout(r, 300));
    await page.click('.filter-topic-chip[data-topic="motion-retargeting"]');
    await new Promise(r => setTimeout(r, 800));
    await page.click('#filter-close');
    await new Promise(r => setTimeout(r, 200));
    await page.screenshot({ path: path.resolve(outDir, 'mobile-toolbar-mr.png'), clip: { x:0, y:0, width:390, height:100 } });

    console.log('Done');
  } finally {
    await browser.close();
  }
})().catch(e => { console.error(e); process.exit(1); });
