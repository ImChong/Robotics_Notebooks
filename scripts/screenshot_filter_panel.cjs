// Screenshot graph.html with filter panel opened to verify topic section integration
const puppeteer = require('puppeteer-core');
const path = require('path');
const fs = require('fs');

(async () => {
  const [, , url, outPath, viewport, topic] = process.argv;
  const [W, H] = (viewport || '1440x900').split('x').map(Number);
  const exe = '/opt/pw-browsers/chromium-1194/chrome-linux/chrome';
  const d3Body = fs.readFileSync(path.resolve(__dirname, '..', 'node_modules', 'd3', 'dist', 'd3.min.js'));
  const browser = await puppeteer.launch({
    executablePath: exe, headless: 'new',
    args: ['--no-sandbox', '--disable-gpu', '--ignore-certificate-errors'],
    ignoreHTTPSErrors: true,
  });
  try {
    const page = await browser.newPage();
    await page.setViewport({ width: W, height: H, deviceScaleFactor: W < 500 ? 2 : 1 });
    await page.setRequestInterception(true);
    page.on('request', req => {
      if (req.url().includes('cdn.jsdelivr.net/npm/d3')) {
        req.respond({ status: 200, contentType: 'application/javascript', body: d3Body });
      } else { req.continue(); }
    });
    await page.goto(url, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => {
      const el = document.getElementById('graph-node-count');
      return el && el.textContent && !el.textContent.includes('加载中');
    }, { timeout: 25000 }).catch(()=>{});
    await new Promise(r => setTimeout(r, 4500));
    await page.evaluate(() => {
      const ld = document.getElementById('graph-loading');
      if (ld) ld.style.display = 'none';
    });
    // open filter panel
    await page.click('#filter-toggle');
    await new Promise(r => setTimeout(r, 400));
    // open the collapsed topic <details> so chips are visible / clickable
    await page.evaluate(() => {
      const det = document.getElementById('filter-topic-section');
      if (det && !det.open) det.open = true;
    });
    await new Promise(r => setTimeout(r, 300));
    // optionally select a topic chip
    if (topic && topic !== 'all') {
      await page.click(`.filter-topic-chip[data-topic="${topic}"]`);
      await new Promise(r => setTimeout(r, 1200));
    }
    await page.screenshot({ path: path.resolve(outPath), fullPage: false });
    console.log('Saved:', outPath);
  } finally {
    await browser.close();
  }
})().catch(e => { console.error(e); process.exit(1); });
