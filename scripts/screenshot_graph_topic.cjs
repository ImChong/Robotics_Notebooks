// Headless screenshot of docs/graph.html, exercising the V22 topic-view dropdown.
// Usage: node screenshot_graph_topic.cjs <url> <out.png> [topicValue]
const puppeteer = require('puppeteer-core');
const path = require('path');
const fs = require('fs');

(async () => {
  const [, , url, out, topic] = process.argv;
  if (!url || !out) {
    console.error('Usage: node screenshot_graph_topic.cjs <url> <out.png> [topic]');
    process.exit(2);
  }
  const exe = process.env.PUPPETEER_EXECUTABLE_PATH || '/opt/pw-browsers/chromium-1194/chrome-linux/chrome';
  const browser = await puppeteer.launch({
    executablePath: exe,
    headless: 'new',
    args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--window-size=1440,900', '--ignore-certificate-errors'],
    ignoreHTTPSErrors: true,
  });
  // local d3 source as fallback when CDN is blocked in cloud env
  const d3LocalCandidates = [
    path.resolve(__dirname, '..', 'node_modules', 'd3', 'dist', 'd3.min.js'),
  ];
  const d3Local = d3LocalCandidates.find(p => fs.existsSync(p));
  try {
    const page = await browser.newPage();
    page.on('console', msg => {
      if (msg.type() === 'error' || msg.type() === 'warning') {
        console.log('[browser]', msg.type(), msg.text());
      }
    });
    page.on('pageerror', err => console.log('[pageerror]', err.message));
    if (d3Local) {
      const d3Body = fs.readFileSync(d3Local);
      await page.setRequestInterception(true);
      page.on('request', req => {
        const u = req.url();
        if (/\/d3@7\/dist\/d3(\.min)?\.js$/.test(u) || u.includes('cdn.jsdelivr.net/npm/d3')) {
          req.respond({
            status: 200,
            contentType: 'application/javascript',
            body: d3Body,
          });
        } else {
          req.continue();
        }
      });
    }
    await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1 });
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
    if (topic && topic !== 'all') {
      await page.select('#topic-view', topic);
      await new Promise(r => setTimeout(r, 1800));
    }
    await page.screenshot({ path: path.resolve(out), fullPage: false });
    console.log('Saved:', out);
  } finally {
    await browser.close();
  }
})().catch(e => { console.error(e); process.exit(1); });
