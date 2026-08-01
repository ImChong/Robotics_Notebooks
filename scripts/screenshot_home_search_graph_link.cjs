// 截图验证：首页搜索框输入关键词时，背景图谱预览高亮相关节点、其余淡出
const puppeteer = require('puppeteer-core');
const path = require('path');
const fs = require('fs');

(async () => {
  const [, , url, outPath, query, viewport] = process.argv;
  const q = query || 'slam';
  const [W, H] = (viewport || '1440x1600').split('x').map(Number);
  const candidates = [
    process.env.CHROME_PATH,
    '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
    '/usr/bin/chromium',
    '/usr/bin/google-chrome',
  ].filter(Boolean);
  const exe = candidates.find((p) => fs.existsSync(p));
  if (!exe) throw new Error('No Chrome/Chromium found. Set CHROME_PATH.');
  const d3Body = fs.readFileSync(path.resolve(__dirname, '..', 'node_modules', 'd3', 'dist', 'd3.min.js'));
  const browser = await puppeteer.launch({
    executablePath: exe, headless: 'new',
    args: ['--no-sandbox', '--disable-gpu', '--ignore-certificate-errors'],
    ignoreHTTPSErrors: true,
  });
  try {
    const page = await browser.newPage();
    await page.setViewport({ width: W, height: H, deviceScaleFactor: 1 });
    await page.setRequestInterception(true);
    page.on('request', req => {
      if (req.url().includes('cdn.jsdelivr.net/npm/d3')) {
        req.respond({ status: 200, contentType: 'application/javascript', body: d3Body });
      } else { req.continue(); }
    });
    await page.goto(url, { waitUntil: 'domcontentloaded' });
    // 等待 mini-graph 就绪（RNMiniGraph 暴露 + 节点绘制）
    await page.waitForFunction(
      () => window.RNMiniGraph && document.querySelectorAll('#mini-graph-svg .mini-graph-node').length > 0,
      { timeout: 25000 }
    ).catch(() => {});
    await new Promise(r => setTimeout(r, 2500));
    // 输入查询触发搜索联动
    await page.type('#wikiSearchInput', q);
    await new Promise(r => setTimeout(r, 900));
    const stats = await page.evaluate(() => ({
      hit: document.querySelectorAll('#mini-graph-svg .mini-graph-node.mini-node-hit').length,
      dim: document.querySelectorAll('#mini-graph-svg .mini-graph-node.mini-node-dim').length,
      total: document.querySelectorAll('#mini-graph-svg .mini-graph-node').length,
    }));
    console.log('highlight stats:', JSON.stringify(stats));
    // 滚动到图谱预览区
    await page.evaluate(() => {
      const el = document.getElementById('mini-graph-wrap');
      if (el) el.scrollIntoView({ block: 'center' });
    });
    await new Promise(r => setTimeout(r, 600));
    fs.mkdirSync(path.dirname(path.resolve(outPath)), { recursive: true });
    await page.screenshot({ path: path.resolve(outPath), fullPage: false });
    console.log('Saved:', outPath);
  } finally {
    await browser.close();
  }
})().catch(e => { console.error(e); process.exit(1); });
