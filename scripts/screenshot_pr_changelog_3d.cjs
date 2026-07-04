// PR verification: change-log 30-day window + homepage mini-graph 3D hover tooltip.
// Usage: node scripts/screenshot_pr_changelog_3d.cjs [port] [outDir]
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');

const CHROME_CANDIDATES = [
  process.env.PUPPETEER_EXECUTABLE_PATH,
  process.env.CHROME_PATH,
  '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
  '/usr/local/bin/google-chrome',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
  '/usr/bin/chromium-browser',
].filter(Boolean);
const exe = CHROME_CANDIDATES.find((p) => fs.existsSync(p));
if (!exe) {
  console.error('No Chrome/Chromium found.');
  process.exit(1);
}

const chromeArgs = [
  '--no-sandbox',
  '--disable-gpu',
  '--disable-dev-shm-usage',
  '--window-size=1440,1200',
  '--use-gl=angle',
  '--use-angle=swiftshader',
  '--enable-unsafe-swiftshader',
];

const d3Local = path.resolve(__dirname, '..', 'node_modules', 'd3', 'dist', 'd3.min.js');
const d3Body = fs.existsSync(d3Local) ? fs.readFileSync(d3Local) : null;

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function setupPage(page) {
  await page.setViewport({ width: 1440, height: 1200, deviceScaleFactor: 1 });
  if (d3Body) {
    await page.setRequestInterception(true);
    page.on('request', (req) => {
      const u = req.url();
      if (u.includes('cdn.jsdelivr.net/npm/d3')) {
        req.respond({ status: 200, contentType: 'application/javascript', body: d3Body });
      } else {
        req.continue();
      }
    });
  }
}

async function waitChangeLogReady(page) {
  await page.waitForFunction(() => {
    const mount = document.getElementById('homeLatestWikiModule');
    return mount && !mount.classList.contains('data-loading') && mount.querySelector('.updates-day');
  }, { timeout: 120000 });
}

(async () => {
  const port = process.argv[2] || '8765';
  const outDir = path.resolve(process.argv[3] || path.join(__dirname, '..', '.cursor-artifacts', 'screenshots'));
  fs.mkdirSync(outDir, { recursive: true });
  const base = `http://127.0.0.1:${port}`;
  const shots = [];

  const browser = await puppeteer.launch({
    executablePath: exe,
    headless: 'new',
    args: chromeArgs,
  });

  try {
    // 1) change-log — default 30-day window + action buttons
    {
      const page = await browser.newPage();
      await setupPage(page);
      await page.goto(`${base}/change-log.html`, { waitUntil: 'networkidle2', timeout: 120000 });
      await waitChangeLogReady(page);
      const meta = await page.evaluate(() => ({
        intro: document.querySelector('.home-latest-wiki-intro')?.textContent?.trim() || '',
        dayCount: document.querySelectorAll('.updates-day').length,
        hasMoreBtn: !!document.querySelector('.updates-timeline-more-days'),
        hasAllBtn: !!document.querySelector('.updates-timeline-show-all'),
      }));
      const out = path.join(outDir, 'change-log-default-30d.png');
      await page.screenshot({ path: out, fullPage: true });
      shots.push({ page: 'change-log default', out, meta });
      await page.close();
    }

    // 2) change-log — expand all
    {
      const page = await browser.newPage();
      await setupPage(page);
      await page.goto(`${base}/change-log.html`, { waitUntil: 'networkidle2', timeout: 120000 });
      await waitChangeLogReady(page);
      await page.click('.updates-timeline-show-all');
      await sleep(400);
      const meta = await page.evaluate(() => ({
        intro: document.querySelector('.home-latest-wiki-intro')?.textContent?.trim() || '',
        dayCount: document.querySelectorAll('.updates-day').length,
        hasActions: !!document.querySelector('.updates-timeline-actions'),
      }));
      const out = path.join(outDir, 'change-log-expanded-all.png');
      await page.screenshot({ path: out, fullPage: true });
      shots.push({ page: 'change-log all', out, meta });
      await page.close();
    }

    // 3) index — mini-graph 3D hover tooltip
    {
      const page = await browser.newPage();
      await setupPage(page);
      await page.goto(`${base}/index.html`, { waitUntil: 'networkidle2', timeout: 120000 });
      await page.waitForSelector('#miniGraphMode3d', { timeout: 60000 });
      await page.click('#miniGraphMode3d');
      await page.waitForFunction(() => {
        const wrap = document.getElementById('mini-graph-3d');
        return wrap && !wrap.hidden && wrap.querySelector('canvas');
      }, { timeout: 120000 });
      await sleep(8000);

      const canvasBox = await page.evaluate(() => {
        const canvas = document.querySelector('#mini-graph-3d canvas');
        if (!canvas) return null;
        const r = canvas.getBoundingClientRect();
        return { x: r.x, y: r.y, width: r.width, height: r.height };
      });
      if (!canvasBox) throw new Error('3D canvas not found');

      let tooltipVisible = false;
      const gridX = 7;
      const gridY = 7;
      for (let gy = 1; gy < gridY && !tooltipVisible; gy++) {
        for (let gx = 1; gx < gridX && !tooltipVisible; gx++) {
          const x = canvasBox.x + (canvasBox.width * gx) / gridX;
          const y = canvasBox.y + (canvasBox.height * gy) / gridY;
          await page.mouse.move(x, y);
          await sleep(350);
          tooltipVisible = await page.evaluate(() => {
            const tip = document.getElementById('mini-graph-tooltip');
            return !!(tip && !tip.classList.contains('hidden') && tip.textContent.trim());
          });
        }
      }

      const tooltipMeta = await page.evaluate(() => {
        const tip = document.getElementById('mini-graph-tooltip');
        return {
          visible: !!(tip && !tip.classList.contains('hidden')),
          hasTitle: !!tip?.querySelector('.tt-title'),
          hasLink: !!tip?.querySelector('.tt-link'),
          text: tip?.textContent?.trim().slice(0, 120) || '',
        };
      });

      const section = await page.$('#mini-graph-section');
      const out = path.join(outDir, 'index-mini-graph-3d-tooltip.png');
      if (section) {
        await section.screenshot({ path: out });
      } else {
        await page.screenshot({ path: out });
      }
      shots.push({ page: 'index 3d tooltip', out, meta: { tooltipVisible, tooltipMeta, canvasBox } });
      await page.close();
    }

    const reportPath = path.join(outDir, 'pr-changelog-3d-verify.json');
    fs.writeFileSync(reportPath, JSON.stringify({ ok: true, shots }, null, 2));
    console.log(JSON.stringify({ ok: true, shots, reportPath }, null, 2));
  } finally {
    await browser.close();
  }
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
