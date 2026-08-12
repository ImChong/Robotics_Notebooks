// Verify change-log：默认仅新增；点击「显示维护节点」后展示维护条目。
// Usage: node scripts/verify_changelog_added_only.cjs [port] [outDir]
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');

const CHROME_CANDIDATES = [
  process.env.PUPPETEER_EXECUTABLE_PATH,
  process.env.CHROME_PATH,
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

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

(async () => {
  const port = process.argv[2] || '8765';
  const outDir = path.resolve(
    process.argv[3] || path.join(__dirname, '..', '.cursor-artifacts', 'screenshots')
  );
  fs.mkdirSync(outDir, { recursive: true });
  const base = `http://127.0.0.1:${port}`;

  const browser = await puppeteer.launch({
    executablePath: exe,
    headless: 'new',
    args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--window-size=1440,1100'],
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1440, height: 1100, deviceScaleFactor: 1 });
    await page.goto(`${base}/change-log.html`, { waitUntil: 'networkidle2', timeout: 120000 });
    await page.waitForFunction(() => {
      const mount = document.getElementById('homeLatestWikiModule');
      return (
        mount &&
        !mount.classList.contains('data-loading') &&
        mount.querySelector('.updates-day') &&
        mount.querySelector('button.updates-filter-added-only')
      );
    }, { timeout: 120000 });

    const before = await page.evaluate(() => {
      const days = Array.from(document.querySelectorAll('.updates-day'));
      const badges = Array.from(document.querySelectorAll('.updates-badge'));
      const labels = badges.map((b) => b.textContent.trim());
      return {
        dayCount: days.length,
        hasMaintained: labels.some((t) => t === '维护'),
        hasAdded: labels.some((t) => t === '新增'),
        allBadgesAreAdded: labels.length > 0 && labels.every((t) => t === '新增'),
        btnText: document.querySelector('button.updates-filter-added-only')?.textContent?.trim() || '',
        btnActive: document.querySelector('button.updates-filter-added-only')?.classList.contains('is-active'),
        ariaPressed: document.querySelector('button.updates-filter-added-only')?.getAttribute('aria-pressed') || '',
        intro: document.querySelector('.home-latest-wiki-intro')?.textContent?.trim() || '',
      };
    });

    const outAdded = path.join(outDir, 'change-log-filter-added-only.png');
    await page.screenshot({ path: outAdded, fullPage: false });

    await page.click('button.updates-filter-added-only');
    await sleep(300);
    await page.waitForFunction(() => {
      const btn = document.querySelector('button.updates-filter-added-only');
      return btn && btn.getAttribute('aria-pressed') === 'true';
    }, { timeout: 10000 });

    const after = await page.evaluate(() => {
      const days = Array.from(document.querySelectorAll('.updates-day'));
      const badges = Array.from(document.querySelectorAll('.updates-badge'));
      return {
        dayCount: days.length,
        hasMaintained: badges.some((b) => b.textContent.trim() === '维护'),
        hasAdded: badges.some((b) => b.textContent.trim() === '新增'),
        btnText: document.querySelector('button.updates-filter-added-only')?.textContent?.trim() || '',
        btnActive: document.querySelector('button.updates-filter-added-only')?.classList.contains('is-active'),
        ariaPressed: document.querySelector('button.updates-filter-added-only')?.getAttribute('aria-pressed') || '',
        intro: document.querySelector('.home-latest-wiki-intro')?.textContent?.trim() || '',
      };
    });

    const outAll = path.join(outDir, 'change-log-filter-all.png');
    await page.screenshot({ path: outAll, fullPage: false });

    // scroll filter bar into view for a tighter crop-ish shot of the control
    await page.evaluate(() => {
      document.querySelector('.updates-filter-bar')?.scrollIntoView({ block: 'start' });
    });
    await sleep(200);
    const outBar = path.join(outDir, 'change-log-filter-show-maintained-bar.png');
    await page.screenshot({ path: outBar, fullPage: false });

    const ok =
      before.allBadgesAreAdded &&
      !before.hasMaintained &&
      before.hasAdded &&
      !before.btnActive &&
      before.ariaPressed === 'false' &&
      before.btnText === '显示维护节点' &&
      before.intro.includes('仅新增') &&
      after.hasMaintained &&
      after.hasAdded &&
      after.btnActive &&
      after.ariaPressed === 'true' &&
      after.btnText === '只看新增节点';

    console.log(JSON.stringify({ before, after, shots: [outAdded, outAll, outBar], ok }, null, 2));
    if (!ok) process.exit(2);
  } finally {
    await browser.close();
  }
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
