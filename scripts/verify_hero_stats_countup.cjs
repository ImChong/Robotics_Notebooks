// 验证：首页 Hero 规模数字每次加载/刷新都 count-up。
//
// 前置：仓库根目录生成 home-stats 并起静态服务
//   make export graph   # 或至少有 docs/exports/home-stats.json
//   cd docs && python3 -m http.server 8765
//
// 用法（仓库根目录）：
//   node scripts/verify_hero_stats_countup.cjs [baseUrl] [outDir]
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');

(async () => {
  const base = process.argv[2] || 'http://127.0.0.1:8765';
  const outDir = process.argv[3] || path.resolve(__dirname, '..', '.cursor-artifacts', 'screenshots');
  fs.mkdirSync(outDir, { recursive: true });
  const candidates = [
    process.env.CHROME_PATH,
    '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
    '/usr/bin/chromium',
    '/usr/bin/google-chrome',
    '/usr/bin/google-chrome-stable',
  ].filter(Boolean);
  const exe = candidates.find((p) => fs.existsSync(p));
  if (!exe) throw new Error('No Chrome/Chromium found. Set CHROME_PATH.');

  const browser = await puppeteer.launch({
    executablePath: exe,
    headless: 'new',
    args: ['--no-sandbox', '--disable-gpu'],
  });

  async function sampleHero(page) {
    return page.evaluate(() => {
      const ids = ['heroNodeCount', 'heroEdgeCount', 'heroMainRouteCount', 'heroDepthRouteCount'];
      const out = {};
      for (const id of ids) {
        const el = document.getElementById(id);
        out[id] = el
          ? {
              text: String(el.textContent || '').trim(),
              counting: el.classList.contains('is-counting'),
            }
          : null;
      }
      return out;
    });
  }

  async function waitForCountUpMid(page) {
    await page.waitForFunction(
      () => {
        const node = document.getElementById('heroNodeCount');
        const edge = document.getElementById('heroEdgeCount');
        if (!node || !edge) return false;
        const n = parseInt(node.textContent, 10);
        const e = parseInt(edge.textContent, 10);
        const counting = document.querySelectorAll('.hero-stat-num.is-counting').length;
        return (
          Number.isFinite(n) &&
          Number.isFinite(e) &&
          n > 200 &&
          n < 1800 &&
          e > 1000 &&
          e < 16000 &&
          counting >= 2
        );
      },
      { timeout: 8000 }
    );
  }

  async function waitForCountUpDone(page) {
    await page.waitForFunction(
      () => {
        const ids = ['heroNodeCount', 'heroEdgeCount', 'heroMainRouteCount', 'heroDepthRouteCount'];
        for (const id of ids) {
          const el = document.getElementById(id);
          if (!el || el.classList.contains('is-counting')) return false;
          if (!/^\d+$/.test(String(el.textContent || '').trim())) return false;
        }
        const node = parseInt(document.getElementById('heroNodeCount').textContent, 10);
        const edge = parseInt(document.getElementById('heroEdgeCount').textContent, 10);
        return node > 100 && edge > 100;
      },
      { timeout: 10000 }
    );
  }

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1280, height: 900 });
    const errs = [];
    page.on('pageerror', (e) => errs.push(String(e)));

    await page.goto(base + '/index.html', { waitUntil: 'domcontentloaded', timeout: 30000 });

    // 首次加载：翻滚中
    await waitForCountUpMid(page);
    const mid = await sampleHero(page);
    await page.screenshot({
      path: path.join(outDir, 'home-hero-stats-countup-mid.png'),
      fullPage: false,
    });

    await waitForCountUpDone(page);
    const settledA = await sampleHero(page);
    await new Promise((r) => setTimeout(r, 200));
    const settledB = await sampleHero(page);
    await page.screenshot({
      path: path.join(outDir, 'home-hero-stats-countup-done.png'),
      fullPage: false,
    });

    // 刷新后应再次翻滚
    await page.reload({ waitUntil: 'domcontentloaded', timeout: 30000 });
    await waitForCountUpMid(page);
    const midReload = await sampleHero(page);
    await page.screenshot({
      path: path.join(outDir, 'home-hero-stats-countup-reload-mid.png'),
      fullPage: false,
    });
    await waitForCountUpDone(page);
    const settledReload = await sampleHero(page);
    await page.screenshot({
      path: path.join(outDir, 'home-hero-stats-countup-reload-done.png'),
      fullPage: false,
    });

    const midNode = parseInt(mid.heroNodeCount.text, 10);
    const doneNode = parseInt(settledB.heroNodeCount.text, 10);
    const midEdge = parseInt(mid.heroEdgeCount.text, 10);
    const doneEdge = parseInt(settledB.heroEdgeCount.text, 10);
    const reloadMidNode = parseInt(midReload.heroNodeCount.text, 10);
    const reloadDoneNode = parseInt(settledReload.heroNodeCount.text, 10);
    const reloadMidEdge = parseInt(midReload.heroEdgeCount.text, 10);
    const reloadDoneEdge = parseInt(settledReload.heroEdgeCount.text, 10);

    const okMid = midNode < doneNode && midEdge < doneEdge;
    const okSettled = settledA.heroNodeCount.text === settledB.heroNodeCount.text;
    const okReload =
      midReload.heroNodeCount.counting &&
      midReload.heroEdgeCount.counting &&
      reloadMidNode < reloadDoneNode &&
      reloadMidEdge < reloadDoneEdge &&
      settledReload.heroNodeCount.text === settledB.heroNodeCount.text;

    console.log('pageerrors:', errs.length ? errs : 'none');
    console.log('MID   :', JSON.stringify(mid));
    console.log('DONE  :', JSON.stringify(settledB));
    console.log('RELOAD:', JSON.stringify({ midReload, settledReload }));
    console.log(
      'CHECKS:',
      JSON.stringify({
        okMid,
        okSettled,
        okReload,
        midNode,
        doneNode,
        midEdge,
        doneEdge,
        reloadMidNode,
        reloadDoneNode,
      })
    );

    if (!okMid || !okSettled || !okReload || errs.length) {
      process.exitCode = 1;
    } else {
      console.log('OK hero stats count-up (plays every load/refresh)');
    }
  } finally {
    await browser.close();
  }
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
