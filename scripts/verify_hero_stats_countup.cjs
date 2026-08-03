// 验证：首页 Hero 规模数字首次访问 count-up；刷新 / 新标签不重播（localStorage）。
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
      out.flag = localStorage.getItem('rn_home_hero_stats_countup_played');
      out.sessionFlag = sessionStorage.getItem('rn_home_hero_stats_countup_played');
      return out;
    });
  }

  async function clearCountUpFlags(page) {
    await page.evaluate(() => {
      try { localStorage.removeItem('rn_home_hero_stats_countup_played'); } catch (_) { /* ignore */ }
      try { sessionStorage.removeItem('rn_home_hero_stats_countup_played'); } catch (_) { /* ignore */ }
    });
  }

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1280, height: 900 });
    const errs = [];
    page.on('pageerror', (e) => errs.push(String(e)));

    // 清空持久标记，确保本次为「首次访问」
    await page.goto(base + '/index.html', { waitUntil: 'domcontentloaded', timeout: 30000 });
    await clearCountUpFlags(page);
    await page.reload({ waitUntil: 'domcontentloaded', timeout: 30000 });

    // 动画进行中：节点/边同步翻滚且仍低于终值，便于截到「翻滚中」帧
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
    const mid = await sampleHero(page);
    await page.screenshot({
      path: path.join(outDir, 'home-hero-stats-countup-mid.png'),
      fullPage: false,
    });

    // 等待翻滚结束：四个数字稳定且无 is-counting
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
    // 再采一帧确保不再跳动
    const settledA = await sampleHero(page);
    await new Promise((r) => setTimeout(r, 200));
    const settledB = await sampleHero(page);
    await page.screenshot({
      path: path.join(outDir, 'home-hero-stats-countup-done.png'),
      fullPage: false,
    });

    // 同页刷新：应直接落在终值，且全程不出现 is-counting
    await page.reload({ waitUntil: 'networkidle2', timeout: 30000 });
    await page.waitForFunction(
      () => {
        const node = document.getElementById('heroNodeCount');
        const edge = document.getElementById('heroEdgeCount');
        if (!node || !edge) return false;
        const n = parseInt(node.textContent, 10);
        const e = parseInt(edge.textContent, 10);
        return n > 100 && e > 100 && !document.querySelector('.hero-stat-num.is-counting');
      },
      { timeout: 10000 }
    );
    const earlySecond = await sampleHero(page);
    await new Promise((r) => setTimeout(r, 500));
    const lateSecond = await sampleHero(page);
    const secondHadCounting = earlySecond.heroNodeCount.counting ||
      earlySecond.heroEdgeCount.counting ||
      lateSecond.heroNodeCount.counting ||
      lateSecond.heroEdgeCount.counting;
    await page.screenshot({
      path: path.join(outDir, 'home-hero-stats-countup-second-visit.png'),
      fullPage: false,
    });

    // 新标签（共享 localStorage）：也不应重播
    const page2 = await browser.newPage();
    await page2.setViewport({ width: 1280, height: 900 });
    await page2.goto(base + '/index.html', { waitUntil: 'domcontentloaded', timeout: 30000 });
    let newTabHadCounting = false;
    const newTabTexts = new Set();
    for (let i = 0; i < 25; i++) {
      const snap = await page2.evaluate(() => {
        const node = document.getElementById('heroNodeCount');
        return {
          text: node ? String(node.textContent || '').trim() : '',
          counting: !!document.querySelector('.hero-stat-num.is-counting'),
          flag: localStorage.getItem('rn_home_hero_stats_countup_played'),
        };
      });
      newTabTexts.add(snap.text);
      if (snap.counting) newTabHadCounting = true;
      await new Promise((r) => setTimeout(r, 40));
    }
    const newTabSample = await sampleHero(page2);
    await page2.screenshot({
      path: path.join(outDir, 'home-hero-stats-countup-new-tab.png'),
      fullPage: false,
    });
    await page2.close();

    const midNode = parseInt(mid.heroNodeCount.text, 10);
    const doneNode = parseInt(settledB.heroNodeCount.text, 10);
    const midEdge = parseInt(mid.heroEdgeCount.text, 10);
    const doneEdge = parseInt(settledB.heroEdgeCount.text, 10);
    const okMid = midNode < doneNode && midEdge < doneEdge;
    const okSettled = settledA.heroNodeCount.text === settledB.heroNodeCount.text;
    const okFlag = settledB.flag === '1';
    const okSecond =
      !secondHadCounting &&
      earlySecond.heroNodeCount.text === lateSecond.heroNodeCount.text &&
      earlySecond.heroEdgeCount.text === lateSecond.heroEdgeCount.text &&
      earlySecond.heroNodeCount.text === settledB.heroNodeCount.text &&
      earlySecond.flag === '1';
    const okNewTab =
      !newTabHadCounting &&
      newTabSample.flag === '1' &&
      newTabSample.heroNodeCount.text === settledB.heroNodeCount.text &&
      newTabTexts.size === 1;

    console.log('pageerrors:', errs.length ? errs : 'none');
    console.log('MID   :', JSON.stringify(mid));
    console.log('DONE  :', JSON.stringify(settledB));
    console.log('SECOND:', JSON.stringify({ earlySecond, lateSecond, secondHadCounting }));
    console.log('NEWTAB:', JSON.stringify({ newTabSample, newTabHadCounting, texts: [...newTabTexts] }));
    console.log(
      'CHECKS:',
      JSON.stringify({
        okMid,
        okSettled,
        okFlag,
        okSecond,
        okNewTab,
        midNode,
        doneNode,
        midEdge,
        doneEdge,
      })
    );

    if (!okMid || !okSettled || !okFlag || !okSecond || !okNewTab || errs.length) {
      process.exitCode = 1;
    } else {
      console.log('OK hero stats count-up');
    }
  } finally {
    await browser.close();
  }
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
