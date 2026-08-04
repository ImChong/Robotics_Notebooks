// Verify 3D community floating labels stay readable (not tiny/huge) across viewports.
// Usage: node scripts/verify_graph_community_labels_3d_responsive.cjs [baseUrl] [outDir]
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');

const VIEWPORTS = [
  // phone: effective font should be readable (>= ~7css after zoom≈1)
  {
    name: 'phone-390',
    viewport: { width: 390, height: 844, deviceScaleFactor: 3, isMobile: true, hasTouch: true },
    expect: { fontMin: 7, fontMax: 14, effMin: 6.5, effMax: 16, heightMin: 14, heightMax: 40 },
  },
  {
    name: 'tablet-768',
    viewport: { width: 768, height: 1024, deviceScaleFactor: 2, isMobile: true, hasTouch: true },
    expect: { fontMin: 7, fontMax: 18, effMin: 7, effMax: 20, heightMin: 16, heightMax: 48 },
  },
  {
    name: 'laptop-1280',
    viewport: { width: 1280, height: 800, deviceScaleFactor: 1 },
    expect: { fontMin: 7, fontMax: 18, effMin: 7, effMax: 20, heightMin: 16, heightMax: 48 },
  },
  {
    name: 'desktop-1440',
    viewport: { width: 1440, height: 900, deviceScaleFactor: 1 },
    // design reference: ~8–16px base, viewport scale ≈ 1
    expect: { fontMin: 7.5, fontMax: 17, effMin: 7.5, effMax: 18, heightMin: 16, heightMax: 42 },
  },
  {
    name: 'ultrawide-2560',
    viewport: { width: 2560, height: 1440, deviceScaleFactor: 1 },
    // must grow vs 1440 so labels are not tiny relative to canvas
    expect: { fontMin: 9, fontMax: 22, effMin: 9, effMax: 24, heightMin: 20, heightMax: 55, largerThan: 'desktop-1440' },
  },
  {
    name: 'narrow-900',
    viewport: { width: 900, height: 700, deviceScaleFactor: 1 },
    expect: { fontMin: 7, fontMax: 16, effMin: 7, effMax: 18, heightMin: 14, heightMax: 42 },
  },
];

(async () => {
  const baseUrl = process.argv[2] || 'http://127.0.0.1:8765/graph.html?view=3d';
  const outDir = path.resolve(process.argv[3] || path.join(__dirname, '..', '.cursor-artifacts', 'screenshots'));
  fs.mkdirSync(outDir, { recursive: true });

  const exe = process.env.PUPPETEER_EXECUTABLE_PATH
    || (fs.existsSync('/usr/local/bin/google-chrome') ? '/usr/local/bin/google-chrome' : 'google-chrome');

  const browser = await puppeteer.launch({
    executablePath: exe,
    headless: 'new',
    args: ['--no-sandbox', '--disable-dev-shm-usage',
      '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
  });

  const results = [];
  const byName = {};
  const check = (name, ok, extra) => {
    results.push({ name, ok, extra });
    console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${extra ? ' — ' + extra : ''}`);
  };

  try {
    for (const cfg of VIEWPORTS) {
      const page = await browser.newPage();
      await page.setViewport(cfg.viewport);
      await page.goto(baseUrl, { waitUntil: 'domcontentloaded', timeout: 45000 });
      await page.waitForFunction(() => {
        const loading = document.getElementById('graph-loading');
        const loadingHidden = !loading || loading.hidden
          || window.getComputedStyle(loading).display === 'none';
        const canvas3d = document.querySelector('#graph-canvas-3d:not([hidden]) canvas');
        return loadingHidden && canvas3d;
      }, { timeout: 90000 });
      await page.waitForFunction(() => {
        const view = window.__RN_GRAPH3D_VIEW__;
        return !!(view && view.getBaselineCameraDist && view.getBaselineCameraDist());
      }, { timeout: 30000 }).catch(() => {});
      await page.waitForFunction(() => {
        const view = window.__RN_GRAPH3D_VIEW__;
        if (!view || typeof view.getCommunityLabelZoomScale !== 'function') return false;
        const z = view.getCommunityLabelZoomScale();
        return z > 0.85 && z < 1.2;
      }, { timeout: 20000 }).catch(() => {});
      await new Promise((r) => setTimeout(r, 1200));

      const stats = await page.evaluate(() => {
        const els = Array.from(document.querySelectorAll('#graph-canvas-3d .graph-3d-community-label'))
          .filter((el) => el.style.visibility === 'visible');
        const parseScale = (t) => {
          const m = /scale\(([-0-9.]+)\)/.exec(t || '');
          return m ? parseFloat(m[1]) : 1;
        };
        const rows = els.map((el) => {
          const fs = parseFloat(el.style.fontSize) || 0;
          const sc = parseScale(el.style.transform);
          const r = el.getBoundingClientRect();
          return { fontSize: fs, scale: sc, effectiveFont: fs * sc, height: r.height, width: r.width };
        });
        const wrap = document.getElementById('graph-canvas-3d');
        const view = window.__RN_GRAPH3D_VIEW__;
        return {
          count: rows.length,
          canvas: wrap ? { w: wrap.clientWidth, h: wrap.clientHeight } : null,
          vpScale: view && view.getCommunityLabelViewportScale
            ? view.getCommunityLabelViewportScale() : null,
          zoomScale: view && view.getCommunityLabelZoomScale
            ? view.getCommunityLabelZoomScale() : null,
          fontMin: rows.length ? Math.min(...rows.map((r) => r.fontSize)) : null,
          fontMax: rows.length ? Math.max(...rows.map((r) => r.fontSize)) : null,
          effMin: rows.length ? Math.min(...rows.map((r) => r.effectiveFont)) : null,
          effMax: rows.length ? Math.max(...rows.map((r) => r.effectiveFont)) : null,
          heightMin: rows.length ? Math.min(...rows.map((r) => r.height)) : null,
          heightMax: rows.length ? Math.max(...rows.map((r) => r.height)) : null,
        };
      });

      byName[cfg.name] = stats;
      const ex = cfg.expect;
      const inRange = (v, lo, hi) => v != null && v >= lo && v <= hi;
      check(
        `${cfg.name}: 有可见社区标签`,
        stats.count >= 10,
        `count=${stats.count} canvas=${JSON.stringify(stats.canvas)} vp=${stats.vpScale}`
      );
      check(
        `${cfg.name}: 字号落在合适区间`,
        inRange(stats.fontMin, ex.fontMin, ex.fontMax) && inRange(stats.fontMax, ex.fontMin, ex.fontMax),
        `font=${stats.fontMin}–${stats.fontMax} expect=${ex.fontMin}–${ex.fontMax}`
      );
      check(
        `${cfg.name}: 有效字号（含相机 scale）合适`,
        inRange(stats.effMin, ex.effMin, ex.effMax) && inRange(stats.effMax, ex.effMin, ex.effMax),
        `eff=${stats.effMin}–${stats.effMax} expect=${ex.effMin}–${ex.effMax}`
      );
      check(
        `${cfg.name}: 胶囊高度合适`,
        inRange(stats.heightMin, ex.heightMin, ex.heightMax)
          && inRange(stats.heightMax, ex.heightMin, ex.heightMax),
        `h=${stats.heightMin}–${stats.heightMax} expect=${ex.heightMin}–${ex.heightMax}`
      );
      if (ex.largerThan && byName[ex.largerThan]) {
        const ref = byName[ex.largerThan];
        check(
          `${cfg.name}: 大屏字号大于 ${ex.largerThan}`,
          stats.fontMax != null && ref.fontMax != null && stats.fontMax > ref.fontMax + 0.5,
          `thisMax=${stats.fontMax} refMax=${ref.fontMax}`
        );
      }

      await page.screenshot({
        path: path.join(outDir, `community-labels-responsive-${cfg.name}.png`),
      });
      await page.close();
    }

    fs.writeFileSync(
      path.join(outDir, 'community-labels-responsive-report.json'),
      JSON.stringify({ byName, results }, null, 2)
    );

    const failed = results.filter((r) => !r.ok);
    console.log(failed.length ? `\n${failed.length} 项失败` : '\n全部通过');
    process.exitCode = failed.length ? 1 : 0;
  } finally {
    await browser.close();
  }
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
