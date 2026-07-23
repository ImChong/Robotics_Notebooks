#!/usr/bin/env node
/**
 * Record mobile Chrome toolbar-hide viewport fix verification (MP4),
 * and regenerate README demo GIF (media/graph-demo.gif).
 *
 * Usage:
 *   node scripts/record_graph_viewport_and_demo.cjs [baseUrl] [outDir]
 */
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const baseUrl = process.argv[2] || 'http://127.0.0.1:8765/graph.html';
const outDir = path.resolve(
  process.argv[3] || path.join(__dirname, '..', '.cursor-artifacts', 'screenshots')
);
const mediaGif = path.resolve(__dirname, '..', 'media', 'graph-demo.gif');
const chromePath =
  process.env.PUPPETEER_EXECUTABLE_PATH ||
  (fs.existsSync('/usr/local/bin/google-chrome')
    ? '/usr/local/bin/google-chrome'
    : 'google-chrome');

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function framesToMp4(frameDir, outMp4, fps) {
  const pattern = path.join(frameDir, 'frame_%03d.png');
  execSync(
    `ffmpeg -y -framerate ${fps} -i "${pattern}" -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p -movflags +faststart "${outMp4}"`,
    { stdio: 'inherit' }
  );
}

function framesToGif(frameDir, outGif, fps, width, height) {
  const pattern = path.join(frameDir, 'frame_%03d.png');
  const palette = path.join(frameDir, 'palette.png');
  execSync(
    `ffmpeg -y -framerate ${fps} -i "${pattern}" -vf "scale=${width}:${height}:flags=lanczos,palettegen=max_colors=128:stats_mode=diff" "${palette}"`,
    { stdio: 'inherit' }
  );
  execSync(
    `ffmpeg -y -framerate ${fps} -i "${pattern}" -i "${palette}" -lavfi "scale=${width}:${height}:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3" -loop 0 "${outGif}"`,
    { stdio: 'inherit' }
  );
}

async function waitGraphReady(page) {
  await page.waitForFunction(() => {
    const count = document.getElementById('graph-node-count');
    const loading = document.getElementById('graph-loading');
    const readyCount = count && count.textContent && !/加载|失败/.test(count.textContent);
    const loadingGone = !loading || loading.hidden || loading.classList.contains('is-hidden');
    return readyCount && loadingGone;
  }, { timeout: 120000 });
}

async function injectHud(page) {
  await page.evaluate(() => {
    let hud = document.getElementById('viewport-verify-hud');
    if (!hud) {
      hud = document.createElement('div');
      hud.id = 'viewport-verify-hud';
      hud.style.cssText = [
        'position:fixed',
        'left:10px',
        'top:calc(var(--header-h) + 54px)',
        'z-index:9999',
        'padding:8px 10px',
        'border-radius:8px',
        'background:rgba(0,0,0,0.72)',
        'color:#e6edf3',
        'font:600 12px/1.45 -apple-system,BlinkMacSystemFont,sans-serif',
        'pointer-events:none',
        'backdrop-filter:blur(6px)',
        'border:1px solid rgba(0,212,255,0.35)',
        'max-width:78%',
      ].join(';');
      document.body.appendChild(hud);
    }
    window.__updateViewportHud = function () {
      const wrap = document.getElementById('graph-wrap');
      const rect = wrap ? wrap.getBoundingClientRect() : null;
      const gap = rect ? Math.round(window.innerHeight - rect.bottom) : null;
      const appVh = getComputedStyle(document.documentElement).getPropertyValue('--app-vh').trim();
      hud.innerHTML =
        '<div>视口 innerHeight = <b style="color:#7dd3fc">' + window.innerHeight + 'px</b></div>' +
        '<div>壳层 bottom gap = <b style="color:' + (gap === 0 ? '#34d399' : '#f87171') + '">' + gap + 'px</b></div>' +
        '<div>--app-vh = ' + (appVh || '(dvh)') + '</div>';
      return { gap, innerHeight: window.innerHeight, wrapBottom: rect ? Math.round(rect.bottom) : null };
    };
    return window.__updateViewportHud();
  });
}

async function captureSequence(page, frameDir, count, delayMs, onTick) {
  ensureDir(frameDir);
  for (let i = 0; i < count; i++) {
    if (onTick) await onTick(i);
    const file = path.join(frameDir, `frame_${String(i).padStart(3, '0')}.png`);
    await page.screenshot({ path: file });
    if (i + 1 < count) await sleep(delayMs);
  }
}

async function recordMobileVerify(browser) {
  const page = await browser.newPage();
  const shortH = 700;
  const tallH = 844;
  await page.setViewport({ width: 390, height: shortH, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
  await page.goto(baseUrl + (baseUrl.includes('?') ? '&' : '?') + 'v=' + Date.now(), {
    waitUntil: 'domcontentloaded',
    timeout: 60000,
  });
  await waitGraphReady(page);
  await sleep(2500);
  await page.evaluate(() => {
    const btn = document.getElementById('fit-to-screen');
    if (btn) btn.click();
  });
  await sleep(900);
  await injectHud(page);

  const frameDir = path.join(outDir, 'viewport-verify-frames');
  fs.rmSync(frameDir, { recursive: true, force: true });
  ensureDir(frameDir);

  // Phase A: toolbar visible (short viewport)
  await page.evaluate(() => {
    const el = document.getElementById('viewport-verify-hud');
    if (el) {
      const tag = document.createElement('div');
      tag.id = 'viewport-verify-phase';
      tag.style.cssText = 'margin-top:6px;color:#fbbf24';
      tag.textContent = '阶段 A：模拟 Chrome 底栏可见';
      el.appendChild(tag);
    }
    window.__updateViewportHud();
  });
  await captureSequence(page, frameDir, 10, 180, async () => {
    await page.evaluate(() => window.__updateViewportHud && window.__updateViewportHud());
  });

  // Phase B: toolbar hides → taller viewport + user taps graph
  let frameIdx = 10;
  await page.setViewport({ width: 390, height: tallH, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
  await page.waitForFunction((h) => window.innerHeight >= h - 4, {}, tallH);
  await page.evaluate(() => {
    const phase = document.getElementById('viewport-verify-phase');
    if (phase) phase.textContent = '阶段 B：底栏隐藏 → 点按画布同步壳层';
    window.dispatchEvent(new Event('resize'));
    window.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, clientX: 200, clientY: 420 }));
  });
  await page.waitForFunction(() => {
    const wrap = document.getElementById('graph-wrap');
    if (!wrap) return false;
    return Math.abs(window.innerHeight - wrap.getBoundingClientRect().bottom) <= 2;
  }, { timeout: 5000 });
  await sleep(300);

  for (let i = 0; i < 14; i++) {
    await page.evaluate(() => window.__updateViewportHud && window.__updateViewportHud());
    const file = path.join(frameDir, `frame_${String(frameIdx).padStart(3, '0')}.png`);
    await page.screenshot({ path: file });
    frameIdx += 1;
    await sleep(160);
  }

  const metrics = await page.evaluate(() => window.__updateViewportHud());
  const outMp4 = path.join(outDir, 'graph-mobile-viewport-fix.mp4');
  framesToMp4(frameDir, outMp4, 8);
  await page.close();
  return { outMp4, metrics, frames: frameIdx };
}

async function recordReadmeDemo(browser) {
  const page = await browser.newPage();
  await page.setViewport({ width: 800, height: 450, deviceScaleFactor: 1 });
  await page.goto(baseUrl + (baseUrl.includes('?') ? '&' : '?') + 'demo=' + Date.now(), {
    waitUntil: 'domcontentloaded',
    timeout: 60000,
  });
  await waitGraphReady(page);
  await sleep(2800);
  await page.evaluate(() => {
    const btn = document.getElementById('fit-to-screen');
    if (btn) btn.click();
  });
  await sleep(1000);

  const frameDir = path.join(outDir, 'readme-demo-frames');
  fs.rmSync(frameDir, { recursive: true, force: true });
  ensureDir(frameDir);

  let idx = 0;
  const shot = async () => {
    const file = path.join(frameDir, `frame_${String(idx).padStart(3, '0')}.png`);
    await page.screenshot({ path: file });
    idx += 1;
  };

  // Settle / gentle pan of layout
  for (let i = 0; i < 8; i++) {
    await shot();
    await sleep(140);
  }

  // Hover a few nodes via DOM centers
  const hoverTargets = await page.evaluate(() => {
    const nodes = Array.from(document.querySelectorAll('#graph-canvas .node-circle')).slice(0, 40);
    const pts = [];
    for (const c of nodes) {
      const r = c.getBoundingClientRect();
      if (r.width < 2) continue;
      pts.push({ x: r.left + r.width / 2, y: r.top + r.height / 2 });
      if (pts.length >= 5) break;
    }
    return pts;
  });

  for (const p of hoverTargets) {
    await page.mouse.move(p.x, p.y, { steps: 6 });
    await sleep(220);
    await shot();
    await sleep(180);
    await shot();
  }

  // Zoom in / out with wheel at center
  await page.mouse.move(400, 240);
  for (let i = 0; i < 6; i++) {
    await page.mouse.wheel({ deltaY: -90 });
    await sleep(120);
    await shot();
  }
  for (let i = 0; i < 4; i++) {
    await page.mouse.wheel({ deltaY: 110 });
    await sleep(120);
    await shot();
  }

  // Fit again
  await page.evaluate(() => {
    const btn = document.getElementById('fit-to-screen');
    if (btn) btn.click();
  });
  await sleep(700);
  for (let i = 0; i < 4; i++) {
    await shot();
    await sleep(150);
  }

  // Try 3D toggle if available
  const has3d = await page.evaluate(() => {
    const btn = document.getElementById('view-mode-3d') || document.getElementById('physics-toggle');
    if (!btn) return false;
    if (btn.id === 'physics-toggle') {
      btn.click();
      const b3 = document.getElementById('view-mode-3d');
      if (!b3) return false;
      b3.click();
      return true;
    }
    btn.click();
    return true;
  });
  if (has3d) {
    await sleep(1800);
    for (let i = 0; i < 10; i++) {
      await shot();
      await sleep(160);
    }
    await page.evaluate(() => {
      const b2 = document.getElementById('view-mode-2d');
      if (b2) b2.click();
    });
    await sleep(900);
    for (let i = 0; i < 4; i++) {
      await shot();
      await sleep(140);
    }
  }

  framesToGif(frameDir, mediaGif, 8, 800, 450);
  const st = fs.statSync(mediaGif);
  await page.close();
  return { outGif: mediaGif, frames: idx, bytes: st.size };
}

(async () => {
  ensureDir(outDir);
  const browser = await puppeteer.launch({
    executablePath: chromePath,
    headless: 'new',
    args: [
      '--no-sandbox',
      '--disable-dev-shm-usage',
      '--disable-gpu',
      '--use-gl=angle',
      '--use-angle=swiftshader',
      '--enable-unsafe-swiftshader',
      '--window-size=900,700',
    ],
  });

  try {
    console.log('Recording mobile viewport verification…');
    const verify = await recordMobileVerify(browser);
    console.log('verify', verify);

    console.log('Recording README demo GIF…');
    const demo = await recordReadmeDemo(browser);
    console.log('demo', demo);

    const report = { verify, demo, at: new Date().toISOString() };
    fs.writeFileSync(path.join(outDir, 'graph-viewport-demo-record.json'), JSON.stringify(report, null, 2));
    if (!verify.metrics || verify.metrics.gap !== 0) {
      console.error('FAIL: verification gap is not 0', verify.metrics);
      process.exitCode = 1;
    } else {
      console.log('PASS');
    }
  } finally {
    await browser.close();
  }
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
