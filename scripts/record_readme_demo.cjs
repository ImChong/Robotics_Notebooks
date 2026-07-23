#!/usr/bin/env node
/**
 * Record the README site demo GIF (media/site-demo.gif):
 * homepage usage (entry cards → live search → mini graph preview)
 * followed by graph view basics (hover / zoom / node sidebar / 3D toggle).
 *
 * Usage:
 *   node scripts/record_readme_demo.cjs [baseUrl] [outDir]
 *   # baseUrl default: http://127.0.0.1:8765  (cd docs && python3 -m http.server 8765)
 */
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const baseUrl = (process.argv[2] || 'http://127.0.0.1:8765').replace(/\/$/, '');
const outDir = path.resolve(
  process.argv[3] || path.join(__dirname, '..', '.cursor-artifacts', 'screenshots')
);
const mediaGif = path.resolve(__dirname, '..', 'media', 'site-demo.gif');
const VIEW_W = 880;
const VIEW_H = 495;
const FPS = 6;
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

function framesToMp4(frameDir, outMp4, fps) {
  const pattern = path.join(frameDir, 'frame_%03d.png');
  execSync(
    `ffmpeg -y -framerate ${fps} -i "${pattern}" -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p -movflags +faststart "${outMp4}"`,
    { stdio: 'inherit' }
  );
}

/** Fixed caption banner so the GIF is self-explanatory without audio. */
async function setCaption(page, text) {
  await page.evaluate((t) => {
    let cap = document.getElementById('readme-demo-caption');
    if (!cap) {
      cap = document.createElement('div');
      cap.id = 'readme-demo-caption';
      cap.style.cssText = [
        'position:fixed',
        'left:50%',
        'bottom:14px',
        'transform:translateX(-50%)',
        'z-index:99999',
        'padding:8px 16px',
        'border-radius:999px',
        'background:rgba(10,14,20,0.85)',
        'color:#e6edf3',
        'font:600 15px/1.4 -apple-system,BlinkMacSystemFont,"PingFang SC","Noto Sans CJK SC",sans-serif',
        'pointer-events:none',
        'white-space:nowrap',
        'border:1px solid rgba(0,212,255,0.45)',
        'box-shadow:0 4px 16px rgba(0,0,0,0.45)',
      ].join(';');
      document.body.appendChild(cap);
    }
    cap.textContent = t;
  }, text);
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

async function recordHomeAndGraph(browser, frameDir) {
  const page = await browser.newPage();
  await page.setViewport({ width: VIEW_W, height: VIEW_H, deviceScaleFactor: 1 });

  let idx = 0;
  const shot = async () => {
    const file = path.join(frameDir, `frame_${String(idx).padStart(3, '0')}.png`);
    await page.screenshot({ path: file });
    idx += 1;
  };
  const shotN = async (n, delayMs) => {
    for (let i = 0; i < n; i++) {
      await shot();
      await sleep(delayMs);
    }
  };

  // ── Part 1: homepage ──────────────────────────────────────────────
  await page.goto(`${baseUrl}/index.html?demo=${Date.now()}`, {
    waitUntil: 'networkidle2',
    timeout: 60000,
  });
  await page.waitForSelector('.home-entry-grid', { timeout: 30000 });
  await sleep(1200);

  await setCaption(page, '① 首页：按目标选入口 — 路线 / 搜索 / 图谱');
  await shotN(7, 220);

  // Scroll down to the search section and run a live search.
  await page.evaluate(() => {
    const el = document.getElementById('wiki-search');
    if (el) el.scrollIntoView({ behavior: 'auto', block: 'start' });
  });
  await sleep(500);
  await setCaption(page, '② 全库即时搜索：输入关键词直接命中知识页');
  await shot();
  await page.click('#wikiSearchInput');
  for (const ch of 'MPC') {
    await page.keyboard.type(ch);
    await sleep(240);
    await shot();
  }
  await page.waitForFunction(
    () => {
      const box = document.getElementById('wikiSearchResults');
      return box && box.children.length > 0;
    },
    { timeout: 15000 }
  );
  await sleep(400);
  await shotN(7, 240);

  // Mini knowledge-graph preview at the bottom of the homepage.
  await page.evaluate(() => {
    const el = document.getElementById('mini-graph-section');
    if (el) el.scrollIntoView({ behavior: 'auto', block: 'start' });
  });
  await sleep(2600);
  await setCaption(page, '③ 首页图谱预览 → 「打开完整图谱」进入交互视图');
  await shotN(6, 240);

  // ── Part 2: full graph view ───────────────────────────────────────
  await page.goto(`${baseUrl}/graph.html?demo=${Date.now()}`, {
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
  await setCaption(page, '④ 知识图谱：颜色 = 技术社区，连线 = 页面互链');
  await shotN(8, 220);

  // Hover a few nodes to show the tooltip.
  await setCaption(page, '⑤ 悬停节点查看简介与关联边');
  const hoverTargets = await page.evaluate(() => {
    const nodes = Array.from(document.querySelectorAll('#graph-canvas .node-circle'));
    const pts = [];
    for (const c of nodes) {
      const r = c.getBoundingClientRect();
      if (r.width < 7) continue;
      const x = r.left + r.width / 2;
      const y = r.top + r.height / 2;
      if (x < 120 || x > window.innerWidth - 140 || y < 120 || y > window.innerHeight - 130) continue;
      pts.push({ x, y });
      if (pts.length >= 3) break;
    }
    return pts;
  });
  for (let hi = 0; hi < hoverTargets.length; hi++) {
    const p = hoverTargets[hi];
    await page.mouse.move(p.x, p.y, { steps: 8 });
    await sleep(260);
    await shotN(hi === hoverTargets.length - 1 ? 5 : 3, 220);
  }

  // Wheel zoom in / out around the centre of the canvas.
  await setCaption(page, '⑥ 滚轮缩放、拖拽平移，自由探索');
  await page.mouse.move(VIEW_W / 2, VIEW_H / 2 - 20);
  for (let i = 0; i < 5; i++) {
    await page.mouse.wheel({ deltaY: -110 });
    await sleep(140);
    await shot();
  }
  for (let i = 0; i < 3; i++) {
    await page.mouse.wheel({ deltaY: 130 });
    await sleep(140);
    await shot();
  }

  // Click a node to open the detail sidebar.
  await setCaption(page, '⑦ 点击节点：右侧详情栏 + 一键进知识页');
  const clickTarget = await page.evaluate(() => {
    const nodes = Array.from(document.querySelectorAll('#graph-canvas .node-circle'));
    let best = null;
    for (const c of nodes) {
      const r = c.getBoundingClientRect();
      if (r.width < 8) continue;
      const x = r.left + r.width / 2;
      const y = r.top + r.height / 2;
      if (x < 150 || x > window.innerWidth - 340 || y < 130 || y > window.innerHeight - 150) continue;
      const d = Math.hypot(x - window.innerWidth / 2, y - window.innerHeight / 2);
      if (!best || d < best.d) best = { x, y, d };
    }
    return best;
  });
  if (clickTarget) {
    await page.mouse.move(clickTarget.x, clickTarget.y, { steps: 6 });
    await sleep(250);
    await page.mouse.click(clickTarget.x, clickTarget.y);
    await page
      .waitForFunction(() => {
        const sb = document.getElementById('graph-sidebar');
        return sb && sb.classList.contains('open');
      }, { timeout: 6000 })
      .catch(() => {});
    // Park the cursor on empty canvas so the hover tooltip does not cover the sidebar.
    await page.mouse.move(120, VIEW_H - 120, { steps: 4 });
    await sleep(700);
    await shotN(10, 260);
    await page.evaluate(() => {
      const close = document.getElementById('sb-close');
      if (close) close.click();
    });
    await sleep(400);
  }

  // Switch to the 3D view.
  const has3d = await page.evaluate(() => {
    const btn = document.getElementById('view-mode-3d');
    if (!btn) return false;
    btn.click();
    return true;
  });
  if (has3d) {
    await setCaption(page, '⑧ 一键切换 3D 立体视图');
    await sleep(2200);
    await shotN(9, 240);
  }

  await page.close();
  return idx;
}

(async () => {
  ensureDir(outDir);
  const frameDir = path.join(outDir, 'readme-site-demo-frames');
  fs.rmSync(frameDir, { recursive: true, force: true });
  ensureDir(frameDir);

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
      `--window-size=${VIEW_W},${VIEW_H + 100}`,
    ],
  });

  try {
    console.log('Recording README site demo…');
    const frames = await recordHomeAndGraph(browser, frameDir);
    console.log(`captured ${frames} frames`);

    ensureDir(path.dirname(mediaGif));
    framesToGif(frameDir, mediaGif, FPS, VIEW_W, VIEW_H);
    const st = fs.statSync(mediaGif);
    console.log(`GIF: ${mediaGif} (${(st.size / 1024 / 1024).toFixed(2)} MB, ${frames} frames)`);

    const outMp4 = path.join(outDir, 'readme-site-demo.mp4');
    framesToMp4(frameDir, outMp4, FPS);
    console.log(`MP4 preview: ${outMp4}`);
  } finally {
    await browser.close();
  }
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
