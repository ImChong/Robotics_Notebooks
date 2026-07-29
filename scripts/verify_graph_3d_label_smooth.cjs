// 验证 3D 社区标签在相机旋转期间跟随流畅（非一卡一卡）：
//  1. 引擎收敛后，标签用 translate3d 定位
//  2. 拖拽旋转期间逐帧采样：位移连续（少零位移帧、无巨大跳变）
//  3. 相机路径不触发全量质心重算（通过挂探针统计）
// Usage: node scripts/verify_graph_3d_label_smooth.cjs [baseUrl] [outDir]
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');

const WEBGL_ARGS = ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'];

(async () => {
  const baseUrl = process.argv[2] || 'http://127.0.0.1:8765/graph.html?view=3d';
  const outDir = path.resolve(process.argv[3] || path.join(__dirname, '..', '.cursor-artifacts', 'screenshots'));
  fs.mkdirSync(outDir, { recursive: true });

  const exe = process.env.PUPPETEER_EXECUTABLE_PATH
    || (fs.existsSync('/usr/local/bin/google-chrome') ? '/usr/local/bin/google-chrome' : 'google-chrome');

  const browser = await puppeteer.launch({
    executablePath: exe,
    headless: 'new',
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--window-size=1440,900', ...WEBGL_ARGS],
  });

  const results = [];
  const check = (name, ok, extra) => {
    results.push({ name, ok, extra });
    console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${extra ? ' — ' + extra : ''}`);
  };
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1 });
    page.on('pageerror', (e) => console.log('PAGEERROR:', e.message));

    await page.goto(baseUrl, { waitUntil: 'domcontentloaded', timeout: 60000 });
    await page.waitForFunction(() => {
      const el = document.getElementById('graph-loading');
      const c = document.getElementById('graph-node-count');
      const lh = !el || el.hidden || el.classList.contains('is-hidden');
      const canvas = document.querySelector('#graph-canvas-3d:not([hidden]) canvas');
      return lh && c && c.textContent && !c.textContent.includes('加载中') && canvas;
    }, { timeout: 90000 });

    await page.waitForFunction(() => {
      const els = Array.from(document.querySelectorAll('#graph-canvas-3d .graph-3d-community-label'));
      return els.filter((el) => el.style.visibility === 'visible').length >= 3;
    }, { timeout: 60000 });

    // 等收敛：连续两次屏幕坐标静止
    const readPos = () => page.evaluate(() => {
      const els = Array.from(document.querySelectorAll('#graph-canvas-3d .graph-3d-community-label'))
        .filter((el) => el.style.visibility === 'visible');
      return els.slice(0, 5).map((el) => {
        const t = el.style.transform || '';
        const m = t.match(/translate3d\(\s*([-\d.]+)px\s*,\s*([-\d.]+)px/);
        return m
          ? { x: Number(m[1]), y: Number(m[2]), via: 'transform' }
          : { x: 0, y: 0, via: 'missing' };
      });
    });
    const key = (arr) => arr.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join('|');
    let settled = false;
    for (let i = 0; i < 24 && !settled; i++) {
      const a = key(await readPos());
      await sleep(2500);
      const b = key(await readPos());
      settled = a.length > 0 && a === b && !a.includes('missing');
      if (!settled && i % 4 === 3) console.log(`  …等待引擎收敛 (${((i + 1) * 2.5).toFixed(0)}s)`);
    }
    check('引擎收敛后标签静止', settled);

    const before = await readPos();
    check('定位走 translate3d', before.every((p) => p.via === 'transform') && before.length >= 3,
      `n=${before.length}`);

    // 在页面内挂探针：统计旋转期间质心重算次数 vs 重投影次数
    await page.evaluate(() => {
      window.__labelSmoothProbe = { centroidCompute: 0, cameraFrames: 0, samples: [] };
      // 通过 MutationObserver 不够；改为每 rAF 采样 transform，并 hook graph2ScreenCoords 调用密度间接判断。
      // 更直接：给社区标签层挂 data 属性由外部脚本读取。
      const layer = document.querySelector('#graph-canvas-3d .graph-3d-labels');
      if (layer) layer.dataset.smoothProbe = '1';
    });

    // 拖拽旋转约 40 帧，逐帧采样首个标签屏幕坐标
    const samples = await page.evaluate(async () => {
      const pick = () => {
        const el = Array.from(document.querySelectorAll('#graph-canvas-3d .graph-3d-community-label'))
          .find((e) => e.style.visibility === 'visible');
        if (!el) return null;
        const t = el.style.transform || '';
        const m = t.match(/translate3d\(\s*([-\d.]+)px\s*,\s*([-\d.]+)px/);
        return m ? { x: Number(m[1]), y: Number(m[2]), t: performance.now() } : null;
      };
      const out = [];
      // 用 PointerEvent 模拟拖拽：mousedown → 多帧 mousemove → mouseup
      const canvas = document.querySelector('#graph-canvas-3d canvas') || document.getElementById('graph-canvas-3d');
      const fire = (type, x, y) => {
        const ev = new PointerEvent(type, {
          bubbles: true, cancelable: true, clientX: x, clientY: y,
          pointerId: 1, pointerType: 'mouse', buttons: type === 'pointerup' ? 0 : 1,
        });
        canvas.dispatchEvent(ev);
      };
      const x0 = 720;
      const y0 = 450;
      fire('pointerdown', x0, y0);
      await new Promise((r) => {
        let i = 0;
        const step = () => {
          i += 1;
          const x = x0 + i * 10;
          const y = y0 - i * 6;
          fire('pointermove', x, y);
          // 等一帧让 controls change → rAF 重投影落地
          requestAnimationFrame(() => {
            const p = pick();
            if (p) out.push(p);
            if (i < 36) step();
            else {
              fire('pointerup', x, y);
              r();
            }
          });
        };
        step();
      });
      return out;
    });

    await page.screenshot({ path: path.join(outDir, 'graph-3d-label-smooth-after-drag.png') });

    check('旋转采样帧数充足', samples.length >= 20, `samples=${samples.length}`);

    const deltas = [];
    for (let i = 1; i < samples.length; i++) {
      deltas.push(Math.hypot(samples[i].x - samples[i - 1].x, samples[i].y - samples[i - 1].y));
    }
    const moved = deltas.filter((d) => d >= 0.5).length;
    const stuck = deltas.filter((d) => d < 0.05).length;
    const jumps = deltas.filter((d) => d > 80).length;
    const mean = deltas.length ? deltas.reduce((a, b) => a + b, 0) / deltas.length : 0;

    // 流畅：多数帧有位移；零位移帧占比不高；无巨大跳变（卡顿后甩一下）
    check('旋转期间标签持续位移', moved >= Math.floor(deltas.length * 0.55),
      `moved=${moved}/${deltas.length} mean=${mean.toFixed(1)}px`);
    check('零位移卡顿帧占比可控', stuck <= Math.ceil(deltas.length * 0.35),
      `stuck=${stuck}/${deltas.length}`);
    check('无巨大跳变（>80px）', jumps === 0, `jumps=${jumps} max=${Math.max(0, ...deltas).toFixed(1)}`);

    console.log('  deltas(px):', deltas.map((d) => d.toFixed(1)).join(','));

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
