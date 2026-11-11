// node recall.mjs <ep> [--force]   -> out/recall/<id>.png
// 「回顾」画面：一集里讲到以前几集的内容时，分镜上叠出当时的画面（发布清单「每集一个主题」）。
// 要哪些画面写在这一集的 EpisodeVideo.recall：{ id: { ep: 'ep013', scene: 5, t: 47.5 } }（scene 是分镜下标，或 'intro'；t 是那一幕的分镜时间，
// 照那一集 episodes/<ep>.py 的 [from, to] 取）。这里打开那一集的舞台，按 render.mjs scene 模式（不用配音，分镜时间 = 真实时间）放到 t，
// 截下分镜框（.stage，片头截主图 .hero）。截图由源码生成、不入库（out/ 已 gitignore），渲染这一集之前先跑一次；已有的不重截（--force 重截）。
import { chromium } from 'playwright-core';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { serve } from './serve.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const [name] = process.argv.slice(2), force = process.argv.includes('--force');
const OUT = path.join(HERE, 'out', 'recall');
fs.mkdirSync(OUT, { recursive: true });

// 同 render.mjs：episodes/<ep>/*.js 按文件名顺序装进 stage.html
function buildPage(ep, tag) {
  const dir = path.join(HERE, 'episodes', ep);
  const scripts = fs.existsSync(dir)
    ? fs.readdirSync(dir).filter((f) => f.endsWith('.js')).sort().map((f) => `episodes/${ep}/${f}`)
    : [`episodes/${ep}.js`];
  const html = fs.readFileSync(path.join(HERE, 'stage.html'), 'utf8')
    .replace('<script src="episodes/EPISODE.js"></script>', scripts.map((s) => `<script src="${s}"></script>`).join('\n'));
  const file = `stage.${ep}.${tag}.built.html`;
  fs.writeFileSync(path.join(HERE, file), html);
  return file;
}

const browser = await chromium.launch({ executablePath: process.env.CHROME || undefined });
const server = await serve(HERE);
const pg = await browser.newPage({ viewport: { width: 1080, height: 1920 }, deviceScaleFactor: 1 });
await pg.goto(server.url + buildPage(name, 'recall'));
const list = await pg.evaluate(() => (window.EpisodeVideo && window.EpisodeVideo.recall) || {});
await pg.close();

const groups = new Map();
for (const [id, r] of Object.entries(list)) {
  if (!force && fs.existsSync(path.join(OUT, id + '.png'))) continue;
  const key = r.ep + '|' + r.scene;
  if (!groups.has(key)) groups.set(key, []);
  groups.get(key).push({ id, ...r });
}
for (const items of groups.values()) {
  const { ep, scene } = items[0];
  const page = await browser.newPage({ viewport: { width: 1080, height: 1920 }, deviceScaleFactor: 1 });
  page.on('pageerror', (e) => console.error('pageerror', ep, e.message));
  await page.goto(server.url + buildPage(ep, 'recall'));
  await page.evaluate(() => window.robotsReady);
  const dur = await page.evaluate((k) => { const E = window.EpisodeVideo; return k === 'intro' ? E.introDur : E.scenes[k].dur; }, scene);
  const tl = { total: dur + 0.5, segments: [{ scene, t0: 0, t1: dur + 0.5, from: 0, to: dur + 0.5, lead: 0 }], subs: [] };
  await page.evaluate((t) => window.setup(t), tl);
  await page.evaluate(() => document.fonts.ready);
  await page.evaluate(() => window.renderAt(0.5));
  await page.evaluate(() => document.fonts.ready);
  await page.evaluate(() => window.fitStages());
  for (const it of items) {
    await page.evaluate((x) => window.renderAt(x), it.t);
    await page.locator(scene === 'intro' ? '.hero' : '.stage').first().screenshot({ path: path.join(OUT, it.id + '.png') });
    console.error(`${it.id}: ${ep} ${scene === 'intro' ? '片头' : '第 ' + (scene + 1) + ' 幕'} t = ${it.t}`);
  }
  await page.close();
}
await browser.close();
server.close();
