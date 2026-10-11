// node render.mjs <ep> stills 1,20,40        -> out/<ep>/still_<t>.png
// node render.mjs <ep> video [fps]            -> out/<ep>/<ep>_video.mp4
// node render.mjs <ep> cover                  -> out/<ep>/cover.png（封面版式、无字幕）
// node render.mjs <ep> clip 140.5,191 [fps]   -> out/<ep>/clip_140.5-191.mp4（只渲染这一段，配同一段旁白）
// node render.mjs <ep> scene 3 0,5,12.5        -> out/<ep>/scene3_<t>.png（不用配音：第 3 幕按分镜时间 1 倍速播放，抽这几个时刻；
//                                               也可写 intro / outro。做分镜时先看画面用）
// <ep> 对应 episodes/<ep>.{js,py}。改自 Robot_Learning_Paper_Notebooks 的 scripts/paper_video/render.mjs。
import { chromium } from 'playwright-core';
import { spawn, execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FFMPEG = execFileSync('python3', ['-c', 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())']).toString().trim();
const [name, mode, arg] = process.argv.slice(2);
const OUT = path.join(HERE, 'out', name);
// a cover needs no narration: without build.py's timeline, fake a 30 s intro so every block has faded in
const tlPath = path.join(OUT, 'timeline.json');
let tl = mode === 'scene' ? null : fs.existsSync(tlPath) ? JSON.parse(fs.readFileSync(tlPath, 'utf8'))
  : mode === 'cover' ? { total: 30, segments: [{ scene: 'intro', t0: 0, t1: 30, from: 0, to: 30, lead: 0 }], subs: [] }
  : JSON.parse(fs.readFileSync(tlPath, 'utf8'));
fs.mkdirSync(OUT, { recursive: true });

// 一集可以是一个文件 episodes/<ep>.js，也可以拆成目录 episodes/<ep>/*.js（按文件名顺序加载，最后一个设 window.EpisodeVideo）
const epDir = path.join(HERE, 'episodes', name);
const scripts = fs.existsSync(epDir)
  ? fs.readdirSync(epDir).filter((f) => f.endsWith('.js')).sort().map((f) => `episodes/${name}/${f}`)
  : [`episodes/${name}.js`];
const html = fs.readFileSync(path.join(HERE, 'stage.html'), 'utf8')
  .replace('<script src="episodes/EPISODE.js"></script>', scripts.map((s) => `<script src="${s}"></script>`).join('\n'));
// 每集一份（scene 模式每幕一份），几个进程可以同时渲染
const built = path.join(HERE, mode === 'scene' ? `stage.${name}.scene${arg}.built.html` : `stage.${name}.built.html`);
fs.writeFileSync(built, html);

const browser = await chromium.launch({ executablePath: process.env.CHROME || undefined });
const page = await browser.newPage({ viewport: { width: 1080, height: 1920 }, deviceScaleFactor: 1 });
page.on('pageerror', (e) => console.error('pageerror', e.message));
page.on('console', (m) => { if (m.type() === 'error') console.error('console', m.text()); });
await page.goto('file://' + built);
if (mode === 'scene') {
  // 只放这一幕：分镜时间 = 真实时间，时长取分镜里写的 dur（片头 / 片尾取 introDur / outroDur）
  const key = /^\d+$/.test(arg) ? Number(arg) : arg;
  const dur = await page.evaluate((k) => { const E = window.EpisodeVideo; return k === 'intro' ? E.introDur : k === 'outro' ? E.outroDur : E.scenes[k].dur; }, key);
  tl = { total: dur + 0.5, segments: [{ scene: key, t0: 0, t1: dur + 0.5, from: 0, to: dur + 0.5, lead: 0 }], subs: [] };
}
await page.evaluate((t) => window.setup(t), tl);
await page.evaluate(() => document.fonts.ready);
// warm every scene once so KaTeX + fonts are laid out before the first real frame
for (const s of tl.segments) await page.evaluate((t) => window.renderAt(t), s.t0 + 0.5);
await page.evaluate(() => document.fonts.ready);
// zoom each storyboard to what it draws, now that text has its real widths
for (const [i, f] of (await page.evaluate(() => window.fitStages())).entries()) {
  console.error(`scene ${i + 1}: viewBox ${f.viewBox}  zoom ${f.zoom.toFixed(3)}`);
}

function encoder(mp4, extra, crf, from, to) {
  const audio = from === undefined ? ['-i', path.join(OUT, 'narration.wav')]
    : ['-ss', String(from), '-t', String(to - from), '-i', path.join(OUT, 'narration.wav')];
  return spawn(FFMPEG, ['-y', '-loglevel', 'error',
    '-f', 'image2pipe', '-framerate', String(extra.fps), '-c:v', 'mjpeg', '-i', '-', ...audio,
    '-c:v', 'libx264', '-preset', 'medium', '-crf', String(crf), '-vf', 'scale=in_range=pc:out_range=tv', '-pix_fmt', 'yuv420p',
    '-color_range', 'tv', '-r', String(extra.fps),
    '-c:a', 'aac', '-b:a', '160k', '-ar', '48000', '-movflags', '+faststart', mp4],
  { stdio: ['pipe', 'inherit', 'inherit'] });
}

async function frames(ff, t0, n, fps) {
  const start = Date.now();
  for (let i = 0; i < n; i++) {
    await page.evaluate((x) => window.renderAt(x), t0 + i / fps);
    const buf = await page.screenshot({ type: 'jpeg', quality: 93 });
    if (!ff.stdin.write(buf)) await new Promise((r) => ff.stdin.once('drain', r));
    if (i % 300 === 0) console.error(`frame ${i}/${n}  ${((Date.now() - start) / 1000).toFixed(0)}s`);
  }
  ff.stdin.end();
  await new Promise((r) => ff.on('close', r));
}

if (mode === 'scene') {
  for (const t of process.argv[5].split(',').map(Number)) {
    await page.evaluate((x) => window.renderAt(x), t);
    await page.screenshot({ path: path.join(OUT, `scene${arg}_${t}.png`) });
  }
} else if (mode === 'stills') {
  for (const t of arg.split(',').map(Number)) {
    await page.evaluate((x) => window.renderAt(x), t);
    await page.screenshot({ path: path.join(OUT, `still_${t}.png`) });
  }
} else if (mode === 'cover') {
  // intro frame with every block shown, no subtitle, laid out by stage.html's cover() -> out/<ep>/cover.png
  const intro = tl.segments.filter((s) => s.scene === 'intro').pop();
  await page.evaluate((x) => { window.__cover = true; window.renderAt(x); }, intro.t1 - 0.5);
  await page.screenshot({ path: path.join(OUT, 'cover.png') });
} else if (mode === 'clip') {
  const [from, to] = arg.split(',').map(Number);
  const fps = Number(process.argv[5] || 30);
  const mp4 = path.join(OUT, `clip_${from}-${to}.mp4`);
  await frames(encoder(mp4, { fps }, 21, from, to), from, Math.ceil((to - from) * fps), fps);
  console.error('done', mp4);
} else {
  // 不加 -shortest：论文系列 BeyondMimic 用它时截掉了最后 4 帧；画面帧数按 timeline.total 取整，旁白末尾本就留了 0.8 s 静音
  const fps = Number(arg || 30);
  const mp4 = path.join(OUT, name + '_video.mp4');
  await frames(encoder(mp4, { fps }, 19), 0, Math.ceil(tl.total * fps), fps);
  console.error('done', mp4);
}
await browser.close();
