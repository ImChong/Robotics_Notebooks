// node check_layout.mjs <ep>        （先跑过 build.py，并用 render.mjs 生成过 stage.<ep>.built.html）
// node check_layout.mjs <ep> cover  封面帧（stage.html 的 buildCover，论文系列同款版式）：块互不重叠、都在画面里、大标题 / 中文名 /
//                                   一句话 / 每条目录各占一行、集数只在右上角季胶囊里（「第一季 · 001」）、别处没有集数和路线层级、
//                                   3:4 置顶裁剪后目录字底不过 1405（不需要 build.py）
// 每 0.5 s 抽一帧，用 DOM 量各块位置：可见块之间不许重叠、不许压字幕、片头 / 片尾相邻块间距 ≥ 10px，要读的东西必须在安全区 330–1440 之内
// （视频号 9:16 画面会被裁成居中的 6:7，底部还压着作者、标题、转赞评与浮评）。改自论文系列的 check_layout.mjs。
import { chromium } from 'playwright-core';
import fs from 'node:fs';
import { serve } from './serve.mjs';

const [name, mode] = process.argv.slice(2);
const tlPath = `out/${name}/timeline.json`;
const tl = fs.existsSync(tlPath) ? JSON.parse(fs.readFileSync(tlPath, 'utf8'))
  : { total: 30, segments: [{ scene: 'intro', t0: 0, t1: 30, from: 0, to: 30, lead: 0 }], subs: [] };
const b = await chromium.launch({ executablePath: process.env.CHROME });
const pg = await b.newPage({ viewport: { width: 1080, height: 1920 } });
const server = await serve(process.cwd());
await pg.goto(server.url + `stage.${name}.built.html`);
await pg.evaluate(() => window.robotsReady);
await pg.evaluate((t) => window.setup(t), tl);
await pg.evaluate(() => document.fonts.ready);

if (mode === 'cover') {
  const intro = tl.segments.filter((s) => s.scene === 'intro').pop();
  const r = await pg.evaluate((x) => {
    window.__cover = true;
    window.renderAt(x);
    const out = [];
    const sc = document.getElementById('scene');
    // 居中的整宽块：左右取文字本身，上下取盒子
    const text = (e) => {
      const box = e.getBoundingClientRect(), g = document.createRange(), w = document.createTreeWalker(e, NodeFilter.SHOW_TEXT);
      let l = Infinity, r = -Infinity;
      for (let n = w.nextNode(); n; n = w.nextNode()) {
        if (!n.nodeValue.trim()) continue;
        g.selectNodeContents(n);
        for (const q of g.getClientRects()) { l = Math.min(l, q.left); r = Math.max(r, q.right); }
      }
      return { left: l, right: r, top: box.top, bottom: box.bottom };
    };
    const hit = (a, c) => a.bottom > c.top + 2 && c.bottom > a.top + 2 && a.right > c.left + 2 && c.right > a.left + 2;
    const bigs = [...sc.querySelectorAll('.big')], meta = sc.querySelector('.meta'), toc = sc.querySelector('.toc');
    if (bigs.length < 2 || !meta || !toc) return ['cover 不是「大标题 / 中文名 / 信息框 / 目录」四块'];
    const blocks = [['title', text(bigs[0])], ['name', text(bigs[1])], ['meta', meta.getBoundingClientRect()], ['toc', toc.getBoundingClientRect()],
      ['brand', document.querySelector('.brand').getBoundingClientRect()]];
    for (let i = 0; i < blocks.length; i++) {
      const [n, ri] = blocks[i];
      if (ri.left < 30 || ri.right > 1050 || ri.top < 40 || ri.bottom > 1840) out.push(`${n} off the frame (${Math.round(ri.left)}–${Math.round(ri.right)}, ${Math.round(ri.top)}–${Math.round(ri.bottom)})`);
      for (let j = i + 1; j < blocks.length; j++) if (hit(ri, blocks[j][1])) out.push(`${n} x ${blocks[j][0]}`);
    }
    const oneLine = (n, e) => {
      const lh = parseFloat(getComputedStyle(e).lineHeight) || parseFloat(getComputedStyle(e).fontSize) * 1.5;
      if (e.getBoundingClientRect().height > lh * 1.3) out.push(`${n} wraps`);
    };
    oneLine('title', bigs[0].firstElementChild);
    [...bigs[1].children].forEach((e, k) => oneLine(k ? 'pitch' : 'name', e));
    [...toc.children].forEach((e, k) => oneLine(`toc row ${k + 1}`, e));
    // 集数只写在右上角的季胶囊里（「第一季 · 001」）；别处不出现集数和路线层级
    const E = window.EpisodeVideo, badge = document.getElementById('badge').textContent;
    if (badge !== E.badge + ' · ' + E.ep) out.push(`season pill reads "${badge}", want "${E.badge} · ${E.ep}"`);
    const all = sc.textContent + document.querySelector('.brand').textContent.replace(badge, '');
    if (/\b0\d\d\b|L[−-]?\d/.test(all)) out.push('cover shows an episode number or roadmap level outside the season pill');
    // 视频号「设置封面」按 3:4 置顶裁剪到 y 1440：最后一行目录的字底要离裁剪线 ≥ 35px
    const g = document.createRange(); g.selectNodeContents(toc.lastElementChild);
    const ink = Math.max(...[...g.getClientRects()].map((r) => r.bottom));
    if (ink > 1405) out.push(`toc ink bottom ${Math.round(ink)} > 1405 (3:4 top crop at 1440 leaves < 35px)`);
    return out;
  }, intro.t1 - 0.5);
  console.log(name, 'cover', r.length ? '\n  ' + r.join('\n  ') : 'no overlap');
  await b.close();
  server.close();
  process.exit(0);
}
for (const s of tl.segments) await pg.evaluate((t) => window.renderAt(t), s.t0 + 0.5);
const issues = new Map();
const BLOCKS = '.head, .prog, .stage, .cue, .big, .meta, .toc, .hero, .pt, .footer';
const KEYS = '.head, .stage, .cue, .big, .meta, .toc, .hero, .pt, .footer';
for (let t = 0.2; t < tl.total; t += 0.5) {
  const r = await pg.evaluate(([t, BLOCKS, KEYS]) => {
    window.renderAt(t);
    const R = (e) => e.getBoundingClientRect();
    const sc = document.getElementById('scene');
    const sub = document.getElementById('sub');
    const out = [];
    const subR = sub.textContent.trim() ? R(sub) : null;
    const hit = (a, c) => a.bottom > c.top + 2 && c.bottom > a.top + 2 && a.right > c.left + 2 && c.right > a.left + 2;
    const vis = (e) => +getComputedStyle(e).opacity >= 0.05 && getComputedStyle(e).display !== 'none';
    const blocks = [...sc.querySelectorAll(BLOCKS)].filter(vis);
    for (let i = 0; i < blocks.length; i++) {
      const ri = R(blocks[i]);
      if (ri.height === 0) continue;
      for (let j = i + 1; j < blocks.length; j++) {
        const rj = R(blocks[j]);
        if (rj.height && hit(ri, rj)) out.push(blocks[i].className + ' x ' + blocks[j].className);
      }
      if (subR && hit(ri, subR)) out.push(blocks[i].className + ' x subtitle');
      if (ri.bottom > 1920 || ri.right > 1082 || ri.left < -2) out.push(blocks[i].className + ' offscreen');
    }
    // 安全区：要读的东西只能在 330–1440 之间（淡入位移不算，用布局位置）
    const keys = [...sc.querySelectorAll(KEYS)].filter((e) => vis(e) && e.offsetHeight);
    for (const e of keys) {
      const top = e.offsetTop, bot = e.offsetTop + e.offsetHeight;
      if (top < 328 || bot > 1442) out.push(e.className + ' outside safe zone (' + top + '–' + bot + ')');
    }
    // 片头 / 片尾的块由 stack() 排：放不下时它会压缩间距，压到 10px 以下画面上就像贴在一起
    const stacked = keys.filter((e) => e.matches('.big, .meta, .toc, .hero, .pt, .footer'));
    for (const a of stacked) {
      for (const c of stacked) {
        if (a === c || a.offsetParent !== c.offsetParent) continue;
        const gap = c.offsetTop - (a.offsetTop + a.offsetHeight);
        const xo = Math.min(a.offsetLeft + a.offsetWidth, c.offsetLeft + c.offsetWidth) - Math.max(a.offsetLeft, c.offsetLeft);
        if (xo > 0 && gap >= -2 && gap < 10) out.push(a.className + ' / ' + c.className + ' too close (' + gap + 'px)');
      }
    }
    const tt = sc.querySelector('.title');
    if (tt) {
      const r2 = tt.getBoundingClientRect();
      if (r2.top < 328 || r2.bottom > 1442) out.push('title outside safe zone');
      if (r2.height > 100) out.push('title wraps');
    }
    for (const s of ['.big .t', '.big .zh', '.big .pitch']) {
      const e = sc.querySelector(s);
      if (e && vis(e) && e.scrollWidth > e.clientWidth + 2) out.push(s + ' wider than the frame');
    }
    if (subR && (subR.top < 328 || subR.bottom > 1442)) out.push('subtitle outside safe zone (' + Math.round(subR.top) + '–' + Math.round(subR.bottom) + ')');
    return out;
  }, [t, BLOCKS, KEYS]);
  for (const x of r) if (!issues.has(x)) issues.set(x, Math.round(t * 10) / 10);
}
console.log(name, issues.size ? [...issues].map(([k, v]) => `${k} @${v}s`).join('\n  ') : 'no overlap');
await b.close();
server.close();
