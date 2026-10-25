// node check_layout.mjs <ep>        （先跑过 build.py，并用 render.mjs 生成过 stage.<ep>.built.html）
// node check_layout.mjs <ep> cover  封面帧（stage.html 的 buildCover，论文系列同款版式）：块互不重叠、都在画面里、大标题 / 中文名 /
//                                   一句话 / 每条目录各占一行、集数只在顶栏正中（「No.001」，不碰左右两块）、别处没有集数和路线层级、
//                                   3:4 置顶裁剪后目录字底不过 1405（不需要 build.py）
// 每 0.5 s 抽一帧，用 DOM 量各块位置：可见块之间不许重叠、不许压字幕、片头 / 片尾相邻块间距 ≥ 10px，要读的东西必须在安全区 330–1440 之内
// （视频号 9:16 画面会被裁成居中的 6:7，底部还压着作者、标题、转赞评与浮评）。改自论文系列的 check_layout.mjs。
// 分镜 / 片头主图里面也查：看得见的字、公式、图形、机器人不许超出画面被裁掉，字和字（含公式）不许互相压住。
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
    // 集数只写在顶栏正中（「No.001」），和左边品牌、右边季胶囊各留 ≥ 10px；别处不出现集数和路线层级
    const E = window.EpisodeVideo, epno = document.getElementById('epno'), badge = document.getElementById('badge');
    if (epno.textContent !== 'No.' + E.ep) out.push(`episode number reads "${epno.textContent}", want "No.${E.ep}"`);
    if (badge.textContent !== E.badge) out.push(`season pill reads "${badge.textContent}", want "${E.badge}"`);
    const en = text(epno), bl = text(document.querySelector('.brand').firstElementChild), pr = badge.getBoundingClientRect();
    if (en.left < bl.right + 10 || en.right > pr.left - 10) out.push('episode number crowds the brand or the season pill');
    if (Math.abs((en.left + en.right) / 2 - 540) > 2) out.push('episode number is not centred');
    const all = sc.textContent + document.querySelector('.brand').textContent.replace(epno.textContent, '');
    if (/\b0\d\d\b|L[−-]?\d/.test(all)) out.push('cover shows an episode number or roadmap level outside the top-centre number');
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
    // 分镜 / 主图的 svg 里面：舞台框（圆角留 2px）与 svg 视口的交集就是看得见的范围
    const host = sc.querySelector('.stage > svg, .hero > svg');
    if (host) {
      const fr = host.parentElement.getBoundingClientRect(), sv = host.getBoundingClientRect();
      const view = { left: Math.max(fr.left + 2, sv.left), right: Math.min(fr.right - 2, sv.right), top: Math.max(fr.top + 2, sv.top), bottom: Math.min(fr.bottom - 2, sv.bottom) };
      const alpha = (n) => {
        let a = 1;
        for (let e = n; e && e !== host.parentElement; e = e.parentElement) {
          const cs = getComputedStyle(e);
          if (cs.display === 'none' || cs.visibility === 'hidden') return 0;
          a *= +cs.opacity;
        }
        return a;
      };
      const cut = (r) => Math.max(view.left - r.left, r.right - view.right, view.top - r.top, r.bottom - view.bottom);
      const name = (n) => (n.textContent || '').trim().slice(0, 14) || n.tagName;
      // 字的外框按行高算，比字形高出一截：上下各收 15% 再比（只看墨迹会不会被裁、会不会碰到）
      const ink = (r) => ({ left: r.left, right: r.right, top: r.top + 0.15 * r.height, bottom: r.bottom - 0.15 * r.height });
      // 图形与 svg 字：机器人是 robot3d.js 跟着模型投影外框走的透明 rect（后面紧跟画布的 foreignObject）
      for (const n of host.querySelectorAll('rect, circle, ellipse, line, path, polyline, polygon, text')) {
        if (n.closest('defs, mask, clipPath, marker')) continue;
        const r = n.getBoundingClientRect();
        if (!(r.width > 0 || r.height > 0) || alpha(n) < 0.1) continue;
        const robot = n.tagName === 'rect' && n.nextElementSibling && n.nextElementSibling.tagName === 'foreignObject' && n.getAttribute('fill') === 'none';
        const d = cut(n.tagName === 'text' ? ink(r) : r);
        if (d > 2) out.push(`${robot ? 'robot' : n.tagName + ' "' + name(n) + '"'} cut off by ${Math.round(d)}px`);
      }
      // 字块：svg 的 text 各算一块；KaTeX 公式按文字逐段量（不算 MathML 副本），同一条公式里的段不互查
      const words = [];
      for (const n of host.querySelectorAll('text')) {
        const r = n.getBoundingClientRect(), a = alpha(n);
        if (r.width > 0 && a >= 0.5 && n.textContent.trim()) words.push({ key: n, r: ink(r), s: name(n), g: n.textContent.trim() });
      }
      const g = document.createRange();
      for (const fo of host.querySelectorAll('foreignObject')) {
        const kh = fo.querySelector('.katex-html');
        if (!kh || alpha(fo) < 0.1) continue;
        // 公式整体（括号是 svg / 带边框的 span，取 .katex-html 里所有元素外框的并集）不许被裁掉
        let u = null;
        for (const e of [kh, ...kh.querySelectorAll('*')]) {
          const r = e.getBoundingClientRect();
          if (!(r.width > 0 && r.height > 0)) continue;
          u = u ? { left: Math.min(u.left, r.left), right: Math.max(u.right, r.right), top: Math.min(u.top, r.top), bottom: Math.max(u.bottom, r.bottom) } : { left: r.left, right: r.right, top: r.top, bottom: r.bottom };
        }
        const tex = (fo.texSource || fo.textSource || kh.textContent).slice(0, 24);
        if (u && cut(u) > 2) out.push(`formula "${tex}" cut off by ${Math.round(cut(u))}px`);
        if (alpha(fo) < 0.5) continue;
        const walk = document.createTreeWalker(kh, NodeFilter.SHOW_TEXT);
        for (let t = walk.nextNode(); t; t = walk.nextNode()) {
          if (!t.nodeValue.trim()) continue;
          g.selectNodeContents(t);
          for (const r of g.getClientRects()) if (r.width > 0) words.push({ key: fo, r: ink(r), s: tex, g: t.nodeValue.trim() });
        }
      }
      const seen = new Set();
      for (let i = 0; i < words.length; i++) {
        for (let j = i + 1; j < words.length; j++) {
          const a = words[i], c = words[j];
          if (a.key === c.key) continue;
          // 同一个字落在同一处（飞过去的数字交接给公式里的那个数）：叠在一起也看不出两层
          if (a.g === c.g && Math.abs(a.r.left + a.r.right - c.r.left - c.r.right) < 8 && Math.abs(a.r.top + a.r.bottom - c.r.top - c.r.bottom) < 8) continue;
          const w = Math.min(a.r.right, c.r.right) - Math.max(a.r.left, c.r.left), h = Math.min(a.r.bottom, c.r.bottom) - Math.max(a.r.top, c.r.top);
          const msg = `"${a.s}" x "${c.s}"`;
          if (w > 3 && h > 3 && !seen.has(msg)) { seen.add(msg); out.push('text overlap ' + msg); }
        }
      }
    }
    return out;
  }, [t, BLOCKS, KEYS]);
  for (const x of r) if (!issues.has(x)) issues.set(x, Math.round(t * 10) / 10);
}
console.log(name, issues.size ? [...issues].map(([k, v]) => `${k} @${v}s`).join('\n  ') : 'no overlap');
await b.close();
server.close();
