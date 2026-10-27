// 零基础篇 006：欧拉角、四元数、旋转矩阵——三种表示怎么选（路线 §L0 自测题 3「矩阵指数和欧拉角 / 四元数描述旋转，分别的优劣是什么」）
// 本目录按文件名顺序加载：00 共用部件 → 10 片头主图 → 21…26 六幕分镜 → 90 组装 window.EpisodeVideo。旁白在 ../ep006.py。
// 工具函数取自 ../ep005/00-common.js（命名空间换成 EP006；可动箭头、坐标网格、KaTeX、三维小坐标架、G1、骨盆上的三根小箭头）；
// 手画矩阵合成一个 E.grid（任意行列，四元数的 4×1 也用它）；G1 改成能按任意旋转矩阵摆（setR）、能抬高贴地（趴下用）；
// 另加三维里的一圈 E.ring3（前后半圈分两层画）和三个环套在一起的万向节 E.gimbal（第 3 幕）。
// 分镜都画在 800×420 的网格上。draw(st, t)：st 是分镜时间（旁白念完前画面停在 to 之前，一次性事件用它），
// t 是这一幕的真实时间（呼吸、来回摆这类循环动作用它，停帧时也不会卡住）。
(function () {
  var K = window.PaperDemoKit;
  var C = K.xColors;
  var E = window.EP006 = { K: K, scenes: [] };

  /* 颜色同 005（RELEASE_PLAN「画面配色」）：坐标轴 x 红、y 绿、z 蓝（E.X / E.Y / E.Z）——身子的三根小箭头、三根转轴、矩阵对应的列都用它，
     绕哪根轴转（横滚绕 x、俯仰绕 y、偏航绕 z）的箭头和圈就用那根轴的颜色；坑、出毛病、不是旋转用橙 E.ERR（红色留给 x 轴），打勾这类「对」用 good（绿）；
     三种写法：欧拉角 warn、四元数 accent、旋转矩阵 violet（同 003–005 的矩阵色）；本系列亮黄 --hi 只做强调。
     KaTeX 的 \textcolor 不认 CSS 变量，公式里用 E.HEX 的同一组色值（与 stage.html 的 :root 一致） */
  E.ACC = C.accent; E.GOOD = C.good; E.WARN = C.warn; E.BAD = C.bad; E.VIOLET = 'var(--violet)'; E.HI = 'var(--hi)';
  E.MUT = C.muted; E.INK = C.ink; E.INK2 = C.ink2; E.S1 = C.surface; E.S2 = C.surface2; E.BORDER = C.border; E.GRID = C.grid;
  E.HEX = { acc: '#569cd6', good: '#4ec9a4', warn: '#d7a75f', bad: '#e06c75', violet: '#c586c0', hi: '#ffd166', ink: '#e6e6e6', ink2: '#a8a8a8', mut: '#8a8a8a',
    x: '#e06c75', y: '#4ec9a4', z: '#569cd6', err: '#ff7b3a' };
  E.X = E.HEX.x; E.Y = E.HEX.y; E.Z = E.HEX.z; E.ERR = E.HEX.err;

  /* ── 基础工具 ── */
  function el(tag, attrs, parent) { var n = K.svgEl(tag, attrs || {}); if (parent) parent.appendChild(n); return n; }
  // 文字：size 不小于 18；anchor 默认居中；weight 如 700 / 900
  function txt(parent, x, y, s, size, color, anchor, weight) {
    var n = K.svgText(x, y, s, '', size || 20, anchor || 'middle');
    n.style.fill = color || E.INK;
    if (weight) n.style.fontWeight = weight;
    parent.appendChild(n);
    return n;
  }
  function op(node, v) { node.style.opacity = v; }
  function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }
  // 从 a 起 d 秒内 0 → 1（缓动）
  function fade(t, a, d) { return K.ease(K.seg(t, a, a + (d || 0.5))); }
  // a 起淡入、b 起淡出
  function window_(t, a, b, d) { return Math.min(fade(t, a, d), 1 - fade(t, b, d)); }
  // 弹出：scale 从 s0 到 1，以 (cx, cy) 为中心。g 不要再有别的 transform（需要就外面再包一层 g）
  function pop(g, cx, cy, p, s0) {
    var s = (s0 === undefined ? 0.8 : s0) + (1 - (s0 === undefined ? 0.8 : s0)) * p;
    g.setAttribute('transform', 'translate(' + cx + ' ' + cy + ') scale(' + s.toFixed(4) + ') translate(' + (-cx) + ' ' + (-cy) + ')');
    op(g, p);
  }
  function scaleAt(g, cx, cy, s) {
    g.setAttribute('transform', 'translate(' + cx + ' ' + cy + ') scale(' + s.toFixed(4) + ') translate(' + (-cx) + ' ' + (-cy) + ')');
  }
  // 线条「画出来」：p 0 → 1。实线用 dashoffset；虚线（有 stroke-dasharray）套一个同形状的遮罩来画，保留虚线节奏。
  // 要挂到页面上才量得到长度，所以第一次真正生效是在 draw 里（build 时调用只按 p 显隐）。
  var maskN = 0;
  function drawOn(path, p) {
    if (!path.isConnected) { op(path, p > 0 ? 1 : 0); return; }
    if (!path.__len) {
      path.__len = path.getTotalLength();
      if (path.getAttribute('stroke-dasharray')) {
        var svg = path.ownerSVGElement, defs = svg.querySelector('defs') || el('defs', {}, svg);
        var id = 'ep006-draw-mask-' + (++maskN);
        var mask = el('mask', { id: id, maskUnits: 'userSpaceOnUse', x: -2000, y: -2000, width: 6000, height: 6000 }, defs);
        var w = parseFloat(path.getAttribute('stroke-width') || '3') + 14;
        path.__mask = el('path', { d: path.getAttribute('d'), fill: 'none', stroke: '#fff', 'stroke-width': w, 'stroke-linecap': 'round',
          'stroke-dasharray': path.__len + ' ' + path.__len }, mask);
        path.setAttribute('mask', 'url(#' + id + ')');
      }
    }
    var L = path.__len, off = (L * (1 - clamp(p, 0, 1))).toFixed(2);
    if (path.__mask) path.__mask.style.strokeDashoffset = off;
    else { path.setAttribute('stroke-dasharray', L + ' ' + L); path.style.strokeDashoffset = off; }
    op(path, p > 0 ? 1 : 0);
  }
  // 线段 / 折线 + 自己画的三角箭头（粗细不影响箭头大小）。pts: [[x,y],...]；返回 { g, line, head }
  function arrow(parent, pts, color, w, opts) {
    var o = opts || {};
    var g = el('g', {}, parent);
    var line = el('path', { d: K.polyPath(pts), fill: 'none', 'stroke-width': w || 3, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, g);
    K.paint(line, null, color || E.INK2);
    if (o.dash) line.setAttribute('stroke-dasharray', o.dash);
    var a = pts[pts.length - 2], b = pts[pts.length - 1], ang = Math.atan2(b[1] - a[1], b[0] - a[0]);
    var L = o.head || 12, W = (o.head || 12) * 0.6;
    var p1 = [b[0] - L * Math.cos(ang) + W * Math.sin(ang), b[1] - L * Math.sin(ang) - W * Math.cos(ang)];
    var p2 = [b[0] - L * Math.cos(ang) - W * Math.sin(ang), b[1] - L * Math.sin(ang) + W * Math.cos(ang)];
    var head = el('path', { d: 'M ' + b[0] + ' ' + b[1] + ' L ' + p1[0].toFixed(1) + ' ' + p1[1].toFixed(1) + ' L ' + p2[0].toFixed(1) + ' ' + p2[1].toFixed(1) + ' Z' }, g);
    K.paint(head, color || E.INK2);
    return { g: g, line: line, head: head };
  }
  // 曲线 path（d 字符串）+ 末端箭头；方向取 path 末端切线（需挂到页面后才准，先按 endDir 给的角度画）
  function curveArrow(parent, d, color, w, endDirDeg, opts) {
    var o = opts || {};
    var g = el('g', {}, parent);
    var line = el('path', { d: d, fill: 'none', 'stroke-width': w || 3, 'stroke-linecap': 'round' }, g);
    K.paint(line, null, color || E.INK2);
    if (o.dash) line.setAttribute('stroke-dasharray', o.dash);
    var m = d.match(/(-?[\d.]+)[ ,](-?[\d.]+)\s*$/), bx = +m[1], by = +m[2];
    var ang = endDirDeg * Math.PI / 180, L = o.head || 12, W = L * 0.6;
    var p1 = [bx - L * Math.cos(ang) + W * Math.sin(ang), by - L * Math.sin(ang) - W * Math.cos(ang)];
    var p2 = [bx - L * Math.cos(ang) - W * Math.sin(ang), by - L * Math.sin(ang) + W * Math.cos(ang)];
    var head = el('path', { d: 'M ' + bx + ' ' + by + ' L ' + p1[0].toFixed(1) + ' ' + p1[1].toFixed(1) + ' L ' + p2[0].toFixed(1) + ' ' + p2[1].toFixed(1) + ' Z' }, g);
    K.paint(head, color || E.INK2);
    return { g: g, line: line, head: head };
  }
  // 用 path 画的 ✓（不依赖字体），(x, y) 为左上角，size 为宽
  function check(parent, x, y, size, color, w) {
    var s = (size || 18) / 18;
    var p = el('path', { d: 'M ' + x + ' ' + (y + 8 * s) + ' L ' + (x + 6 * s) + ' ' + (y + 14 * s) + ' L ' + (x + 18 * s) + ' ' + y,
      fill: 'none', 'stroke-width': w || 3.5, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, parent);
    K.paint(p, null, color || E.GOOD);
    return p;
  }
  // 用 path 画的 ×，中心 (cx, cy)，半臂长 r
  function cross(parent, cx, cy, r, color, w) {
    var p = el('path', { d: 'M ' + (cx - r) + ' ' + (cy - r) + ' L ' + (cx + r) + ' ' + (cy + r) + ' M ' + (cx + r) + ' ' + (cy - r) + ' L ' + (cx - r) + ' ' + (cy + r),
      fill: 'none', 'stroke-width': w || 3, 'stroke-linecap': 'round' }, parent);
    K.paint(p, null, color || E.ERR);
    return p;
  }
  // 圆角胶囊标签：rect + 文字；返回 { g, rect, text }
  function tag(parent, x, y, w, h, s, color, opts) {
    var o = opts || {};
    var g = el('g', {}, parent);
    var r = el('rect', { x: x, y: y, width: w, height: h, rx: o.rx === undefined ? h / 2 : o.rx, 'stroke-width': o.sw || 2 }, g);
    K.paint(r, o.fill || 'color-mix(in srgb, ' + color + ' 12%, var(--demo-surface))', color);
    var t = txt(g, x + w / 2, y + h / 2 + (o.size || 20) * 0.36, s, o.size || 20, o.ink || E.INK, 'middle', o.weight);
    return { g: g, rect: r, text: t };
  }
  // 填色混一点本色：color-mix(color pct%, base)
  function tint(color, pct, base) { return 'color-mix(in srgb, ' + color + ' ' + pct + '%, ' + (base || 'var(--demo-surface-2)') + ')'; }
  function glow(node, color, on) { node.style.filter = on ? 'drop-shadow(0 0 6px ' + color + ')' : ''; }
  // 压在网格线上的字描一圈底色（同 003 的坐标标签），线从字后面过去也看得清
  function halo(node, w) { node.style.paintOrder = 'stroke'; node.style.stroke = 'var(--demo-surface)'; node.style.strokeWidth = (w || 6) + 'px'; node.style.strokeLinejoin = 'round'; return node; }

  // 圆角矩形的 path（drawOn 画虚线框要 path 的 d，rect 没有）
  function rrect(x, y, w, h, r) {
    return 'M ' + (x + r) + ' ' + y + ' H ' + (x + w - r) + ' A ' + r + ' ' + r + ' 0 0 1 ' + (x + w) + ' ' + (y + r) + ' V ' + (y + h - r) +
      ' A ' + r + ' ' + r + ' 0 0 1 ' + (x + w - r) + ' ' + (y + h) + ' H ' + (x + r) + ' A ' + r + ' ' + r + ' 0 0 1 ' + x + ' ' + (y + h - r) +
      ' V ' + (y + r) + ' A ' + r + ' ' + r + ' 0 0 1 ' + (x + r) + ' ' + y + ' Z';
  }
  E.rrect = rrect;

  E.el = el; E.txt = txt; E.op = op; E.clamp = clamp; E.fade = fade; E.win = window_; E.pop = pop; E.scaleAt = scaleAt;
  E.drawOn = drawOn; E.arrow = arrow; E.curveArrow = curveArrow; E.check = check; E.cross = cross; E.tag = tag; E.tint = tint; E.glow = glow; E.halo = halo;

  /* ── 可动箭头：set(x0, y0, x1, y1) 每帧改起止点，箭头三角跟着转；太短时三角按长度缩小，长度 0 时整支不画 ── */
  E.vec = function (parent, color, w, opts) {
    var o = opts || {};
    var g = el('g', {}, parent);
    var line = el('path', { fill: 'none', 'stroke-width': w || 4, 'stroke-linecap': 'round' }, g);
    var head = el('path', {}, g);
    var api = {
      g: g, line: line, head: head, color: color,
      paint: function (c) { api.color = c; K.paint(line, null, c); K.paint(head, c); return api; },
      set: function (x0, y0, x1, y1) {
        var dx = x1 - x0, dy = y1 - y0, L = Math.hypot(dx, dy);
        if (L < 0.5) { line.setAttribute('d', ''); head.setAttribute('d', ''); return api; }
        var hl = Math.min(o.head || 16, L * 0.45), hw = hl * 0.6, ux = dx / L, uy = dy / L;
        var bx = x1 - ux * hl * 0.8, by = y1 - uy * hl * 0.8;   // 线收在三角里面，线头不从尖上冒出来
        line.setAttribute('d', 'M ' + x0.toFixed(2) + ' ' + y0.toFixed(2) + ' L ' + bx.toFixed(2) + ' ' + by.toFixed(2));
        var p1 = [x1 - ux * hl - uy * hw, y1 - uy * hl + ux * hw], p2 = [x1 - ux * hl + uy * hw, y1 - uy * hl - ux * hw];
        head.setAttribute('d', 'M ' + x1.toFixed(2) + ' ' + y1.toFixed(2) + ' L ' + p1[0].toFixed(2) + ' ' + p1[1].toFixed(2) + ' L ' + p2[0].toFixed(2) + ' ' + p2[1].toFixed(2) + ' Z');
        return api;
      }
    };
    if (o.dash) line.setAttribute('stroke-dasharray', o.dash);
    api.paint(color || E.INK);
    return api;
  };

  /* ── 线段裁到矩形里（Liang–Barsky）：返回 [[x,y],[x,y]] 或 null ── */
  function clipSeg(p, q, b) {
    var t0 = 0, t1 = 1, dx = q[0] - p[0], dy = q[1] - p[1];
    var P = [-dx, dx, -dy, dy], Q = [p[0] - b[0], b[2] - p[0], p[1] - b[1], b[3] - p[1]];
    for (var i = 0; i < 4; i++) {
      if (Math.abs(P[i]) < 1e-9) { if (Q[i] < 0) return null; continue; }
      var r = Q[i] / P[i];
      if (P[i] < 0) { if (r > t1) return null; if (r > t0) t0 = r; } else { if (r < t0) return null; if (r < t1) t1 = r; }
    }
    return [[p[0] + t0 * dx, p[1] + t0 * dy], [p[0] + t1 * dx, p[1] + t1 * dy]];
  }
  E.clipSeg = clipSeg;

  /* ── 坐标网格（数学坐标：x 往右、y 往上）：
     o = { ox, oy：原点在画面里的位置；u：一格多少 px；x0, x1, y0, y1：画出的整数范围（含两端）；axes（默认 true）：画 x / y 轴与箭头；
           labels（默认 true）：轴端写 x、y（y 写在箭头尖右边）；ticks：轴上标 1、2、3…（默认 false）；color：格线颜色 }
     返回 { g, P(x, y) → [X, Y]：数学坐标 → 画面坐标, box：网格外框（画面坐标 [x0, y0, x1, y1]）,
            warp：一层可变形的格线（默认不显示），warp.set(m) 按矩阵 m = [a, b, c, d]（即 [[a, b], [c, d]]）变形，变完仍裁在外框里；
            warp.g / warp.lines 可改颜色、透明度 } ── */
  E.plane = function (parent, o) {
    var u = o.u, ox = o.ox, oy = o.oy;
    function P(x, y) { return [ox + u * x, oy - u * y]; }
    var g = el('g', { 'data-grid': '1' }, parent);   // data-grid：画在格子上的字本来就描了底色，查「字压线」时不算格线
    var box = [ox + u * o.x0, oy - u * o.y1, ox + u * o.x1, oy - u * o.y0];
    var grid = el('g', {}, g), x, y;
    for (x = o.x0; x <= o.x1; x++) K.paint(el('line', { x1: P(x, 0)[0], y1: box[1], x2: P(x, 0)[0], y2: box[3], 'stroke-width': 1.5 }, grid), null, o.color || E.GRID);
    for (y = o.y0; y <= o.y1; y++) K.paint(el('line', { x1: box[0], y1: P(0, y)[1], x2: box[2], y2: P(0, y)[1], 'stroke-width': 1.5 }, grid), null, o.color || E.GRID);
    // 可变形的一层：格线取得比外框大一圈（转、拉、推之后外框里也铺满），每帧变完再裁回外框
    var N = Math.max(Math.abs(o.x0), Math.abs(o.x1), Math.abs(o.y0), Math.abs(o.y1)) * 3 + 2;
    var warpG = el('g', {}, g), lines = [];
    for (var k = -N; k <= N; k++) {
      lines.push({ a: [k, -N], b: [k, N], el: el('path', { fill: 'none', 'stroke-width': 2 }, warpG) });
      lines.push({ a: [-N, k], b: [N, k], el: el('path', { fill: 'none', 'stroke-width': 2 }, warpG) });
    }
    lines.forEach(function (l) { K.paint(l.el, null, E.VIOLET); });
    op(warpG, 0);
    var warp = {
      g: warpG, lines: lines,
      set: function (m) {
        lines.forEach(function (l) {
          var p = P(m[0] * l.a[0] + m[1] * l.a[1], m[2] * l.a[0] + m[3] * l.a[1]);
          var q = P(m[0] * l.b[0] + m[1] * l.b[1], m[2] * l.b[0] + m[3] * l.b[1]);
          var s = clipSeg(p, q, box);
          l.el.setAttribute('d', s ? 'M ' + s[0][0].toFixed(1) + ' ' + s[0][1].toFixed(1) + ' L ' + s[1][0].toFixed(1) + ' ' + s[1][1].toFixed(1) : '');
        });
      }
    };
    warp.set([1, 0, 0, 1]);
    var axes = el('g', {}, g);
    if (o.axes !== false) {
      var ax = E.vec(axes, E.INK2, 2.5, { head: 12 }), ay = E.vec(axes, E.INK2, 2.5, { head: 12 });
      ax.set(box[0], oy, box[2] + 14, oy); ay.set(ox, box[3], ox, box[1] - 14);
      if (o.labels !== false) {
        txt(axes, box[2] + 22, oy + 8, 'x', 24, E.INK2, 'start', 700);
        txt(axes, ox + 12, box[1] - 8, 'y', 24, E.INK2, 'start', 700);   // 写在箭头尖右边，不往上多占一行
      }
      if (o.ticks) {
        for (x = o.x0; x <= o.x1; x++) if (x) txt(axes, P(x, 0)[0], oy + 24, String(x).replace('-', '−'), 18, E.MUT);
        for (y = o.y0; y <= o.y1; y++) if (y) txt(axes, ox - 12, P(0, y)[1] + 6, String(y).replace('-', '−'), 18, E.MUT, 'end');
      }
    }
    return { g: g, grid: grid, axes: axes, warp: warp, P: P, box: box, u: u };
  };

  /* ── 公式：K.svgMath 包一层，(x, y) 同 txt 的基线；size 默认 28；color 是 CSS 颜色（整条公式的默认色） ── */
  E.tex = function (parent, x, y, tex, size, color, anchor, opts) {
    var o = opts || {};
    var fo = K.svgMath(x, y, tex, { size: size || 28, anchor: anchor || 'middle', w: o.w, h: o.h });
    if (color) fo.setTone(color);
    parent.appendChild(fo);
    return fo;
  };

  /* ── 手画的矩阵 / 列向量（不用 KaTeX，取自 004 / 005 的 E.mat、E.mat3，合成一个）：每个数的位置都知道，好逐个高亮、逐个写出来。
     E.grid(parent, cx, cy, M, o)：M 按行存（[[..], [..]]，数或字符串）；o = { size（默认 28）, dx / dy：列距 / 排距, colors：每列一个颜色（或 cellColors 每个数一个）,
     digits, pad：括号离最外一列中心多远（默认 0.95 字号）, cw：框的宽（默认 dx − 6） }
     返回 { g, cells[k]（k = 行 × 列数 + 列）, pos(k), set(M), cellBox(k, color, p), colBox(j, color, p), brackets, box } ── */
  E.grid = function (parent, cx, cy, M, o) {
    o = o || {};
    var R = M.length, Cn = M[0].length, size = o.size || 28;
    var dx = o.dx || size * 2.4, dy = o.dy || size * 1.3;
    var g = el('g', {}, parent), under = el('g', {}, g);
    var bh = (R - 1) * dy + size * 1.15, bx = (Cn - 1) * dx / 2 + (o.pad || size * 0.95), CW = o.cw || dx - 6;
    var brackets = [-1, 1].map(function (sd) {
      var xb = cx + sd * bx, lip = size * 0.28;
      return K.paint(el('path', { d: 'M ' + (xb - sd * lip) + ' ' + (cy - bh / 2) + ' H ' + xb + ' V ' + (cy + bh / 2) + ' H ' + (xb - sd * lip),
        fill: 'none', 'stroke-width': Math.max(2, size / 14), 'stroke-linecap': 'square' }, g), null, o.bracket || E.INK);
    });
    function pos(k) { var r = Math.floor(k / Cn), c = k % Cn; return [cx + (c - (Cn - 1) / 2) * dx, cy + (r - (R - 1) / 2) * dy]; }
    var colR = [], cellR = [], k;
    for (k = 0; k < Cn; k++) { colR.push(el('rect', { rx: 10, 'stroke-width': 2.5 }, under)); op(colR[k], 0); }
    for (k = 0; k < R * Cn; k++) { cellR.push(el('rect', { rx: 8, 'stroke-width': 2.5 }, under)); op(cellR[k], 0); }
    var cells = [];
    for (k = 0; k < R * Cn; k++) {
      var p = pos(k), c = (o.cellColors || [])[k] || (o.colors || [])[k % Cn] || E.INK;
      var n = txt(g, p[0], p[1] + size * 0.35, '', size, c, 'middle');
      n.style.fontFamily = 'KaTeX_Main, serif';
      cells.push(n);
    }
    var api = {
      g: g, cells: cells, pos: pos, brackets: brackets, size: size, dx: dx, dy: dy,
      box: [cx - bx, cy - bh / 2, cx + bx, cy + bh / 2],
      set: function (m) { for (var q = 0; q < R * Cn; q++) cells[q].textContent = E.num(m[Math.floor(q / Cn)][q % Cn], o.digits); return api; },
      colBox: function (j, color, p) {
        var b = colR[j], xc = pos(j)[0], w = Math.min(CW, dx - 4);
        b.setAttribute('x', (xc - w / 2).toFixed(1)); b.setAttribute('width', w.toFixed(1));
        b.setAttribute('y', (cy - bh / 2 - size * 0.05).toFixed(1)); b.setAttribute('height', (bh + size * 0.1).toFixed(1));
        K.paint(b, tint(color, 16, 'var(--demo-surface)'), color); op(b, clamp(p, 0, 1));
      },
      cellBox: function (q, color, p) {
        var b = cellR[q], c = pos(q), w = Math.min(CW, dx - 6), hh = Math.min(size * 1.1, dy - 4);
        b.setAttribute('x', (c[0] - w / 2).toFixed(1)); b.setAttribute('width', w.toFixed(1));
        b.setAttribute('y', (c[1] - hh / 2).toFixed(1)); b.setAttribute('height', hh.toFixed(1));
        K.paint(b, tint(color, 22, 'var(--demo-surface)'), color); op(b, clamp(p, 0, 1));
      }
    };
    api.set(M);
    return api;
  };

  /* ── 数：小数写成「0.8」，负号用真正的减号「−」 ── */
  E.num = function (x, digits) {
    var s = typeof x === 'string' ? x : (Math.abs(x) < 5e-3 ? '0' : x.toFixed(digits === undefined ? 2 : digits).replace(/\.?0+$/, ''));
    return s.replace('-', '−');
  };

  /* ── 三维：数学坐标 x 前、y 左、z 上，斜着看。E.view3(az, el) 给出投影：P(v) → [右, 上]（单位长度的画面分量），P.toward 是朝着看的人的方向
     （点乘 > 0 的在前面）；az / el 的定义同 ../../robot3d.js（az 0 = 从 −y 方向看，el > 0 = 从上往下看） ── */
  E.view3 = function (az, el) {
    var a = az * Math.PI / 180, e = el * Math.PI / 180;
    var r = [Math.cos(a), Math.sin(a), 0], u = [-Math.sin(a) * Math.sin(e), Math.cos(a) * Math.sin(e), Math.cos(e)];
    var P = function (v) { return [v[0] * r[0] + v[1] * r[1] + v[2] * r[2], v[0] * u[0] + v[1] * u[1] + v[2] * u[2]]; };
    P.toward = [Math.sin(a) * Math.cos(e), -Math.cos(a) * Math.cos(e), Math.sin(e)];
    return P;
  };
  // 3×3：按行存；绕 x / y / z 轴转 a 弧度、相乘、取第 j 列；E.zyx(偏航, 俯仰, 横滚) = Rz·Ry·Rx，就是 RPY：先绕不动的 x 横滚、再绕 y 俯仰、最后绕 z 偏航
  E.Rx = function (a) { var c = Math.cos(a), s = Math.sin(a); return [[1, 0, 0], [0, c, -s], [0, s, c]]; };
  E.Ry = function (a) { var c = Math.cos(a), s = Math.sin(a); return [[c, 0, s], [0, 1, 0], [-s, 0, c]]; };
  E.Rz = function (a) { var c = Math.cos(a), s = Math.sin(a); return [[c, -s, 0], [s, c, 0], [0, 0, 1]]; };
  E.mul3 = function (A, B) {
    return A.map(function (row) { return [0, 1, 2].map(function (j) { return row[0] * B[0][j] + row[1] * B[1][j] + row[2] * B[2][j]; }); });
  };
  E.col3 = function (M, j) { return [M[0][j], M[1][j], M[2][j]]; };
  E.zyx = function (yaw, pitch, roll) { return E.mul3(E.mul3(E.Rz(yaw), E.Ry(pitch)), E.Rx(roll)); };
  function dot3(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

  /* ── 三维里一圈（圆心 c、半径 r、所在平面由单位向量 a、b 张成）画成两段折线：前半圈（朝着看的人）和后半圈。
     E.ring3(back, front, color, w)：back / front 是两层 <g>（后半圈画在小箭头后面，前半圈画在前面）；set(c, a, b, r, proj, P) ── */
  E.ring3 = function (back, front, color, w) {
    var pb = K.paint(el('path', { fill: 'none', 'stroke-width': (w || 7) - 2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, back), null, color);
    var pf = K.paint(el('path', { fill: 'none', 'stroke-width': w || 7, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, front), null, color);
    pb.style.strokeOpacity = 0.4;
    return {
      back: pb, front: pf,
      set: function (c, a, b, r, proj, P) {
        var N = 96, df = '', db = '', prevF = null;
        for (var k = 0; k <= N; k++) {
          var t = 2 * Math.PI * k / N, ct = Math.cos(t), st = Math.sin(t);
          var v = [c[0] + r * (ct * a[0] + st * b[0]), c[1] + r * (ct * a[1] + st * b[1]), c[2] + r * (ct * a[2] + st * b[2])];
          var isF = dot3([ct * a[0] + st * b[0], ct * a[1] + st * b[1], ct * a[2] + st * b[2]], proj.toward) >= -0.02;
          var q = P(v), s = q[0].toFixed(1) + ' ' + q[1].toFixed(1);
          // 前后交界处两段各多画一个点，接得上
          if (isF) df += (prevF === true ? ' L ' : ' M ') + s; else db += (prevF === false ? ' L ' : ' M ') + s;
          if (prevF !== null && prevF !== isF) { if (isF) db += ' L ' + s; else df += ' L ' + s; }
          prevF = isF;
        }
        pf.setAttribute('d', df); pb.setAttribute('d', db);
      }
    };
  };

  /* ── 万向节：三个套在一起的环，外环绕竖轴转（偏航，蓝）、中环绕外环上左右方向的轴转（俯仰，绿）、内环绕中环上前后方向的轴转（横滚，红）；
     中间是身子的三根小箭头（x 前 红、y 左 绿、z 上 蓝）。底下一根立柱和底座托着外环。
     E.gimbal(parent, cx, cy, L, proj)：中心 (cx, cy)，单位长 L px。set(yaw, pitch, roll)（弧度）；
     返回 { P, rings[外, 中, 内], axes：三根转轴虚线 { z 外环、y 中环、x 内环 }（默认不显示）, body：身子的三根小箭头, R：身子的旋转矩阵 } ── */
  E.gimbal = function (parent, cx, cy, L, proj) {
    var g = el('g', {}, parent);
    function P(v) { var q = proj(v); return [cx + L * q[0], cy - L * q[1]]; }
    var RAD = [1, 0.84, 0.68], BASE = -1.32;
    var stand = el('g', {}, g);
    var foot = K.paint(el('path', { 'stroke-width': 2.5 }, stand), E.tint(E.MUT, 18, 'var(--demo-surface)'), E.MUT);
    var post = K.paint(el('path', { fill: 'none', 'stroke-width': 5, 'stroke-linecap': 'round' }, stand), null, E.MUT);
    var back = el('g', {}, g), axesG = el('g', {}, g), bodyG = el('g', {}, g), front = el('g', {}, g);
    var cols = [E.Z, E.Y, E.X];
    var rings = cols.map(function (c) { return E.ring3(back, front, c, 8); });
    // 环和环之间的小轴（连接点）：外环上下两点接立柱、中环左右两点接外环、内环前后两点接中环
    var axles = cols.map(function (c) { return K.paint(el('path', { fill: 'none', 'stroke-width': 5, 'stroke-linecap': 'round' }, front), null, c); });
    var pins = [];
    for (var k = 0; k < 6; k++) pins.push(K.paint(el('circle', { r: 5.5, 'stroke-width': 2 }, front), E.S1, cols[Math.floor(k / 2)]));
    var axes = {};
    [['z', E.Z], ['y', E.Y], ['x', E.X]].forEach(function (d) {
      var p = K.paint(el('path', { fill: 'none', 'stroke-width': 3.5, 'stroke-dasharray': '10 7', 'stroke-linecap': 'round' }, axesG), null, d[1]);
      op(p, 0);
      axes[d[0]] = p;
    });
    var body = [E.X, E.Y, E.Z].map(function (c) { var a = E.vec(bodyG, c, 6, { head: 17 }); return a; });
    var hub = K.paint(el('circle', { r: 7, 'stroke-width': 2.5 }, bodyG), E.INK, E.S1);
    function seg(path, a, b) { var p = P(a), q = P(b); path.setAttribute('d', 'M ' + p[0].toFixed(1) + ' ' + p[1].toFixed(1) + ' L ' + q[0].toFixed(1) + ' ' + q[1].toFixed(1)); }
    function sc(v, s) { return [v[0] * s, v[1] * s, v[2] * s]; }
    var api = {
      g: g, P: P, rings: rings, axes: axes, body: body, axles: axles, pins: pins, stand: stand, hub: hub, R: null, RAD: RAD,
      set: function (yaw, pitch, roll) {
        var F0 = E.Rz(yaw), F1 = E.mul3(F0, E.Ry(pitch)), F2 = E.mul3(F1, E.Rx(roll));
        var z = [0, 0, 1], y0 = E.col3(F0, 1), x1 = E.col3(F1, 0), y1 = E.col3(F1, 1), x2 = E.col3(F2, 0), z2 = E.col3(F2, 2);
        rings[0].set([0, 0, 0], y0, z, RAD[0], proj, P);     // 外环：竖着，平面里有竖轴和中环的轴
        rings[1].set([0, 0, 0], x1, y1, RAD[1], proj, P);    // 中环：平面里有自己的轴（左右）和内环的轴（前后）
        rings[2].set([0, 0, 0], x2, z2, RAD[2], proj, P);    // 内环：平面里有自己的轴（前后）和身子的上
        seg(post, [0, 0, BASE], sc(z, -RAD[0]));
        var fp = [];
        for (var k = 0; k <= 32; k++) { var t = 2 * Math.PI * k / 32, q = P([0.42 * Math.cos(t), 0.42 * Math.sin(t), BASE]); fp.push((k ? 'L ' : 'M ') + q[0].toFixed(1) + ' ' + q[1].toFixed(1)); }
        foot.setAttribute('d', fp.join(' ') + ' Z');
        // 小轴：外环顶上一小截竖轴；中环 ↔ 外环（沿 y0）；内环 ↔ 中环（沿 x1）
        axles[0].setAttribute('d', (function () { var a = P(sc(z, RAD[0])), b = P(sc(z, RAD[0] + 0.1)); return 'M ' + a.join(' ') + ' L ' + b.join(' '); })());
        axles[1].setAttribute('d', [1, -1].map(function (s) { var a = P(sc(y0, s * RAD[1])), b = P(sc(y0, s * RAD[0])); return 'M ' + a[0].toFixed(1) + ' ' + a[1].toFixed(1) + ' L ' + b[0].toFixed(1) + ' ' + b[1].toFixed(1); }).join(' '));
        axles[2].setAttribute('d', [1, -1].map(function (s) { var a = P(sc(x1, s * RAD[2])), b = P(sc(x1, s * RAD[1])); return 'M ' + a[0].toFixed(1) + ' ' + a[1].toFixed(1) + ' L ' + b[0].toFixed(1) + ' ' + b[1].toFixed(1); }).join(' '));
        [[z, RAD[0]], [y0, RAD[1]], [x1, RAD[2]]].forEach(function (d, i) {
          [1, -1].forEach(function (s, j) { var q = P(sc(d[0], s * d[1])); pins[2 * i + j].setAttribute('cx', q[0].toFixed(1)); pins[2 * i + j].setAttribute('cy', q[1].toFixed(1)); });
        });
        seg(axes.z, [0, 0, -1.25], [0, 0, 1.3]);
        seg(axes.y, sc(y0, -1.25), sc(y0, 1.25));
        seg(axes.x, sc(x2, -1.25), sc(x2, 1.25));
        var o0 = P([0, 0, 0]);
        [0, 1, 2].forEach(function (j) { var tp = P(sc(E.col3(F2, j), 0.52)); body[j].set(o0[0], o0[1], tp[0], tp[1]); });
        hub.setAttribute('cx', o0[0].toFixed(1)); hub.setAttribute('cy', o0[1].toFixed(1));
        api.R = F2;
        return api;
      }
    };
    api.set(0, 0, 0);
    return api;
  };

  /* ── 三维小坐标架（取自 005）：原点 (cx, cy)，单位长 L px；三根小箭头（x 红、y 绿、z 蓝）+ 三根小箭头围成的小方块（淡线）。
     set(M)：M 是 3×3（列 = 三根小箭头转完以后的方向）；proj：E.view3 的投影 ── */
  E.gizmo = function (parent, cx, cy, L, proj, opts) {
    var o = opts || {};
    var g = el('g', {}, parent), box = el('g', {}, g);
    var edges = [];
    for (var k = 0; k < 12; k++) edges.push(K.paint(el('line', { 'stroke-width': 1.6, 'stroke-dasharray': '5 5' }, box), null, E.MUT));
    var cols = [E.X, E.Y, E.Z];
    var arrows = cols.map(function (c) { return E.vec(g, c, o.w || 4.5, { head: o.head || 14 }); });
    var labs = ['x', 'y', 'z'].map(function (s, i) { return halo(txt(g, 0, 0, s, o.lab || 22, cols[i], 'middle', 700)); });
    function P(v) { var q = proj(v); return [cx + L * q[0], cy - L * q[1]]; }
    var api = {
      g: g, box: box, arrows: arrows, labels: labs, P: P,
      set: function (M) {
        var c = [E.col3(M, 0), E.col3(M, 1), E.col3(M, 2)];
        function at(a, b, d) { return P([a * c[0][0] + b * c[1][0] + d * c[2][0], a * c[0][1] + b * c[1][1] + d * c[2][1], a * c[0][2] + b * c[1][2] + d * c[2][2]]); }
        var n = 0;
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]].forEach(function (p) {
          [0, 1, 2].forEach(function (ax) {
            if (p[ax]) return;
            var q = p.slice(); q[ax] = 1;
            var a = at(p[0], p[1], p[2]), b = at(q[0], q[1], q[2]), ln = edges[n++];
            ln.setAttribute('x1', a[0].toFixed(1)); ln.setAttribute('y1', a[1].toFixed(1)); ln.setAttribute('x2', b[0].toFixed(1)); ln.setAttribute('y2', b[1].toFixed(1));
          });
        });
        var o0 = P([0, 0, 0]);
        c.forEach(function (v, i) {
          var tp = P(v); arrows[i].set(o0[0], o0[1], tp[0], tp[1]);
          var dx = tp[0] - o0[0], dy = tp[1] - o0[1], len = Math.hypot(dx, dy) || 1;
          labs[i].setAttribute('x', (tp[0] + dx / len * 20).toFixed(1)); labs[i].setAttribute('y', (tp[1] + dy / len * 20 + 7).toFixed(1));
        });
        return api;
      }
    };
    api.set([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
    return api;
  };

  /* ── G1（Robot Description Gallery Online 收录的 URDF，由 ../../robot3d.js 渲染），斜着看。
     E.g1(parent, x, y, h, o)：站着时两脚中间的地面点放在 (x, y)，身高 h px；o = { az, el（默认 135、20）, w：画布宽（默认 0.62 h）, up / down：画布往上 / 往下多留几 px }。
     身子绕骨盆转：set(yaw) 原地转身（从上往下看逆时针为正，往机器人的左边转）；setR(M) 按 3×3 旋转矩阵摆（M 的列 = 身子的前、左、上）；lift(dz)：骨盆抬高 dz 米。
     frame(L)：骨盆上三根小箭头（机器人的前、左、上，各长 L 米）在画面里的起点和三个箭头尖；ring(r)：脚下半径 r 米的地面圆（折线点）；
     lowest()：此刻模型最低点比站着时的脚底高多少米（趴下时用来贴地） ── */
  E.g1 = function (parent, x, y, h, o) {
    o = o || {};
    var R3 = window.Robot3D, B = R3.measure('g1'), sole = B.min.z, top = B.max.z;
    var az = o.az === undefined ? 135 : o.az, elv = o.el === undefined ? 20 : o.el, S = h / (top - sole);
    var g = el('g', {}, parent);
    var w = o.w || 0.62 * h;
    var v = R3.view(g, { robot: 'g1', origin: [x, y + sole * S * Math.cos(elv * Math.PI / 180)], scale: S, az: az, el: elv,
      box: [x - w / 2, y - h - 12 - (o.up || 0), w, h + 22 + (o.up || 0) + (o.down || 0)], res: o.res || 2 });
    v.joints({ left_shoulder_roll_joint: 8 * Math.PI / 180, right_shoulder_roll_joint: -8 * Math.PI / 180, left_elbow_joint: 1.45, right_elbow_joint: 1.45 });
    var st = { M: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], dz: 0 };
    var m4 = new R3.THREE.Matrix4();
    var api = {
      g: g, view: v, scale: S, sole: sole,
      setR: function (M) {
        st.M = M;
        m4.set(M[0][0], M[0][1], M[0][2], 0, M[1][0], M[1][1], M[1][2], 0, M[2][0], M[2][1], M[2][2], 0, 0, 0, 0, 1);
        v.robot.quaternion.setFromRotationMatrix(m4);
        v.robot.position.set(0, 0, st.dz);
        return api;
      },
      set: function (yaw) { return api.setR(E.Rz(yaw)); },
      lift: function (dz) { st.dz = dz; v.robot.position.set(0, 0, dz); return api; },
      lowest: function () { return v.box3().min.z - sole; },
      frame: function (L, lift) {
        var z0 = (lift || 0) + st.dz, o0 = v.project([0, 0, z0]);
        return { o: o0, tips: [0, 1, 2].map(function (j) { var c = E.col3(st.M, j); return v.project([L * c[0], L * c[1], L * c[2] + z0]); }) };
      },
      ring: function (r, n) {
        var pts = [];
        for (var k = 0; k <= (n || 48); k++) { var a = 2 * Math.PI * k / (n || 48); pts.push(v.project([r * Math.cos(a), r * Math.sin(a), sole])); }
        return pts;
      },
      project: function (p) { return v.project(p); },
      tint: function (part, color, amount, opacity) { v.tint(part, color, amount, opacity); }
    };
    api.set(0);
    return api;
  };

  /* 骨盆上的三根小箭头（x 前 红、y 左 绿、z 上 蓝）：E.frame3(parent) → { g, set(f)：f 是 E.g1(...).frame() 的结果 } */
  E.frame3 = function (parent, w) {
    var g = el('g', {}, parent), cols = [E.X, E.Y, E.Z];
    var arrows = cols.map(function (c) { var a = E.vec(g, c, w || 4.5, { head: 13 }); a.line.style.filter = 'drop-shadow(0 0 2px #000)'; return a; });
    var dot = K.paint(el('circle', { r: 4.5, 'stroke-width': 2 }, g), E.INK, E.S1);
    return {
      g: g, arrows: arrows,
      set: function (f) {
        f.tips.forEach(function (tp, i) { arrows[i].set(f.o[0], f.o[1], tp[0], tp[1]); });
        dot.setAttribute('cx', f.o[0].toFixed(1)); dot.setAttribute('cy', f.o[1].toFixed(1));
      }
    };
  };

  /* 一行字拆成几段 <tspan>（共用一个 <text>，间距照字体排），返回各段；逐段显隐用 style.fillOpacity。数字用 KaTeX_Main（同矩阵里的数） */
  E.spans = function (parent, x, y, parts, size, color, anchor) {
    var n = txt(parent, x, y, '', size, color, anchor || 'middle', 700);
    n.style.fontFamily = 'KaTeX_Main, serif';
    return parts.map(function (p) { var sp = document.createElementNS('http://www.w3.org/2000/svg', 'tspan'); sp.textContent = p; n.appendChild(sp); return sp; });
  };

  /* 折线点 → path 的 d（闭合可选） */
  E.ptsPath = function (pts, close) { return pts.map(function (p, k) { return (k ? 'L ' : 'M ') + p[0].toFixed(1) + ' ' + p[1].toFixed(1); }).join(' ') + (close ? ' Z' : ''); };

  /* 一张写法卡片（欧拉角 / 四元数 / 旋转矩阵）：本色圆角框 + 名字；返回 { g, rect, name }。
     不加左边的粗色条：框和名字已经是这种写法的本色，色条不多说什么（片尾总结卡片的色条是 stage.html 的版式，各集统一） */
  E.card = function (parent, x, y, w, h, name, color, size) {
    var g = el('g', {}, parent);
    var r = K.paint(el('rect', { x: x, y: y, width: w, height: h, rx: 14, 'stroke-width': 2.5 }, g), tint(color, 9, 'var(--demo-surface)'), color);
    var n = txt(g, x + 22, y + (size || 26) + 12, name, size || 26, color, 'start', 900);
    return { g: g, rect: r, name: n };
  };
})();
