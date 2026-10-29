// 零基础篇 007：齐次变换与 SE(3)——位姿 = 朝向 + 位置（路线 §L0 自测题 2「给定 SE(3) 元素 g = (R, p)，向量 v 在新坐标系下表示是什么形式？」）
// 本目录按文件名顺序加载：00 共用部件 → 10 片头主图 → 21…25 五幕分镜 → 90 组装 window.EpisodeVideo。旁白在 ../ep007.py。
// 工具函数取自 ../ep005/00-common.js（命名空间换成 EP007；坐标网格、可动箭头、KaTeX 公式）；另加手画的任意大小矩阵 E.matN（每个数的位置都知道，
// 好逐个框亮、弹出，矩阵能先分成几块各带括号、再拼成一个）、俯视的扫地机 E.vac（带自己的两根小箭头）、袜子 E.sock，
// 和斜着看的三维房间 E.room（地面网格、地图的三根轴、站在地图上某处的 G1，骨盆上三根小箭头）。
// 分镜都画在 800×420 的网格上。draw(st, t)：st 是分镜时间（旁白念完前画面停在 to 之前，一次性事件用它），
// t 是这一幕的真实时间（呼吸、闪烁这类循环动作用它，停帧时也不会卡住）。
(function () {
  var K = window.PaperDemoKit;
  var C = K.xColors;
  var E = window.EP007 = { K: K, scenes: [] };

  /* 颜色同 005、006（RELEASE_PLAN「画面配色」）：坐标轴 x 红、y 绿、z 蓝（E.X / E.Y / E.Z）——扫地机和骨盆上往前 / 往左 / 往上的小箭头、
     矩阵对应的前三列都用它；平面里的转动（绕 z 轴）用蓝；算错用橙 E.ERR（红色留给 x 轴），打勾这类「对」用 good（绿）；
     位置 p（地图原点指到机器人的箭头、齐次变换矩阵里位置那一列）violet；点（袜子、杯子）和补上的 1 / 0 用本系列亮黄 --hi；
     最下面补的一排（0、0、1）muted；「向量」这个说法用白字（绿留给「对」和 y 轴）。
     KaTeX 的 \textcolor 不认 CSS 变量，公式里用 E.HEX 的同一组色值（与 stage.html 的 :root 一致） */
  E.ACC = C.accent; E.GOOD = C.good; E.WARN = C.warn; E.VIOLET = 'var(--violet)'; E.HI = 'var(--hi)';
  E.MUT = C.muted; E.INK = C.ink; E.INK2 = C.ink2; E.S1 = C.surface; E.S2 = C.surface2; E.BORDER = C.border; E.GRID = C.grid;
  E.HEX = { acc: '#569cd6', good: '#4ec9a4', warn: '#d7a75f', violet: '#c586c0', hi: '#ffd166', ink: '#e6e6e6', ink2: '#a8a8a8', mut: '#8a8a8a',
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
        var id = 'ep007-draw-mask-' + (++maskN);
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

  E.el = el; E.txt = txt; E.op = op; E.clamp = clamp; E.fade = fade; E.pop = pop; E.scaleAt = scaleAt;
  E.drawOn = drawOn; E.arrow = arrow; E.check = check; E.cross = cross; E.tag = tag; E.tint = tint; E.glow = glow; E.halo = halo;

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
  /* ── 数：小数写成「0.8」，负号用真正的减号「−」 ── */
  E.num = function (x, digits) {
    var s = typeof x === 'string' ? x : (Math.abs(x) < 5e-3 ? '0' : x.toFixed(digits === undefined ? 2 : digits).replace(/\.?0+$/, ''));
    return s.replace('-', '−');
  };

  /* ── 一对方括号：把 [x0, x1] × [y0, y1] 这一块括起来（lip 是括号上下两个小钩的长度） ── */
  E.brk = function (parent, x0, y0, x1, y1, color, w, lipMax) {
    var lip = Math.min(Math.max(7, (y1 - y0) * 0.09), lipMax || 99), g = el('g', { 'data-bracket': '' }, parent);
    [[x0, 1], [x1, -1]].forEach(function (d) {
      K.paint(el('path', { d: 'M ' + (d[0] + d[1] * lip) + ' ' + y0 + ' H ' + d[0] + ' V ' + y1 + ' H ' + (d[0] + d[1] * lip),
        fill: 'none', 'stroke-width': w || 2.5, 'stroke-linecap': 'square' }, g), null, color || E.INK);
    });
    return g;
  };

  /* ── 手画的矩阵 / 列向量（不用 KaTeX）：每个数的位置都知道，好逐个框亮、弹出。
     E.matN(parent, cx, cy, M, o)：M 按行存（数或字符串）；o = { size：字号（默认 30），dx / dy：列距 / 排距，
       color(i, j) → 这个数的颜色（默认 INK），wide：按哪组数留宽度（默认 M；数会变的矩阵给最宽的那组） }
     框和括号按每列最宽的数留空（同 006 的修法）：框比数宽出 0.25 字号，括号再离框 0.36 字号，数不贴框、框不压括号。
     返回 { g, cells[i][j], pos(i, j) → [x, y]（这个数的中心）, set(M), frame（整块的括号）, brk(i0, j0, i1, j1)：给其中一块另画括号,
       hl(key, i0, j0, i1, j1, color, p)：在这一块底下垫一个框（key 相同就复用同一个框）, box（括号外框） } ── */
  // KaTeX_Main 的字宽（em）：数字 0.5、小数点 0.278、减号 0.778，斜体字母约 0.55
  function textW(s, size) {
    var w = 0;
    String(s).split('').forEach(function (c) { w += /[0-9]/.test(c) ? 0.5 : c === '.' || c === ',' ? 0.278 : c === '−' ? 0.778 : 0.55; });
    return w * size;
  }
  // 每列最宽的数有多宽；括号竖笔离最左 / 最右一列中心多远（E.flow 排一行算式时也用）
  function colWidths(M, o, size) {
    var out = [];
    for (var j = 0; j < M[0].length; j++) out.push(Math.max.apply(null, (o.wide || M).map(function (row) { return textW(E.num(row[j]), size); })));
    return out;
  }
  E.matExt = function (M, o) {
    o = o || {};
    var size = o.size || 30, dx = o.dx || size * 1.9, cw = colWidths(M, o, size), w = (M[0].length - 1) * dx, k = size * (0.25 + 0.36);
    return [w / 2 + cw[0] / 2 + k + 1.5, w / 2 + cw[cw.length - 1] / 2 + k + 1.5];   // 左、右（含括号线宽的一半）
  };
  E.matN = function (parent, cx, cy, M, o) {
    o = o || {};
    var size = o.size || 30, dx = o.dx || size * 1.9, dy = o.dy || size * 1.3;
    var rows = M.length, cols = M[0].length, w = (cols - 1) * dx, h = (rows - 1) * dy;
    var g = el('g', { 'data-mgrid': '' }, parent), under = el('g', {}, g), boxes = {};   // data-mgrid / data-bracket / data-hl：给版面检查认
    var colW = colWidths(M, o, size);
    var IN = size * 0.25, GAP = size * 0.36;
    function pos(i, j) { return [cx - w / 2 + j * dx, cy - h / 2 + i * dy]; }
    function block(i0, j0, i1, j1) {
      var a = pos(i0, j0), b = pos(i1, j1);
      return [a[0] - colW[j0] / 2 - IN, a[1] - size * 0.62, b[0] + colW[j1] / 2 + IN, b[1] + size * 0.62];
    }
    function brkBox(i0, j0, i1, j1) { var b = block(i0, j0, i1, j1); return [b[0] - GAP, b[1] - size * 0.12, b[2] + GAP, b[3] + size * 0.12]; }   // 括号上下也比框长一点，小钩不碰框角
    var fb = brkBox(0, 0, rows - 1, cols - 1);
    var frame = E.brk(g, fb[0], fb[1], fb[2], fb[3], o.bracket || E.INK, Math.max(2, size / 14), GAP - 1);   // 小钩不伸到框上面
    var cells = M.map(function (row, i) {
      return row.map(function (v, j) {
        var p = pos(i, j), n = txt(g, p[0], p[1] + size * 0.35, E.num(v), size, o.color ? o.color(i, j) : E.INK, 'middle');
        n.style.fontFamily = 'KaTeX_Main, serif';
        return n;
      });
    });
    var api = {
      g: g, cells: cells, pos: pos, frame: frame, size: size, box: fb,
      set: function (m) { m.forEach(function (row, i) { row.forEach(function (v, j) { cells[i][j].textContent = E.num(v); }); }); return api; },
      brk: function (i0, j0, i1, j1, color) {
        var b = brkBox(i0, j0, i1, j1);
        return E.brk(g, b[0], b[1], b[2], b[3], color || E.INK, Math.max(2, size / 14), GAP - 1);
      },
      hl: function (key, i0, j0, i1, j1, color, p) {
        var r = boxes[key] || (boxes[key] = el('rect', { rx: 9, 'stroke-width': 2.5, 'data-hl': '' }, under)), b = block(i0, j0, i1, j1);
        r.setAttribute('x', b[0].toFixed(1)); r.setAttribute('y', b[1].toFixed(1));
        r.setAttribute('width', (b[2] - b[0]).toFixed(1)); r.setAttribute('height', (b[3] - b[1]).toFixed(1));
        K.paint(r, tint(color, 18, 'var(--demo-surface)'), color); op(r, clamp(p, 0, 1));
        return r;
      }
    };
    return api;
  };

  /* ── 一行算式从左往右排：E.flow(x, y, gap) 记着排到哪儿了，按矩阵括号的真实宽度和字宽留空（gap：括号和旁边的字之间默认 12）。
     .mat(parent, M, o) → E.matN；.sym(parent, s, size, color) → 字（KaTeX_Main，如 =、+、×、2）；.txt(parent, s, size, color, weight) → 中文标签；
     .skip(d) 多空一点；.x 现在排到的位置 ── */
  // 估一行字有多宽（px）：汉字和圈码 1 字号，数字 0.5，=、+、−、× 0.78，小数点、逗号、空格 0.28，其余 0.55
  function estW(s0, size) {
    var w = 0;
    String(s0).split('').forEach(function (c) {
      w += /[\u2460-\u24ff\u2e80-\uffff]/.test(c) ? 1 : /[0-9]/.test(c) ? 0.5 : /[=+\u2212\u00d7]/.test(c) ? 0.78 : /[., ]/.test(c) ? 0.28 : 0.55;
    });
    return w * size;
  }
  E.flow = function (x, y, gap) {
    var G = gap === undefined ? 12 : gap;
    function step(g) { return g === undefined ? G : g; }
    var api = {
      x: x,
      mat: function (parent, M, o, g) {
        var e = E.matExt(M, o), cx = api.x + step(g) + e[0];
        api.x = cx + e[1];
        return E.matN(parent, cx, y, M, o);
      },
      sym: function (parent, s0, size, color, g) {
        var w = estW(s0, size), cx = api.x + step(g) + w / 2;
        api.x = cx + w / 2;
        var n = txt(parent, cx, y + size * 0.34, s0, size, color || E.INK, 'middle', 400);
        n.style.fontFamily = 'KaTeX_Main, serif';
        return n;
      },
      txt: function (parent, s0, size, color, weight, g) {
        var x0 = api.x + step(g);
        api.x = x0 + estW(s0, size);
        return txt(parent, x0, y + size * 0.36, s0, size, color || E.INK, 'start', weight || 900);
      },
      skip: function (d) { api.x += d; return api; }
    };
    return api;
  };

  /* ── 俯视的扫地机（iRobot Create 2 的 URDF，由 ../../robot3d.js 渲染，同 003 第 2 幕）+ 它自己的两根小箭头：
     往前（红）、往左（绿），各长一格。E.vac(parent, P, s)：P 是坐标网格的 P(x, y)；s 是缩放（宽 92·s px）。
     set(x, y, heading)：停在网格上 (x, y)，车头朝 heading 弧度（0 = 朝 +x，逆时针为正）；arrows：两根小箭头（E.vec），
     tips() → 两根小箭头尖的画面坐标 ── */
  E.vac = function (parent, P, s) {
    var g = el('g', {}, parent), body = el('g', {}, g);
    window.Robot3D.create2(body, 0, 0, s, { top: true });
    var ar = el('g', {}, g);
    var arrows = [E.X, E.Y].map(function (c) { var a = E.vec(ar, c, 5, { head: 15 }); a.line.style.filter = 'drop-shadow(0 0 2px #000)'; return a; });
    var dot = K.paint(el('circle', { r: 4.5, 'stroke-width': 2 }, ar), E.INK, E.S1);
    var st = { x: 0, y: 0, h: 0 };
    var api = {
      g: g, body: body, arrowsG: ar, arrows: arrows,
      set: function (x, y, h) {
        st.x = x; st.y = y; st.h = h;
        var c = P(x, y);
        body.setAttribute('transform', 'translate(' + c[0].toFixed(2) + ' ' + c[1].toFixed(2) + ') rotate(' + (-h * 180 / Math.PI).toFixed(3) + ')');
        var t = api.tips();
        arrows.forEach(function (a, i) { a.set(c[0], c[1], t[i][0], t[i][1]); });
        dot.setAttribute('cx', c[0].toFixed(2)); dot.setAttribute('cy', c[1].toFixed(2));
        return api;
      },
      tips: function () {
        return [P(st.x + Math.cos(st.h), st.y + Math.sin(st.h)), P(st.x - Math.sin(st.h), st.y + Math.cos(st.h))];
      }
    };
    return api;
  };

  /* ── 袜子（地图上的一个点）：画面坐标 (x, y) 是脚面中间，size 是高 ── */
  E.sock = function (parent, x, y, size, color) {
    var k = (size || 30) / 30, g = el('g', {}, parent), c = color || E.HI;
    var d = 'M -9 -16 L 3 -16 L 3 -1 L 10 -1 Q 17 -1 17 6 Q 17 13 10 13 L -4 13 Q -9 13 -9 8 Z';
    var inner = el('g', { transform: 'translate(' + x + ' ' + y + ') scale(' + k + ')' }, g);
    K.paint(el('path', { d: d, 'stroke-width': 2.5 / k, 'stroke-linejoin': 'round' }, inner), tint(c, 35, 'var(--demo-surface)'), c);
    K.paint(el('path', { d: 'M -9 -10 L 3 -10', fill: 'none', 'stroke-width': 2.5 / k }, inner), null, c);
    return g;
  };

  /* ── 三维：数学坐标 x、y、z（z 朝上），斜着看。E.view3(az, el) 给出投影：P(v) → [右, 上]（单位长度的画面分量）；
     az / el 的定义同 ../../robot3d.js（az 0 = 从 −y 方向看，el > 0 = 从上往下看） ── */
  E.view3 = function (az, el) {
    var a = az * Math.PI / 180, e = el * Math.PI / 180;
    var r = [Math.cos(a), Math.sin(a), 0], u = [-Math.sin(a) * Math.sin(e), Math.cos(a) * Math.sin(e), Math.cos(e)];
    return function (v) { return [v[0] * r[0] + v[1] * r[1] + v[2] * r[2], v[0] * u[0] + v[1] * u[1] + v[2] * u[2]]; };
  };

  /* ── 斜着看的三维房间（地面 z = 0，单位：米）：地面网格、地图原点的三根轴（灰）、站在地图上 (px, py) 的 G1（Robot Description Gallery Online
     收录的 URDF，由 ../../robot3d.js 渲染），骨盆朝 yaw。E.room(parent, o)：o = { ox, oy：地图原点在画面里的位置；scale：一米多少 px；az, el；
     floor：[x0, x1, y0, y1] 地面网格的范围；step：网格间距（默认 0.5）；at：[px, py]；yaw；margin：G1 画布框比模型外框多留几 px（默认 8，原地转身时放宽） }
     返回 { g, floor, axes, P(v)：三维点 → 画面坐标, robot（view）, pelvis：骨盆原点（三维）, R(v)：按骨盆朝向转一个三维方向, setYaw(yaw)：原地转身,
            frame(L)、at(link, local)、linkFrame(link, L) } ── */
  E.room = function (parent, o) {
    var R3 = window.Robot3D, B = R3.measure('g1'), H = -B.min.z;          // 站直时骨盆离地 H 米
    var proj = E.view3(o.az, o.el), S = o.scale;
    function P(v) { var q = proj(v); return [o.ox + S * q[0], o.oy - S * q[1]]; }
    var g = el('g', {}, parent), floor = el('g', { 'data-grid': '1' }, g), axes = el('g', {}, g), bot = el('g', {}, g);   // data-grid：同 E.plane
    var f = o.floor, st = o.step || 0.5, x, y;
    for (x = f[0]; x <= f[1] + 1e-6; x += st) { var a = P([x, f[2], 0]), b = P([x, f[3], 0]); K.paint(el('line', { x1: a[0], y1: a[1], x2: b[0], y2: b[1], 'stroke-width': 1.5 }, floor), null, E.GRID); }
    for (y = f[2]; y <= f[3] + 1e-6; y += st) { var c = P([f[0], y, 0]), d = P([f[1], y, 0]); K.paint(el('line', { x1: c[0], y1: c[1], x2: d[0], y2: d[1], 'stroke-width': 1.5 }, floor), null, E.GRID); }
    var O = P([0, 0, 0]), L = o.axis || 0.5;
    ['x', 'y', 'z'].forEach(function (n, i) {
      var e = [0, 0, 0]; e[i] = L;
      var t = P(e), v = E.vec(axes, E.INK2, 3, { head: 12 });
      v.set(O[0], O[1], t[0], t[1]);
      var dx = t[0] - O[0], dy = t[1] - O[1], len = Math.hypot(dx, dy) || 1;
      halo(txt(axes, t[0] + dx / len * 16, t[1] + dy / len * 16 + 7, n, 20, E.INK2, 'middle', 700));
    });
    K.paint(el('circle', { cx: O[0], cy: O[1], r: 4.5 }, axes), E.INK2);
    // G1：按 robot3d 的同一套投影（origin = 地图原点），骨盆放在 (px, py, H)，朝 yaw；画布框取模型投影外框再放宽一点
    var pel = [o.at[0], o.at[1], H], yaw = o.yaw || 0;
    var v = R3.view(bot, { robot: 'g1', origin: [o.ox, o.oy], scale: S, az: o.az, el: o.el, box: [0, 0, 10, 10], res: o.res || 2 });
    var D = Math.PI / 180;
    v.joints({ left_shoulder_roll_joint: 8 * D, right_shoulder_roll_joint: -8 * D, left_elbow_joint: 1.45, right_elbow_joint: 1.45 });
    v.base(pel, [0, 0, yaw]);
    var bb = v.box3(), x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (var k = 0; k < 8; k++) {
      var p = P([k & 1 ? bb.max.x : bb.min.x, k & 2 ? bb.max.y : bb.min.y, k & 4 ? bb.max.z : bb.min.z]);
      x0 = Math.min(x0, p[0]); y0 = Math.min(y0, p[1]); x1 = Math.max(x1, p[0]); y1 = Math.max(y1, p[1]);
    }
    var mg = o.margin || 8;
    v.camera({ box: [x0 - mg, y0 - mg, x1 - x0 + 2 * mg, y1 - y0 + 2 * mg] });
    function R(d) { var c = Math.cos(yaw), s = Math.sin(yaw); return [c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]]; }
    return {
      g: g, floor: floor, axes: axes, bot: bot, robot: v, P: P, pelvis: pel, R: R, H: H,
      setYaw: function (y) { yaw = y; v.base(pel, [0, 0, yaw]); },
      // 骨盆上三根小箭头（机器人的前、左、上，各长 L 米）在画面里的起点和三个箭头尖
      frame: function (Lf) {
        return { o: P(pel), tips: [[Lf, 0, 0], [0, Lf, 0], [0, 0, Lf]].map(function (d) { var q = R(d); return P([pel[0] + q[0], pel[1] + q[1], pel[2] + q[2]]); }) };
      },
      // 连杆 name 上一点（连杆坐标系里，米）→ 画面坐标；这一节自己的三根小箭头（各长 Lf 米）
      at: function (name, local) { var w = v.world(name, local); return P([w.x, w.y, w.z]); },
      linkFrame: function (name, Lf) {
        var o0 = v.world(name, [0, 0, 0]);
        return { o: P([o0.x, o0.y, o0.z]), tips: [[Lf, 0, 0], [0, Lf, 0], [0, 0, Lf]].map(function (d) { var w = v.world(name, d); return P([w.x, w.y, w.z]); }) };
      }
    };
  };

  /* 三根小箭头（x 前 红、y 左 绿、z 上 蓝）：E.frame3(parent, w) → { g, set(f)：f 是 room.frame() / linkFrame() 的结果 } */
  E.frame3 = function (parent, w, head) {
    var g = el('g', {}, parent), cols = [E.X, E.Y, E.Z];
    var arrows = cols.map(function (c) { var a = E.vec(g, c, w || 4.5, { head: head || 13 }); a.line.style.filter = 'drop-shadow(0 0 2px #000)'; return a; });
    var dot = K.paint(el('circle', { r: 4.5, 'stroke-width': 2 }, g), E.INK, E.S1);
    return {
      g: g, arrows: arrows,
      set: function (f) {
        f.tips.forEach(function (tp, i) { arrows[i].set(f.o[0], f.o[1], tp[0], tp[1]); });
        dot.setAttribute('cx', f.o[0].toFixed(1)); dot.setAttribute('cy', f.o[1].toFixed(1));
      }
    };
  };

  /* ── 三维的杯子（房间里的一个点）：一个圆柱，按房间同一个投影 P 画出来——底面、口都是椭圆，侧面取左右两条切线，口里面压暗，
     把手是立在侧面的半个圈（同样按 P 投影）。E.cup3(parent, P, at, o)：at = [x, y] 杯底中心（米，地面 z = 0）；o = { r（默认 0.07）, h（默认 0.13）, color }
     返回 { g, top：杯口中心在画面里的位置 } ── */
  var cupN = 0;
  E.cup3 = function (parent, P, at, o) {
    o = o || {};
    var r = o.r || 0.07, h = o.h || 0.13, c = o.color || E.HI, n = 48, g = el('g', {}, parent);
    var B = [], T = [];
    for (var k = 0; k < n; k++) {
      var a = 2 * Math.PI * k / n;
      B.push(P([at[0] + r * Math.cos(a), at[1] + r * Math.sin(a), 0])); T.push(P([at[0] + r * Math.cos(a), at[1] + r * Math.sin(a), h]));
    }
    // 正交投影里竖直向量只往画面上下挪：底面和口的左右切点是同一个角度
    var iL = 0, iR = 0;
    B.forEach(function (q, k) { if (q[0] < B[iL][0]) iL = k; if (q[0] > B[iR][0]) iR = k; });
    function arc(R, from, to, step) { var out = [], k = from; for (;;) { out.push(R[k]); if (k === to) break; k = (k + step + n) % n; } return out; }
    // 底面靠近看的人那半圈：画面上更低（y 更大）的那一边
    var fwd = arc(B, iL, iR, 1), bwd = arc(B, iL, iR, -1);
    var avg = function (pts) { return pts.reduce(function (s0, q) { return s0 + q[1]; }, 0) / pts.length; };
    var step = avg(fwd) > avg(bwd) ? 1 : -1;
    var front = arc(B, iL, iR, step), topFront = arc(T, iR, iL, -step);
    function path(pts, close) { return 'M ' + pts.map(function (q) { return q[0].toFixed(1) + ' ' + q[1].toFixed(1); }).join(' L ') + (close ? ' Z' : ''); }
    var id = 'ep007-cup-' + (++cupN), defs = el('defs', {}, g);
    var gr = el('linearGradient', { id: id, x1: '0', y1: '0', x2: '1', y2: '0' }, defs);
    [['0', 48], ['0.55', 30], ['1', 16]].forEach(function (d) { var st = el('stop', { offset: d[0] }, gr); st.style.stopColor = tint(c, d[1], 'var(--demo-surface)'); });
    // 把手：杯身朝画面右边那一侧、半高处，立着的半个圈
    var side = [P([1, 0, 0])[0] - P([0, 0, 0])[0], P([0, 1, 0])[0] - P([0, 0, 0])[0]], sl = Math.hypot(side[0], side[1]);
    var dir = [side[0] / sl, side[1] / sl], hp = [];
    for (k = 0; k <= 16; k++) {
      var ph = Math.PI * k / 16, rr = r + 0.045 * Math.sin(ph);
      hp.push(P([at[0] + dir[0] * rr, at[1] + dir[1] * rr, h * 0.5 + 0.035 * Math.cos(ph)]));
    }
    K.paint(el('path', { d: path(hp), fill: 'none', 'stroke-width': 4, 'stroke-linecap': 'round' }, g), null, c);
    var body = el('path', { d: path([T[iL]].concat(front, topFront), true), 'stroke-width': 2.5, 'stroke-linejoin': 'round' }, g);
    body.style.fill = 'url(#' + id + ')'; K.paint(body, null, c);
    K.paint(el('path', { d: path(T, true), 'stroke-width': 2.5 }, g), tint(c, 8, 'var(--demo-surface-2)'), c);   // 杯口：里面压暗
    return { g: g, top: P([at[0], at[1], h]) };
  };

  E.bump = function (st, a, d) { var u = K.seg(st, a, a + d); return u > 0 && u < 1 ? Math.sin(Math.PI * u) : 0; };
})();
