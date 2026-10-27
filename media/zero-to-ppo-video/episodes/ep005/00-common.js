// 零基础篇 005：旋转矩阵——为什么旋转不能直接相加（路线 §L0 自测题 1「旋转矩阵 R 为什么不能直接做加法 / 插值」）
// 本目录按文件名顺序加载：00 共用部件 → 10 片头主图 → 21…25 五幕分镜 → 90 组装 window.EpisodeVideo。旁白在 ../ep005.py。
// 工具函数取自 ../ep003/00-common.js（命名空间换成 EP005；坐标网格、可动箭头、KaTeX 向量 / 矩阵），描边 E.halo 与手画矩阵 E.mat 取自 ../ep004/00-common.js；
// 另加三维小坐标架 E.gizmo（三根小箭头 + 小方块，斜着看）和能原地转身、能被「压瘦」的 G1（E.g1）。
// 分镜都画在 800×420 的网格上。draw(st, t)：st 是分镜时间（旁白念完前画面停在 to 之前，一次性事件用它），
// t 是这一幕的真实时间（呼吸、旋转、绕圈这类循环动作用它，停帧时也不会卡住）。
(function () {
  var K = window.PaperDemoKit;
  var C = K.xColors;
  var E = window.EP005 = { K: K, scenes: [] };

  /* 颜色：坐标轴按机器人领域的习惯 x 红、y 绿、z 蓝（E.X / E.Y / E.Z）——往右的一格 / 第一列 / x 轴红，往上的一格 / 第二列 / y 轴绿，
     第三根小箭头 / 第三列 / z 轴蓝；平面里的转动是绕 z 轴转，转动的弧线、小弯箭头也用蓝。矩阵 / 变换 violet；
     不合格、跳出旋转矩阵、镜子的背面用橙 E.ERR（红色留给 x 轴）；打勾、规矩牌这类「对」仍用 good（绿）；本系列亮黄 --hi 只做强调（小旗、小方块）。
     KaTeX 的 \textcolor 不认 CSS 变量，公式里用 E.HEX 的同一组色值（与 stage.html 的 :root 一致） */
  E.ACC = C.accent; E.GOOD = C.good; E.WARN = C.warn; E.BAD = C.bad; E.VIOLET = 'var(--violet)'; E.HI = 'var(--hi)';
  E.MUT = C.muted; E.INK = C.ink; E.INK2 = C.ink2; E.S1 = C.surface; E.S2 = C.surface2; E.BORDER = C.border; E.GRID = C.grid;
  E.HEX = { acc: '#569cd6', good: '#4ec9a4', warn: '#d7a75f', bad: '#e06c75', violet: '#c586c0', hi: '#ffd166', ink: '#e6e6e6', ink2: '#a8a8a8', mut: '#8a8a8a',
    x: '#e06c75', y: '#4ec9a4', z: '#569cd6', err: '#ff7b3a' };
  E.Z = E.HEX.z; E.ERR = E.HEX.err;
  /* 003、004 的两根小箭头是 x 绿、y 琥珀：片头和第 1 幕开头先沿用，第 1 幕讲颜色约定时渐变成 x 红、y 绿。
     E.X / E.Y 是 CSS 变量（这一幕没设就是新配色）；E.axes(svg, px, py) 在这一幕的 svg 上设：0 = 旧色、1 = 新色，返回两个 hex 给 KaTeX 用 */
  E.HEX.x0 = E.HEX.good; E.HEX.y0 = E.HEX.warn;
  E.X = 'var(--ax-x, ' + E.HEX.x + ')'; E.Y = 'var(--ax-y, ' + E.HEX.y + ')';
  function mixHex(a, b, p) {
    var A = parseInt(a.slice(1), 16), B = parseInt(b.slice(1), 16);
    return '#' + [16, 8, 0].map(function (s) { var u = (A >> s) & 255, v = (B >> s) & 255; return ('0' + Math.round(u + (v - u) * p).toString(16)).slice(-2); }).join('');
  }
  E.axes = function (svg, px, py) {
    var x = mixHex(E.HEX.x0, E.HEX.x, px), y = mixHex(E.HEX.y0, E.HEX.y, py);
    svg.style.setProperty('--ax-x', x); svg.style.setProperty('--ax-y', y);
    return [x, y];
  };

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
        var id = 'ep005-draw-mask-' + (++maskN);
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
  // 上色的一个数：\textcolor{#hex}{x}（「-」在 KaTeX 数学模式里本来就排成减号）
  function cnum(x, hex) { var s = String(x); return hex ? '\\textcolor{' + hex + '}{' + s + '}' : s; }
  // 列向量：col([3, 2]) → \begin{bmatrix}3\\2\end{bmatrix}；hex 可以是一个颜色或每个数各一个
  E.col = function (xs, hex) {
    return '\\begin{bmatrix}' + xs.map(function (x, i) { return cnum(x, Array.isArray(hex) ? hex[i] : hex); }).join('\\\\') + '\\end{bmatrix}';
  };
  // 2×2 矩阵：mat2([a, b, c, d]) 即 [[a, b], [c, d]]；colHex = [第一列颜色, 第二列颜色]
  E.mat2 = function (m, colHex) {
    var h = colHex || [];
    return '\\begin{bmatrix}' + cnum(m[0], h[0]) + '&' + cnum(m[1], h[1]) + '\\\\' + cnum(m[2], h[0]) + '&' + cnum(m[3], h[1]) + '\\end{bmatrix}';
  };
  // 2×2 矩阵乘法、乘向量
  E.mul = function (A, B) { return [A[0] * B[0] + A[1] * B[2], A[0] * B[1] + A[1] * B[3], A[2] * B[0] + A[3] * B[2], A[2] * B[1] + A[3] * B[3]]; };
  E.apply = function (m, v) { return [m[0] * v[0] + m[1] * v[1], m[2] * v[0] + m[3] * v[1]]; };
  // 转 a 弧度的矩阵（逆时针为正）；从单位阵到 m 的线性插值（拉长、斜推用；转动用 E.rot 插角度，免得中途被压扁）
  E.rot = function (a) { return [Math.cos(a), -Math.sin(a), Math.sin(a), Math.cos(a)]; };
  E.lerpM = function (m, p) { return [1 + (m[0] - 1) * p, m[1] * p, m[2] * p, 1 + (m[3] - 1) * p]; };

  /* ── 数：小数写成「0.8」，负号用真正的减号「−」 ── */
  E.num = function (x, digits) {
    var s = typeof x === 'string' ? x : (Math.abs(x) < 5e-3 ? '0' : x.toFixed(digits === undefined ? 2 : digits).replace(/\.?0+$/, ''));
    return s.replace('-', '−');
  };

  /* ── 手画的 2×2 矩阵（不用 KaTeX，取自 004）：每个数的位置都知道，好逐个高亮、让数飞进飞出。
     E.mat(parent, cx, cy, m, o)：m = [a, b, c, d] 即 [[a, b], [c, d]]（数或字符串）；o = { size：字号（默认 40），dx / dy：列距 / 排距，colors：四个数各自的颜色，
     pad：括号离列中心多远（默认 0.95 字号），cw：框列 / 框数的宽（默认 1.6 字号，「−0.6」这种要放宽） }
     返回 { g, cells[k]（文字）, pos(k) → [x, y]（第 k 个数的中心）, set(m)：换成另一组数, colBox(c, color, p), cellBox(k, color, p), box（外框） } ── */
  E.mat = function (parent, cx, cy, m, o) {
    o = o || {};
    var size = o.size || 40, dx = o.dx || size * 1.7, dy = o.dy || size * 1.35;
    var g = el('g', {}, parent), under = el('g', {}, g);
    var bh = dy + size * 1.15, bx = dx / 2 + (o.pad || size * 0.95), CW = o.cw || size * 1.6;
    var brackets = [-1, 1].map(function (sd) {
      var xb = cx + sd * bx, lip = size * 0.28;
      var b = el('path', { d: 'M ' + (xb - sd * lip) + ' ' + (cy - bh / 2) + ' H ' + xb + ' V ' + (cy + bh / 2) + ' H ' + (xb - sd * lip),
        fill: 'none', 'stroke-width': Math.max(2, size / 16), 'stroke-linecap': 'square' }, g);
      K.paint(b, null, o.bracket || E.INK);
      return b;
    });
    function pos(k) { return [cx + (k % 2 ? 1 : -1) * dx / 2, cy + (k < 2 ? -1 : 1) * dy / 2]; }
    var cols = [0, 1].map(function () { return el('rect', { rx: 10, 'stroke-width': 2.5 }, under); });
    var cellR = [0, 1, 2, 3].map(function () { return el('rect', { rx: 8, 'stroke-width': 2.5 }, under); });
    cols.concat(cellR).forEach(function (r) { op(r, 0); });
    var cells = m.map(function (v, k) {
      var p = pos(k), n = txt(g, p[0], p[1] + size * 0.35, E.num(v), size, (o.colors || [])[k] || E.INK, 'middle');
      n.style.fontFamily = 'KaTeX_Main, serif';
      return n;
    });
    return {
      g: g, cells: cells, pos: pos, brackets: brackets, size: size, dx: dx, dy: dy,
      box: [cx - bx, cy - bh / 2, cx + bx, cy + bh / 2],
      set: function (mm) { mm.forEach(function (v, k) { cells[k].textContent = E.num(v); }); },
      colBox: function (c, color, p) {
        var b = cols[c], xc = cx + (c ? 1 : -1) * dx / 2, w = Math.min(CW, dx - 6);
        b.setAttribute('x', (xc - w / 2).toFixed(1)); b.setAttribute('width', w.toFixed(1));
        b.setAttribute('y', (cy - dy / 2 - size * 0.62).toFixed(1)); b.setAttribute('height', (dy + size * 1.24).toFixed(1));
        K.paint(b, tint(color, 16, 'var(--demo-surface)'), color); op(b, clamp(p, 0, 1));
      },
      cellBox: function (k, color, p) {
        var b = cellR[k], q = pos(k), w = Math.min(CW, dx - 8), hh = Math.min(size * 1.1, dy - 6);
        b.setAttribute('x', (q[0] - w / 2).toFixed(1)); b.setAttribute('width', w.toFixed(1));
        b.setAttribute('y', (q[1] - hh / 2).toFixed(1)); b.setAttribute('height', hh.toFixed(1));
        K.paint(b, tint(color, 22, 'var(--demo-surface)'), color); op(b, clamp(p, 0, 1));
      }
    };
  };

  /* ── 手画的 3×3 矩阵：E.mat3(parent, cx, cy, M, o)，M 按行存；o = { size（默认 28）, dx / dy：列距 / 排距, colors：三列各自的颜色, digits }
     返回 { g, set(M), colBox(j, color, p), box } ── */
  E.mat3 = function (parent, cx, cy, M, o) {
    o = o || {};
    var size = o.size || 28, dx = o.dx || size * 2.7, dy = o.dy || size * 1.3;
    var g = el('g', {}, parent), under = el('g', {}, g);
    var bh = 2 * dy + size * 1.15, bx = dx + size * 1.9;
    [-1, 1].forEach(function (sd) {
      var xb = cx + sd * bx, lip = size * 0.28;
      K.paint(el('path', { d: 'M ' + (xb - sd * lip) + ' ' + (cy - bh / 2) + ' H ' + xb + ' V ' + (cy + bh / 2) + ' H ' + (xb - sd * lip),
        fill: 'none', 'stroke-width': Math.max(2, size / 14), 'stroke-linecap': 'square' }, g), null, E.INK);
    });
    var boxes = [0, 1, 2].map(function () { var r = el('rect', { rx: 10, 'stroke-width': 2.5 }, under); op(r, 0); return r; });
    var cells = [];
    for (var i = 0; i < 3; i++) for (var j = 0; j < 3; j++) {
      var n = txt(g, cx + (j - 1) * dx, cy + (i - 1) * dy + size * 0.35, '', size, (o.colors || [])[j] || E.INK, 'middle');
      n.style.fontFamily = 'KaTeX_Main, serif';
      cells.push(n);
    }
    var api = {
      g: g, cells: cells, box: [cx - bx, cy - bh / 2, cx + bx, cy + bh / 2],
      set: function (m) { for (var k = 0; k < 9; k++) cells[k].textContent = E.num(m[Math.floor(k / 3)][k % 3], o.digits); return api; },
      colBox: function (j, color, p) {
        var b = boxes[j], w = dx - 4;
        b.setAttribute('x', (cx + (j - 1) * dx - w / 2).toFixed(1)); b.setAttribute('width', w.toFixed(1));
        b.setAttribute('y', (cy - dy - size * 0.62).toFixed(1)); b.setAttribute('height', (2 * dy + size * 1.24).toFixed(1));
        K.paint(b, tint(color, 16, 'var(--demo-surface)'), color); op(b, clamp(p, 0, 1));
      }
    };
    api.set(M);
    return api;
  };

  /* ── 三维：数学坐标 x、y、z（z 朝上），斜着看。E.view3(az, el) 给出投影：P(v) → [右, 上]（单位长度的画面分量）；
     az / el 的定义同 ../../robot3d.js（az 0 = 从 −y 方向看，el > 0 = 从上往下看） ── */
  E.view3 = function (az, el) {
    var a = az * Math.PI / 180, e = el * Math.PI / 180;
    var r = [Math.cos(a), Math.sin(a), 0], u = [-Math.sin(a) * Math.sin(e), Math.cos(a) * Math.sin(e), Math.cos(e)];
    return function (v) { return [v[0] * r[0] + v[1] * r[1] + v[2] * r[2], v[0] * u[0] + v[1] * u[1] + v[2] * u[2]]; };
  };
  // 3×3：按行存 [[..], [..], [..]]；绕单位轴 n 转 a 弧度（罗德里格斯公式）、相乘、乘向量、取第 j 列
  E.rot3 = function (n, a) {
    var L = Math.hypot(n[0], n[1], n[2]), x = n[0] / L, y = n[1] / L, z = n[2] / L, c = Math.cos(a), s = Math.sin(a), C = 1 - c;
    return [[c + x * x * C, x * y * C - z * s, x * z * C + y * s], [y * x * C + z * s, c + y * y * C, y * z * C - x * s], [z * x * C - y * s, z * y * C + x * s, c + z * z * C]];
  };
  E.mul3 = function (A, B) {
    return A.map(function (row) { return [0, 1, 2].map(function (j) { return row[0] * B[0][j] + row[1] * B[1][j] + row[2] * B[2][j]; }); });
  };
  E.col3 = function (M, j) { return [M[0][j], M[1][j], M[2][j]]; };
  // KaTeX 的 3×3 矩阵，三列各一个颜色
  E.mat3tex = function (M, hex, digits) {
    return '\\begin{bmatrix}' + M.map(function (row) {
      return row.map(function (v, j) { var s = E.num(v, digits).replace('−', '-'); return hex ? '\\textcolor{' + hex[j] + '}{' + s + '}' : s; }).join('&');
    }).join('\\\\') + '\\end{bmatrix}';
  };

  /* ── 三维小坐标架：原点 (cx, cy)，单位长 L px；三根小箭头（x 红、y 绿、z 蓝）+ 三根小箭头围成的小方块（淡线）。
     set(M)：M 是 3×3（列 = 三根小箭头转完以后的方向，可以不是旋转矩阵）；proj：E.view3 的投影 ── */
  E.gizmo = function (parent, cx, cy, L, proj, opts) {
    var o = opts || {};
    var g = el('g', {}, parent), box = el('g', {}, g);
    var edges = [];
    for (var k = 0; k < 12; k++) edges.push(K.paint(el('line', { 'stroke-width': 1.6, 'stroke-dasharray': '5 5' }, box), null, E.MUT));
    var cols = [E.X, E.Y, E.Z];
    var arrows = cols.map(function (c) { return E.vec(g, c, o.w || 4.5, { head: 14 }); });
    var labs = ['x', 'y', 'z'].map(function (s, i) { return halo(txt(g, 0, 0, s, 22, cols[i], 'middle', 700)); });
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
          labs[i].setAttribute('x', (tp[0] + dx / len * 22).toFixed(1)); labs[i].setAttribute('y', (tp[1] + dy / len * 22 + 8).toFixed(1));
        });
        return api;
      }
    };
    api.set([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
    return api;
  };

  /* ── G1（Robot Description Gallery Online 收录的 URDF，由 ../../robot3d.js 渲染）：站着，斜着看，能原地转身、能被「压瘦」。
     E.g1(parent, x, y, h, o)：两脚中间的地面点放在 (x, y)，身高 h px；o = { az, el（默认 55、16） }。
     set(yaw, s)：绕竖直轴转 yaw 弧度（从上往下看逆时针为正，往机器人的左边转），水平方向压成 s 倍（s = 1 不变形）——
     也就是身子按矩阵 Rz(yaw)·diag(s, s, 1) 摆：零度和 θ 两个朝向直接平均，正好是 Rz(θ/2)·diag(cos(θ/2), cos(θ/2), 1)。
     frame(L)：骨盆上三根小箭头（机器人的前、左、上，各长 L 米，跟着 set 一起转、一起压）在画面里的起点和三个箭头尖；
     ring(r)：脚下半径 r 米的地面圆（画面里的折线点） ── */
  E.g1 = function (parent, x, y, h, o) {
    o = o || {};
    var R3 = window.Robot3D, B = R3.measure('g1'), sole = B.min.z, top = B.max.z;
    var az = o.az === undefined ? 55 : o.az, elv = o.el === undefined ? 16 : o.el, S = h / (top - sole);
    var g = el('g', {}, parent);
    var w = 0.62 * h;
    var v = R3.view(g, { robot: 'g1', origin: [x, y + sole * S * Math.cos(elv * Math.PI / 180)], scale: S, az: az, el: elv,
      box: [x - w / 2, y - h - 12, w, h + 22], res: o.res || 2 });
    var D = Math.PI / 180;
    v.joints({ left_shoulder_roll_joint: 8 * D, right_shoulder_roll_joint: -8 * D, left_elbow_joint: 1.45, right_elbow_joint: 1.45 });
    var st = { yaw: 0, s: 1 };
    function M(p) {   // Rz(yaw)·diag(s, s, 1)·p
      var c = Math.cos(st.yaw), sn = Math.sin(st.yaw), a = st.s * p[0], b = st.s * p[1];
      return [c * a - sn * b, sn * a + c * b, p[2]];
    }
    var api = {
      g: g, view: v, scale: S,
      set: function (yaw, s) {
        st.yaw = yaw; st.s = Math.max(0.03, s === undefined ? 1 : s);
        v.base([0, 0, 0], [0, 0, yaw]);
        v.robot.scale.set(st.s, st.s, 1);
        return api;
      },
      frame: function (L, lift) {
        var z0 = lift || 0, o0 = v.project([0, 0, z0]);
        return { o: o0, tips: [[L, 0, 0], [0, L, 0], [0, 0, L]].map(function (p) { var q = M(p); return v.project([q[0], q[1], q[2] + z0]); }) };
      },
      ring: function (r, n) {
        var pts = [];
        for (var k = 0; k <= (n || 48); k++) { var a = 2 * Math.PI * k / (n || 48); pts.push(v.project([r * Math.cos(a), r * Math.sin(a), sole])); }
        return pts;
      },
      ground: function (p) { return v.project([p[0], p[1], sole]); },
      tint: function (part, color, amount, opacity) { v.tint(part, color, amount, opacity); }
    };
    api.set(0, 1);
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
})();
