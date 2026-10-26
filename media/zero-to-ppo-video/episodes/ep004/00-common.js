// 零基础篇 004：用 Python / NumPy 写第一段矩阵代码（路线 §L0 前置知识「会写 Python」、学完输出「能用 NumPy 写简单矩阵运算」）
// 本目录按文件名顺序加载：00 共用部件 → 10 片头主图 → 21…25 五幕分镜 → 90 组装 window.EpisodeVideo。旁白在 ../ep004.py。
// 工具函数取自 ../ep003/00-common.js（命名空间换成 EP004），另加代码窗口 E.code（逐字打出、光标、滚动、按语义上色）。
// 分镜都画在 800×420 的网格上。draw(st, t)：st 是分镜时间（旁白念完前画面停在 to 之前，一次性事件用它），
// t 是这一幕的真实时间（光标闪烁这类循环动作用它，停帧时也不会卡住）。
(function () {
  var K = window.PaperDemoKit;
  var C = K.xColors;
  var E = window.EP004 = { K: K, scenes: [] };

  /* 颜色同 003：位置 / 结果向量 accent（蓝）、矩阵 / 变换 violet；矩阵的两列——第一列 good（绿）、第二列 warn（琥珀）；错误 bad；
     本系列亮黄 --hi 只做强调。代码窗口里只给有含义的数上这几种颜色，其余代码一律是正文色，提示符灰色。
     KaTeX 的 \textcolor 不认 CSS 变量，公式里用 E.HEX 的同一组色值（与 stage.html 的 :root 一致） */
  E.ACC = C.accent; E.GOOD = C.good; E.WARN = C.warn; E.BAD = C.bad; E.VIOLET = 'var(--violet)'; E.HI = 'var(--hi)';
  E.MUT = C.muted; E.INK = C.ink; E.INK2 = C.ink2; E.S1 = C.surface; E.S2 = C.surface2; E.BORDER = C.border; E.GRID = C.grid;
  E.HEX = { acc: '#569cd6', good: '#4ec9a4', warn: '#d7a75f', bad: '#e06c75', violet: '#c586c0', hi: '#ffd166', ink: '#e6e6e6', ink2: '#a8a8a8', mut: '#8a8a8a' };

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
        var id = 'ep004-draw-mask-' + (++maskN);
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
    K.paint(p, null, color || E.BAD);
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
    var g = el('g', {}, parent);
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

  /* ── 机器人：UR5（Robot Description Gallery Online 收录的 URDF，由 ../../robot3d.js 渲染，接口同 ep003）：
     底座底面中心在 (x, y)；opts.az / opts.el 换视角，网格上用 az 0、el 0 的正侧视，胳膊就在画面这个平面里转 ── */
  E.arm = function (parent, x, y, s, opts) { return window.Robot3D.ur5(parent, x, y, s, opts); };

  /* UR5 侧视、肩关节放在 (ox, oy)：把整台挪到肩关节落在这一点；pose(a1, a2) 同 ur5.set，turn(rad) 只转肩关节（肩膀以外整条胳膊刚性地转），
     正值在画面上是逆时针（数学正向，和 E.rot 一致）；shoulder() / tool() 是肩关节、手（tool0）在画面里的位置。
     aim(x, y, a1)：大臂离竖直 a1 度，二分小臂角，让「肩 → 手」正好指向数学方向 (x, y)，返回一格多少 px（|肩 → 手| / |(x, y)|），
     拿它当 E.plane 的 u，手就正好落在格点 (x, y) 上。画布外框按 reach（画面 px）往左、右、上放开，往下到底座底面，转到哪都不会被裁掉 */
  E.armAt = function (parent, ox, oy, s, reach) {
    var r = E.arm(parent, 0, 0, s, { az: 0, el: 0 }), v = r.view, base = { lift: 0, elbow: 0, w1: 0 };
    var sh = v.at('upper_arm_link');
    var x = ox - sh[0], y = oy - sh[1];
    v.camera({ origin: [x, y], box: [ox - reach, oy - reach, 2 * reach, reach + (y - oy) + 6] });
    function snap() { base.lift = v.jointValue('shoulder_lift_joint'); base.elbow = v.jointValue('elbow_joint'); base.w1 = v.jointValue('wrist_1_joint'); }
    snap();
    return {
      g: r.g, view: v, origin: [x, y],
      pose: function (a1, a2) { r.set(a1, a2); snap(); return v.at('tool0'); },
      turn: function (a) { v.joints({ shoulder_lift_joint: base.lift - a, elbow_joint: base.elbow, wrist_1_joint: base.w1 }); return v.at('tool0'); },
      aim: function (x, y, a1) {
        var want = Math.atan2(y, x), lo = 0, hi = 170, sp = v.at('upper_arm_link'), tp;
        for (var i = 0; i < 40; i++) {
          var m = (lo + hi) / 2;
          r.set(a1, m); tp = v.at('tool0');
          if (Math.atan2(sp[1] - tp[1], tp[0] - sp[0]) > want) lo = m; else hi = m;   // 小臂越弯，手越往下
        }
        snap(); tp = v.at('tool0');
        return Math.hypot(tp[0] - sp[0], tp[1] - sp[1]) / Math.hypot(x, y);
      },
      shoulder: function () { return v.at('upper_arm_link'); },
      tool: function () { return v.at('tool0'); }
    };
  };

  /* ── 代码窗口：E.code(parent, x, y, w, h, o) ──
     o = { title：标题栏文字（「Python」「hand.py」「终端」），size：代码字号（默认 24），lh：行距（默认 size × 1.42），pad：左边距（默认 16），
           cursor：false 不画光标（编辑器里整段贴进来时用） }
     api.line(s, at, opts) 加一句，返回它的下标：
       s 是代码文本，`{g:0}` 把「0」涂成绿（g / w / a / v / b / h / m / i = good / warn / acc / violet / bad / hi / muted / ink2，n 不上色），
       `{g.id:0}` 再给这一段起个名字，api.seg(id) 拿到它的 tspan 与列号（分镜里改颜色、量位置）；开头的「>>> 」「... 」「$ 」涂灰、不用打。
       opts = { out：输出（整句一下子出来，默认颜色 ink2），dur：打字用多久（默认每秒 16 个字），keys：每个字各自出现的分镜时刻（和旁白逐词对上，
                可以不按从左到右的顺序：没出现的字先空着，等宽字体下别的字不挪位置），page：第几页（新一页开始时上一页淡出，从第一排重新排），
                row：排在第几排（默认接着上一句），alpha：整句的透明度系数 }
     超出窗口能放的排数时整体往上滚一排，滚出去的句子淡掉；光标跟着正在打的那一句，打完在句尾闪，下一句开始或出了输出就收掉。
     api.draw(st, t) 之后可以调 api.hl(i, color, p) 给第 i 句垫一条底色、api.box(id, color, p) 给一段框一圈、
     api.boxCol(i, col, len, color, p) 按列号框住第 i 句的一段（跨好几段颜色时用）；
     api.x(col) 是第 col 个字符左缘的 x，api.y(i) 是第 i 句此刻的基线 y（要先 draw），api.cw 是一个字符宽 */
  E.MONO = "'JetBrains Mono', monospace";
  var CODE_COL = { g: E.GOOD, w: E.WARN, a: E.ACC, v: E.VIOLET, b: E.BAD, h: E.HI, m: E.MUT, i: E.INK2, n: null };
  function parseCode(s) {
    var segs = [], at = 0, m, re = /\{([a-z])(?:\.(\w+))?:([^}]*)\}/g;
    var pr = s.match(/^(>>> |\.\.\. |\$ )/);
    if (pr) { segs.push({ t: pr[1], c: E.MUT, prompt: true }); at = pr[1].length; }
    re.lastIndex = at;
    while ((m = re.exec(s)) !== null) {
      if (m.index > at) segs.push({ t: s.slice(at, m.index) });
      segs.push({ t: m[3], c: CODE_COL[m[1]], id: m[2] });
      at = m.index + m[0].length;
    }
    if (at < s.length) segs.push({ t: s.slice(at) });
    return segs;
  }
  E.code = function (parent, x, y, w, h, o) {
    o = o || {};
    var size = o.size || 24, lh = o.lh || Math.round(size * 1.42), pad = o.pad || 16, cw = size * 0.6, BAR = 34;
    var g = el('g', {}, parent);
    K.paint(el('rect', { x: x, y: y, width: w, height: h, rx: 14, 'stroke-width': 2 }, g), '#121216', E.BORDER);
    K.paint(el('path', { d: 'M ' + (x + 1) + ' ' + (y + BAR) + ' V ' + (y + 14) + ' A 13 13 0 0 1 ' + (x + 14) + ' ' + (y + 1) +
      ' H ' + (x + w - 14) + ' A 13 13 0 0 1 ' + (x + w - 1) + ' ' + (y + 14) + ' V ' + (y + BAR) + ' Z' }, g), E.S2);
    [0, 1, 2].forEach(function (k) { K.paint(el('circle', { cx: x + 20 + 16 * k, cy: y + BAR / 2, r: 5 }, g), '#4a4a52'); });
    var title = txt(g, x + w / 2, y + BAR / 2 + 6.5, o.title || '', 18, E.INK2, 'middle', 700);
    var hlG = el('g', {}, g), body = el('g', {}, g);
    var cursor = K.paint(el('rect', { width: cw, height: Math.round(size * 1.12), rx: 2 }, g), E.INK);
    var top = y + BAR + 10, cap = Math.floor((h - BAR - 16) / lh);
    var lines = [], segs = {}, boxes = {};
    var api = {
      g: g, title: title, cw: cw, size: size, lh: lh, cap: cap, lines: lines,
      x: function (col) { return x + pad + cw * col; },
      y: function (i) { return lines[i].y; },
      line: function (s, at, op2) {
        var p = op2 || {}, sg = parseCode(s), i = lines.length;
        var prev = lines.filter(function (L) { return (L.page || 0) === (p.page || 0); }).pop();
        var n = el('text', { x: x + pad, 'font-size': size, 'xml:space': 'preserve' }, body);
        n.style.fontFamily = E.MONO; n.style.whiteSpace = 'pre';
        var col = 0, chars = [];
        sg.forEach(function (q) {
          q.el = el('tspan', {}, n);
          q.el.style.fill = q.c || (p.out ? E.INK2 : E.INK);
          q.col = col; col += q.t.length;
          if (q.id) segs[q.id] = { el: q.el, line: i, col: q.col, len: q.t.length };
          q.c0 = chars.length;
          if (!q.prompt) for (var k = 0; k < q.t.length; k++) chars.push({ q: q, k: k });
        });
        var L = { segs: sg, chars: chars, at: at, out: !!p.out, page: p.page || 0, row: p.row !== undefined ? p.row : prev ? prev.row + 1 : 0,
          keys: p.keys, dur: p.dur !== undefined ? p.dur : chars.length / 16, alpha: p.alpha === undefined ? 1 : p.alpha, el: n, shown: null, y: 0,
          hl: K.paint(el('rect', { x: x + 6, width: w - 12, height: lh - 4, rx: 6 }, hlG), 'none') };
        L.hl.style.opacity = 0;
        lines.push(L);
        return i;
      },
      seg: function (id) { return segs[id]; },
      draw: function (st, t) {
        // 当前页：开头时刻不晚于 st 的最后一页；上一页在新一页开始前 0.35 s 淡出
        var starts = {};
        lines.forEach(function (L) { if (starts[L.page] === undefined || L.at < starts[L.page]) starts[L.page] = L.at; });
        var cur = 0;
        Object.keys(starts).forEach(function (pg) { if (starts[pg] <= st + 1e-6 && +pg > cur) cur = +pg; });
        var off = 0;
        lines.forEach(function (L) { if (L.page === cur && L.row >= cap) off += fade(st, L.at - 0.25, 0.25); });
        var curLine = null;
        lines.forEach(function (L, i) {
          var a;
          if (L.page === cur) a = st >= L.at - 1e-6 ? (L.out ? fade(st, L.at, 0.15) : 1) : 0;
          else if (L.page === cur - 1) a = 1 - fade(st, starts[cur] - 0.35, 0.3);
          else a = 0;
          var r = L.row - (L.page === cur ? off : 0);
          a *= clamp(r * 2 + 1, 0, 1) * L.alpha;
          L.y = top + r * lh + size * 0.92;
          L.el.setAttribute('y', L.y.toFixed(1));
          L.hl.setAttribute('y', (L.y - size * 0.92 - (lh - size * 1.2) / 2 + 2).toFixed(1));
          L.hl.style.opacity = 0;
          op(L.el, a.toFixed(3));
          // 打到第几个字：keys 给了就按各字的时刻，没给就按 dur 匀速
          var vis = L.chars.map(function (c, k) {
            if (L.out) return true;
            if (L.keys) return st >= L.keys[Math.min(k, L.keys.length - 1)] - 1e-6;
            return st >= L.at + (L.dur * (k + 0.5)) / Math.max(L.chars.length, 1) - 1e-6 || L.dur === 0;
          });
          var key = vis.join('');
          if (key !== L.shown) {
            L.shown = key;
            L.segs.forEach(function (q) {
              if (q.prompt) { q.el.textContent = q.t; return; }
              var s2 = '';
              for (var k = 0; k < q.t.length; k++) s2 += vis[q.c0 + k] ? q.t[k] : ' ';
              q.el.textContent = s2;
            });
          }
          L.last = -1;
          vis.forEach(function (v, k) { if (v) L.last = k; });
          L.done = vis.every(Boolean);
          if (L.page === cur && !L.out && st >= L.at - 1e-6 && a > 0.5) curLine = L;
        });
        // 光标：最后一句开始打的输入句；它后面已经出了下一句（输入或输出）就收掉
        var showC = false;
        if (curLine) {
          var later = lines.some(function (L) { return L.page === cur && L.at > curLine.at && st >= L.at - 1e-6; });
          if (!later) {
            var pl = curLine.segs[0] && curLine.segs[0].prompt ? curLine.segs[0].t.length : 0;   // 空行没有段
            var c = curLine.chars[curLine.last];
            var col = curLine.last < 0 ? pl : c.q.col + c.k + 1;
            cursor.setAttribute('x', (x + pad + cw * col + 1).toFixed(1));
            cursor.setAttribute('y', (curLine.y - size * 0.86).toFixed(1));
            showC = !curLine.done || (t % 1.0) < 0.6;
          }
        }
        op(cursor, showC && o.cursor !== false ? 0.85 : 0);
      },
      hl: function (i, color, p) {
        var L = lines[i];
        K.paint(L.hl, tint(color, 22, '#121216'));
        L.hl.style.opacity = clamp(p, 0, 1) * +L.el.style.opacity;
      },
      box: function (id, color, p) {
        var sgm = segs[id];
        return api.boxCol(sgm.line, sgm.col, sgm.len, color, p, 'seg:' + id);
      },
      boxCol: function (i, col, len, color, p, key) {
        var L = lines[i], k = key || i + ':' + col + ':' + len;
        var bx = boxes[k] = boxes[k] || el('rect', { rx: 6, fill: 'none', 'stroke-width': 2.5 }, hlG);
        bx.style.stroke = color;
        bx.setAttribute('x', (x + pad + cw * col - 3).toFixed(1));
        bx.setAttribute('y', (L.y - size * 0.95).toFixed(1));
        bx.setAttribute('width', (cw * len + 6).toFixed(1));
        bx.setAttribute('height', (size * 1.3).toFixed(1));
        bx.style.opacity = clamp(p, 0, 1) * +L.el.style.opacity;
        return bx;
      }
    };
    return api;
  };

  /* ── 手画的 2×2 矩阵（不用 KaTeX）：每个数的位置都知道，好逐排 / 逐列高亮、让数飞出去。
     E.mat(parent, cx, cy, m, o)：m = [a, b, c, d] 即 [[a, b], [c, d]]；o = { size：字号（默认 40），dx / dy：列距 / 排距，colors：四个数各自的颜色 }
     返回 { g, cells[k]（文字）, pos(k) → [x, y]（第 k 个数的中心）, rowBand(r, color, p), colBox(c, color, p), brackets（两个括号 path） } ── */
  E.mat = function (parent, cx, cy, m, o) {
    o = o || {};
    var size = o.size || 40, dx = o.dx || size * 1.7, dy = o.dy || size * 1.35;
    var g = el('g', {}, parent), under = el('g', {}, g);
    var bh = dy + size * 1.15, bx = dx / 2 + size * 0.95;
    var brackets = [-1, 1].map(function (sd) {
      var xb = cx + sd * bx, lip = size * 0.28;
      var b = el('path', { d: 'M ' + (xb - sd * lip) + ' ' + (cy - bh / 2) + ' H ' + xb + ' V ' + (cy + bh / 2) + ' H ' + (xb - sd * lip),
        fill: 'none', 'stroke-width': Math.max(2, size / 16), 'stroke-linecap': 'square' }, g);
      K.paint(b, null, o.bracket || E.INK);
      return b;
    });
    function pos(k) { return [cx + (k % 2 ? 1 : -1) * dx / 2, cy + (k < 2 ? -1 : 1) * dy / 2]; }
    var cells = m.map(function (v, k) {
      var p = pos(k), n = txt(g, p[0], p[1] + size * 0.35, String(v).replace('-', '−'), size, (o.colors || [])[k] || E.INK, 'middle');
      n.style.fontFamily = 'KaTeX_Main, serif';
      return n;
    });
    var bands = [0, 1].map(function () { return el('rect', { rx: 8 }, under); });
    var cols = [0, 1].map(function () { return el('rect', { rx: 10, fill: 'none', 'stroke-width': 3 }, under); });
    bands.concat(cols).forEach(function (r) { op(r, 0); });
    return {
      g: g, cells: cells, pos: pos, brackets: brackets, size: size, dx: dx, dy: dy,
      box: [cx - bx, cy - bh / 2, cx + bx, cy + bh / 2],
      rowBand: function (r, color, p) {
        var b = bands[r], y0 = cy + (r ? 0 : -dy) - size * 0.05;
        b.setAttribute('x', cx - bx + 8); b.setAttribute('width', 2 * bx - 16);
        b.setAttribute('y', (y0 + 4).toFixed(1)); b.setAttribute('height', (dy - 8).toFixed(1));
        K.paint(b, tint(color, 26, 'var(--demo-surface)')); op(b, clamp(p, 0, 1));
      },
      colBox: function (c, color, p) {
        var b = cols[c], xc = cx + (c ? 1 : -1) * dx / 2;
        b.setAttribute('x', (xc - size * 0.72).toFixed(1)); b.setAttribute('width', (size * 1.44).toFixed(1));
        b.setAttribute('y', (cy - dy / 2 - size * 0.72).toFixed(1)); b.setAttribute('height', (dy + size * 1.44).toFixed(1));
        b.style.stroke = color; op(b, clamp(p, 0, 1));
      }
    };
  };
})();
