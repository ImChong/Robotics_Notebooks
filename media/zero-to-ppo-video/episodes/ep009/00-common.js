// 零基础篇 009：运动旋量与螺旋轴——用一个六维量描述刚体速度（路线 §L1.1「Twist / Screw Axis」，Modern Robotics 3.3.2）
// 本目录按文件名顺序加载：00 共用部件 → 10 片头主图 → 21…25 五幕分镜 → 90 组装 window.EpisodeVideo。旁白在 ../ep009.py。
// 工具函数取自 ../ep005/00-common.js（命名空间换成 EP009；坐标网格、可动箭头、KaTeX、三维投影 E.view3、三维小坐标架 E.gizmo），
// UR5 侧视放到网格上取自 ../ep003/00-common.js 的 E.armAt，另加：小臂绕肘刚性转动（bend）、小臂上一点的位置、六维列向量、螺旋线。
// 分镜都画在 800×420 的网格上。draw(st, t)：st 是分镜时间（旁白念完前画面停在 to 之前，一次性事件用它），
// t 是这一幕的真实时间（呼吸、绕圈这类循环动作用它，停帧时也不会卡住）。
(function () {
  var K = window.PaperDemoKit;
  var C = K.xColors;
  var E = window.EP009 = { K: K, scenes: [] };

  /* 颜色：角速度 ω / 转动 violet（同 003、005 的变换色）；点的速度、线速度 v accent（蓝，同 003 的向量）；
     转轴 / 肘（转动中心）亮黄 --hi；螺旋轴 warn（琥珀）；螺距、沿轴移动 good（绿）；说错的、不合格 bad。
     三维坐标轴沿用 005：x 绿、y 琥珀、z 蓝。KaTeX 的 \textcolor 不认 CSS 变量，公式里用 E.HEX 的同一组色值（与 stage.html 的 :root 一致） */
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
  // 一下鼓起：a 起 d 秒内 0 → 1 → 0
  function bump(t, a, d) { var u = K.seg(t, a, a + (d || 0.6)); return u > 0 && u < 1 ? Math.sin(Math.PI * u) : 0; }
  // 线条「画出来」：p 0 → 1。实线用 dashoffset；虚线（有 stroke-dasharray）套一个同形状的遮罩来画，保留虚线节奏。
  // 要挂到页面上才量得到长度，所以第一次真正生效是在 draw 里（build 时调用只按 p 显隐）。形状会变的 path 不要用它。
  var maskN = 0;
  function drawOn(path, p) {
    if (!path.isConnected) { op(path, p > 0 ? 1 : 0); return; }
    if (!path.__len) {
      path.__len = path.getTotalLength();
      if (path.getAttribute('stroke-dasharray')) {
        var svg = path.ownerSVGElement, defs = svg.querySelector('defs') || el('defs', {}, svg);
        var id = 'ep009-draw-mask-' + (++maskN);
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
  // 压在网格线 / 机械臂上的字描一圈底色，线从字后面过去也看得清
  function halo(node, w) { node.style.paintOrder = 'stroke'; node.style.stroke = 'var(--demo-surface)'; node.style.strokeWidth = (w || 6) + 'px'; node.style.strokeLinejoin = 'round'; return node; }
  // 折线 / 多边形的 d
  function poly(pts, close) { return pts.map(function (p, k) { return (k ? 'L ' : 'M ') + p[0].toFixed(1) + ' ' + p[1].toFixed(1); }).join(' ') + (close ? ' Z' : ''); }

  E.el = el; E.txt = txt; E.op = op; E.clamp = clamp; E.fade = fade; E.win = window_; E.pop = pop; E.scaleAt = scaleAt; E.bump = bump;
  E.drawOn = drawOn; E.check = check; E.cross = cross; E.tag = tag; E.tint = tint; E.glow = glow; E.halo = halo; E.poly = poly;

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

  /* ── 圆弧（画面坐标，圆心 (cx, cy)、半径 r，从 a0 到 a1，数学角度：逆时针为正、画面 y 朝下） + 末端三角：转动方向的小弯箭头。
     set(a0, a1) 每帧可改；返回 { g, arc, head, set } ── */
  E.arcArrow = function (parent, cx, cy, r, color, w, opts) {
    var o = opts || {};
    var g = el('g', {}, parent);
    var arc = el('path', { fill: 'none', 'stroke-width': w || 3, 'stroke-linecap': 'round' }, g);
    var head = el('path', {}, g);
    K.paint(arc, null, color); K.paint(head, color);
    if (o.dash) arc.setAttribute('stroke-dasharray', o.dash);
    var api = {
      g: g, arc: arc, head: head,
      set: function (a0, a1) {
        var hl = o.head || 13, hw = hl * 0.62, sgn = a1 >= a0 ? 1 : -1;
        var back = sgn * (hl * 0.8) / r;                         // 弧收在三角里面
        var e0 = a1 - back;
        function P(a) { return [cx + r * Math.cos(a), cy - r * Math.sin(a)]; }
        var p0 = P(a0), p1 = P(e0), large = Math.abs(e0 - a0) > Math.PI ? 1 : 0, sweep = sgn > 0 ? 0 : 1;
        arc.setAttribute('d', Math.abs(e0 - a0) < 1e-3 ? '' : 'M ' + p0[0].toFixed(2) + ' ' + p0[1].toFixed(2) + ' A ' + r + ' ' + r + ' 0 ' + large + ' ' + sweep + ' ' + p1[0].toFixed(2) + ' ' + p1[1].toFixed(2));
        var tip = P(a1), tx = -Math.sin(a1) * sgn, ty = -Math.cos(a1) * sgn;  // 画面里的切向（沿转动方向）
        var b = [tip[0] - tx * hl, tip[1] - ty * hl];
        head.setAttribute('d', 'M ' + tip[0].toFixed(2) + ' ' + tip[1].toFixed(2) + ' L ' + (b[0] - ty * hw).toFixed(2) + ' ' + (b[1] + tx * hw).toFixed(2) +
          ' L ' + (b[0] + ty * hw).toFixed(2) + ' ' + (b[1] - tx * hw).toFixed(2) + ' Z');
        return api;
      }
    };
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

  /* ── 坐标网格（数学坐标：x 往右、y 往上），同 003 / 005：
     o = { ox, oy：原点在画面里的位置；u：一格多少 px；x0, x1, y0, y1：画出的整数范围（含两端）；axes（默认 true）；labels（默认 true）；ticks（默认 false）；color }
     返回 { g, P(x, y) → [X, Y], box, u } ── */
  E.plane = function (parent, o) {
    var u = o.u, ox = o.ox, oy = o.oy;
    function P(x, y) { return [ox + u * x, oy - u * y]; }
    var g = el('g', { 'data-grid': '1' }, parent);   // data-grid：画在格子上的字本来就描了底色，查「字压线」时不算格线
    var box = [ox + u * o.x0, oy - u * o.y1, ox + u * o.x1, oy - u * o.y0];
    var grid = el('g', {}, g), x, y;
    for (x = o.x0; x <= o.x1; x++) K.paint(el('line', { x1: P(x, 0)[0], y1: box[1], x2: P(x, 0)[0], y2: box[3], 'stroke-width': 1.5 }, grid), null, o.color || E.GRID);
    for (y = o.y0; y <= o.y1; y++) K.paint(el('line', { x1: box[0], y1: P(0, y)[1], x2: box[2], y2: P(0, y)[1], 'stroke-width': 1.5 }, grid), null, o.color || E.GRID);
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
    return { g: g, grid: grid, axes: axes, P: P, box: box, u: u };
  };

  /* ── 公式：K.svgMath 包一层，(x, y) 同 txt 的基线；size 默认 28；color 是 CSS 颜色（整条公式的默认色） ── */
  E.tex = function (parent, x, y, tex, size, color, anchor, opts) {
    var o = opts || {};
    var fo = K.svgMath(x, y, tex, { size: size || 28, anchor: anchor || 'middle', w: o.w, h: o.h });
    if (color) fo.setTone(color);
    parent.appendChild(fo);
    return fo;
  };
  // 上色的一个数：\textcolor{#hex}{x}
  function cnum(x, hex) { var s = typeof x === 'number' ? E.num(x).replace('−', '-') : String(x); return hex ? '\\textcolor{' + hex + '}{' + s + '}' : s; }
  E.cnum = cnum;
  // 列向量：col([3, 2]) → \begin{bmatrix}3\\2\end{bmatrix}；hex 可以是一个颜色或每个数各一个
  E.col = function (xs, hex) {
    return '\\begin{bmatrix}' + xs.map(function (x, i) { return cnum(x, Array.isArray(hex) ? hex[i] : hex); }).join('\\\\') + '\\end{bmatrix}';
  };

  /* ── 数：小数写成「0.8」，负号用真正的减号「−」 ── */
  E.num = function (x, digits) {
    var s = typeof x === 'string' ? x : (Math.abs(x) < 5e-3 ? '0' : x.toFixed(digits === undefined ? 2 : digits).replace(/\.?0+$/, ''));
    return s.replace('-', '−');
  };
  // 平面里逆时针转 90°：(x, y) → (−y, x)（003 第 3 幕）
  E.rot90 = function (v) { return [-v[1], v[0]]; };

  /* ── 手画的列向量（不用 KaTeX）：每个数的位置都知道，好逐个弹出、框亮。
     E.colvec(parent, cx, cy, xs, o)：xs 为数（或字符串）；o = { size（默认 30）, dy：排距（默认 1.3 字号）, colors：每个数的颜色, w：括号内宽（默认 2.2 字号） }
     返回 { g, cells[k], pos(k) → [x, y]（第 k 个数的中心）, set(xs), box, cellBox(k, color, p), rangeBox(k0, k1, color, p) } ── */
  E.colvec = function (parent, cx, cy, xs, o) {
    o = o || {};
    var size = o.size || 30, dy = o.dy || size * 1.3, n = xs.length, w = o.w || size * 2.2;
    var g = el('g', {}, parent), under = el('g', {}, g);
    var h = (n - 1) * dy + size * 1.25, bx = w / 2;
    [-1, 1].forEach(function (sd) {
      var xb = cx + sd * bx, lip = size * 0.3;
      K.paint(el('path', { d: 'M ' + (xb - sd * lip) + ' ' + (cy - h / 2) + ' H ' + xb + ' V ' + (cy + h / 2) + ' H ' + (xb - sd * lip),
        fill: 'none', 'stroke-width': Math.max(2, size / 14), 'stroke-linecap': 'square' }, g), null, o.bracket || E.INK);
    });
    function pos(k) { return [cx, cy + (k - (n - 1) / 2) * dy]; }
    var boxes = xs.map(function () { var r = el('rect', { rx: 8, 'stroke-width': 2.5 }, under); op(r, 0); return r; });
    var ranges = [];
    var cells = xs.map(function (v, k) {
      var p = pos(k), c = txt(g, p[0], p[1] + size * 0.35, E.num(v), size, (o.colors || [])[k] || E.INK, 'middle');
      c.style.fontFamily = 'KaTeX_Main, serif';
      return c;
    });
    return {
      g: g, cells: cells, pos: pos, size: size, dy: dy,
      box: [cx - bx, cy - h / 2, cx + bx, cy + h / 2],
      set: function (ys) { ys.forEach(function (v, k) { cells[k].textContent = E.num(v); }); },
      cellBox: function (k, color, p) {
        var b = boxes[k], q = pos(k), bw = w - 10, bh = Math.min(size * 1.1, dy - 4);
        b.setAttribute('x', (q[0] - bw / 2).toFixed(1)); b.setAttribute('width', bw.toFixed(1));
        b.setAttribute('y', (q[1] - bh / 2).toFixed(1)); b.setAttribute('height', bh.toFixed(1));
        K.paint(b, tint(color, 20, 'var(--demo-surface)'), color); op(b, clamp(p, 0, 1));
      },
      // k0…k1 几个数框成一组（ω 三个、v 三个）
      rangeBox: function (k0, k1, color, p) {
        var key = k0 + '-' + k1, r = ranges[key];
        if (!r) { r = ranges[key] = el('rect', { rx: 10, 'stroke-width': 2.5 }, under); }
        var a = pos(k0), b = pos(k1), bw = w - 6;
        r.setAttribute('x', (cx - bw / 2).toFixed(1)); r.setAttribute('width', bw.toFixed(1));
        r.setAttribute('y', (a[1] - size * 0.62).toFixed(1)); r.setAttribute('height', (b[1] - a[1] + size * 1.24).toFixed(1));
        K.paint(r, tint(color, 14, 'var(--demo-surface)'), color); op(r, clamp(p, 0, 1));
      }
    };
  };

  /* ── 三维：数学坐标 x、y、z（z 朝上），斜着看。E.view3(az, el) 给出投影：P(v) → [右, 上]（单位长度的画面分量）；
     az / el 的定义同 ../../robot3d.js（az 0 = 从 −y 方向看，el > 0 = 从上往下看） ── */
  E.view3 = function (az, elv) {
    var a = az * Math.PI / 180, e = elv * Math.PI / 180;
    var r = [Math.cos(a), Math.sin(a), 0], u = [-Math.sin(a) * Math.sin(e), Math.cos(a) * Math.sin(e), Math.cos(e)];
    var d = [-Math.sin(a) * Math.cos(e), Math.cos(a) * Math.cos(e), -Math.sin(e)];   // 视线方向（朝里）：深度 = v·d，越小越靠近镜头
    var f = function (v) { return [v[0] * r[0] + v[1] * r[1] + v[2] * r[2], v[0] * u[0] + v[1] * u[1] + v[2] * u[2]]; };
    f.depth = function (v) { return v[0] * d[0] + v[1] * d[1] + v[2] * d[2]; };
    return f;
  };
  E.rot3 = function (n, a) {
    var L = Math.hypot(n[0], n[1], n[2]), x = n[0] / L, y = n[1] / L, z = n[2] / L, c = Math.cos(a), s = Math.sin(a), Cc = 1 - c;
    return [[c + x * x * Cc, x * y * Cc - z * s, x * z * Cc + y * s], [y * x * Cc + z * s, c + y * y * Cc, y * z * Cc - x * s], [z * x * Cc - y * s, z * y * Cc + x * s, c + z * z * Cc]];
  };
  E.mv3 = function (M, v) { return [0, 1, 2].map(function (i) { return M[i][0] * v[0] + M[i][1] * v[1] + M[i][2] * v[2]; }); };
  E.col3 = function (M, j) { return [M[0][j], M[1][j], M[2][j]]; };
  E.cross3 = function (a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; };

  /* ── 三维小坐标架（取自 005）：原点 (cx, cy)，单位长 L px；三根小箭头（x 绿、y 琥珀、z 蓝）。set(M)：M 是 3×3（列 = 三根小箭头的方向）；
     opts.box：画三根小箭头围成的小方块（淡虚线） ── */
  E.gizmo = function (parent, cx, cy, L, proj, opts) {
    var o = opts || {};
    var g = el('g', {}, parent), box = el('g', {}, g);
    var edges = [];
    if (o.box) for (var k = 0; k < 12; k++) edges.push(K.paint(el('line', { 'stroke-width': 1.6, 'stroke-dasharray': '5 5' }, box), null, E.MUT));
    var cols = [E.GOOD, E.WARN, E.ACC];
    var arrows = cols.map(function (c) { return E.vec(g, c, o.w || 4.5, { head: 14 }); });
    var labs = ['x', 'y', 'z'].map(function (s, i) { return halo(txt(g, 0, 0, s, 22, cols[i], 'middle', 700)); });
    function P(v) { var q = proj(v); return [cx + L * q[0], cy - L * q[1]]; }
    var api = {
      g: g, box: box, arrows: arrows, labels: labs, P: P,
      set: function (M, off) {
        var c = [E.col3(M, 0), E.col3(M, 1), E.col3(M, 2)], t0 = off || [0, 0, 0];
        function at(a, b, d) { return P([t0[0] + a * c[0][0] + b * c[1][0] + d * c[2][0], t0[1] + a * c[0][1] + b * c[1][1] + d * c[2][1], t0[2] + a * c[0][2] + b * c[1][2] + d * c[2][2]]); }
        if (o.box) {
          var n = 0;
          [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]].forEach(function (p) {
            [0, 1, 2].forEach(function (ax) {
              if (p[ax]) return;
              var q = p.slice(); q[ax] = 1;
              var a = at(p[0], p[1], p[2]), b = at(q[0], q[1], q[2]), ln = edges[n++];
              ln.setAttribute('x1', a[0].toFixed(1)); ln.setAttribute('y1', a[1].toFixed(1)); ln.setAttribute('x2', b[0].toFixed(1)); ln.setAttribute('y2', b[1].toFixed(1));
            });
          });
        }
        var o0 = at(0, 0, 0);
        c.forEach(function (v, i) {
          var tp = at(i === 0 ? 1 : 0, i === 1 ? 1 : 0, i === 2 ? 1 : 0); arrows[i].set(o0[0], o0[1], tp[0], tp[1]);
          var dx = tp[0] - o0[0], dy = tp[1] - o0[1], len = Math.hypot(dx, dy) || 1;
          labs[i].setAttribute('x', (tp[0] + dx / len * 20).toFixed(1)); labs[i].setAttribute('y', (tp[1] + dy / len * 20 + 8).toFixed(1));
        });
        return api;
      }
    };
    api.set([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
    return api;
  };

  // 第 1–3 幕共用的摆法：肩关节（原点）在画面 (OX, OY)，UR5 缩放 S；速度箭头每 1（格每秒）画 KV 格；右栏从 RX 起
  E.ARM = { OX: 110, OY: 330, S: 2.4, KV: 0.42, RX: 445 };

  /* ── UR5 正侧视（az 0、el 0：胳膊就在画面这个平面里转），肩关节放在 (ox, oy)，同 003 的 E.armAt：
     aim(qx, qy, a2)：大臂指向数学方向 (qx, qy)（肘落在格点 (qx, qy) 上），小臂离水平 a2 度（往上为正），返回一格多少 px（|肩 → 肘| / |(qx, qy)|）；
     bend(a)：小臂连同手腕、手刚性地绕肘转 a 弧度（逆时针为正）；elbow() / wrist() / tool()：肘、腕（wrist_1）、手（tool0）在画面里的位置；
     画布外框按 reach（画面 px）往左、右、上放开，往下到底座底面，转到哪都不会被裁掉 ── */
  E.armAt = function (parent, ox, oy, s, reach) {
    var r = window.Robot3D.ur5(parent, 0, 0, s, { az: 0, el: 0 }), v = r.view, base = { elbow: 0 };
    var sh = v.at('upper_arm_link');
    var x = ox - sh[0], y = oy - sh[1];
    v.camera({ origin: [x, y], box: [ox - reach, oy - reach, 2 * reach, reach + (y - oy) + 6] });
    var api = {
      g: r.g, view: v,
      shoulder: function () { return v.at('upper_arm_link'); },
      elbow: function () { return v.at('forearm_link'); },
      wrist: function () { return v.at('wrist_1_link'); },
      tool: function () { return v.at('tool0'); },
      aim: function (qx, qy, a2) {
        // 大臂：shoulder_lift 让「肩 → 肘」指向 (qx, qy)；二分找角度
        var want = Math.atan2(qy, qx), lo = -Math.PI, hi = 0, sp = v.at('upper_arm_link'), ep;
        for (var i = 0; i < 50; i++) {
          var m = (lo + hi) / 2;
          v.joints({ shoulder_lift_joint: m });
          ep = v.at('forearm_link');
          if (Math.atan2(sp[1] - ep[1], ep[0] - sp[0]) > want) lo = m; else hi = m;
        }
        // 小臂：elbow 让「肘 → 腕」离水平 a2 度；手腕保持朝下（同 003 的 ur5.set）
        var lift = v.jointValue('shoulder_lift_joint'), want2 = (a2 || 0) * Math.PI / 180;
        lo = -Math.PI; hi = Math.PI;
        for (i = 0; i < 50; i++) {
          m = (lo + hi) / 2;
          v.joints({ elbow_joint: m, wrist_1_joint: -(lift + m) - Math.PI / 2, wrist_2_joint: -Math.PI / 2, wrist_3_joint: 0 });
          var e = v.at('forearm_link'), w = v.at('wrist_1_link');
          if (Math.atan2(e[1] - w[1], w[0] - e[0]) > want2) lo = m; else hi = m;
        }
        base.elbow = v.jointValue('elbow_joint');
        ep = v.at('forearm_link');
        return Math.hypot(ep[0] - sp[0], ep[1] - sp[1]) / Math.hypot(qx, qy);
      },
      bend: function (a) { v.joints({ elbow_joint: base.elbow - a }); return api; }
    };
    return api;
  };
})();
