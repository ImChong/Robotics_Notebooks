// 零基础篇 011：正运动学——用指数积公式从关节角算末端位姿（路线 §L1.2「正逆运动学」、推荐做什么「手写 FKinSpace」，§L1 自测题 3，Modern Robotics 第 4 章）
// 本目录按文件名顺序加载：00 共用部件 → 10 片头主图 → 21…27 七幕分镜 → 90 组装 window.EpisodeVideo。旁白在 ../ep011.py。
// 工具函数取自 ../ep010/00-common.js（命名空间换成 EP011；坐标网格、可动箭头、小弯箭头、KaTeX、手画矩阵 E.matN、一行算式 E.flow、
// 三维投影、UR5 侧视放到网格上 E.armAt），另取 ../ep004/00-common.js 的代码窗口 E.code（逐字打出、光标、滚动、按语义上色），
// 新加 E.arm2：UR5 侧视只动肩、肘两个关节（角度从第 9、10 集的姿势算起），E.fk2 / E.hand2：两关节的位置和手的坐标系。
// 分镜都画在 800×420 的网格上。draw(st, t)：st 是分镜时间（旁白念完前画面停在 to 之前，一次性事件用它），
// t 是这一幕的真实时间（光标闪烁这类循环动作用它，停帧时也不会卡住）。
(function () {
  var K = window.PaperDemoKit;
  var C = K.xColors;
  var E = window.EP011 = { K: K, scenes: [] };

  /* 颜色按发布清单「画面配色（005 起）」：坐标轴 x 红、y 绿、z 蓝（矩阵的第一、二、三列同色，扫地机、方块上的小箭头同色）；
     平面里的转动是绕 z 轴转，小弯箭头、转过的角度 θ 用蓝（E.OMEGA）；错误、不合格、走歪的直线用橙（E.ERR，红色留给 x 轴），
     打勾这类「对」、真正的圆弧用绿。照 009：刚体（小臂、方块）淡黄（E.BODY）；肘（转动中心）亮黄；螺旋轴、ŝ 琥珀；螺距、沿轴移动绿；
     点的位置、速度和位姿里的位置 p 用紫（E.VEL，同 007 的位置 p、009 的 v）。
     KaTeX 的 \textcolor 不认 CSS 变量，公式里用 E.HEX 的同一组色值（与 stage.html 的 :root 一致） */
  E.ACC = C.accent; E.GOOD = C.good; E.WARN = C.warn; E.VIOLET = 'var(--violet)'; E.HI = 'var(--hi)';
  E.MUT = C.muted; E.INK = C.ink; E.INK2 = C.ink2; E.S1 = C.surface; E.S2 = C.surface2; E.BORDER = C.border; E.GRID = C.grid;
  E.HEX = { acc: '#569cd6', good: '#4ec9a4', warn: '#d7a75f', violet: '#c586c0', hi: '#ffd166', ink: '#e6e6e6', ink2: '#a8a8a8', mut: '#8a8a8a',
    x: '#e06c75', y: '#4ec9a4', z: '#569cd6', err: '#ff7b3a' };
  E.X = E.HEX.x; E.Y = E.HEX.y; E.Z = E.HEX.z; E.ERR = E.HEX.err;
  E.OMEGA = E.ACC; E.VEL = E.VIOLET; E.BODY = E.HI;

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
        var id = 'ep011-draw-mask-' + (++maskN);
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
  // 压在网格线 / 机械臂上的字描一圈底色，线从字后面过去也看得清
  function halo(node, w) { node.style.paintOrder = 'stroke'; node.style.stroke = 'var(--demo-surface)'; node.style.strokeWidth = (w || 6) + 'px'; node.style.strokeLinejoin = 'round'; return node; }
  // 折线 / 多边形的 d
  function poly(pts, close) { return pts.map(function (p, k) { return (k ? 'L ' : 'M ') + p[0].toFixed(1) + ' ' + p[1].toFixed(1); }).join(' ') + (close ? ' Z' : ''); }

  /* ── 右栏的一行：几段不同颜色的字挨着往右排。E.rowKit() → { line(g, x, y, parts, size), layout() }；parts = [[字, 颜色, 粗细], …]；
     要挂到页面上才量得到宽度，所以 layout() 在 draw 里第一次 s.isConnected 时调一次（同 009 第 3 幕的 layoutRows） ── */
  E.rowKit = function () {
    var rows = [];
    return {
      line: function (g, x, y, parts, size) {
        var r = { g: el('g', {}, g), x: x };
        r.parts = parts.map(function (pt) { var n = txt(r.g, x, y, pt[0], size || 23, pt[1] || E.INK, 'start', pt[2] || 700); n.style.whiteSpace = 'pre'; return n; });
        rows.push(r);
        return r;
      },
      layout: function () {
        rows.forEach(function (r) {
          var x = r.x;
          r.parts.forEach(function (n) {
            // 开头、末尾的半角空格既不画出来、也不算进量出来的宽度：按字号补上（「 → 」「= 」两边要留空）
            var k = parseFloat(n.style.fontSize || 23) * 0.3, s0 = n.textContent;
            var lead = s0.length - s0.replace(/^ +/, '').length, tail = s0.length - s0.replace(/ +$/, '').length;
            x += lead * k;
            n.setAttribute('x', x.toFixed(1));
            x += n.getComputedTextLength() + 2 + tail * k;
          });
          r.right = x;
        });
      }
    };
  };

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
    var size = o.size || 30, dx = o.dx || size * 1.9, cw = colWidths(M, o, size), w = (M[0].length - 1) * dx, k = size * 0.25 + Math.max(9, size * 0.36);
    return [w / 2 + cw[0] / 2 + k + 1.5, w / 2 + cw[cw.length - 1] / 2 + k + 1.5];   // 左、右（含括号线宽的一半）
  };
  E.matN = function (parent, cx, cy, M, o) {
    o = o || {};
    var size = o.size || 30, dx = o.dx || size * 1.9, dy = o.dy || size * 1.3;
    var rows = M.length, cols = M[0].length, w = (cols - 1) * dx, h = (rows - 1) * dy;
    var g = el('g', { 'data-mgrid': '' }, parent), under = el('g', {}, g), boxes = {};   // data-mgrid / data-bracket / data-hl：给版面检查认
    var colW = colWidths(M, o, size);
    var IN = size * 0.25, GAP = Math.max(9, size * 0.36);   // 高亮框离中括号至少 9 px（同 006、009）
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

  /* ── 三维小坐标架（取自 005）：原点 (cx, cy)，单位长 L px；三根小箭头（x 红、y 绿、z 蓝）。set(M)：M 是 3×3（列 = 三根小箭头的方向）；
     opts.box：画三根小箭头围成的小方块（淡虚线） ── */
  E.gizmo = function (parent, cx, cy, L, proj, opts) {
    var o = opts || {};
    var g = el('g', {}, parent), box = el('g', {}, g);
    var edges = [];
    if (o.box) for (var k = 0; k < 12; k++) edges.push(K.paint(el('line', { 'stroke-width': 1.6, 'stroke-dasharray': '5 5' }, box), null, E.MUT));
    var cols = [E.X, E.Y, E.Z];
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

  // 第 1–4、7 幕共用的摆法：肩关节（原点）在画面 (OX, OY)，UR5 缩放 S（肘落在格点 (1, 2) 上时一格约 40 px）；网格 x −5…4、y −2…5
  // （肩、肘各转 90° 后手在 (−4, 1)，顺序反了手在 (0, −1)）；右栏从 RX 起
  E.ARM = { OX: 228, OY: 296, S: 1.46, RX: 446, X0: -5, X1: 4, Y0: -2, Y1: 5 };
  E.ARM.U = 0.425 * 118 * E.ARM.S / 0.817 / Math.sqrt(5);

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

  /* ── 两个关节的手臂（UR5 侧视，只动肩 shoulder_lift、肘 elbow）：零位是第 9、10 集的姿势（大臂指向 (1, 2)、小臂水平），
     角度从这里算起、逆时针为正。E.arm2(parent, reach) → armAt 的 api 再加 u（一格多少 px）、pose(θ1, θ2)（弧度） ── */
  E.arm2 = function (parent, reach) {
    var A = E.ARM, arm = E.armAt(parent, A.OX, A.OY, A.S, reach || 300), u = arm.aim(1, 2, 0), V = arm.view;
    var lift0 = V.jointValue('shoulder_lift_joint'), elbow0 = V.jointValue('elbow_joint');
    arm.u = u;
    arm.pose = function (t1, t2) { V.joints({ shoulder_lift_joint: lift0 - t1, elbow_joint: elbow0 - (t2 || 0) }); return arm; };
    return arm;
  };
  // 网格坐标里的两关节正运动学（几何）：肘 = 转 θ1 的 (1, 2)；手 = 肘 + 转 θ1 + θ2 的 (2, 0)；手的 x 轴朝 θ1 + θ2
  E.rotP = function (p, a, c) { c = c || [0, 0]; var dx = p[0] - c[0], dy = p[1] - c[1], cs = Math.cos(a), sn = Math.sin(a); return [c[0] + cs * dx - sn * dy, c[1] + sn * dx + cs * dy]; };
  E.fk2 = function (t1, t2) {
    var el_ = E.rotP([1, 2], t1), a = t1 + t2;
    return { elbow: el_, hand: [el_[0] + 2 * Math.cos(a), el_[1] + 2 * Math.sin(a)], ang: a };
  };
  /* ── 手的坐标系：一个点 + x 轴（红）、y 轴（绿）两根小箭头，长 L 格。E.hand2(parent, P, L) → { g, set(x, y, ang), ax, ay, dot } ── */
  E.hand2 = function (parent, P, L, opts) {
    var o = opts || {}, g = el('g', {}, parent);
    var ax = E.vec(g, E.X, o.w || 4.5, { head: 14, dash: o.dash }), ay = E.vec(g, E.Y, o.w || 4.5, { head: 14, dash: o.dash });
    var dot = K.paint(el('circle', { r: 6, 'stroke-width': 3 }, g), o.fill || E.S1, o.ring || E.VEL);
    var api = {
      g: g, ax: ax, ay: ay, dot: dot,
      set: function (x, y, a) {
        var c = P(x, y), px = P(x + L * Math.cos(a), y + L * Math.sin(a)), py = P(x - L * Math.sin(a), y + L * Math.cos(a));
        ax.set(c[0], c[1], px[0], px[1]); ay.set(c[0], c[1], py[0], py[1]);
        dot.setAttribute('cx', c[0].toFixed(1)); dot.setAttribute('cy', c[1].toFixed(1));
        return api;
      }
    };
    return api;
  };


  /* ── 三维的 UR5（斜着看）：六根关节轴（琥珀虚线 + S₁…S₆）和《Modern Robotics》例 4.5 的手坐标系 {b}（同 010 第 6 幕的画法）。
     E.ur6(parent, x, y, s, az, el, box) → { g, view, axG, lines, labs, hand（x 红、y 绿、z 蓝三根小箭头）, update(lab, len), handAt() }
     例 4.5 的 {b} 和 URDF 的 ee_link 差一个固定的转动：x_b = −y_ee、y_b = −z_ee、z_b = x_ee（零位时 M = [[−1, 0, 0], [0, 0, 1], [0, 1, 0]]，
     URDF 的 ee_link 是 [[0, 1, 0], [1, 0, 0], [0, 0, −1]]，见 ../ep011.py）。lab：每根轴标签 [哪一端（0 / 1 / 'mid'）, dx, dy, 对齐]；len：从关节往两边画多长（米） ── */
  E.JN6 = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'];
  E.ur6 = function (parent, x, y, s, az, elv, box) {
    var ur = window.Robot3D.ur5(parent, x, y, s, { az: az, el: elv }), V = ur.view, THREE = window.Robot3D.THREE;
    if (box) V.camera({ box: box });
    var g = el('g', {}, parent), axG = el('g', {}, g);
    var lines = E.JN6.map(function () { return K.paint(el('path', { fill: 'none', 'stroke-width': 3.5, 'stroke-dasharray': '8 5' }, axG), null, E.WARN); });
    var labs = E.JN6.map(function (n, i) { return halo(txt(axG, 0, 0, 'S' + '₁₂₃₄₅₆'[i], 22, E.WARN, 'middle', 900)); });
    var handG = el('g', {}, g);
    var hand = [E.X, E.Y, E.Z].map(function (c) { return E.vec(handG, c, 4, { head: 12 }); });
    function axisOf(name) {
      var j = V.robot.joints[name];
      j.updateMatrixWorld(true);
      return { o: new THREE.Vector3().setFromMatrixPosition(j.matrixWorld), d: j.axis.clone().transformDirection(j.matrixWorld) };
    }
    var api = {
      g: g, view: V, axG: axG, lines: lines, labs: labs, handG: handG, hand: hand, THREE: THREE,
      update: function (lab, len) {
        E.JN6.forEach(function (n, i) {
          var a = axisOf(n), L = len[i];
          var p0 = V.project(a.o.clone().addScaledVector(a.d, L[0])), p1 = V.project(a.o.clone().addScaledVector(a.d, L[1]));
          lines[i].setAttribute('d', 'M ' + p0[0].toFixed(1) + ' ' + p0[1].toFixed(1) + ' L ' + p1[0].toFixed(1) + ' ' + p1[1].toFixed(1));
          var lp = lab[i], at = lp[0] === 'mid' ? [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2] : lp[0] ? p1 : p0;
          labs[i].setAttribute('x', (at[0] + lp[1]).toFixed(1)); labs[i].setAttribute('y', (at[1] + lp[2]).toFixed(1)); labs[i].style.textAnchor = lp[3];
        });
        var h = api.handAt(), o = h.o;
        h.ax.forEach(function (d, i) { var tp = V.project(o.clone().addScaledVector(d, 0.13)), op0 = V.project(o); hand[i].set(op0[0], op0[1], tp[0], tp[1]); });
        return api;
      },
      // 例 4.5 的手坐标系 {b}：原点、三根轴（世界坐标，单位长）
      handAt: function () {
        var tl = V.robot.links.ee_link;
        V.robot.updateMatrixWorld(true);
        var o = new THREE.Vector3().setFromMatrixPosition(tl.matrixWorld);
        var ex = new THREE.Vector3(1, 0, 0).transformDirection(tl.matrixWorld), ey = new THREE.Vector3(0, 1, 0).transformDirection(tl.matrixWorld), ez = new THREE.Vector3(0, 0, 1).transformDirection(tl.matrixWorld);
        return { o: o, ax: [ey.clone().negate(), ez.clone().negate(), ex] };
      }
    };
    V.joints({ shoulder_pan_joint: 0, shoulder_lift_joint: 0, elbow_joint: 0, wrist_1_joint: 0, wrist_2_joint: 0, wrist_3_joint: 0 });
    return api;
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
  var CODE_COL = { g: E.GOOD, w: E.WARN, a: E.ACC, v: E.VIOLET, b: E.ERR, h: E.HI, m: E.MUT, i: E.INK2, n: null, x: E.X, y: E.Y, z: E.Z };   // 011 加 x / y / z：坐标轴的红、绿、蓝
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

})();
