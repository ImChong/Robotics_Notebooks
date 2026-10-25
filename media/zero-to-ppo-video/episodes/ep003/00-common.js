// 零基础篇 003：向量与矩阵——位置是向量，变换是矩阵（路线 §L0 核心问题「线性代数在机器人里到底怎么用」）
// 本目录按文件名顺序加载：00 共用部件 → 10 片头主图 → 21…26 六幕分镜 → 90 组装 window.EpisodeVideo。旁白在 ../ep003.py。
// 工具函数取自 ../ep002/00-common.js（命名空间换成 EP003），另加画坐标网格、可动箭头、KaTeX 向量 / 矩阵的部件。
// 分镜都画在 800×420 的网格上。draw(st, t)：st 是分镜时间（旁白念完前画面停在 to 之前，一次性事件用它），
// t 是这一幕的真实时间（呼吸、旋转、绕圈这类循环动作用它，停帧时也不会卡住）。
(function () {
  var K = window.PaperDemoKit;
  var C = K.xColors;
  var E = window.EP003 = { K: K, scenes: [] };

  /* 颜色：位置 / 结果向量 accent（蓝）、矩阵 / 变换 violet；向量的两个分量与矩阵的两列——往右的一格 / 第一列 good（绿）、
     往上的一格 / 第二列 warn（琥珀），乘法里「第一列取几份、第二列取几份」颜色一路跟着走；错误 bad；本系列亮黄 --hi 只做强调。
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
        var id = 'ep003-draw-mask-' + (++maskN);
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

  // 圆角矩形的 path（drawOn 画虚线框要 path 的 d，rect 没有）
  function rrect(x, y, w, h, r) {
    return 'M ' + (x + r) + ' ' + y + ' H ' + (x + w - r) + ' A ' + r + ' ' + r + ' 0 0 1 ' + (x + w) + ' ' + (y + r) + ' V ' + (y + h - r) +
      ' A ' + r + ' ' + r + ' 0 0 1 ' + (x + w - r) + ' ' + (y + h) + ' H ' + (x + r) + ' A ' + r + ' ' + r + ' 0 0 1 ' + x + ' ' + (y + h - r) +
      ' V ' + (y + r) + ' A ' + r + ' ' + r + ' 0 0 1 ' + (x + r) + ' ' + y + ' Z';
  }
  E.rrect = rrect;

  E.el = el; E.txt = txt; E.op = op; E.clamp = clamp; E.fade = fade; E.win = window_; E.pop = pop; E.scaleAt = scaleAt;
  E.drawOn = drawOn; E.arrow = arrow; E.curveArrow = curveArrow; E.check = check; E.cross = cross; E.tag = tag; E.tint = tint; E.glow = glow;

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
           labels（默认 true）：轴端写 x、y；ticks：轴上标 1、2、3…（默认 false）；color：格线颜色 }
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
        txt(axes, ox, box[1] - 24, 'y', 24, E.INK2, 'middle', 700);
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

  /* ── 机器人：都是 Robot Description Gallery Online 收录的 URDF，由 ../../robot3d.js 渲染（接口同 ep002）：
     E.arm  UR5（底座底面中心在 (x, y)；opts.az / opts.el 换视角，网格上用 az 0、el 0 的正侧视，胳膊就在画面这个平面里转）；
     E.front 正面 G1（两脚中间 x = 0、脚底 y = 388、头顶 y = 92）；E.vacuum 扫地机 iRobot Create 2（opts.top 俯视，(x, y) 为圆心，车头朝 +x） ── */
  E.arm = function (parent, x, y, s, opts) { return window.Robot3D.ur5(parent, x, y, s, opts); };
  E.front = function (parent, opts) { return window.Robot3D.g1Front(parent, opts); };
  E.vacuum = function (parent, x, y, s, opts) { return window.Robot3D.create2(parent, x, y, s, opts); };

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
})();
