// 零基础篇 008：优化的直觉——变量、目标、约束（路线 §L0 核心问题「优化问题的直觉是什么」）
// 本目录按文件名顺序加载：00 共用部件 → 10 片头主图 → 21…25 五幕分镜 → 90 组装 window.EpisodeVideo。旁白在 ../ep008.py。
// 基础工具（文字、淡入、弹出、画线、箭头、✓ / ×、胶囊标签、描边、可动箭头）取自 ../ep005/00-common.js（命名空间换成 EP008）；
// 另加贯穿全集的最小例子：离终点线 4 格的跑道（侧视 G1 + 脚印 + 「迈 u 格 / 还剩 4 − u 格」）E.track，
// 代价图 E.chart（横轴迈几格 u，竖轴扣分；两项扣分 (4 − u)²、u² 与总代价——一只碗），约束的墙与斜线区 E.wall。
// 分镜都画在 800×420 的网格上。draw(st, t)：st 是分镜时间（旁白念完前画面停在 to 之前，一次性事件用它），
// t 是这一幕的真实时间（呼吸、晃动这类循环动作用它，停帧时也不会卡住）。
(function () {
  var K = window.PaperDemoKit;
  var C = K.xColors;
  var E = window.EP008 = { K: K, scenes: [] };

  /* 颜色：变量 u（迈几格、脚印）good 绿；第一项扣分「还剩几格的平方」accent 蓝；第二项扣分「步子的平方」warn 琥珀；
     代价 / 目标函数（两项相加、那只碗）violet；约束（墙、不能选的斜线区、出界、不能拉）用橙 E.BAD = #ff7b3a（005 起的配色约定：
     错误、不合格用橙，红色留给 x 轴，见 RELEASE_PLAN「画面配色」）；最好的选择（碗底、小球）本系列亮黄 --hi。
     KaTeX 的 \textcolor 不认 CSS 变量，公式里用 E.HEX 的同一组色值（与 stage.html 的 :root 一致） */
  E.ACC = C.accent; E.GOOD = C.good; E.WARN = C.warn; E.VIOLET = 'var(--violet)'; E.HI = 'var(--hi)';
  E.MUT = C.muted; E.INK = C.ink; E.INK2 = C.ink2; E.S1 = C.surface; E.S2 = C.surface2; E.BORDER = C.border; E.GRID = C.grid;
  E.HEX = { acc: '#569cd6', good: '#4ec9a4', warn: '#d7a75f', bad: '#ff7b3a', violet: '#c586c0', hi: '#ffd166', ink: '#e6e6e6', ink2: '#a8a8a8', mut: '#8a8a8a' };
  E.BAD = E.HEX.bad;
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
        var id = 'ep008-draw-mask-' + (++maskN);
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


  /* ── 公式：K.svgMath 包一层，(x, y) 同 txt 的基线；size 默认 28；color 是 CSS 颜色（整条公式的默认色） ── */
  E.tex = function (parent, x, y, tex, size, color, anchor, opts) {
    var o = opts || {};
    var fo = K.svgMath(x, y, tex, { size: size || 28, anchor: anchor || 'middle', w: o.w, h: o.h });
    if (color) fo.setTone(color);
    parent.appendChild(fo);
    return fo;
  };
  E.ctex = function (hex, s) { return '\\textcolor{' + hex + '}{' + s + '}'; };

  /* ── 数：小数写成「1.5」，负号用真正的减号「−」 ── */
  E.num = function (x, digits) {
    var s = Math.abs(x) < 5e-3 ? '0' : x.toFixed(digits === undefined ? 1 : digits).replace(/\.?0+$/, '');
    return s.replace('-', '−');
  };
  E.lerp = function (a, b, p) { return a + (b - a) * p; };
  // 一段时间里鼓一下：a 起 d 秒内 0 → 1 → 0
  E.bump = function (st, a, d) { var u = K.seg(st, a, a + d); return u > 0 && u < 1 ? Math.sin(Math.PI * u) : 0; };
  // 分段取值：keys = [[t0, v0], [t1, v1], …]，每到 ti 用 d 秒缓到 vi（v 是数）
  E.steps = function (st, keys, d) {
    var v = keys[0][1];
    for (var i = 1; i < keys.length; i++) v = v + (keys[i][1] - v) * E.fade(st, keys[i][0], d === undefined ? 0.5 : d);
    return v;
  };

  /* ── 最小例子：离终点线 4 格，这一步迈 u 格。代价 = 还剩格数的平方 + 步子的平方 = (4 − u)² + u²，迈 2 格最小（8 分） ── */
  E.rem = function (u) { return (4 - u) * (4 - u); };
  E.eff = function (u) { return u * u; };
  E.cost = function (u) { return E.rem(u) + E.eff(u); };
  // 顺着下坡挪：每步 u ← u − 0.1 × 坡度（这只碗在 u 处的坡度是 4u − 8）；lim：约束的上限，挪过了就停在边上
  E.descent = function (u0, n, lim) {
    var us = [u0];
    for (var k = 0; k < n; k++) { var u = us[us.length - 1] - 0.1 * (4 * us[us.length - 1] - 8); us.push(lim === undefined ? u : Math.min(u, lim)); }
    return us;
  };

  /* ── 斜线填充（不能选的区域）：每张 svg 一个 pattern，id 全页唯一 ── */
  var hatchN = 0;
  E.hatch = function (svg, color) {
    var defs = svg.querySelector('defs') || el('defs', {}, svg);
    var id = 'ep008-hatch-' + (++hatchN);
    var p = el('pattern', { id: id, patternUnits: 'userSpaceOnUse', width: 12, height: 12, patternTransform: 'rotate(45)' }, defs);
    K.paint(el('rect', { width: 12, height: 12 }, p), 'color-mix(in srgb, ' + (color || E.BAD) + ' 10%, transparent)');
    K.paint(el('line', { x1: 0, y1: 0, x2: 0, y2: 12, 'stroke-width': 3 }, p), null, color || E.BAD);
    p.lastChild.style.strokeOpacity = 0.45;
    return 'url(#' + id + ')';
  };
  /* 约束的墙（橙）：竖线 x = xa（从 y0 到 y1）+ 它右边到 xb 的斜线区。set(xa, xb)；墙顶写字 label（可空） */
  E.wall = function (parent, svg, y0, y1, label) {
    var g = el('g', {}, parent);
    var zone = el('rect', { y: y0, height: y1 - y0 }, g);
    zone.style.fill = E.hatch(svg);
    var line = K.paint(el('line', { y1: y0 - 6, y2: y1, 'stroke-width': 4, 'stroke-linecap': 'round' }, g), null, E.BAD);
    var lab = label ? halo(txt(g, 0, y0 - 14, label, 20, E.BAD, 'middle', 900)) : null;
    var api = {
      g: g, zone: zone, line: line, label: lab,
      set: function (xa, xb) {
        zone.setAttribute('x', xa.toFixed(1)); zone.setAttribute('width', Math.max(0, xb - xa).toFixed(1));
        line.setAttribute('x1', xa.toFixed(1)); line.setAttribute('x2', xa.toFixed(1));
        if (lab) lab.setAttribute('x', xa.toFixed(1));
        return api;
      }
    };
    return api;
  };

  /* ── 跑道：起点（u = 0）在 x0，地面 y，一格 w px，4 格后是终点线。侧视 G1 站在起点左边（脚尖对着起点）。
     o = { x0, y, w, scale：G1 缩放（默认 1），robot：false 就不画机器人, nums：false 不写格号 }
     返回 { g, X(u), robot, finish, nums, foot（落脚点）, step（迈的弧形箭头 + 「迈 u 格」）, rem（还剩的尺寸线 + 「还剩 … 格」）,
            setU(u)：落脚点、弧、尺寸线、两个标签一起跟着 u 走 } ── */
  E.track = function (parent, o) {
    var x0 = o.x0, y = o.y, w = o.w, s = o.scale || 1;
    function X(u) { return x0 + w * u; }
    var g = el('g', {}, parent);
    var floor = el('g', {}, g);
    K.paint(el('line', { x1: x0 - 150 * s, y1: y, x2: X(4) + 46, y2: y, 'stroke-width': 3, 'stroke-linecap': 'round' }, floor), null, E.BORDER);
    var tiles = [0, 1, 2, 3].map(function (k) {
      var r = el('rect', { x: X(k) + 2, y: y + 3, width: w - 4, height: 12, rx: 3, 'stroke-width': 1.5 }, floor);
      K.paint(r, k % 2 ? E.S2 : 'color-mix(in srgb, var(--demo-surface-2) 60%, var(--demo-surface))', E.BORDER);
      return r;
    });
    var nums = o.nums === false ? [] : [0, 1, 2, 3, 4].map(function (k) { return txt(floor, X(k), y + 40, String(k), 20, E.INK2, 'middle', 700); });
    // 终点线：黑白格的竖条 + 顶上「终点线」
    var finish = el('g', {}, g), FH = o.flagH || 72;
    for (var k = 0; k < 6; k++) for (var j = 0; j < 2; j++) {
      K.paint(el('rect', { x: X(4) - 8 + 8 * j, y: y - FH + FH / 6 * k, width: 8, height: FH / 6 }, finish), (k + j) % 2 ? E.INK : 'var(--demo-surface-2)');
    }
    K.paint(el('rect', { x: X(4) - 8, y: y - FH, width: 16, height: FH, fill: 'none', 'stroke-width': 1.5 }, finish), null, E.INK2);
    if (o.flagLab !== false) finish.finishLab = txt(finish, X(4), y - FH - 12, '终点线', 20, E.INK, 'middle', 900);
    var robot = null;
    if (o.robot !== false) {
      robot = window.Robot3D.g1Side(g, 0);
      // 侧视 G1 本地坐标：脚底 y = 388，脚尖约在 x = +28（见 robot3d.js g1Side）；脚尖放到起点
      robot.place({ x: x0 - 30 * s, scale: s, dy: y - 388 });
    }
    // 落脚点：地面上一块绿色的扁椭圆（脚印）
    var foot = el('g', {}, g);
    var footE = K.paint(el('ellipse', { rx: 20, ry: 6, 'stroke-width': 2.5 }, foot), E.tint(E.GOOD, 45, 'var(--demo-surface)'), E.GOOD);
    // 迈的弧形箭头：从起点到落脚点，弧顶高 h
    var step = el('g', {}, g);
    var arc = K.paint(el('path', { fill: 'none', 'stroke-width': 3.5, 'stroke-linecap': 'round' }, step), null, E.GOOD);
    var head = K.paint(el('path', {}, step), E.GOOD);
    var stepLab = halo(txt(step, 0, 0, '', 22, E.GOOD, 'middle', 900));
    // 还剩：落脚点到终点线的尺寸线（两头小竖线）
    var rem = el('g', {}, g);
    var remLine = K.paint(el('path', { fill: 'none', 'stroke-width': 2.5 }, rem), null, E.ACC);
    var remLab = halo(txt(rem, 0, 0, '', 22, E.ACC, 'middle', 900));
    var H = o.arcH || 58, RY = y - (o.remDy || 26);
    var api = {
      g: g, X: X, robot: robot, finish: finish, nums: nums, tiles: tiles, floor: floor,
      foot: foot, footE: footE, step: step, stepLab: stepLab, rem: rem, remLab: remLab, u: 0,
      setU: function (u, stepText, remText) {
        api.u = u;
        var xf = X(u);
        footE.setAttribute('cx', xf.toFixed(1)); footE.setAttribute('cy', (y - 2).toFixed(1));
        // 弧：二次贝塞尔，起点在起点脚尖上方一点，终点落在脚印上
        var xa = x0, ya = y - 8, xb = xf, yb = y - 10, hh = Math.min(H, 18 + 0.5 * Math.abs(xb - xa));
        var cx = (xa + xb) / 2, cy = Math.min(ya, yb) - 2 * hh;
        if (Math.abs(xb - xa) < 4) { arc.setAttribute('d', ''); head.setAttribute('d', ''); }
        else {
          arc.setAttribute('d', 'M ' + xa.toFixed(1) + ' ' + ya.toFixed(1) + ' Q ' + cx.toFixed(1) + ' ' + cy.toFixed(1) + ' ' + xb.toFixed(1) + ' ' + yb.toFixed(1));
          var ang = Math.atan2(yb - cy, xb - cx), L = 13, W = 7.5;
          var p1 = [xb - L * Math.cos(ang) + W * Math.sin(ang), yb - L * Math.sin(ang) - W * Math.cos(ang)];
          var p2 = [xb - L * Math.cos(ang) - W * Math.sin(ang), yb - L * Math.sin(ang) + W * Math.cos(ang)];
          head.setAttribute('d', 'M ' + xb.toFixed(1) + ' ' + yb.toFixed(1) + ' L ' + p1[0].toFixed(1) + ' ' + p1[1].toFixed(1) + ' L ' + p2[0].toFixed(1) + ' ' + p2[1].toFixed(1) + ' Z');
        }
        stepLab.textContent = stepText === undefined ? '迈 ' + E.num(u) + ' 格' : stepText;
        stepLab.setAttribute('x', ((xa + xb) / 2).toFixed(1)); stepLab.setAttribute('y', (Math.min(ya, (ya + cy) / 2) - 12).toFixed(1));
        // 尺寸线：落脚点到终点线
        var r0 = xf, r1 = X(4);
        if (r1 - r0 < 3) remLine.setAttribute('d', '');
        else remLine.setAttribute('d', 'M ' + r0.toFixed(1) + ' ' + (RY - 7) + ' V ' + (RY + 7) + ' M ' + r0.toFixed(1) + ' ' + RY + ' H ' + r1.toFixed(1) +
          ' M ' + r1.toFixed(1) + ' ' + (RY - 7) + ' V ' + (RY + 7));
        var at = 4 - u < 0.05;   // 到线了：字写在终点线右边，不压在终点线上
        remLab.textContent = remText === undefined ? (at ? '到线了' : '还剩 ' + E.num(4 - u) + ' 格') : remText;
        remLab.setAttribute('text-anchor', at ? 'start' : 'middle');
        remLab.setAttribute('x', (at ? r1 + 16 : (r0 + r1) / 2).toFixed(1)); remLab.setAttribute('y', (at ? y - 8 : RY - 12).toFixed(1));
        return api;
      }
    };
    api.setU(0);
    return api;
  };

  /* ── 代价图：横轴迈几格 u（0…4），竖轴扣分（0…16）。o = { ox, oy：原点位置；su：一格 u 多少 px；sc：一分多少 px；
     cMax（默认 16）；ticks：竖轴标的分数（默认 [0, 4, 8, 12, 16]）；xLab / yLab：两根轴的名字 }
     返回 { g, P(u, c), path(fn, a, b), curve(fn, color, w)：一条画好的曲线, dot(color, r)：可挪的圆点 { g, set(u, c) } } ── */
  E.chart = function (parent, o) {
    var ox = o.ox, oy = o.oy, su = o.su, sc = o.sc, cMax = o.cMax || 16;
    function P(u, c) { return [ox + su * u, oy - sc * c]; }
    var g = el('g', {}, parent);
    var grid = el('g', {}, g);
    (o.ticks || [0, 4, 8, 12, 16]).forEach(function (c) {
      if (c) K.paint(el('line', { x1: ox, y1: P(0, c)[1], x2: P(4.25, 0)[0], y2: P(0, c)[1], 'stroke-width': 1.2, 'stroke-dasharray': '3 6' }, grid), null, E.GRID);
      txt(grid, ox - 12, P(0, c)[1] + 7, String(c), 18, E.MUT, 'end', 700);
    });
    [0, 1, 2, 3, 4].forEach(function (u) {
      K.paint(el('line', { x1: P(u, 0)[0], y1: oy, x2: P(u, 0)[0], y2: oy + 6, 'stroke-width': 2 }, grid), null, E.INK2);
      txt(grid, P(u, 0)[0], oy + 28, String(u), 20, E.INK2, 'middle', 700);
    });
    var axes = el('g', {}, g);
    E.vec(axes, E.INK2, 2.5, { head: 12 }).set(ox, oy, P(4.45, 0)[0], oy);
    E.vec(axes, E.INK2, 2.5, { head: 12 }).set(ox, oy, ox, P(0, cMax + 1.6)[1]);
    var xLab = txt(axes, P(4.45, 0)[0] + 10, oy + 7, o.xLab || 'u', 20, E.GOOD, 'start', 900);
    var yLab = halo(txt(axes, ox + 12, P(0, cMax + 1.6)[1] + 6, o.yLab || '扣分', 20, E.INK2, 'start', 900));   // 描底色：第 3 幕斜线区会盖到这里
    function path(fn, a, b, n) {
      var pts = [], N = n || 60;
      for (var k = 0; k <= N; k++) { var u = a + (b - a) * k / N; pts.push(P(u, fn(u))); }
      return pts.map(function (p, k) { return (k ? 'L ' : 'M ') + p[0].toFixed(1) + ' ' + p[1].toFixed(1); }).join(' ');
    }
    return {
      g: g, grid: grid, axes: axes, xLab: xLab, yLab: yLab, P: P, path: path,
      curve: function (fn, color, w, a, b) {
        var c = el('path', { d: path(fn, a === undefined ? 0 : a, b === undefined ? 4 : b), fill: 'none', 'stroke-width': w || 4, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, g);
        K.paint(c, null, color);
        return c;
      },
      dot: function (color, r, par) {
        var d = el('g', {}, par || g);
        var c = K.paint(el('circle', { r: r || 8, 'stroke-width': 2.5 }, d), color, 'var(--demo-surface)');
        var api = { g: d, c: c, set: function (u, cc) { var p = P(u, cc); c.setAttribute('cx', p[0].toFixed(1)); c.setAttribute('cy', p[1].toFixed(1)); return api; } };
        return api;
      }
    };
  };

  /* ── 第 2、3 幕左半边的同一套：上面小跑道（和代价图横轴对齐：u 格的脚印正好在图上 u 的正上方），下面代价图 ── */
  E.MINI = { x0: 100, y: 88, w: 68, scale: 0.5, nums: false, arcH: 30, flagH: 44, flagLab: false };
  E.CH = { ox: 100, oy: 384, su: 68, sc: 14 };

  /* ── 三样东西的胶囊：变量 绿、目标 紫、约束 橙（片头、第 4 幕用） ── */
  E.THREE = [{ s: '变量', c: E.GOOD }, { s: '目标', c: E.VIOLET }, { s: '约束', c: E.BAD }];
})();
