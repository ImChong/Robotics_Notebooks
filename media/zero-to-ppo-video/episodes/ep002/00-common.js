// 零基础篇 002：为什么从人形 / 双足学起——高自由度、浮动基、接触切换、强不稳定（路线 §L−1「为什么以"人形 / 双足"为主载体」）
// 本目录按文件名顺序加载：00 共用部件 → 10 片头主图 → 21…25 五幕分镜 → 90 组装 window.EpisodeVideo。旁白在 ../ep002.py。
// 工具函数、侧视人形、机械臂与火柴人图标取自 ../ep001/00-common.js（命名空间换成 EP002）。
// 分镜都画在 800×420 的网格上。draw(st, t)：st 是分镜时间（旁白念完前画面停在 to 之前，一次性事件用它），
// t 是这一幕的真实时间（呼吸、旋转、绕圈这类循环动作用它，停帧时也不会卡住）。
(function () {
  var K = window.PaperDemoKit;
  var C = K.xColors;
  var E = window.EP002 = { K: K, scenes: [] };

  /* 颜色：片头回顾上一集的四步沿用 001（感知 good、规划 violet、控制 accent、执行 warn）；
     本集四个难点各用一色：关节多 accent、浮动基 violet、接触切换 good、强不稳定 warn；错误 / 要倒 bad；地面推力 good */
  E.ACC = C.accent; E.GOOD = C.good; E.WARN = C.warn; E.BAD = C.bad; E.VIOLET = 'var(--violet)';
  E.MUT = C.muted; E.INK = C.ink; E.INK2 = C.ink2; E.S1 = C.surface; E.S2 = C.surface2; E.BORDER = C.border; E.GRID = C.grid;
  E.STEPS = [
    { name: '感知', color: E.GOOD },
    { name: '规划', color: E.VIOLET },
    { name: '控制', color: E.ACC },
    { name: '执行', color: E.WARN }
  ];
  E.DIFF = [
    { name: '关节多', color: E.ACC },
    { name: '浮动基', color: E.VIOLET },
    { name: '接触切换', color: E.GOOD },
    { name: '强不稳定', color: E.WARN }
  ];

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
        var id = 'ep002-draw-mask-' + (++maskN);
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

  /* ── 侧视人形机器人（朝 +x），本地坐标里髋在 (0, 302)，脚底在 y = 388，头顶约 y = 190。
     place({ x, rot, pivot, scale, dy })：先在本地绕 pivot（本地坐标）转 rot 度、以脚底 (0,388) 为心缩放，再平移到 x。
     legs(near, far)：两条腿 [髋角, 膝角]（度，0 = 竖直向下，正值朝 +x，膝角也是全局角）；arm([肩, 肘])。
     walk(ph)：kit 的 poseWalk 步态；腿长 42 + 42、髋摆 ±22°，一个周期约走 4·84·sin22° ≈ 126，ph = 已走距离 / 126 时脚不打滑。
     pts()：近腿髋 / 膝 / 踝与摄像头、小盒子的本地坐标（加上 x 就是画面坐标，未计旋转缩放）。
     legIK(hip, ankle)：给髋、踝的本地坐标，返回 [髋角, 膝角]（膝盖朝 +x 弯），S3 走路时脚踩实地面用。 ── */
  E.robot = function (parent, X) {
    var outer = el('g', {}, parent), g = el('g', {}, outer);
    var R = Math.PI / 180;
    function seg(len, ang, p) { return [p[0] + len * Math.sin(ang * R), p[1] + len * Math.cos(ang * R)]; }
    function line(w, c, parentG) { var l = el('line', { 'stroke-width': w, 'stroke-linecap': 'round' }, parentG || g); K.paint(l, null, c); return l; }
    function put(l, a, b) { l.setAttribute('x1', a[0].toFixed(1)); l.setAttribute('y1', a[1].toFixed(1)); l.setAttribute('x2', b[0].toFixed(1)); l.setAttribute('y2', b[1].toFixed(1)); }
    var far = el('g', {}, g); op(far, 0.5);
    var farThigh = line(7, E.INK2, far), farShin = line(7, E.INK2, far), farFoot = line(4, E.INK2, far);
    var torso = el('rect', { x: -15, y: 228, width: 30, height: 72, rx: 8, 'stroke-width': 3 }, g);
    K.paint(torso, E.S2, E.INK);
    var neck = line(4, E.INK); put(neck, [0, 222], [0, 228]);
    var head = el('circle', { cx: 0, cy: 206, r: 16, 'stroke-width': 3 }, g);
    K.paint(head, E.S2, E.INK);
    var cam = K.paint(el('rect', { x: 5, y: 199, width: 12, height: 8, rx: 2 }, g), E.INK2);
    var thigh = line(8, E.INK), shin = line(8, E.INK), foot = line(4, E.INK);
    var upper = line(6, E.INK), fore = line(6, E.INK);
    var imu = K.paint(el('rect', { x: -5, y: 250, width: 10, height: 10, rx: 2 }, g), E.MUT);  // 画在手臂之后，不被挡住
    var hipDot = K.paint(el('circle', { r: 5 }, g), E.INK), kneeDot = K.paint(el('circle', { r: 5 }, g), E.INK), ankleDot = K.paint(el('circle', { r: 5 }, g), E.INK);
    var P = { hip: [0, 302], knee: [0, 344], ankle: [0, 386], cam: [11, 203], imu: [0, 255], shoulder: [0, 238] };
    var legA = [4, 0], legB = [-4, 0], armA = [8, 4];
    function leg(hipP, a, th, sh, ft) {
      var k = seg(42, a[0], hipP), an = seg(42, a[1], k);
      put(th, hipP, k); put(sh, k, an); put(ft, [an[0] - 4, an[1] + 2], [an[0] + 18, an[1] + 2]);
      return [k, an];
    }
    function redraw() {
      var kn = leg([0, 302], legA, thigh, shin, foot);
      leg([4, 302], legB, farThigh, farShin, farFoot);
      P.knee = kn[0]; P.ankle = kn[1];
      [[hipDot, P.hip], [kneeDot, P.knee], [ankleDot, P.ankle]].forEach(function (d) { d[0].setAttribute('cx', d[1][0].toFixed(1)); d[0].setAttribute('cy', d[1][1].toFixed(1)); });
      var e = seg(30, armA[0], P.shoulder); put(upper, P.shoulder, e); put(fore, e, seg(28, armA[1], e));
    }
    var state = { x: X || 0, rot: 0, pivot: [0, 388], scale: 1, dy: 0 };
    function place(o) {
      for (var k in o) state[k] = o[k];
      outer.setAttribute('transform', 'translate(' + state.x.toFixed(2) + ' ' + state.dy.toFixed(2) + ') translate(0 388) scale(' + state.scale.toFixed(4) + ') translate(0 -388)');
      g.setAttribute('transform', state.rot ? 'rotate(' + state.rot.toFixed(2) + ' ' + state.pivot[0] + ' ' + state.pivot[1] + ')' : '');
    }
    redraw(); place({});
    return {
      g: outer, body: g, head: head, cam: cam, imu: imu, torso: torso, neck: neck,
      thigh: thigh, shin: shin, foot: foot, far: far, upper: upper, fore: fore,
      hipDot: hipDot, kneeDot: kneeDot, ankleDot: ankleDot,
      nearLeg: [thigh, shin, foot, hipDot, kneeDot, ankleDot],
      place: place, state: state,
      legs: function (near, farA) { legA = near; legB = farA || legB; redraw(); },
      arm: function (a) { armA = a; redraw(); },
      stand: function () { legA = [4, 0]; legB = [-4, 0]; armA = [8, 4]; redraw(); },
      walk: function (ph) { var w = K.poseWalk(ph); legA = w.legA; legB = w.legB; armA = [w.armA[0] * 0.8, w.armA[1] * 0.8]; redraw(); },
      legIK: function (hip, ankle) {
        var dx = ankle[0] - hip[0], dy = ankle[1] - hip[1], r = Math.min(84, Math.hypot(dx, dy));
        var phi = Math.atan2(dx, dy) / R, a = Math.acos(r / 84) / R;
        return [phi + a, phi - a];
      },
      pts: function () { return P; },
      // 近腿描边换色（如执行时变 warn）；dash 为虚线；c 为空恢复 ink
      legStyle: function (c, dash, opacity) {
        // 虚线时用平头线帽，否则圆头会把空隙填满、看起来还是实线
        [thigh, shin, foot].forEach(function (l) { l.style.stroke = c || E.INK; l.setAttribute('stroke-dasharray', dash || ''); l.setAttribute('stroke-linecap', dash ? 'butt' : 'round'); });
        [hipDot, kneeDot, ankleDot].forEach(function (d) { d.style.fill = c || E.INK; });
        var o = opacity === undefined ? 1 : opacity;
        [thigh, shin, foot, hipDot, kneeDot, ankleDot].forEach(function (n) { op(n, o); });
        far.querySelectorAll('line').forEach(function (l) { l.setAttribute('stroke-dasharray', dash || ''); l.setAttribute('stroke-linecap', dash ? 'butt' : 'round'); });
        op(far, 0.5 * o);
      }
    };
  };

  /* ── 机器人简笔图标（片头、S1、S2、S5）。(x, y) 为脚底中心，s 为缩放 ── */
  function holder(parent, x, y, s) { return el('g', { transform: 'translate(' + x + ' ' + y + ') scale(' + (s || 1) + ')' }, parent); }
  // 机械臂（侧视）：底座 + 两节连杆 + 夹爪；set(a1, a2) 两个关节角（度，0 = 竖直向上），返回夹爪位置（本地）
  E.arm = function (parent, x, y, s, color) {
    var g = holder(parent, x, y, s), c = color || E.INK2;
    K.paint(el('path', { d: 'M -34 0 L 34 0 L 22 -18 L -22 -18 Z', 'stroke-width': 3 }, g), E.S2, c);
    var l1 = el('line', { 'stroke-width': 12, 'stroke-linecap': 'round' }, g), l2 = el('line', { 'stroke-width': 10, 'stroke-linecap': 'round' }, g);
    K.paint(l1, null, c); K.paint(l2, null, c);
    var j = [0, 1, 2].map(function () { return K.paint(el('circle', { r: 7, 'stroke-width': 3 }, g), E.S1, c); });
    var f1 = el('line', { 'stroke-width': 5, 'stroke-linecap': 'round' }, g), f2 = el('line', { 'stroke-width': 5, 'stroke-linecap': 'round' }, g);
    K.paint(f1, null, c); K.paint(f2, null, c);
    function set(a1, a2) {
      var r = Math.PI / 180, p0 = [0, -18];
      var p1 = [p0[0] + 64 * Math.sin(a1 * r), p0[1] - 64 * Math.cos(a1 * r)];
      var p2 = [p1[0] + 54 * Math.sin((a1 + a2) * r), p1[1] - 54 * Math.cos((a1 + a2) * r)];
      var a = (a1 + a2) * r, ux = Math.sin(a), uy = -Math.cos(a), nx = -uy, ny = ux;
      l1.setAttribute('x1', p0[0]); l1.setAttribute('y1', p0[1]); l1.setAttribute('x2', p1[0].toFixed(1)); l1.setAttribute('y2', p1[1].toFixed(1));
      l2.setAttribute('x1', p1[0].toFixed(1)); l2.setAttribute('y1', p1[1].toFixed(1)); l2.setAttribute('x2', p2[0].toFixed(1)); l2.setAttribute('y2', p2[1].toFixed(1));
      [p0, p1, p2].forEach(function (p, k) { j[k].setAttribute('cx', p[0].toFixed(1)); j[k].setAttribute('cy', p[1].toFixed(1)); });
      [[f1, 1], [f2, -1]].forEach(function (f) {
        f[0].setAttribute('x1', (p2[0] + 8 * nx * f[1]).toFixed(1)); f[0].setAttribute('y1', (p2[1] + 8 * ny * f[1]).toFixed(1));
        f[0].setAttribute('x2', (p2[0] + 8 * nx * f[1] + 16 * ux).toFixed(1)); f[0].setAttribute('y2', (p2[1] + 8 * ny * f[1] + 16 * uy).toFixed(1));
      });
      return p2;
    }
    set(25, 60);
    return { g: g, set: set };
  };
  // 人形机器人图标：kit 的火柴人（髋在 y−48，脚底在 y）；walk(ph) 走一步的相位 0…1
  E.humanoid = function (parent, x, y, s, color) {
    var g = holder(parent, x, y, s);
    var f = K.stickFigure(color || E.INK2, 6, false, 0.55);
    g.appendChild(f.el);
    function stand() { f.pose(0, -48, { lean: 0, armA: [14, 8], armB: [-14, -8], legA: [8, 2], legB: [-8, -2] }); }   // 双脚略分开，不然像一根杆
    stand();
    // idle(t)：站着轻轻晃（重心微移、手臂小幅摆）。图标不往前走，就不要用 walk，原地迈步看起来像在跑步机上
    function idle(t) {
      var w = Math.sin(2 * Math.PI * t / 3.2);
      f.pose(0, -48, { lean: 1.5 * w, armA: [14 + 4 * w, 8 + 6 * w], armB: [-14 + 4 * w, -8 + 6 * w], legA: [8, 2], legB: [-8, -2] });
    }
    return { g: g, walk: function (ph) { f.pose(0, -48, K.poseWalk(ph)); }, stand: stand, idle: idle };
  };
  // 四足（侧视，朝 +x）：身子 + 头 + 四条两节腿（远侧两条半透明）
  E.quadruped = function (parent, x, y, s, color) {
    var g = holder(parent, x, y, s), c = color || E.INK2;
    function leg(hx, kx, far) {
      var l = el('path', { d: 'M ' + hx + ' -50 L ' + kx + ' -26 L ' + hx + ' 0', fill: 'none', 'stroke-width': 5, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, g);
      K.paint(l, null, c);
      if (far) op(l, 0.5);
    }
    leg(-26, -20, true); leg(30, 36, true);
    K.paint(el('rect', { x: -44, y: -72, width: 88, height: 24, rx: 10, 'stroke-width': 4 }, g), E.S2, c);
    K.paint(el('line', { x1: 40, y1: -66, x2: 50, y2: -74, 'stroke-width': 5, 'stroke-linecap': 'round' }, g), null, c);
    K.paint(el('circle', { cx: 56, cy: -78, r: 11, 'stroke-width': 3 }, g), E.S2, c);
    leg(-34, -28); leg(22, 28);
    return { g: g };
  };
  // 轮式底盘（侧视）：车身 + 两个轮子 + 顶上的传感器
  E.wheeled = function (parent, x, y, s, color) {
    var g = holder(parent, x, y, s), c = color || E.INK2;
    K.paint(el('rect', { x: -8, y: -66, width: 16, height: 16, rx: 3, 'stroke-width': 3 }, g), E.S1, c);
    K.paint(el('rect', { x: -52, y: -52, width: 104, height: 34, rx: 8, 'stroke-width': 4 }, g), E.S2, c);
    [-30, 30].forEach(function (wx) {
      K.paint(el('circle', { cx: wx, cy: -14, r: 14, 'stroke-width': 4 }, g), E.S1, c);
      K.paint(el('circle', { cx: wx, cy: -14, r: 3 }, g), c);
    });
    return { g: g };
  };

  /* ── 正面人形（S1 数关节、S4 站不稳）。本地坐标：两脚中间 x = 0，脚底 y = 388，头顶 y ≈ 92。
     14 个关节点：脖子、肩 ×2、肘 ×2、腕 ×2、腰、髋 ×2、膝 ×2、踝 ×2（一个点代表那一处的关节，不是一个自由度）。
     pose({ shift, lean, lift: [左, 右], arm: [[肩, 肘], [肩, 肘]], tilt, pivot })：
       shift 上身（含髋）横移；lean 上身绕腰转（度）；lift 两只脚离地高度；arm 两臂外张角（度，0 = 竖直下垂）；
       tilt 整个人绕 pivot（本地坐标）转（度，正值往右倒）。左右都按画面左右说。
     place(x, s)：平移到画面 x、以 (0, 388) 为心缩放。com()：重心的本地坐标（腰附近，跟着 shift / lean 走）。 ── */
  E.front = function (parent) {
    var outer = el('g', {}, parent), g = el('g', {}, outer), R = Math.PI / 180;
    var FOOT = [-24, 24], HIP = [-22, 22], LEG = 60, SH = [-46, 46];
    function line(w, c, par) { var l = el('line', { 'stroke-width': w, 'stroke-linecap': 'round' }, par || g); K.paint(l, null, c || E.INK); return l; }
    function put(l, a, b) { l.setAttribute('x1', a[0].toFixed(1)); l.setAttribute('y1', a[1].toFixed(1)); l.setAttribute('x2', b[0].toFixed(1)); l.setAttribute('y2', b[1].toFixed(1)); }
    function at(c, p) { c.setAttribute('cx', p[0].toFixed(1)); c.setAttribute('cy', p[1].toFixed(1)); }
    var legs = [0, 1].map(function () { return { thigh: line(9), shin: line(9), foot: line(7) }; });
    var upper = el('g', {}, g);
    var arms = [0, 1].map(function () { return { up: line(7, null, upper), fore: line(7, null, upper), hand: K.paint(el('circle', { r: 6 }, upper), E.INK) }; });
    var torso = K.paint(el('path', { d: 'M -46 150 L 46 150 L 30 230 L -30 230 Z', 'stroke-width': 3, 'stroke-linejoin': 'round' }, upper), E.S2, E.INK);
    var neck = line(5, null, upper); put(neck, [0, 132], [0, 150]);
    var head = K.paint(el('circle', { cx: 0, cy: 112, r: 20, 'stroke-width': 3 }, upper), E.S2, E.INK);
    var pelvis = K.paint(el('rect', { x: -32, y: 240, width: 64, height: 22, rx: 6, 'stroke-width': 3 }, g), E.S2, E.INK);
    var waistLink = line(5); 
    function dot(par) { return K.paint(el('circle', { r: 6.5, 'stroke-width': 2 }, par || g), E.INK, E.S1); }
    var D = {
      neck: dot(upper), sh: [dot(upper), dot(upper)], el: [dot(upper), dot(upper)], wr: [dot(upper), dot(upper)],
      waist: dot(), hip: [dot(), dot()], knee: [dot(), dot()], ankle: [dot(), dot()]
    };
    var groups = { legs: D.hip.concat(D.knee, D.ankle), waist: [D.waist], arms: D.sh.concat(D.el, D.wr), neck: [D.neck] };
    var all = groups.legs.concat(groups.waist, groups.arms, groups.neck);
    var st = { shift: 0, lean: 0, lift: [0, 0], arm: [[8, 4], [8, 4]], tilt: 0, pivot: [0, 388] };
    var X = 0, S = 1;
    function draw() {
      var sx = st.shift, w = [sx, 236];
      upper.setAttribute('transform', 'translate(' + sx.toFixed(2) + ' 0) rotate(' + st.lean.toFixed(2) + ' 0 236)');
      pelvis.setAttribute('x', (sx - 32).toFixed(1));
      put(waistLink, [sx, 230], [sx, 244]);
      at(D.waist, w); at(D.neck, [0, 140]);
      [0, 1].forEach(function (i) {
        var hip = [HIP[i] + sx, 258], an = [FOOT[i], 376 - st.lift[i]];
        var dx = an[0] - hip[0], dy = an[1] - hip[1], r = Math.min(2 * LEG, Math.hypot(dx, dy));
        // 正面看不出膝盖往前弯，就让弯曲的膝盖往外鼓一点（按透视压扁 35%）
        var bulge = 0.35 * Math.sqrt(Math.max(0, LEG * LEG - r * r / 4)) * (i ? 1 : -1);
        var kn = [(hip[0] + an[0]) / 2 + bulge, (hip[1] + an[1]) / 2];
        put(legs[i].thigh, hip, kn); put(legs[i].shin, kn, an); put(legs[i].foot, [an[0] - 15, an[1] + 8], [an[0] + 15, an[1] + 8]);
        at(D.hip[i], hip); at(D.knee[i], kn); at(D.ankle[i], an);
        var sgn = i ? 1 : -1, shp = [SH[i], 156], a = st.arm[i];
        var e = [shp[0] + sgn * 46 * Math.sin(a[0] * R), shp[1] + 46 * Math.cos(a[0] * R)];
        var h = [e[0] + sgn * 42 * Math.sin(a[1] * R), e[1] + 42 * Math.cos(a[1] * R)];
        put(arms[i].up, shp, e); put(arms[i].fore, e, h); at(arms[i].hand, h);
        at(D.sh[i], shp); at(D.el[i], e); at(D.wr[i], h);
      });
      g.setAttribute('transform', st.tilt ? 'rotate(' + st.tilt.toFixed(2) + ' ' + st.pivot[0] + ' ' + st.pivot[1] + ')' : '');
    }
    function place(x, s) { X = x; S = s || 1; outer.setAttribute('transform', 'translate(' + X + ' 0) translate(0 388) scale(' + S + ') translate(0 -388)'); }
    draw(); place(0, 1);
    return {
      g: outer, body: g, upper: upper, head: head, torso: torso, pelvis: pelvis, dots: D, groups: groups, all: all, legs: legs, arms: arms,
      place: place,
      pose: function (o) { for (var k in o) st[k] = o[k]; draw(); },
      reset: function () { st.shift = 0; st.lean = 0; st.lift = [0, 0]; st.arm = [[8, 4], [8, 4]]; st.tilt = 0; st.pivot = [0, 388]; draw(); },
      com: function () { return [st.shift, 236]; },
      // 关节点 d 的画面坐标：上身的点先按 upper 的变换（绕腰转 lean、再横移 shift）
      locate: function (d) {
        var p = [+d.getAttribute('cx'), +d.getAttribute('cy')];
        if (d.parentNode === upper) {
          var a = st.lean * R, px = p[0], py = p[1] - 236;
          p = [px * Math.cos(a) - py * Math.sin(a) + st.shift, px * Math.sin(a) + py * Math.cos(a) + 236];
        }
        return this.world(p);
      },
      // 本地坐标 p 转成画面坐标（含 tilt、place）
      world: function (p) {
        var a = st.tilt * R, px = p[0] - st.pivot[0], py = p[1] - st.pivot[1];
        var q = [st.pivot[0] + px * Math.cos(a) - py * Math.sin(a), st.pivot[1] + px * Math.sin(a) + py * Math.cos(a)];
        return [X + q[0] * S, 388 + (q[1] - 388) * S];
      },
      footX: FOOT
    };
  };

  /* ── 重心标记：圆圈里对角两块涂色（工程图里的重心符号），(x, y) 为圆心 ── */
  E.comMark = function (parent, color, r) {
    var g = el('g', {}, parent), rr = r || 11, c = color || E.WARN;
    K.paint(el('circle', { r: rr, 'stroke-width': 2.5 }, g), E.S1, c);
    K.paint(el('path', { d: 'M 0 0 L ' + rr + ' 0 A ' + rr + ' ' + rr + ' 0 0 1 0 ' + rr + ' Z' }, g), c);
    K.paint(el('path', { d: 'M 0 0 L ' + (-rr) + ' 0 A ' + rr + ' ' + rr + ' 0 0 1 0 ' + (-rr) + ' Z' }, g), c);
    return { g: g, at: function (x, y) { g.setAttribute('transform', 'translate(' + x.toFixed(2) + ' ' + y.toFixed(2) + ')'); } };
  };
})();
