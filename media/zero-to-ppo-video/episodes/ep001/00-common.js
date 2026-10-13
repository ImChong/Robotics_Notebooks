// 零基础篇 001：四步循环——感知 → 规划 → 控制 → 执行（路线 §L−1「30 秒看懂"一台机器人在干嘛"」）
// 本目录按文件名顺序加载：00 共用部件 → 10 片头主图 → 21…28 八幕分镜 → 90 组装 window.EpisodeVideo。旁白在 ../ep001.py。
// 分镜都画在 800×420 的网格上。draw(st, t)：st 是分镜时间（旁白念完前画面停在 to 之前，一次性事件用它），
// t 是这一幕的真实时间（呼吸、旋转、绕圈这类循环动作用它，停帧时也不会卡住）。
(function () {
  var K = window.PaperDemoKit;
  var C = K.xColors;
  var E = window.EP001 = { K: K, scenes: [] };

  /* 颜色：感知 good、规划 violet、控制 accent、执行 warn；错误 / 警示 bad */
  E.ACC = C.accent; E.GOOD = C.good; E.WARN = C.warn; E.BAD = C.bad; E.VIOLET = 'var(--violet)';
  E.MUT = C.muted; E.INK = C.ink; E.INK2 = C.ink2; E.S1 = C.surface; E.S2 = C.surface2; E.BORDER = C.border; E.GRID = C.grid;
  E.STEPS = [
    { name: '感知', color: E.GOOD },
    { name: '规划', color: E.VIOLET },
    { name: '控制', color: E.ACC },
    { name: '执行', color: E.WARN }
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
        var id = 'ep001-draw-mask-' + (++maskN);
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

  E.el = el; E.txt = txt; E.op = op; E.clamp = clamp; E.fade = fade; E.win = window_; E.pop = pop; E.scaleAt = scaleAt;
  E.drawOn = drawOn; E.arrow = arrow; E.curveArrow = curveArrow; E.check = check; E.cross = cross; E.tag = tag; E.tint = tint; E.glow = glow;

  /* ── 四步条（S2–S6 顶部常驻）。set(states, t)：states 为 4 个 'off' | 'cur' | 'done'；t 用来呼吸 ── */
  E.stepBar = function (parent) {
    var g = el('g', {}, parent), xs = [40, 235, 430, 625], rects = [], labels = [];
    E.STEPS.forEach(function (st, i) {
      rects.push(el('rect', { x: xs[i], y: 12, width: 150, height: 40, rx: 10 }, g));
      labels.push(txt(g, xs[i] + 75, 39, st.name, 20, E.INK2, 'middle', 700));
    });
    var arrows = [[192, 231], [387, 426], [582, 621]].map(function (a) { return arrow(g, [[a[0], 32], [a[1], 32]], E.INK2, 2.5, { head: 9 }); });
    var fb = arrow(g, [[700, 52], [700, 62], [115, 62], [115, 54]], E.MUT, 2, { dash: '6 5', head: 8 });
    var star = txt(g, 566, 39, '★', 18, E.ACC);
    op(star, 0);
    return {
      g: g, rects: rects, labels: labels, arrows: arrows, fb: fb, star: star, xs: xs,
      set: function (states, t) {
        states.forEach(function (s, i) {
          var c = E.STEPS[i].color, r = rects[i], lb = labels[i];
          if (s === 'cur') {
            K.paint(r, tint(c, 22), c); r.style.strokeWidth = 3;
            r.style.strokeOpacity = (0.6 + 0.4 * Math.sin(2 * Math.PI * (t || 0))).toFixed(3);
            lb.style.fill = c;
          } else if (s === 'done') {
            K.paint(r, E.S2, c); r.style.strokeWidth = 2; r.style.strokeOpacity = 0.5; lb.style.fill = E.INK2;
          } else {
            K.paint(r, E.S2, E.BORDER); r.style.strokeWidth = 2; r.style.strokeOpacity = 1; lb.style.fill = E.INK2;
          }
        });
      }
    };
  };

  /* ── 客厅：地面 + 门（顶 y 148、中心 (741,268)、标签 (741,138)）+ 纸箱。纸箱单独一组（box），需要遮住机器人时把它 appendChild 到最后 ── */
  E.room = function (parent) {
    var back = el('g', {}, parent);
    var floor = el('line', { x1: 20, y1: 388, x2: 780, y2: 388, 'stroke-width': 3 }, back);
    K.paint(floor, null, E.BORDER);
    for (var x = 40; x <= 760; x += 60) K.paint(el('line', { x1: x, y1: 390, x2: x, y2: 398, 'stroke-width': 2 }, back), null, E.GRID);
    // 门高 240，约为机器人身高（≈ 199）的 1.2 倍：1.7 m 的机器人配 2 m 的门，比例才对
    var door = el('rect', { x: 712, y: 148, width: 58, height: 240, rx: 4, 'stroke-width': 2.5 }, back);
    K.paint(door, E.S2, E.INK2);
    var knob = K.paint(el('circle', { cx: 760, cy: 272, r: 4 }, back), E.INK2);
    var doorLabel = txt(back, 741, 138, '门口', 20, E.INK2);
    var boxG = el('g', {}, parent);
    var box = el('rect', { x: 380, y: 342, width: 66, height: 46, rx: 3, 'stroke-width': 3 }, boxG);
    K.paint(box, E.S2, E.MUT);
    var tape = el('line', { x1: 413, y1: 342, x2: 413, y2: 388, 'stroke-width': 2 }, boxG);
    K.paint(tape, null, E.MUT);
    return { back: back, floor: floor, door: door, knob: knob, doorLabel: doorLabel, boxG: boxG, box: box, tape: tape };
  };

  /* ── 侧视人形机器人（朝 +x），本地坐标里髋在 (0, 302)，脚底在 y = 388，头顶约 y = 190。
     place({ x, rot, pivot, scale, dy })：先在本地绕 pivot（本地坐标）转 rot 度、以脚底 (0,388) 为心缩放，再平移到 x。
     legs(near, far)：两条腿 [髋角, 膝角]（度，0 = 竖直向下，正值朝 +x，膝角也是全局角）；arm([肩, 肘])。
     walk(ph)：kit 的 poseWalk 步态；腿长 42 + 42、髋摆 ±22°，一个周期约走 4·84·sin22° ≈ 126，ph = 已走距离 / 126 时脚不打滑。
     pts()：近腿髋 / 膝 / 踝与摄像头、小盒子的本地坐标（加上 x 就是画面坐标，未计旋转缩放）。 ── */
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

  /* ── 俯视小地图（S3、S6）：面板 + 网格 + 迷雾 + 机器人白点 + 纸箱 + 门 + 绕行路线 + 脚印。
     setReveal(x)：迷雾只盖 x 右边；at(u)：路线上比例 u 处的点（u = (侧视 X − 110) / 580）；
     dot(u)：白点放到路线上 u；路线 route 是 path，可配合 E.drawOn 画出。 ── */
  E.MAP_K = 320 / 350;   // 小地图缩放（见 E.miniMap 末尾）
  E.miniMap = function (parent) {
    var g = el('g', {}, parent);
    var panel = el('rect', { x: 430, y: 70, width: 350, height: 126, rx: 12, 'stroke-width': 2 }, g);
    K.paint(panel, E.S1, E.BORDER);
    var title = txt(g, 444, 92, '心里的地图（俯视）', 18, E.INK2, 'start');
    var grid = el('g', {}, g);
    for (var x = 442; x <= 768; x += 22) K.paint(el('line', { x1: x, y1: 100, x2: x, y2: 188, 'stroke-width': 1 }, grid), null, E.GRID);
    for (var y = 100; y <= 188; y += 22) K.paint(el('line', { x1: 442, y1: y, x2: 768, y2: y, 'stroke-width': 1 }, grid), null, E.GRID);
    var box = el('rect', { x: 585, y: 140, width: 24, height: 24, 'stroke-width': 2 }, g);
    K.paint(box, E.S2, E.MUT);
    var fog = el('rect', { x: 442, y: 100, width: 326, height: 88 }, g);
    K.paint(fog, E.S1); fog.style.fillOpacity = 0.92;
    var door = el('line', { x1: 766, y1: 130, x2: 766, y2: 174, 'stroke-width': 4 }, g);
    K.paint(door, null, E.INK2);
    var doorLabel = txt(g, 752, 124, '门', 18, E.INK2);
    var route = el('path', { d: 'M 466 152 C 530 152 550 118 597 118 S 670 152 766 152', fill: 'none', 'stroke-width': 3, 'stroke-linecap': 'round' }, g);
    K.paint(route, null, E.VIOLET);
    var prints = el('g', {}, g);
    var dotEl = K.paint(el('circle', { cx: 466, cy: 152, r: 6 }, g), E.INK);
    function at(u) {
      if (!route.isConnected) return [466 + 300 * u, 152];
      if (!route.__L) route.__L = route.getTotalLength();
      var p = route.getPointAtLength(clamp(u, 0, 1) * route.__L);
      return [p.x, p.y];
    }
    // 整张地图放在门的左边：面板 (430,70,350,126) 映射到 (380,70,320,115)，右缘 700，不压门（门在 712–770、顶 148）；
    // 机器人走到地图下面时头顶 ≈ 188，面板底 185。里面的字按缩放补大，成片里仍不小于 18
    g.setAttribute('transform', 'translate(380 70) scale(' + E.MAP_K + ') translate(-430 -70)');
    [title, doorLabel].forEach(function (n) { n.setAttribute('font-size', 20); });
    return {
      g: g, panel: panel, title: title, grid: grid, box: box, fog: fog, door: door, doorLabel: doorLabel,
      route: route, prints: prints, dotEl: dotEl, at: at,
      setReveal: function (rx) { rx = clamp(rx, 442, 768); fog.setAttribute('x', rx); fog.setAttribute('width', Math.max(0, 768 - rx)); },
      dot: function (u) { var p = at(u); dotEl.setAttribute('cx', p[0].toFixed(1)); dotEl.setAttribute('cy', p[1].toFixed(1)); return p; },
      // 在路线 u 处放一个小脚印（左右交替偏 side·5px），返回椭圆
      footprint: function (u, side, color, rx) {
        var p = at(u), q = at(Math.min(1, u + 0.01)), ang = Math.atan2(q[1] - p[1], q[0] - p[0]);
        var cx = (p[0] - side * 5 * Math.sin(ang)).toFixed(1), cy = (p[1] + side * 5 * Math.cos(ang)).toFixed(1);  // 绕脚印自己的中心转
        var e = el('ellipse', { cx: cx, cy: cy, rx: rx || 4.5, ry: 2.5, transform: 'rotate(' + (ang * 180 / Math.PI).toFixed(1) + ' ' + cx + ' ' + cy + ')' }, prints);
        K.paint(e, color || E.VIOLET);
        return e;
      }
    };
  };

  /* ── 四台机器人的简笔图标（片头、S7）。(x, y) 为脚底中心（扫地机为圆心），s 为缩放 ── */
  function holder(parent, x, y, s) { return el('g', { transform: 'translate(' + x + ' ' + y + ') scale(' + (s || 1) + ')' }, parent); }
  // 扫地机（俯视）：圆盘 + 顶上的激光雷达 + 前方保险杠（朝上；需要朝右就外面套 rotate(90)）
  E.vacuum = function (parent, x, y, s, color) {
    var g = holder(parent, x, y, s), c = color || E.INK2;
    var body = K.paint(el('circle', { r: 46, 'stroke-width': 4 }, g), E.S2, c);
    var lidar = K.paint(el('circle', { r: 14, 'stroke-width': 3 }, g), E.S1, c);
    var bumper = K.paint(el('path', { d: 'M -38 -26 A 46 46 0 0 1 38 -26', fill: 'none', 'stroke-width': 7, 'stroke-linecap': 'round' }, g), null, c);
    return { g: g, body: body, lidar: lidar, bumper: bumper };
  };
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
  // 自动驾驶汽车（侧视）：车身 + 车窗 + 两个轮子 + 车顶传感器；spin(圈数) 转动轮辐——只在车真的往前走时按走过的距离转，停着的车别转轮子
  E.car = function (parent, x, y, s, color) {
    var g = holder(parent, x, y, s), c = color || E.INK2;
    K.paint(el('path', { d: 'M -78 -16 L -78 -40 L -48 -44 L -26 -70 L 30 -70 L 54 -44 L 78 -38 L 78 -16 Z', 'stroke-width': 4, 'stroke-linejoin': 'round' }, g), E.S2, c);
    K.paint(el('path', { d: 'M -20 -64 L 26 -64 L 44 -46 L -38 -46 Z', 'stroke-width': 2 }, g), E.S1, c);
    K.paint(el('rect', { x: -10, y: -84, width: 20, height: 12, rx: 4, 'stroke-width': 3 }, g), E.S1, c);
    var wheels = [-46, 46].map(function (wx) {
      var w = el('g', { transform: 'translate(' + wx + ' -14)' }, g);
      K.paint(el('circle', { r: 16, 'stroke-width': 4 }, w), E.S1, c);
      var sp = el('line', { x1: -10, y1: 0, x2: 10, y2: 0, 'stroke-width': 3 }, w);
      K.paint(sp, null, c);
      return sp;
    });
    return {
      g: g,
      spin: function (t) { wheels.forEach(function (sp) { sp.setAttribute('transform', 'rotate(' + ((t * 360) % 360).toFixed(1) + ')'); }); }
    };
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
})();
