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
    // 门高 240：G1 身高 1.32 m 画出来约 165，门按同一比例约 1.9 m，比例才对
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

  /* ── 侧视人形机器人 = Unitree G1 的 URDF（Robot Description Gallery Online 收录，../../robot3d.js 渲染）。
     接口与原先的手画版一致：朝 +x，本地坐标里右髋在 (0, 302)、膝在 (0, 344)、脚底 y = 388（头顶约 223，G1 身高 1.32 m）；
     place / legs / arm / walk / stand / pts / legStyle 照旧。躯干、大腿、小腿、脚、远侧腿是「把手」——设它的 stroke / filter
     就给 G1 对应的连杆上色发光；髋 / 膝 / 踝圆点、摄像头（G1 头上的 d435）、IMU（imu_in_torso）只在被上色时显示，位置按正运动学算。 ── */
  E.robot = function (parent, X, opts) { return window.Robot3D.g1Side(parent, X, opts); };

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

  /* ── 四台机器人（片头、S7）：都是 Robot Description Gallery Online 收录的 URDF，由 ../../robot3d.js 渲染。
     (x, y) 为底下地面中心（俯视扫地机为圆心），s 为缩放 ── */
  // 扫地机 = iRobot Create 2（Roomba 600 系列机身）：从右前方斜看，宽 ≈ 92·s；{ top: true } 俯视、车头朝 +x，直径 ≈ 92·s
  E.vacuum = function (parent, x, y, s, opts) { return window.Robot3D.create2(parent, x, y, s, opts); };
  // 机械臂 = UR5 的 URDF：set(a1, a2) 同手画版（大臂离竖直的角、小臂再弯的角，度），joints() 给 6 个关节的画面位置
  E.arm = function (parent, x, y, s, opts) { return window.Robot3D.ur5(parent, x, y, s, opts); };

  // 自动驾驶汽车 = Toyota Prius（OSRF car_demo）：从右前方斜看，车头朝右、宽 ≈ 156·s；
  // spin(圈数) 转轮子——只在车真的往前走时按走过的距离转，停着的车别转轮子
  E.car = function (parent, x, y, s, opts) { return window.Robot3D.prius(parent, x, y, s, opts); };
  // 人形机器人图标 = G1 从右前方斜看（脚底在 (x, y)、高 ≈ 113·s）：stand / idle(t) 同手画版
  E.humanoid = function (parent, x, y, s, opts) { return window.Robot3D.g1Icon(parent, x, y, s, opts); };
})();
