// 011 第 3 幕：再转肩。接第 2 幕的结尾（肘已转 90°，手在 (1, 4)、x 朝上、y 朝左）。右栏上方 exp([S₁]θ₁) × 刚才的结果，下面「= 结果」、最下面几行算式。
// 句 1「再算肩」肩上的 ⊙ 亮，肩的矩阵出来（位置一列先是问号）；「朝向」「第三集转九十度的矩阵」左边两列填数、框蓝。
// 句 2「位置这一列」框紫；「轴上的点」「轴穿过原点」原点亮；「所以是零、零」位置一列填 0、0。
// 句 3「肩一转」「大臂、肘、小臂和手」整条胳膊依次鼓一下（上色）；「拿它乘上刚才的结果」右边出第 2 幕的结果；「一列一列」三列轮流亮。
// 句 4–5 第 1、2 列：原点上画同色的自由向量，「第二列拿一份」「第一列拿负一份」A 的那一列亮，结果那一列填数，「朝左 / 朝下」画满 90° 的蓝弧转过去。
// 句 6 第 3 列 (1, 4, 1)：(1, 4) 出空心紫圈；「第一列拿一份」从原点往上画红的 (0, 1)，「第二列拿四份」接着往左画绿的 (−4, 0)，
//    「第三列拿一份」(0, 0, 1)：位置那一份是 (0, 0)，不再挪；句 7 逐项相加，结果第三列填 −4、1、1，(−4, 1) 出点。
// 句 8「这就是答案」(−4, 1) 出虚线的手坐标系（x 朝左、y 朝下），整条胳膊绕原点转 90°（画满的蓝弧），手的坐标系落在虚线上。
// 句 9 验算朝向：肩、肘各画满 90° 的蓝弧，写 90° + 90° = 180°；句 10 验算位置：肘从 (1, 2) 沿虚线弧转到 (−2, 1)，「两格」小臂旁标 2 格，
//    「负二减二」「手正好在」(−4, 1) 打勾。
// 时刻取自配音的逐词时间，见 ../ep011.py。
(function () {
  var E = window.EP011, K = E.K, H = E.HEX;
  var D = Math.PI / 180, A = E.ARM, RX = A.RX, X = RX + 10;
  var T = {
    sh: 0.97, rot: 2.2, ep3: 3.23, r90: 4.07,
    pos: 6.42, ep10: 7.66, axisPt: 8.57, stay: 9.25, thru: 10.18, orig: 10.67, still: 11.41, z1: 13.47, z2: 13.91,
    whole: 15.36, parts: [16.13, 16.77, 17.35, 17.9], all: 18.32, mul: 20.14, prev: 20.79, cols: [22.18, 22.5, 22.84],
    c1: 24.42, xax: 25.4, c1v: 26.08, take1: 27.55, r1: 28.6, left: [30.62, 31.6],
    c2: 33.24, yax: 34.25, c2v: 34.83, take2: 36.39, r2: 37.65, down: [39.73, 40.7],
    c3: 42.3, pos3: 43.45, c3v: 43.97, t1: 45.22, t1r: 46.26, t2: 47.69, t2r: 48.78, t3: 50.55, t3r: 51.79,
    add: 54.31, s1: 54.98, s1r: 56.29, s2: 56.88, s2r: 58.2, s3: 58.59, s3r: 59.66,
    ans: 61.33, stop: 62.42, turn: [62.42, 64.4], gx: 64.18, gy: 65.24,
    chk: 67.37, both: 68.37, a90: [69.77, 70.34], deg180: 71.03, cx: 72.45, cy: 73.5,
    carry: 75.59, from12: 77.05, arc: [77.66, 78.9], to21: 78.12, ep10b: 79.09, fore: 80.41, two: 81.75, calc: 82.43, calcR: 83.56, ok: 84.55
  };
  var END = 87.41, R0 = 90 * D;

  E.scenes[2] = {
    title: '再转肩',
    dur: END,
    cues: [
      { at: 0, s: '再算肩：**朝向**也是第 3 集转 90° 的矩阵' },
      { at: 6.32, s: '**位置**一列：轴上的点转完不动；肩的轴穿过原点，所以是 (0, 0)' },
      { at: 15.26, s: '肩一转，大臂、肘、小臂和手一起绕原点转：拿它乘上刚才的结果，一列一列算' },
      { at: 24.32, s: '第 1 列 = 手的 x 轴 (0, 1, 0)：拿 1 份第 2 列 = (−1, 0, 0)，x 轴朝左' },
      { at: 33.14, s: '第 2 列 = 手的 y 轴 (−1, 0, 0)：拿 −1 份第 1 列 = (0, −1, 0)，y 轴朝下' },
      { at: 42.2, s: '第 3 列 = 手的**位置** (1, 4, 1)：1 × (0, 1, 0) + 4 × (−1, 0, 0) + 1 × (0, 0, 1)' },
      { at: 54.21, s: '0 − 4 + 0 = −4，1 + 0 + 0 = 1，0 + 0 + 1 = 1' },
      { at: 61.23, s: '答案：手停在 (−4, 1)，x 轴朝左，y 轴朝下' },
      { at: 67.27, s: '验算**朝向**：一共转了 90° + 90° = 180°，所以 x 轴朝左、y 轴朝下' },
      { at: 75.49, s: '验算**位置**：肘被带到 (−2, 1)（第 10 集算过），小臂朝左 2 格：−2 − 2 = −4' }
    ],
    build: function (K) {
      var s = K.sceneSvg('再转肩：肩的矩阵指数，朝向是第三集转九十度的矩阵，位置一列让轴上的点不动，肩的轴穿过原点，所以是 (0, 0)。肩一转，整条胳膊一起绕原点转：拿它乘第 2 幕的结果，一列一列算：第一列手的 x 轴 (0, 1, 0) 拿一份第二列得 (−1, 0, 0)，x 轴朝左；第二列手的 y 轴 (−1, 0, 0) 拿负一份第一列得 (0, −1, 0)，y 轴朝下；第三列手的位置 (1, 4, 1)：1 × (0, 1, 0) + 4 × (−1, 0, 0) + 1 × (0, 0, 1) = (−4, 1, 1)。答案：手停在 (−4, 1)，x 轴朝左，y 轴朝下。验算：一共转了 180°；肘被带到 (−2, 1)，小臂朝左两格，−2 − 2 = −4');

      var under = E.el('g', {}, s), armG = E.el('g', {}, s);
      var arm = E.arm2(armG, 320), u = arm.u, V = arm.view;
      var P = E.plane(under, { ox: A.OX, oy: A.OY, u: u, x0: A.X0, x1: A.X1, y0: A.Y0, y1: A.Y1 }).P;
      var UPPER = ['upper_arm_link'], RIGID = ['forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link'];
      var top = E.el('g', {}, s), O = P(0, 0);
      function seg(vec, a, b, p) { var A0 = P(a[0], a[1]), B0 = P(a[0] + (b[0] - a[0]) * p, a[1] + (b[1] - a[1]) * p); vec.set(A0[0], A0[1], B0[0], B0[1]); E.op(vec.g, p > 0 ? 1 : 0); }
      function arcPts(c, p0, a0, a1) { var pts = []; for (var k = 0; k <= 40; k++) { var q = E.rotP(p0, a0 + (a1 - a0) * k / 40, c); pts.push(P(q[0], q[1])); } return pts; }
      // 肩、肘的 ⊙（肘跟着胳膊走）
      var sRing = K.paint(E.el('circle', { cx: O[0], cy: O[1], r: 12, 'stroke-width': 3.5 }, top), E.S1, E.HI);
      K.paint(E.el('circle', { cx: O[0], cy: O[1], r: 4 }, top), E.HI);
      var eRing = K.paint(E.el('circle', { r: 12, 'stroke-width': 3.5 }, top), E.S1, E.HI), eDot = K.paint(E.el('circle', { r: 4 }, top), E.HI);
      var shArc = E.arcArrow(top, O[0], O[1], 0.75 * u, E.OMEGA, 3.5, { head: 12 });
      var elArc = E.arcArrow(top, 0, 0, 0.75 * u, E.OMEGA, 3.5, { head: 12 });
      // 原点上的自由向量
      var vx = E.vec(top, E.X, 5, { head: 14 }), vy = E.vec(top, E.Y, 5, { head: 14 });
      var arcX = E.arcArrow(top, O[0], O[1], 0.62 * u, E.OMEGA, 3, { head: 11 }), arcY = E.arcArrow(top, O[0], O[1], 0.62 * u, E.OMEGA, 3, { head: 11 });
      var vxLab = E.halo(E.txt(top, 0, 0, 'x 轴', 20, E.X, 'start', 900)), vyLab = E.halo(E.txt(top, 0, 0, 'y 轴', 20, E.Y, 'start', 900));
      // 第 3 列：按列拿份
      var p14 = K.paint(E.el('circle', { cx: P(1, 4)[0], cy: P(1, 4)[1], r: 8, 'stroke-width': 3, fill: 'none' }, top), null, E.VEL);
      var cA = E.vec(top, E.X, 4.5, { head: 13 }), cB = E.vec(top, E.Y, 4.5, { head: 13 });
      var cAL = E.halo(E.txt(top, P(0, 0.5)[0] + 12, P(0, 0.5)[1] + 6, '1 份', 18, E.X, 'start', 900));
      var cBL = E.halo(E.txt(top, P(-2, 1)[0], P(-2, 1)[1] - 12, '4 份', 18, E.Y, 'middle', 900));
      var pZero = E.halo(E.txt(top, P(-4, 1)[0], P(-4, 1)[1] - 18, '+ (0, 0)', 18, E.VEL, 'middle', 900));
      var p41 = P(-4, 1), d41 = K.paint(E.el('circle', { cx: p41[0], cy: p41[1], r: 7, 'stroke-width': 3 }, top), E.VEL, E.S1);
      var l41 = E.halo(E.txt(top, p41[0], p41[1] - 20, '(−4, 1)', 22, E.GOOD, 'middle', 900));
      var ok41 = E.check(top, p41[0] + 40, p41[1] - 40, 22, E.GOOD, 4);
      var ghost = E.hand2(top, P, 0.9, { dash: '5 4', w: 3.5, fill: 'none' });
      ghost.set(-4, 1, Math.PI);
      var hand = E.hand2(top, P, 0.9);
      // 句 10：肘从 (1, 2) 转到 (−2, 1)；小臂 2 格
      var eTrack = K.paint(E.el('path', { d: E.poly(arcPts([0, 0], [1, 2], 0, R0)), fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '7 6' }, top), null, E.HI);
      var e12 = K.paint(E.el('circle', { cx: P(1, 2)[0], cy: P(1, 2)[1], r: 6, 'stroke-width': 3, fill: 'none' }, top), null, E.HI);
      var e12L = E.halo(E.txt(top, P(1, 2)[0] + 14, P(1, 2)[1] + 6, '(1, 2)', 20, E.HI, 'start', 900));
      var e21L = E.halo(E.txt(top, P(-2, 1)[0] + 14, P(-2, 1)[1] - 14, '(−2, 1)', 20, E.HI, 'start', 900));
      var twoBr = K.paint(E.el('path', { d: 'M ' + P(-2.1, 1)[0] + ' ' + (P(-2, 1)[1] + 24) + ' v 6 H ' + P(-3.9, 1)[0] + ' v -6', fill: 'none', 'stroke-width': 2.5 }, top), null, E.INK);
      var twoL = E.halo(E.txt(top, P(-3, 1)[0], P(-3, 1)[1] + 52, '2 格', 18, E.INK, 'middle', 900));
      var degL = E.halo(E.txt(top, P(-3.2, 2.6)[0], P(-3.2, 2.6)[1], '90° + 90° = 180°', 20, E.OMEGA, 'middle', 900));

      /* ══ 右栏 ══ */
      var RK = E.rowKit();
      var MY = 88, mG = E.el('g', {}, s);
      var BM = [[0, -1, 0], [1, 0, 0], [0, 0, 1]], TM = [[0, -1, 1], [1, 0, 4], [0, 0, 1]];
      var colC = function (i, j) { return i === 2 ? E.MUT : j === 0 ? E.X : j === 1 ? E.Y : E.VEL; };
      var fl = E.flow(X - 12, MY, 10);
      var Bm = fl.mat(mG, [['?', '?', '?'], ['?', '?', '?'], [0, 0, 1]], { size: 22, dx: 42, dy: 30, wide: BM, color: colC });
      Bm.cells.forEach(function (r) { r.forEach(function (n) { n.__col = n.style.fill; }); });
      var tG = E.el('g', {}, mG);
      var Tm = fl.mat(tG, TM, { size: 22, dx: 42, dy: 30, color: colC }, 16);
      var bLab = E.txt(mG, Bm.pos(0, 1)[0], MY - 52, '肩转 90°', 18, E.WARN, 'middle', 900);
      var tLab = E.txt(tG, Tm.pos(0, 1)[0], MY - 52, '刚才的结果', 18, E.GOOD, 'middle', 900);
      var RY = 196, rG = E.el('g', {}, s);
      var rEq = E.txt(rG, X + 4, RY + 8, '=', 26, E.INK, 'start', 400); rEq.style.fontFamily = 'KaTeX_Main, serif';
      var RES = [[-1, 0, -4], [0, -1, 1], [0, 0, 1]];
      var Rm = E.matN(rG, X + 104, RY, [['?', '?', '?'], ['?', '?', '?'], ['?', '?', '?']], { size: 22, dx: 42, dy: 30, wide: RES, color: colC });
      Rm.cells.forEach(function (r) { r.forEach(function (n) { n.__col = n.style.fill; }); });
      var SY = 268;
      function rows(lines) { var g = E.el('g', {}, s); return { g: g, r: lines.map(function (l) { return RK.line(g, X, SY + l[0], l[1], l[2] || 19); }) }; }
      var g1 = rows([[0, [['肩转 90°：', E.INK2], ['朝向 = 第 3 集转 90° 的矩阵', E.OMEGA, 900]]],
        [34, [['位置：轴上的点转完不动', E.INK2]]], [66, [['轴穿过原点 → 原点还在原点：', E.INK2], ['(0, 0)', E.VEL, 900]]]]);
      var g2 = rows([[0, [['肩一转：', E.INK2], ['大臂、肘、小臂和手', E.BODY, 900]]], [32, [['一起绕原点转', E.INK2]]],
        [72, [['乘上刚才的结果：', E.INK2], ['一列一列地算', E.INK, 900]]]]);
      var g3 = rows([[0, [['第 1 列 ', E.INK2], ['(0, 1, 0)', E.X, 900], ['：手的 x 轴', E.INK2]]],
        [32, [['1 × 第 2 列 = ', E.INK2], ['(−1, 0, 0)', E.X, 900]]], [66, [['x 轴转到朝左', E.X, 900]]]]);
      var g4 = rows([[0, [['第 2 列 ', E.INK2], ['(−1, 0, 0)', E.Y, 900], ['：手的 y 轴', E.INK2]]],
        [32, [['−1 × 第 1 列 = ', E.INK2], ['(0, −1, 0)', E.Y, 900]]], [66, [['y 轴转到朝下', E.Y, 900]]]]);
      var g5 = rows([[0, [['第 3 列 ', E.INK2], ['(1, 4, 1)', E.VEL, 900], ['：手的位置', E.INK2]]],
        [30, [['1 × ', E.INK2], ['(0, 1, 0)', E.X, 900], [' = ', E.INK2], ['(0, 1, 0)', E.X, 900]]],
        [58, [['4 × ', E.INK2], ['(−1, 0, 0)', E.Y, 900], [' = ', E.INK2], ['(−4, 0, 0)', E.Y, 900]]],
        [86, [['1 × ', E.INK2], ['(0, 0, 1)', E.VEL, 900], [' = ', E.INK2], ['(0, 0, 1)', E.VEL, 900]]]]);
      var g6 = rows([[0, [['加起来：', E.INK2], ['0 − 4 + 0 = ', E.INK2], ['−4', E.GOOD, 900]]], [30, [['1 + 0 + 0 = ', E.INK2], ['1', E.GOOD, 900]]],
        [58, [['0 + 0 + 1 = ', E.INK2], ['1', E.GOOD, 900]]]]);
      g6.r[1].x = X + 80; g6.r[2].x = X + 80;
      var g7 = rows([[0, [['答案：手停在 ', E.INK2], ['(−4, 1)', E.GOOD, 900]]], [32, [['x 轴', E.X, 900], ['朝左，', E.INK2], ['y 轴', E.Y, 900], ['朝下', E.INK2]]]]);
      var g8 = rows([[0, [['验算朝向：', E.INK2], ['90° + 90° = 180°', E.OMEGA, 900]]], [32, [['x 轴', E.X, 900], ['：朝右 → 朝左', E.INK2]]],
        [64, [['y 轴', E.Y, 900], ['：朝上 → 朝下', E.INK2]]]]);
      var g9 = rows([[0, [['验算位置：肘 ', E.INK2], ['(1, 2) → (−2, 1)', E.HI, 900]]], [32, [['小臂朝左伸出 2 格', E.INK2]]],
        [64, [['−2 − 2 = ', E.INK2], ['−4', E.GOOD, 900], ['：手在 ', E.INK2], ['(−4, 1)', E.GOOD, 900]]]]);

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) { RK.layout(); laid = true; }
          var f = E.fade, b = E.bump;

          /* ── 胳膊：肘一直是 90°；句 8 整条胳膊绕原点转 90° ── */
          var tp = f(st, T.turn[0], T.turn[1] - T.turn[0]), t1 = R0 * tp;
          arm.pose(t1, R0);
          V.clearTint();
          var bu = Math.max(b(st, T.parts[0], 0.7), b(st, T.all, 0.9)), bf = Math.max(b(st, T.parts[2], 0.7), b(st, T.all, 0.9));
          V.tint(UPPER, E.BODY, 0.45 * f(st, T.whole, 0.4) + 0.35 * bu);
          V.tint(RIGID, E.BODY, 0.45 + 0.35 * bf);
          var fk = E.fk2(t1, R0), ec = P(fk.elbow[0], fk.elbow[1]);
          hand.set(fk.hand[0], fk.hand[1], fk.ang);
          var bx = b(st, T.xax, 0.9), by = b(st, T.yax, 0.9), bp = Math.max(b(st, T.pos3, 0.9), b(st, T.parts[3], 0.7));
          hand.ax.line.style.strokeWidth = 4.5 + 2.5 * bx; E.glow(hand.ax.g, E.X, bx > 0.05);
          hand.ay.line.style.strokeWidth = 4.5 + 2.5 * by; E.glow(hand.ay.g, E.Y, by > 0.05);
          hand.dot.setAttribute('r', (6 + 4 * bp).toFixed(2));
          eRing.setAttribute('cx', ec[0].toFixed(1)); eRing.setAttribute('cy', ec[1].toFixed(1)); eDot.setAttribute('cx', ec[0].toFixed(1)); eDot.setAttribute('cy', ec[1].toFixed(1));
          var gs = Math.max(b(st, T.sh, 0.8), b(st, T.thru, 0.8)), ge = b(st, T.parts[1], 0.7);
          sRing.setAttribute('r', (12 + 5 * gs).toFixed(2)); E.glow(sRing, E.HI, gs > 0.05);
          eRing.setAttribute('r', (12 + 5 * ge).toFixed(2)); E.glow(eRing, E.HI, ge > 0.05);
          // 转动的弧：句 8 肩转；句 9 肩、肘各画满 90°
          var aS = Math.max(tp, f(st, T.a90[0], 0.6)), onS = f(st, T.turn[0], 0.2) * (1 - f(st, T.carry, 0.4));
          shArc.set(0, Math.max(1e-3, R0 * aS)); E.op(shArc.g, onS);
          var aE = f(st, T.a90[1], 0.6);
          elArc.g.setAttribute('transform', 'translate(' + ec[0].toFixed(1) + ' ' + ec[1].toFixed(1) + ')');
          elArc.set(t1, t1 + Math.max(1e-3, R0 * aE)); E.op(elArc.g, f(st, T.a90[1], 0.2) * (1 - f(st, T.carry, 0.4)));
          E.op(degL, fr(st, T.deg180) * (1 - f(st, T.carry, 0.4)));

          /* ── 原点上的自由向量 ── */
          var ax1 = R0 * f(st, T.left[0], T.left[1] - T.left[0]), ay1 = R0 * f(st, T.down[0], T.down[1] - T.down[0]);
          var axA = R0 + ax1, ayA = Math.PI + ay1;   // x 轴从朝上转到朝左，y 轴从朝左转到朝下
          vx.set(O[0], O[1], O[0] + Math.cos(axA) * u, O[1] - Math.sin(axA) * u);
          vy.set(O[0], O[1], O[0] + Math.cos(ayA) * u, O[1] - Math.sin(ayA) * u);
          var onX = f(st, T.c1v, 0.3) * (1 - f(st, T.c2 - 0.2, 0.4)), onY = f(st, T.c2v, 0.3) * (1 - f(st, T.c3, 0.4));
          E.op(vx.g, onX); E.op(vy.g, onY);
          arcX.set(R0, R0 + Math.max(1e-3, ax1)); E.op(arcX.g, f(st, T.left[0], 0.2) * (1 - f(st, T.c2 - 0.2, 0.4)));
          arcY.set(Math.PI, Math.PI + Math.max(1e-3, ay1)); E.op(arcY.g, f(st, T.down[0], 0.2) * (1 - f(st, T.c3, 0.4)));
          var tx = [O[0] + Math.cos(axA) * u, O[1] - Math.sin(axA) * u], ty = [O[0] + Math.cos(ayA) * u, O[1] - Math.sin(ayA) * u];
          vxLab.setAttribute('x', (tx[0] + (ax1 > 0.8 ? -48 : 10)).toFixed(1)); vxLab.setAttribute('y', (tx[1] + (ax1 > 0.8 ? -12 : 4)).toFixed(1));
          vyLab.setAttribute('x', (ty[0] + (ay1 > 0.8 ? 10 : -48)).toFixed(1)); vyLab.setAttribute('y', (ty[1] + (ay1 > 0.8 ? 4 : -12)).toFixed(1));
          E.op(vxLab, onX); E.op(vyLab, onY);

          /* ── 第 3 列 ── */
          var on5 = 1 - f(st, T.ans, 0.4);
          E.op(p14, f(st, T.c3v, 0.3) * on5);
          seg(cA, [0, 0], [0, 1], f(st, T.t1, 0.5)); E.op(cA.g, (f(st, T.t1, 0.1) > 0 ? 1 : 0) * on5);
          seg(cB, [0, 1], [-4, 1], f(st, T.t2, 0.8)); E.op(cB.g, (f(st, T.t2, 0.1) > 0 ? 1 : 0) * on5);
          E.op(cAL, fr(st, T.t1r) * on5); E.op(cBL, fr(st, T.t2r) * on5); E.op(pZero, fr(st, T.t3r) * (1 - f(st, T.s3r + 0.3, 0.3)));
          E.op(d41, f(st, T.s1r, 0.3) * (1 - f(st, T.turn[1] - 0.2, 0.3))); d41.setAttribute('r', (7 + 4 * b(st, T.s3r + 0.3, 0.9)).toFixed(2));
          E.op(l41, f(st, T.ans, 0.3));
          E.op(ghost.ax.g, f(st, T.gx, 0.3) * 0.9 * (1 - f(st, T.chk, 0.4))); E.op(ghost.ay.g, f(st, T.gy, 0.3) * 0.9 * (1 - f(st, T.chk, 0.4)));
          E.op(ghost.dot, f(st, T.stop, 0.3) * (1 - f(st, T.turn[1] - 0.2, 0.3)));

          /* ── 句 10：肘的轨迹、2 格 ── */
          var ep = f(st, T.arc[0], T.arc[1] - T.arc[0]);
          E.drawOn(eTrack, ep); E.op(eTrack, ep > 0 ? 0.9 : 0);
          E.op(e12, fr(st, T.from12)); E.op(e12L, fr(st, T.from12)); E.op(e21L, fr(st, T.to21));
          E.op(twoBr, fr(st, T.two)); E.op(twoL, fr(st, T.two));
          E.drawOn(ok41, f(st, T.ok, 0.3)); E.op(ok41, f(st, T.ok, 0.1) > 0 ? 1 : 0);

          /* ── 右栏：矩阵 ── */
          E.op(mG, fr(st, T.sh));
          var fill = [[0, 0, '0', T.r90], [1, 0, '1', T.r90 + 0.2], [0, 1, '−1', T.r90 + 0.4], [1, 1, '0', T.r90 + 0.6], [0, 2, '0', T.z1], [1, 2, '0', T.z2]];
          fill.forEach(function (c) {
            var n = Bm.cells[c[0]][c[1]], q = Bm.pos(c[0], c[1]), p = f(st, c[3], 0.3);
            n.textContent = st >= c[3] ? c[2] : '?';
            E.scaleAt(n, q[0], q[1], st >= c[3] ? 1 + 0.25 * (1 - p) : 1);
            n.style.fill = st >= c[3] ? n.__col : E.MUT;
          });
          Bm.hl('rot', 0, 0, 2, 1, E.OMEGA, f(st, T.rot, 0.3) * (1 - f(st, T.pos, 0.3)));
          Bm.hl('pos', 0, 2, 2, 2, E.VEL, f(st, T.pos, 0.3) * (1 - f(st, T.whole, 0.3)));
          E.op(tG, fr(st, T.mul)); E.op(tLab, fr(st, T.prev));
          var cur = st >= T.c3 ? 2 : st >= T.c2 ? 1 : st >= T.c1 ? 0 : -1;
          [0, 1, 2].forEach(function (j) {
            var col = [E.X, E.Y, E.VEL][j];
            Tm.hl('m' + j, 0, j, 2, j, col, Math.max(b(st, T.cols[j], 0.6), cur === j ? f(st, [T.c1, T.c2, T.c3][j], 0.3) * (1 - f(st, T.ans, 0.3)) : 0));
          });
          Bm.hl('b0', 0, 0, 2, 0, E.X, Math.max(cur === 1 ? f(st, T.take2, 0.3) * (1 - f(st, T.r2 + 0.6, 0.3)) : 0, cur === 2 ? f(st, T.t1, 0.3) * (1 - f(st, T.t2, 0.3)) : 0));
          Bm.hl('b1', 0, 1, 2, 1, E.Y, Math.max(cur === 0 ? f(st, T.take1, 0.3) * (1 - f(st, T.r1 + 0.6, 0.3)) : 0, cur === 2 ? f(st, T.t2, 0.3) * (1 - f(st, T.t3, 0.3)) : 0));
          Bm.hl('b2', 0, 2, 2, 2, E.VEL, cur === 2 ? f(st, T.t3, 0.3) * (1 - f(st, T.add, 0.3)) : 0);
          E.op(rG, fr(st, T.c1));
          var rf = [[0, 0, '−1', T.r1], [1, 0, '0', T.r1 + 0.3], [2, 0, '0', T.r1 + 0.5], [0, 1, '0', T.r2], [1, 1, '−1', T.r2 + 0.3], [2, 1, '0', T.r2 + 0.5],
            [0, 2, '−4', T.s1r], [1, 2, '1', T.s2r], [2, 2, '1', T.s3r]];
          rf.forEach(function (c) {
            var n = Rm.cells[c[0]][c[1]], q = Rm.pos(c[0], c[1]), p = f(st, c[3], 0.3);
            n.textContent = st >= c[3] ? c[2] : '?';
            E.scaleAt(n, q[0], q[1], st >= c[3] ? 1 + 0.25 * (1 - p) : 1);
            n.style.fill = st >= c[3] ? n.__col : E.MUT;
          });
          Rm.hl('all', 0, 0, 2, 2, E.HI, b(st, T.ans, 1.0));
          [0, 1, 2].forEach(function (j) { Rm.hl('r' + j, 0, j, 2, j, [E.X, E.Y, E.VEL][j], cur === j ? f(st, [T.r1, T.r2, T.s1r][j], 0.3) * (1 - f(st, [T.c2, T.c3, T.ans][j], 0.3)) : 0); });

          /* ── 右栏：算式 ── */
          E.op(g1.g, fr(st, T.rot) * (1 - f(st, T.whole - 0.3, 0.3)));
          E.op(g1.r[1].g, fr(st, T.pos)); E.op(g1.r[2].g, fr(st, T.thru)); E.op(g1.r[2].parts[1], fr(st, T.z1));
          E.op(g2.g, fr(st, T.whole) * (1 - f(st, T.c1 - 0.3, 0.3)));
          E.op(g2.r[0].parts[1], fr(st, T.parts[0])); E.op(g2.r[1].g, fr(st, T.all)); E.op(g2.r[2].g, fr(st, T.mul)); E.op(g2.r[2].parts[1], fr(st, T.cols[0]));
          E.op(g3.g, fr(st, T.c1) * (1 - f(st, T.c2 - 0.3, 0.3)));
          E.op(g3.r[0].parts[1], fr(st, T.c1v)); E.op(g3.r[0].parts[2], fr(st, T.xax)); E.op(g3.r[1].g, fr(st, T.take1)); E.op(g3.r[1].parts[1], fr(st, T.r1)); E.op(g3.r[2].g, fr(st, T.left[0]));
          E.op(g4.g, fr(st, T.c2) * (1 - f(st, T.c3 - 0.3, 0.3)));
          E.op(g4.r[0].parts[1], fr(st, T.c2v)); E.op(g4.r[0].parts[2], fr(st, T.yax)); E.op(g4.r[1].g, fr(st, T.take2)); E.op(g4.r[1].parts[1], fr(st, T.r2)); E.op(g4.r[2].g, fr(st, T.down[0]));
          E.op(g5.g, fr(st, T.c3) * (1 - f(st, T.add - 0.3, 0.3)));
          E.op(g5.r[0].parts[1], fr(st, T.c3v));
          [[T.t1, T.t1r], [T.t2, T.t2r], [T.t3, T.t3r]].forEach(function (tt, k) { var r = g5.r[k + 1]; E.op(r.g, fr(st, tt[0])); E.op(r.parts[2], fr(st, tt[1])); E.op(r.parts[3], fr(st, tt[1])); });
          E.op(g6.g, fr(st, T.add) * (1 - f(st, T.ans - 0.3, 0.3)));
          E.op(g6.r[0].parts[1], fr(st, T.s1)); E.op(g6.r[0].parts[2], fr(st, T.s1r)); E.op(g6.r[1].g, fr(st, T.s2)); E.op(g6.r[1].parts[1], fr(st, T.s2r));
          E.op(g6.r[2].g, fr(st, T.s3)); E.op(g6.r[2].parts[1], fr(st, T.s3r));
          E.op(g7.g, fr(st, T.ans) * (1 - f(st, T.chk - 0.3, 0.3)));
          E.op(g7.r[0].parts[1], fr(st, T.stop)); E.op(g7.r[1].g, fr(st, T.gx)); E.op(g7.r[1].parts[2], fr(st, T.gy)); E.op(g7.r[1].parts[3], fr(st, T.gy));
          E.op(g8.g, fr(st, T.chk) * (1 - f(st, T.carry - 0.3, 0.3)));
          E.op(g8.r[0].parts[1], fr(st, T.a90[0])); E.op(g8.r[1].g, fr(st, T.cx)); E.op(g8.r[2].g, fr(st, T.cy));
          E.op(g9.g, fr(st, T.carry));
          E.op(g9.r[0].parts[1], fr(st, T.from12)); E.op(g9.r[1].g, fr(st, T.fore)); E.op(g9.r[2].g, fr(st, T.calc)); E.op(g9.r[2].parts[1], fr(st, T.calcR));
          E.op(g9.r[2].parts[2], fr(st, T.ok)); E.op(g9.r[2].parts[3], fr(st, T.ok));
        }
      };
    }
  };
})();
