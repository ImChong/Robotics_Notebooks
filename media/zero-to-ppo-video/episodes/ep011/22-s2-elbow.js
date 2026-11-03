// 011 第 2 幕：先转肘。左边同第 1 幕的网格 + UR5；右栏上方一行公式 T = exp([S₁]θ₁) exp([S₂]θ₂) M（三块各有底框），
// 中间 exp([S₂]θ₂) × M 两个手画的 3 × 3（列按 x 红、y 绿、位置紫上色），下面「= 结果」先是问号、按列填数，最下面几行算式。
// 句 1「公式」三块依次出来，「肩」「肘」「位姿」下面写名字；句 2「从右往左」箭头从右往左画，「先算」框住后两块，「肘」那块亮。
// 句 3「肘转九十度的矩阵指数」两个矩阵出来，「第十集第三幕」右栏写出处，「左边两列」框蓝，「位置这一列」框紫。
// 句 4「矩阵乘矩阵」「每一列」M 的三列轮流亮一下；「按列拿份」写一行。
// 句 5–6 第 1、2 列：M 的这一列、手的这根小箭头亮；网格原点上画一根同色的小箭头（自由向量，画在哪都一样），「第一列拿一份」A 的那一列亮，
//    结果那一列填数，「转到了朝上 / 朝左」原点的小箭头画满 90° 的蓝弧转过去；「位置那一列拿了零份」右栏写「只转、不平移」（A 的第三列不框：没参与计算）。
// 句 7 第 3 列 (3, 2, 1)：(3, 2) 出空心紫圈；「第一列拿三份」从原点往上画红的 (0, 3)，「第二列拿两份」接着往左画绿的 (−2, 0)，
//    「第三列拿一份」再接紫的 (3, 1)，头尾相接停在 (1, 4)（同 010 第 3 幕的验算）；句 8 逐项相加，结果第三列填 1、4、1，(1, 4) 出点。
// 句 9「三列拼起来」结果亮；「手停在一、四」(1, 4) 出虚线的手坐标系（x 朝上、y 朝左）；「小臂绕着肘」小臂绕肘转 90°（画满的蓝弧），
//    手的坐标系跟着走，正好落在虚线上。
// 时刻取自配音的逐词时间，见 ../ep011.py。
(function () {
  var E = window.EP011, K = E.K, H = E.HEX;
  var D = Math.PI / 180, A = E.ARM, RX = A.RX, X = RX + 10;
  var T = {
    f: 0.7, fS: 2.23, fE: 4.18, fM: 6.27, rl: [9.21, 9.9], first: 10.4, near: 11.14, elb: 13.69, why: 14.43, ep4: 16.02,
    A: 18.13, ep10: 20.25, rot: 21.97, pos: 24.93, p31: 26.1,
    mm: 28.15, mat: 31.1, cols: [33.84, 34.3, 34.76], share: 36.35,
    c1: 39.42, xax: 40.6, c1v: 41.21, take1: 42.53, zero23: 43.59, r1: 45.79, up: [47.44, 48.44],
    posz: 50.04, only: 52.75, outR: 53.95, c2: 54.53, yax: 55.68, c2v: 56.32, take2: 57.69, r2: 59.04, left: [60.78, 61.75],
    c3: 63.37, pos3: 64.74, c3v: 65.28, t1: 66.52, t1r: 67.66, t2: 69.31, t2r: 70.38, t3: 72.01, t3r: 73.24,
    add: 75.93, s1: 76.61, s1r: 77.86, s2: 78.27, s2r: 79.41, s3: 80.08, s3r: 81.16, at14: 82.09,
    glue: 84.68, done: 85.8, stop: 86.61, gx: 88.22, gy: 89.26, grid: 90.72, turn: [92.08, 94.44]
  };
  var END = 95.7;

  E.scenes[1] = {
    title: '先转肘',
    dur: END,
    cues: [
      { at: 0, s: '**指数积公式**：T = exp([S₁]θ₁) · exp([S₂]θ₂) · M（肩、肘、**零位**时手的**位姿**）' },
      { at: 9.11, s: '从右往左算：先算最靠近 M 的那一个，也就是肘（为什么，第 4 幕讲）' },
      { at: 18.03, s: '肘转 90° 的**矩阵指数**（第 10 集第 3 幕）：左边两列是转 90° 的矩阵，**位置**一列 (3, 1)' },
      { at: 28.05, s: '矩阵 × 矩阵：右边矩阵的每一列各算一次，照第 3 集按列拿份' },
      { at: 39.32, s: '第 1 列 = 手的 x 轴 (1, 0, 0)：拿 1 份第 1 列 = (0, 1, 0)，x 轴朝上' },
      { at: 49.94, s: '**位置**一列拿 0 份：只转、不平移。第 2 列 = y 轴 (0, 1, 0) → (−1, 0, 0)，朝左' },
      { at: 63.27, s: '第 3 列 = 手的**位置** (3, 2, 1)：3 × (0, 1, 0) + 2 × (−1, 0, 0) + 1 × (3, 1, 1)' },
      { at: 75.83, s: '0 − 2 + 3 = 1，3 + 0 + 1 = 4，0 + 0 + 1 = 1：手到了 (1, 4)' },
      { at: 84.58, s: '肘转完：手停在 (1, 4)，x 轴朝上，y 轴朝左 = 小臂绕肘转了 90°' }
    ],
    build: function (K) {
      var s = K.sceneSvg('先转肘：指数积公式 T = exp([S₁]θ₁) exp([S₂]θ₂) M 从右往左算，先算最靠近 M 的肘。肘转九十度的矩阵指数第十集算过：左边两列是转九十度的矩阵，位置一列是 (3, 1)。拿它乘 M，右边矩阵每一列各算一次：第一列手的 x 轴 (1, 0, 0) 拿一份第一列得 (0, 1, 0)，x 轴朝上，位置那一列拿零份所以只转不平移；第二列手的 y 轴 (0, 1, 0) 得 (−1, 0, 0)，朝左；第三列手的位置 (3, 2, 1)：3 × (0, 1, 0) + 2 × (−1, 0, 0) + 1 × (3, 1, 1) = (1, 4, 1)。三列拼起来，手停在 (1, 4)，x 轴朝上、y 轴朝左，正是小臂绕肘往上转了九十度');

      var under = E.el('g', {}, s), armG = E.el('g', {}, s);
      var arm = E.arm2(armG, 320), u = arm.u, V = arm.view;
      var P = E.plane(under, { ox: A.OX, oy: A.OY, u: u, x0: A.X0, x1: A.X1, y0: A.Y0, y1: A.Y1 }).P;
      var RIGID = ['forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link'];
      var top = E.el('g', {}, s), O = P(0, 0), Q = P(1, 2);
      function seg(vec, a, b, p) { var A0 = P(a[0], a[1]), B0 = P(a[0] + (b[0] - a[0]) * p, a[1] + (b[1] - a[1]) * p); vec.set(A0[0], A0[1], B0[0], B0[1]); E.op(vec.g, p > 0 ? 1 : 0); }
      // 肘的 ⊙
      var ring = K.paint(E.el('circle', { cx: Q[0], cy: Q[1], r: 12, 'stroke-width': 3.5 }, top), E.S1, E.HI);
      K.paint(E.el('circle', { cx: Q[0], cy: Q[1], r: 4 }, top), E.HI);
      var elbArc = E.arcArrow(top, Q[0], Q[1], 0.8 * u, E.OMEGA, 3.5, { head: 12 });
      // 原点上的自由向量（x 轴红、y 轴绿）和转 90° 的蓝弧
      var vx = E.vec(top, E.X, 5, { head: 14 }), vy = E.vec(top, E.Y, 5, { head: 14 });
      var arcX = E.arcArrow(top, O[0], O[1], 0.62 * u, E.OMEGA, 3, { head: 11 }), arcY = E.arcArrow(top, O[0], O[1], 0.62 * u, E.OMEGA, 3, { head: 11 });
      var vxLab = E.halo(E.txt(top, 0, 0, 'x 轴', 20, E.X, 'start', 900)), vyLab = E.halo(E.txt(top, 0, 0, 'y 轴', 20, E.Y, 'start', 900));
      // 第 3 列：按列拿份，头尾相接
      var p32 = K.paint(E.el('circle', { cx: P(3, 2)[0], cy: P(3, 2)[1], r: 8, 'stroke-width': 3, fill: 'none' }, top), null, E.VEL);
      var cA = E.vec(top, E.X, 4.5, { head: 13 }), cB = E.vec(top, E.Y, 4.5, { head: 13 }), cC = E.vec(top, E.VEL, 4.5, { head: 13 });
      var cAL = E.halo(E.txt(top, P(0, 1.5)[0] + 10, P(0, 1.5)[1] + 4, '3 份', 18, E.X, 'start', 900));
      var cBL = E.halo(E.txt(top, P(-1, 3)[0], P(-1, 3)[1] - 10, '2 份', 18, E.Y, 'middle', 900));
      var p14 = P(1, 4), d14 = K.paint(E.el('circle', { cx: p14[0], cy: p14[1], r: 7, 'stroke-width': 3 }, top), E.VEL, E.S1);
      var l14 = E.halo(E.txt(top, p14[0] + 16, p14[1] + 26, '(1, 4)', 22, E.GOOD, 'start', 900));
      // 结果：虚线的手坐标系；真正的手坐标系
      var ghost = E.hand2(top, P, 0.9, { dash: '5 4', w: 3.5, fill: 'none' });
      ghost.set(1, 4, Math.PI / 2);
      var hand = E.hand2(top, P, 0.9);

      /* ══ 右栏 ══ */
      var RK = E.rowKit(), FY = 40;
      var fG = E.el('g', {}, s);
      E.tex(fG, X, FY, 'T=', 24, E.INK, 'start', { w: 40, h: 40 });
      function factor(x, w, tex, lab, col) {
        var g = E.el('g', {}, fG);
        var box = K.paint(E.el('rect', { x: x - 6, y: FY - 27, width: w, height: 38, rx: 8, 'stroke-width': 2 }, g), E.tint(col, 14, 'var(--demo-surface)'), col);
        E.tex(g, x, FY, tex, 21, E.INK, 'start', { w: w - 8, h: 40 });
        var l = E.txt(g, x + w / 2 - 6, FY + 34, lab, 18, col, 'middle', 900);
        return { g: g, box: box, lab: l, x: x, w: w };
      }
      var F1 = factor(X + 44, 124, '\\exp([\\mathcal S_1]\\,\\theta_1)', '肩', E.WARN);
      var F2 = factor(X + 176, 124, '\\exp([\\mathcal S_2]\\,\\theta_2)', '肘', E.WARN);
      var F3 = factor(X + 308, 34, 'M', '零位', E.GOOD);
      var rl = E.vec(fG, E.INK2, 3, { head: 11 });
      var rlLab = E.txt(fG, X + 160, FY + 80, '从右往左', 18, E.INK2, 'middle', 900);
      var firstBox = K.paint(E.el('path', { d: E.poly([[X + 164, FY - 33], [X + 340, FY - 33], [X + 340, FY + 41], [X + 164, FY + 41]], true), 'stroke-width': 2.5, fill: 'none', 'stroke-dasharray': '7 5' }, fG), null, E.HI);
      var why = E.txt(fG, X + 340, FY + 80, '为什么？第 4 幕', 16, E.MUT, 'end', 700);
      // 乘法：exp([S₂]θ₂) × M
      var MY = 88, mG = E.el('g', {}, s);
      var AM = [[0, -1, 3], [1, 0, 1], [0, 0, 1]], MM = [[1, 0, 3], [0, 1, 2], [0, 0, 1]];
      var colC = function (i, j) { return i === 2 ? E.MUT : j === 0 ? E.X : j === 1 ? E.Y : E.VEL; };
      var fl = E.flow(X - 12, MY, 10);
      var Am = fl.mat(mG, AM, { size: 22, dx: 42, dy: 30, color: colC });
      var Mm = fl.mat(mG, MM, { size: 22, dx: 42, dy: 30, color: colC }, 16);
      var aLab = E.txt(mG, Am.pos(0, 1)[0], MY - 52, '肘转 90°', 18, E.WARN, 'middle', 900);
      var mLab = E.txt(mG, Mm.pos(0, 1)[0], MY - 52, '零位时、手的位姿', 17, E.GOOD, 'middle', 900);
      var RY = 196, rG = E.el('g', {}, s);
      var rEq = E.txt(rG, X + 4, RY + 8, '=', 26, E.INK, 'start', 400); rEq.style.fontFamily = 'KaTeX_Main, serif';
      var Rm = E.matN(rG, X + 100, RY, [['?', '?', '?'], ['?', '?', '?'], ['?', '?', '?']], { size: 22, dx: 42, dy: 30, wide: [[0, -1, 1], [1, 0, 4], [0, 0, 1]], color: colC });
      Rm.cells.forEach(function (r) { r.forEach(function (n) { n.__col = n.style.fill; }); });
      var resLab = E.txt(rG, X + 172, RY + 8, '', 20, E.HI, 'start', 900);
      // 算式：几组，轮流出
      var SY = 268;
      function rows(lines) { var g = E.el('g', {}, s); return { g: g, r: lines.map(function (l) { return RK.line(g, X, SY + l[0], l[1], l[2] || 19); }) }; }
      var g2 = rows([[0, [['肘转 90° 的矩阵指数：', E.INK2], ['第 10 集第 3 幕', E.INK, 900]]],
        [34, [['左边两列：', E.INK2], ['转 90° 的矩阵', E.OMEGA, 900]]], [66, [['位置这一列：', E.INK2], ['(3, 1)', E.VEL, 900]]]]);
      var g3 = rows([[0, [['矩阵 × 矩阵：', E.INK2], ['右边每一列，各算一次', E.INK, 900]]], [34, [['每一列：照第 3 集，', E.INK2], ['按列拿份', E.INK, 900]]]]);
      var g4 = rows([[0, [['第 1 列 ', E.INK2], ['(1, 0, 0)', E.X, 900], ['：手的 x 轴', E.INK2]]],
        [32, [['1 × 第 1 列 + 0 + 0 = ', E.INK2], ['(0, 1, 0)', E.X, 900]]], [66, [['x 轴转到朝上', E.X, 900]]]]);
      var g5 = rows([[0, [['位置那一列拿 0 份：', E.INK2], ['只转、不平移', E.INK, 900]]],
        [36, [['第 2 列 ', E.INK2], ['(0, 1, 0)', E.Y, 900], ['：手的 y 轴', E.INK2]]],
        [68, [['1 × 第 2 列 = ', E.INK2], ['(−1, 0, 0)', E.Y, 900], ['：朝左', E.Y, 900]]]]);
      var g6 = rows([[0, [['第 3 列 ', E.INK2], ['(3, 2, 1)', E.VEL, 900], ['：手的位置', E.INK2]]],
        [30, [['3 × ', E.INK2], ['(0, 1, 0)', E.X, 900], [' = ', E.INK2], ['(0, 3, 0)', E.X, 900]]],
        [58, [['2 × ', E.INK2], ['(−1, 0, 0)', E.Y, 900], [' = ', E.INK2], ['(−2, 0, 0)', E.Y, 900]]],
        [86, [['1 × ', E.INK2], ['(3, 1, 1)', E.VEL, 900], [' = ', E.INK2], ['(3, 1, 1)', E.VEL, 900]]]]);
      var g7 = rows([[0, [['加起来：', E.INK2], ['0 − 2 + 3 = ', E.INK2], ['1', E.GOOD, 900]]], [30, [['3 + 0 + 1 = ', E.INK2], ['4', E.GOOD, 900]]],
        [58, [['0 + 0 + 1 = ', E.INK2], ['1', E.GOOD, 900]]], [88, [['手到了 ', E.INK2], ['(1, 4)', E.GOOD, 900]]]]);
      g7.r[1].x = X + 80; g7.r[2].x = X + 80;
      var g8 = rows([[0, [['手停在 ', E.INK2], ['(1, 4)', E.GOOD, 900]]], [32, [['x 轴', E.X, 900], ['朝上，', E.INK2], ['y 轴', E.Y, 900], ['朝左', E.INK2]]],
        [72, [['= 小臂绕着肘，往上转 90°', E.OMEGA, 900]], 20]]);

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) { RK.layout(); laid = true; }
          var f = E.fade, b = E.bump;

          /* ── 胳膊：句 9 小臂绕肘转 90° ── */
          var tp = f(st, T.turn[0], T.turn[1] - T.turn[0]), t2 = 90 * D * tp;
          arm.pose(0, t2);
          V.clearTint(); V.tint(RIGID, E.BODY, 0.45 + 0.3 * b(st, T.fE, 0.8));
          var fk = E.fk2(0, t2);
          hand.set(fk.hand[0], fk.hand[1], fk.ang);
          var bx = b(st, T.xax, 0.9), by = b(st, T.yax, 0.9), bp = b(st, T.pos3, 0.9);
          hand.ax.line.style.strokeWidth = 4.5 + 2.5 * bx; E.glow(hand.ax.g, E.X, bx > 0.05);
          hand.ay.line.style.strokeWidth = 4.5 + 2.5 * by; E.glow(hand.ay.g, E.Y, by > 0.05);
          hand.dot.setAttribute('r', (6 + 4 * bp).toFixed(2));
          var ge = Math.max(b(st, T.fE, 0.8), b(st, T.elb, 0.8), b(st, T.A, 0.8));
          ring.setAttribute('r', (12 + 5 * ge).toFixed(2)); E.glow(ring, E.HI, ge > 0.05);
          elbArc.set(0, Math.max(1e-3, t2)); E.op(elbArc.g, f(st, T.turn[0], 0.2));

          /* ── 原点上的自由向量：x 轴、y 轴各转 90° ── */
          var ax1 = 90 * D * f(st, T.up[0], T.up[1] - T.up[0]), ay1 = 90 * D * f(st, T.left[0], T.left[1] - T.left[0]);
          vx.set(O[0], O[1], O[0] + Math.cos(ax1) * u, O[1] - Math.sin(ax1) * u);
          vy.set(O[0], O[1], O[0] + Math.cos(Math.PI / 2 + ay1) * u, O[1] - Math.sin(Math.PI / 2 + ay1) * u);
          var onX = f(st, T.c1v, 0.3) * (1 - f(st, T.outR, 0.4)), onY = f(st, T.c2v, 0.3) * (1 - f(st, T.c3, 0.4));
          E.op(vx.g, onX); E.op(vy.g, onY);
          arcX.set(0, Math.max(1e-3, ax1)); E.op(arcX.g, f(st, T.up[0], 0.2) * (1 - f(st, T.outR, 0.4)));
          arcY.set(Math.PI / 2, Math.PI / 2 + Math.max(1e-3, ay1)); E.op(arcY.g, f(st, T.left[0], 0.2) * (1 - f(st, T.c3, 0.4)));
          var tx = [O[0] + Math.cos(ax1) * u, O[1] - Math.sin(ax1) * u], ty = [O[0] + Math.cos(Math.PI / 2 + ay1) * u, O[1] - Math.sin(Math.PI / 2 + ay1) * u];
          vxLab.setAttribute('x', (tx[0] + (ax1 > 0.8 ? 10 : 6)).toFixed(1)); vxLab.setAttribute('y', (tx[1] + (ax1 > 0.8 ? 4 : 26)).toFixed(1));
          vyLab.setAttribute('x', (ty[0] + (ay1 > 0.8 ? -48 : 10)).toFixed(1)); vyLab.setAttribute('y', (ty[1] + (ay1 > 0.8 ? -12 : 4)).toFixed(1));
          E.op(vxLab, onX); E.op(vyLab, onY);

          /* ── 第 3 列：按列拿份 ── */
          var on6 = 1 - f(st, T.glue, 0.4);
          E.op(p32, f(st, T.c3v, 0.3) * on6);
          seg(cA, [0, 0], [0, 3], f(st, T.t1, 0.6)); E.op(cA.g, (f(st, T.t1, 0.1) > 0 ? 1 : 0) * on6);
          seg(cB, [0, 3], [-2, 3], f(st, T.t2, 0.6)); E.op(cB.g, (f(st, T.t2, 0.1) > 0 ? 1 : 0) * on6);
          seg(cC, [-2, 3], [1, 4], f(st, T.t3, 0.7)); E.op(cC.g, (f(st, T.t3, 0.1) > 0 ? 1 : 0) * on6);
          E.op(cAL, fr(st, T.t1r) * on6); E.op(cBL, fr(st, T.t2r) * on6);
          E.op(d14, f(st, T.at14, 0.3) * (1 - f(st, T.turn[1] - 0.2, 0.3))); E.op(l14, f(st, T.at14, 0.3));
          d14.setAttribute('r', (7 + 4 * b(st, T.at14, 0.9)).toFixed(2));
          E.op(ghost.ax.g, f(st, T.gx, 0.3) * 0.9); E.op(ghost.ay.g, f(st, T.gy, 0.3) * 0.9); E.op(ghost.dot, f(st, T.stop, 0.3) * (1 - f(st, T.turn[1] - 0.2, 0.3)));

          /* ── 右栏：公式 ── */
          E.op(fG, fr(st, T.f) * (1 - f(st, T.A - 0.5, 0.4)));
          E.op(F1.g, fr(st, T.f)); E.op(F2.g, fr(st, T.f + 0.2)); E.op(F3.g, fr(st, T.f + 0.4));
          E.op(F1.lab, fr(st, T.fS)); E.op(F2.lab, fr(st, T.fE)); E.op(F3.lab, fr(st, T.fM));
          [F1, F2, F3].forEach(function (F, k) {
            var hl = k === 1 ? Math.max(b(st, T.fE, 0.8), b(st, T.elb, 0.9)) : k === 0 ? b(st, T.fS, 0.8) : b(st, T.fM, 0.8);
            F.box.style.strokeWidth = 2 + 2 * hl; E.glow(F.box, E.HI, hl > 0.05);
          });
          var rp = f(st, T.rl[0], T.rl[1] - T.rl[0]);
          rl.set(X + 330, FY + 58, X + 330 - 300 * rp, FY + 58); E.op(rl.g, (rp > 0 ? 1 : 0) * (1 - f(st, T.A - 0.3, 0.4)));
          E.op(rlLab, fr(st, T.rl[0]) * (1 - f(st, T.A - 0.3, 0.4)));
          E.drawOn(firstBox, f(st, T.first, 0.6)); E.op(firstBox, (f(st, T.first, 0.1) > 0 ? 1 : 0) * (1 - f(st, T.A - 0.3, 0.4)));
          E.op(why, fr(st, T.ep4) * (1 - f(st, T.A - 0.3, 0.4)));

          /* ── 右栏：乘法 ── */
          E.op(mG, fr(st, T.A)); E.op(aLab, fr(st, T.ep10)); E.op(mLab, fr(st, T.mm));
          E.op(Mm.g, fr(st, T.mm));
          Am.hl('rot', 0, 0, 2, 1, E.OMEGA, f(st, T.rot, 0.3) * (1 - f(st, T.pos, 0.3)));
          Am.hl('pos', 0, 2, 2, 2, E.VEL, f(st, T.pos, 0.3) * (1 - f(st, T.mm, 0.3)));
          // M 的列：句 4 轮流亮；句 5–7 当前这一列
          var cur = st >= T.c3 ? 2 : st >= T.c2 ? 1 : st >= T.c1 ? 0 : -1;
          [0, 1, 2].forEach(function (j) {
            var col = [E.X, E.Y, E.VEL][j];
            Mm.hl('m' + j, 0, j, 2, j, col, Math.max(b(st, T.cols[j], 0.7), cur === j ? f(st, [T.c1, T.c2, T.c3][j], 0.3) * (1 - f(st, T.glue, 0.3)) : 0));
          });
          // A 的列：只在真拿份的时候亮（2026-11-03 用户反馈：算第 1 列时第 3 列亮灰框没必要——它没参与计算，同样拿 0 份的第 2 列也没框，去掉）
          Am.hl('a0', 0, 0, 2, 0, E.X, Math.max(cur === 0 ? f(st, T.take1, 0.3) * (1 - f(st, T.r1 + 0.6, 0.3)) : 0, cur === 2 ? f(st, T.t1, 0.3) * (1 - f(st, T.t2, 0.3)) : 0));
          Am.hl('a1', 0, 1, 2, 1, E.Y, Math.max(cur === 1 ? f(st, T.take2, 0.3) * (1 - f(st, T.r2 + 0.6, 0.3)) : 0, cur === 2 ? f(st, T.t2, 0.3) * (1 - f(st, T.t3, 0.3)) : 0));
          Am.hl('a2', 0, 2, 2, 2, E.VEL, cur === 2 ? f(st, T.t3, 0.3) * (1 - f(st, T.add, 0.3)) : 0);
          // 结果
          E.op(rG, fr(st, T.c1));
          var fill = [[0, 0, '0', T.r1], [1, 0, '1', T.r1 + 0.3], [2, 0, '0', T.r1 + 0.5], [0, 1, '−1', T.r2], [1, 1, '0', T.r2 + 0.3], [2, 1, '0', T.r2 + 0.5],
            [0, 2, '1', T.s1r], [1, 2, '4', T.s2r], [2, 2, '1', T.s3r]];
          fill.forEach(function (c) {
            var n = Rm.cells[c[0]][c[1]], q = Rm.pos(c[0], c[1]), p = f(st, c[3], 0.3);
            n.textContent = st >= c[3] ? c[2] : '?';
            E.scaleAt(n, q[0], q[1], st >= c[3] ? 1 + 0.25 * (1 - p) : 1);
            n.style.fill = st >= c[3] ? n.__col : E.MUT;
          });
          Rm.hl('all', 0, 0, 2, 2, E.HI, b(st, T.glue, 1.0));
          [0, 1, 2].forEach(function (j) { Rm.hl('r' + j, 0, j, 2, j, [E.X, E.Y, E.VEL][j], cur === j ? f(st, [T.r1, T.r2, T.at14][j], 0.3) * (1 - f(st, [T.c2, T.c3, T.glue][j], 0.3)) : 0); });

          /* ── 右栏：算式 ── */
          E.op(g2.g, fr(st, T.ep10) * (1 - f(st, T.mm - 0.3, 0.3)));
          E.op(g2.r[1].g, fr(st, T.rot)); E.op(g2.r[2].g, fr(st, T.pos)); E.op(g2.r[2].parts[1], fr(st, T.p31));
          E.op(g3.g, fr(st, T.mat) * (1 - f(st, T.c1 - 0.3, 0.3)));
          E.op(g3.r[0].g, fr(st, T.mat)); E.op(g3.r[1].g, fr(st, T.share));
          E.op(g4.g, fr(st, T.c1) * (1 - f(st, T.posz - 0.3, 0.3)));
          E.op(g4.r[0].parts[1], fr(st, T.c1v)); E.op(g4.r[0].parts[2], fr(st, T.xax)); E.op(g4.r[1].g, fr(st, T.take1)); E.op(g4.r[1].parts[1], fr(st, T.r1)); E.op(g4.r[2].g, fr(st, T.up[0]));
          E.op(g5.g, fr(st, T.posz) * (1 - f(st, T.c3 - 0.3, 0.3)));
          E.op(g5.r[0].parts[1], fr(st, T.only)); E.op(g5.r[1].g, fr(st, T.c2)); E.op(g5.r[1].parts[1], fr(st, T.c2v));
          E.op(g5.r[2].g, fr(st, T.take2)); E.op(g5.r[2].parts[1], fr(st, T.r2)); E.op(g5.r[2].parts[2], fr(st, T.left[0]));
          E.op(g6.g, fr(st, T.c3) * (1 - f(st, T.add - 0.3, 0.3)));
          E.op(g6.r[0].parts[1], fr(st, T.c3v));
          [[T.t1, T.t1r], [T.t2, T.t2r], [T.t3, T.t3r]].forEach(function (tt, k) { var r = g6.r[k + 1]; E.op(r.g, fr(st, tt[0])); E.op(r.parts[2], fr(st, tt[1])); E.op(r.parts[3], fr(st, tt[1])); });
          E.op(g7.g, fr(st, T.add) * (1 - f(st, T.glue - 0.3, 0.3)));
          E.op(g7.r[0].parts[1], fr(st, T.s1)); E.op(g7.r[0].parts[2], fr(st, T.s1r)); E.op(g7.r[1].g, fr(st, T.s2)); E.op(g7.r[1].parts[1], fr(st, T.s2r));
          E.op(g7.r[2].g, fr(st, T.s3)); E.op(g7.r[2].parts[1], fr(st, T.s3r)); E.op(g7.r[3].g, fr(st, T.at14));
          E.op(g8.g, fr(st, T.glue));
          E.op(g8.r[0].g, fr(st, T.stop)); E.op(g8.r[1].g, fr(st, T.gx)); E.op(g8.r[1].parts[2], fr(st, T.gy)); E.op(g8.r[1].parts[3], fr(st, T.gy)); E.op(g8.r[2].g, fr(st, T.grid));
        }
      };
    }
  };
})();
