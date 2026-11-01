// 010 第 3 幕：手算——转九十度后的位姿。左边同第 1 幕的网格 + UR5（往左、往下多放两格），右栏上方是 3 × 3 的位姿矩阵 T，下方一行行写算式。
// 2026-11-01 按 009 的用户反馈重做：矩阵乘向量照第 3 集按列拿份、逐项算，每一步网格上同时画出来、头尾相接；
// 配色照「画面配色」：T 的第一列（x）红、第二列（y）绿、位置那一列紫（同 007 的 p），转动蓝，走歪的橙。
// 句 1 右栏「不用真的分无穷多步」「现成的公式」「是一个位姿」；
// 句 2「三乘三的矩阵」T 出来（上两排先是问号，最下面一排 0、0、1），「左边两列是朝向」框蓝、写「朝向」，「右边一列是位置」框紫、写「位置」；
// 句 3「小臂转了九十度」小臂绕肘转到竖直，小臂上 (3, 2) 那一点沿圆弧走到 (1, 4)；「转九十度的矩阵」原点上往右（红）、往上（绿）两根小箭头转 90°，
//    「两列是零、一」第一列填 0、1（红），「和负一、零」第二列填 −1、0（绿）；
// 句 4「轴上的点」「肘就在这根轴上」肘亮一圈，「还在原地」右栏写；
// 句 5「先转」「一、二」从原点画到肘的虚线，「第一列拿一份」从原点往上画红的一段 (0, 1)，「第二列拿两份」接着往左画绿的一段 (−2, 0)，
//    「加起来」虚线绕原点转 90°（蓝弧）落在 (−2, 1)，右栏逐项写出；
// 句 6「再平移」「回到一、二」从 (−2, 1) 画紫箭头到肘，「一减负二是三」先画往右 3 的红段，「二减一是一」再画往上 1 的绿段，「三、一」第三列填 3、1；
// 句 7「拼起来」T 亮一下，「验算」右栏写，「三、二后面补一」(3, 2) 处出空心点，「乘上它」；
// 句 8「第一列拿三份」从原点往上画 3 格红段，「第二列拿两份」接着往左 2 格绿段，「第三列拿一份」再接紫的 (3, 1)，头尾相接停在 (1, 4)；右栏逐项写；
// 句 9「零减二加三是一」「三加零加一是四」「零加零加一是一」右栏逐项写，「一、四、一」小臂上的点鼓一下、写 (1, 4)；
// 句 10「位置这一列」第三列框鼓一下，「透明板上、原来在原点的那一点」原点亮出紫点，「转到了三、一」它沿圆弧走到 (3, 1)；
// 句 11「二、负一」「直接乘角度」从原点沿 (2, −1) 画橙虚线，「二乘一点五七」「负一乘上一点五七」右栏逐项写，「走歪了」打橙 ×；
// 句 12 UR5 淡出：「扫地机」(3, 1) 出现车头朝上的扫地机（第 7 集），「原点」原点出一台车头朝右的，「绕着一、二，转了九十度」它绕肘转到 (3, 1)。
// 时刻取自配音的逐词时间，见 ../ep010.py。
(function () {
  var E = window.EP010, K = E.K, H = E.HEX;
  var D = Math.PI / 180, A = E.ARM, RX = A.RX;
  var T = {
    nope: 0.91, ready: 3.11, pose: 4.82, out0: 6.3,
    mat: 9.56, rot: 10.89, pos: 12.34,
    simple: 14.78, turn: [16.48, 17.52], ep3: 18.21, frameTurn: [18.21, 19.9], col1: 21.07, col2: 21.85, out2: 23.6,
    posRow: 23.87, axisPt: 25.41, elbow: 26.35, stay: 27.66,
    first: 29.95, qq: 30.7, mult: 31.43, take1: 33.7, got1: 34.75, take2: 35.88, got2: 37.0, sum: 38.26, sumR: 38.96,
    shift: 41.23, back: [42.04, 42.97], diff: 43.45, d1: 46.46, d1r: 47.36, d2: 47.83, d2r: 48.23, p31: 50.33, p31b: 50.88, out6: 51.9,
    glue: 52.31, check: 54.59, p321: 55.63, one: 56.75, mult2: 57.33,
    t1: 59.18, t1r: 60.29, t2: 61.9, t2r: 62.95, t3: 64.62, t3r: 65.85,
    add: 68.51, s1: 69.11, s2: 70.45, s3: 71.9, res: 73.16, res2: 73.76, out9: 75.6,
    col: 76.61, plate: 78.59, travel: [81.09, 82.6], lab31: 82.5,
    vth: 85.55, vline: [86.48, 87.4], v1: 87.62, v1r: 88.66, v2: 89.8, v2r: 91.27, wrong: 92.54,
    armOut: 94.3, ep7: 95.45, vac: 97.19, up: 99.3, ghost: 101.03, around: [103.29, 105.38]
  };
  var END = 106.37;

  E.scenes[2] = {
    title: '手算：转九十度后的位姿',
    dur: END,
    cues: [
      { at: 0, s: '不用真的分无穷多步：现成的公式，算出来是一个**位姿**' },
      { at: 6.48, s: '平面上的**位姿**写成 3 × 3：左边两列是**朝向**，右边一列是**位置**' },
      { at: 14.69, s: '**朝向**：转 90° 的矩阵，两列是 (0, 1) 和 (−1, 0)' },
      { at: 23.78, s: '**位置**：肘就在轴上，转完还在原地' },
      { at: 29.86, s: '先转：1 × (0, 1) + 2 × (−1, 0) = (0, 1) + (−2, 0) = (−2, 1)' },
      { at: 41.14, s: '再平移回 (1, 2)：1 − (−2) = 3，2 − 1 = 1，**位置** = (3, 1)' },
      { at: 52.22, s: '验算：T × (3, 2, 1)' },
      { at: 59.09, s: '3 × (0, 1, 0) + 2 × (−1, 0, 0) + 1 × (3, 1, 1)' },
      { at: 68.42, s: '0 − 2 + 3 = 1，3 + 0 + 1 = 4，0 + 0 + 1 = 1：正好是 (1, 4, 1)' },
      { at: 75.84, s: '(3, 1) 是透明板上原来在原点的那一点，绕肘转过去的地方' },
      { at: 84.29, s: '直接乘角度：2 × 1.57 = 3.14，−1 × 1.57 = −1.57，一样走歪了' },
      { at: 94.44, s: '第 7 集的扫地机：停在 (3, 1)、车头朝上 = 绕 (1, 2) 转 90°' }
    ],
    build: function (K) {
      var s = K.sceneSvg('矩阵指数算出来是一个位姿：平面上写成三乘三的矩阵，左边两列是朝向，右边一列是位置；小臂转了九十度，朝向就是第三集转九十度的矩阵，两列是 (0, 1) 和 (−1, 0)；位置这一列让肘转完还在原地：先转，(1, 2) 乘这个矩阵，第一列拿一份 (0, 1)，第二列拿两份 (−2, 0)，加起来 (−2, 1)；再平移回 (1, 2)，1 − (−2) = 3，2 − 1 = 1，位置是 (3, 1)；验算 T 乘 (3, 2, 1)：第一列拿三份 (0, 3, 0)，第二列拿两份 (−2, 0, 0)，第三列拿一份 (3, 1, 1)，加起来 (1, 4, 1)；(3, 1) 是透明板上原来在原点的点绕肘转过去的地方，拿螺旋轴里的 (2, −1) 直接乘角度得 (3.14, −1.57)，走歪了；这个矩阵正是第七集扫地机停在 (3, 1)、车头朝上的位姿');

      var under = E.el('g', {}, s);
      var armG = E.el('g', {}, s);
      var arm = E.armAt(armG, A.OX, A.OY, A.S, 330);
      var u = arm.aim(1, 2, 0);
      var pl = E.plane(under, { ox: A.OX, oy: A.OY, u: u, x0: -2, x1: 4, y0: -2, y1: 5 });
      var P = pl.P, V = arm.view, RIGID = ['forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link'];
      var top = E.el('g', {}, s), Q = P(1, 2), O = P(0, 0);
      function around(c, p0, a) { var d = [p0[0] - c[0], p0[1] - c[1]], cs = Math.cos(a), sn = Math.sin(a); return [c[0] + cs * d[0] - sn * d[1], c[1] + sn * d[0] + cs * d[1]]; }
      function arcPts(c, p0, a0, a1) { var pts = []; for (var k = 0; k <= 40; k++) { var q = around(c, p0, a0 + (a1 - a0) * k / 40); pts.push(P(q[0], q[1])); } return pts; }
      function seg(vec, a, b, p) { var A0 = P(a[0], a[1]), B0 = P(a[0] + (b[0] - a[0]) * p, a[1] + (b[1] - a[1]) * p); vec.set(A0[0], A0[1], B0[0], B0[1]); E.op(vec.g, p > 0 ? 1 : 0); }

      // 小臂上的点：(3, 2) → (1, 4)
      var pArc = K.paint(E.el('path', { d: E.poly(arcPts([1, 2], [3, 2], 0, Math.PI / 2)), fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '7 6' }, top), null, E.VEL);
      var pDot = K.paint(E.el('circle', { r: 7, 'stroke-width': 3 }, top), E.VEL, E.S1);
      var p14 = P(1, 4);
      var p14Lab = E.halo(E.txt(top, p14[0] - 22, p14[1] - 16, '(1, 4)', 24, E.GOOD, 'end', 900));   // 点的左上：右上是手腕
      // 句 3：原点上往右（红）、往上（绿）两根小箭头，转 90°
      var fx = E.vec(top, E.X, 4.5, { head: 13 }), fy = E.vec(top, E.Y, 4.5, { head: 13 });
      var fTurn = E.arcArrow(top, O[0], O[1], 0.62 * u, E.OMEGA, 3, { head: 11 });
      // 句 5：(1, 2) 先转：原点 → 肘的虚线、红 (0, 1)、绿 (−2, 0)、转 90° 的蓝弧、空心点 (−2, 1)
      var qLine = E.vec(top, E.HI, 3, { head: 12, dash: '6 5' });
      var rA = E.vec(top, E.X, 4.5, { head: 13 }), rB = E.vec(top, E.Y, 4.5, { head: 13 });
      var eArc = K.paint(E.el('path', { d: E.poly(arcPts([0, 0], [1, 2], 0, Math.PI / 2)), fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '7 6' }, top), null, E.OMEGA);
      var R12 = P(-2, 1);
      var rDot = K.paint(E.el('circle', { cx: R12[0], cy: R12[1], r: 7, 'stroke-width': 3 }, top), E.S1, E.HI);
      var rLab = E.halo(E.txt(top, R12[0] - 6, R12[1] + 30, '(−2, 1)', 22, E.HI, 'start', 900));
      // 句 6：再平移 (−2, 1) → (1, 2)：紫箭头，红 3、绿 1 两段
      var shift = E.vec(top, E.VEL, 4.5, { head: 15 });
      var dA = E.vec(top, E.X, 3.5, { head: 11, dash: '6 4' }), dB = E.vec(top, E.Y, 3.5, { head: 11, dash: '6 4' });
      // 句 8：验算 T (3, 2, 1)：红 (0, 3)、绿 (−2, 0)、紫 (3, 1)，头尾相接
      var g321 = K.paint(E.el('circle', { cx: P(3, 2)[0], cy: P(3, 2)[1], r: 7, 'stroke-width': 3, fill: 'none' }, top), null, E.VEL);
      var g321Lab = E.halo(E.txt(top, P(3, 2)[0] + 12, P(3, 2)[1] + 28, '(3, 2)', 20, E.VEL, 'start', 900));
      var vA = E.vec(top, E.X, 4.5, { head: 13 }), vB = E.vec(top, E.Y, 4.5, { head: 13 }), vC = E.vec(top, E.VEL, 4.5, { head: 13 });
      // 句 10：透明板上原来在原点的点：(0, 0) → (3, 1)
      var oArc = K.paint(E.el('path', { d: E.poly(arcPts([1, 2], [0, 0], 0, Math.PI / 2)), fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '7 6' }, top), null, E.VEL);
      var oDot = K.paint(E.el('circle', { r: 7.5, 'stroke-width': 3 }, top), E.VEL, E.S1);
      var p31 = P(3, 1);
      var oLab = E.halo(E.txt(top, p31[0] + 14, p31[1] + 8, '(3, 1)', 24, E.VEL, 'start', 900));
      // 句 11：(2, −1) × 1.57，从原点走直线
      var vEnd = P(2 * Math.PI / 2, -Math.PI / 2);
      var vLine = K.paint(E.el('path', { d: 'M ' + O[0] + ' ' + O[1] + ' L ' + vEnd[0].toFixed(1) + ' ' + vEnd[1].toFixed(1), fill: 'none', 'stroke-width': 4, 'stroke-dasharray': '9 6' }, top), null, E.ERR);
      var vDot = K.paint(E.el('circle', { cx: vEnd[0], cy: vEnd[1], r: 7, 'stroke-width': 3 }, top), E.S1, E.ERR);
      var vLab = E.halo(E.txt(top, vEnd[0] + 14, vEnd[1] + 8, '(3.14, −1.57)', 22, E.ERR, 'start', 900));
      var vX = E.cross(top, vEnd[0] - 4, vEnd[1] - 26, 8, E.ERR, 4);
      // 肘：⊙ + (1, 2)
      var outOf = E.el('g', {}, top);
      var ring = K.paint(E.el('circle', { cx: Q[0], cy: Q[1], r: 12, 'stroke-width': 3.5 }, outOf), E.S1, E.HI);
      K.paint(E.el('circle', { cx: Q[0], cy: Q[1], r: 4 }, outOf), E.HI);
      var qLab = E.halo(E.txt(top, Q[0] + 18, Q[1] + 36, '(1, 2)', 22, E.HI, 'start', 900));   // 离肘的圈远一点：圈鼓起来半径到 17
      // 句 12：两台扫地机
      var vacs = E.el('g', {}, top);
      var vacA = E.vac(vacs, P, 0.46), vacB = E.vac(vacs, P, 0.46);
      vacA.set(3, 1, Math.PI / 2);

      /* ══ 右栏：T + 算式 ══ */
      var RK = E.rowKit(), Rg = E.el('g', {}, s), X = RX + 10, MX = RX + 178, MY = 108;
      var g0 = E.el('g', {}, Rg);
      var a1 = RK.line(g0, X, 96, [['不用真的分无穷多步', E.INK, 900]]);
      var a2 = RK.line(g0, X, 140, [['有', E.INK2], ['现成的公式', E.GOOD, 900]]);
      var a2ok = E.check(a2.g, X + 150, 122, 22, E.GOOD, 4);
      var a3 = E.el('g', {}, g0);
      E.tex(a3, X, 204, '\\textcolor{' + H.hi + '}{\\exp([\\mathcal S]\\,\\theta)}=', 28, E.INK, 'start', { w: 170, h: 50 });
      E.txt(a3, X + 176, 204, '一个位姿 T', 24, E.HI, 'start', 900);
      var mG = E.el('g', {}, Rg);
      E.tex(mG, X + 2, MY + 10, 'T=', 30, E.INK, 'start', { w: 60, h: 50 });
      var M = E.matN(mG, MX, MY, [['?', '?', '?'], ['?', '?', '?'], [0, 0, 1]], { size: 28, dx: 62, dy: 40, wide: [[0, -1, 3], [1, 0, 1], [0, 0, 1]],
        color: function (i, j) { return i === 2 ? E.MUT : j === 0 ? E.X : j === 1 ? E.Y : E.VEL; } });
      M.cells.forEach(function (row) { row.forEach(function (n) { n.__col = n.style.fill; }); });   // 填数前先写灰色问号
      var rotLab = E.txt(mG, M.pos(0, 0)[0] + 31, MY + 92, '朝向', 22, E.OMEGA, 'middle', 900);
      var posLab = E.txt(mG, M.pos(0, 2)[0], MY + 92, '位置', 22, E.VEL, 'middle', 900);
      var ROWS = 236, SZ = 20;
      function rows(lines) {   // 一组算式：lines = [[y 偏移, parts, size], …]
        var g = E.el('g', {}, Rg);
        return { g: g, r: lines.map(function (l) { return RK.line(g, X, ROWS + l[0], l[1], l[2] || SZ); }) };
      }
      var b3 = rows([[10, [['转了 90°：', E.INK2], ['第 3 集的矩阵', E.OMEGA, 900]], 22], [52, [['两列：', E.INK2], ['(0, 1)', E.X, 900], [' 和 ', E.INK2], ['(−1, 0)', E.Y, 900]], 22]]);
      var b4 = rows([[10, [['位置：用轴上的点', E.INK, 900]], 22], [52, [['肘就在轴上，转完还在 ', E.INK2], ['(1, 2)', E.HI, 900]], 22]]);
      var b5 = rows([[0, [['先转：', E.INK2], ['(1, 2)', E.HI, 900], [' 乘上这个矩阵', E.INK2]]],
        [34, [['1 × ', E.INK2], ['(0, 1)', E.X, 900], [' = ', E.INK2], ['(0, 1)', E.X, 900]]],
        [66, [['2 × ', E.INK2], ['(−1, 0)', E.Y, 900], [' = ', E.INK2], ['(−2, 0)', E.Y, 900]]],
        [98, [['加起来：', E.INK2], ['(0, 1)', E.X, 900], [' + ', E.INK2], ['(−2, 0)', E.Y, 900], [' = ', E.INK2], ['(−2, 1)', E.HI, 900]]]]);
      var b6 = rows([[0, [['再平移，回到 ', E.INK2], ['(1, 2)', E.HI, 900], ['：', E.INK2]]],
        [34, [['位置 = ', E.VEL, 900], ['(1, 2)', E.HI, 900], [' − ', E.INK2], ['(−2, 1)', E.HI, 900]]],
        [66, [['1 − (−2) = ', E.INK2], ['3', E.X, 900], ['，2 − 1 = ', E.INK2], ['1', E.Y, 900]]],
        [104, [['位置 = ', E.VEL, 900], ['(3, 1)', E.VEL, 900]], 24]]);
      var b7 = rows([[0, [['验算：', E.INK2], ['T', E.INK, 900], [' × ', E.INK2], ['(3, 2, 1)', E.VEL, 900]]],
        [34, [['3 × ', E.INK2], ['(0, 1, 0)', E.X, 900], [' = ', E.INK2], ['(0, 3, 0)', E.X, 900]]],
        [66, [['2 × ', E.INK2], ['(−1, 0, 0)', E.Y, 900], [' = ', E.INK2], ['(−2, 0, 0)', E.Y, 900]]],
        [98, [['1 × ', E.INK2], ['(3, 1, 1)', E.VEL, 900], [' = ', E.INK2], ['(3, 1, 1)', E.VEL, 900]]]]);
      var b9 = rows([[34, [['0 − 2 + 3 = ', E.INK2], ['1', E.GOOD, 900]]], [66, [['3 + 0 + 1 = ', E.INK2], ['4', E.GOOD, 900]]],
        [98, [['0 + 0 + 1 = ', E.INK2], ['1', E.GOOD, 900]]], [136, [['= ', E.INK2], ['(1, 4, 1)', E.GOOD, 900]], 24]]);
      var b9ok = E.check(b9.g, X + 130, ROWS + 118, 22, E.GOOD, 4);
      var b10 = rows([[10, [['(3, 1)', E.VEL, 900], ['：透明板上', E.INK, 900]], 22], [50, [['原来在原点的那一点', E.INK, 900]], 22],
        [92, [['绕着肘，转到了 ', E.INK2], ['(3, 1)', E.VEL, 900]], 22]]);
      var b11 = rows([[10, [['螺旋轴里的 ', E.INK2], ['(2, −1)', E.INK, 900], [' × 角度：', E.INK2]], 22],
        [50, [['2 × 1.57 = ', E.INK2], ['3.14', E.ERR, 900]], 22], [86, [['−1 × 1.57 = ', E.INK2], ['−1.57', E.ERR, 900]], 22],
        [126, [['→ ', E.INK2], ['(3.14, −1.57)', E.ERR, 900], ['：走歪了', E.ERR, 900]], 22]]);
      var b12 = rows([[10, [['第 7 集：扫地机停在 ', E.INK2], ['(3, 1)', E.VEL, 900]], 22], [50, [['车头朝上，就是这个位姿', E.INK2]], 22],
        [100, [['= 绕 ', E.INK, 900], ['(1, 2)', E.HI, 900], [' 转 90°', E.OMEGA, 900]], 22]]);

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) { RK.layout(); laid = true; }
          var f = E.fade, b = E.bump;

          /* ── 小臂：句 3 转到竖直；句 12 淡出 ── */
          var phi = 90 * D * f(st, T.turn[0], T.turn[1] - T.turn[0]);
          arm.bend(phi);
          V.clearTint(); V.tint(RIGID, E.BODY, 0.45);
          E.op(armG, 1 - f(st, T.armOut, 0.5));
          var pp = around([1, 2], [3, 2], phi), ps = P(pp[0], pp[1]);
          pDot.setAttribute('cx', ps[0].toFixed(1)); pDot.setAttribute('cy', ps[1].toFixed(1));
          pDot.setAttribute('r', (7 + 4 * b(st, T.res2, 0.9)).toFixed(2));
          E.op(pDot, 1 - f(st, T.armOut, 0.4));
          E.drawOn(pArc, f(st, T.turn[0], T.turn[1] - T.turn[0]));
          E.op(pArc, (f(st, T.turn[0], 0.1) > 0 ? 0.8 : 0) * (1 - f(st, T.first - 0.3, 0.4)));
          E.op(p14Lab, f(st, T.res2, 0.3) * (1 - f(st, T.armOut, 0.4)));

          /* ── 句 3：原点上的两根小箭头转 90° ── */
          var ft = f(st, T.frameTurn[0], T.frameTurn[1] - T.frameTurn[0]), fa = 90 * D * ft;
          fx.set(O[0], O[1], O[0] + Math.cos(fa) * u, O[1] - Math.sin(fa) * u);
          fy.set(O[0], O[1], O[0] - Math.sin(fa) * u, O[1] - Math.cos(fa) * u);
          var fOn = f(st, T.ep3 - 0.3, 0.3) * (1 - f(st, T.out2, 0.4));
          E.op(fx.g, fOn); E.op(fy.g, fOn);
          fTurn.set(4 * D, Math.max(5 * D, fa - 4 * D)); E.op(fTurn.g, f(st, T.frameTurn[0], 0.2) * (1 - f(st, T.out2, 0.4)));
          fx.line.style.strokeWidth = 4.5 + 2.5 * b(st, T.col1, 0.8); fy.line.style.strokeWidth = 4.5 + 2.5 * b(st, T.col2, 0.8);

          /* ── 肘 ── */
          E.op(outOf, 1); E.op(qLab, 1 - f(st, T.vac - 0.4, 0.4));   // 扫地机往左那根小箭头的尖在 (2, 1)，紧挨着「(1, 2)」
          ring.setAttribute('r', (12 + 5 * Math.max(b(st, T.elbow, 0.8), b(st, T.back[1], 0.8))).toFixed(2));
          E.glow(ring, E.HI, b(st, T.elbow, 0.8) > 0.05);

          /* ── 句 5：(1, 2) 先转：红 1 份 + 绿 2 份 = (−2, 1) ── */
          var on5 = 1 - f(st, T.out6, 0.4);
          var ql = f(st, T.qq, 0.4); qLine.set(O[0], O[1], O[0] + (Q[0] - O[0]) * ql, O[1] + (Q[1] - O[1]) * ql);
          E.op(qLine.g, (ql > 0 ? 1 : 0) * (1 - f(st, T.sum + 0.6, 0.3)));
          seg(rA, [0, 0], [0, 1], f(st, T.take1, 0.5)); E.op(rA.g, (f(st, T.take1, 0.1) > 0 ? 1 : 0) * on5);
          seg(rB, [0, 1], [-2, 1], f(st, T.take2, 0.6)); E.op(rB.g, (f(st, T.take2, 0.1) > 0 ? 1 : 0) * on5);
          E.drawOn(eArc, f(st, T.sum, 0.8)); E.op(eArc, (f(st, T.sum, 0.1) > 0 ? 1 : 0) * on5);
          E.op(rDot, f(st, T.sumR, 0.3) * on5); E.op(rLab, f(st, T.sumR, 0.3) * on5);
          /* ── 句 6：再平移：紫 (−2, 1) → (1, 2)，红 3、绿 1 ── */
          var sp = f(st, T.back[0], T.back[1] - T.back[0]);
          shift.set(R12[0], R12[1], R12[0] + (Q[0] - 14 - R12[0]) * sp, R12[1] + (Q[1] + 4 - R12[1]) * sp);
          E.op(shift.g, (sp > 0 ? 1 : 0) * on5);
          seg(dA, [-2, 1], [1, 1], f(st, T.d1, 0.6)); E.op(dA.g, (f(st, T.d1, 0.1) > 0 ? 1 : 0) * on5);
          seg(dB, [1, 1], [1, 2], f(st, T.d2, 0.4)); E.op(dB.g, (f(st, T.d2, 0.1) > 0 ? 1 : 0) * on5);

          /* ── 句 7–9：验算 T (3, 2, 1) ── */
          var on8 = 1 - f(st, T.out9, 0.4);
          E.op(g321, f(st, T.p321, 0.3) * on8); E.op(g321Lab, f(st, T.p321, 0.3) * on8);
          seg(vA, [0, 0], [0, 3], f(st, T.t1, 0.6)); E.op(vA.g, (f(st, T.t1, 0.1) > 0 ? 1 : 0) * on8);
          seg(vB, [0, 3], [-2, 3], f(st, T.t2, 0.6)); E.op(vB.g, (f(st, T.t2, 0.1) > 0 ? 1 : 0) * on8);
          seg(vC, [-2, 3], [1, 4], f(st, T.t3, 0.7)); E.op(vC.g, (f(st, T.t3, 0.1) > 0 ? 1 : 0) * on8);

          /* ── 句 10：原点那一点 → (3, 1) ── */
          var tv = f(st, T.travel[0], T.travel[1] - T.travel[0]), oq = around([1, 2], [0, 0], Math.PI / 2 * tv), os = P(oq[0], oq[1]);
          oDot.setAttribute('cx', os[0].toFixed(1)); oDot.setAttribute('cy', os[1].toFixed(1));
          oDot.setAttribute('r', (7.5 + 4 * b(st, T.plate + 1.3, 0.8)).toFixed(2));
          E.op(oDot, f(st, T.plate, 0.3) * (1 - f(st, T.vac - 0.4, 0.4)));
          E.drawOn(oArc, tv); E.op(oArc, (tv > 0 ? 0.85 : 0) * (1 - f(st, T.armOut, 0.4) + 0.7 * f(st, T.ghost, 0.4)));   // 句 12 扫地机沿它走
          E.op(oLab, f(st, T.lab31, 0.3) * (1 - f(st, T.vac - 0.4, 0.4)));

          /* ── 句 11：(2, −1) × 角度 ── */
          var on11 = 1 - f(st, T.armOut, 0.4);
          E.drawOn(vLine, f(st, T.vline[0], T.vline[1] - T.vline[0])); E.op(vLine, (f(st, T.vline[0], 0.1) > 0 ? 1 : 0) * on11);
          E.op(vDot, f(st, T.vline[1] - 0.1, 0.2) * on11); E.op(vLab, f(st, T.v2r, 0.3) * on11);
          E.drawOn(vX, f(st, T.wrong, 0.3)); E.op(vX, (f(st, T.wrong, 0.1) > 0 ? 1 : 0) * on11);

          /* ── 句 12：扫地机 ── */
          E.op(vacA.g, f(st, T.vac, 0.4));
          var ga = f(st, T.around[0], T.around[1] - T.around[0]), vq = around([1, 2], [0, 0], Math.PI / 2 * ga);
          vacB.set(vq[0], vq[1], Math.PI / 2 * ga);
          E.op(vacB.g, f(st, T.ghost, 0.4) * (0.75 - 0.75 * f(st, T.around[1], 0.4)));

          /* ── 右栏 ── */
          E.op(g0, 1 - f(st, T.out0, 0.4));
          E.op(a1.g, fr(st, T.nope)); E.op(a2.g, fr(st, T.ready)); E.drawOn(a2ok, f(st, T.ready + 0.3, 0.3)); E.op(a3, fr(st, T.pose));
          E.pop(mG, MX, MY, f(st, T.mat, 0.35), 0.9);
          M.hl('rot', 0, 0, 1, 1, E.OMEGA, f(st, T.rot, 0.3) * (1 - f(st, T.out2, 0.4)) + 0.6 * b(st, T.simple, 0.9));
          M.hl('pos', 0, 2, 1, 2, E.VEL, f(st, T.pos, 0.3) * (1 - f(st, T.ep3, 0.4)) + f(st, T.posRow, 0.3) * (1 - f(st, T.glue, 0.4)) + b(st, T.col, 0.9));
          M.hl('all', 0, 0, 2, 2, E.HI, b(st, T.glue, 1.0));
          E.op(rotLab, f(st, T.rot, 0.3)); E.op(posLab, f(st, T.pos, 0.3));
          var cells = [[0, 0, '0', T.col1], [1, 0, '1', T.col1 + 0.3], [0, 1, '−1', T.col2], [1, 1, '0', T.col2 + 0.4], [0, 2, '3', T.p31], [1, 2, '1', T.p31b]];
          cells.forEach(function (c) {
            var n = M.cells[c[0]][c[1]], q = M.pos(c[0], c[1]), p = f(st, c[3], 0.3);
            n.textContent = st >= c[3] ? c[2] : '?';
            E.op(n, st >= c[3] ? p : 1); E.scaleAt(n, q[0], q[1], st >= c[3] ? 1 + 0.2 * (1 - p) : 1);
            n.style.fill = st >= c[3] ? n.__col : E.MUT;
          });
          E.op(b3.g, fr(st, T.ep3) * (1 - f(st, T.out2, 0.4))); E.op(b3.r[1].g, fr(st, T.col1)); E.op(b3.r[1].parts[2], fr(st, T.col2)); E.op(b3.r[1].parts[3], fr(st, T.col2));
          E.op(b4.g, fr(st, T.posRow) * (1 - f(st, T.first - 0.3, 0.3)));
          E.op(b4.r[1].g, fr(st, T.elbow)); E.op(b4.r[1].parts[1], fr(st, T.stay));
          E.op(b5.g, fr(st, T.first) * (1 - f(st, T.shift - 0.3, 0.3)));
          E.op(b5.r[0].parts[1], fr(st, T.qq)); E.op(b5.r[0].parts[2], fr(st, T.mult));
          E.op(b5.r[1].g, fr(st, T.take1)); E.op(b5.r[1].parts[2], fr(st, T.got1)); E.op(b5.r[1].parts[3], fr(st, T.got1));
          E.op(b5.r[2].g, fr(st, T.take2)); E.op(b5.r[2].parts[2], fr(st, T.got2)); E.op(b5.r[2].parts[3], fr(st, T.got2));
          E.op(b5.r[3].g, fr(st, T.sum)); E.op(b5.r[3].parts[5], fr(st, T.sumR)); E.op(b5.r[3].parts[4], fr(st, T.sumR));
          E.op(b6.g, fr(st, T.shift) * (1 - f(st, T.glue + 0.4, 0.3)));
          E.op(b6.r[1].g, fr(st, T.diff)); E.op(b6.r[2].g, fr(st, T.d1)); E.op(b6.r[2].parts[1], fr(st, T.d1r));
          E.op(b6.r[2].parts[2], fr(st, T.d2)); E.op(b6.r[2].parts[3], fr(st, T.d2r)); E.op(b6.r[3].g, fr(st, T.p31));
          E.op(b7.g, fr(st, T.check) * (1 - f(st, T.out9, 0.4)));
          E.op(b7.r[0].parts[3], fr(st, T.p321)); E.op(b7.r[0].parts[1], fr(st, T.mult2)); E.op(b7.r[0].parts[2], fr(st, T.mult2));
          var prodOff = 1 - f(st, T.add, 0.3);
          [[T.t1, T.t1r], [T.t2, T.t2r], [T.t3, T.t3r]].forEach(function (tt, k) {
            var r = b7.r[k + 1]; E.op(r.g, fr(st, tt[0]) * prodOff); E.op(r.parts[2], fr(st, tt[1])); E.op(r.parts[3], fr(st, tt[1]));
          });
          E.op(b9.g, fr(st, T.add) * (1 - f(st, T.out9, 0.4)));
          E.op(b9.r[0].g, fr(st, T.s1)); E.op(b9.r[1].g, fr(st, T.s2)); E.op(b9.r[2].g, fr(st, T.s3)); E.op(b9.r[3].g, fr(st, T.res)); E.drawOn(b9ok, f(st, T.res2 + 0.4, 0.3));
          E.op(b10.g, fr(st, T.col) * (1 - f(st, T.vth - 0.4, 0.3)));
          E.op(b10.r[0].parts[1], fr(st, T.plate)); E.op(b10.r[1].g, fr(st, T.plate)); E.op(b10.r[2].g, fr(st, T.travel[0]));
          E.op(b11.g, fr(st, T.vth) * (1 - f(st, T.armOut, 0.4)));
          E.op(b11.r[0].parts[2], fr(st, T.vline[0])); E.op(b11.r[1].g, fr(st, T.v1)); E.op(b11.r[1].parts[1], fr(st, T.v1r));
          E.op(b11.r[2].g, fr(st, T.v2)); E.op(b11.r[2].parts[1], fr(st, T.v2r)); E.op(b11.r[3].g, fr(st, T.wrong));
          E.op(b12.g, fr(st, T.ep7));
          E.op(b12.r[0].parts[1], fr(st, T.vac)); E.op(b12.r[1].g, fr(st, T.up)); E.op(b12.r[2].g, fr(st, T.around[0]));
        }
      };
    }
  };
})();
