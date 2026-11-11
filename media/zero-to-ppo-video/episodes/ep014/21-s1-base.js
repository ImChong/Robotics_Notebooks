// 014 第 1 幕：站在底座上看——空间运动旋量的后 3 个数，是原点那一点的速度（《Modern Robotics》3.3.2 节式 (3.73)(3.74)、图 3.17；第 9 集讲过透明板）。
// 左边第 9–13 集的侧视网格 + UR5，姿势照第 13 集第 6 幕「肩先转了 90°」：肘 (−2, 1)，小臂竖着朝上，手 (−2, 3)。
// 句 1「网格」网格淡入，「零位时，肘在一、二，手在三、二」先标零位的坐标（第 11 集），「肩先转了九十度」胳膊逆时针转 90°（蓝弧）；
// 句 2（2026-11-10 用户要求「不要只说第几集说过，要快速过一下怎么得到的」）「乘第三集的矩阵」右栏出 [[0, −1], [1, 0]]，肘、手各按列拿份：
//    1 × (0, 1) + 2 × (−1, 0) = (−2, 1)，3 × (0, 1) + 2 × (−1, 0) = (−2, 3)，念到哪一列、哪一列框亮，左边标出新坐标、「小臂竖着潮上」；
// 句 2「底座上画着两根轴」原点画底座的坐标系（x 红、y 绿），「站在这里看」底座亮，「固定坐标系」右栏写；
// 句 3「只让肩每秒转一弧度」肩旁蓝色小弯箭头，胳膊逆时针转（句 2 末尾、画面上没有方向标记时先顺时针挪开 0.3 弧度，这里一路逆时针转回这一集的姿势），
//    「肩的螺旋轴」右栏 6 个数 (0, 0, 1, 0, 0, 0) 逐个出；
// 句 4「角速度」前 3 个数框蓝，「后三个数却全是零」后 3 个数框橙，「手明明在走」手上出紫色虚线箭头 + 问号（沿手真的运动方向）；
// 句 5「透明板」淡黄的大板子和胳膊粘在一起，「原点那一点」原点画白圈；
// 句 6「原地转」胳膊连同板子绕肩逆时针转（句 5 开头先顺时针挪开），原点那一点不动，「所以是零」写 速度 = 0，「不是手的速度」；
// 句 7「从肩关节指向手」白箭头，「转九十度」白箭头绕原点转 90°，「第一列取负二份」红 (0, −2)、「第二列取三份」绿 (−3, 0) 头尾相接，「加起来」紫 (−3, −2)；
// 句 8 紫箭头挪到手上：手每秒往左 3 格、往下 2 格；句 9 第 13 集第 6 幕为什么正好相等；句 10「站在手上看」手亮。
// 时刻取自配音的逐词时间，见 ../ep014.py。
(function () {
  var E = window.EP014, K = E.K, H = E.HEX;
  var X = 455, D90 = Math.PI / 2;
  var T = {
    grid: 1.88, zpose: 2.58, zE: 3.33, zH: 4.53, sh: 6.63, rot: [7.09, 8.5], mat: 10.37, cols: 11.98, el: 14.18, eC1: 15.19, eC2: 16.25, e21: 17.37,
    hand: 18.66, hC1: 19.71, hC2: 20.86, h23: 22.02, fore: 23.36, up: 23.8, out1: 25.6,
    base: 25.87, axes: 26.7, xr: 27.73, yu: 28.7, ep13: 30.06, here: 32.07, book: 33.38, fixed: 34.54, pre1: [34.97, 36.37], out2: 36.52,
    only: 36.75, sh3: 37.11, rate: 37.83, whole: 38.83, rot3: [38.07, 40.97], ep13b: 41.26, twist: 43.17, isS: 44.72,
    nums: [45.6, 46.08, 46.57, 47.11, 47.63, 48.14],
    front: 49.6, omega: 50.54, z: 51.61, perSec: 52.27, back: 54.04, zero: 54.82, moving: 56.01, why: 57.26, outQ: 58.97,
    ep9: 59.33, pre2: [59.37, 60.47], boardOn: 61.37, back3: 63.08, onBoard: 64.06, atO: 65.08, speed: 66.05,
    origin: 67.64, isSh: 68.82, rot6: [69.37, 71.47], noMove: 70.62, soZero: 71.55, notHand: 73.31, outB: 74.37,
    hp: 75.57, walk: 76.95, shJ: 77.72, n23: 79.16, turn: [80.3, 81.17], c1: 81.26, take1: 81.91, r02: 82.78, c2: 83.8, take2: 84.44, r30: 85.2,
    add: 86.29, sum: 86.98, out7: 88.67,
    hv: 89.26, left3: 90.15, down2: 91.22,
    ep13c: 93.42, rates: 96.41, vNums: 97.39, equal: 99.78, because: 101.05, trans: 103.16, every: 104.94, same: 106.26,
    want: 107.78, hpt: 108.78, change: 110.38, onHand: 111.47
  };
  var END = 114.07;

  E.scenes[0] = {
    title: '站在底座上看',
    dur: END,
    cues: [
      { at: 0, s: '第 13 集的网格：零位时肘在 $(1,\\,2)$、手在 $(3,\\,2)$；现在肩先转了 $90^\\circ$' },
      { at: 9.13, s: '转 $90^\\circ$ = 乘第 3 集的矩阵、按列拿份：肘到 $(-2,\\,1)$，手到 $(-2,\\,3)$' },
      { at: 25.77, s: '底座上的坐标系：$x$ 往右、$y$ 往上。第 13 集的**运动旋量**都是站在这里看的——书上叫**固定坐标系**' },
      { at: 36.65, s: '只让肩每秒转 $1$ 弧度：这时的**运动旋量**就是肩的**螺旋轴** $(0,\\,0,\\,1,\\,0,\\,0,\\,0)$（第 13 集）' },
      { at: 49.5, s: '前 $3$ 个数是角速度：绕 $z$ 轴每秒转 $1$ 弧度。后 $3$ 个数全是 $0$——手明明在走？' },
      { at: 59.23, s: '第 9 集：把胳膊想成一张很大的透明板，后 $3$ 个数是板上**此刻在原点那一点**的速度' },
      { at: 67.54, s: '原点这一点就是肩：只在原地转、不挪地方，所以是 $0$——后 $3$ 个数**不是手的速度**' },
      { at: 75.47, s: '手这一点：从肩关节指向手 $(-2,\\,3)$，转 $90^\\circ$：$(-2)\\times(0,\\,1)+3\\times(-1,\\,0)=(-3,\\,-2)$' },
      { at: 89.16, s: '**手的速度**：每秒往左走 $3$ 格、往下走 $2$ 格' },
      { at: 93.32, s: '第 13 集第 6 幕：转速 $(1,\\,-1)$，后 $3$ 个数字 $(-2,\\,1)$ = 手的速度：小臂只平移' },
      { at: 107.68, s: '想直接知道手这一点的速度：换个地方站——**站在手上看**' }
    ],
    build: function (K) {
      var s = K.sceneSvg('还是第十三集的网格：肩先转了九十度，肘到了 (−2, 1)，小臂竖着朝上，手在 (−2, 3)。底座上画着两根轴：往右是 x、往上是 y，第十三集的运动旋量都是站在这里看的，书上叫固定坐标系。只让肩每秒转一弧度，这时的运动旋量就是肩的螺旋轴 (0, 0, 1, 0, 0, 0)：前三个数是角速度，后三个数全是零，手明明在走。第九集讲过：把胳膊想成一张很大的透明板，后三个数是板上此刻在原点那一点的速度；原点这一点就是肩，只在原地转、不挪地方，所以是零，后三个数不是手的速度。手这一点的速度：从肩关节指向手 (−2, 3)，转九十度得 (−3, −2)，每秒往左走三格、往下走两格。第十三集第六幕正好相等，是因为那时小臂只平移不转。想直接知道手这一点的速度，就站在手上看');

      var pn = E.panel14(s), P = pn.P, u = pn.u, top = pn.top;
      var board = E.board(pn.under, P);
      var O = P(0, 0);
      var bF = E.frame2(top, P, 1, '底座', { labAt: [0.75, -0.62], tipOff: E.BASE_TIP });
      var jS = E.joint(top, E.HI), jE = E.joint(top, E.HI), hand = E.dot(top, E.V, 7);
      var arc90 = E.arcArrow(top, O[0], O[1], 0.95 * u, E.OMEGA, 3.5, { head: 12 });
      var a90L = E.halo(E.txt(top, 0, 0, '90°', 20, E.OMEGA, 'middle', 900));
      var rArc = E.arcArrow(top, O[0], O[1], 0.6 * u, E.OMEGA, 3.5, { head: 12 });
      var rLab = E.tex(top, O[0] - 132, O[1] + 46, '\\dot\\theta_1=1', 22, E.OMEGA, 'start', { w: 90, h: 36 });
      var eLab = E.halo(E.txt(top, P(-2, 1)[0] - 18, P(-2, 1)[1] + 30, '肘 (−2, 1)', 20, E.HI, 'end', 900));
      var hLab = E.halo(E.txt(top, P(-2, 3)[0] - 18, P(-2, 3)[1] + 8, '手 (−2, 3)', 20, E.V, 'end', 900));
      var foreL = E.halo(E.txt(top, P(-2, 2)[0] - 18, P(-2, 2)[1] + 6, '竖着朝上', 18, E.INK2, 'end', 900));
      // 零位（第 11 集）：肘 (1, 2)、手 (3, 2)，肩转 90° 之前标出来
      var zEL = E.halo(E.txt(top, P(1, 2)[0] + 12, P(1, 2)[1] + 34, '肘 (1, 2)', 20, E.HI, 'start', 900));
      var zHL = E.halo(E.txt(top, P(3, 2)[0] - 4, P(3, 2)[1] - 22, '手 (3, 2)', 20, E.V, 'middle', 900));
      // 原点那一点（透明板上）：白圈 + 「速度 = 0」
      var oPt = E.el('g', {}, top);
      K.paint(E.el('circle', { cx: O[0], cy: O[1], r: 13, fill: 'none', 'stroke-width': 3.5 }, oPt), null, E.INK);
      var oLab = E.halo(E.txt(top, P(0.45, 1.75)[0], P(0.45, 1.75)[1] + 6, '原点那一点', 18, E.INK, 'start', 900));
      var oZero = E.halo(E.txt(top, P(0.45, 1.2)[0], P(0.45, 1.2)[1] + 6, '速度 = 0', 18, E.INK, 'start', 900));
      // 句 4：问号箭头；句 7：白 → 转 90° → 红绿 → 紫
      var qV = E.vec(top, E.V, 4.5, { head: 15, dash: '8 6' }), qM = E.halo(E.txt(top, 0, 0, '?', 34, E.V, 'middle', 900));
      var wA = E.gv(top, P, E.INK, 4, { head: 14, dash: '9 6' }), wT = E.gv(top, P, E.INK, 4, { head: 14 });
      var tArc = E.arcArrow(top, O[0], O[1], 0.75 * u, E.OMEGA, 3, { head: 10 });
      var red = E.gv(top, P, E.X, 5, { head: 14 }), grn = E.gv(top, P, E.Y, 5, { head: 14 }), vio = E.gv(top, P, E.V, 5.5, { head: 17 });
      [red, grn, vio, wT].forEach(function (a) { a.line.style.filter = 'drop-shadow(0 0 2px #000)'; });
      var wL = E.halo(E.txt(top, 0, 0, '(−2, 3)', 19, E.INK, 'middle', 900));
      var redL = E.halo(E.txt(top, P(0, -1.55)[0] + 12, P(0, -1.55)[1] + 6, '(0, −2)', 18, E.X, 'start', 900));
      var grnL = E.halo(E.txt(top, P(-1.5, -2)[0], P(-1.5, -2)[1] + 24, '(−3, 0)', 18, E.Y, 'middle', 900));
      var vioL = E.halo(E.txt(top, 0, 0, '(−3, −2)', 19, E.V, 'middle', 900));
      var standL = E.halo(E.txt(top, P(-2, 3)[0], P(-2, 3)[1] - 26, '站在手上看', 19, E.V, 'middle', 900));

      /* ══ 右栏 ══ */
      var RK = E.rowKit(), Rg = E.el('g', {}, s);
      function grp() { return E.el('g', {}, Rg); }
      var g1 = grp();
      var a1 = RK.line(g1, X, 46, [['第 13 集的网格', E.INK, 900]], 22);
      var a2 = RK.line(g1, X, 82, [['零位：', E.INK2], ['肘 (1, 2)', E.HI, 900], ['，', E.INK2], ['手 (3, 2)', E.V, 900]], 19);
      var a3 = RK.line(g1, X, 118, [['肩先转了 ', E.INK2], ['90°', E.OMEGA, 900]], 19);
      var a4 = RK.line(g1, X, 160, [['转 90° = 乘第 3 集的矩阵', E.INK2]], 18);
      var R90 = E.matN(g1, X + 300, 152, [[0, -1], [1, 0]], { size: 17, dx: 30, dy: 24, color: function (i, j) { return j === 0 ? E.X : E.Y; } });
      var a5 = RK.line(g1, X, 214, [['肘：', E.HI, 900], ['1 × ', E.INK, 900], ['(0, 1)', E.X, 900], [' + ', E.INK2], ['2 × ', E.INK, 900], ['(−1, 0)', E.Y, 900]], 18);
      var a6 = RK.line(g1, X, 242, [['　= ', E.INK2], ['(−2, 1)', E.HI, 900]], 18);
      var a7 = RK.line(g1, X, 284, [['手：', E.V, 900], ['3 × ', E.INK, 900], ['(0, 1)', E.X, 900], [' + ', E.INK2], ['2 × ', E.INK, 900], ['(−1, 0)', E.Y, 900]], 18);
      var a8 = RK.line(g1, X, 312, [['　= ', E.INK2], ['(−2, 3)', E.V, 900]], 18);
      var a9 = RK.line(g1, X, 354, [['小臂', E.INK, 900], ['竖着朝上', E.INK2]], 19);
      var g2 = grp();
      var b1 = RK.line(g2, X, 60, [['底座上的坐标系', E.INK, 900]], 24);
      var b2 = RK.line(g2, X, 104, [['x', E.X, 900], [' 往右　', E.INK2], ['y', E.Y, 900], [' 往上', E.INK2]], 22);
      var b3 = RK.line(g2, X, 160, [['第 13 集的', E.INK2], ['运动旋量', E.WARN, 900], ['：', E.INK2]], 21);
      var b4 = RK.line(g2, X, 196, [['都是站在这里看的', E.INK, 900]], 21);
      var b5 = RK.line(g2, X, 252, [['书上：', E.INK2], ['固定坐标系', E.INK, 900]], 22);
      var b6 = RK.line(g2, X, 286, [['（fixed frame，记作 {s}）', E.MUT, 700]], 17);
      var g3 = grp();
      var c1 = E.symRow(g3, RK, X, 50, '\\dot\\theta_1=1', E.OMEGA, 80, [['只转肩，每秒 1 弧度', E.INK2]], 20);
      var c2 = RK.line(g3, X, 92, [['运动旋量', E.WARN, 900], [' = 肩的', E.INK2], ['螺旋轴', E.WARN, 900], ['（第 13 集）', E.MUT, 700]], 20);
      var tw = E.tw6(g3, X + 46, 238, [0, 0, 1, 0, 0, 0], { size: 23, dy: 28, w: 50 });
      var twW = E.txt(g3, X + 92, 207, '角速度', 20, E.OMEGA, 'start', 900), twV = E.txt(g3, X + 92, 291, '线速度', 20, E.V, 'start', 900);
      var twW2 = E.txt(g3, X + 170, 207, '绕 z 轴，每秒 1 弧度', 18, E.OMEGA, 'start', 700);
      var twV2 = E.txt(g3, X + 170, 291, '全是 0？', 20, E.ERR, 'start', 900);
      var c3 = RK.line(g3, X, 380, [['手', E.V, 900], ['明明在走 → 为什么是 0？', E.INK, 900]], 20);
      var g5 = grp();
      var d1 = RK.line(g5, X, 52, [['第 9 集：', E.INK2], ['透明板', E.HI, 900]], 22);
      var d2 = RK.line(g5, X, 92, [['后 3 个数 = 板上', E.INK2]], 21);
      var d3 = RK.line(g5, X, 128, [['此刻在原点那一点', E.INK, 900], ['的速度', E.INK2]], 21);
      var d4 = RK.line(g5, X, 190, [['原点那一点 = ', E.INK2], ['肩', E.HI, 900]], 21);
      var d5 = RK.line(g5, X, 226, [['只在原地转，不挪地方', E.INK2]], 20);
      var d6 = RK.line(g5, X, 262, [['→ 速度 = 0 ', E.INK, 900], ['✓', E.GOOD, 900]], 21);
      var d7 = RK.line(g5, X, 320, [['后 3 个数 ', E.INK2], ['≠', E.ERR, 900], [' 手的速度', E.V, 900]], 22);
      var g7 = grp();
      var f1 = RK.line(g7, X, 48, [['手这一点的速度：', E.INK, 900]], 22);
      var f2 = RK.line(g7, X, 90, [['从肩关节指向手 = ', E.INK2], ['(−2, 3)', E.INK, 900]], 19);
      var f3 = RK.line(g7, X, 128, [['转 90°：', E.OMEGA, 900], ['(−2) × ', E.INK, 900], ['(0, 1)', E.X, 900], [' + ', E.INK2], ['3 × ', E.INK, 900], ['(−1, 0)', E.Y, 900]], 19);
      var f4 = RK.line(g7, X, 164, [['= ', E.INK2], ['(0, −2)', E.X, 900], [' + ', E.INK2], ['(−3, 0)', E.Y, 900]], 19);
      var f5 = RK.line(g7, X, 200, [['= ', E.INK2], ['(−3, −2)', E.V, 900]], 21);
      var f6 = RK.line(g7, X, 252, [['每秒：往左 ', E.INK2], ['3', E.INK, 900], [' 格、往下 ', E.INK2], ['2', E.INK, 900], [' 格', E.INK2]], 20);
      var g9 = grp();
      var h1 = RK.line(g9, X, 302, [['第 13 集第 6 幕：转速 ', E.INK2], ['(1, −1)', E.OMEGA, 900]], 17);
      var h1b = RK.line(g9, X, 326, [['后 3 个数字 ', E.INK2], ['(−2, 1)', E.V, 900], [' = 手的速度', E.INK2]], 17);
      var h2 = RK.line(g9, X, 352, [['小臂只平移、不转', E.INK, 900]], 18);
      var h3 = RK.line(g9, X, 378, [['板上每一点都走得一样', E.INK2]], 18);
      var g10 = grp();
      var k1 = RK.line(g10, X, 330, [['想直接知道手这一点的速度', E.INK2]], 19);
      var k2 = RK.line(g10, X, 368, [['→ ', E.INK2], ['站在手上看', E.V, 900]], 22);

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) { RK.layout(); laid = true; }
          var f = E.fade, b = E.bump;
          /* ── UR5：肩转过的角度 th（第 11 集的零位算 0）。只朝蓝色小弯箭头的方向（逆时针）转；要归位时趁画面上没有方向标记先顺时针挪开 ── */
          var th = D90 * f(st, T.rot[0], T.rot[1] - T.rot[0])
            - 0.3 * f(st, T.pre1[0], T.pre1[1] - T.pre1[0]) + 0.3 * f(st, T.rot3[0], T.rot3[1] - T.rot3[0])
            - 0.3 * f(st, T.pre2[0], T.pre2[1] - T.pre2[0]) + 0.3 * f(st, T.rot6[0], T.rot6[1] - T.rot6[0]);
          pn.pose(E.ZERO[0] + th, th);
          pn.view.clearTint();
          var J = pn.joints(), pE = P(J[1][0], J[1][1]), pH = P(J[2][0], J[2][1]);
          E.op(pn.under, f(st, T.grid, 0.5));
          board.set(th - D90);
          E.op(board.g, E.win(st, T.boardOn, T.outB, 0.5));
          jS.at(O); jE.at(pE); hand.at(pH, 7 + 4 * Math.max(b(st, T.hand, 0.8), b(st, T.moving, 0.8), b(st, T.hp, 0.8), b(st, T.hpt, 0.8), b(st, T.onHand, 0.8)));
          E.op(jS, fr(st, T.zpose)); E.op(jE, fr(st, T.zE)); E.op(hand, fr(st, T.zH));
          E.glow(jS, E.HI, b(st, T.sh, 0.8) > 0.05 || b(st, T.sh3, 0.8) > 0.05 || b(st, T.isSh, 0.8) > 0.05);
          E.glow(jE, E.HI, b(st, T.zE, 0.8) > 0.05 || b(st, T.e21, 0.8) > 0.05);
          E.glow(hand, E.V, st > T.onHand);
          // 句 1：转 90° 的蓝弧、标签
          arc90.set(E.ZERO[0], E.ZERO[0] + Math.max(0.01, D90 * f(st, T.rot[0], T.rot[1] - T.rot[0])));
          var on90 = fr(st, T.rot[0]) * (1 - fr(st, T.out1));
          E.op(arc90.g, on90);
          var la = E.ZERO[0] + D90 * 0.5, lq = P(1.35 * Math.cos(la), 1.35 * Math.sin(la));
          a90L.setAttribute('x', lq[0].toFixed(1)); a90L.setAttribute('y', (lq[1] + 6).toFixed(1)); E.op(a90L, fr(st, T.rot[1] - 0.3) * (1 - fr(st, T.out1)));
          E.op(zEL, fr(st, T.zE) * (1 - fr(st, T.rot[0] - 0.3))); E.op(zHL, fr(st, T.zH) * (1 - fr(st, T.rot[0] - 0.3)));
          E.op(eLab, fr(st, T.e21) * (1 - fr(st, T.out1))); E.op(foreL, fr(st, T.up) * (1 - fr(st, T.out1))); E.op(hLab, fr(st, T.h23) * (1 - fr(st, T.out1)));
          // 句 2：底座的坐标系（之后一直在）
          bF.set(0, 0, 0);
          E.op(bF.g, fr(st, T.axes)); E.op(bF.ax.g, fr(st, T.xr)); E.op(bF.lx, fr(st, T.xr)); E.op(bF.ay.g, fr(st, T.yu)); E.op(bF.ly, fr(st, T.yu));
          E.op(bF.lab, fr(st, T.base));
          E.glow(bF.g, E.INK, b(st, T.here, 1.2) > 0.05 || b(st, T.base, 1.0) > 0.05);
          // 句 3、6：肩旁的蓝色小弯箭头（逆时针，同胳膊转的方向）
          rArc.set(3.45, 4.55);   // 左下：右上是底座的坐标系，左上是大臂
          var rOn = E.win(st, T.rate, T.ep13b, 0.3) + E.win(st, T.rot6[0] - 0.2, T.noMove + 1.2, 0.3);
          E.op(rArc.g, rOn); E.op(rLab, E.win(st, T.rate, T.ep13b, 0.3));
          // 句 4：手上的问号箭头（方向 = 只转肩时手真的运动方向）
          var vq = E.vHand(J, 1, 0), lv = Math.hypot(vq[0], vq[1]), dq = [vq[0] / lv * 1.6, vq[1] / lv * 1.6];
          var qT = P(J[2][0] + dq[0], J[2][1] + dq[1]), qL = P(J[2][0] + dq[0] * 1.32, J[2][1] + dq[1] * 1.32);
          var qOn = fr(st, T.moving) * (1 - fr(st, T.outQ));
          qV.set(pH[0], pH[1], qT[0], qT[1]); E.op(qV.g, qOn);
          qM.setAttribute('x', qL[0].toFixed(1)); qM.setAttribute('y', (qL[1] + 10).toFixed(1)); E.op(qM, qOn * (0.65 + 0.35 * b(st, T.why, 0.8)));
          // 句 5–6：原点那一点
          var oOn = E.win(st, T.atO, T.outB, 0.3);
          E.op(oPt, oOn); E.op(oLab, oOn); E.op(oZero, E.win(st, T.soZero, T.outB, 0.3));
          E.glow(oPt, E.INK, b(st, T.origin, 1.0) > 0.05 || b(st, T.soZero, 1.0) > 0.05);
          // 句 7：白箭头 (−2, 3) → 绕原点转 90° → 红、绿头尾相接 → 紫 (−3, −2)；句 8 紫箭头挪到手上
          var on7 = 1 - fr(st, T.out7);
          wA.at(0, 0, -2, 3); E.op(wA.g, fr(st, T.shJ) * on7);
          wL.setAttribute('x', P(-1.25, 1.75)[0] - 10); wL.setAttribute('y', P(-1.25, 1.75)[1] - 2); wL.style.textAnchor = 'end'; E.op(wL, fr(st, T.n23) * (1 - fr(st, T.turn[0])));   // 白箭头左上（右边是底座 y 轴的字母）
          var tr = f(st, T.turn[0], T.turn[1] - T.turn[0]), ang = Math.atan2(3, -2) + D90 * tr, L13 = Math.sqrt(13);
          wT.at(0, 0, L13 * Math.cos(ang), L13 * Math.sin(ang)); E.op(wT.g, fr(st, T.turn[0] - 0.2) * (1 - fr(st, T.add)));
          tArc.set(Math.atan2(3, -2), Math.atan2(3, -2) + D90 * Math.max(tr, 0.01)); E.op(tArc.g, (tr > 0.01 ? 1 : 0) * (1 - fr(st, T.take1)));
          red.at(0, 0, 0, -2 * f(st, T.take1, 0.6)); E.op(red.g, (st > T.take1 ? 1 : 0) * on7); E.op(redL, fr(st, T.r02) * on7);
          grn.at(0, -2, -3 * f(st, T.take2, 0.6), -2); E.op(grn.g, (st > T.take2 ? 1 : 0) * on7); E.op(grnL, fr(st, T.r30) * on7);
          var mv = f(st, T.hv, 0.9), vx0 = -2 * mv, vy0 = 3 * mv;
          vio.at(vx0, vy0, vx0 - 3, vy0 - 2); E.op(vio.g, fr(st, T.sum));
          var vl = mv > 0.5 ? P(vx0 - 1.5, vy0 - 1) : P(-2.25, -0.62), vlOff = mv > 0.5 ? [-22, -16] : [0, 6];
          vioL.setAttribute('x', (vl[0] + vlOff[0]).toFixed(1)); vioL.setAttribute('y', (vl[1] + vlOff[1]).toFixed(1));
          E.op(vioL, fr(st, T.sum) * (1 - E.win(st, T.hv - 0.1, T.hv + 1.0, 0.25)));
          E.op(standL, fr(st, T.onHand));

          /* ── 右栏 ── */
          E.op(g1, 1 - fr(st, T.out1));
          E.op(a1.g, fr(st, T.grid)); E.op(a2.g, fr(st, T.zpose)); E.op(a2.parts[1], fr(st, T.zE)); [2, 3].forEach(function (k) { E.op(a2.parts[k], fr(st, T.zH)); });
          E.op(a3.g, fr(st, T.sh)); E.op(a3.parts[1], fr(st, T.rot[0])); E.op(a4.g, fr(st, T.mat)); E.op(R90.g, fr(st, T.cols));
          R90.hl('c1', 0, 0, 1, 0, E.X, E.win(st, T.eC1, T.eC2, 0.25) + E.win(st, T.hC1, T.hC2, 0.25)); R90.hl('c2', 0, 1, 1, 1, E.Y, E.win(st, T.eC2, T.e21, 0.25) + E.win(st, T.hC2, T.h23, 0.25));
          E.op(a5.g, fr(st, T.el)); [1, 2].forEach(function (k) { E.op(a5.parts[k], fr(st, T.eC1)); }); [3, 4, 5].forEach(function (k) { E.op(a5.parts[k], fr(st, T.eC2)); }); E.op(a6.g, fr(st, T.e21));
          E.op(a7.g, fr(st, T.hand)); [1, 2].forEach(function (k) { E.op(a7.parts[k], fr(st, T.hC1)); }); [3, 4, 5].forEach(function (k) { E.op(a7.parts[k], fr(st, T.hC2)); }); E.op(a8.g, fr(st, T.h23));
          E.op(a9.g, fr(st, T.fore)); E.op(a9.parts[1], fr(st, T.up));
          E.op(g2, E.win(st, T.base - 0.1, T.out2, 0.3));
          E.op(b1.g, fr(st, T.base)); E.op(b2.g, fr(st, T.xr)); E.op(b2.parts[2], fr(st, T.yu)); E.op(b2.parts[3], fr(st, T.yu));
          E.op(b3.g, fr(st, T.ep13)); E.op(b4.g, fr(st, T.here)); E.op(b5.g, fr(st, T.book)); E.op(b5.parts[1], fr(st, T.fixed)); E.op(b6.g, fr(st, T.fixed + 0.4));
          E.op(g3, E.win(st, T.only - 0.1, T.ep9 - 0.2, 0.3));
          E.op(c1.g, fr(st, T.rate)); E.op(c2.g, fr(st, T.twist)); E.op(c2.parts[2], fr(st, T.isS)); E.op(c2.parts[3], fr(st, T.isS));
          E.op(tw.g, fr(st, T.nums[0] - 0.4));
          tw.cells.forEach(function (c, k) { E.op(c, fr(st, T.nums[k])); });
          tw.rangeBox(0, 2, E.OMEGA, fr(st, T.omega)); tw.rangeBox(3, 5, E.ERR, fr(st, T.zero));
          E.op(twW, fr(st, T.omega)); E.op(twW2, fr(st, T.z)); E.op(twV, fr(st, T.back)); E.op(twV2, fr(st, T.zero));
          E.op(c3.g, fr(st, T.moving)); E.op(c3.parts[1], fr(st, T.moving));
          E.op(g5, E.win(st, T.ep9 - 0.1, T.hp - 0.2, 0.3));
          E.op(d1.g, fr(st, T.ep9)); E.op(d1.parts[1], fr(st, T.boardOn - 0.5)); E.op(d2.g, fr(st, T.back3)); E.op(d3.g, fr(st, T.atO));
          E.op(d4.g, fr(st, T.origin)); E.op(d4.parts[1], fr(st, T.isSh)); E.op(d5.g, fr(st, T.rot6[0])); E.op(d6.g, fr(st, T.soZero)); E.op(d7.g, fr(st, T.notHand));
          E.op(g7, fr(st, T.hp - 0.1) * (1 - 0.55 * fr(st, T.ep13c - 0.2)));
          E.op(f1.g, fr(st, T.hp)); E.op(f2.g, fr(st, T.shJ)); E.op(f2.parts[1], fr(st, T.n23));
          E.op(f3.g, fr(st, T.turn[0])); E.op(f3.parts[1], fr(st, T.take1)); E.op(f3.parts[2], fr(st, T.take1)); [3, 4, 5].forEach(function (k) { E.op(f3.parts[k], fr(st, T.take2)); });
          E.op(f4.g, fr(st, T.r02)); E.op(f4.parts[2], fr(st, T.r30)); E.op(f4.parts[3], fr(st, T.r30));
          E.op(f5.g, fr(st, T.sum)); E.op(f6.g, fr(st, T.left3)); E.op(f6.parts[2], fr(st, T.down2)); E.op(f6.parts[3], fr(st, T.down2)); E.op(f6.parts[4], fr(st, T.down2));
          // 句 9–10：上面留着手的速度那几行（淡一点），下面写
          E.op(g9, E.win(st, T.ep13c - 0.1, T.want - 0.1, 0.3));
          E.op(h1.g, fr(st, T.ep13c)); E.op(h1.parts[1], fr(st, T.rates)); E.op(h1b.g, fr(st, T.vNums)); E.op(h1b.parts[2], fr(st, T.equal)); E.op(h2.g, fr(st, T.trans)); E.op(h3.g, fr(st, T.every));
          E.op(g10, fr(st, T.want - 0.1));
          E.op(k1.g, fr(st, T.want)); E.op(k2.g, fr(st, T.change)); E.op(k2.parts[1], fr(st, T.onHand));
        }
      };
    }
  };
})();
