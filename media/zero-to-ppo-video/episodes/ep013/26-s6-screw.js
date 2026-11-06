// 013 第 6 幕：六个关节——写成运动旋量，雅可比的每一列就是关节轴此刻的螺旋轴（兑现 011 第 7 幕；《Modern Robotics》5.1 节：第 i 列是只让第 i 个关节
// 每秒转 1 弧度时的运动旋量，和第 4 章求螺旋轴的办法一样，只是用此刻的 θ；5.1.1 节式 (5.11)；伴随变换这一步留到 014 不讲）。
// 前后两段是斜着看的 UR5（URDF）和六根关节轴（琥珀虚线，同 011 第 5 幕），中间一段回到平面上的两个关节（侧视网格）算一遍。
// 句 1「真的机械臂」斜着看的 UR5，「位置」手上紫点，「潮向」手上三根小箭头，「运动旋量」右栏写 6 个数：角速度 3 个 + 线速度 3 个（第 9 集）；
// 句 2「运动旋量等于转速乘以螺旋轴」右栏写，「只让一个关节」肘的轴（S₃）亮、小臂绕它转一点，「就是这个关节的螺旋轴」右栏写；
// 句 3「每一列」「关节轴此刻的螺旋轴」六根轴都出来，右栏写，「第十一集提过」；
// 句 4 换成侧视网格：「零位」肘 (1, 2)、手 (3, 2)，「两列」右栏出 6 × 2 的雅可比（两列就是第 11 集的 S₁、S₂，亮黄、青）；
// 句 5「乘以转速一和负一」右边乘一列 (1, −1)，「第一列取一份」「第二列取负一份」框亮，「加起来」= (0, 0, 0, −2, 1, 0)（紫）；
// 句 6「角速度是零」前 3 个数框亮，「小臂不转，只是平移」两个关节按 (1, −1) 来回转、小臂一直水平，「每一点」小臂上三点各一支紫箭头 (−2, 1)；
// 句 7「肩先转了九十度」胳膊绕肩转 90°（同第 11 集：肘到 (−2, 1)、手到 (−4, 1)），「肘的轴挪了地方」肘的 ⊙ 亮；
// 句 8 照第 9 集：白箭头 (−2, 1) → 原点 = (2, −1)，转 90°：红 2 × (0, 1)、绿 (−1) × (−1, 0)，加起来 (1, 2)；
// 句 9「零、零、一、一、二、零」右栏新的一列，和零位时的 (0, 0, 1, 2, −1, 0) 对比，变了的两个数框亮；
// 句 10 回到斜着看的 UR5：「六乘六」右栏 6 个格子一排（J₁…J₆），「每一列」各对应一根轴，「前面的关节一转」底座来回转，后面五根轴跟着挪；
// 句 11「站在不动的底座上看」底座上画固定的坐标系 {s}，「站在手上看」手上的坐标系 {b} 亮，「第十四集讲」右栏写。
// 时刻取自配音的逐词时间，见 ../ep013.py。
(function () {
  var E = window.EP013, K = E.K, H = E.HEX;
  var RX = E.ARM.RX, X = RX + 10, D90 = Math.PI / 2;
  var T = {
    real: 0.7, pos: 2.24, ori: 3.69, vel: 5.0, twist: 5.8, w3: 7.96, v3: 9.21,
    ep9: 11.51, eq: 12.76, eqS: 13.65, one: 15.88, perSec: 17.2, tw2: 18.64, isS: 19.54,
    so: 22.42, jac: 24.3, each: 24.98, axisNow: 25.68, ep11: 28.25, out3d: 30.8,
    two: 31.22, now: 32.68, zero: 33.72, cols: 35.78, colsS: 36.32,
    times: 39.9, take1: 41.55, take2: 42.68, add: 43.99, res: 44.68,
    w0: 49.08, noRot: 50.37, trans: 51.32, every: 52.27, same: 54.05, before: 55.4, wig: [50.3, 56.4],
    rot90: 57.85, rotA: [58.4, 59.6], elbow: 59.69, el21: 60.8, moved: 61.96, col: 63.53, byNow: 64.26,
    ep9b: 67.13, toO: 68.2, d21: 70.3, turn: [71.49, 72.3], take2b: 72.49, n02: 73.61, takeM1: 74.73, n10: 76.07, sum: 77.14, n12: 77.84,
    newCol: 79.72, becomes: 80.74, nums: 81.31, diff: 84.3, out2d: 86.6,
    six: 86.98, j66: 89.12, eachCol: 90.8, axisNow2: 91.44, front: 94.03, follow: 95.48, wag: [94.0, 97.9],
    base: 99.54, other: 101.91, handV: 103.08, ep14: 104.25
  };
  var END = 106.47;
  // 斜着看的 UR5：标签摆法和每根轴画多长，同 011 第 5 幕（零位：胳膊平伸）
  var LAB = [[1, 0, -12, 'middle'], [0, -18, 6, 'end'], [0, -18, 6, 'end'], [0, -12, 16, 'end'], [0, -10, 4, 'end'], [1, 14, 6, 'start']];
  var LEN = [[-0.05, 0.32], [-0.2, 0.2], [-0.18, 0.18], [-0.16, 0.16], [-0.15, 0.15], [-0.14, 0.14]];

  E.scenes[5] = {
    title: '六个关节：每一列是此刻的螺旋轴',
    dur: END,
    cues: [
      { at: 0, s: '真的机械臂：手有位置，还有朝向——速度要用第 9 集的**运动旋量**：角速度 $3$ 个数 + 线速度 $3$ 个数' },
      { at: 11.41, s: '**运动旋量** = **转速** × **螺旋轴**（第 9 集）：只让一个关节每秒转 $1$ 弧度，**运动旋量**就是它的**螺旋轴**' },
      { at: 22.32, s: '所以写成**运动旋量**，**雅可比**的每一列，就是关节轴此刻的**螺旋轴**（第 11 集提过）' },
      { at: 31.12, s: '还是这 $2$ 个关节：现在正好是第 11 集的零位，$2$ 列就是第 11 集的 $2$ 根**螺旋轴**' },
      { at: 39.8, s: '× **转速** $(1,\\,-1)$：$1\\times$ 第 $1$ 列 $+(-1)\\times$ 第 $2$ 列 $=(0,\\,0,\\,0,\\,-2,\\,1,\\,0)$' },
      { at: 48.98, s: '角速度是 $0$：小臂不转，只是平移，每一点都以 $(-2,\\,1)$ 走——和前面算的一样 ✓' },
      { at: 57.75, s: '要是肩先转了 $90^\\circ$：肘被带到了 $(-2,\\,1)$，肘的轴挪了地方，这一列要按现在的位置算' },
      { at: 67.03, s: '照第 9 集：$(-2,\\,1)$ 指到原点是 $(2,\\,-1)$，转 $90^\\circ$：$2\\times(0,\\,1)+(-1)\\times(-1,\\,0)=(1,\\,2)$' },
      { at: 79.62, s: '这一列变成 $(0,\\,0,\\,1,\\,1,\\,2,\\,0)$，和零位时的 $(0,\\,0,\\,1,\\,2,\\,-1,\\,0)$ 不一样了' },
      { at: 86.88, s: '$6$ 个关节：**雅可比**是 $6\\times 6$，每一列都是一根关节轴此刻的**螺旋轴**——前面的关节一转，后面的轴都跟着挪' },
      { at: 97.96, s: '这样写的**运动旋量**是站在不动的底座上看的；站在手上看的另一种写法，第 14 集讲' }
    ],
    build: function (K) {
      var s = K.sceneSvg('真的机械臂：手不光有位置还有朝向，速度要用第九集的运动旋量，角速度三个数、线速度三个数；运动旋量等于转速乘以螺旋轴，只让一个关节每秒转一弧度，运动旋量就是这个关节的螺旋轴；所以写成运动旋量，雅可比的每一列就是关节轴此刻的螺旋轴。平面上的两个关节：现在正好是第十一集的零位，两列就是 S₁ = (0, 0, 1, 0, 0, 0)、S₂ = (0, 0, 1, 2, −1, 0)；乘以转速 (1, −1)：加起来 (0, 0, 0, −2, 1, 0)，角速度是零，小臂只平移，每一点都以 (−2, 1) 走；肩先转九十度，肘被带到 (−2, 1)，从 (−2, 1) 指到原点是 (2, −1)，转九十度是 (1, 2)，这一列变成 (0, 0, 1, 1, 2, 0)。六个关节的雅可比是六乘六，每一列都是一根关节轴此刻的螺旋轴，前面的关节一转后面的轴都跟着挪；这样写的运动旋量是站在不动的底座上看的，站在手上看的写法第十四集讲');

      /* ══ 斜着看的 UR5（前后两段） ══ */
      var g3 = E.el('g', {}, s);
      var R6 = E.ur6(g3, 100, 352, 2.25, 32, 16, [0, 0, 440, 420]), V = R6.view;
      var hDot = E.dot(g3, E.V, 7);
      // 底座的坐标系 {s}（三根小箭头，世界坐标）
      var baseG = E.el('g', {}, g3), baseAx = [E.X, E.Y, E.Z].map(function (c) { return E.vec(baseG, c, 4, { head: 12 }); });
      var baseL = E.halo(E.txt(baseG, 0, 0, '底座', 18, E.INK, 'middle', 900));
      var handL = E.halo(E.txt(g3, 0, 0, '手', 18, E.INK, 'middle', 900));
      var THREE = R6.THREE;

      /* ══ 平面上的两个关节（中间一段）：侧视网格 x −5…4、y −1…4，一格约 41 px ══ */
      var g2 = E.el('g', {}, s);
      var pn = E.panel(g2, { ox: 235, oy: 300, S: 1.5, x0: -5, x1: 4, y0: -1, y1: 4, box: [0, 40, 440, 380] }), P = pn.P, u = pn.u, top = pn.top;
      var jS = E.joint(top, E.WARN), jE = E.joint(top, E.WARN), hand = E.dot(top, E.V, 6.5);
      var sLab = E.halo(E.txt(top, 0, 0, 'S₁', 20, E.J1, 'end', 900)), eLab = E.halo(E.txt(top, 0, 0, 'S₂', 20, E.J2, 'start', 900));
      var e21 = E.halo(E.txt(top, P(-2, 1)[0] - 14, P(-2, 1)[1] + 30, '(−2, 1)', 18, E.WARN, 'end', 900));
      var pts = [0, 2].map(function () { return E.gv(top, P, E.V, 4.5, { head: 14 }); });   // 肘、手两点（三点挨得太近，箭头叠在一起）
      var rotArc = E.arcArrow(top, P(0, 0)[0], P(0, 0)[1], 0.8 * u, E.OMEGA, 3, { head: 11 });
      var dA = E.gv(top, P, E.INK, 4, { head: 14 }), dTurn = E.gv(top, P, E.INK, 4, { head: 14 });
      var red = E.gv(top, P, E.X, 4.5, { head: 13 }), grn = E.gv(top, P, E.Y, 4.5, { head: 13 });
      var res12 = E.gv(top, P, E.J2, 5, { head: 15 });
      var turnArc = E.arcArrow(top, P(0, 0)[0], P(0, 0)[1], 0.75 * u, E.OMEGA, 3, { head: 10 });

      /* ══ 右栏 ══ */
      var RK = E.rowKit(), Rg = E.el('g', {}, s);
      function grp() { return E.el('g', {}, Rg); }
      // 句 1–3
      var gA = grp();
      var a0 = RK.line(gA, X, 40, [['手：', E.INK2], ['位置', E.V, 900], [' + ', E.INK2], ['朝向', E.INK, 900]], 23);
      var a1 = RK.line(gA, X, 80, [['速度：', E.INK2], ['运动旋量', E.WARN, 900], ['（第 9 集）', E.MUT, 700]], 22);
      var twV = E.colvec(gA, X + 44, 210, ['ω₁', 'ω₂', 'ω₃', 'v₁', 'v₂', 'v₃'], { size: 24, dy: 30, w: 60, colors: [E.OMEGA, E.OMEGA, E.OMEGA, E.V, E.V, E.V] });
      var twA = E.txt(gA, X + 92, 171, '角速度 3 个数', 20, E.OMEGA, 'start', 900), twB = E.txt(gA, X + 92, 261, '线速度 3 个数', 20, E.V, 'start', 900);
      var gA2 = grp();
      var b0 = E.tex(gA2, X, 46, '\\textcolor{' + H.warn + '}{\\text{运动旋量}}=\\textcolor{' + H.acc + '}{\\dot\\theta}\\times\\textcolor{' + H.warn + '}{\\text{螺旋轴}}', 24, E.INK, 'start', { w: 330, h: 44 });
      var b1 = RK.line(gA2, X, 92, [['（第 9 集）', E.MUT, 700]], 17);
      var b2 = RK.line(gA2, X, 134, [['只让一个关节转：', E.INK2]], 20);
      var b3 = E.symRow(gA2, RK, X, 168, '\\dot\\theta=1', E.OMEGA, 60, [['（每秒 1 弧度）', E.INK2]], 20);
      var b4 = RK.line(gA2, X, 206, [['→ 运动旋量 = ', E.INK2], ['这个关节的螺旋轴', E.WARN, 900]], 20);
      var b5 = RK.line(gA2, X, 262, [['雅可比', E.HI, 900], ['的每一列', E.INK2]], 22);
      var b6 = RK.line(gA2, X, 298, [['= 关节轴', E.INK2], ['此刻', E.INK, 900], ['的', E.INK2], ['螺旋轴', E.WARN, 900]], 22);
      var b7 = RK.line(gA2, X, 334, [['（第 11 集提过）', E.MUT, 700]], 17);
      // 句 4–6：6 × 2 的雅可比 × (1, −1)
      var gB = grp(), MY = 196, fl = E.flow(X - 2, MY, 10);
      var Jm = fl.mat(gB, [[0, 0], [0, 0], [1, 1], [0, 2], [0, -1], [0, 0]], { size: 21, dx: 42, dy: 27, color: function (i, j) { return j === 0 ? E.J1 : E.J2; } }, 0);
      var Tm = fl.mat(gB, [[1], [-1]], { size: 21, dy: 27, color: function () { return E.OMEGA; } });
      var eqS = fl.sym(gB, '=', 22, E.INK);
      var Vm = fl.mat(gB, [[0], [0], [0], [-2], [1], [0]], { size: 21, dy: 27, wide: [['−2']], color: function () { return E.V; } });
      var bL1 = E.txt(gB, Jm.pos(0, 0)[0], MY - 98, 'S₁', 18, E.J1, 'middle', 900), bL2 = E.txt(gB, Jm.pos(0, 1)[0], MY - 98, 'S₂', 18, E.J2, 'middle', 900);
      var c0 = RK.line(gB, X, 40, [['零位（第 11 集）：', E.INK2]], 19);
      var c1r = RK.line(gB, X, 66, [['2 列 = 第 11 集的 2 根螺旋轴', E.INK, 900]], 18);
      var wLab = E.txt(gB, Vm.box[2] + 8, Vm.pos(1, 0)[1] + 6, 'ω = 0', 18, E.OMEGA, 'start', 900);
      var vLab = E.txt(gB, Vm.box[2] + 8, Vm.pos(4, 0)[1] - 6, 'v', 18, E.V, 'start', 900);
      var c2r = RK.line(gB, X, 364, [['小臂只平移：每一点都以 ', E.INK2], ['(−2, 1)', E.V, 900], [' 走', E.INK2]], 18);
      var c3r = RK.line(gB, X, 394, [['= 第 3 幕算的 ✓', E.GOOD, 900]], 18);
      // 句 7–9：肩转 90° 以后，肘那一列
      var gC = grp();
      var d0 = RK.line(gC, X, 36, [['肩先转 90°：', E.INK, 900], ['肘到了 ', E.INK2], ['(−2, 1)', E.WARN, 900]], 19);
      var d1 = RK.line(gC, X, 68, [['这一列按现在的位置算', E.INK2]], 18);
      var d2 = RK.line(gC, X, 108, [['照第 9 集：', E.INK2], ['(−2, 1)', E.WARN, 900], [' 指到原点 = ', E.INK2], ['(2, −1)', E.INK, 900]], 18);
      var d3 = RK.line(gC, X, 140, [['转 90°：', E.OMEGA, 900], ['2 × ', E.INK, 900], ['(0, 1)', E.X, 900], [' + ', E.INK2], ['(−1) × ', E.INK, 900], ['(−1, 0)', E.Y, 900]], 18);
      var d4 = RK.line(gC, X, 170, [['= ', E.INK2], ['(0, 2)', E.X, 900], [' + ', E.INK2], ['(1, 0)', E.Y, 900], [' = ', E.INK2], ['(1, 2)', E.J2, 900]], 18);
      var colOld = E.colvec(gC, X + 70, 300, [0, 0, 1, 2, -1, 0], { size: 20, dy: 23, w: 44, colors: [E.MUT, E.MUT, E.MUT, E.MUT, E.MUT, E.MUT], bracket: E.MUT });
      var colNew = E.colvec(gC, X + 230, 300, [0, 0, 1, 1, 2, 0], { size: 20, dy: 23, w: 44, colors: [E.J2, E.J2, E.J2, E.J2, E.J2, E.J2], bracket: E.J2 });
      var oldL = E.txt(gC, X + 70, 214, '零位时', 17, E.MUT, 'middle', 900), newL = E.txt(gC, X + 230, 214, '肩转 90° 后', 17, E.J2, 'middle', 900);
      var arrOld = E.vec(gC, E.INK2, 3, { head: 12 }); arrOld.set(X + 110, 300, X + 190, 300);
      var neq = E.txt(gC, X + 150, 290, '≠', 24, E.ERR, 'middle', 900);
      // 句 10–11：6 × 6
      var gD = grp();
      var e0 = RK.line(gD, X, 40, [['6 个关节：', E.INK, 900], ['雅可比', E.HI, 900], [' 6 × 6', E.INK, 900]], 22);
      var boxes = [0, 1, 2, 3, 4, 5].map(function (k) {
        var g = E.el('g', {}, gD), x0 = X + 6 + k * 52;
        K.paint(E.el('rect', { x: x0, y: 64, width: 42, height: 132, rx: 8, 'stroke-width': 2.5 }, g), E.tint(E.WARN, 14, 'var(--demo-surface)'), E.WARN);
        E.txt(g, x0 + 21, 136, 'S' + '₁₂₃₄₅₆'[k], 20, E.WARN, 'middle', 900);
        return g;
      });
      var e1 = RK.line(gD, X, 230, [['每一列：一根关节轴', E.INK2], ['此刻', E.INK, 900], ['的螺旋轴', E.WARN, 900]], 19);
      var e2 = RK.line(gD, X, 264, [['前面的关节一转，后面的轴都跟着挪', E.INK2]], 18);
      var e3 = RK.line(gD, X, 312, [['站在不动的', E.INK2], ['底座', E.INK, 900], ['上看', E.INK2]], 19);
      var e4 = RK.line(gD, X, 346, [['站在', E.INK2], ['手', E.INK, 900], ['上看：另一种写法', E.INK2]], 19);
      var e5 = RK.line(gD, X, 378, [['（第 14 集）', E.MUT, 700]], 17);

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }
      function rot(v, a) { return [v[0] * Math.cos(a) - v[1] * Math.sin(a), v[0] * Math.sin(a) + v[1] * Math.cos(a)]; }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) { RK.layout(); laid = true; }
          var f = E.fade, b = E.bump;
          var on3 = (1 - fr(st, T.out3d)) + fr(st, T.out2d), on2 = fr(st, T.out3d) * (1 - fr(st, T.out2d));
          E.op(g3, on3); E.op(g2, on2);
          /* ── 斜着看的 UR5 ── */
          if (on3 > 0) {
            var elW = -E.win(st, T.one, T.so, 0.5) * 0.3 * (0.5 - 0.5 * Math.cos((st - T.one) * 2.2));   // 只往上抬：往下转手会伸出视景框
            var pan = E.win(st, T.wag[0], T.wag[1], 0.5) * 0.32 * Math.sin((st - T.wag[0]) * 1.9);
            V.joints({ shoulder_pan_joint: pan, shoulder_lift_joint: 0, elbow_joint: elW, wrist_1_joint: 0, wrist_2_joint: 0, wrist_3_joint: 0 });
            R6.update(LAB, LEN);
            V.clearTint();
            if (st < T.out3d) V.tint(['forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link'], E.J2, 0.5 * E.win(st, T.one, T.so, 0.4));
            var allAx = st < T.out3d ? fr(st, T.each) : fr(st, T.six);
            R6.lines.forEach(function (ln, k) {
              var on = k === 2 ? Math.max(E.win(st, T.one, T.so + 0.5, 0.3), allAx) : allAx;
              E.op(ln, on); E.op(R6.labs[k], on);
              E.glow(ln, E.WARN, (k === 2 && E.win(st, T.isS, T.so, 0.3) > 0.5) || (st > T.out2d && b(st, T.eachCol + k * 0.3, 0.6) > 0.05) || (st > T.wag[0] && st < T.wag[1] && k > 0));
            });
            var h = R6.handAt(), hp = V.project(h.o);
            hDot.at(hp, 7 + 4 * b(st, T.pos, 0.8)); E.op(hDot, fr(st, T.pos));
            E.op(R6.handG, Math.max(E.win(st, T.ori, T.out3d, 0.3), fr(st, T.handV)));
            E.glow(R6.handG, E.INK, b(st, T.handV, 1.2) > 0.05);
            handL.setAttribute('x', (hp[0] + 6).toFixed(1)); handL.setAttribute('y', (hp[1] + 40).toFixed(1)); E.op(handL, Math.max(E.win(st, T.pos, T.out3d, 0.3), fr(st, T.handV)));
            // 底座的坐标系 {s}：画在底座旁边（世界原点）
            var o0 = V.project(new THREE.Vector3(0, 0, 0));
            [[0.16, 0, 0], [0, 0.16, 0], [0, 0, 0.16]].forEach(function (d, k) { var q = V.project(new THREE.Vector3(d[0], d[1], d[2])); baseAx[k].set(o0[0], o0[1], q[0], q[1]); });
            baseL.setAttribute('x', (o0[0] - 4).toFixed(1)); baseL.setAttribute('y', (o0[1] + 34).toFixed(1));
            E.op(baseG, fr(st, T.base)); E.glow(baseG, E.INK, b(st, T.base, 1.2) > 0.05);
          }
          /* ── 平面上的两个关节 ── */
          if (on2 > 0) {
            var w = E.win(st, T.wig[0], T.wig[1], 0.5), phi = 0.3 * Math.sin((st - T.wig[0]) * 1.8) * w;
            var r90 = D90 * f(st, T.rotA[0], T.rotA[1] - T.rotA[0]);
            pn.pose(E.ZERO[0] + phi + r90, r90);
            pn.view.clearTint();
            var J = pn.joints(), pE = P(J[1][0], J[1][1]), pH = P(J[2][0], J[2][1]);
            jS.at(P(0, 0)); jE.at(pE); hand.at(pH);
            E.glow(jE, E.WARN, b(st, T.moved, 1.0) > 0.05 || b(st, T.col, 0.8) > 0.05);
            sLab.setAttribute('x', (P(0, 0)[0] + 40).toFixed(1)); sLab.setAttribute('y', (P(0, 0)[1] + 30).toFixed(1));   // 原点右下：左上会被转动的胳膊和箭头扫过
            var rotd = st > T.rotA[0];   // 零位时写在肘的右下（上方是紫色的速度箭头）；转过 90° 以后写在左上（右下是从肘指到原点的白箭头）
            eLab.setAttribute('x', (pE[0] + (rotd ? -16 : 14)).toFixed(1)); eLab.setAttribute('y', (pE[1] + (rotd ? -14 : 30)).toFixed(1)); eLab.style.textAnchor = rotd ? 'end' : 'start';
            E.op(sLab, fr(st, T.colsS) * (1 - fr(st, T.rot90))); E.op(eLab, fr(st, T.colsS));   // 原点那边接着要画转 90°，S₁ 先收
            E.op(e21, fr(st, T.el21) * (1 - fr(st, T.toO)));
            // 句 6：小臂上三点的速度（按 (1, −1)：都是 rot90(肘的位置)）
            var ve = rot(J[1], D90), on6 = E.win(st, T.every, T.rot90, 0.3);
            pts.forEach(function (a, k) { var x0 = J[1][0] + 2 * k, y0 = J[1][1]; a.at(x0, y0, x0 + ve[0], y0 + ve[1]); E.op(a.g, on6); });
            rotArc.set(Math.atan2(2, 1), Math.atan2(2, 1) + Math.max(r90, 0.01)); E.op(rotArc.g, (r90 > 0.01 ? 1 : 0) * (1 - fr(st, T.toO)));
            // 句 8：(−2, 1) → 原点，转 90°，红绿两段
            var g8 = f(st, T.toO, 0.8), tr = f(st, T.turn[0], T.turn[1] - T.turn[0]);
            dA.at(-2, 1, -2 + 2 * g8, 1 - 1 * g8); E.op(dA.g, (st > T.toO ? 1 : 0) * (1 - fr(st, T.out2d)));
            var dv = rot([2, -1], D90 * tr);
            dTurn.at(0, 0, dv[0], dv[1]); E.op(dTurn.g, (st > T.turn[0] ? 1 : 0) * (1 - fr(st, T.sum)));
            turnArc.set(Math.atan2(-1, 2), Math.atan2(-1, 2) + D90 * Math.max(tr, 0.01)); E.op(turnArc.g, (tr > 0.01 ? 1 : 0) * (1 - fr(st, T.take2b)));
            red.at(0, 0, 0, 2 * f(st, T.take2b, 0.6)); E.op(red.g, (st > T.take2b ? 1 : 0) * (1 - fr(st, T.out2d)));
            grn.at(0, 2, 1 * f(st, T.takeM1, 0.6), 2); E.op(grn.g, (st > T.takeM1 ? 1 : 0) * (1 - fr(st, T.out2d)));
            res12.at(0, 0, 1, 2); E.op(res12.g, fr(st, T.n12) * (1 - fr(st, T.out2d)));
          }

          /* ── 右栏 ── */
          E.op(gA, 1 - fr(st, T.ep9 - 0.3));
          E.op(a0.g, fr(st, T.pos)); E.op(a0.parts[1], fr(st, T.pos)); E.op(a0.parts[2], fr(st, T.ori)); E.op(a0.parts[3], fr(st, T.ori));
          E.op(a1.g, fr(st, T.vel)); E.op(a1.parts[1], fr(st, T.twist)); E.op(a1.parts[2], fr(st, T.twist));
          E.op(twV.g, fr(st, T.twist)); E.op(twA, fr(st, T.w3)); E.op(twB, fr(st, T.v3));
          twV.rangeBox(0, 2, E.OMEGA, E.win(st, T.w3, T.v3 + 1.2, 0.3)); twV.rangeBox(3, 5, E.V, E.win(st, T.v3, T.ep9, 0.3));
          E.op(gA2, E.win(st, T.ep9 - 0.2, T.out3d, 0.3));
          E.op(b0, fr(st, T.eq)); E.op(b1.g, fr(st, T.ep9)); E.op(b2.g, fr(st, T.one)); E.op(b3.g, fr(st, T.perSec)); E.op(b4.g, fr(st, T.tw2)); E.op(b4.parts[1], fr(st, T.isS));
          E.op(b5.g, fr(st, T.jac)); E.op(b6.g, fr(st, T.axisNow)); E.op(b7.g, fr(st, T.ep11));
          E.op(gB, E.win(st, T.two - 0.2, T.rot90, 0.3));
          E.op(c0.g, fr(st, T.zero)); E.op(c1r.g, fr(st, T.cols));
          E.op(Jm.g, fr(st, T.cols)); E.op(bL1, fr(st, T.colsS)); E.op(bL2, fr(st, T.colsS));
          Jm.hl('a', 0, 0, 5, 0, E.J1, E.win(st, T.take1, T.take2, 0.3)); Jm.hl('b', 0, 1, 5, 1, E.J2, E.win(st, T.take2, T.add, 0.3));
          E.op(Tm.g, fr(st, T.times)); E.op(eqS, fr(st, T.add)); E.op(Vm.g, fr(st, T.res));
          Tm.hl('a', 0, 0, 0, 0, E.OMEGA, E.win(st, T.take1, T.take2, 0.3)); Tm.hl('b', 1, 0, 1, 0, E.OMEGA, E.win(st, T.take2, T.add, 0.3));
          Vm.hl('w', 0, 0, 2, 0, E.OMEGA, fr(st, T.w0)); Vm.hl('v', 3, 0, 4, 0, E.V, fr(st, T.same));
          E.op(wLab, fr(st, T.w0)); E.op(vLab, fr(st, T.same));
          E.op(c2r.g, fr(st, T.every)); E.op(c2r.parts[1], fr(st, T.same)); E.op(c2r.parts[2], fr(st, T.same)); E.op(c3r.g, fr(st, T.before));
          E.op(gC, E.win(st, T.rot90 - 0.2, T.out2d, 0.3));
          E.op(d0.g, fr(st, T.rot90)); E.op(d0.parts[1], fr(st, T.elbow)); E.op(d0.parts[2], fr(st, T.el21)); E.op(d1.g, fr(st, T.byNow));
          E.op(d2.g, fr(st, T.ep9b)); E.op(d2.parts[1], fr(st, T.toO)); E.op(d2.parts[2], fr(st, T.d21)); E.op(d2.parts[3], fr(st, T.d21));
          E.op(d3.g, fr(st, T.turn[0])); E.op(d3.parts[1], fr(st, T.take2b)); E.op(d3.parts[2], fr(st, T.take2b)); [3, 4, 5].forEach(function (k) { E.op(d3.parts[k], fr(st, T.takeM1)); });
          E.op(d4.g, fr(st, T.n02)); E.op(d4.parts[2], fr(st, T.n10)); E.op(d4.parts[3], fr(st, T.n10)); E.op(d4.parts[4], fr(st, T.n12)); E.op(d4.parts[5], fr(st, T.n12));
          E.op(colOld.g, fr(st, T.newCol)); E.op(oldL, fr(st, T.newCol));
          E.op(colNew.g, fr(st, T.nums)); E.op(newL, fr(st, T.nums)); E.op(arrOld.g, fr(st, T.becomes)); E.op(neq, fr(st, T.diff));
          colOld.rangeBox(3, 4, E.ERR, fr(st, T.diff)); colNew.rangeBox(3, 4, E.ERR, fr(st, T.diff));
          E.op(gD, fr(st, T.six - 0.2));
          E.op(e0.g, fr(st, T.six)); E.op(e0.parts[1], fr(st, T.j66)); E.op(e0.parts[2], fr(st, T.j66));
          boxes.forEach(function (g, k) { E.op(g, fr(st, T.j66 + 0.12 * k)); E.glow(g, E.WARN, b(st, T.eachCol + k * 0.3, 0.6) > 0.05); });
          E.op(e1.g, fr(st, T.eachCol)); E.op(e1.parts[1], fr(st, T.axisNow2)); E.op(e1.parts[2], fr(st, T.axisNow2)); E.op(e2.g, fr(st, T.front));
          E.op(e3.g, fr(st, T.base)); E.op(e4.g, fr(st, T.other)); E.op(e5.g, fr(st, T.ep14));
        }
      };
    }
  };
})();
