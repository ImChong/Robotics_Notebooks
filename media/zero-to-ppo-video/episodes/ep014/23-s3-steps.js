// 014 第 3 幕：先换一个点，再换一套轴——把只转肘那一列从底座上看的换成手上看的（《Modern Robotics》式 (3.84)：v_b = Rᵀ(v_s + ω_s × p)；
// 平面里 ω × p 用第 9 集「从原点指向这一点、转 90°、再乘以角速度」）。左边这一集的姿势（肩先转了 90°），底座、手两个坐标系一直画着。
// 句 1 右栏写问题和两步；句 2–7「只转肘」肘旁蓝色小弯箭头，站在底座上看把这一列逐步算出来（2026-11-10 用户反馈原先没交代 (0, 0, 1, 1, 2, 0) 怎么来的）：
// 前 3 个数字绕 z 轴每秒 1 弧度 (0, 0, 1)；后 3 个数是原点那一点的速度（板子跟着小臂绕肘转，虚线弧），从肘关节指向原点 (2, −1)，绕肘转 90°，挪到原点，
// 红 2 × (0, 1)、绿 (−1) × (−1, 0) 头尾相接 = 紫 (1, 2)，× 转速 1 还是 (1, 2)；以下句号顺延；
// 句 3「换一个点」原点、手各画白圈，紫色虚线箭头从原点指到手；句 4 右栏写第 9 集的式子；
// 句 5「从原点指向手」白虚线箭头 (−2, 3)，「转九十度」绕原点转到 (−3, −2)（蓝弧），「乘上去」还是 (−3, −2)；
// 句 6「原点那一点的速度」原点画 (1, 2)（紫，细），「加上」(−3, −2) 挪到它的尖上头尾相接，「手这一点的速度」从原点到 (−2, 0) 的紫箭头，再整支挪到手上；
// 句 7 验算：从肘关节指向手 (0, 2)（白虚线，压在小臂上），绕肘转 90° 是 (−2, 0)，「一样」打勾；
// 句 8「换一套轴」手的两根轴亮：「x 轴潮上」沿 x 轴是 0（红色的 0），「y 轴潮左」沿 y 轴是 2（绿箭头盖在紫箭头上）；
// 句 9 右栏写 (0, 0, 1, 0, 2, 0)；句 10 右栏两个 6 × 2：站在手上看的（物体雅可比 J_b）、站在底座上看的（空间雅可比 J_s）。
// 时刻取自配音的逐词时间，见 ../ep014.py。
(function () {
  var E = window.EP014, K = E.K, H = E.HEX;
  var X = 455, D90 = Math.PI / 2;
  var T = {
    rev: 0.7, hand6: 1.72, base6: 4.48, calc: 5.34, two: 6.96, steps: 7.97,
    elbow: 10.55, perSec: 11.52, baseV: 13.61, calcCol: 15.79,
    w3: 17.78, wz: 19.76, wNums: [22.23, 22.74, 23.21],
    v3: 24.7, board: 25.83, oPt: 26.86, now: 28.75, withF: 30.13, aroundE: 31.76,
    ep9d: 33.54, eToO: 34.83, oMinus: 36.56, dx: 37.91, dy: 39.62, got21: 40.23,
    turnD: [42.6, 43.35], mvD: [43.4, 44.05], take2: 44.18, r02: 44.97, takeM1: 46.39, r10: 47.48, addD: 48.41, sumD: 49.33, times1: 50.99, still12: 52.11,
    so: 53.96, baseSo: 54.62, col31: 55.59, nums: [56.53, 57.01, 57.51, 57.98, 58.47, 58.9], ep13: 59.44,
    s1: 63.04, pt: 63.65, fromO: 64.57, toH: 65.95, out3: 67.72,
    ep9: 68.28, f1: 70.11, f2: 71.44, f3: 73.21, f4: 75.29, f5: 76.47,
    oh: 97.14, n23: 98.56, turn: [99.73, 101.15], before: 100.5, r32: 106.45, cT1: 101.35, cT2: 103.89, w1: 108.02, times: 109.1, still: 109.89,
    vo: 112.53, v12: 113.91, plus: 114.7, m32: 115.09, a1: 116.29, a2: 118.09, hv: 120.05, res: 121.68, mvH: [120.22, 121.43],
    chk: 123.87, onlyE: 124.66, around: 125.78, eh: 126.74, n02: 128.56, turn2: [129.54, 130.32], r20: 130.77, same: 131.94, out7: 133.12,
    s2: 133.55, axes: 134.24, n20: 135.33, left: 136.75, noUD: 137.98, xUp: 139.7, x0: 141.76, yLeft: 142.34, y2: 144.43,
    w: 145.99, standH: 147.61, colB: 148.85, numsB: [150.05, 150.59, 151.15, 151.7, 152.29, 152.74],
    two2: 154.18, jb: 155.64, book: 157.49, jbName: 158.26, js: 159.74, jsName: 161.33,
    ex0: 78.58, exE: 80.44, exO: 83.91, exV: 85.29, exP: 86.42, exR: 88.91, exT: 90.12, exT2: 91.1, exAdd: 92.2, exZ: 94.43, exOk: 94.93
  };
  var END = 164.16;

  E.scenes[2] = {
    title: '先换一个点，再换一套轴',
    dur: END,
    cues: [
      { at: 0, s: '反过来：手上看到的 $6$ 个数，能不能从底座看到的算出来？照 $2$ 处不同，分 $2$ 步' },
      { at: 9.78, s: '只转肘，每秒转 $1$ 弧度：先**站在底座上看**，把这一列算出来' },
      { at: 17.68, s: '前 $3$ 个数字是角速度：绕 $z$ 轴每秒转 $1$ 弧度，$(0,\\,0,\\,1)$' },
      { at: 24.6, s: '后 $3$ 个数：透明板上原点那一点的速度——板子跟着小臂绕肘转' },
      { at: 33.44, s: '照第 9 集：从肘关节指向原点 $(0-(-2),\\,0-1)=(2,\\,-1)$' },
      { at: 42.5, s: '转 $90^\\circ$：$2\\times(0,\\,1)+(-1)\\times(-1,\\,0)=(1,\\,2)$，$\\times 1$ 还是 $(1,\\,2)$' },
      { at: 53.86, s: '所以**站在底座上看**，这一列是 $(0,\\,0,\\,1,\\,1,\\,2,\\,0)$（第 13 集第 6 幕也这样算）' },
      { at: 62.94, s: '第 $1$ 步，**换一个点**：从原点那一点，换到手这一点' },
      { at: 68.18, s: '第 9 集：板上一点的速度 $=$ 原点那一点的速度 $+$（从原点指向这一点，转 $90^\\circ$）$\\times$ 角速度' },
      { at: 78.53, s: '第 9 集验算肘：$(4,\\,-2)+(-4,\\,2)=(0,\\,0)$，肘果然不动——这条规矩就是这样来的' },
      { at: 97.04, s: '从原点指向手 $(-2,\\,3)$，转 $90^\\circ$：$(0,\\,-2)+(-3,\\,0)=(-3,\\,-2)$，$\\times 1$ 还是 $(-3,\\,-2)$' },
      { at: 112.43, s: '$(1,\\,2)+(-3,\\,-2)=(-2,\\,0)$：手这一点的速度' },
      { at: 123.77, s: '验算：只转肘，从肘关节指向手 $(0,\\,2)$，转 $90^\\circ$ 是 $(-2,\\,0)$ ✓' },
      { at: 133.45, s: '第 $2$ 步，**换一套轴**：$(-2,\\,0)$ 往左 $2$ 格——沿手的 $x$ 轴（朝上）是 $0$，沿手的 $y$ 轴（朝左）是 $2$' },
      { at: 145.89, s: '角速度还是 $1$：**站在手上看**，这一列是 $(0,\\,0,\\,1,\\,0,\\,2,\\,0)$' },
      { at: 154.08, s: '站在手上看的雅可比：**物体雅可比**；站在底座上看的：**空间雅可比**' }
    ],
    build: function (K) {
      var s = K.sceneSvg('反过来：手上看到的六个数，能不能直接从底座看到的算出来？分两步。只转肘，站在底座上看这一列是 (0, 0, 1, 1, 2, 0)。第一步换一个点：从原点那一点换到手这一点；第九集验算肘时用过，板上一点的速度等于原点那一点的速度，加上从原点指向这一点、转九十度、再乘以角速度：从原点指向手 (−2, 3)，转九十度是 (−3, −2)，乘以角速度一还是 (−3, −2)；(1, 2) + (−3, −2) = (−2, 0)。验算：从肘关节指向手 (0, 2)，转九十度是 (−2, 0)。第二步换一套轴：(−2, 0) 是往左走两格，手的 x 轴朝上，沿 x 轴是 0；手的 y 轴朝左，沿 y 轴是 2。角速度还是一，站在手上看这一列是 (0, 0, 1, 0, 2, 0)。两列排在一起是物体雅可比，站在底座上看的叫空间雅可比');

      var pn = E.panel14(s), P = pn.P, u = pn.u, top = pn.top, O = P(0, 0);
      pn.pose(E.P90[0], E.P90[1]);
      var J = pn.joints(), pE = P(J[1][0], J[1][1]), pH = P(J[2][0], J[2][1]);
      var bF = E.frame2(top, P, 1, '底座', { labAt: [0.75, -0.62] }), hF = E.frame2(top, P, 1, '手', { labAt: [0.62, 0.5], labColor: E.V, tipOff: [[-0.3, -0.05], [0.1, 0.38]] });   // 手的 y 轴朝左，箭头尖外面要画 (−2, 0)：字母写在尖的上面
      bF.set(0, 0, 0); hF.set(-2, 3, D90);
      var jS = E.joint(top, E.HI), jE = E.joint(top, E.HI), hand = E.dot(top, E.V, 7);
      jS.at(O); jE.at(pE); hand.at(pH);
      var eArc = E.arcArrow(top, pE[0], pE[1], 0.55 * u, E.OMEGA, 3.5, { head: 12 });
      eArc.set(3.3, 4.5);   // 肘的左下（右下是大臂）
      var eLab = E.tex(top, pE[0] - 112, pE[1] + 50, '\\dot\\theta_2=1', 22, E.OMEGA, 'start', { w: 90, h: 36 });
      function ring(p) { return K.paint(E.el('circle', { cx: p[0], cy: p[1], r: 14, fill: 'none', 'stroke-width': 3.5 }, top), null, E.INK); }
      var rgO = ring(O), rgH = ring(pH);
      var mvA = E.gv(top, P, E.STEP1, 3.5, { head: 13, dash: '7 6' });
      // 句 5–6：白 (−2, 3) → 转 90° (−3, −2)；原点的 (1, 2)；头尾相接；紫 (−2, 0)
      var wA = E.gv(top, P, E.INK, 4, { head: 14, dash: '9 6' }), wT = E.gv(top, P, E.INK, 4, { head: 14 });
      var tArc = E.arcArrow(top, O[0], O[1], 0.75 * u, E.OMEGA, 3, { head: 10 });
      var vO = E.gv(top, P, E.V, 3.5, { head: 13 }), sum = E.gv(top, P, E.V, 5.5, { head: 17 });
      [wT, vO, sum].forEach(function (a) { a.line.style.filter = 'drop-shadow(0 0 2px #000)'; });
      var wL = E.halo(E.txt(top, P(-1.25, 1.75)[0] - 10, P(-1.25, 1.75)[1] - 2, '(−2, 3)', 19, E.INK, 'end', 900));   // 白箭头左上（右边是底座 y 轴的字母）
      var tL = E.halo(E.txt(top, 0, 0, '(−3, −2)', 18, E.INK, 'middle', 900));
      var vOL = E.halo(E.txt(top, P(0.5, 1)[0] + 14, P(0.5, 1)[1] + 4, '(1, 2)', 19, E.V, 'start', 900));
      // 句 2–7（2026-11-10 用户反馈「(0, 0, 1, 1, 2, 0) 没有交代怎么算出来的」）：只转肘、站在底座上看，原点那一点绕肘走圆；
      // 从肘关节指向原点 (2, −1)（白虚线，挪到大臂下面一点，不压在大臂上），绕肘转 90° 到 (1, 2)，整支挪到原点，红 (0, 2)、绿 (1, 0) 头尾相接，紫 (1, 2) 就是上面的 vO
      var ringD = K.paint(E.el('circle', { cx: O[0], cy: O[1], r: 14, fill: 'none', 'stroke-width': 3.5 }, top), null, E.INK);
      var arcD = E.arcArrow(top, pE[0], pE[1], Math.sqrt(5) * u, E.V, 2.5, { head: 12, dash: '7 6' });
      var aEO = Math.atan2(-1, 2);
      arcD.set(aEO - 0.8, aEO + 0.18);   // 箭头尖刚过原点（再往前会压到底座的 y 轴）
      var oDL = E.halo(E.txt(top, P(0.45, -1.15)[0], P(0.45, -1.15)[1] + 6, '原点那一点', 17, E.INK, 'start', 900));
      var dA = E.gv(top, P, E.INK, 4, { head: 14, dash: '9 6' }), dT = E.gv(top, P, E.INK, 4, { head: 14 });
      var dArc = E.arcArrow(top, pE[0], pE[1], 0.75 * u, E.OMEGA, 3, { head: 10 });
      var dL = E.halo(E.txt(top, P(-1.15, 0.05)[0] - 6, P(-1.15, 0.05)[1] + 22, '(2, −1)', 18, E.INK, 'end', 900));
      var dRed = E.gv(top, P, E.X, 5, { head: 14 }), dGrn = E.gv(top, P, E.Y, 5, { head: 14 });
      [dT, dRed, dGrn].forEach(function (a) { a.line.style.filter = 'drop-shadow(0 0 2px #000)'; });
      var dRedL = E.halo(E.txt(top, P(0, 1.2)[0] - 12, P(0, 1.2)[1] + 6, '(0, 2)', 17, E.X, 'end', 900));
      var dGrnL = E.halo(E.txt(top, P(0.5, 2)[0], P(0.5, 2)[1] - 14, '(1, 0)', 17, E.Y, 'middle', 900));
      var sumL = E.halo(E.txt(top, 0, 0, '(−2, 0)', 19, E.V, 'middle', 900));
      // 句 7：从肘关节指向手 (0, 2)，绕肘转 90°
      var eA = E.gv(top, P, E.INK, 4, { head: 14, dash: '9 6' }), eT = E.gv(top, P, E.INK, 4, { head: 14 });
      var eTArc = E.arcArrow(top, pE[0], pE[1], 0.7 * u, E.OMEGA, 3, { head: 10 });
      var eTL = E.halo(E.txt(top, P(-4, 1)[0] - 10, P(-4, 1)[1] + 7, '(−2, 0)', 18, E.INK, 'end', 900));
      var ok7 = E.check(top, P(-4.5, 3.4)[0], P(-4.5, 3.4)[1] - 18, 22, E.GOOD, 4);
      // 句 8：沿手的 x 轴 0（红）、沿 y 轴 2（绿）
      var x0L = E.halo(E.txt(top, P(-2, 4)[0] + 34, P(-2, 4)[1] + 8, '→ 0', 22, E.X, 'start', 900));
      var gY = E.gv(top, P, E.Y, 5, { head: 15 });
      var y2L = E.halo(E.txt(top, P(-3, 3)[0], P(-3, 3)[1] + 30, '2', 22, E.Y, 'middle', 900));

      /* ══ 右栏 ══ */
      var RK = E.rowKit(), Rg = E.el('g', {}, s);
      function grp() { return E.el('g', {}, Rg); }
      var gA = grp();
      var a1 = RK.line(gA, X, 60, [['反过来：', E.INK, 900]], 23);
      var a2 = RK.line(gA, X, 104, [['手上看到的 6 个数', E.V, 900]], 21);
      var a3 = RK.line(gA, X, 140, [['能不能从', E.INK2], ['底座看到的', E.INK, 900], ['算出来？', E.INK2]], 21);
      var a4 = RK.line(gA, X, 200, [['照 2 处不同，分 2 步：', E.INK2]], 21);
      var a5 = RK.line(gA, X, 240, [['① 换一个点', E.STEP1, 900]], 22);
      var a6 = RK.line(gA, X, 278, [['② 换一套轴', E.STEP2, 900]], 22);
      var gB = grp();
      var b1 = RK.line(gB, X, 40, [['只转肘', E.J2, 900], ['，每秒 1 弧度，站在底座上看：', E.INK2]], 18);
      var b2 = RK.line(gB, X, 72, [['(0, 0, 1, ', E.OMEGA, 900], ['1, 2, 0)', E.V, 900]], 22);
      var gD0 = grp();
      var q1 = RK.line(gD0, X, 116, [['前 3 个数字：', E.INK, 900], ['角速度 ', E.INK2], ['(0, 0, 1)', E.OMEGA, 900]], 18);
      var q2 = RK.line(gD0, X, 144, [['（绕 ', E.MUT, 700], ['z', E.Z, 900], [' 轴，每秒 1 弧度）', E.MUT, 700]], 15);
      var q3 = RK.line(gD0, X, 184, [['后 3 个数：', E.INK, 900], ['原点那一点的速度', E.V, 900]], 18);
      var q4 = RK.line(gD0, X, 210, [['（板子跟着小臂绕肘转，第 9 集）', E.MUT, 700]], 15);
      var q5 = RK.line(gD0, X, 248, [['从肘关节指向原点：', E.INK2], ['(2, −1)', E.INK, 900]], 17);
      var q6 = RK.line(gD0, X, 274, [['0 − (−2) = 2，0 − 1 = −1', E.INK2]], 15);
      var q7 = RK.line(gD0, X, 310, [['转 90°：', E.OMEGA, 900], ['2 × ', E.INK, 900], ['(0, 1)', E.X, 900], [' + ', E.INK2], ['(−1) × ', E.INK, 900], ['(−1, 0)', E.Y, 900]], 17);
      var q8 = RK.line(gD0, X, 338, [['= ', E.INK2], ['(0, 2)', E.X, 900], [' + ', E.INK2], ['(1, 0)', E.Y, 900], [' = ', E.INK2], ['(1, 2)', E.V, 900]], 17);
      var q9 = RK.line(gD0, X, 366, [['× 转速 1 → 还是 ', E.INK2], ['(1, 2)', E.V, 900]], 17);
      var q10 = RK.line(gD0, X, 396, [['第 13 集第 6 幕也这样算 ✓', E.GOOD, 900]], 16);
      var gC = grp();
      var c1 = RK.line(gC, X, 118, [['① 换一个点', E.STEP1, 900], ['：原点那一点 → 手这一点', E.INK2]], 19);
      var c2 = RK.line(gC, X, 156, [['板上一点的速度 = ', E.INK2], ['原点那一点的速度', E.V, 900]], 17);
      var c3 = RK.line(gC, X, 184, [['+（从原点指向这一点，', E.INK2], ['转 90°', E.OMEGA, 900], ['）× ', E.INK2], ['角速度', E.OMEGA, 900]], 17);
      var c3b = RK.line(gC, X, 210, [['（第 9 集）', E.MUT, 700]], 16);
      // 第 9 集是怎么验算的（2026-11-10 用户要求「不要只说第几集说过，要快速过一下怎么得到的」）：肘 (1, 2)、绕肘每秒 2 弧度，原点那一点 (4, −2)，
      // 肘的位置 (1, 2) 转 90° × 2 = (−4, 2)，加起来 (0, 0)：肘不动。写在下面几排（「从原点指向手」那几句出来之前收起）
      var gEx = grp();
      var x1 = RK.line(gEx, X, 240, [['第 9 集验算肘：', E.INK, 900], ['肘 (1, 2)，每秒 2 弧度', E.INK2]], 16);
      var x2 = RK.line(gEx, X, 266, [['原点那一点的速度 ', E.INK2], ['(4, −2)', E.V, 900]], 16);
      var x3 = RK.line(gEx, X, 292, [['肘 (1, 2) 转 90° → ', E.INK2], ['(−2, 1)', E.INK, 900], [' × 2 → ', E.INK2], ['(−4, 2)', E.INK, 900]], 16);
      var x4 = RK.line(gEx, X, 318, [['(4, −2) + (−4, 2) = ', E.INK2], ['(0, 0)', E.GOOD, 900], ['：肘不动 ✓', E.GOOD, 900]], 16);
      var c4 = RK.line(gC, X, 240, [['(−2, 3)', E.INK, 900], [' 转 90°：', E.OMEGA, 900], ['(0, −2)', E.X, 900], ['+', E.INK2], ['(−3, 0)', E.Y, 900], ['=', E.INK2], ['(−3, −2)', E.INK, 900]], 15);
      var c4b = RK.line(gC, X, 266, [['× 角速度 1 → 还是 ', E.INK2], ['(−3, −2)', E.INK, 900]], 17);
      var c5 = RK.line(gC, X, 298, [['(1, 2)', E.V, 900], [' + ', E.INK2], ['(−3, −2)', E.INK, 900], [' = ', E.INK2], ['(−2, 0)', E.V, 900]], 20);
      var c5b = RK.line(gC, X, 322, [['1 + (−3) = −2，', E.INK2], ['2 + (−2) = 0', E.INK2]], 16);
      var c6 = RK.line(gC, X, 356, [['验算：从肘关节指向手 ', E.INK2], ['(0, 2)', E.INK, 900]], 17);
      var c7 = RK.line(gC, X, 382, [['转 90° → ', E.OMEGA, 900], ['(−2, 0)', E.INK, 900], [' 一样 ', E.INK2], ['✓', E.GOOD, 900]], 17);
      var gD = grp();
      var d1 = RK.line(gD, X, 122, [['② 换一套轴', E.STEP2, 900], ['：按手的轴拿份', E.INK2]], 20);
      var d2 = RK.line(gD, X, 162, [['(−2, 0)', E.V, 900], ['：往左 2 格，不上也不下', E.INK2]], 18);
      var d3 = RK.line(gD, X, 202, [['手的 ', E.INK2], ['x', E.X, 900], [' 轴朝上 → ', E.INK2], ['0', E.X, 900]], 20);
      var d4 = RK.line(gD, X, 236, [['手的 ', E.INK2], ['y', E.Y, 900], [' 轴朝左 → ', E.INK2], ['2', E.Y, 900]], 20);
      var d5 = RK.line(gD, X, 276, [['角速度还是 ', E.INK2], ['1', E.OMEGA, 900]], 20);
      var d6 = RK.line(gD, X, 326, [['站在手上看：', E.V, 900]], 20);
      var d7 = RK.line(gD, X, 360, [['(0, 0, 1, ', E.OMEGA, 900], ['0, 2, 0)', E.V, 900]], 22);
      var gE = grp(), MY = 196;
      var JbM = E.matN(gE, X + 80, MY, [[0, 0], [0, 0], [1, 1], [-2, 0], [3, 2], [0, 0]], { size: 20, dx: 42, dy: 26, color: function (i, j) { return j === 0 ? E.J1 : E.J2; } });
      var JsM = E.matN(gE, X + 262, MY, [[0, 0], [0, 0], [1, 1], [0, 1], [0, 2], [0, 0]], { size: 20, dx: 42, dy: 26, color: function (i, j) { return j === 0 ? E.J1 : E.J2; } });
      var eL1 = E.txt(gE, X + 80, 56, '站在手上看', 18, E.V, 'middle', 900), eL2 = E.txt(gE, X + 262, 56, '站在底座上看', 18, E.INK, 'middle', 900);
      var eN1 = E.txt(gE, X + 80, 352, '物体雅可比', 20, E.INK, 'middle', 900), eN2 = E.txt(gE, X + 262, 352, '空间雅可比', 20, E.INK, 'middle', 900);
      var eS1 = E.tex(gE, X + 80, 386, 'J_b', 22, E.INK2, 'middle', { w: 50, h: 34 }), eS2 = E.tex(gE, X + 262, 386, 'J_s', 22, E.INK2, 'middle', { w: 50, h: 34 });
      var eH1 = E.txt(gE, JbM.pos(0, 0)[0], 86, '肩', 17, E.J1, 'middle', 900), eH2 = E.txt(gE, JbM.pos(0, 1)[0], 86, '肘', 17, E.J2, 'middle', 900);
      var eH3 = E.txt(gE, JsM.pos(0, 0)[0], 86, '肩', 17, E.J1, 'middle', 900), eH4 = E.txt(gE, JsM.pos(0, 1)[0], 86, '肘', 17, E.J2, 'middle', 900);

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) { RK.layout(); laid = true; }
          var f = E.fade, b = E.bump;
          pn.pose(E.P90[0], E.P90[1]); pn.view.clearTint();
          E.glow(jE, E.HI, b(st, T.elbow, 0.8) > 0.05 || b(st, T.onlyE, 0.8) > 0.05);
          var eOn = E.win(st, T.elbow, T.s1, 0.3) + E.win(st, T.onlyE, T.out7, 0.3);
          E.op(eArc.g, eOn); E.op(eLab, eOn);
          E.op(rgO, E.win(st, T.fromO, T.out3, 0.3)); E.op(rgH, E.win(st, T.toH, T.out3, 0.3));
          mvA.at(0.25, 0.3, -1.75, 2.7); E.op(mvA.g, E.win(st, T.toH, T.out3, 0.3));
          E.glow(hand, E.V, b(st, T.toH, 1.0) > 0.05 || b(st, T.hv, 1.0) > 0.05);
          [bF.lx, bF.ly].forEach(function (n) { E.op(n, 1 - E.win(st, T.take2 - 0.3, T.s1, 0.25) - E.win(st, T.plus - 0.2, T.hv + 0.3, 0.25)); });   // (−3, −2) 挪到 (1, 2) 的尖上时从底座的 x、y 字母上扫过：先收起，(1, 2) 收起后再出
          // 只转肘、站在底座上看：算出 (0, 0, 1, 1, 2, 0)
          var onD = E.win(st, T.oPt, T.s1 - 0.3, 0.3);
          E.op(ringD, onD); E.op(oDL, onD); E.op(arcD.g, E.win(st, T.aroundE, T.s1 - 0.3, 0.3));
          dA.at(-2.08, 0.84, -0.08, -0.16); E.op(dA.g, E.win(st, T.eToO, T.turnD[0] + 0.2, 0.3)); E.op(dL, E.win(st, T.got21, T.turnD[0], 0.3));
          var trD = f(st, T.turnD[0], T.turnD[1] - T.turnD[0]), angD = aEO + D90 * trD, mvk = f(st, T.mvD[0], T.mvD[1] - T.mvD[0]), R5 = Math.sqrt(5);
          var dx0 = -2 + 2 * mvk, dy0 = 1 - 1 * mvk;   // 转完以后整支从肘挪到原点
          dT.at(dx0, dy0, dx0 + R5 * Math.cos(angD), dy0 + R5 * Math.sin(angD)); E.op(dT.g, fr(st, T.turnD[0] - 0.2) * (1 - fr(st, T.addD)));
          dArc.set(aEO, aEO + D90 * Math.max(trD, 0.01)); E.op(dArc.g, (trD > 0.01 ? 1 : 0) * (1 - fr(st, T.mvD[0])));
          dRed.at(0, 0, 0, 2 * f(st, T.take2, 0.6)); E.op(dRed.g, (st > T.take2 ? 1 : 0) * (1 - fr(st, T.so)));
          dGrn.at(0, 2, 1 * f(st, T.takeM1, 0.6), 2); E.op(dGrn.g, (st > T.takeM1 ? 1 : 0) * (1 - fr(st, T.so)));
          E.op(dRedL, E.win(st, T.r02, T.so, 0.3)); E.op(dGrnL, E.win(st, T.r10, T.so, 0.3));
          // 句 5：白 (−2, 3)，绕原点转 90° 到 (−3, −2)；句 6 挪到 (1, 2) 的尖上
          var on6 = 1 - fr(st, T.hv);
          wA.at(0, 0, -2, 3); E.op(wA.g, E.win(st, T.oh, T.vo, 0.3));
          E.op(wL, E.win(st, T.n23, T.turn[0], 0.3));
          var tr = f(st, T.turn[0], T.turn[1] - T.turn[0]), ang = Math.atan2(3, -2) + D90 * tr, L13 = Math.sqrt(13);
          var sh = f(st, T.plus, 0.9), ox = sh * 1, oy = sh * 2;
          wT.at(ox, oy, ox + L13 * Math.cos(ang), oy + L13 * Math.sin(ang)); E.op(wT.g, fr(st, T.turn[0] - 0.2) * on6);
          tArc.set(Math.atan2(3, -2), Math.atan2(3, -2) + D90 * Math.max(tr, 0.01)); E.op(tArc.g, (tr > 0.01 ? 1 : 0) * (1 - fr(st, T.before + 0.6)));
          var tq = P(ox - 1.5, oy - 1); tL.setAttribute('x', (tq[0] - 26).toFixed(1)); tL.setAttribute('y', (tq[1] + 22).toFixed(1));
          E.op(tL, E.win(st, T.r32, T.plus, 0.3));
          vO.at(0, 0, 1, 2); E.op(vO.g, fr(st, T.sumD) * on6); E.op(vOL, fr(st, T.sumD + 0.3) * on6);   // (1, 2) 在第 2 句算出来，一直留到第 6 句
          // 紫 (−2, 0)：先从原点画，再整支挪到手上
          var mh = f(st, T.mvH[0], T.mvH[1] - T.mvH[0]), sx = -2 * mh, sy = 3 * mh, sg = f(st, T.a1, 0.6);
          sum.at(sx, sy, sx - 2 * sg, sy); E.op(sum.g, (st > T.a1 ? 1 : 0) * (1 - fr(st, T.w)));
          var sq = P(sx - 1.2, sy); sumL.setAttribute('x', sq[0].toFixed(1)); sumL.setAttribute('y', (sq[1] + (mh > 0.5 ? 30 : 26)).toFixed(1));   // 挪到手上以后写在箭头下面（上面是手的 y 轴的字母）
          E.op(sumL, fr(st, T.a2) * (1 - E.win(st, T.mvH[0] - 0.1, T.mvH[1], 0.25)) * (1 - fr(st, T.eh)));   // 验算时白箭头绕肘转，会扫过它：先收起
          // 句 7：从肘指向手，绕肘转 90°
          var on7 = E.win(st, T.eh, T.out7, 0.3);
          eA.at(-2.15, 1, -2.15, 3); E.op(eA.g, on7 * (1 - fr(st, T.turn2[0])));
          var tr2 = f(st, T.turn2[0], T.turn2[1] - T.turn2[0]), ang2 = D90 + D90 * tr2;
          eT.at(-2, 1, -2 + 2 * Math.cos(ang2), 1 + 2 * Math.sin(ang2)); E.op(eT.g, fr(st, T.turn2[0] - 0.2) * (1 - fr(st, T.out7)));
          eTArc.set(D90 + 0.15, D90 + D90 * Math.max(tr2, 0.17)); E.op(eTArc.g, (tr2 > 0.01 ? 1 : 0) * (1 - fr(st, T.out7)));
          E.op(eTL, E.win(st, T.r20, T.out7, 0.3)); E.op(ok7, E.win(st, T.same, T.out7 + 0.6, 0.3));
          // 句 8：换轴
          E.glow(hF.ax.g, E.X, E.win(st, T.xUp, T.yLeft, 0.2) > 0.5); E.glow(hF.ay.g, E.Y, E.win(st, T.yLeft, T.w, 0.2) > 0.5);
          E.glow(hF.g, E.V, b(st, T.axes, 1.2) > 0.05);
          E.op(x0L, E.win(st, T.x0, T.w + 0.5, 0.3));
          gY.at(-2, 3, -2 - 2 * f(st, T.y2, 0.6), 3); E.op(gY.g, (st > T.y2 ? 1 : 0) * (1 - fr(st, T.two2)));
          E.op(y2L, E.win(st, T.y2, T.two2, 0.3));
          [hF.lx, hF.ly].forEach(function (n) { E.op(n, 1 - E.win(st, T.y2 - 0.3, T.two2, 0.3)); });   // 绿箭头沿手的 y 轴画出来时，x、y 两个字母先收起（不压线）

          /* ── 右栏 ── */
          E.op(gA, 1 - fr(st, T.elbow - 0.3));
          E.op(a1.g, fr(st, T.rev)); E.op(a2.g, fr(st, T.hand6)); E.op(a3.g, fr(st, T.base6)); E.op(a3.parts[2], fr(st, T.calc));
          E.op(a4.g, fr(st, T.two)); E.op(a5.g, fr(st, T.steps)); E.op(a6.g, fr(st, T.steps + 0.4));
          E.op(gB, E.win(st, T.elbow - 0.1, T.two2 - 0.2, 0.3));
          E.op(b1.g, fr(st, T.elbow)); E.op(b1.parts[1], fr(st, T.baseV)); E.op(b2.g, fr(st, T.nums[0]));
          E.op(gD0, E.win(st, T.w3 - 0.1, T.s1 - 0.2, 0.3));
          E.op(q1.g, fr(st, T.w3)); E.op(q1.parts[1], fr(st, T.w3 + 0.5)); E.op(q1.parts[2], fr(st, T.wNums[0])); E.op(q2.g, fr(st, T.wz));
          E.op(q3.g, fr(st, T.v3)); E.op(q3.parts[1], fr(st, T.oPt)); E.op(q4.g, fr(st, T.now));
          E.op(q5.g, fr(st, T.eToO)); E.op(q5.parts[1], fr(st, T.got21)); E.op(q6.g, fr(st, T.oMinus));
          E.op(q7.g, fr(st, T.turnD[0])); E.op(q7.parts[1], fr(st, T.take2)); E.op(q7.parts[2], fr(st, T.take2)); [3, 4, 5].forEach(function (k) { E.op(q7.parts[k], fr(st, T.takeM1)); });
          E.op(q8.g, fr(st, T.r02)); E.op(q8.parts[2], fr(st, T.r10)); E.op(q8.parts[3], fr(st, T.r10)); E.op(q8.parts[4], fr(st, T.sumD)); E.op(q8.parts[5], fr(st, T.sumD));
          E.op(q9.g, fr(st, T.times1)); E.op(q9.parts[1], fr(st, T.still12)); E.op(q10.g, fr(st, T.ep13));
          E.op(gC, E.win(st, T.s1 - 0.1, T.s2 - 0.2, 0.3));
          E.op(c1.g, fr(st, T.s1)); E.op(c1.parts[1], fr(st, T.fromO));
          E.op(c2.g, fr(st, T.f1)); E.op(c2.parts[1], fr(st, T.f2)); E.op(c3.g, fr(st, T.f3)); E.op(c3.parts[1], fr(st, T.f4)); E.op(c3.parts[2], fr(st, T.f5)); E.op(c3.parts[3], fr(st, T.f5)); E.op(c3b.g, fr(st, T.f1));
          E.op(c4.g, fr(st, T.n23)); E.op(c4.parts[1], fr(st, T.turn[0])); E.op(c4.parts[2], fr(st, T.cT1)); [3, 4].forEach(function (k) { E.op(c4.parts[k], fr(st, T.cT2)); }); [5, 6].forEach(function (k) { E.op(c4.parts[k], fr(st, T.r32)); });
          E.op(gEx, E.win(st, T.ex0 - 0.1, T.oh - 0.2, 0.3));
          E.op(x1.g, fr(st, T.ex0)); E.op(x1.parts[1], fr(st, T.exE)); E.op(x2.g, fr(st, T.exO)); E.op(x2.parts[1], fr(st, T.exV));
          E.op(x3.g, fr(st, T.exP)); E.op(x3.parts[1], fr(st, T.exR)); E.op(x3.parts[2], fr(st, T.exT)); E.op(x3.parts[3], fr(st, T.exT2));
          E.op(x4.g, fr(st, T.exAdd)); E.op(x4.parts[1], fr(st, T.exZ)); E.op(x4.parts[2], fr(st, T.exOk)); E.op(c4b.g, fr(st, T.w1)); E.op(c4b.parts[1], fr(st, T.still));
          E.op(c5.g, fr(st, T.v12)); E.op(c5.parts[1], fr(st, T.plus)); E.op(c5.parts[2], fr(st, T.m32)); E.op(c5.parts[3], fr(st, T.res)); E.op(c5.parts[4], fr(st, T.res));
          E.op(c5b.g, fr(st, T.a1)); E.op(c5b.parts[1], fr(st, T.a2));
          E.op(c6.g, fr(st, T.chk)); E.op(c6.parts[1], fr(st, T.n02)); E.op(c7.g, fr(st, T.turn2[0])); E.op(c7.parts[1], fr(st, T.r20)); E.op(c7.parts[2], fr(st, T.same)); E.op(c7.parts[3], fr(st, T.same));
          E.op(gD, E.win(st, T.s2 - 0.1, T.two2 - 0.2, 0.3));
          E.op(d1.g, fr(st, T.s2)); E.op(d1.parts[1], fr(st, T.axes)); E.op(d2.g, fr(st, T.n20)); E.op(d2.parts[1], fr(st, T.left));
          E.op(d3.g, fr(st, T.xUp)); E.op(d3.parts[3], fr(st, T.x0)); E.op(d4.g, fr(st, T.yLeft)); E.op(d4.parts[3], fr(st, T.y2));
          E.op(d5.g, fr(st, T.w)); E.op(d6.g, fr(st, T.standH)); E.op(d7.g, fr(st, T.numsB[0]));
          E.op(gE, fr(st, T.two2 - 0.1));
          E.op(JbM.g, fr(st, T.two2)); E.op(eL1, fr(st, T.two2)); [eH1, eH2].forEach(function (n) { E.op(n, fr(st, T.two2)); });
          JbM.hl('c2', 0, 1, 5, 1, E.J2, E.win(st, T.two2, T.jb, 0.3));
          E.op(eN1, fr(st, T.jbName)); E.op(eS1, fr(st, T.jbName));
          E.op(JsM.g, fr(st, T.js)); E.op(eL2, fr(st, T.js)); [eH3, eH4].forEach(function (n) { E.op(n, fr(st, T.js)); });
          E.op(eN2, fr(st, T.jsName)); E.op(eS2, fr(st, T.jsName));
        }
      };
    }
  };
})();
