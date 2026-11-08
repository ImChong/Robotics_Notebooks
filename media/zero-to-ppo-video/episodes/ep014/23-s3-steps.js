// 014 第 3 幕：先换一个点，再换一套轴——把只转肘那一列从底座上看的换成手上看的（《Modern Robotics》式 (3.84)：v_b = Rᵀ(v_s + ω_s × p)；
// 平面里 ω × p 用第 9 集「从原点指向这一点、转 90°、再乘以角速度」）。左边这一集的姿势（肩先转了 90°），底座、手两个坐标系一直画着。
// 句 1 右栏写问题和两步；句 2「只转肘」肘旁蓝色小弯箭头，右栏写第 13 集第 6 幕的 (0, 0, 1, 1, 2, 0)；
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
    elbow: 10.09, ep13: 11.24, base: 12.62, col: 14.03, nums: [14.75, 15.23, 15.73, 16.23, 16.75, 17.16],
    s1: 18.62, pt: 19.23, fromO: 20.15, toH: 21.53, out3: 23.3,
    ep9: 23.86, f1: 25.69, f2: 27.02, f3: 28.79, f4: 30.87, f5: 32.05,
    oh: 34.21, n23: 35.62, turn: [36.81, 38.0], before: 37.7, r32: 38.76, w1: 40.02, times: 41.11, still: 41.87,
    vo: 44.51, v12: 45.89, plus: 46.68, m32: 47.07, a1: 48.27, a2: 50.07, hv: 52.03, res: 53.65, mvH: [52.2, 53.4],
    chk: 55.85, onlyE: 56.64, around: 57.76, eh: 58.73, n02: 60.54, turn2: [61.52, 62.3], r20: 62.75, same: 63.92, out7: 65.1,
    s2: 65.53, axes: 66.22, n20: 67.31, left: 68.74, noUD: 69.96, xUp: 71.68, x0: 73.74, yLeft: 74.32, y2: 76.41,
    w: 77.97, standH: 79.59, colB: 80.83, numsB: [82.03, 82.58, 83.13, 83.68, 84.27, 84.72],
    two2: 86.16, jb: 87.62, book: 89.47, jbName: 90.24, js: 91.72, jsName: 93.31
  };
  var END = 96.14;

  E.scenes[2] = {
    title: '先换一个点，再换一套轴',
    dur: END,
    cues: [
      { at: 0, s: '反过来：手上看到的 $6$ 个数，能不能从底座看到的算出来？照 $2$ 处不同，分 $2$ 步' },
      { at: 9.78, s: '只转肘：**站在底座上看**，这一列是 $(0,\\,0,\\,1,\\,1,\\,2,\\,0)$（第 13 集）' },
      { at: 18.52, s: '第 $1$ 步，**换一个点**：从原点那一点，换到手这一点' },
      { at: 23.76, s: '第 9 集：板上一点的速度 $=$ 原点那一点的速度 $+$（从原点指向这一点，转 $90^\\circ$）$\\times$ 角速度' },
      { at: 34.11, s: '从原点指向手 $(-2,\\,3)$，转 $90^\\circ$ 是 $(-3,\\,-2)$，$\\times 1$ 还是 $(-3,\\,-2)$' },
      { at: 44.41, s: '$(1,\\,2)+(-3,\\,-2)=(-2,\\,0)$：手这一点的速度' },
      { at: 55.75, s: '验算：只转肘，从肘关节指向手 $(0,\\,2)$，转 $90^\\circ$ 是 $(-2,\\,0)$ ✓' },
      { at: 65.43, s: '第 $2$ 步，**换一套轴**：$(-2,\\,0)$ 往左 $2$ 格——沿手的 $x$ 轴（朝上）是 $0$，沿手的 $y$ 轴（朝左）是 $2$' },
      { at: 77.87, s: '角速度还是 $1$：**站在手上看**，这一列是 $(0,\\,0,\\,1,\\,0,\\,2,\\,0)$' },
      { at: 86.06, s: '站在手上看的雅可比：**物体雅可比**；站在底座上看的：**空间雅可比**' }
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
      var b1 = RK.line(gB, X, 40, [['只转肘', E.J2, 900], ['，站在底座上看（第 13 集）：', E.INK2]], 18);
      var b2 = RK.line(gB, X, 72, [['(0, 0, 1, ', E.OMEGA, 900], ['1, 2, 0)', E.V, 900]], 22);
      var gC = grp();
      var c1 = RK.line(gC, X, 118, [['① 换一个点', E.STEP1, 900], ['：原点那一点 → 手这一点', E.INK2]], 19);
      var c2 = RK.line(gC, X, 156, [['板上一点的速度 = ', E.INK2], ['原点那一点的速度', E.V, 900]], 17);
      var c3 = RK.line(gC, X, 184, [['+（从原点指向这一点，', E.INK2], ['转 90°', E.OMEGA, 900], ['）× ', E.INK2], ['角速度', E.OMEGA, 900]], 17);
      var c3b = RK.line(gC, X, 210, [['（第 9 集）', E.MUT, 700]], 16);
      var c4 = RK.line(gC, X, 240, [['(−2, 3)', E.INK, 900], [' 转 90° → ', E.OMEGA, 900], ['(−3, −2)', E.INK, 900]], 18);
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
          [bF.lx, bF.ly].forEach(function (n) { E.op(n, 1 - E.win(st, T.plus - 0.2, T.hv + 0.3, 0.25)); });   // (−3, −2) 挪到 (1, 2) 的尖上时从底座的 x、y 字母上扫过：先收起，(1, 2) 收起后再出
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
          vO.at(0, 0, 1, 2); E.op(vO.g, fr(st, T.vo) * on6); E.op(vOL, fr(st, T.v12) * on6);
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

          /* ── 右栏 ── */
          E.op(gA, 1 - fr(st, T.elbow - 0.3));
          E.op(a1.g, fr(st, T.rev)); E.op(a2.g, fr(st, T.hand6)); E.op(a3.g, fr(st, T.base6)); E.op(a3.parts[2], fr(st, T.calc));
          E.op(a4.g, fr(st, T.two)); E.op(a5.g, fr(st, T.steps)); E.op(a6.g, fr(st, T.steps + 0.4));
          E.op(gB, E.win(st, T.elbow - 0.1, T.two2 - 0.2, 0.3));
          E.op(b1.g, fr(st, T.elbow)); E.op(b1.parts[1], fr(st, T.ep13)); E.op(b2.g, fr(st, T.nums[0]));
          E.op(gC, E.win(st, T.s1 - 0.1, T.s2 - 0.2, 0.3));
          E.op(c1.g, fr(st, T.s1)); E.op(c1.parts[1], fr(st, T.fromO));
          E.op(c2.g, fr(st, T.f1)); E.op(c2.parts[1], fr(st, T.f2)); E.op(c3.g, fr(st, T.f3)); E.op(c3.parts[1], fr(st, T.f4)); E.op(c3.parts[2], fr(st, T.f5)); E.op(c3.parts[3], fr(st, T.f5)); E.op(c3b.g, fr(st, T.f1));
          E.op(c4.g, fr(st, T.n23)); E.op(c4.parts[1], fr(st, T.turn[0])); E.op(c4.parts[2], fr(st, T.r32)); E.op(c4b.g, fr(st, T.w1)); E.op(c4b.parts[1], fr(st, T.still));
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
