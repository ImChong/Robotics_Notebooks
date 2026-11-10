// 014 第 6 幕：目标写在哪，就配哪一个雅可比（路线 §L1 自测题 1 后半；《Modern Robotics》6.2 节：物体雅可比配物体运动旋量，「an equivalent form can be derived in the
// space frame, using J_s and V_s = [Ad_Tsb] V_b」；11.3.3 节任务空间运动控制用 V_b 配 J_b；5.4 节：关心手这一点的线速度时用 J_b 更合适）。
// 左边这一集的姿势（肩先转了 90°），底座、手两个坐标系。
// 句 3「站在底座上写」一个淡紫的「目标①」坐标系从手出发、绕原点逆时针转 75°（虚线大圆弧）；句 4「站在手上写」「目标②」在手这一点原地逆时针转 75°；
// 句 8「原点那一点，也跟着走一个圆」以手为圆心、过原点的一段虚线弧，「从手这一点指向原点」白虚线 (2, −3)，「转九十度」绕手转到 (3, 2)，再整支挪到原点，
// 「第一列取两份」红 (0, 2)、「第二列取负三份」绿 (3, 0) 头尾相接，「加起来」紫 (3, 2)。右栏逐句写。时刻取自配音的逐词时间，见 ../ep014.py。
(function () {
  var E = window.EP014, K = E.K, H = E.HEX;
  var X = 455, D90 = Math.PI / 2, SW = 75 * Math.PI / 180;
  var T = {
    last: 0.7, diff: 3.58, ctrl: 4.49, give: 5.76, target: 8.19, joints: 9.27, fast: 10.4,
    two: 12.04, write: 13.16, nums: 14.28,
    t1: 18.5, base1: 19.11, around: 20.63, rot1: [21, 25.6], circle: 24.22,
    t2: 26.4, hand2: 27.06, still: 28.43, self: 29.64, rot2: [29.9, 33.6], screw: 32.22,
    same6: 34.21, diffMove: 35.61, so: 37.29, where: 38.44, match: 40.27, out2: 43.4,
    j1: 43.75, js: 44.88, j2: 46.05, jb: 46.87,
    other: 48.99, adj: 50.97, conv: 51.98, to1: 53.83, ep: 56.53, n1: 62.44,
    to2: 66.73, spin: 68.33, oPt: 69.56, circ: 70.67, fromH: 72.43, n23: 74.35, turn: [75.38, 76.14], mv: [76.19, 76.94],
    c1: 76.35, take2: 76.96, r02: 77.7, c2: 78.62, takeM3: 79.26, r30: 80.16, add: 81.15, sum: 81.88,
    res: 84.43, both: 87.79, sameAdj: 89.87, onlyIf: 91.41, sameRate: 94.38, out8: 96.14,
    book: 96.53, care: 97.4, useB: 99.62, better: 101.08, back3: 101.99, isHand: 103.23,
    ep15: 105.63, ep15b: 108.39,
    ep2: 55.45, hv6: 57.44, rG6: 59.34
  };
  var END = 110.98;

  E.scenes[5] = {
    title: '目标写在哪，就配哪一个',
    dur: END,
    cues: [
      { at: 0, s: '控制手：先给手定一个想要的速度，叫做**目标**，再算各个关节该转多快' },
      { at: 11.94, s: '$2$ 个**目标**，写出来都是 $(0,\\,0,\\,1,\\,0,\\,0,\\,0)$' },
      { at: 18.4, s: '① 站在底座上写：绕着原点每秒转 $1$ 弧度——手绕着肩转一个大圈' },
      { at: 26.3, s: '② 站在手上写：手这一点不动，自己原地转——像拧螺丝刀' },
      { at: 34.11, s: '$6$ 个数一模一样，动作完全不同：**目标**站在什么地方写，就配站在同一个地方看的雅可比' },
      { at: 43.65, s: '① 配**空间雅可比**；② 配**物体雅可比**' },
      { at: 48.89, s: '偏要用另一个：先用**伴随变换**把**目标**换过去。① 换到手上 = 第 2 幕只转肩那个 $(0,\\,0,\\,1,\\,-2,\\,3,\\,0)$' },
      { at: 66.63, s: '② 换到底座上：从手指向原点 $(2,\\,-3)$，转 $90^\\circ$：$2\\times(0,\\,1)+(-3)\\times(-1,\\,0)=(3,\\,2)$' },
      { at: 83.77, s: '$(0,\\,0,\\,1,\\,3,\\,2,\\,0)$。两边乘的是同一个**伴随变换**：只要站在同一个地方，算出的关节转速一样' },
      { at: 96.43, s: '书上：关心手这一点自己的速度，用**物体雅可比**更合适——后 $3$ 个数就是手这一点的速度' },
      { at: 105.53, s: '从**目标**怎么反过来算出关节转速：第 15 集' }
    ],
    build: function (K) {
      var s = K.sceneSvg('这两种雅可比用起来有什么不同？控制手的时候，先给手定一个想要的速度，叫做目标，再算各个关节该转多快。两个目标写出来都是 (0, 0, 1, 0, 0, 0)：第一个站在底座上写，绕着原点每秒转一弧度，手绕着肩转一个大圈；第二个站在手上写，手这一点不动，自己原地转，像拧螺丝刀。六个数一模一样，动作完全不同，所以目标站在什么地方写，就配站在同一个地方看的雅可比：第一个配空间雅可比，第二个配物体雅可比。偏要用另一个，就先用伴随变换把目标换过去：第一个换到手上是 (0, 0, 1, −2, 3, 0)；第二个换到底座上，从手指向原点 (2, −3)，转九十度得 (3, 2)，是 (0, 0, 1, 3, 2, 0)。两边乘的是同一个伴随变换，算出来的关节转速一样。书上说，关心手这一点自己的速度时，用物体雅可比更合适。从目标怎么反过来算出关节转速，第十五集讲');

      var pn = E.panel14(s), P = pn.P, u = pn.u, top = pn.top, O = P(0, 0);
      pn.pose(E.P90[0], E.P90[1]);
      var J = pn.joints(), pE = P(J[1][0], J[1][1]), pH = P(J[2][0], J[2][1]);
      var bF = E.frame2(top, P, 1, '底座', { labAt: [0.75, -0.62] }), hF = E.frame2(top, P, 1, '手', { labAt: [0.45, 0.45], labColor: E.V });
      bF.set(0, 0, 0); hF.set(-2, 3, D90);
      var jS = E.joint(top, E.HI), jE = E.joint(top, E.HI), hand = E.dot(top, E.V, 7);
      jS.at(O); jE.at(pE); hand.at(pH);
      // 目标①、②：淡一点的坐标系（只是想要的动作，机械臂这两个关节做不出②）
      var R13 = Math.sqrt(13), A0 = Math.atan2(3, -2);
      var arc1 = K.paint(E.el('path', { fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '7 6' }, top), null, E.V);
      var g1 = E.frame2(top, P, 0.8, '目标①', { labAt: [0.85, 0.3], labColor: E.V, w: 4, size: 17 });
      var g2 = E.frame2(top, P, 0.8, '目标②', { labAt: [0.95, -0.62], labColor: E.V, w: 4, size: 17 });
      var spin2 = E.arcArrow(top, pH[0], pH[1], 0.45 * u, E.OMEGA, 3, { head: 10 });
      var big1 = E.arcArrow(top, O[0], O[1], R13 * u, E.OMEGA, 2.5, { head: 12, dash: '7 6' });
      // 句 8：原点那一点绕手走圆
      var arcO = E.arcArrow(top, pH[0], pH[1], R13 * u, E.V, 2.5, { head: 12, dash: '7 6' });
      var ringO = K.paint(E.el('circle', { cx: O[0], cy: O[1], r: 14, fill: 'none', 'stroke-width': 3.5 }, top), null, E.INK);
      var wA = E.gv(top, P, E.INK, 4, { head: 14, dash: '9 6' }), wT = E.gv(top, P, E.INK, 4, { head: 14 });
      var tArc = E.arcArrow(top, pH[0], pH[1], 0.7 * u, E.OMEGA, 3, { head: 10 });
      var red = E.gv(top, P, E.X, 5, { head: 14 }), grn = E.gv(top, P, E.Y, 5, { head: 14 }), vio = E.gv(top, P, E.V, 5.5, { head: 17 });
      [wT, red, grn, vio].forEach(function (a) { a.line.style.filter = 'drop-shadow(0 0 2px #000)'; });
      var wL = E.halo(E.txt(top, P(-0.6, 2.35)[0], P(-0.6, 2.35)[1] + 4, '(2, −3)', 19, E.INK, 'start', 900));
      var redL = E.halo(E.txt(top, P(0, 1.5)[0] - 12, P(0, 1.5)[1] + 6, '(0, 2)', 18, E.X, 'end', 900));
      var grnL = E.halo(E.txt(top, P(1.5, 2)[0], P(1.5, 2)[1] - 14, '(3, 0)', 18, E.Y, 'middle', 900));
      var vioL = E.halo(E.txt(top, P(2.2, 0.8)[0], P(2.2, 0.8)[1] + 8, '(3, 2)', 19, E.V, 'start', 900));

      /* ══ 右栏 ══ */
      var RK = E.rowKit(), Rg = E.el('g', {}, s);
      function grp() { return E.el('g', {}, Rg); }
      var gA = grp();
      var a1 = RK.line(gA, X, 60, [['控制手的时候：', E.INK, 900]], 22);
      var a2 = RK.line(gA, X, 104, [['① 给手定一个想要的速度：', E.INK2], ['目标', E.GOOD, 900]], 19);
      var a3 = RK.line(gA, X, 144, [['② 再算各个关节该转多快', E.INK2]], 19);
      var gB = grp();
      var b1 = RK.line(gB, X, 44, [['2 个', E.INK2], ['目标', E.GOOD, 900], ['，写出来都是', E.INK2]], 19);
      var b2 = RK.line(gB, X, 76, [['(0, 0, 1, ', E.OMEGA, 900], ['0, 0, 0)', E.V, 900]], 22);
      var b3 = RK.line(gB, X, 120, [['① 站在底座上写：', E.INK, 900]], 18);
      var b4 = RK.line(gB, X, 148, [['绕原点每秒转 1 弧度 → 手绕着肩转大圈', E.INK2]], 16);
      var b5 = RK.line(gB, X, 190, [['② 站在手上写：', E.V, 900]], 18);
      var b6 = RK.line(gB, X, 218, [['手这一点不动、原地转 → 像拧螺丝刀', E.INK2]], 16);
      var b7 = RK.line(gB, X, 262, [['6 个数一样，动作完全不同', E.ERR, 900]], 18);
      var b8 = RK.line(gB, X, 300, [['目标', E.GOOD, 900], ['在什么地方写，', E.INK2]], 18);
      var b9 = RK.line(gB, X, 328, [['就配', E.INK2], ['同一个地方', E.INK, 900], ['看的雅可比', E.INK2]], 18);
      var b10 = RK.line(gB, X, 370, [['① → ', E.INK2], ['空间雅可比', E.INK, 900], ['　② → ', E.INK2], ['物体雅可比', E.V, 900]], 18);
      var gC = grp();
      var c1 = RK.line(gC, X, 44, [['偏要用另一个：先用', E.INK2], ['伴随变换', E.HI, 900], ['换目标', E.INK2]], 17);
      var c2 = RK.line(gC, X, 80, [['① 换到手上：', E.INK, 900], ['(0, 0, 1, −2, 3, 0)', E.V, 900]], 17);
      var c2b = RK.line(gC, X, 104, [['（第 2 幕只转肩：', E.MUT, 700], ['(−3, −2)', E.V, 900], [' 按手的轴 → ', E.MUT, 700], ['(−2, 3)', E.INK, 900], ['）', E.MUT, 700]], 15);
      var c3 = RK.line(gC, X, 140, [['② 换到底座上：', E.V, 900], ['原点那一点绕手走圆', E.INK2]], 17);
      var c4 = RK.line(gC, X, 168, [['从手指向原点 ', E.INK2], ['(2, −3)', E.INK, 900], ['，', E.INK2], ['转 90°', E.OMEGA, 900]], 17);
      var c5 = RK.line(gC, X, 196, [['2 × ', E.INK, 900], ['(0, 1)', E.X, 900], [' + ', E.INK2], ['(−3) × ', E.INK, 900], ['(−1, 0)', E.Y, 900]], 17);
      var c6 = RK.line(gC, X, 224, [['= ', E.INK2], ['(0, 2)', E.X, 900], [' + ', E.INK2], ['(3, 0)', E.Y, 900], [' = ', E.INK2], ['(3, 2)', E.V, 900]], 17);
      var c7 = RK.line(gC, X, 256, [['→ ', E.INK2], ['(0, 0, 1, 3, 2, 0)', E.INK, 900]], 18);
      var c8 = E.tex(gC, X, 300, 'J_s\\,\\dot\\theta=\\mathcal V_s\\;\\Leftrightarrow\\;\\textcolor{' + H.violet + '}{J_b\\,\\dot\\theta=\\mathcal V_b}', 22, E.INK, 'start', { w: 330, h: 40 });
      var c9 = RK.line(gC, X, 340, [['两边乘同一个', E.INK2], ['伴随变换', E.HI, 900]], 17);
      var c10 = RK.line(gC, X, 370, [['站在同一个地方 → 关节转速一样', E.GOOD, 900]], 17);
      var gD = grp();
      var d1 = RK.line(gD, X, 60, [['书上（《Modern Robotics》5.4 节）：', E.INK2]], 17);
      var d2 = RK.line(gD, X, 100, [['关心', E.INK2], ['手这一点自己的速度', E.V, 900]], 20);
      var d3 = RK.line(gD, X, 138, [['→ 用', E.INK2], ['物体雅可比', E.V, 900], ['更合适', E.INK2]], 20);
      var d4 = RK.line(gD, X, 176, [['后 3 个数 = 手这一点的速度', E.INK2]], 18);
      var d5 = RK.line(gD, X, 260, [['从', E.INK2], ['目标', E.GOOD, 900], ['反过来算关节转速：', E.INK2]], 18);
      var d6 = RK.line(gD, X, 292, [['第 15 集', E.HI, 900]], 20);

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }
      function arcD(cx, cy, r, a0, a1) {   // 画面里的圆弧 path（数学角度、逆时针为正）
        var p0 = [cx + r * Math.cos(a0), cy - r * Math.sin(a0)], p1 = [cx + r * Math.cos(a1), cy - r * Math.sin(a1)];
        return 'M ' + p0[0].toFixed(1) + ' ' + p0[1].toFixed(1) + ' A ' + r + ' ' + r + ' 0 0 0 ' + p1[0].toFixed(1) + ' ' + p1[1].toFixed(1);
      }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) { RK.layout(); laid = true; }
          var f = E.fade, b = E.bump;
          pn.pose(E.P90[0], E.P90[1]); pn.view.clearTint();
          E.glow(bF.g, E.INK, E.win(st, T.base1, T.around, 0.2) > 0.5 || b(st, T.js, 1.0) > 0.05);
          E.glow(hF.g, E.V, E.win(st, T.hand2, T.still, 0.2) > 0.5 || b(st, T.jb, 1.0) > 0.05);
          // 目标①：绕原点逆时针转（虚线大弧 + 跟着转的坐标系）
          var k1 = f(st, T.rot1[0], T.rot1[1] - T.rot1[0]), ang1 = A0 + SW * k1, on1 = E.win(st, T.around, T.out2, 0.3);
          g1.set(R13 * Math.cos(ang1), R13 * Math.sin(ang1), D90 + SW * k1); E.op(g1.g, 0.85 * on1);
          var away = E.clamp((k1 - 0.12) / 0.2, 0, 1);   // 刚出发时和手的坐标系叠在一起：字母、标签等它挪开一点再出
          E.op(g1.lab, away); [g1.lx, g1.ly, g2.lx, g2.ly].forEach(function (n) { E.op(n, 0); });   // 目标的两根轴只用红、绿区分，不写字母（转的时候会扫过别的线）
          var lq = P((R13 + 1.25) * Math.cos(ang1), (R13 + 1.25) * Math.sin(ang1));   // 「目标①」写在大圆弧外面，跟着走
          g1.lab.setAttribute('x', lq[0].toFixed(1)); g1.lab.setAttribute('y', (lq[1] + 6).toFixed(1));
          big1.set(A0, A0 + Math.max(0.01, SW * k1)); E.op(big1.g, k1 > 0.01 ? on1 : 0);
          // 目标②：原地转
          var k2 = f(st, T.rot2[0], T.rot2[1] - T.rot2[0]), on2 = E.win(st, T.still, T.out2, 0.3);
          g2.set(-2, 3, D90 + SW * k2); E.op(g2.g, 0.85 * on2);
          spin2.set(D90 + 0.5, D90 + 0.5 + Math.max(0.01, SW * k2)); E.op(spin2.g, k2 > 0.01 ? on2 : 0);
          E.op(arc1, 0);
          E.op(hF.g, 1 - 0.7 * E.win(st, T.around, T.out2, 0.3));   // 目标①、②从手出发、画在手上时，手自己的坐标系先淡下去（不叠在一起）
          [hF.lx, hF.ly, hF.lab].forEach(function (n) { E.op(n, 1 - E.win(st, T.around, T.out2, 0.3) - E.win(st, T.fromH, T.out8, 0.3)); });   // 句 8 白箭头绕手转时也先收起字
          // 句 8：原点那一点绕手走圆
          var aO = Math.atan2(-3, 2), on8 = E.win(st, T.oPt, T.out8, 0.3);
          arcO.set(aO - 0.65, aO + 0.6); E.op(arcO.g, E.win(st, T.circ, T.out8, 0.3));
          E.op(ringO, on8);
          wA.at(-2, 3, 0, 0); E.op(wA.g, E.win(st, T.fromH, T.mv[1], 0.3));
          E.op(wL, E.win(st, T.n23, T.turn[0], 0.3));
          var tr = f(st, T.turn[0], T.turn[1] - T.turn[0]), mvk = f(st, T.mv[0], T.mv[1] - T.mv[0]), ang = aO + D90 * tr;
          var tx0 = -2 + 2 * mvk, ty0 = 3 - 3 * mvk;   // 转完以后整支从手挪到原点
          wT.at(tx0, ty0, tx0 + R13 * Math.cos(ang), ty0 + R13 * Math.sin(ang)); E.op(wT.g, fr(st, T.turn[0] - 0.2) * (1 - fr(st, T.add)));
          tArc.set(aO, aO + D90 * Math.max(tr, 0.01)); E.op(tArc.g, (tr > 0.01 ? 1 : 0) * (1 - fr(st, T.mv[0])));
          red.at(0, 0, 0, 2 * f(st, T.take2, 0.6)); E.op(red.g, (st > T.take2 ? 1 : 0) * (1 - fr(st, T.out8)));
          grn.at(0, 2, 3 * f(st, T.takeM3, 0.6), 2); E.op(grn.g, (st > T.takeM3 ? 1 : 0) * (1 - fr(st, T.out8)));
          vio.at(0, 0, 3, 2); E.op(vio.g, E.win(st, T.sum, T.out8, 0.3));
          E.op(redL, E.win(st, T.r02, T.out8, 0.3)); E.op(grnL, E.win(st, T.r30, T.out8, 0.3)); E.op(vioL, E.win(st, T.sum, T.out8, 0.3));
          E.glow(hand, E.V, E.win(st, T.care, T.ep15, 0.2) > 0.5);
          [bF.lx, bF.ly].forEach(function (n) { E.op(n, 1 - E.win(st, T.mv[0], T.out8, 0.3)); });   // 红 (0, 2) 画在底座的 y 轴上：字母先收起

          /* ── 右栏 ── */
          E.op(gA, 1 - fr(st, T.two - 0.3));
          E.op(a1.g, fr(st, T.ctrl)); E.op(a2.g, fr(st, T.give)); E.op(a2.parts[1], fr(st, T.target)); E.op(a3.g, fr(st, T.joints));
          E.op(gB, E.win(st, T.two - 0.1, T.other - 0.2, 0.3));
          E.op(b1.g, fr(st, T.two)); E.op(b2.g, fr(st, T.nums));
          E.op(b3.g, fr(st, T.t1)); E.op(b4.g, fr(st, T.around)); E.op(b5.g, fr(st, T.t2)); E.op(b6.g, fr(st, T.still));
          E.op(b7.g, fr(st, T.same6)); E.op(b8.g, fr(st, T.so)); E.op(b9.g, fr(st, T.match)); E.op(b10.g, fr(st, T.j1)); E.op(b10.parts[1], fr(st, T.js)); [2, 3].forEach(function (k) { E.op(b10.parts[k], fr(st, T.j2)); });
          E.op(gC, E.win(st, T.other - 0.1, T.book - 0.2, 0.3));
          E.op(c1.g, fr(st, T.other)); E.op(c1.parts[1], fr(st, T.adj)); E.op(c1.parts[2], fr(st, T.conv));
          E.op(c2.g, fr(st, T.to1)); E.op(c2.parts[1], fr(st, T.n1)); E.op(c2b.g, fr(st, T.ep2)); E.op(c2b.parts[1], fr(st, T.hv6)); [2, 3, 4].forEach(function (k) { E.op(c2b.parts[k], fr(st, T.rG6)); });
          E.op(c3.g, fr(st, T.to2)); E.op(c3.parts[1], fr(st, T.oPt)); E.op(c4.g, fr(st, T.fromH)); E.op(c4.parts[1], fr(st, T.n23)); E.op(c4.parts[2], fr(st, T.turn[0])); E.op(c4.parts[3], fr(st, T.turn[0]));
          E.op(c5.g, fr(st, T.c1)); E.op(c5.parts[0], fr(st, T.take2)); E.op(c5.parts[1], fr(st, T.take2)); [2, 3, 4].forEach(function (k) { E.op(c5.parts[k], fr(st, T.takeM3)); });
          E.op(c6.g, fr(st, T.r02)); E.op(c6.parts[2], fr(st, T.r30)); E.op(c6.parts[3], fr(st, T.r30)); E.op(c6.parts[4], fr(st, T.sum)); E.op(c6.parts[5], fr(st, T.sum));
          E.op(c7.g, fr(st, T.res)); E.op(c8, fr(st, T.both)); E.op(c9.g, fr(st, T.sameAdj)); E.op(c10.g, fr(st, T.onlyIf)); E.op(c10.parts[0], fr(st, T.onlyIf));
          E.glow(c10.g, E.GOOD, b(st, T.sameRate, 1.0) > 0.05);
          E.op(gD, fr(st, T.book - 0.1));
          E.op(d1.g, fr(st, T.book)); E.op(d2.g, fr(st, T.care)); E.op(d3.g, fr(st, T.useB)); E.op(d3.parts[2], fr(st, T.better)); E.op(d4.g, fr(st, T.back3));
          E.op(d5.g, fr(st, T.ep15)); E.op(d6.g, fr(st, T.ep15b));
        }
      };
    }
  };
})();
