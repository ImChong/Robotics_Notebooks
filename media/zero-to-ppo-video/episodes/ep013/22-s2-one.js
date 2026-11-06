// 013 第 2 幕：每次只转一个关节——雅可比的两列（《Modern Robotics》图 5.1：第 i 列 = 只让第 i 个关节每秒转 1 弧度时手的速度）。
// 左边侧视网格 + UR5（零位：肘 (1, 2)，手 (3, 2)），右栏逐行写算式；速度箭头 1 : 1 画（同 009）。
// 句 1「只转肩」肩旁出蓝色小弯箭头 θ̇₁ = 1，肘上一把灰锁「不弯」，「整条胳膊绕着肩转」胳膊来回转，「手走的是一个圆」描出手走的一段圆弧（亮黄虚线）；
// 句 2 停在零位：「连线」肩—手画白虚线，「垂直」手上出直角记号和淡淡的速度方向，「大小」「转速乘以距离」右栏写；
// 句 3「从肩关节指向手」白箭头从原点画到 (3, 2)，「转九十度」箭头绕原点逆时针转 90°（蓝弧），「这个矩阵」右栏出第 3 集转 90° 的矩阵（两列红、绿），
//    「零、一」「负一、零」两列各亮一下；
// 句 4 按列拿份（同 009 第 3 幕）：「第一列取三份」网格上红色一段 (0, 0) → (0, 3)，「第二列取两份」绿色一段接着画到 (−2, 3)，「加起来」转好的箭头亮；
// 句 5「再乘以转速一」箭头平移到手上、变亮黄（肩那一列），「往左走两格」「往上走三格」手上画两段分量（先往上绿、再往左红，横着走会压在小臂上）；
// 句 6「再只转肘」肘旁蓝色小弯箭头 θ̇₂ = 1，肩上灰锁「不动」，小臂绕着肘来回转，「小一点的圆」描出手绕肘的一段圆弧（青虚线）；
// 句 7 停在零位：「从肘关节指向手」白箭头肘 → 手，「三减一」「二减二」右栏逐项算，「转九十度」箭头绕肘转 90°，「第一列取两份」红色一段 (1, 2) → (1, 4)，
//    「第二列取零份」右栏写 0 份，「加起来」(0, 2)；
// 句 8「乘以转速一」箭头平移到手上、变青（肘那一列），「往上走两格」；最后两支箭头一起留在手上（肩亮黄、肘青）。
// 时刻取自配音的逐词时间，见 ../ep013.py。
(function () {
  var E = window.EP013, K = E.K, H = E.HEX;
  var RX = E.ARM.RX, X = RX + 10, D90 = Math.PI / 2;
  var T = {
    only: 0.7, rate1: 1.92, rad1: 2.81, noBend: 3.46, swing: [4.69, 8.4], circle: 6.42, round: 7.33,
    ep9: 8.85, hv: 10.04, line: 12.31, perp: 12.67, size: 13.54, prod: 14.38,
    from: 16.83, hand3: 18.08, c3: 18.63, c2: 19.17, turn: [19.94, 21.0], mat: 22.43, cols: 23.1, col1: 23.64, col2: 24.51,
    byCol: 26.63, take3: 27.57, n03: 29.0, take2: 30.04, n20: 31.41, sum: 32.55, n23: 33.45,
    times1: 35.69, still: 36.97, move: [36.97, 37.9], shOnly: 38.6, left: 40.45, left2: 41.04, up: 41.68, up3: 42.29, out1: 43.6,
    elOnly: 44.03, rate2: 45.24, shLock: 46.77, swing2: [47.88, 51.6], circle2: 50.37, round2: 51.3,
    from2: 52.78, hand2: 54.02, m31: 54.58, m22: 55.78, got: 56.95, n2: 57.33, turn2: [58.51, 59.4], take2b: 59.54, n02: 61.0,
    take0: 62.01, sum2: 63.33, n02b: 64.34,
    times1b: 66.3, move2: [66.6, 67.4], elOnlyT: 67.59, up2: 69.41, two2: 70.0
  };
  var END = 71.54;
  var A0 = Math.atan2(2, 3);   // 肩 → 手的方向

  E.scenes[1] = {
    title: '每次只转一个关节',
    dur: END,
    cues: [
      { at: 0, s: '只转肩（每秒 1 弧度），肘不弯：整条胳膊绕着肩转，手走一个圆' },
      { at: 8.75, s: '**手的速度**（第 9 集）：和它到转动中心的连线垂直；大小 = **转速** × 距离' },
      { at: 16.73, s: '从肩关节指向手：$(3,\\,2)$；转 $90^\\circ$ = 乘第 3 集的矩阵，$2$ 列是 $(0,\\,1)$、$(-1,\\,0)$' },
      { at: 26.53, s: '**按列拿份**：$3\\times(0,\\,1)+2\\times(-1,\\,0)=(0,\\,3)+(-2,\\,0)=(-2,\\,3)$' },
      { at: 35.59, s: '× **转速** 1：只转肩时，手每秒往左 2 格、往上 3 格：$(-2,\\,3)$' },
      { at: 43.93, s: '只转肘（每秒 1 弧度），肩不动：小臂绕着肘转，手走一个小一点的圆' },
      { at: 52.68, s: '从肘关节指向手：$(3-1,\\,2-2)=(2,\\,0)$；转 $90^\\circ$：$2\\times(0,\\,1)+0\\times(-1,\\,0)=(0,\\,2)$' },
      { at: 66.2, s: '× **转速** 1：只转肘时，手每秒往上 2 格：$(0,\\,2)$' }
    ],
    build: function (K) {
      var s = K.sceneSvg('每次只转一个关节：只转肩，每秒一弧度，肘不弯，整条胳膊绕着肩转，手走一个圆；手的速度和它到转动中心的连线垂直，大小是转速乘以距离；从肩关节指向手是 (3, 2)，转九十度就是乘第三集的矩阵（两列 (0, 1)、(−1, 0)），按列拿份：3 × (0, 1) + 2 × (−1, 0) = (0, 3) + (−2, 0) = (−2, 3)，再乘以转速 1，只转肩时手每秒往左走 2 格、往上走 3 格。只转肘，每秒一弧度，肩不动，小臂绕着肘转，手走一个小一点的圆；从肘关节指向手 (3 − 1, 2 − 2) = (2, 0)，转九十度：2 × (0, 1) + 0 × (−1, 0) = (0, 2)，乘以转速 1，只转肘时手每秒往上走 2 格');

      var pn = E.panel(s), P = pn.P, u = pn.u, top = pn.top, O = P(0, 0), Q = P(1, 2), HH = P(3, 2);
      // 手走的圆弧：绕肩（半径 √13，亮黄虚线）、绕肘（半径 2，青虚线）；只画网格里的一段
      function arcPath(cx, cy, r, a0, a1) {
        var pts = [];
        for (var k = 0; k <= 40; k++) { var a = a0 + (a1 - a0) * k / 40; pts.push(P(cx + r * Math.cos(a), cy + r * Math.sin(a))); }
        return E.poly(pts);
      }
      var under2 = E.el('g', {}, top);
      var cir1 = K.paint(E.el('path', { d: arcPath(0, 0, Math.sqrt(13), -0.14, 1.92), fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '8 6' }, under2), null, E.J1);
      var cir2 = K.paint(E.el('path', { d: arcPath(1, 2, 2, -0.7, 2.44), fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '8 6' }, under2), null, E.J2);
      var jS = E.joint(top, E.HI), jE = E.joint(top, E.HI);
      var arcS = E.arcArrow(top, O[0], O[1], 0.62 * u, E.OMEGA, 3.5, { head: 12 }), arcE = E.arcArrow(top, 0, 0, 0.5 * u, E.OMEGA, 3.5, { head: 11 });
      arcS.set(-0.5, 1.0); arcE.set(-0.3, 1.3);
      var thS = E.tex(top, O[0] - 64, O[1] - 30, '\\dot\\theta_1=1', 22, E.OMEGA, 'middle', { w: 90, h: 36 });
      var thE = E.tex(top, 0, 0, '\\dot\\theta_2=1', 22, E.OMEGA, 'middle', { w: 90, h: 36 });
      // 不动的关节：灰色小锁（字「不弯」「不动」）
      function lock(lbl) {
        var g = E.el('g', {}, top);
        K.paint(E.el('rect', { x: -9, y: -4, width: 18, height: 14, rx: 3 }, g), E.MUT);
        K.paint(E.el('path', { d: 'M -5.5 -4 V -9 A 5.5 5.5 0 0 1 5.5 -9 V -4', fill: 'none', 'stroke-width': 3 }, g), null, E.MUT);
        E.halo(E.txt(g, 14, 9, lbl, 18, E.INK2, 'start', 900));
        return g;
      }
      var lockE = lock('不弯'), lockS = lock('不动');
      var hand = E.dot(top, E.V, 7);
      // 句 2：连线、直角、方向
      var link = K.paint(E.el('path', { d: 'M ' + O[0] + ' ' + O[1] + ' L ' + HH[0] + ' ' + HH[1], fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '6 6' }, top), null, E.INK);
      var e1 = [3 / Math.sqrt(13), 2 / Math.sqrt(13)], e2 = [-e1[1], e1[0]], sq = 0.32;
      var rightMk = K.paint(E.el('path', { d: E.poly([P(3 - sq * e1[0], 2 - sq * e1[1]), P(3 - sq * e1[0] + sq * e2[0], 2 - sq * e1[1] + sq * e2[1]), P(3 + sq * e2[0], 2 + sq * e2[1])]),
        fill: 'none', 'stroke-width': 2.5 }, top), null, E.INK);
      var dirHint = E.gv(top, P, E.J1, 3, { head: 12, dash: '6 6' });
      var distL = E.halo(E.txt(top, P(1.3, 0.55)[0], P(1.3, 0.55)[1], '距离', 18, E.INK, 'middle', 900));
      // 句 3–5：白箭头（从肩指向手）、转 90°、红绿两段、平移到手上
      var dShadow = E.gv(top, P, E.INK2, 3, { head: 12, dash: '6 6' });
      var red1 = E.gv(top, P, E.X, 5, { head: 14 }), grn1 = E.gv(top, P, E.Y, 5, { head: 14 });
      var dArrow = E.gv(top, P, E.INK, 4.5, { head: 15 });
      var turnArc = E.arcArrow(top, O[0], O[1], 1.1 * u, E.OMEGA, 3, { head: 11 });
      var d32 = E.halo(E.txt(top, 0, 0, '(3, 2)', 20, E.INK, 'start', 900));
      var c1 = E.gv(top, P, E.J1, 5.5, { head: 17 });
      var c1L = E.halo(E.txt(top, 0, 0, '(−2, 3)', 21, E.J1, 'start', 900));
      var legU1 = K.paint(E.el('path', { d: 'M ' + HH[0] + ' ' + HH[1] + ' V ' + P(3, 5)[1], fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '7 5' }, top), null, E.Y);
      var legL1 = K.paint(E.el('path', { d: 'M ' + P(3, 5)[0] + ' ' + P(3, 5)[1] + ' H ' + P(1, 5)[0], fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '7 5' }, top), null, E.X);
      var legL1L = E.halo(E.txt(top, P(2.1, 5)[0], P(2.1, 5)[1] + 24, '往左 2', 18, E.X, 'middle', 900));
      var legU1L = E.halo(E.txt(top, P(3, 3.5)[0] + 10, P(3, 3.5)[1] + 6, '往上 3', 18, E.Y, 'start', 900));
      // 句 7–8：肘
      var dArrow2 = E.gv(top, P, E.INK, 4.5, { head: 15 });
      var red2 = E.gv(top, P, E.X, 5, { head: 14 });   // 画在白箭头上面：这一列只有红色这一段，和转好的白箭头重合
      var dShadow2 = E.gv(top, P, E.INK2, 3, { head: 12, dash: '6 6' });
      var turnArc2 = E.arcArrow(top, Q[0], Q[1], 0.95 * u, E.OMEGA, 3, { head: 11 });
      var d20 = E.halo(E.txt(top, P(2, 2)[0], P(2, 2)[1] + 30, '(2, 0)', 20, E.INK, 'middle', 900));
      var c2 = E.gv(top, P, E.J2, 5.5, { head: 17 });
      var c2L = E.halo(E.txt(top, 0, 0, '(0, 2)', 21, E.J2, 'start', 900));

      /* ══ 右栏 ══ */
      var RK = E.rowKit(), Rg = E.el('g', {}, s);
      function grp() { return E.el('g', {}, Rg); }
      var g1 = grp();
      var a1 = RK.line(g1, X, 46, [['① 只转', E.INK, 900], ['肩', E.HI, 900]], 24);
      var a2 = E.symRow(g1, RK, X, 88, '\\dot\\theta_1=1', E.OMEGA, 74, [['：每秒转 1 弧度', E.INK2]], 21);
      var a3 = RK.line(g1, X, 124, [['肘不弯', E.INK2]], 21);
      var a4 = RK.line(g1, X, 160, [['手：绕着肩走一个', E.INK2], ['圆', E.J1, 900]], 21);
      var a5 = RK.line(g1, X, 214, [['手的速度', E.V, 900], ['（第 9 集）', E.MUT, 700]], 22);
      var a6 = RK.line(g1, X, 250, [['和到转动中心的连线', E.INK2], ['垂直', E.INK, 900]], 21);
      var a7 = RK.line(g1, X, 286, [['大小 = ', E.INK2], ['转速', E.OMEGA, 900], [' × ', E.INK2], ['距离', E.INK, 900]], 21);
      // 句 3–5 的算式
      var g2 = grp();
      var b1 = RK.line(g2, X, 40, [['从肩关节指向手：', E.INK2], ['(3, 2)', E.INK, 900]], 21);
      var b2 = RK.line(g2, X, 80, [['转 90°', E.OMEGA, 900], [' = 乘第 3 集的矩阵', E.INK2]], 21);
      var Rm = E.matN(g2, X + 190, 132, [[0, -1], [1, 0]], { size: 24, dx: 46, dy: 32, color: function (i, j) { return j === 0 ? E.X : E.Y; } });
      var rmL = E.txt(g2, X + 110, 140, 'R =', 22, E.INK2, 'end', 700);
      var b3 = RK.line(g2, X, 204, [['3 × ', E.INK, 900], ['(0, 1)', E.X, 900], [' = ', E.INK2], ['(0, 3)', E.X, 900]], 22);
      var b4 = RK.line(g2, X, 240, [['2 × ', E.INK, 900], ['(−1, 0)', E.Y, 900], [' = ', E.INK2], ['(−2, 0)', E.Y, 900]], 22);
      var b5 = RK.line(g2, X, 276, [['加起来：', E.INK2], ['(−2, 3)', E.INK, 900]], 22);
      var b6 = RK.line(g2, X, 322, [['× 转速 1：', E.OMEGA, 900], ['(−2, 3)', E.J1, 900]], 22);
      var b7 = RK.line(g2, X, 360, [['手每秒：', E.INK2], ['往左 2 格', E.X, 900], ['、', E.INK2], ['往上 3 格', E.Y, 900]], 21);
      // 句 6–8：肘
      var g3 = grp();
      var e1r = RK.line(g3, X, 46, [['② 只转', E.INK, 900], ['肘', E.HI, 900]], 24);
      var e2r = E.symRow(g3, RK, X, 88, '\\dot\\theta_2=1', E.OMEGA, 74, [['：每秒转 1 弧度', E.INK2]], 21);
      var e3r = RK.line(g3, X, 124, [['肩不动；手绕着肘走一个', E.INK2], ['圆', E.J2, 900]], 21);
      var f1 = RK.line(g3, X, 174, [['从肘关节指向手：', E.INK2]], 21);
      var f2 = RK.line(g3, X, 208, [['(3 − 1, 2 − 2)', E.INK, 900], [' = ', E.INK2], ['(2, 0)', E.INK, 900]], 21);
      var f3 = RK.line(g3, X, 248, [['转 90°：', E.OMEGA, 900], ['2 × ', E.INK, 900], ['(0, 1)', E.X, 900], [' = ', E.INK2], ['(0, 2)', E.X, 900]], 21);
      var f4 = RK.line(g3, X, 282, [['　　　 ', E.INK2], ['0 × ', E.INK, 900], ['(−1, 0)', E.Y, 900], [' = ', E.INK2], ['(0, 0)', E.Y, 900]], 21);
      var f5 = RK.line(g3, X, 316, [['加起来：', E.INK2], ['(0, 2)', E.INK, 900], ['；× 转速 1：', E.OMEGA, 900], ['(0, 2)', E.J2, 900]], 21);
      var f6 = RK.line(g3, X, 352, [['手每秒：', E.INK2], ['往上 2 格', E.Y, 900]], 21);
      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }
      function rot(v, a) { return [v[0] * Math.cos(a) - v[1] * Math.sin(a), v[0] * Math.sin(a) + v[1] * Math.cos(a)]; }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) { RK.layout(); laid = true; }
          var f = E.fade, b = E.bump;
          /* ── UR5：句 1 绕肩来回转，句 6 绕肘来回转，别的时候停在零位 ── */
          var w1 = E.win(st, T.swing[0], T.swing[1], 0.5), w2 = E.win(st, T.swing2[0], T.swing2[1], 0.5);
          var phi = 0.42 * Math.sin((st - T.swing[0]) * 1.9) * w1, psi = 0.6 * Math.sin((st - T.swing2[0]) * 1.9) * w2;
          pn.pose(E.ZERO[0] + phi, E.ZERO[1] + phi + psi);
          pn.view.clearTint();
          var J = pn.joints(), pS = P(0, 0), pE = P(J[1][0], J[1][1]), pH = P(J[2][0], J[2][1]);
          jS.at(pS); jE.at(pE);
          var sec1 = st < T.out1;
          E.glow(jS, E.HI, sec1 && st > T.only && st < T.n23); E.glow(jE, E.HI, !sec1 && st > T.elOnly && st < T.n02b);
          hand.at(pH, 7 + 4 * Math.max(b(st, T.circle, 0.8), b(st, T.circle2, 0.8)));
          // 转速、锁
          E.op(arcS.g, E.win(st, T.rate1, T.out1, 0.3)); E.op(thS, E.win(st, T.rate1, T.out1, 0.3));
          arcE.g.setAttribute('transform', 'translate(' + pE[0].toFixed(1) + ' ' + pE[1].toFixed(1) + ')');
          E.op(arcE.g, E.win(st, T.rate2, END + 1, 0.3));
          thE.setAttribute('transform', 'translate(' + (pE[0] + 4).toFixed(1) + ' ' + (pE[1] - 50).toFixed(1) + ')'); E.op(thE, E.win(st, T.rate2, T.from2, 0.3));
          lockE.setAttribute('transform', 'translate(' + (pE[0] - 10).toFixed(1) + ' ' + (pE[1] + 30).toFixed(1) + ')'); E.op(lockE, E.win(st, T.noBend, T.ep9, 0.3));
          lockS.setAttribute('transform', 'translate(' + (pS[0] + 22).toFixed(1) + ' ' + (pS[1] + 28).toFixed(1) + ')'); E.op(lockS, E.win(st, T.shLock, T.from2, 0.3));
          // 手走的圆弧
          E.drawOn(cir1, f(st, T.circle, 1.0)); E.op(cir1, (st > T.circle ? 0.85 : 0) * (1 - fr(st, T.from)));
          E.drawOn(cir2, f(st, T.circle2, 1.0)); E.op(cir2, (st > T.circle2 ? 0.85 : 0) * (1 - fr(st, T.from2)));
          // 句 2：连线、直角、方向
          var s2 = 1 - fr(st, T.from);
          E.drawOn(link, f(st, T.line, 0.5)); if (st > T.line) E.op(link, s2);
          E.op(rightMk, fr(st, T.perp) * s2); E.op(distL, fr(st, T.prod + 0.6) * s2);
          dirHint.at(3, 2, 3 + 0.9 * e2[0] * Math.sqrt(13) / 2, 2 + 0.9 * e2[1] * Math.sqrt(13) / 2); E.op(dirHint.g, fr(st, T.perp) * s2);
          // 句 3–5：从肩指向手、转 90°、按列拿份、平移到手上
          var grow = f(st, T.from, 1.0), turn = f(st, T.turn[0], T.turn[1] - T.turn[0]), mv = f(st, T.move[0], T.move[1] - T.move[0]);
          var dv = rot([3, 2], D90 * turn);
          var on1 = 1 - fr(st, T.out1);
          dArrow.at(0, 0, dv[0] * grow, dv[1] * grow); E.op(dArrow.g, (grow > 0 ? 1 : 0) * (1 - fr(st, T.n23 + 0.6)));
          dShadow.at(0, 0, 3, 2); E.op(dShadow.g, (turn > 0.02 ? 0.7 : 0) * (1 - fr(st, T.times1)));
          turnArc.set(A0, A0 + D90 * Math.max(turn, 0.01)); E.op(turnArc.g, (turn > 0.01 ? 1 : 0) * (1 - fr(st, T.byCol)));
          d32.setAttribute('x', (HH[0] - 8).toFixed(1)); d32.setAttribute('y', (HH[1] + 30).toFixed(1)); E.op(d32, fr(st, T.c3) * (1 - fr(st, T.turn[0])));
          red1.at(0, 0, 0, 3 * f(st, T.take3, 0.9)); E.op(red1.g, (st > T.take3 ? 1 : 0) * (1 - fr(st, T.times1)));
          grn1.at(0, 3, -2 * f(st, T.take2, 0.9), 3); E.op(grn1.g, (st > T.take2 ? 1 : 0) * (1 - fr(st, T.times1)));
          // 平移到手上：(−2, 3) 的尾巴从原点挪到 (3, 2)
          var tx = 3 * mv, ty = 2 * mv;
          c1.at(tx, ty, tx - 2, ty + 3);
          E.op(c1.g, fr(st, T.n23) * (st < T.out1 ? 1 : 0) + (st >= T.out1 ? E.fade(st, T.two2 - 0.2, 0.4) : 0));
          c1L.setAttribute('x', (P(tx - 2, ty + 3)[0] - 10).toFixed(1)); c1L.setAttribute('y', (P(tx - 2, ty + 3)[1] - 14).toFixed(1));
          E.op(c1L, (fr(st, T.n23) * (st < T.out1 ? 1 : 0) + (st >= T.out1 ? E.fade(st, T.two2 - 0.2, 0.4) : 0)) * (mv > 0.02 && mv < 0.98 ? 0 : 1));   // 平移途中不写：会扫过 y 轴的标签
          // 箭头颜色：先白（转好的方向），「加起来」亮，乘转速、平移后变亮黄
          var yel = f(st, T.times1, 0.4);
          c1.paint(yel > 0.5 ? E.J1 : E.INK);
          c1L.style.fill = yel > 0.5 ? E.J1 : E.INK;
          E.glow(c1.line, E.INK, b(st, T.n23, 0.8) > 0.05);
          E.drawOn(legU1, f(st, T.up, 0.5)); E.drawOn(legL1, f(st, T.left, 0.5));
          if (st > T.out1 - 0.4) { E.op(legU1, on1); E.op(legL1, on1); }
          E.op(legL1L, fr(st, T.left2) * on1); E.op(legU1L, fr(st, T.up3) * on1);
          // 句 7–8：肘
          var grow2 = f(st, T.from2, 1.0), turn2 = f(st, T.turn2[0], T.turn2[1] - T.turn2[0]), mv2 = f(st, T.move2[0], T.move2[1] - T.move2[0]);
          var dv2 = rot([2, 0], D90 * turn2);
          dArrow2.at(1, 2, 1 + dv2[0] * grow2, 2 + dv2[1] * grow2); E.op(dArrow2.g, (grow2 > 0 ? 1 : 0) * (1 - fr(st, T.n02b + 0.6)));
          dShadow2.at(1, 2, 3, 2); E.op(dShadow2.g, (turn2 > 0.02 ? 0.7 : 0) * (1 - fr(st, T.times1b)));
          turnArc2.set(0, D90 * Math.max(turn2, 0.01)); E.op(turnArc2.g, (turn2 > 0.01 ? 1 : 0) * (1 - fr(st, T.take0)));
          E.op(d20, fr(st, T.n2) * (1 - fr(st, T.turn2[0])));
          red2.at(1, 2, 1, 2 + 2 * f(st, T.take2b, 0.9)); E.op(red2.g, (st > T.take2b ? 1 : 0) * (1 - fr(st, T.n02b)));
          var ex = 2 * mv2;
          c2.at(1 + ex, 2, 1 + ex, 4); E.op(c2.g, fr(st, T.n02b));
          c2.paint(f(st, T.times1b, 0.4) > 0.5 ? E.J2 : E.INK);
          E.glow(c2.line, E.INK, b(st, T.n02b, 0.8) > 0.05);
          c2L.setAttribute('x', (P(1 + ex, 4)[0] + 12).toFixed(1)); c2L.setAttribute('y', (P(1 + ex, 4)[1] + 4).toFixed(1));
          c2L.style.fill = f(st, T.times1b, 0.4) > 0.5 ? E.J2 : E.INK; E.op(c2L, fr(st, T.n02b));

          /* ── 右栏 ── */
          E.op(g1, 1 - fr(st, T.from - 0.3));
          E.op(a1.g, fr(st, T.only)); E.op(a2.g, fr(st, T.rate1)); E.op(a3.g, fr(st, T.noBend)); E.op(a4.g, fr(st, T.swing[0])); E.op(a4.parts[1], fr(st, T.round));
          E.op(a5.g, fr(st, T.hv)); E.op(a6.g, fr(st, T.line)); E.op(a6.parts[1], fr(st, T.perp)); E.op(a7.g, fr(st, T.size)); E.op(a7.parts[1], fr(st, T.prod)); E.op(a7.parts[2], fr(st, T.prod + 0.5)); E.op(a7.parts[3], fr(st, T.prod + 0.5));
          E.op(g2, f(st, T.from - 0.2, 0.3) * on1);
          E.op(b1.g, fr(st, T.from)); E.op(b1.parts[1], fr(st, T.c3)); E.op(b2.g, fr(st, T.turn[0])); E.op(b2.parts[1], fr(st, T.mat - 1.0));
          E.op(Rm.g, fr(st, T.mat)); E.op(rmL, fr(st, T.mat));
          Rm.hl('c1', 0, 0, 1, 0, E.X, E.win(st, T.col1, T.byCol, 0.3) + E.win(st, T.take3, T.take2, 0.3));
          Rm.hl('c2', 0, 1, 1, 1, E.Y, E.win(st, T.col2, T.byCol, 0.3) + E.win(st, T.take2, T.sum, 0.3));
          E.op(b3.g, fr(st, T.take3)); E.op(b3.parts[2], fr(st, T.n03)); E.op(b3.parts[3], fr(st, T.n03));
          E.op(b4.g, fr(st, T.take2)); E.op(b4.parts[2], fr(st, T.n20)); E.op(b4.parts[3], fr(st, T.n20));
          E.op(b5.g, fr(st, T.sum)); E.op(b5.parts[1], fr(st, T.n23));
          E.op(b6.g, fr(st, T.times1)); E.op(b6.parts[1], fr(st, T.still));
          E.op(b7.g, fr(st, T.left)); E.op(b7.parts[2], fr(st, T.up)); E.op(b7.parts[3], fr(st, T.up));
          E.op(g3, f(st, T.elOnly - 0.2, 0.3));
          E.op(e1r.g, fr(st, T.elOnly)); E.op(e2r.g, fr(st, T.rate2)); E.op(e3r.g, fr(st, T.shLock)); E.op(e3r.parts[1], fr(st, T.round2));
          E.op(f1.g, fr(st, T.from2)); E.op(f2.g, fr(st, T.m31)); E.op(f2.parts[1], fr(st, T.got)); E.op(f2.parts[2], fr(st, T.n2));
          E.op(f3.g, fr(st, T.turn2[0])); E.op(f3.parts[1], fr(st, T.take2b)); E.op(f3.parts[2], fr(st, T.take2b)); E.op(f3.parts[3], fr(st, T.n02)); E.op(f3.parts[4], fr(st, T.n02));
          E.op(f4.g, fr(st, T.take0));
          E.op(f5.g, fr(st, T.sum2)); E.op(f5.parts[1], fr(st, T.n02b)); E.op(f5.parts[2], fr(st, T.times1b)); E.op(f5.parts[3], fr(st, T.times1b + 0.4));
          E.op(f6.g, fr(st, T.up2));
        }
      };
    }
  };
})();
