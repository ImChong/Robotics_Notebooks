// 014 第 2 幕：站在手上看——物体运动旋量（《Modern Robotics》3.3.2 节式 (3.69)(3.70)：ω_b 按 {b} 的轴写，v_b 是 {b} 原点的速度、按 {b} 的轴写）。
// 句 1「手上也有一个坐标系」第 11 集的零位：手在 (3, 2)，手的坐标系 x 红朝右、y 绿朝上，「物体坐标系」右栏写；
// 句 2「肩转了九十度」胳膊连同手的坐标系逆时针转 90°（蓝弧），手到 (−2, 3)，「x 轴潮上」「y 轴潮左」各亮一下；
// 句 3「物体运动旋量」右栏 6 个数（ω₁…v₃）：「角速度」「手这一点自己的速度」；句 4「沿手的 x 轴」「沿手的 y 轴」手的两根轴各画一条长虚线；
// 句 5「只转肩」肩旁蓝色小弯箭头，「往左走三格、往下走两格」手上紫箭头 (−3, −2)（第 1 幕），「各要取几份」问号；
// 句 6「沿 x 轴就是负二」紫箭头尖的左边画红箭头往下 2 格（沿手的 x 轴反着走），「沿 y 轴就是三」从手往左画绿箭头 3 格，红绿头尾相接到紫箭头尖；
// 句 7 验算（右栏）；句 8「垂直屏幕潮外」原点和手各画一个 ⊙（z 轴朝外，蓝）；句 9 右栏 V_b = (0, 0, 1, −2, 3, 0) 逐个出；
// 句 10–12 右栏两列并排：站在底座上看 (0, 0, 1, 0, 0, 0)、站在手上看 (0, 0, 1, −2, 3, 0)，「两处」：① 原点那一点 / 手这一点（左边原点、手各画白圈），② 底座的轴 / 手的轴（两个坐标系各亮）。
// 时刻取自配音的逐词时间，见 ../ep014.py。
(function () {
  var E = window.EP014, K = E.K, H = E.HEX;
  var X = 455, D90 = Math.PI / 2;
  var T = {
    hand: 0.7, frame: 1.6, ep11: 2.29, xr: 3.61, yg: 5.27, book: 6.45, body: 7.04,
    rot: [9.2, 12.6],   // 「肩转了九十度，手的坐标系也跟着转九十度」：慢一点，转到「九十度」念完
    follow: 11.66, xWas: 13.05, xUp: 14.43, yWas: 15.56, yLeft: 16.95, out1: 18.76,
    stand: 19.06, twist: 20.24, six: 21.18, bodyTw: 22.27, front: 23.61, omega: 24.72, back: 25.63, self: 26.59,
    both: 29.27, byHand: 29.94, alongX: 31.79, alongY: 33.54,
    only: 36.2, hv: 43.26, left3: 44.12, down2: 45.19, axes: 46.73, howMany: 47.89,
    xUp2: 51.55, down2b: 52.8, isM2: 55.07, yLeft2: 56.43, left3b: 57.84, is3: 59.83,
    check: 61.29, cx: 61.92, cx2: 63.24, cy: 64.54, cy2: 65.81, sum: 67.05, sumR: 67.74, ok: 69,
    omegaZ: 70.78, handZ: 71.6, baseZ: 72.58, same: 73.61, out: 74.22, still1: 75.54,
    vb: 77.56, nums: [80.27, 80.85, 81.44, 82.01, 82.8, 83.39],
    sameMove: 84.84, two: 85.85, six2: 87.25, diff: 87.73, twoPlaces: 88.34,
    p1: 90.94, where: 92.33, baseP: 93.6, originP: 94.79, handP: 95.89, handPt: 97.05,
    p2: 98.99, whose: 99.75, baseAx: 100.87, handAx: 102.32,
    ep1: 37.02, fromSH: 37.96, n23s: 39.65, turnS: 40.83, r32s: 41.92
  };
  var END = 105.09;

  E.scenes[1] = {
    title: '站在手上看',
    dur: END,
    cues: [
      { at: 0, s: '手上的坐标系（第 11 集）：$x$ 红、$y$ 绿——书上叫**物体坐标系**' },
      { at: 9.06, s: '肩转了 $90^\\circ$，手的坐标系也跟着转 $90^\\circ$：$x$ 轴朝右 → 朝上，$y$ 轴朝上 → 朝左' },
      { at: 18.96, s: '**站在手上看**：**物体运动旋量**也是 $6$ 个数——前 $3$ 个角速度，后 $3$ 个是**手这一点自己的速度**' },
      { at: 29.17, s: '都按手的轴来算：沿手的 $x$ 轴走几格，沿手的 $y$ 轴走几格' },
      { at: 35.66, s: '只转肩（第 1 幕）：$(-2,\\,3)$ 转 $90^\\circ$ → **手的速度** $(-3,\\,-2)$；手的 $2$ 根轴各取几份？' },
      { at: 51.05, s: '$x$ 轴朝上：往下 $2$ 格 → $-2$；$y$ 轴朝左：往左 $3$ 格 → $3$' },
      { at: 61.19, s: '验算：$(-2)\\times(0,\\,1)+3\\times(-1,\\,0)=(-3,\\,-2)$ ✓' },
      { at: 70.68, s: '角速度：手的 $z$ 轴和底座的一样，都垂直屏幕朝外，还是 $1$' },
      { at: 77.46, s: '**站在手上看**：$(0,\\,0,\\,1,\\,-2,\\,3,\\,0)$' },
      { at: 84.74, s: '同一个动作，$6$ 个数不一样——不同的地方有 $2$ 处' },
      { at: 90.84, s: '① 后 $3$ 个数是什么地方的速度：底座看的是原点那一点，手上看的是手这一点' },
      { at: 98.89, s: '② 按谁的轴来算：一个按底座的轴，一个按手的轴' }
    ],
    build: function (K) {
      var s = K.sceneSvg('手上也有一个坐标系，第十一集讲过：红色是手的 x 轴，绿色是 y 轴，书上叫物体坐标系。肩转了九十度，手的坐标系也跟着转：手的 x 轴朝上，y 轴朝左。站在手上看，运动旋量也是六个数，叫做物体运动旋量：前三个数是角速度，后三个数是手这一点自己的速度，都按手的轴来算。只转肩时手每秒往左走三格、往下走两格：手的 x 轴朝上，往下走两格，沿 x 轴是负二；手的 y 轴朝左，往左走三格，沿 y 轴是三；验算 (−2) × (0, 1) + 3 × (−1, 0) = (−3, −2)。角速度：手的 z 轴和底座的一样，都垂直屏幕朝外，还是一。站在手上看，这个运动旋量是 (0, 0, 1, −2, 3, 0)；站在底座上看是 (0, 0, 1, 0, 0, 0)。不同的地方有两处：后三个数是什么地方的速度（原点那一点 / 手这一点），按谁的轴来算（底座的轴 / 手的轴）');

      var pn = E.panel14(s), P = pn.P, u = pn.u, top = pn.top, O = P(0, 0);
      var under2 = E.el('g', {}, top);
      var bF = E.frame2(top, P, 1, '底座', { labAt: [0.75, -0.62] });
      var jS = E.joint(top, E.HI), jE = E.joint(top, E.HI);
      var hF = E.frame2(top, P, 1, '手', { labAt: [0.62, 0.5], labColor: E.V });
      var hand = E.dot(top, E.V, 7);
      var arc90 = E.arcArrow(top, O[0], O[1], 0.95 * u, E.OMEGA, 3.5, { head: 12 });
      var a90L = E.halo(E.txt(top, 0, 0, '90°', 20, E.OMEGA, 'middle', 900));
      // 句 4：手的两根轴各一条长虚线
      var lnX = K.paint(E.el('path', { fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '7 6' }, under2), null, E.X);
      var lnY = K.paint(E.el('path', { fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '7 6' }, under2), null, E.Y);
      var rArc = E.arcArrow(top, O[0], O[1], 0.6 * u, E.OMEGA, 3.5, { head: 12 });
      var rLab = E.tex(top, O[0] - 132, O[1] + 46, '\\dot\\theta_1=1', 22, E.OMEGA, 'start', { w: 90, h: 36 });
      var wSH = E.gv(top, P, E.INK, 3.5, { head: 13, dash: '8 6' });
      var vio = E.gv(top, P, E.V, 5.5, { head: 17 }), vioL = E.halo(E.txt(top, P(-3.5, 2)[0] + 18, P(-3.5, 2)[1] + 30, '(−3, −2)', 19, E.V, 'middle', 900));
      var qM = E.halo(E.txt(top, P(-4.2, 3.4)[0], P(-4.2, 3.4)[1] + 10, '?', 32, E.INK, 'middle', 900));
      var red = E.gv(top, P, E.X, 5, { head: 14 }), grn = E.gv(top, P, E.Y, 5, { head: 14 });
      [red, grn, vio].forEach(function (a) { a.line.style.filter = 'drop-shadow(0 0 2px #000)'; });
      var redL = E.halo(E.txt(top, P(-5, 2)[0] - 12, P(-5, 2)[1] + 7, '−2', 21, E.X, 'end', 900));
      var grnL = E.halo(E.txt(top, P(-3.5, 3)[0], P(-3.5, 3)[1] - 14, '3', 21, E.Y, 'middle', 900));
      // 句 8：z 轴朝外的 ⊙（原点、手）
      function zOut(side) {
        var g = E.el('g', {}, top);
        K.paint(E.el('circle', { r: 10, 'stroke-width': 3 }, g), E.S1, E.Z);
        K.paint(E.el('circle', { r: 3.5 }, g), E.Z);
        var l = E.halo(E.txt(g, side * 14, -8, 'z', 19, E.Z, side > 0 ? 'start' : 'end', 900));
        g.at = function (p) { g.setAttribute('transform', 'translate(' + p[0].toFixed(1) + ' ' + p[1].toFixed(1) + ')'); return g; };
        return g;
      }
      var zB = zOut(1).at(P(0.75, 0.6)), zH = zOut(-1).at(P(-2.6, 3.62));   // 底座那个 z 写在右边（左上是底座 y 轴的字母）   // 手的两根轴中间（右边是 UR5 的手腕）
      // 句 11：原点那一点、手这一点的白圈
      function ring(p) { return K.paint(E.el('circle', { cx: p[0], cy: p[1], r: 14, fill: 'none', 'stroke-width': 3.5 }, top), null, E.INK); }
      var rgO = ring(O), rgH = ring(P(-2, 3));

      /* ══ 右栏 ══ */
      var RK = E.rowKit(), Rg = E.el('g', {}, s);
      function grp() { return E.el('g', {}, Rg); }
      var gA = grp();
      var a1 = RK.line(gA, X, 60, [['手的坐标系', E.V, 900], ['（第 11 集）', E.MUT, 700]], 23);
      var a2 = RK.line(gA, X, 104, [['x', E.X, 900], [' 红　', E.INK2], ['y', E.Y, 900], [' 绿', E.INK2]], 22);
      var a3 = RK.line(gA, X, 148, [['书上：', E.INK2], ['物体坐标系', E.INK, 900]], 22);
      var a4 = RK.line(gA, X, 182, [['（body frame，记作 {b}）', E.MUT, 700]], 17);
      var a5 = RK.line(gA, X, 240, [['肩转了 ', E.INK2], ['90°', E.OMEGA, 900], ['，跟着转：', E.INK2]], 21);
      var a6 = RK.line(gA, X, 280, [['手的 ', E.INK2], ['x', E.X, 900], [' 轴：朝右 → ', E.INK2], ['朝上', E.INK, 900]], 20);
      var a7 = RK.line(gA, X, 316, [['手的 ', E.INK2], ['y', E.Y, 900], [' 轴：朝上 → ', E.INK2], ['朝左', E.INK, 900]], 20);
      var gB = grp();
      var b1 = RK.line(gB, X, 50, [['站在手上看：', E.V, 900], ['物体运动旋量', E.WARN, 900]], 22);
      var twS = E.tw6(gB, X + 40, 180, ['ω₁', 'ω₂', 'ω₃', 'v₁', 'v₂', 'v₃'], { size: 22, dy: 28, w: 54 });
      var bW = E.txt(gB, X + 84, 152, '角速度', 20, E.OMEGA, 'start', 900);
      var bV = E.txt(gB, X + 84, 236, '手这一点自己的速度', 20, E.V, 'start', 900);
      var b2 = RK.line(gB, X, 318, [['都按', E.INK2], ['手的轴', E.V, 900], ['来算：', E.INK2]], 20);
      var b3 = RK.line(gB, X, 352, [['沿手的 ', E.INK2], ['x', E.X, 900], [' 轴走几格', E.INK2]], 19);
      var b4 = RK.line(gB, X, 384, [['沿手的 ', E.INK2], ['y', E.Y, 900], [' 轴走几格', E.INK2]], 19);
      var gC = grp();
      var c1 = RK.line(gC, X, 46, [['只转肩：', E.INK, 900], ['手的速度 ', E.INK2], ['(−3, −2)', E.V, 900]], 20);
      var c2 = RK.line(gC, X, 78, [['第 1 幕：', E.MUT, 700], ['(−2, 3)', E.INK, 900], [' 转 90° → ', E.OMEGA, 900], ['(−3, −2)', E.V, 900]], 16);
      var c3 = RK.line(gC, X, 116, [['手的两根轴，各取几份？', E.INK, 900]], 20);
      var c4 = RK.line(gC, X, 160, [['x', E.X, 900], [' 轴朝上：往下 2 格 → ', E.INK2], ['−2', E.X, 900]], 20);
      var c5 = RK.line(gC, X, 196, [['y', E.Y, 900], [' 轴朝左：往左 3 格 → ', E.INK2], ['3', E.Y, 900]], 20);
      var c6 = RK.line(gC, X, 244, [['验算：', E.INK2], ['(−2) × ', E.INK, 900], ['(0, 1)', E.X, 900], [' + ', E.INK2], ['3 × ', E.INK, 900], ['(−1, 0)', E.Y, 900]], 18);
      var c7 = RK.line(gC, X, 276, [['= ', E.INK2], ['(0, −2)', E.X, 900], [' + ', E.INK2], ['(−3, 0)', E.Y, 900], [' = ', E.INK2], ['(−3, −2)', E.V, 900], [' ✓', E.GOOD, 900]], 18);
      var c8 = RK.line(gC, X, 330, [['角速度：', E.OMEGA, 900], ['z', E.Z, 900], [' 轴和底座的一样', E.INK2]], 19);
      var c9 = RK.line(gC, X, 362, [['都垂直屏幕朝外 → 还是 ', E.INK2], ['1', E.OMEGA, 900]], 19);
      var gD = grp(), VY = 150;
      var dS = E.tw6(gD, X + 70, VY, [0, 0, 1, 0, 0, 0], { size: 21, dy: 25, w: 46, bracket: E.INK });
      var dB = E.tw6(gD, X + 250, VY, [0, 0, 1, -2, 3, 0], { size: 21, dy: 25, w: 46, bracket: E.V });
      var dSL = E.txt(gD, X + 70, 52, '站在底座上看', 18, E.INK, 'middle', 900), dBL = E.txt(gD, X + 250, 52, '站在手上看', 18, E.V, 'middle', 900);
      var neq = E.txt(gD, X + 160, VY + 8, '≠', 30, E.ERR, 'middle', 900);
      var e1 = RK.line(gD, X, 290, [['① 后 3 个数：', E.INK, 900], ['原点那一点', E.INK, 900], [' / ', E.MUT], ['手这一点', E.V, 900]], 18);
      var e2 = RK.line(gD, X, 330, [['② 按谁的轴：', E.INK, 900], ['底座的轴', E.INK, 900], [' / ', E.MUT], ['手的轴', E.V, 900]], 18);
      var e0 = RK.line(gD, X, 254, [['不同的地方有 2 处：', E.INK2]], 18);

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) { RK.layout(); laid = true; }
          var f = E.fade, b = E.bump;
          // UR5：开场是第 11 集的零位，句 2 逆时针转 90°（蓝弧），之后停在这一集的姿势
          var th = D90 * f(st, T.rot[0], T.rot[1] - T.rot[0]);
          pn.pose(E.ZERO[0] + th, th);
          pn.view.clearTint();
          var J = pn.joints(), pE = P(J[1][0], J[1][1]), pH = P(J[2][0], J[2][1]);
          jS.at(O); jE.at(pE); hand.at(pH, 7 + 4 * Math.max(b(st, T.hand, 0.8), b(st, T.self, 0.8), b(st, T.handPt, 0.8)));
          bF.set(0, 0, 0);
          hF.set(J[2][0], J[2][1], th);
          var lOff = 1 - E.win(st, T.down2b - 0.3, T.vb, 0.3);   // 红绿两段画在手的两根轴上时，x、y 两个字母先收起（不压线）
          E.op(hF.g, fr(st, T.frame)); E.op(hF.ax.g, fr(st, T.xr)); E.op(hF.lx, fr(st, T.xr) * lOff); E.op(hF.ay.g, fr(st, T.yg)); E.op(hF.ly, fr(st, T.yg) * lOff);
          E.glow(hF.ax.g, E.X, b(st, T.xUp, 1.0) > 0.05 || b(st, T.xUp2, 1.0) > 0.05);
          E.glow(hF.ay.g, E.Y, b(st, T.yLeft, 1.0) > 0.05 || b(st, T.yLeft2, 1.0) > 0.05);
          E.glow(hF.g, E.V, b(st, T.handAx, 1.2) > 0.05 || b(st, T.body, 1.0) > 0.05);
          E.glow(bF.g, E.INK, b(st, T.baseAx, 1.2) > 0.05);
          [bF.lx, bF.ly].forEach(function (n) { E.op(n, 1 - E.win(st, T.rot[0] - 0.3, T.out1, 0.3)); });   // 转 90° 的蓝弧从底座 y 轴的字母上扫过：先收起
          arc90.set(E.ZERO[0], E.ZERO[0] + Math.max(0.01, th));
          var on90 = fr(st, T.rot[0]) * (1 - fr(st, T.out1));
          E.op(arc90.g, on90);
          var la = 0.7, lq = P(1.6 * Math.cos(la), 1.6 * Math.sin(la));   // 右上：底座的 y 轴标签在 (0, 1.3)
          a90L.setAttribute('x', lq[0].toFixed(1)); a90L.setAttribute('y', (lq[1] + 6).toFixed(1)); E.op(a90L, fr(st, T.rot[1] - 0.3) * (1 - fr(st, T.out1)));
          // 句 4：手的两根轴的长虚线（这一集的姿势：x 轴是过手的竖线，y 轴是过手的横线）
          var hx = P(-2, -1.8), hx2 = P(-2, 5), hy = P(-6, 3), hy2 = P(2.8, 3);
          lnX.setAttribute('d', 'M ' + hx[0] + ' ' + hx[1] + ' L ' + hx2[0] + ' ' + hx2[1]);
          lnY.setAttribute('d', 'M ' + hy[0] + ' ' + hy[1] + ' L ' + hy2[0] + ' ' + hy2[1]);
          E.op(lnX, 0.8 * E.win(st, T.alongX, T.only, 0.4)); E.op(lnY, 0.8 * E.win(st, T.alongY, T.only, 0.4));
          // 句 5–7：只转肩（蓝色小弯箭头），紫箭头 (−3, −2)，红绿两段
          rArc.set(3.45, 4.55); var rOn = E.win(st, T.only, T.omegaZ, 0.3);
          E.op(rArc.g, rOn); E.op(rLab, rOn);
          vio.at(-2, 3, -5, 1); E.op(vio.g, E.win(st, T.hv, T.vb, 0.3)); E.op(vioL, E.win(st, T.left3, T.vb, 0.3));
          E.op(qM, E.win(st, T.axes, T.xUp2, 0.3) * (0.65 + 0.35 * b(st, T.howMany, 0.8)));
          var gr = f(st, T.left3b, 0.6), rd = f(st, T.down2b, 0.6);
          red.at(-5, 3, -5, 3 - 2 * rd); E.op(red.g, (st > T.down2b ? 1 : 0) * (1 - fr(st, T.vb)));
          grn.at(-2, 3, -2 - 3 * gr, 3); E.op(grn.g, (st > T.left3b ? 1 : 0) * (1 - fr(st, T.vb)));
          E.op(redL, E.win(st, T.isM2, T.vb, 0.3)); E.op(grnL, E.win(st, T.is3, T.vb, 0.3));
          // 句 8：⊙
          E.op(zH, E.win(st, T.handZ, T.vb, 0.3)); E.op(zB, E.win(st, T.baseZ, T.vb, 0.3));
          E.glow(zH, E.Z, b(st, T.out, 1.0) > 0.05); E.glow(zB, E.Z, b(st, T.out, 1.0) > 0.05);
          // 句 11：白圈
          E.op(rgO, E.win(st, T.originP, T.p2, 0.3)); E.op(rgH, E.win(st, T.handPt, T.p2, 0.3));

          /* ── 右栏 ── */
          E.op(gA, 1 - fr(st, T.out1));
          E.op(a1.g, fr(st, T.frame)); E.op(a1.parts[1], fr(st, T.ep11)); E.op(a2.g, fr(st, T.xr)); E.op(a2.parts[2], fr(st, T.yg)); E.op(a2.parts[3], fr(st, T.yg));
          E.op(a3.g, fr(st, T.book)); E.op(a3.parts[1], fr(st, T.body)); E.op(a4.g, fr(st, T.body + 0.4));
          E.op(a5.g, fr(st, T.rot[0])); E.op(a5.parts[2], fr(st, T.follow)); E.op(a6.g, fr(st, T.xWas)); E.op(a6.parts[3], fr(st, T.xUp)); E.op(a7.g, fr(st, T.yWas)); E.op(a7.parts[3], fr(st, T.yLeft));
          E.op(gB, E.win(st, T.stand - 0.1, T.only - 0.2, 0.3));
          E.op(b1.g, fr(st, T.stand)); E.op(b1.parts[1], fr(st, T.bodyTw)); E.op(twS.g, fr(st, T.six));
          twS.rangeBox(0, 2, E.OMEGA, E.win(st, T.omega, T.back, 0.3)); twS.rangeBox(3, 5, E.V, fr(st, T.self));
          E.op(bW, fr(st, T.omega)); E.op(bV, fr(st, T.self));
          E.op(b2.g, fr(st, T.both)); E.op(b2.parts[1], fr(st, T.byHand)); E.op(b2.parts[2], fr(st, T.byHand)); E.op(b3.g, fr(st, T.alongX)); E.op(b4.g, fr(st, T.alongY));
          E.op(gC, E.win(st, T.only - 0.1, T.vb - 0.2, 0.3));
          E.op(c1.g, fr(st, T.only)); E.op(c1.parts[1], fr(st, T.hv)); E.op(c1.parts[2], fr(st, T.hv)); E.op(c2.g, fr(st, T.ep1)); E.op(c2.parts[1], fr(st, T.n23s)); E.op(c2.parts[2], fr(st, T.turnS)); E.op(c2.parts[3], fr(st, T.r32s));
          wSH.at(0, 0, -2, 3); E.op(wSH.g, E.win(st, T.fromSH, T.hv + 0.4, 0.3));   // 第 1 幕的算法：从肩关节指向手（白虚线），转 90° 就是手的速度
          E.op(c3.g, fr(st, T.axes));
          E.op(c4.g, fr(st, T.xUp2)); E.op(c4.parts[2], fr(st, T.isM2)); E.op(c5.g, fr(st, T.yLeft2)); E.op(c5.parts[2], fr(st, T.is3));
          E.op(c6.g, fr(st, T.check)); E.op(c6.parts[1], fr(st, T.cx)); E.op(c6.parts[2], fr(st, T.cx)); [3, 4, 5].forEach(function (k) { E.op(c6.parts[k], fr(st, T.cy)); });
          E.op(c7.g, fr(st, T.cx2)); E.op(c7.parts[2], fr(st, T.cy2)); E.op(c7.parts[3], fr(st, T.cy2)); [4, 5].forEach(function (k) { E.op(c7.parts[k], fr(st, T.sumR)); }); E.op(c7.parts[6], fr(st, T.ok));
          E.op(c8.g, fr(st, T.omegaZ)); E.op(c8.parts[1], fr(st, T.handZ)); E.op(c8.parts[2], fr(st, T.baseZ)); E.op(c9.g, fr(st, T.out)); E.op(c9.parts[1], fr(st, T.still1));
          E.op(gD, fr(st, T.vb - 0.1));
          E.op(dB.g, fr(st, T.vb)); E.op(dBL, fr(st, T.vb));
          dB.cells.forEach(function (c, k) { E.op(c, fr(st, T.nums[k])); });
          E.op(dS.g, fr(st, T.two)); E.op(dSL, fr(st, T.two)); E.op(neq, fr(st, T.diff));
          dS.rangeBox(3, 5, E.ERR, fr(st, T.diff)); dB.rangeBox(3, 5, E.ERR, fr(st, T.diff));
          E.op(e0.g, fr(st, T.twoPlaces));
          E.op(e1.g, fr(st, T.p1)); E.op(e1.parts[1], fr(st, T.originP)); E.op(e1.parts[2], fr(st, T.handP)); E.op(e1.parts[3], fr(st, T.handP));
          E.op(e2.g, fr(st, T.p2)); E.op(e2.parts[1], fr(st, T.baseAx)); E.op(e2.parts[2], fr(st, T.handAx)); E.op(e2.parts[3], fr(st, T.handAx));
        }
      };
    }
  };
})();
