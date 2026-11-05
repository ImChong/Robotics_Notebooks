// 011 第 1 幕：两样东西——零位和螺旋轴。左边同第 9、10 集的侧视网格 + UR5（肩关节为原点，肘在 (1, 2)，小臂水平），
// 网格放开到 x −5…4、y −2…5（后面几幕手会转到 (−4, 1)、(0, −1)）；右栏一组组换。
// 句 1「两个关节」右栏写；「原点」「肩关节」原点亮一圈、写「肩」；「肘」「一、二」肘亮一圈、写 (1, 2)；「小臂水平」小臂鼓一下。
// 句 2「只让肩和肘转」肩、肘各晃一下又回来；「两根关节轴」肩、肘上出 ⊙（垂直于屏幕、朝向你，同 009）。
// 句 3「零位」右栏出「零位」牌；「这时候都是零度」写 θ₁ = 0°、θ₂ = 0°。
// 句 4「三、二」小臂上 (3, 2) 出紫点、写 (3, 2)；「手的坐标系」出手的两根小箭头：「x」红、「y」绿。
// 句 5「正向运动学」右栏写定义；「两个关节的角度」胳膊转到别的角度（θ₁ = 30°、θ₂ = 45°），手的坐标系跟着走、旁边打问号，念完回到零位。
// 句 6「两样东西」「第一样」右栏出 ①；「三乘三」M 出来（先是问号，最下面一排灰的 0、0、1）。
// 句 7 M 逐列填数：「x 轴是一、零」红的第一列（手的 x 轴鼓一下），「y 轴是零、一」绿的第二列，「补零」两列最下面的 0 亮；
//    「位置：三、二」紫的第三列（手的点鼓一下），「补一」最下面的 1 亮。
// 句 8「第二样」右栏出 ②；「肘关节的螺旋轴」肘上的 ⊙ 鼓一下，「零、零、一…」六个数逐个出来（琥珀）。
// 句 9 肩的螺旋轴逐步写：「方向」(0, 0, 1)，「轴上的点就是原点」「指到原点」原点 − 原点 = (0, 0)，「转九十度」还是 (0, 0)，「螺距」0；
//    「所以」三行换成 S₁ = (0, 0, 1, 0, 0, 0)。
// 句 10「肩转九十度」肩上画满 90° 的蓝弧、写 θ₁ = 90°，「肘也转九十度」肘上同样；「手最后停在」手旁打问号。
// 时刻取自配音的逐词时间，见 ../ep011.py。
(function () {
  var E = window.EP011, K = E.K, H = E.HEX;
  var D = Math.PI / 180, A = E.ARM, RX = A.RX, X = RX + 10;
  var T = {
    two: 1.19, view: 2.46, orig: 4.93, sh: 5.52, elb: 6.33, e12: 6.72, fore: 7.58,
    only: 9.66, wigS: [9.99, 11], wigE: [10.35, 11.4], axes: 11.14, perp: 12.48, you: 13.66,
    pick: 15.58, zero: 17.39, ang: 18.84, from: 19.65, deg0: 21.63, use: 22.53, outA: 26,
    on: 26.59, two2: 27.44, p32: 29.89, put: 30.86, hand: 31.33, ep7b: 32.71, ax: 34.66, ay: 35.75,
    fk: 37.78, know: 39.28, angs: 40.38, move: [40.38, 41.66], get: 41.02, pos: 43.13, dir: 43.72, back: [44.76, 45.76], outB: 45.76,
    two3: 47.2, first: 48.74, M: 50.74, ep7: 51.7, m33: 54.9,
    rot: 58.1, cx: 59.1, cx1: 59.71, cy: 60.53, cy1: 61.1, pad0: 62.59, pos3: 63.95, p3: 64.57, pad1: 66.34,
    second: 67.75, zeroPos: 70.46, elbS: 71.54, ep9: 73.02, nums: [74.27, 74.74, 75.28, 75.8, 76.28, 76.99],
    shS: 78.44, sh9: 79.7, dir1: 80.59, pt: 82.44, orig2: 83.44, to0: 84.43, z0: 85.64, turn: 86.68, still: 87.59, pitch: 88.98, so: 90.27, S1: 91,
    ex: 96.19, sh90: [97.45, 98.64], el90: [98.86, 100.03], where: 100.58
  };
  var END = 104.62;

  E.scenes[0] = {
    title: '两样东西：零位和螺旋轴',
    dur: END,
    cues: [
      { at: 0, s: '还是第 9、10 集的侧视图：原点在肩关节，肘在 (1, 2)，小臂水平' },
      { at: 9.56, s: '只动肩和肘：两根关节轴都垂直于屏幕，朝向你' },
      { at: 15.48, s: '**零位**：两个关节的**角度**都从这个姿势算起，这时 θ₁ = θ₂ = 0°' },
      { at: 26.29, s: '手的坐标系（同第 7 集的扫地机）放在小臂上的 (3, 2)：x 轴朝右，y 轴朝上' },
      { at: 37.68, s: '**正向运动学**：知道关节的**角度**，得到手停在什么**位置**、**朝向**什么方向' },
      { at: 46.02, s: '**指数积公式**只要两样东西。① **零位**时、手的**位姿** M：3 × 3 的矩阵' },
      { at: 57.22, s: '左边两列是**朝向**：x 轴 (1, 0)、y 轴 (0, 1)，补 0；右边一列是**位置** (3, 2)，补 1' },
      { at: 67.65, s: '② 每个关节的**螺旋轴**，按**零位**写：肘 S₂ = (0, 0, 1, 2, −1, 0)（第 9 集）' },
      { at: 78.34, s: '肩（照第 9 集算）：方向 (0, 0, 1)；原点 − 原点 = (0, 0)，转 90° 还是 (0, 0)；螺距 0 → S₁ = (0, 0, 1, 0, 0, 0)' },
      { at: 96.09, s: '例子：肩转 90°，肘也转 90°。手停在什么**位置**、**朝向**什么方向？' }
    ],
    build: function (K) {
      var s = K.sceneSvg('两个关节：第九、十集的侧视图，原点在肩关节，肘在 (1, 2)，小臂水平；肩和肘的关节轴都垂直于屏幕、朝向你。先选定一个姿势叫零位，两个关节的角度都从这里算起，这时都是零度。在小臂上离肘两格的 (3, 2) 放一个手的坐标系，x 轴朝右、y 轴朝上。正向运动学：知道两个关节的角度，得到手停在什么位置、朝向什么方向。指数积公式只要两样东西：零位时手的位姿 M = [[1, 0, 3], [0, 1, 2], [0, 0, 1]]，左边两列是朝向、右边一列是位置；每个关节的螺旋轴，也按零位写：肘 S₂ = (0, 0, 1, 2, −1, 0)，肩 S₁ = (0, 0, 1, 0, 0, 0)。例子：肩转九十度，肘也转九十度，手停在哪里？');

      var under = E.el('g', {}, s), armG = E.el('g', {}, s);
      var arm = E.arm2(armG, 320), u = arm.u, V = arm.view;
      var P = E.plane(under, { ox: A.OX, oy: A.OY, u: u, x0: A.X0, x1: A.X1, y0: A.Y0, y1: A.Y1 }).P;
      var RIGID = ['forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link'];
      var top = E.el('g', {}, s), O = P(0, 0);
      // 原点 / 肩、肘：亮一圈；⊙（关节轴垂直于屏幕、朝向你）
      function joint(x, y) {
        var g = E.el('g', {}, top), c = P(x, y);
        var ring = K.paint(E.el('circle', { cx: c[0], cy: c[1], r: 12, 'stroke-width': 3.5 }, g), E.S1, E.HI);
        var dot = K.paint(E.el('circle', { cx: c[0], cy: c[1], r: 4 }, g), E.HI);
        return { g: g, ring: ring, dot: dot, c: c };
      }
      var jS = joint(0, 0), jE = joint(1, 2);
      var shLab = E.halo(E.txt(top, O[0] - 22, O[1] + 8, '肩', 22, E.HI, 'end', 900));
      var elLab = E.halo(E.txt(top, jE.c[0] - 18, jE.c[1] - 18, '肘 (1, 2)', 22, E.HI, 'end', 900));
      // 句 4：手的坐标系
      var hand = E.hand2(top, P, 0.9);
      var p32 = E.halo(E.txt(top, 0, 0, '(3, 2)', 22, E.VEL, 'start', 900));
      var handLab = E.halo(E.txt(top, 0, 0, '手', 22, E.VEL, 'start', 900));
      var qm = E.halo(E.txt(top, 0, 0, '?', 34, E.HI, 'middle', 900));
      // 句 10：两段画满 90° 的蓝弧
      var arcS = E.arcArrow(top, O[0], O[1], 0.8 * u, E.OMEGA, 3.5, { head: 12 });
      var arcE = E.arcArrow(top, jE.c[0], jE.c[1], 0.8 * u, E.OMEGA, 3.5, { head: 12 });
      var th1 = E.halo(E.txt(top, O[0] - 14, O[1] - 0.95 * u, 'θ₁ = 90°', 22, E.OMEGA, 'end', 900));
      var th2 = E.halo(E.txt(top, jE.c[0] + 0.5 * u, jE.c[1] - 0.95 * u, 'θ₂ = 90°', 22, E.OMEGA, 'start', 900));
      var qm2 = E.halo(E.txt(top, P(3, 2)[0] + 40, P(3, 2)[1] - 30, '?', 34, E.HI, 'middle', 900));

      /* ══ 右栏 ══ */
      var RK = E.rowKit();
      function row(g, y, parts, size) { return RK.line(g, X, y, parts, size || 22); }
      // A：两个关节、零位
      var gA = E.el('g', {}, s);
      var a1 = row(gA, 40, [['两个关节：', E.INK2], ['肩', E.HI, 900], ['、', E.INK2], ['肘', E.HI, 900]], 24);
      var a2 = row(gA, 76, [['第 9、10 集的侧视图', E.INK2]]);
      var a3 = row(gA, 112, [['原点 = ', E.INK2], ['肩关节', E.HI, 900]]);
      var a4 = row(gA, 146, [['肘在 ', E.INK2], ['(1, 2)', E.HI, 900], ['，小臂水平', E.INK2]]);
      var a5 = row(gA, 192, [['⊙ ', E.HI, 900], ['关节轴：', E.INK2], ['垂直于屏幕', E.INK, 900]]);
      var a6 = row(gA, 224, [['　 朝向你（', E.INK2], ['z', E.Z, 900], [' 的方向）', E.INK2]]);
      var zeroTag = E.tag(gA, X, 252, 104, 44, '零位', E.GOOD, { size: 24, weight: 900, ink: E.GOOD });
      var a7 = row(gA, 284, [['：角度从这里算起', E.INK2]]);
      a7.x = X + 112;
      var a8 = row(gA, 334, [['θ₁ = 0°', E.OMEGA, 900], ['，', E.INK2], ['θ₂ = 0°', E.OMEGA, 900]], 24);
      var a9 = row(gA, 374, [['下面手算的例子：就用它', E.INK2]], 20);
      // B：手的坐标系、正向运动学
      var gB = E.el('g', {}, s);
      var b1 = row(gB, 40, [['手的坐标系', E.VEL, 900]], 24);
      var b2 = row(gB, 76, [['放在小臂上，离肘 2 格：', E.INK2], ['(3, 2)', E.VEL, 900]]);
      var b3 = row(gB, 110, [['x 轴', E.X, 900], ['朝右，', E.INK2], ['y 轴', E.Y, 900], ['朝上', E.INK2]]);
      var b3b = row(gB, 140, [['（坐标系：同第 7 集的扫地机）', E.MUT, 700]], 18);
      var b4 = row(gB, 176, [['正向运动学', E.HI, 900]], 28);
      var b5 = row(gB, 220, [['知道：两个关节的', E.INK2], ['角度', E.OMEGA, 900], [' θ₁、θ₂', E.OMEGA, 900]]);
      var b6 = row(gB, 258, [['得到：手的坐标系', E.INK2]]);
      var b7 = row(gB, 294, [['　停在什么', E.INK2], ['位置', E.VEL, 900]]);
      var b8 = row(gB, 328, [['　朝向什么', E.INK2], ['方向', E.OMEGA, 900]]);
      // C：两样东西
      var gC = E.el('g', {}, s);
      var c0 = row(gC, 30, [['指数积公式', E.HI, 900], ['只要两样东西：', E.INK2]], 22);
      var c1 = row(gC, 66, [['① ', E.HI, 900], ['零位时、手的位姿', E.INK, 900]], 22);
      var MX = RX + 196, MY = 124, MW = [[1, 0, 3], [0, 1, 2], [0, 0, 1]];
      var EXT = E.matExt(MW, { size: 26, dx: 54 });
      var mG = E.el('g', {}, gC);
      E.tex(mG, MX - EXT[0] - 8, MY + 10, 'M=', 28, E.INK, 'end', { w: 60, h: 46 });
      var M = E.matN(mG, MX, MY, [['?', '?', '?'], ['?', '?', '?'], [0, 0, 1]], { size: 26, dx: 54, dy: 34, wide: MW,
        color: function (i, j) { return i === 2 ? E.MUT : j === 0 ? E.X : j === 1 ? E.Y : E.VEL; } });
      M.cells.forEach(function (r) { r.forEach(function (n) { n.__col = n.style.fill; }); });
      var rotLab = E.txt(mG, (M.pos(0, 0)[0] + M.pos(0, 1)[0]) / 2, MY + 80, '朝向', 20, E.OMEGA, 'middle', 900);
      var posLab = E.txt(mG, M.pos(0, 2)[0], MY + 80, '位置', 20, E.VEL, 'middle', 900);
      var c2 = row(gC, 236, [['② ', E.HI, 900], ['每个关节的', E.INK, 900], ['螺旋轴', E.WARN, 900], ['（按零位写）', E.INK2, 700]], 22);
      var S2 = row(gC, 270, [['肘：', E.INK2], ['S₂ = (', E.WARN, 900], ['0', E.WARN, 900], [', ', E.WARN, 900], ['0', E.WARN, 900], [', ', E.WARN, 900], ['1', E.WARN, 900],
        [', ', E.WARN, 900], ['2', E.WARN, 900], [', ', E.WARN, 900], ['−1', E.WARN, 900], [', ', E.WARN, 900], ['0', E.WARN, 900], [')', E.WARN, 900]], 21);
      var dG = E.el('g', {}, gC);
      var d1 = row(dG, 304, [['肩（照第 9 集）：方向 ', E.INK2], ['(0, 0, 1)', E.Z, 900]], 20);
      var d2 = row(dG, 334, [['原点 − 原点 = ', E.INK2], ['(0, 0)', E.VEL, 900], ['，转 90°：', E.INK2], ['(0, 0)', E.VEL, 900]], 20);
      var d3 = row(dG, 364, [['螺距 ', E.INK2], ['0', E.GOOD, 900]], 20);
      var S1 = row(gC, 304, [['肩：', E.INK2], ['S₁ = (0, 0, 1, 0, 0, 0)', E.WARN, 900]], 21);
      var e1 = row(gC, 350, [['例子：', E.INK2], ['θ₁ = 90°', E.OMEGA, 900], ['，', E.INK2], ['θ₂ = 90°', E.OMEGA, 900]], 22);
      var e2 = row(gC, 388, [['手停在什么', E.INK2], ['位置', E.VEL, 900], ['、朝向什么', E.INK2], ['方向', E.OMEGA, 900], ['？', E.HI, 900]], 21);

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) { RK.layout(); laid = true; }
          var f = E.fade, b = E.bump;

          /* ── 胳膊：句 2 肩、肘各晃一下；句 5 转到别的角度再回来 ── */
          var w1 = 12 * D * Math.sin(Math.PI * E.clamp((st - T.wigS[0]) / (T.wigS[1] - T.wigS[0]), 0, 1));
          var w2 = 15 * D * Math.sin(Math.PI * E.clamp((st - T.wigE[0]) / (T.wigE[1] - T.wigE[0]), 0, 1));
          var mv = f(st, T.move[0], T.move[1] - T.move[0]) * (1 - f(st, T.back[0], T.back[1] - T.back[0]));
          var t1 = w1 + 30 * D * mv, t2 = w2 + 45 * D * mv;
          arm.pose(t1, t2);
          V.clearTint(); V.tint(RIGID, E.BODY, 0.45 + 0.35 * b(st, T.fore, 0.8));
          var fk = E.fk2(t1, t2);

          /* ── 原点、肘、⊙ ── */
          var ec = P(fk.elbow[0], fk.elbow[1]);
          [jE.ring, jE.dot].forEach(function (n) { n.setAttribute('cx', ec[0].toFixed(1)); n.setAttribute('cy', ec[1].toFixed(1)); });
          E.op(jS.g, f(st, T.orig, 0.3)); E.op(jE.g, f(st, T.elb, 0.3));
          var gs = Math.max(b(st, T.orig, 0.8), b(st, T.axes, 0.8), b(st, T.shS, 0.9)), ge = Math.max(b(st, T.elb, 0.8), b(st, T.axes + 0.3, 0.8), b(st, T.elbS, 0.9));
          jS.ring.setAttribute('r', (12 + 5 * gs).toFixed(2)); E.glow(jS.ring, E.HI, gs > 0.05);
          jE.ring.setAttribute('r', (12 + 5 * ge).toFixed(2)); E.glow(jE.ring, E.HI, ge > 0.05);
          E.op(shLab, f(st, T.sh, 0.3)); E.op(elLab, f(st, T.e12, 0.3) * (1 - mv) * (1 - f(st, T.ex, 0.3)));

          /* ── 手的坐标系 ── */
          hand.set(fk.hand[0], fk.hand[1], fk.ang);
          E.op(hand.dot, f(st, T.p32, 0.3)); E.op(hand.ax.g, f(st, T.ax, 0.3)); E.op(hand.ay.g, f(st, T.ay, 0.3));
          var bx = b(st, T.cx, 0.9), by = b(st, T.cy, 0.9), bp = Math.max(b(st, T.pos3, 0.9), b(st, T.p32, 0.8));
          hand.ax.line.style.strokeWidth = 4.5 + 2.5 * bx; E.glow(hand.ax.g, E.X, bx > 0.05);
          hand.ay.line.style.strokeWidth = 4.5 + 2.5 * by; E.glow(hand.ay.g, E.Y, by > 0.05);
          hand.dot.setAttribute('r', (6 + 4 * bp).toFixed(2));
          var hc = P(fk.hand[0], fk.hand[1]);
          p32.setAttribute('x', (hc[0] + 12).toFixed(1)); p32.setAttribute('y', (hc[1] + 30).toFixed(1));
          E.op(p32, f(st, T.p32, 0.3) * (1 - mv));
          handLab.setAttribute('x', (hc[0] + 22).toFixed(1)); handLab.setAttribute('y', (hc[1] - 12).toFixed(1));
          E.op(handLab, f(st, T.hand, 0.3) * (1 - mv) * (1 - f(st, T.ex, 0.3)));
          qm.setAttribute('x', (hc[0] + 34).toFixed(1)); qm.setAttribute('y', (hc[1] - 18).toFixed(1));
          E.op(qm, f(st, T.pos, 0.3) * (1 - f(st, T.back[0], 0.3)));

          /* ── 句 10：两段 90° 的弧 ── */
          var aS = f(st, T.sh90[0], T.sh90[1] - T.sh90[0]), aE = f(st, T.el90[0], T.el90[1] - T.el90[0]);
          arcS.set(0, Math.max(1e-3, 90 * D * aS)); E.op(arcS.g, f(st, T.sh90[0], 0.2));
          arcE.set(0, Math.max(1e-3, 90 * D * aE)); E.op(arcE.g, f(st, T.el90[0], 0.2));
          E.op(th1, f(st, T.sh90[1] - 0.3, 0.3)); E.op(th2, f(st, T.el90[1] - 0.3, 0.3)); E.op(qm2, f(st, T.where, 0.3));

          /* ── 右栏 ── */
          E.op(gA, 1 - f(st, T.outA, 0.4));
          E.op(a1.g, fr(st, T.two)); E.op(a2.g, fr(st, T.view)); E.op(a3.g, fr(st, T.orig)); E.op(a3.parts[1], fr(st, T.sh));
          E.op(a4.g, fr(st, T.elb)); E.op(a4.parts[2], fr(st, T.fore));
          E.op(a5.g, fr(st, T.axes)); E.op(a5.parts[2], fr(st, T.perp)); E.op(a6.g, fr(st, T.you));
          E.pop(zeroTag.g, X + 52, 274, f(st, T.zero, 0.35), 0.85); E.op(a7.g, fr(st, T.from));
          E.op(a8.g, fr(st, T.deg0)); E.op(a9.g, fr(st, T.use));
          E.op(gB, fr(st, T.on) * (1 - f(st, T.outB, 0.4)));
          E.op(b1.g, fr(st, T.hand)); E.op(b2.g, fr(st, T.two2)); E.op(b2.parts[1], fr(st, T.p32));
          E.op(b3b.g, fr(st, T.ep7b)); E.op(b3.g, fr(st, T.ax)); E.op(b3.parts[2], fr(st, T.ay)); E.op(b3.parts[3], fr(st, T.ay));
          E.op(b4.g, fr(st, T.fk)); E.op(b5.g, fr(st, T.know)); E.op(b5.parts[1], fr(st, T.angs)); E.op(b5.parts[2], fr(st, T.angs));
          E.op(b6.g, fr(st, T.get)); E.op(b7.g, fr(st, T.pos)); E.op(b8.g, fr(st, T.dir));
          E.op(gC, fr(st, T.two3));
          E.op(c0.g, 1); E.op(c1.g, fr(st, T.first)); E.pop(mG, MX, MY, f(st, T.m33, 0.35), 0.9);
          E.op(rotLab, fr(st, T.rot)); E.op(posLab, fr(st, T.pos3));
          var fill = [[0, 0, '1', T.cx1], [1, 0, '0', T.cx1 + 0.35], [0, 1, '0', T.cy1], [1, 1, '1', T.cy1 + 0.35], [0, 2, '3', T.p3], [1, 2, '2', T.p3 + 0.45]];
          fill.forEach(function (c) {
            var n = M.cells[c[0]][c[1]], q = M.pos(c[0], c[1]), p = f(st, c[3], 0.3);
            n.textContent = st >= c[3] ? c[2] : '?';
            E.scaleAt(n, q[0], q[1], st >= c[3] ? 1 + 0.25 * (1 - p) : 1);
            n.style.fill = st >= c[3] ? n.__col : E.MUT;
          });
          // 最下面一排：「补零」两列的 0 亮，「补一」位置的 1 亮
          M.hl('c0', 0, 0, 2, 0, E.X, b(st, T.cx, 1.0)); M.hl('c1', 0, 1, 2, 1, E.Y, b(st, T.cy, 1.0));
          M.hl('pad0', 2, 0, 2, 1, E.INK2, b(st, T.pad0, 1.0)); M.hl('c2', 0, 2, 2, 2, E.VEL, b(st, T.pos3, 1.0)); M.hl('pad1', 2, 2, 2, 2, E.INK2, b(st, T.pad1, 0.9));
          E.op(c2.g, fr(st, T.second)); E.op(S2.g, fr(st, T.elbS));
          S2.parts.forEach(function (n, k) {
            if (k < 2) return;
            var j = Math.floor((k - 2) / 2), ok = k === 13 ? fr(st, T.nums[5]) : fr(st, T.nums[Math.min(j, 5)]);
            E.op(n, ok);
          });
          E.op(dG, fr(st, T.shS) * (1 - f(st, T.so, 0.4)));
          E.op(d1.g, fr(st, T.shS)); E.op(d1.parts[1], fr(st, T.dir1)); E.op(d2.g, fr(st, T.to0)); E.op(d2.parts[1], fr(st, T.z0));
          E.op(d2.parts[2], fr(st, T.turn)); E.op(d2.parts[3], fr(st, T.still)); E.op(d3.g, fr(st, T.pitch));
          E.op(S1.g, fr(st, T.S1));
          E.op(e1.g, fr(st, T.ex)); E.op(e1.parts[1], fr(st, T.sh90[0])); E.op(e1.parts[3], fr(st, T.el90[0])); E.op(e2.g, fr(st, T.where));
        }
      };
    }
  };
})();
