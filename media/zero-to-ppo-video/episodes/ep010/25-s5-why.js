// 010 第 5 幕：比一比——矩阵指数好在哪。左边先是第 1 幕的网格 + UR5，中段换成三张对照卡片（占满整幅），最后是斜着看的 UR5 和它的六根关节轴。
// 2026-11-01 按 005–009 的新配色复查：小臂淡黄，转动和角度蓝，打 × 用橙，手上的坐标架 x 红、y 绿、z 蓝；「九十除以二」右栏写 90° ÷ 2 = 45°。
// 句 1「两个朝向」小臂 0° 和 90° 两个位置画成虚线（第 5 集的两个朝向），「不能直接平均」右栏打橙 ×，「矩阵指数」右栏写 exp([S] θ / 2)，
//    「同一根轴」肘亮一圈，「只转一半的角度」小臂从 0° 转到 45°、标 45°，「九十除以二」写出 90° ÷ 2 = 45°；
// 句 2（2026-10-29 补）「两个朝向都不是原样」「矩阵对数」右栏下方写「两个都不是原样：先求矩阵对数」，「再只转一半」出 R₀ exp(½ log(R₀ᵀR₁))
//    （《Modern Robotics》3.2.3.3 节、Solà 等式 (25)(26)；005 的例子里一个朝向是原样，才是「绕同一根轴转一半」）；
// 句 3 换成三张卡片：欧拉角、四元数、矩阵指数（亮黄框）；
// 句 4「三个数」「万向锁」「四个数」「没有万向锁」「两套数」依次写进前两张卡片（2026-11-01 去掉「最直观」：《Modern Robotics》第 1 章说最直观的是指数坐标）；
// 句 5「旋转矩阵」「没有万向锁」「直接连乘」「运动旋量、螺旋轴」「九个数，有冗余」依次写进第三张；
// 句 6（2026-11-01 「都用它来写」改「也可以用它来写」：《Modern Robotics》4.5 节说多数做法用 D-H 参数）换成斜着看的 UR5：「关节」「一根螺旋轴」六根关节轴依次画出、标 S₁…S₆，「转了多少」第二根轴亮、胳膊绕它转一个角度、标 θ₂；
// 句 7「各算一个矩阵指数」右栏指数积公式的六个因子依次框亮、对应的轴鼓一下，「再乘上」出 M，「零度时、手的位姿」下面注明 M
//    （《Modern Robotics》式 (4.14) 最后乘 M：关节都在零位时手的位姿），「手现在的位姿」手上出坐标架，
//    「指数积公式」写名字，「第十一集」出第 11 集卡片。
// 时刻取自配音的逐词时间，见 ../ep010.py。
(function () {
  var E = window.EP010, K = E.K, H = E.HEX;
  var D = Math.PI / 180, A = E.ARM, RX = A.RX;
  var T = {
    ep5: 0.65, two: 1.76, mid: 2.44, avg: 3.33, mexp: 4.94, axis: 6.94, half: [8.17, 9.06], halfCalc: 9.72, log: 15.02, logEq: 19.3, out1: 21.15,
    cards: 21.4,
    eu: [25.73, 26.76], qu: [28.1, 28.74, 29.89], mx: [33.17, 34.7, 35.8, 37.2, 39.88], out2: 42.75,
    arm: 42.98, joint: 44.1, axes: [46.75, 46.99, 47.23, 47.47, 47.71, 47.95], turn: [48.52, 49.38], ang: 49.8,
    each: 51.65, facs: [52.48, 52.68, 53.04, 53.26, 53.48, 53.7], mult: 54.03, mM: 55.48, zero: 56.93, hand: 59.23, name: 61.3, ep11: 62.44
  };
  var END = 64.3;
  var JN = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'];

  E.scenes[4] = {
    title: '比一比：矩阵指数好在哪',
    dur: END,
    cues: [
      { at: 0, s: '两个**朝向**取中间值：绕同一根轴，只转一半的**角度**：90° ÷ 2 = 45°' },
      { at: 12.5, s: '两个朝向都不是原样：先求**矩阵对数**，再转一半' },
      { at: 21.31, s: '和第 6 集的另外 2 种写法比一比' },
      { at: 24.91, s: '欧拉角 3 个数，可会万向锁；四元数 4 个数，可一个朝向 2 套数' },
      { at: 32.28, s: '**矩阵指数**给出的旋转矩阵：没有万向锁、能直接相乘、和**螺旋轴**一套说法；代价是 9 个数' },
      { at: 42.89, s: '关节的运动也可以用它写：每个关节是一根**螺旋轴**，转了多少就是**角度**' },
      { at: 51.55, s: '各算一个**矩阵指数**按顺序相乘，再乘 M（关节都在 0° 时手的位姿）：手的**位姿**（第 11 集）' }
    ],
    build: function (K) {
      var s = K.sceneSvg('两个朝向取中间值不能直接平均，用矩阵指数绕同一根轴只转一半的角度；两个朝向都不是原样时先用矩阵对数求出两者之间绕什么轴、转了多少，再转一半；和第六集另外两种写法比：欧拉角三个数但会万向锁，四元数四个数没有万向锁但一个朝向两套数，矩阵指数给出的旋转矩阵没有万向锁、能直接相乘、和运动旋量螺旋轴是一套说法，代价是九个数有冗余；机械臂每个关节是一根螺旋轴，各算一个矩阵指数按顺序相乘，再乘上所有关节都在零度时手的位姿，就得到手现在的位姿，这叫指数积公式，第十一集讲');

      /* ══ ① 网格 + UR5：取中间值 ══ */
      var g1 = E.el('g', {}, s), under = E.el('g', {}, g1);
      var arm = E.armAt(g1, A.OX, A.OY, A.S, 330);
      var u = arm.aim(1, 2, 0);
      var pl = E.plane(under, { ox: A.OX, oy: A.OY, u: u, x0: -1, x1: 4, y0: -1, y1: 5 });
      var P = pl.P, V1 = arm.view, RIGID = ['forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link'], Q = P(1, 2);
      var top1 = E.el('g', {}, g1);
      var ghosts = [0, 90].map(function (a) {
        var g = E.el('g', {}, top1), e = P(1 + 2.06 * Math.cos(a * D), 2 + 2.06 * Math.sin(a * D));
        K.paint(E.el('path', { d: E.poly([Q, e]), fill: 'none', 'stroke-width': 4, 'stroke-dasharray': '8 6', 'stroke-linecap': 'round' }, g), null, E.BODY);
        K.paint(E.el('circle', { cx: e[0], cy: e[1], r: 6, 'stroke-width': 2.5 }, g), E.S1, E.BODY);
        E.halo(E.txt(g, e[0] + (a ? -14 : 0.75 * u), e[1] + 8, a + '°', 22, E.OMEGA, a ? 'end' : 'start', 900));   // 0° 写在手腕右边：手腕往右还伸出一截
        return g;
      });
      var ring = K.paint(E.el('circle', { cx: Q[0], cy: Q[1], r: 12, 'stroke-width': 3.5 }, top1), E.S1, E.HI);
      K.paint(E.el('circle', { cx: Q[0], cy: Q[1], r: 4 }, top1), E.HI);
      var halfArc = E.arcArrow(top1, Q[0], Q[1], 1.2 * u, E.OMEGA, 3.5, { head: 13 });
      var halfLab = E.halo(E.txt(top1, 0, 0, '45°', 24, E.OMEGA, 'start', 900));
      var RK = E.rowKit(), X = RX + 10;
      var r1 = E.el('g', {}, g1);
      var a1 = RK.line(r1, X, 64, [['第 5 集：两个朝向取中间值', E.INK, 900]], 22);
      var a2 = RK.line(r1, X, 104, [['直接平均', E.ERR, 900], ['：跳出旋转矩阵', E.INK2]], 22);
      var a2x = E.cross(a2.g, X + 260, 96, 9, E.ERR, 4);
      var a3 = RK.line(r1, X, 166, [['矩阵指数：', E.HI, 900], ['绕同一根轴', E.WARN, 900]], 22);
      var a4 = RK.line(r1, X, 206, [['　　　　　只转一半的角度', E.OMEGA, 900]], 22);
      var a5 = E.el('g', {}, r1);
      E.tex(a5, X, 268, '\\textcolor{' + H.hi + '}{\\exp\\!\\big([\\mathcal S]\\,\\tfrac{\\theta}{2}\\big)}', 30, E.INK, 'start', { w: 190, h: 60 });
      var a5c = E.txt(a5, X + 196, 268, '90° ÷ 2 = 45°', 21, E.OMEGA, 'start', 900);
      var a5ok = E.check(a5, X + 200, 296, 22, E.GOOD, 4);
      var a6 = RK.line(r1, X, 346, [['两个都不是原样：', E.INK, 900], ['先求矩阵对数', E.HI, 900]], 22);
      var a7 = E.el('g', {}, r1);
      E.tex(a7, X, 398, 'R_0\\exp\\!\\big(\\tfrac12\\log(R_0^{\\top}R_1)\\big)', 26, E.INK, 'start', { w: 300, h: 56 });

      /* ══ ② 三张卡片 ══ */
      var g2 = E.el('g', {}, s);
      E.txt(g2, 400, 40, '和第 6 集的另外 2 种写法比一比', 24, E.INK, 'middle', 900);
      var CW = 244, CX = [22, 278, 534], CY = 62, CH = 340;
      function card(i, name, color, hi) {
        var g = E.el('g', {}, g2);
        K.paint(E.el('rect', { x: CX[i], y: CY, width: CW, height: CH, rx: 16, 'stroke-width': hi ? 3.5 : 2.5 }, g), E.tint(color, hi ? 12 : 6, 'var(--demo-surface)'), color);
        E.txt(g, CX[i] + CW / 2, CY + 42, name, 26, color, 'middle', 900);
        return g;
      }
      var cE = card(0, '欧拉角', E.INK2), cQ = card(1, '四元数', E.INK2), cM = card(2, '矩阵指数', E.HI, true);
      var sub = E.txt(cM, CX[2] + CW / 2, CY + 74, '给出的旋转矩阵', 20, E.INK2, 'middle', 700);
      function item(gc, i, k, str, good, size) {   // 卡片里的一行：✓ / ✗ + 字
        var g = E.el('g', {}, gc), y = CY + 118 + 46 * k;
        if (good === true) E.check(g, CX[i] + 18, y - 16, 18, E.GOOD, 3.5);
        else if (good === false) E.cross(g, CX[i] + 27, y - 7, 7, E.ERR, 3.5);
        E.txt(g, CX[i] + (good === null ? CW / 2 : 46), y, str, size || 21, good === null ? E.HI : good ? E.INK : E.ERR, good === null ? 'middle' : 'start', 900);
        return g;
      }
      // 不写「最直观」：《Modern Robotics》第 1 章说最自然、最直观的是指数坐标（绕哪根轴转多少），欧拉角没有这个说法
      var iE = [item(cE, 0, 0, '3 个数', null, 24), item(cE, 0, 1, '会万向锁', false)];
      var iQ = [item(cQ, 1, 0, '4 个数', null, 24), item(cQ, 1, 1, '没有万向锁', true), item(cQ, 1, 2, '一个朝向两套数', false)];
      var iM = [item(cM, 2, 0.35, '没有万向锁', true), item(cM, 2, 1.25, '能直接相乘', true), item(cM, 2, 2.15, '和运动旋量、', true), item(cM, 2, 4.4, '9 个数，有冗余', false)];
      var iM2 = E.txt(cM, CX[2] + 46, CY + 118 + 46 * 2.15 + 30, '螺旋轴一套说法', 21, E.INK, 'start', 900);

      /* ══ ③ UR5 斜视 + 六根关节轴 + 指数积公式 ══ */
      var g3 = E.el('g', {}, s);
      var ur = window.Robot3D.ur5(g3, 92, 404, 2.9, { az: 32, el: 16 }), V = ur.view, THREE = window.Robot3D.THREE;
      var POSE0 = { shoulder_pan_joint: 20 * D, shoulder_lift_joint: -95 * D, elbow_joint: 75 * D, wrist_1_joint: -70 * D, wrist_2_joint: -90 * D, wrist_3_joint: 0 };
      V.camera({ box: [0, 0, 440, 420] });
      V.joints(POSE0);
      var axG = E.el('g', {}, g3);
      var axLines = JN.map(function () { return K.paint(E.el('path', { fill: 'none', 'stroke-width': 3.5, 'stroke-dasharray': '8 5' }, axG), null, E.WARN); });
      var axLabs = JN.map(function (n, i) { return E.halo(E.txt(axG, 0, 0, 'S' + '₁₂₃₄₅₆'[i], 22, E.WARN, 'middle', 900)); });
      var thLab = E.halo(E.txt(axG, 0, 0, 'θ₂', 24, E.OMEGA, 'start', 900));
      var toolG = E.el('g', {}, g3);
      var toolAx = [E.X, E.Y, E.Z].map(function (c) { return E.vec(toolG, c, 4, { head: 12 }); });
      function axisOf(name) {
        var j = V.robot.joints[name];
        j.updateMatrixWorld(true);
        return { o: new THREE.Vector3().setFromMatrixPosition(j.matrixWorld), d: j.axis.clone().transformDirection(j.matrixWorld) };
      }
      // 标签摆法同 009 第 5 幕：[哪一端（0 / 1 / 'mid'）, dx, dy, 对齐]；每根轴从关节往两边画多长（米）
      var LAB = [['mid', -26, 7, 'end'], [1, 22, 2, 'start'], [0, -24, 2, 'end'], [1, -8, -18, 'end'], [1, 12, 7, 'start'], [0, 0, -12, 'middle']];
      var LEN = [[-0.05, 0.3], [-0.22, 0.22], [-0.2, 0.2], [-0.16, 0.16], [-0.15, 0.15], [-0.14, 0.14]];
      var RK3 = E.rowKit(), r3 = E.el('g', {}, g3);
      var b1 = RK3.line(r3, X, 60, [['关节的运动也可以用它来写：', E.INK, 900]], 22);
      var b2 = RK3.line(r3, X, 96, [['每个关节 = 一根', E.INK2], ['螺旋轴', E.WARN, 900]], 22);
      var b3 = RK3.line(r3, X, 132, [['转了多少 = ', E.INK2], ['角度 θ', E.OMEGA, 900]], 22);
      var fm = E.el('g', {}, r3), FY = 190;
      // 两行：T手 = exp([S₁]θ₁) exp([S₂]θ₂) / ⋯ exp([S₆]θ₆) M（中间三个因子写成 ⋯；上标太小，不用 e 的上标写法）
      var tEq = E.tex(fm, X, FY, 'T_{\\text{手}}=', 24, E.HI, 'start', { w: 72, h: 44 });
      var facs = [[1, X + 70, FY], [2, X + 184, FY], [6, X + 104, FY + 46]].map(function (q) {
        var g = E.el('g', {}, fm);
        var box = K.paint(E.el('rect', { x: q[1] - 4, y: q[2] - 28, width: 110, height: 40, rx: 8, 'stroke-width': 2 }, g), E.tint(E.HI, 16, 'var(--demo-surface)'), E.HI);
        E.tex(g, q[1], q[2], '\\exp([\\mathcal S_' + q[0] + ']\\,\\theta_' + q[0] + ')', 22, E.INK, 'start', { w: 104, h: 44 });
        return { g: g, box: box };
      });
      var dots = E.tex(fm, X + 70, FY + 46, '\\cdots', 22, E.INK, 'start', { w: 30, h: 44 });
      var mTex = E.tex(fm, X + 220, FY + 46, 'M', 24, E.INK, 'start', { w: 30, h: 44 });
      var mNote = RK3.line(r3, X, FY + 82, [['M：所有关节都在 0° 时，手的位姿', E.MUT, 700]], 17);
      var nm = RK3.line(r3, X, FY + 122, [['指数积公式', E.HI, 900], ['（PoE）', E.INK2, 700]], 24);
      var c11 = E.el('g', {}, r3);
      K.paint(E.el('rect', { x: RX, y: 334, width: 340, height: 64, rx: 14, 'stroke-width': 2.5 }, c11), E.tint(E.GOOD, 10, 'var(--demo-surface)'), E.GOOD);
      E.txt(c11, RX + 18, 362, '第 11 集 · 正运动学', 22, E.GOOD, 'start', 900);
      E.txt(c11, RX + 18, 388, '六个关节连乘 → 手的位姿', 19, E.INK, 'start', 700);

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) {
            RK.layout(); RK3.layout(); laid = true;
          }
          var f = E.fade, b = E.bump;

          /* ── ① ── */
          E.op(g1, 1 - f(st, T.out1, 0.4));
          var phi = 45 * D * f(st, T.half[0], T.half[1] - T.half[0]);
          arm.bend(phi); V1.clearTint(); V1.tint(RIGID, E.BODY, 0.45);
          ghosts.forEach(function (g) { E.op(g, f(st, T.two, 0.4)); });
          ring.setAttribute('r', (12 + 5 * b(st, T.axis, 0.9)).toFixed(2)); E.glow(ring, E.HI, b(st, T.axis, 0.9) > 0.05);
          halfArc.set(4 * D, Math.max(5 * D, phi - 4 * D)); E.op(halfArc.g, f(st, T.half[0], 0.2));
          var hl = P(1 + 1.45 * Math.cos(22 * D), 2 + 1.45 * Math.sin(22 * D)); halfLab.setAttribute('x', hl[0].toFixed(1)); halfLab.setAttribute('y', (hl[1] + 8).toFixed(1));
          E.op(halfLab, f(st, T.half[1], 0.3));
          E.op(a1.g, fr(st, T.ep5)); E.op(a2.g, fr(st, T.avg)); E.drawOn(a2x, f(st, T.avg + 0.4, 0.3));
          E.op(a3.g, fr(st, T.mexp)); E.op(a3.parts[1], fr(st, T.axis)); E.op(a4.g, fr(st, T.half[0]));
          E.op(a5, fr(st, T.half[0] + 0.2)); E.op(a5c, fr(st, T.halfCalc)); E.drawOn(a5ok, f(st, T.halfCalc + 1.0, 0.3));
          E.op(a6.g, fr(st, T.log)); E.op(a7, fr(st, T.logEq));

          /* ── ② ── */
          E.op(g2, f(st, T.cards, 0.4) * (1 - f(st, T.out2, 0.4)));
          E.op(cE, 1); E.op(cQ, 1); E.op(cM, 1);
          iE.forEach(function (g, k) { E.op(g, fr(st, T.eu[k])); });
          iQ.forEach(function (g, k) { E.op(g, fr(st, T.qu[k])); });
          E.op(sub, fr(st, T.mx[0]));
          iM.forEach(function (g, k) { E.op(g, fr(st, [T.mx[1], T.mx[2], T.mx[3], T.mx[4]][k])); });
          E.op(iM2, fr(st, T.mx[3]));

          /* ── ③ ── */
          E.op(g3, f(st, T.arm, 0.4));
          var turn = 28 * D * f(st, T.turn[0], T.turn[1] - T.turn[0]);
          var pose = Object.assign({}, POSE0); pose.shoulder_lift_joint = POSE0.shoulder_lift_joint + turn;
          V.joints(pose);
          JN.forEach(function (n, i) {
            var a = axisOf(n), L = LEN[i];
            var p0 = V.project(a.o.clone().addScaledVector(a.d, L[0])), p1 = V.project(a.o.clone().addScaledVector(a.d, L[1]));
            axLines[i].setAttribute('d', 'M ' + p0[0].toFixed(1) + ' ' + p0[1].toFixed(1) + ' L ' + p1[0].toFixed(1) + ' ' + p1[1].toFixed(1));
            var on = f(st, T.axes[i], 0.35);
            E.op(axLines[i], on); E.op(axLabs[i], on);
            var lp = LAB[i], at = lp[0] === 'mid' ? [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2] : lp[0] ? p1 : p0;
            axLabs[i].setAttribute('x', (at[0] + lp[1]).toFixed(1)); axLabs[i].setAttribute('y', (at[1] + lp[2]).toFixed(1)); axLabs[i].style.textAnchor = lp[3];
            var pulse = Math.max(b(st, T.axes[i], 0.6), b(st, T.facs[i], 0.6), i === 1 ? f(st, T.turn[0], 0.3) * (1 - f(st, T.each, 0.4)) : 0);
            axLines[i].style.strokeWidth = 3.5 + 3 * pulse; E.glow(axLines[i], E.WARN, pulse > 0.05);
            if (i === 1) { thLab.setAttribute('x', (p1[0] + 16).toFixed(1)); thLab.setAttribute('y', (p1[1] + 32).toFixed(1)); }   // S₂ 标签下面
          });
          E.op(thLab, f(st, T.ang, 0.3) * (1 - f(st, T.each, 0.4)));
          var tl = V.robot.links.tool0 || V.robot.frames && V.robot.frames.tool0;
          V.robot.updateMatrixWorld(true);
          var to = new THREE.Vector3().setFromMatrixPosition(tl.matrixWorld), op0 = V.project(to);
          [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)].forEach(function (ax, i) {
            var tp = V.project(to.clone().addScaledVector(ax.transformDirection(tl.matrixWorld), 0.13));
            toolAx[i].set(op0[0], op0[1], tp[0], tp[1]);
          });
          var tb = b(st, T.hand, 0.9);
          E.op(toolG, f(st, T.hand, 0.4)); E.glow(toolG, E.HI, tb > 0.05);
          toolAx.forEach(function (a) { a.line.style.strokeWidth = 4 + 2 * tb; });
          E.op(b1.g, fr(st, T.joint)); E.op(b2.g, fr(st, T.axes[0])); E.op(b3.g, fr(st, T.turn[0]));
          E.op(fm, fr(st, T.each));
          [0, 1, 5].forEach(function (i, k) { E.op(facs[k].box, b(st, T.facs[i], 0.7)); });
          E.op(dots, 1); E.op(mTex, fr(st, T.mM)); E.op(tEq, fr(st, T.hand));
          E.op(mNote.g, fr(st, T.zero));
          E.op(nm.g, fr(st, T.name));
          E.pop(c11, RX + 170, 366, f(st, T.ep11, 0.35), 0.92);
        }
      };
    }
  };
})();
