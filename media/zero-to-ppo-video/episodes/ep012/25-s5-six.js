// 012 第 5 幕：六个关节——最多几组解。UR5 的 8 组解是同一个手的位姿（tool0）用 URDF 正运动学从 300 个初值做数值逆解、去重得到的
// （../ep012.py 文件头；结构同 ur_kinematics：底座 q1 两个 × 手腕 q5 两个 × 肘 q3 两个），每组的 tool0 和参考姿势逐项相同（误差 < 1e-8）。
// 句 1「真实的机械臂」左边一台斜着看的 UR5（参考姿势），「六个关节」六个关节亮圆点，「位置和朝向」手上出三根小箭头（目标位姿，之后一直留着）；
// 句 2「现成的公式」「开源代码里」右栏写 ur_kinematics，「三处，各有两个答案」写；
// 句 3「底座那个关节」「朝前伸」→「转过去、反着伸过来」UR5 换成底座反过来的那组，「肘，可以朝上」→「也可以朝下」换成肘下那组，
//    「手腕，也有两种翻法」换成手腕翻过去的那组（每次手都停在同一个位姿）；右栏逐行写；
// 句 4「两种、两种、再两种，一共八种」右栏淡出，整幅铺开 2 × 4 八台 UR5（上排朝前伸、下排反着伸；列：肘上 / 肘下 × 手腕两种翻法），
//    按念到的「两种」依次亮出，「手停在同一个位姿」八台手上的小箭头一起鼓一下；
// 句 5「最多十六组解」八台淡出，右栏写 16；
// 句 6「关节比需要的多」「平面里三个关节」左边画一条三节的平面链，「手腕不动，前面的关节还能一起挪」末端钉住、三个关节一起来回挪，
//    「人形机器人的一条胳膊有七个关节」换成正面的 G1，左臂七个关节亮圆点；
// 句 7「手腕转到某些姿势」「万向锁」UR5 手腕中间那个关节转到 0，两根转轴画成平行的蓝线；
// 句 8 总结：左边 UR5 依次摆出 8 组解，右栏写「这台最多 8 组，最一般最多 16 组；肘上、肘下 = 开平方的正负两个」。
// 时刻取自配音的逐词时间，见 ../ep012.py。
(function () {
  var E = window.EP012, K = E.K, H = E.HEX;
  var D = Math.PI / 180, RX = E.ARM.RX;
  var T = {
    real: 0.95, six: 2.16, ori: 3.31,
    ik: 7.13, formula: 8.37, code: 9.39, three: 12.3, two: 13.39,
    base: 15.14, front: 16.99, back: [18.26, 19.44], elbow: 19.98, up: 20.66, down: [21.48, 22.3], wrist: 22.44, flip: [23.11, 24.0], out3: 25.0,
    grid: 25.3, g2: [25.39, 26.11, 26.78], eight: 27.54, same: 28.51, most8: 31.22, out4: 33.8,
    gen: 34.1, proved: 36.87, s16: 37.51,
    more: 39.99, inf: 42.19, planar: 43.55, three3: 44.05, still: 46.44, slide: 47.32, human: 49.43, seven: 51.66,
    wristS: 54.64, pose7: 55.75, gimbal: 57.99, inf2: 59.22,
    sum: 61.39, finite: 64.21, s8: 65.76, s16b: 68.05, elb: 68.98, sqrt: 71.62
  };
  var END = 74.15;
  // 8 组解（关节顺序：shoulder_pan、shoulder_lift、elbow、wrist_1、wrist_2、wrist_3，弧度），排成 2 × 4：上排底座朝前，下排反着；
  // 每排：肘上（肘在肩—手腕连线上方）·手腕 ①、肘上·手腕 ②、肘下·手腕 ①、肘下·手腕 ②（手腕 ① / ②：wrist_2 为负 / 正）
  var SOL = [
    [0.3000, -1.2500, 1.7000, -2.0000, -1.5700, 0.4000], [0.3000, -0.8121, 1.0166, 1.3871, 1.5700, -2.7416],
    [0.3000, 0.3588, -1.7000, -0.2088, -1.5700, 0.4000], [0.3000, 0.1599, -1.0166, 2.4483, 1.5700, -2.7416],
    [-2.4707, -2.3294, -1.0163, 1.7559, -1.5791, -2.3708], [-2.4707, -1.8918, -1.7002, -1.1395, 1.5791, 0.7708],
    [-2.4707, 2.9821, 1.0163, 0.6949, -1.5791, -2.3708], [-2.4707, 2.7824, 1.7002, -2.9309, 1.5791, 0.7708]
  ];
  var JN = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'];
  function setQ(v, q) { var o = {}; JN.forEach(function (n, i) { o[n] = q[i]; }); v.joints(o); }
  function lerpQ(a, b, k) { return a.map(function (x, i) { var d = b[i] - x; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI; return x + d * k; }); }

  E.scenes[4] = {
    title: '六个关节：最多几组解',
    dur: END,
    cues: [
      { at: 0, s: '真实的机械臂：6 个关节，手的位置和朝向都要对上' },
      { at: 5.95, s: '这台的逆向运动学有现成的公式（开源代码 ur_kinematics）：3 处各有 2 个答案' },
      { at: 15.05, s: '底座：朝前伸 / 反着伸；肘：朝上 / 朝下；手腕：两种翻法' },
      { at: 25.3, s: '$2\\times 2\\times 2=8$：手停在同一个位姿，最多 **8 组**' },
      { at: 34.01, s: '最一般的 6 个关节：已有人证明，最多 **16 组**' },
      { at: 39.89, s: '关节比需要的多：解有**无穷多组**（平面 3 个关节管手腕位置；人形一条胳膊 7 个关节）' },
      { at: 54.55, s: '手腕转到某些姿势（像第 6 集的万向锁）：也有**无穷多组**' },
      { at: 61.29, s: '6 个关节一般有限几组：这台最多 **8 组**，最一般最多 **16 组**；**肘上**、**肘下** = 开平方的正负两个' }
    ],
    build: function (K) {
      var s = K.sceneSvg('真实的机械臂有六个关节，手的位置和朝向都要对上；这台机械臂的逆向运动学有现成的公式，开源代码 ur_kinematics 里就有：一步一步解下来，有三处各有两个答案——底座朝前伸或反着伸，肘朝上或朝下，手腕两种翻法，2 × 2 × 2 = 8，手停在同一个位姿最多摆出八种姿势；最一般的六关节机械臂已有人证明最多十六组解；关节比需要的多，解有无穷多组：平面里三个关节只管手腕位置，手腕不动前面的关节还能一起挪，人形机器人一条胳膊七个关节也是这样；手腕转到某些姿势像第六集的万向锁，也有无穷多组解；所以六关节机械臂一般有限几组解：这台最多八组，最一般最多十六组，肘上肘下就是开平方的正负两个');

      /* ── 左：一台大的 UR5 ── */
      var bigG = E.el('g', {}, s);
      var big = window.Robot3D.ur5(bigG, 215, 330, 2.3, { az: 32, el: 16 }), BV = big.view;
      BV.camera({ box: [0, 0, 440, 420] });
      setQ(BV, SOL[0]);
      var top = E.el('g', {}, s);
      var LINKS = ['shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link'];
      var jd = LINKS.map(function () { var c = K.paint(E.el('circle', { r: 7, 'stroke-width': 3 }, top), E.S1, E.HI); return c; });
      function frame(parent, w) { return [E.X, E.Y, E.Z].map(function (c) { return E.vec(parent, c, w || 4, { head: 11 }); }); }
      var bigF = frame(top, 4.5);
      var axesG = E.el('g', {}, top), ax1 = K.paint(E.el('path', { fill: 'none', 'stroke-width': 4, 'stroke-linecap': 'round' }, axesG), null, E.OMEGA), ax3 = K.paint(E.el('path', { fill: 'none', 'stroke-width': 4, 'stroke-linecap': 'round' }, axesG), null, E.OMEGA);
      var axLab = E.halo(E.txt(axesG, 0, 0, '两根轴平行', 20, E.OMEGA, 'end', 900));
      var solLab = E.halo(E.txt(top, 30, 40, '', 22, E.HI, 'start', 900));
      var GOAL = SOL[0];
      // 目标位姿：参考姿势下 tool0 的三根小箭头在画面里的位置（大 UR5）
      function toolArrows(v, arr, L) {
        var p = v.at('tool0');
        [[L, 0, 0], [0, L, 0], [0, 0, L]].forEach(function (d, k) { var q = v.at('tool0', d); arr[k].set(p[0], p[1], q[0], q[1]); });
      }

      /* ── 句 4：2 × 4 八台 ── */
      var gridG = E.el('g', {}, s), tiles = [];
      var heads = ['肘上 · 腕①', '肘上 · 腕②', '肘下 · 腕①', '肘下 · 腕②'];
      SOL.forEach(function (q, i) {
        var col = i % 4, row = Math.floor(i / 4), tx = 10 + col * 196, ty = 24 + row * 196;
        var g = E.el('g', {}, gridG);
        K.paint(E.el('rect', { x: tx + 2, y: ty + 2, width: 188, height: 168, rx: 12, 'stroke-width': 2 }, g), E.tint(row ? E.DOWN : E.UP, 6, 'var(--demo-surface)'), E.BORDER);
        var r = window.Robot3D.ur5(g, tx + 98, ty + 124, 0.82, { az: 32, el: 16 });
        r.view.camera({ box: [tx + 4, ty + 4, 184, 164] });
        setQ(r.view, q);
        var arr = frame(g, 3);
        E.txt(g, tx + 96, ty + 162, heads[col], 17, col < 2 ? E.UP : E.DOWN, 'middle', 900);
        tiles.push({ g: g, v: r.view, arr: arr, row: row });
      });
      var rowLab = [E.txt(gridG, 6, 16, '底座朝前伸', 18, E.INK2, 'start', 900), E.txt(gridG, 6, 214, '底座反着伸', 18, E.INK2, 'start', 900)];
      var eq8 = E.tex(gridG, 400, 412, '2\\times 2\\times 2=' + '\\textcolor{' + H.hi + '}{8}', 30, E.INK, 'middle', { w: 300, h: 50 });

      /* ── 句 6：平面三关节链（末端钉住，前面的关节一起挪）、G1 正面 ── */
      var chainG = E.el('g', {}, s), CU = 52, CO = [70, 300], LL = [1.7, 1.5, 1.3], TGC = [2.7, 1.6];
      function CP(x, y) { return [CO[0] + CU * x, CO[1] - CU * y]; }
      for (var gx = 0; gx <= 6; gx++) K.paint(E.el('line', { x1: CP(gx, 0)[0], y1: CP(0, -1)[1], x2: CP(gx, 0)[0], y2: CP(0, 4)[1], 'stroke-width': 1.2 }, chainG), null, E.GRID);
      for (var gy = -1; gy <= 4; gy++) K.paint(E.el('line', { x1: CP(0, 0)[0], y1: CP(0, gy)[1], x2: CP(6, 0)[0], y2: CP(0, gy)[1], 'stroke-width': 1.2 }, chainG), null, E.GRID);
      var links3 = [0, 1, 2].map(function (k) { return K.paint(E.el('path', { fill: 'none', 'stroke-width': 10, 'stroke-linecap': 'round' }, chainG), null, E.tint(E.INK2, 70, 'var(--demo-surface)')); });
      var jd3 = [0, 1, 2].map(function () { return K.paint(E.el('circle', { r: 8, 'stroke-width': 3 }, chainG), E.S1, E.HI); });
      var tg3 = E.target(chainG, CP(TGC[0], TGC[1])[0], CP(TGC[0], TGC[1])[1], 11, E.TGT);
      var ghosts3 = E.el('g', {}, chainG);
      var g1G = E.el('g', {}, s);
      var g1 = window.Robot3D.g1Front(g1G, {});
      g1.place(220, 0.95, 400);
      g1.showDots(true);
      g1.joints.forEach(function (j) { var on = /^left_(shoulder|elbow|wrist)/.test(j.name); j.el.style.display = on ? '' : 'none'; if (on) { j.el.style.fill = 'var(--hi)'; j.el.setAttribute('r', 7); } });
      g1.pose({ arm: [[8, 4], [40, 70]] });
      var g1Lab = E.halo(E.txt(g1G, 330, 150, '7 个关节', 24, E.HI, 'start', 900));

      /* ══ 右栏 ══ */
      var RK = E.rowKit(), Rg = E.el('g', {}, s), X = RX + 10;
      var r1g = E.el('g', {}, Rg);
      var a1 = RK.line(r1g, X, 40, [['真实的机械臂：', E.INK2], ['6 个关节', E.HI, 900]]);
      var a2 = RK.line(r1g, X, 78, [['手的', E.INK2], ['位置', E.TGT, 900], [' + ', E.INK2], ['朝向', E.OMEGA, 900], ['，都要对上', E.INK2]], 21);
      var a3 = RK.line(r1g, X, 134, [['现成的公式：', E.INK2], ['ur_kinematics', E.INK, 900]], 21);
      var a3b = RK.line(r1g, X, 166, [['（UR 机械臂的开源代码）', E.MUT, 700]], 18);
      var a4 = RK.line(r1g, X, 206, [['一步一步解：', E.INK2], ['3 处', E.INK, 900], ['，各有 ', E.INK2], ['2 个答案', E.HI, 900]], 21);
      var a5 = RK.line(r1g, X, 260, [['① 底座：', E.INK, 900], ['朝前伸', E.INK2], [' / ', E.MUT], ['反着伸', E.INK2]], 21);
      var a6 = RK.line(r1g, X, 300, [['② 肘：', E.INK, 900], ['朝上', E.UP, 900], [' / ', E.MUT], ['朝下', E.DOWN, 900]], 21);
      var a7 = RK.line(r1g, X, 340, [['③ 手腕：', E.INK, 900], ['两种翻法', E.INK2]], 21);
      var r3g = E.el('g', {}, Rg);
      var b1 = RK.line(r3g, X, 44, [['最一般的 6 个关节：', E.INK2]], 21);
      var b2 = RK.line(r3g, X, 84, [['已有人证明，最多 ', E.INK2], ['16 组', E.HI, 900]], 23);
      var b3 = RK.line(r3g, X, 144, [['关节比需要的多 → ', E.INK2], ['无穷多组', E.HI, 900]], 21);
      var b4 = RK.line(r3g, X, 182, [['平面 3 个关节，只管手腕的位置', E.INK2]], 20);
      var b5 = RK.line(r3g, X, 216, [['手腕不动，前面的关节还能一起挪', E.INK2]], 20);
      var b6 = RK.line(r3g, X, 256, [['人形机器人：一条胳膊 ', E.INK2], ['7 个关节', E.HI, 900]], 20);
      var b7 = RK.line(r3g, X, 316, [['手腕某些姿势（像第 6 集的万向锁）', E.INK2]], 19);
      var b8 = RK.line(r3g, X, 350, [['→ 也有 ', E.INK2], ['无穷多组', E.HI, 900]], 21);
      var r4g = E.el('g', {}, Rg);
      var c1 = RK.line(r4g, X, 56, [['6 个关节：一般是', E.INK2], ['有限几组', E.INK, 900]], 21);
      var c2 = RK.line(r4g, X, 104, [['这台（UR5）最多 ', E.INK2], ['8 组', E.HI, 900]], 23);
      var c3 = RK.line(r4g, X, 146, [['最一般的最多 ', E.INK2], ['16 组', E.HI, 900]], 23);
      var c4 = RK.line(r4g, X, 214, [['肘上', E.UP, 900], ['、', E.INK2], ['肘下', E.DOWN, 900], ['：求肘那一步', E.INK2]], 21);
      var c5 = RK.line(r4g, X, 252, [['开平方', E.WARN, 900], ['的正负两个', E.INK2]], 21);

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) { RK.layout(); laid = true; }
          var f = E.fade, b = E.bump;

          /* ── 大 UR5：参考姿势 → 底座反着 → 肘下 → 手腕翻过去；句 7 手腕奇异；句 8 依次摆出 8 组 ── */
          var q = SOL[0];
          var kB = f(st, T.back[0], T.back[1] - T.back[0]), kD = f(st, T.down[0], T.down[1] - T.down[0]), kF = f(st, T.flip[0], T.flip[1] - T.flip[0]);
          if (kB > 0) q = lerpQ(SOL[0], SOL[4], kB);
          if (kD > 0) q = lerpQ(SOL[4], SOL[6], kD);
          if (kF > 0) q = lerpQ(SOL[6], SOL[7], kF);
          if (st >= T.out3) q = SOL[0];
          var ws = f(st, T.pose7, 0.8);
          if (st >= T.wristS - 0.5 && st < T.sum) { q = SOL[0].slice(); q[4] = SOL[0][4] * (1 - ws); }
          var kk = -1;
          if (st >= T.s8) { kk = Math.min(7, Math.floor((st - T.s8) / 1.0)); q = SOL[kk]; }
          setQ(BV, q);
          // 大 UR5：句 4 让给八台，句 6 让给三关节链和 G1
          var bigOn = 1 - f(st, T.grid, 0.4);
          if (st >= T.out4) bigOn = f(st, T.out4 + 0.2, 0.4) * (1 - f(st, T.more - 0.4, 0.4));
          if (st >= T.wristS - 0.6) bigOn = f(st, T.wristS - 0.5, 0.4);
          E.op(bigG, bigOn);
          var pts = LINKS.map(function (l) { return BV.at(l); });
          jd.forEach(function (c, k) { c.setAttribute('cx', pts[k][0].toFixed(1)); c.setAttribute('cy', pts[k][1].toFixed(1)); E.op(c, fr(st, T.six) * (1 - f(st, T.ik, 0.3))); });
          toolArrows(BV, bigF, 0.12);
          var fOn = fr(st, T.ori) * bigOn;
          bigF.forEach(function (a) { E.op(a.g, fOn); a.line.style.strokeWidth = 4.5 + 2 * Math.max(b(st, T.front, 0.8), b(st, T.back[1] + 0.2, 0.8), b(st, T.down[1] + 0.2, 0.8), b(st, T.flip[1] + 0.2, 0.8)); });
          // 句 7：手腕两根转轴（wrist_1、wrist_3 的 y 轴）
          var w1a = BV.at('wrist_1_link', [0, -0.16, 0]), w1b = BV.at('wrist_1_link', [0, 0.16, 0]), w3a = BV.at('wrist_3_link', [0, -0.16, 0]), w3b = BV.at('wrist_3_link', [0, 0.16, 0]);
          ax1.setAttribute('d', 'M ' + w1a[0].toFixed(1) + ' ' + w1a[1].toFixed(1) + ' L ' + w1b[0].toFixed(1) + ' ' + w1b[1].toFixed(1));
          ax3.setAttribute('d', 'M ' + w3a[0].toFixed(1) + ' ' + w3a[1].toFixed(1) + ' L ' + w3b[0].toFixed(1) + ' ' + w3b[1].toFixed(1));
          axLab.setAttribute('x', (Math.min(w1a[0], w3a[0], w1b[0], w3b[0]) - 10).toFixed(1)); axLab.setAttribute('y', (Math.max(w1a[1], w3a[1], w1b[1], w3b[1]) + 28).toFixed(1));   // 轴的左下：右边是右栏
          E.op(axesG, fr(st, T.gimbal - 0.6) * (1 - f(st, T.sum, 0.3)));
          solLab.textContent = kk >= 0 ? '第 ' + (kk + 1) + ' / 8 组' : '';
          E.op(solLab, kk >= 0 ? 1 : 0);

          /* ── 句 4：八台 ── */
          var gOn = f(st, T.grid, 0.4) * (1 - f(st, T.out4, 0.4));
          E.op(gridG, gOn);
          tiles.forEach(function (tl, i) {
            // 「两种」：先亮第 1 列（上下两排 = 底座两种），「两种」再亮第 3 列（肘两种），「再两种」亮第 2、4 列（手腕两种）
            var col = i % 4, at = [T.g2[0], T.g2[2], T.g2[1], T.g2[2]][col];
            E.op(tl.g, f(st, Math.min(at, T.eight), 0.3));
            toolArrows(tl.v, tl.arr, 0.12);
            tl.arr.forEach(function (a) { a.line.style.strokeWidth = 3 + 2.5 * b(st, T.same, 1.0); });
          });
          E.op(eq8, fr(st, T.eight));

          /* ── 句 6：三关节链、G1 ── */
          var chOn = f(st, T.planar, 0.4) * (1 - f(st, T.human, 0.4));
          E.op(chainG, chOn);
          if (chOn > 0) {
            var mv = f(st, T.slide, 0.3) * Math.sin((st - T.slide) * 1.6) * 0.32;
            var th1 = 1.15 + mv, P1 = [LL[0] * Math.cos(th1), LL[0] * Math.sin(th1)];
            var sol = E.ik2(TGC[0] - P1[0], TGC[1] - P1[1], true, LL[1], LL[2]) || [0, 0, [0, 0]];
            var P2 = [P1[0] + sol[2][0], P1[1] + sol[2][1]];
            var pp = [[0, 0], P1, P2, TGC];
            links3.forEach(function (ln, k) { var a = CP(pp[k][0], pp[k][1]), b2 = CP(pp[k + 1][0], pp[k + 1][1]); ln.setAttribute('d', 'M ' + a[0].toFixed(1) + ' ' + a[1].toFixed(1) + ' L ' + b2[0].toFixed(1) + ' ' + b2[1].toFixed(1)); });
            jd3.forEach(function (c, k) { var a = CP(pp[k][0], pp[k][1]); c.setAttribute('cx', a[0].toFixed(1)); c.setAttribute('cy', a[1].toFixed(1)); E.op(c, fr(st, T.three3)); });
            E.pop(tg3, CP(TGC[0], TGC[1])[0], CP(TGC[0], TGC[1])[1], 1 + 0 * b(st, T.still, 0.8), 1);
            tg3.setAttribute('transform', 'translate(' + CP(TGC[0], TGC[1])[0] + ' ' + CP(TGC[0], TGC[1])[1] + ') scale(' + (1 + 0.5 * b(st, T.still, 0.9)) + ') translate(' + (-CP(TGC[0], TGC[1])[0]) + ' ' + (-CP(TGC[0], TGC[1])[1]) + ')');
          }
          E.op(g1G, f(st, T.human, 0.4) * (1 - f(st, T.wristS - 0.5, 0.4)));
          E.op(g1Lab, fr(st, T.seven));
          g1.tint('leftArm', E.HI, 0.6 * f(st, T.seven, 0.3));

          /* ── 右栏 ── */
          E.op(r1g, 1 - f(st, T.grid - 0.2, 0.3));
          E.op(a1.g, fr(st, T.real)); E.op(a1.parts[1], fr(st, T.six)); E.op(a2.g, fr(st, T.ori));
          E.op(a3.g, fr(st, T.formula)); E.op(a3.parts[1], fr(st, T.code)); E.op(a3b.g, fr(st, T.code)); E.op(a4.g, fr(st, T.three)); E.op(a4.parts[3], fr(st, T.two));
          E.op(a5.g, fr(st, T.base)); E.op(a5.parts[3], fr(st, T.back[0])); E.op(a6.g, fr(st, T.elbow)); E.op(a6.parts[3], fr(st, T.down[0])); E.op(a7.g, fr(st, T.wrist));
          E.op(r3g, f(st, T.gen - 0.2, 0.3) * (1 - f(st, T.sum - 0.2, 0.3)));
          E.op(b1.g, fr(st, T.gen)); E.op(b2.g, fr(st, T.proved)); E.op(b2.parts[1], fr(st, T.s16));
          E.op(b3.g, fr(st, T.more)); E.op(b3.parts[1], fr(st, T.inf)); E.op(b4.g, fr(st, T.planar)); E.op(b5.g, fr(st, T.still));
          E.op(b6.g, fr(st, T.human)); E.op(b6.parts[1], fr(st, T.seven)); E.op(b7.g, fr(st, T.wristS)); E.op(b8.g, fr(st, T.inf2));
          E.op(r4g, f(st, T.sum, 0.3));
          E.op(c1.g, fr(st, T.sum)); E.op(c1.parts[1], fr(st, T.finite)); E.op(c2.g, fr(st, T.s8)); E.op(c3.g, fr(st, T.s16b - 0.4));
          E.op(c4.g, fr(st, T.elb)); E.op(c5.g, fr(st, T.sqrt));
        }
      };
    }
  };
})();
