// 009 第 2 幕：角速度——绕什么轴、转多快。左边同第 1 幕的网格 + UR5（肘在 (1, 2)，小臂先停在水平），右栏放要点。
// 句 1「角速度」右栏写「角速度 ω」，「弧度每秒」写单位；
// 句 2 一弧度（右栏放大画一张：网格上半径一格只有 55 px，挤在肘边看不清；旁白去掉了「一弧度是这么大的角」，直接从离轴一格的点讲起）：
//    「离轴一格」轴心长出一根半径（尖上一个黄点，标「1 格」），「转过一弧度」半径带着黄点逆时针转过 1 弧度、身后描出弧，
//    「一格长的弧」弧上标「1 格」、鼓一下；右栏上方写「1 弧度：半径 1 格 → 弧长 1 格」；
// 句 3 57° 是算出来的，一步不跳（用户反馈「全篇的计算都展开讲不要跳过」）：「转一整圈」图缩小、挪到右栏下方，半径接着扫一整圈、身后留下虚线圆，
//    「六点二八格」右栏「一整圈：弧长 ≈ 6.28 格」、圆旁写「整圈弧长 2π ≈ 6.28 格」（π 只上屏，前几集没讲过），「三百六十度」「↔ 360°」、圆里写 360°，
//    「三百六十除以六点二八」右栏「1 弧度 = 360° ÷ 6.28」，「一弧度」画出角、写「1 弧度」，「五十七度」「≈ 57°」；
// 句 4 起小臂开始绕肘慢慢逆时针转（一直转到这一幕结束）：「二」右栏「ω = 2 弧度 / 秒」，「每秒转两弧度」旁注一句，
//    「离肘一格的点」亮一个点、右栏「离肘 1 格 → 每秒」，「每秒走」长出速度箭头，「一乘二」写「1 × 2」，「两格」「= 2 格」、箭头尖上标「2」；
//    「离肘两格」同样，「2 × 2」「= 4 格」、标「4」；句 5「角速度乘距离」右栏「速度大小 = ω × 距离」；
// 句 6 右栏换成右手定则的示意：「箭头」一根竖着的转轴箭头，「顺着」「弯过去」绕着它画出一圈带箭头的椭圆（四指），「大拇指」轴上的箭头鼓一下、写「ω」；
// 句 7「逆时针」肘旁的小弯箭头鼓一下，「垂直于屏幕」肘上出「⊙」（箭头朝向你），右栏换成 x、y、z 小坐标架（z 斜着画、朝向你），
//    「朝向你」z 方向长出 ω 箭头，「z」z 鼓一下，「零、零、一」右栏方向 (0, 0, 1) 的三个数依次弹出，「乘上大小二」写「ω = 2 ×」，
//    「零、零、二」= (0, 0, 2) 的三个数依次弹出。
// 时刻取自配音的逐词时间，见 ../ep009.py。
(function () {
  var E = window.EP009, K = E.K;
  var D = Math.PI / 180, W = 2;
  var OX = E.ARM.OX, OY = E.ARM.OY, ARM_S = E.ARM.S, KV = E.ARM.KV, RX = E.ARM.RX;
  var T = {
    omega: 1.72, unit: 2.86,
    rad: { r: 4.6, one: 4.98, turn: [5.82, 7.25], arc: 7.69, card: 7.69, shrink: 9.45, sweep: [9.7, 11.0], len: 11.33, d360: 12.78, div: 13.92, deg: 15.92, deg57: 16.95, out: 18.6 },
    rot: 19.0, two: 19.64, perSec: 20.62, p1: 21.64, v1: 22.92, m1: 23.47, l1: 24.15, p2: 24.83, v2: 26.12, m2: 26.74, l2: 27.43,
    times: 30.52, clear: 32.51,
    dirArrow: 34.6, fingers: [37.8, 39.24], thumb: 39.75, isIt: 41.19,
    ccw: 44.62, perp: 46.02, you: 47.05, z: 48.25, unitZ: [49.14, 49.6, 50.03], big: 50.48, nums: [52.13, 52.63, 53.1]
  };
  var END = 54.17, SWING = 38 * D;                           // 19.0 起小臂匀速往上转 38°

  E.scenes[1] = {
    title: '角速度：绕什么轴、转多快',
    dur: END,
    cues: [
      { at: 0, s: '转得多快：**角速度**，单位是弧度每秒' },
      { at: 4.51, s: '1 弧度：离轴 1 格的点走过 1 格长的弧' },
      { at: 9.41, s: '一整圈：弧长 2π ≈ 6.28 格，对应 360°；1 弧度 = 360° ÷ 6.28 ≈ 57°' },
      { at: 18.65, s: '**角速度** = 2：每秒转 2 弧度；离肘 1 格每秒走 1 × 2 = 2 格，离肘 2 格每秒走 2 × 2 = 4 格' },
      { at: 28.97, s: '速度大小 = **角速度** × 距离' },
      { at: 32.64, s: '**角速度**也是一支箭头：右手四指顺着转，拇指指的就是它' },
      { at: 43.2, s: '逆时针转：箭头朝向你，方向 (0, 0, 1) × 大小 2 = **ω** = (0, 0, 2)' }
    ],
    build: function (K) {
      var s = K.sceneSvg('UR5 小臂绕肘转；离轴一格的点转过一弧度，正好走过一格长的弧；转一整圈弧长约 6.28 格、对应 360 度，360 除以 6.28，一弧度约 57 度；角速度是 2 弧度每秒时，离肘一格的点每秒走 1 × 2 = 2 格，离肘两格的点每秒走 2 × 2 = 4 格，速度大小等于角速度乘距离；角速度也是一支箭头，画在转轴上，右手四指顺着转动方向弯，大拇指指的就是它的方向；小臂在平面里逆时针转，角速度垂直屏幕朝向你，方向 (0, 0, 1) 乘大小 2，写成 (0, 0, 2)');

      var under = E.el('g', {}, s);
      var arm = E.armAt(s, OX, OY, ARM_S, 330);
      var u = arm.aim(1, 2, 0);
      E.plane(under, { ox: OX, oy: OY, u: u, x0: -1, x1: 4, y0: -1, y1: 3 });
      var top = E.el('g', {}, s);
      var V = arm.view, RIGID = ['forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link'];

      var eDot = K.paint(E.el('circle', { r: 7, 'stroke-width': 2.5 }, top), E.HI, E.S1);
      var spin = E.arcArrow(top, 0, 0, 30, E.OMEGA, 3.5, { head: 12 });
      // 「⊙」：角速度箭头朝向你（圆圈里一个点）
      var outOf = E.el('g', {}, top);
      var outRing = K.paint(E.el('circle', { r: 15, 'stroke-width': 3.5 }, outOf), E.S1, E.OMEGA);
      var outDot = K.paint(E.el('circle', { r: 5 }, outOf), E.OMEGA);
      var outLab = E.halo(E.txt(outOf, 0, 0, 'ω', 26, E.OMEGA, 'middle', 900));

      /* ── 一弧度（右栏）：轴心 + 两根半径 + 黄点 + 弧 + 角 ── */
      var radG = E.el('g', {}, s), RC = [RX + 40, 352], RR = 156;
      var wedge = K.paint(E.el('path', { 'stroke-width': 0 }, radG), E.tint(E.HI, 30, 'var(--demo-surface)'));
      var arc = K.paint(E.el('path', { fill: 'none', 'stroke-width': 5, 'stroke-linecap': 'round' }, radG), null, E.HI);
      var r0 = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3 }, radG), null, E.INK2);
      var r1 = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '7 6' }, radG), null, E.HI);
      var rCen = K.paint(E.el('circle', { cx: RC[0], cy: RC[1], r: 6.5, 'stroke-width': 2.5 }, radG), E.HI, E.S1);
      var rDot = K.paint(E.el('circle', { r: 7, 'stroke-width': 2.5 }, radG), E.HI, E.S1);
      var rOne = E.halo(E.txt(radG, RC[0] + RR / 2, RC[1] + 30, '半径 1 格', 21, E.INK, 'middle', 900));
      var aOne = E.halo(E.txt(radG, 0, 0, '弧长 1 格', 21, E.HI, 'start', 900));
      var deg = E.halo(E.txt(radG, 0, 0, '1 弧度', 21, E.HI, 'start', 900));
      // 句 3「转一整圈」：图缩小、挪到右栏下方，半径接着扫一整圈，身后留下虚线圆
      var RC2 = [RX + 110, 322], RR2 = 76;
      var full = K.paint(E.el('path', { fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '6 6' }, radG), null, E.HI);
      var r2 = K.paint(E.el('path', { fill: 'none', 'stroke-width': 2.5 }, radG), null, E.INK2);
      var lab360 = E.halo(E.txt(radG, RC2[0] - 30, RC2[1] + 34, '360°', 21, E.INK, 'middle', 900));
      var labLen = E.halo(E.txt(radG, RC2[0] + RR2 + 18, RC2[1] + 30, '整圈弧长', 20, E.INK2, 'start', 700));
      var labLen2 = E.halo(E.txt(radG, RC2[0] + RR2 + 18, RC2[1] + 58, '2π ≈ 6.28 格', 20, E.HI, 'start', 900));

      /* ── 两个点：离肘 1 格、2 格；速度箭头 + 尖上的数 ── */
      var dots = [0, 1].map(function () { return K.paint(E.el('circle', { r: 6.5, 'stroke-width': 2.5 }, top), E.VEL, E.S1); });
      var vels = [0, 1].map(function () { return E.vec(top, E.VEL, 4.5, { head: 15 }); });
      var vlab = ['2', '4'].map(function (n) { return E.halo(E.txt(top, 0, 0, n, 22, E.VEL, 'middle', 900)); });

      /* ══ 右栏 ══ */
      var R = E.el('g', {}, s);
      var head = E.el('g', {}, R);
      E.txt(head, RX + 10, 50, '角速度 ω', 28, E.OMEGA, 'start', 900);
      var unit = E.txt(head, RX + 10, 86, '单位：弧度 / 秒', 22, E.INK2, 'start', 700);
      var radCard = E.el('g', {}, R);
      E.txt(radCard, RX + 10, 140, '1 弧度：半径 1 格 → 弧长 1 格', 22, E.HI, 'start', 700);
      var radFull = E.txt(radCard, RX + 10, 178, '一整圈：弧长 ≈ 6.28 格', 22, E.INK, 'start', 700);
      var radFull2 = E.txt(radCard, RX + 252, 178, '↔ 360°', 22, E.INK, 'start', 700);
      var radDiv = E.txt(radCard, RX + 10, 214, '1 弧度 = 360° ÷ 6.28', 22, E.INK, 'start', 700);
      var radDeg = E.txt(radCard, RX + 220, 214, '≈ 57°', 22, E.HI, 'start', 900);
      var spd = E.el('g', {}, R);
      var two = E.txt(spd, RX + 10, 140, 'ω = 2 弧度 / 秒', 24, E.OMEGA, 'start', 900);
      var perSec = E.txt(spd, RX + 210, 140, '每秒转 2 弧度', 20, E.INK2, 'start', 700);
      // 离肘 d 格：每秒走 d × 2 = 2d 格（弧长 = 半径 × 弧度）
      var rows = [['离肘 1 格', '1 × 2', '= 2 格'], ['离肘 2 格', '2 × 2', '= 4 格']].map(function (d, i) {
        var g = E.el('g', {}, spd), y = 184 + 40 * i;
        var lab = E.txt(g, RX + 10, y, d[0] + ' → 每秒', 22, E.INK2, 'start', 700);
        var m = E.txt(g, RX + 164, y, d[1], 22, E.VEL, 'start', 900), r = E.txt(g, RX + 222, y, d[2], 22, E.VEL, 'start', 900);
        return { g: g, lab: lab, m: m, r: r };
      });
      // 一行几段字：挂到页面后量宽度，挨着往右排
      function after(a, b2, gap) { b2.setAttribute('x', (+a.getAttribute('x') + a.getComputedTextLength() + gap).toFixed(1)); }
      var times = E.el('g', {}, spd);
      K.paint(E.el('rect', { x: RX, y: 248, width: 300, height: 50, rx: 12, 'stroke-width': 2.5 }, times), E.tint(E.HI, 10, 'var(--demo-surface)'), E.HI);
      E.txt(times, RX + 150, 281, '速度大小 = ω × 距离', 24, E.INK, 'middle', 900);
      // 右手定则示意：竖着的转轴箭头（拇指）+ 绕着它的一圈带箭头的椭圆（四指）
      var rh = E.el('g', {}, R), CX = RX + 205, CY = 270;
      // 按远近遮挡：后半圈（虚线）画在拇指后面，前半圈压在拇指上
      var ringBack = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3.5, 'stroke-dasharray': '6 6' }, rh), null, E.INK2);
      var axisV = E.vec(rh, E.OMEGA, 5, { head: 18 });
      var ringFront = E.el('path', { fill: 'none', 'stroke-width': 4, 'stroke-linecap': 'round' }, rh);
      K.paint(ringFront, null, E.INK);
      var ringHead = K.paint(E.el('path', {}, rh), E.INK);
      var fingLab = E.txt(rh, CX - 84, CY + 12, '四指：顺着转', 22, E.INK, 'end', 900);
      var thumbLab = E.txt(rh, CX - 18, 150, '拇指 → ω 的方向', 22, E.OMEGA, 'end', 900);
      // 小坐标架：x 右、y 上、z 斜着朝向你（往左下画）；ω 沿 z；旁边 ω 的三个数
      var gz = E.el('g', {}, R), GX = RX + 90, GY = 250, GL = 96;   // 坐标架偏左：右上要写 ω = 2 × (0, 0, 1) = (0, 0, 2)
      function P3(v) { return [GX + GL * (v[0] - 0.55 * v[2] * Math.cos(35 * D)), GY - GL * (v[1] - 0.55 * v[2] * Math.sin(35 * D))]; }
      var axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]].map(function (v, i) {
        var a = E.vec(gz, [E.X, E.Y, E.Z][i], 3, { head: 12 }), p = P3(v);
        a.set(GX, GY, p[0], p[1]);
        var lp = P3([v[0] * 1.22, v[1] * 1.22, v[2] * 1.3]);
        E.halo(E.txt(gz, lp[0], lp[1] + 8, ['x', 'y', 'z'][i], 22, [E.X, E.Y, E.Z][i], 'middle', 900));
        return a;
      });
      var wz = E.vec(gz, E.OMEGA, 6, { head: 18 });
      var wzLab = E.halo(E.txt(gz, 0, 0, 'ω', 26, E.OMEGA, 'middle', 900));
      var youLab = E.txt(gz, 0, 0, '朝向你', 20, E.OMEGA, 'start', 900);
      // ω = 2 × (0, 0, 1) = (0, 0, 2)：方向（z）乘大小 2；放在小坐标架右上方，不挨着 x 的字母
      var ucol = E.colvec(gz, RX + 228, 168, [0, 0, 1], { size: 26, colors: [E.Z, E.Z, E.Z] });
      var wcol = E.colvec(gz, RX + 318, 168, [0, 0, 2], { size: 26, colors: [E.OMEGA, E.OMEGA, E.OMEGA] });
      var wEq = E.txt(gz, RX + 196, 177, 'ω = 2 ×', 22, E.OMEGA, 'end', 900);
      var wEq2 = E.txt(gz, RX + 270, 177, '=', 24, E.OMEGA, 'middle', 900);

      function toMath(p) { return [(p[0] - OX) / u, (OY - p[1]) / u]; }

      return {
        el: s,
        draw: function (st, t) {
          var f = E.fade, b = E.bump;
          if (s.isConnected) {                                   // 每帧都量：网页字体晚到也排得对
            rows.forEach(function (r) { after(r.lab, r.m, 8); after(r.m, r.r, 6); });
            after(radFull, radFull2, 10); after(radDiv, radDeg, 10);
          }
          var k = E.clamp((st - T.rot) / (END - T.rot), 0, 1);
          var phi = SWING * (k < 0.03 ? k * k / 0.06 : k - 0.015) / 0.985;
          arm.bend(phi);
          V.clearTint(); V.tint(RIGID, E.BODY, 0.4);
          var e = arm.elbow(), w = arm.wrist(), dir = [w[0] - e[0], w[1] - e[1]], dl = Math.hypot(dir[0], dir[1]);

          eDot.setAttribute('cx', e[0].toFixed(1)); eDot.setAttribute('cy', e[1].toFixed(1));
          spin.g.setAttribute('transform', 'translate(' + e[0].toFixed(1) + ' ' + e[1].toFixed(1) + ')');
          spin.set(115 * D, 205 * D);
          var sb = b(st, T.ccw, 0.9);
          spin.arc.style.strokeWidth = 3.5 + 2 * sb; E.glow(spin.g, E.OMEGA, sb > 0.05);
          // ⊙ 盖在肘上：箭头朝向你
          outOf.setAttribute('transform', 'translate(' + e[0].toFixed(1) + ' ' + e[1].toFixed(1) + ')');
          outLab.setAttribute('x', -6); outLab.setAttribute('y', -42);   // 写在肘的正上方、肘的外壳和小弯箭头再往上：右上是小臂和速度箭头
          var po = f(st, T.perp, 0.35);
          E.op(outOf, po); E.op(eDot, 1 - po);
          outRing.setAttribute('r', (15 + 4 * b(st, T.you, 0.8)).toFixed(2));

          /* ── 一弧度 ── */
          var sh = f(st, T.rad.shrink, 0.6), U = RR + (RR2 - RR) * sh, C0 = [RC[0] + (RC2[0] - RC[0]) * sh, RC[1] + (RC2[1] - RC[1]) * sh];
          var a1 = 1 * f(st, T.rad.turn[0], T.rad.turn[1] - T.rad.turn[0]);
          function at(r, a) { return [C0[0] + U * r * Math.cos(a), C0[1] - U * r * Math.sin(a)]; }
          rCen.setAttribute('cx', C0[0].toFixed(1)); rCen.setAttribute('cy', C0[1].toFixed(1));
          var pA = at(1, 0), pB = at(1, a1);
          r0.setAttribute('d', 'M ' + C0[0].toFixed(1) + ' ' + C0[1].toFixed(1) + ' L ' + pA[0].toFixed(1) + ' ' + pA[1].toFixed(1));
          r1.setAttribute('d', 'M ' + C0[0].toFixed(1) + ' ' + C0[1].toFixed(1) + ' L ' + pB[0].toFixed(1) + ' ' + pB[1].toFixed(1));
          arc.setAttribute('d', a1 > 0.005 ? 'M ' + pA[0].toFixed(1) + ' ' + pA[1].toFixed(1) + ' A ' + U + ' ' + U + ' 0 0 0 ' + pB[0].toFixed(1) + ' ' + pB[1].toFixed(1) : '');
          var WR = 0.3 + 0.15 * sh, w0 = at(WR, 0), w1 = at(WR, a1);
          wedge.setAttribute('d', a1 > 0.005 ? 'M ' + C0[0].toFixed(1) + ' ' + C0[1].toFixed(1) + ' L ' + w0[0].toFixed(1) + ' ' + w0[1].toFixed(1) +
            ' A ' + (WR * U).toFixed(1) + ' ' + (WR * U).toFixed(1) + ' 0 0 0 ' + w1[0].toFixed(1) + ' ' + w1[1].toFixed(1) + ' Z' : '');
          rDot.setAttribute('cx', pB[0].toFixed(1)); rDot.setAttribute('cy', pB[1].toFixed(1));
          var ao = at(1.08, 0.42); aOne.setAttribute('x', ao[0].toFixed(1)); aOne.setAttribute('y', (ao[1] + 8).toFixed(1));
          // 缩小以后「1 弧度」写在圆外的右上方（圆里太挤）
          var dg = at(1.12, 0.62); deg.setAttribute('x', dg[0].toFixed(1)); deg.setAttribute('y', (dg[1] + 4).toFixed(1));
          deg.textContent = st >= T.rad.deg57 ? '1 弧度 ≈ 57°' : '1 弧度';
          // 整圈：半径 r2 从 1 弧度接着扫到 2π，虚线圆跟着画出来
          var sw = f(st, T.rad.sweep[0], T.rad.sweep[1] - T.rad.sweep[0]), aS = 1 + (2 * Math.PI - 1) * sw;
          var pS = at(1, aS), fullPts = [];
          for (var kk = 0; kk <= 64; kk++) fullPts.push(at(1, 1 + (aS - 1) * kk / 64));
          full.setAttribute('d', sw > 0.005 ? E.poly(fullPts) : '');
          r2.setAttribute('d', 'M ' + C0[0].toFixed(1) + ' ' + C0[1].toFixed(1) + ' L ' + pS[0].toFixed(1) + ' ' + pS[1].toFixed(1));
          E.op(full, sw > 0.005 ? 1 : 0); E.op(r2, sw > 0.005 ? 1 - f(st, T.rad.sweep[1] + 0.3, 0.4) : 0);
          E.op(lab360, f(st, T.rad.d360, 0.3)); E.op(labLen, f(st, T.rad.len, 0.3)); E.op(labLen2, f(st, T.rad.len, 0.3));
          var radOn = 1 - f(st, T.rad.out, 0.4);
          E.op(radG, radOn * f(st, T.rad.r, 0.3));
          E.op(r1, f(st, T.rad.turn[0], 0.2)); E.op(rOne, f(st, T.rad.one, 0.3) * (1 - sh));   // 缩小以后「半径 1 格」「弧长 1 格」收掉：右栏第一行写着
          E.op(arc, a1 > 0 ? 1 : 0); arc.style.strokeWidth = 5 + 3 * b(st, T.rad.arc, 0.8); E.glow(arc, E.HI, b(st, T.rad.arc, 0.8) > 0.05);
          E.op(aOne, f(st, T.rad.arc, 0.3) * (1 - sh)); E.op(wedge, f(st, T.rad.deg, 0.4)); E.op(deg, f(st, T.rad.deg, 0.4));

          /* ── 两个点的速度 ── */
          [1, 2].forEach(function (d, i) {
            var p = [e[0] + dir[0] / dl * d * u, e[1] + dir[1] / dl * d * u];
            var rm = [(p[0] - e[0]) / u, (e[1] - p[1]) / u], v = E.rot90(rm);
            var tip = [p[0] + KV * u * W * v[0], p[1] - KV * u * W * v[1]];
            var on = f(st, i ? T.v2 : T.v1, 0.5);
            vels[i].set(p[0], p[1], p[0] + (tip[0] - p[0]) * on, p[1] + (tip[1] - p[1]) * on);
            E.op(vels[i].g, on > 0 ? 1 : 0);
            dots[i].setAttribute('cx', p[0].toFixed(1)); dots[i].setAttribute('cy', p[1].toFixed(1));
            var hb = b(st, i ? T.p2 : T.p1, 0.7);
            dots[i].setAttribute('r', (6.5 + 3 * hb).toFixed(2));
            E.op(dots[i], f(st, i ? T.p2 : T.p1, 0.3));
            // 数写在箭头旁边（往小臂外端那一侧让开 24 px、比尖低一点），不压箭头的三角
            var ux = dir[0] / dl, uy = dir[1] / dl, wl = Math.hypot(tip[0] - p[0], tip[1] - p[1]) || 1;
            vlab[i].setAttribute('x', (tip[0] + ux * 24 - (tip[0] - p[0]) / wl * 12).toFixed(1)); vlab[i].setAttribute('y', (tip[1] + uy * 24 - (tip[1] - p[1]) / wl * 12 + 8).toFixed(1));
            E.op(vlab[i], f(st, i ? T.l2 : T.l1, 0.3));
          });

          /* ── 右栏 ── */
          E.op(head, f(st, T.omega, 0.35)); E.op(unit, f(st, T.unit, 0.3));
          E.op(radCard, f(st, T.rad.card, 0.3) * (1 - f(st, T.rad.out, 0.3)));
          E.op(radFull, f(st, T.rad.len, 0.3)); E.op(radFull2, f(st, T.rad.d360, 0.3)); E.op(radDiv, f(st, T.rad.div, 0.3)); E.op(radDeg, f(st, T.rad.deg57, 0.3));
          var spdOn = 1 - f(st, T.clear, 0.4);
          E.op(spd, spdOn);
          E.op(two, f(st, T.two, 0.3)); E.op(perSec, f(st, T.perSec, 0.3));
          rows.forEach(function (r, i) { E.op(r.g, f(st, i ? T.p2 : T.p1, 0.3)); E.op(r.m, f(st, i ? T.m2 : T.m1, 0.3)); E.op(r.r, f(st, i ? T.l2 : T.l1, 0.3)); });
          E.pop(times, RX + 150, 273, f(st, T.times, 0.35), 0.9);

          // 右手定则：竖轴箭头（拇指）+ 椭圆（四指，从上往下看逆时针：前面那半往右走）
          var rhOn = f(st, T.clear + 0.3, 0.4) * (1 - f(st, T.ccw, 0.4));
          E.op(rh, rhOn);
          var ga = f(st, T.dirArrow, 0.5);
          axisV.set(CX, CY + 110, CX, CY + 110 - 260 * ga);
          var tb = b(st, T.thumb, 0.9);
          axisV.line.style.strokeWidth = 5 + 3 * tb; E.glow(axisV.g, E.OMEGA, tb > 0.05);
          var fp = f(st, T.fingers[0], T.fingers[1] - T.fingers[0]), RXe = 78, RYe = 22;
          // 后半圈（虚线，从右到左）一次画出；前半圈（实线）从左往右按 fp 画，末端三角朝右
          ringBack.setAttribute('d', 'M ' + (CX + RXe) + ' ' + CY + ' A ' + RXe + ' ' + RYe + ' 0 0 0 ' + (CX - RXe) + ' ' + CY);
          E.op(ringBack, f(st, T.fingers[0], 0.3));
          var th = Math.PI * (1 - fp), ex = CX + RXe * Math.cos(th), ey = CY + RYe * Math.sin(th);
          ringFront.setAttribute('d', fp > 0.01 ? 'M ' + (CX - RXe) + ' ' + CY + ' A ' + RXe + ' ' + RYe + ' 0 0 0 ' + ex.toFixed(1) + ' ' + ey.toFixed(1) : '');
          var tx = Math.sin(th), ty = -Math.cos(th) * RYe / RXe, tl = Math.hypot(tx, ty) || 1;   // 前半圈往右走的切向
          tx /= tl; ty /= tl;
          ringHead.setAttribute('d', fp > 0.15 ? 'M ' + (ex + tx * 14).toFixed(1) + ' ' + (ey + ty * 14).toFixed(1) + ' L ' + (ex - ty * 8).toFixed(1) + ' ' + (ey + tx * 8).toFixed(1) + ' L ' + (ex + ty * 8).toFixed(1) + ' ' + (ey - tx * 8).toFixed(1) + ' Z' : '');
          E.op(fingLab, f(st, T.fingers[0], 0.3)); E.op(thumbLab, f(st, T.isIt, 0.3));

          // 小坐标架 + ω = (0, 0, 2)
          E.op(gz, f(st, T.ccw + 0.2, 0.4));
          var wp = f(st, T.you, 0.5), tip3 = P3([0, 0, 1.05 * wp]);
          wz.set(GX, GY, tip3[0], tip3[1]); E.op(wz.g, wp > 0 ? 1 : 0);
          var wm = P3([0, 0, 0.6 * wp]); wzLab.setAttribute('x', (wm[0] - 16).toFixed(1)); wzLab.setAttribute('y', (wm[1] - 14).toFixed(1));   // 写在 ω 箭头中段的左上方，不和 z 的字母挤在尖上
          E.op(wzLab, wp);
          youLab.setAttribute('x', (tip3[0] + 14).toFixed(1)); youLab.setAttribute('y', (tip3[1] + 26).toFixed(1));
          E.op(youLab, f(st, T.you + 0.2, 0.3));
          var zb = b(st, T.z, 0.8);
          axes[2].line.style.strokeWidth = 3 + 3 * zb; E.glow(axes[2].g, E.Z, zb > 0.05);
          E.op(ucol.g, f(st, T.unitZ[0] - 0.3, 0.3));
          ucol.cells.forEach(function (c, i) { var q = ucol.pos(i), pc = f(st, T.unitZ[i], 0.3); E.op(c, pc); E.scaleAt(c, q[0], q[1], 1 + 0.5 * (1 - pc)); });
          E.op(wEq, f(st, T.big, 0.3)); E.op(wEq2, f(st, T.nums[0] - 0.3, 0.3));
          wcol.cells.forEach(function (c, i) { var q = wcol.pos(i), pc = f(st, T.nums[i], 0.3); E.op(c, pc); E.scaleAt(c, q[0], q[1], 1 + 0.5 * (1 - pc)); });
          E.op(wcol.g, f(st, T.nums[0] - 0.3, 0.3));
        }
      };
    }
  };
})();
