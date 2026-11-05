// 011 第 4 幕：顺序——为什么先转外面的关节（《Modern Robotics》4.1.3 节式 (4.16) 后那段：空间形式里 M 先被最外面的关节变换，
// 远端关节的转动不影响近端关节在固定坐标系里的螺旋轴）。左边同前几幕的网格 + UR5，右栏上方一行公式、下面几行算式。
// 句 1「反过来试试」右栏出顺序反过来的公式（橙框）；「先转肩」整条胳膊绕原点转 90°（画满的蓝弧）。
// 句 2 手 (3, 2) 转 90°：「第一列拿三份」原点往上红 (0, 3)，「第二列拿两份」往左绿 (−2, 0)，「负二、三」出点；「肘也被带到了」肘亮。
// 句 3「再转肘」「按零位写的」(1, 2) 出琥珀虚线圈「S₂ 的轴」；「可肘已经不在那里了」真正的肘 (−2, 1) 标橙。
// 句 4「先减去一、二」从 (1, 2) 到手画紫虚线箭头 (−3, 1)。句 5「转九十度」从 (1, 2) 往下画红 (0, −3)、往左画绿 (−1, 0)。
// 句 6「再加回一、二」(0, −1) 出点；「手跑到了」橙色的小臂虚影绕空的 (1, 2) 转过去，「离肘远了」肘到虚影之间橙虚线、打 ×，写「扯断」。
// 句 7「肘的轴也跟着动了」(1, 2) 沿虚线弧挪到 (−2, 1)；「反过来，先转肘」胳膊回零位、先转肘，「肩的轴不会动」肩亮。
// 句 8「一个一个往里转」再转肩，手到 (−4, 1)、打勾。句 9 公式换回正的顺序：「从右往左」箭头，「最外面的关节，先转」「最里面的关节，最后转」写在下面；
//    「第四集」写 S @ R @ p。句 10「几个关节可以一起转」胳膊回零位、两个关节同时转，落在同一个地方。
// 时刻取自配音的逐词时间，见 ../ep011.py。
(function () {
  var E = window.EP011, K = E.K, H = E.HEX;
  var D = Math.PI / 180, A = E.ARM, RX = A.RX, X = RX + 10, R0 = 90 * D;
  var T = {
    why: 0.7, rev: 3.43, shFirst: 4.56, turnS: [7.23, 8.4],
    hp: 9.28, h32: 10.01, t1: 11.85, t1r: 13.2, t2: 14.24, t2r: 15.36, sum: 16.61, sumR: 17.47, elb: 18.91, elbR: 19.92,
    thenE: 21.99, s2: 23.06, zero: 24.23, around: 25.18, p12: 25.93, but: 27.25, notThere: 27.86,
    rotH: 29.95, sub: 32.13, sub12: 32.81, d1: 33.65, d1r: 34.81, d2: 35.44, d2r: 36.53,
    rot: 37.98, a1: 39.75, a1r: 41.26, a2: 42.3, a2r: 43.33, aSum: 44.65, aSumR: 45.48,
    back: 47.74, b1: 49.13, b1r: 50.33, b2: 50.8, b2r: 51.91, run: [52.89, 54.1], far: 54.75, torn: 56.06,
    prob: 58.02, moved: [59.9, 61.0], reverse: 61.89, reset: [61.89, 62.6], elbFirst: [62.81, 63.6], shStay: 63.46, origin: 65.1,
    so: 66.82, outer: 67.83, inward: [70.15, 71.2], inner: 72.16, zeroPl: 73.84, ok: 76.49,
    formula: 78.28, rl: [78.73, 79.5], first: 82.11, firstT: 83.24, last: 85.51, lastT: 86.52, ep4: 87.71, near: 90.07, nearT: 91.36,
    real: 93.15, reset2: [93.97, 94.6], together: [94.93, 96.7], pose: 96.76, angles: 98.56, oneByOne: 99.5, right: 101.94
  };
  var END = 103.48;

  E.scenes[3] = {
    title: '顺序：从外往里转',
    dur: END,
    cues: [
      { at: 0, s: '为什么先转肘、再转肩？反过来试试：先转肩，整条胳膊绕原点转 90°' },
      { at: 9.18, s: '手 (3, 2) 转 90°：3 × (0, 1) + 2 × (−1, 0) = (−2, 3)；肘被带到 (−2, 1)' },
      { at: 21.89, s: '再转肘：**螺旋轴**按**零位**写，绕的还是 (1, 2)，可肘已经不在那里了' },
      { at: 29.85, s: '手绕 (1, 2) 转 90°：先减去 (1, 2)：−2 − 1 = −3，3 − 2 = 1' },
      { at: 37.88, s: '(−3, 1) 转 90°：−3 × (0, 1) + 1 × (−1, 0) = (−1, −3)' },
      { at: 47.64, s: '加回 (1, 2)：−1 + 1 = 0，−3 + 2 = −1。手跑到 (0, −1)，小臂被**扯断**了' },
      { at: 57.92, s: '问题：肩一转，肘的轴也跟着动了；先转肘，肩的轴不动，还在原点' },
      { at: 66.72, s: '从最外面的关节开始，一个一个往里转：**螺旋轴**都按**零位**写，就对了' },
      { at: 78.18, s: '公式从右往左读：最靠近 M 的、最外面的关节先转（第 4 集：离右边最近的先做）' },
      { at: 93.05, s: '真机上关节可以一起转：最后的**位姿**只看各个关节的**角度**；一个一个转，是为了算对' }
    ],
    build: function (K) {
      var s = K.sceneSvg('顺序：反过来先转肩，手 (3, 2) 转到 (−2, 3)，肘被带到 (−2, 1)；再转肘时螺旋轴按零位写，绕的还是 (1, 2)：(−2, 3) − (1, 2) = (−3, 1)，转九十度得 (−1, −3)，加回 (1, 2) 得 (0, −1)，手离肘远了，小臂被扯断。问题在于肩一转，肘的轴也跟着动了；先转肘，肩的轴不动。所以从最外面的关节开始一个一个往里转，螺旋轴都按零位写就对了。公式从右往左读：最靠近 M 的最外面的关节先转，最左边的最里面的关节最后转（第四集：连着相乘，离右边最近的先做）。真机上关节可以一起转，最后的位姿只看各个关节的角度');

      var under = E.el('g', {}, s), armG = E.el('g', {}, s);
      var arm = E.arm2(armG, 320), u = arm.u, V = arm.view;
      var P = E.plane(under, { ox: A.OX, oy: A.OY, u: u, x0: A.X0, x1: A.X1, y0: A.Y0, y1: A.Y1 }).P;
      var UPPER = ['upper_arm_link'], RIGID = ['forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link'];
      var top = E.el('g', {}, s), O = P(0, 0);
      function seg(vec, a, b, p) { var A0 = P(a[0], a[1]), B0 = P(a[0] + (b[0] - a[0]) * p, a[1] + (b[1] - a[1]) * p); vec.set(A0[0], A0[1], B0[0], B0[1]); E.op(vec.g, p > 0 ? 1 : 0); }
      function arcPts(c, p0, a0, a1) { var pts = []; for (var k = 0; k <= 40; k++) { var q = E.rotP(p0, a0 + (a1 - a0) * k / 40, c); pts.push(P(q[0], q[1])); } return pts; }
      var sRing = K.paint(E.el('circle', { cx: O[0], cy: O[1], r: 12, 'stroke-width': 3.5 }, top), E.S1, E.HI);
      K.paint(E.el('circle', { cx: O[0], cy: O[1], r: 4 }, top), E.HI);
      var eRing = K.paint(E.el('circle', { r: 12, 'stroke-width': 3.5 }, top), E.S1, E.HI), eDot = K.paint(E.el('circle', { r: 4 }, top), E.HI);
      var shArc = E.arcArrow(top, O[0], O[1], 0.75 * u, E.OMEGA, 3.5, { head: 12 });
      var shStay = E.halo(E.txt(top, O[0] + 18, O[1] + 34, '肩的轴：不动', 20, E.HI, 'start', 900));
      // 句 2：手 (3, 2) 转 90°
      var gA = E.vec(top, E.X, 4.5, { head: 13 }), gB = E.vec(top, E.Y, 4.5, { head: 13 });
      var d23 = K.paint(E.el('circle', { cx: P(-2, 3)[0], cy: P(-2, 3)[1], r: 7, 'stroke-width': 3 }, top), E.VEL, E.S1);
      var l23 = E.halo(E.txt(top, P(-2, 3)[0] - 14, P(-2, 3)[1] - 12, '(−2, 3)', 20, E.VEL, 'end', 900));
      var lEl = E.halo(E.txt(top, P(-2, 1)[0] - 16, P(-2, 1)[1] + 30, '肘 (−2, 1)', 20, E.HI, 'middle', 900));
      // 句 3：零位写的肘的轴还在 (1, 2)
      var Q = P(1, 2);
      var s2Ring = K.paint(E.el('circle', { cx: Q[0], cy: Q[1], r: 13, 'stroke-width': 3, 'stroke-dasharray': '5 4', fill: 'none' }, top), null, E.WARN);
      var s2Dot = K.paint(E.el('circle', { cx: Q[0], cy: Q[1], r: 4 }, top), E.WARN);
      var s2Lab = E.halo(E.txt(top, Q[0], Q[1] - 24, 'S₂ 的轴', 20, E.WARN, 'middle', 900));
      // 句 4–6：绕 (1, 2) 转
      var rel = E.vec(top, E.VEL, 3.5, { head: 12, dash: '6 5' });
      var relL = E.halo(E.txt(top, P(-0.5, 2.5)[0], P(-0.5, 2.5)[1] + 30, '(−3, 1)', 20, E.VEL, 'middle', 900));   // 箭头下方：上方是手和 (−2, 3)（线压字检查）
      var wA = E.vec(top, E.X, 4.5, { head: 13 }), wB = E.vec(top, E.Y, 4.5, { head: 13 });
      var wAL = E.halo(E.txt(top, P(1, 0.5)[0] + 12, P(1, 0.5)[1] + 6, '−3 份', 18, E.X, 'start', 900));
      var d01 = K.paint(E.el('circle', { cx: P(0, -1)[0], cy: P(0, -1)[1], r: 7, 'stroke-width': 3 }, top), E.ERR, E.S1);
      var l01 = E.halo(E.txt(top, P(0, -1)[0] + 14, P(0, -1)[1] + 30, '(0, −1)', 20, E.ERR, 'start', 900));
      var ghostG = E.el('g', {}, top);
      var ghostBar = K.paint(E.el('path', { fill: 'none', 'stroke-width': 0.34 * u, 'stroke-linecap': 'round' }, ghostG), null, E.ERR);
      ghostBar.style.opacity = 0.45;
      var ghostH = E.hand2(ghostG, P, 0.8, { dash: '5 4', w: 3.5, ring: E.ERR });
      var gap = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '6 5' }, top), null, E.ERR);
      var gapX = E.cross(top, P(-1, 0)[0], P(-1, 0)[1], 9, E.ERR, 4);
      var tornL = E.halo(E.txt(top, P(-1, 0)[0] - 14, P(-1, 0)[1] + 34, '扯断了', 22, E.ERR, 'end', 900));
      // 句 7：肘的轴跟着动
      var eTrack = K.paint(E.el('path', { d: E.poly(arcPts([0, 0], [1, 2], 0, R0)), fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '7 6' }, top), null, E.WARN);
      // 结果
      var p41 = P(-4, 1), ok41 = E.check(top, p41[0] - 10, p41[1] - 46, 22, E.GOOD, 4);
      var final = E.hand2(top, P, 0.9, { dash: '5 4', w: 3.5, fill: 'none' }); final.set(-4, 1, Math.PI);
      var hand = E.hand2(top, P, 0.9);

      /* ══ 右栏 ══ */
      var RK = E.rowKit(), FY = 36;
      function factor(g, x, w, tex, lab, col, fy) {
        var h = E.el('g', {}, g);
        var box = K.paint(E.el('rect', { x: x - 6, y: fy - 27, width: w, height: 38, rx: 8, 'stroke-width': 2 }, h), E.tint(col, 14, 'var(--demo-surface)'), col);
        E.tex(h, x, fy, tex, 21, E.INK, 'start', { w: w - 8, h: 40 });
        var l = E.txt(h, x + w / 2 - 6, fy + 34, lab, 18, col, 'middle', 900);
        return { g: h, box: box, lab: l };
      }
      // 顺序反了的公式（橙）
      var wG = E.el('g', {}, s);
      E.tex(wG, X, FY, 'T\\overset{?}{=}', 24, E.ERR, 'start', { w: 44, h: 44 });
      var W1 = factor(wG, X + 52, 124, '\\exp([\\mathcal S_2]\\,\\theta_2)', '后转肘', E.ERR, FY);
      var W2 = factor(wG, X + 184, 124, '\\exp([\\mathcal S_1]\\,\\theta_1)', '先转肩', E.ERR, FY);
      var W3 = factor(wG, X + 316, 34, 'M', '零位', E.GOOD, FY);
      // 正的顺序
      var cG = E.el('g', {}, s);
      E.tex(cG, X, FY, 'T=', 24, E.INK, 'start', { w: 40, h: 40 });
      var C1 = factor(cG, X + 50, 124, '\\exp([\\mathcal S_1]\\,\\theta_1)', '肩：最后转', E.WARN, FY);
      var C2 = factor(cG, X + 182, 124, '\\exp([\\mathcal S_2]\\,\\theta_2)', '肘：先转', E.WARN, FY);
      var C3 = factor(cG, X + 314, 34, 'M', '零位', E.GOOD, FY);
      var rlV = E.vec(cG, E.INK2, 3, { head: 11 });
      var SY = 104;
      function rows(lines) { var g = E.el('g', {}, s); return { g: g, r: lines.map(function (l) { return RK.line(g, X, SY + l[0], l[1], l[2] || 19); }) }; }
      var g0 = rows([[0, [['为什么先转肘、再转肩？', E.INK, 900]], 21], [36, [['反过来试试：', E.INK2], ['先转肩', E.ERR, 900]], 20],
        [68, [['整条胳膊绕原点转 90°', E.INK2]], 20]]);
      var g1 = rows([[0, [['手 ', E.INK2], ['(3, 2)', E.VEL, 900], [' 转 90°：', E.INK2]]],
        [30, [['3 × ', E.INK2], ['(0, 1)', E.X, 900], [' = ', E.INK2], ['(0, 3)', E.X, 900]]],
        [58, [['2 × ', E.INK2], ['(−1, 0)', E.Y, 900], [' = ', E.INK2], ['(−2, 0)', E.Y, 900]]],
        [86, [['加起来：', E.INK2], ['(−2, 3)', E.VEL, 900]]], [120, [['肘也被带到了 ', E.INK2], ['(−2, 1)', E.HI, 900]]]]);
      var g2 = rows([[0, [['再转肘：', E.INK2], ['螺旋轴按零位写', E.WARN, 900]]], [32, [['绕的还是 ', E.INK2], ['(1, 2)', E.WARN, 900]]],
        [64, [['可肘已经在 ', E.INK2], ['(−2, 1)', E.ERR, 900], [' 了', E.INK2]]]]);
      var g3 = rows([[0, [['手绕 ', E.INK2], ['(1, 2)', E.WARN, 900], [' 转 90°：', E.INK2]]],
        [30, [['先减去 (1, 2)：−2 − 1 = ', E.INK2], ['−3', E.VEL, 900], ['，3 − 2 = ', E.INK2], ['1', E.VEL, 900]]],
        [64, [['(−3, 1)', E.VEL, 900], [' 转 90°：', E.INK2]]],
        [92, [['−3 × ', E.INK2], ['(0, 1)', E.X, 900], [' = ', E.INK2], ['(0, −3)', E.X, 900]]],
        [120, [['1 × ', E.INK2], ['(−1, 0)', E.Y, 900], [' = ', E.INK2], ['(−1, 0)', E.Y, 900]]],
        [148, [['加起来：', E.INK2], ['(−1, −3)', E.VEL, 900]]],
        [182, [['加回 (1, 2)：−1 + 1 = ', E.INK2], ['0', E.ERR, 900], ['，−3 + 2 = ', E.INK2], ['−1', E.ERR, 900]]],
        [216, [['手跑到 ', E.INK2], ['(0, −1)', E.ERR, 900], ['：小臂被扯断了', E.ERR, 900]]]]);
      var g4 = rows([[0, [['问题：肩一转，', E.INK2], ['肘的轴也跟着动了', E.ERR, 900]]],
        [34, [['反过来，先转肘：', E.INK2], ['肩的轴不动', E.GOOD, 900]]], [66, [['还在原点', E.INK2]]]]);
      var g5 = rows([[0, [['从最外面的关节开始', E.INK, 900]]], [32, [['一个一个往里转', E.INK, 900]]],
        [68, [['里面的关节轴：还在零位的地方', E.INK2]]], [100, [['螺旋轴按零位写，', E.INK2], ['就对了 ✓', E.GOOD, 900]]]]);
      // 整组往下挪 16：「从右往左」箭头在 FY + 50，原先压住第一排字（用户反馈）
      var g6 = rows([[16, [['从右往左读：最靠近 M 的', E.INK2]]], [48, [['最外面的关节，', E.INK, 900], ['先转', E.GOOD, 900]]],
        [82, [['最左边的，', E.INK2], ['最里面的关节，', E.INK, 900], ['最后转', E.WARN, 900]]],
        [126, [['第 4 集：', E.INK2]]], [160, [['连着相乘，离右边最近的，先做', E.INK2]]]]);
      var ep4 = E.el('g', {}, g6.g), ep4T = E.txt(ep4, X + 76, SY + 126, 'S @ R @ p', 20, E.HI, 'start', 700);
      ep4T.style.fontFamily = E.MONO;
      var g7 = rows([[0, [['真机上：', E.INK2], ['几个关节可以一起转', E.INK, 900]]], [34, [['最后的位姿：', E.INK2], ['只看各个关节的角度', E.OMEGA, 900]]],
        [68, [['一个一个转：', E.INK2], ['为了算对', E.GOOD, 900]]]]);

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) { RK.layout(); laid = true; }
          var f = E.fade, b = E.bump;

          /* ── 胳膊：分段 ── */
          var t1 = R0 * (f(st, T.turnS[0], T.turnS[1] - T.turnS[0]) - f(st, T.reset[0], T.reset[1] - T.reset[0]) + f(st, T.inward[0], T.inward[1] - T.inward[0])
            - f(st, T.reset2[0], T.reset2[1] - T.reset2[0]) + f(st, T.together[0], T.together[1] - T.together[0]));
          var t2 = R0 * (f(st, T.elbFirst[0], T.elbFirst[1] - T.elbFirst[0]) - f(st, T.reset2[0], T.reset2[1] - T.reset2[0]) + f(st, T.together[0], T.together[1] - T.together[0]));
          arm.pose(t1, t2);
          V.clearTint(); V.tint(UPPER, E.BODY, 0.3); V.tint(RIGID, E.BODY, 0.45);
          var fk = E.fk2(t1, t2), ec = P(fk.elbow[0], fk.elbow[1]);
          hand.set(fk.hand[0], fk.hand[1], fk.ang);
          eRing.setAttribute('cx', ec[0].toFixed(1)); eRing.setAttribute('cy', ec[1].toFixed(1)); eDot.setAttribute('cx', ec[0].toFixed(1)); eDot.setAttribute('cy', ec[1].toFixed(1));
          var ge = Math.max(b(st, T.elb, 0.9), b(st, T.notThere, 0.9)), gs = Math.max(b(st, T.shStay, 1.0), b(st, T.origin, 0.8));
          eRing.setAttribute('r', (12 + 5 * ge).toFixed(2)); E.glow(eRing, E.HI, ge > 0.05);
          sRing.setAttribute('r', (12 + 5 * gs).toFixed(2)); E.glow(sRing, E.HI, gs > 0.05);
          var on1 = 1 - f(st, T.reset[0], 0.3);
          shArc.set(0, Math.max(1e-3, R0 * f(st, T.turnS[0], T.turnS[1] - T.turnS[0]))); E.op(shArc.g, f(st, T.turnS[0], 0.2) * (1 - f(st, T.hp, 0.4)));
          E.op(shStay, fr(st, T.shStay) * (1 - f(st, T.so, 0.4)));

          /* ── 句 2 ── */
          var on2 = 1 - f(st, T.thenE, 0.4);
          seg(gA, [0, 0], [0, 3], f(st, T.t1, 0.6)); E.op(gA.g, (f(st, T.t1, 0.1) > 0 ? 1 : 0) * on2);
          seg(gB, [0, 3], [-2, 3], f(st, T.t2, 0.6)); E.op(gB.g, (f(st, T.t2, 0.1) > 0 ? 1 : 0) * on2);
          E.op(d23, fr(st, T.sumR) * (1 - f(st, T.run[0], 0.4))); E.op(l23, fr(st, T.sumR) * (1 - f(st, T.rotH, 0.4)));
          E.op(lEl, fr(st, T.elbR) * (1 - f(st, T.rotH, 0.3)));
          lEl.style.fill = st >= T.notThere ? E.ERR : E.HI;

          /* ── 句 3：S₂ 的轴还在 (1, 2)；句 7 跟着挪到 (−2, 1) ── */
          var mv = f(st, T.moved[0], T.moved[1] - T.moved[0]), sq = P.apply(null, E.rotP([1, 2], R0 * mv));
          s2Ring.setAttribute('cx', sq[0].toFixed(1)); s2Ring.setAttribute('cy', sq[1].toFixed(1)); s2Dot.setAttribute('cx', sq[0].toFixed(1)); s2Dot.setAttribute('cy', sq[1].toFixed(1));
          var onS2 = fr(st, T.zero) * (1 - f(st, T.reverse, 0.4));
          E.op(s2Ring, onS2); E.op(s2Dot, onS2); E.op(s2Lab, fr(st, T.p12) * (1 - f(st, T.prob, 0.3)));
          E.drawOn(eTrack, mv); E.op(eTrack, (mv > 0 ? 0.9 : 0) * (1 - f(st, T.reverse, 0.4)));

          /* ── 句 4–6：绕 (1, 2) 转 ── */
          var on4 = 1 - f(st, T.prob, 0.4);
          var rp = f(st, T.sub12, 0.6); rel.set(Q[0], Q[1], Q[0] + (P(-2, 3)[0] - Q[0]) * rp, Q[1] + (P(-2, 3)[1] - Q[1]) * rp);
          E.op(rel.g, (rp > 0 ? 1 : 0) * (1 - f(st, T.rot, 0.4))); E.op(relL, fr(st, T.d2r) * (1 - f(st, T.rot, 0.4)));
          seg(wA, [1, 2], [1, -1], f(st, T.a1, 0.6)); E.op(wA.g, (f(st, T.a1, 0.1) > 0 ? 1 : 0) * on4);
          seg(wB, [1, -1], [0, -1], f(st, T.a2, 0.5)); E.op(wB.g, (f(st, T.a2, 0.1) > 0 ? 1 : 0) * on4);
          E.op(wAL, fr(st, T.a1r) * on4);
          E.op(d01, fr(st, T.b2r) * on4); E.op(l01, fr(st, T.run[1]) * on4);   // 虚影转完再出标签：转的时候从标签上扫过
          // 橙色的小臂虚影：肩转完以后的小臂 (−2, 1) → (−2, 3)，绕空的 (1, 2) 转
          var ga = R0 * f(st, T.run[0], T.run[1] - T.run[0]);
          var g0p = E.rotP([-2, 1], ga, [1, 2]), g1p = E.rotP([-2, 3], ga, [1, 2]);
          ghostBar.setAttribute('d', E.poly([P(g0p[0], g0p[1]), P(g1p[0], g1p[1])]));
          ghostH.set(g1p[0], g1p[1], R0 + ga);
          E.op(ghostG, f(st, T.run[0], 0.3) * on4);
          gap.setAttribute('d', E.poly([P(-2, 1), P(0, -1)])); E.drawOn(gap, f(st, T.far, 0.5)); E.op(gap, (f(st, T.far, 0.1) > 0 ? 1 : 0) * on4);
          E.drawOn(gapX, f(st, T.torn, 0.3)); E.op(gapX, (f(st, T.torn, 0.1) > 0 ? 1 : 0) * on4);
          E.op(tornL, fr(st, T.torn) * on4);

          /* ── 结果 ── */
          var onF = f(st, T.ok, 0.3) * (1 - f(st, T.reset2[0], 0.3)) + f(st, T.together[1] - 0.2, 0.3);
          E.drawOn(ok41, Math.min(1, onF)); E.op(ok41, onF > 0 ? Math.min(1, onF) : 0);
          var onG = f(st, T.reset2[0], 0.3);
          E.op(final.ax.g, onG * 0.8); E.op(final.ay.g, onG * 0.8); E.op(final.dot, onG * 0.8 * (1 - f(st, T.together[1] - 0.2, 0.3)));

          /* ── 右栏：公式 ── */
          E.op(wG, fr(st, T.rev) * (1 - f(st, T.prob, 0.4)));
          E.op(W2.g, fr(st, T.shFirst)); E.op(W1.g, fr(st, T.thenE)); E.op(W3.g, fr(st, T.rev));
          E.op(cG, fr(st, T.formula));
          var rp2 = f(st, T.rl[0], T.rl[1] - T.rl[0]);
          rlV.set(X + 334, FY + 50, X + 334 - 300 * rp2, FY + 50); E.op(rlV.g, (rp2 > 0 ? 1 : 0) * (1 - f(st, T.real - 0.3, 0.3)));
          E.op(C2.lab, fr(st, T.firstT)); E.op(C1.lab, fr(st, T.lastT));
          C2.box.style.strokeWidth = 2 + 2 * b(st, T.first, 0.9); E.glow(C2.box, E.HI, b(st, T.first, 0.9) > 0.05);
          C1.box.style.strokeWidth = 2 + 2 * b(st, T.last, 0.9); E.glow(C1.box, E.HI, b(st, T.last, 0.9) > 0.05);

          /* ── 右栏：算式 ── */
          E.op(g0.g, fr(st, T.why) * (1 - f(st, T.hp - 0.3, 0.3)));
          E.op(g0.r[1].g, fr(st, T.rev)); E.op(g0.r[1].parts[1], fr(st, T.shFirst)); E.op(g0.r[2].g, fr(st, T.turnS[0] - 1));
          E.op(g1.g, fr(st, T.hp) * (1 - f(st, T.thenE - 0.3, 0.3)));
          E.op(g1.r[0].parts[1], fr(st, T.h32));
          [[T.t1, T.t1r], [T.t2, T.t2r]].forEach(function (tt, k) { var r = g1.r[k + 1]; E.op(r.g, fr(st, tt[0])); E.op(r.parts[2], fr(st, tt[1])); E.op(r.parts[3], fr(st, tt[1])); });
          E.op(g1.r[3].g, fr(st, T.sum)); E.op(g1.r[3].parts[1], fr(st, T.sumR)); E.op(g1.r[4].g, fr(st, T.elb)); E.op(g1.r[4].parts[1], fr(st, T.elbR));
          E.op(g2.g, fr(st, T.thenE) * (1 - f(st, T.rotH - 0.3, 0.3)));
          E.op(g2.r[0].parts[1], fr(st, T.zero)); E.op(g2.r[1].g, fr(st, T.around)); E.op(g2.r[1].parts[1], fr(st, T.p12)); E.op(g2.r[2].g, fr(st, T.but));
          E.op(g3.g, fr(st, T.rotH) * (1 - f(st, T.prob - 0.3, 0.3)));
          var g3t = [T.rotH, T.sub, T.rot, T.a1, T.a2, T.aSum, T.back, T.run[0]];
          g3.r.forEach(function (r, k) { E.op(r.g, fr(st, g3t[k])); });
          E.op(g3.r[1].parts[1], fr(st, T.d1r)); E.op(g3.r[1].parts[2], fr(st, T.d2)); E.op(g3.r[1].parts[3], fr(st, T.d2r));
          E.op(g3.r[3].parts[2], fr(st, T.a1r)); E.op(g3.r[3].parts[3], fr(st, T.a1r)); E.op(g3.r[4].parts[2], fr(st, T.a2r)); E.op(g3.r[4].parts[3], fr(st, T.a2r));
          E.op(g3.r[5].parts[1], fr(st, T.aSumR)); E.op(g3.r[6].parts[1], fr(st, T.b1r)); E.op(g3.r[6].parts[2], fr(st, T.b2)); E.op(g3.r[6].parts[3], fr(st, T.b2r));
          E.op(g3.r[7].parts[2], fr(st, T.torn));
          E.op(g4.g, fr(st, T.prob) * (1 - f(st, T.so - 0.3, 0.3)));
          E.op(g4.r[0].parts[1], fr(st, T.moved[0])); E.op(g4.r[1].g, fr(st, T.reverse)); E.op(g4.r[1].parts[1], fr(st, T.shStay)); E.op(g4.r[2].g, fr(st, T.origin));
          E.op(g5.g, fr(st, T.so) * (1 - f(st, T.formula - 0.3, 0.3)));
          E.op(g5.r[0].g, fr(st, T.outer)); E.op(g5.r[1].g, fr(st, T.inward[0])); E.op(g5.r[2].g, fr(st, T.inner)); E.op(g5.r[3].g, fr(st, T.zeroPl)); E.op(g5.r[3].parts[1], fr(st, T.ok));
          E.op(g6.g, fr(st, T.formula) * (1 - f(st, T.real - 0.3, 0.3)));
          E.op(g6.r[0].g, fr(st, T.rl[0])); E.op(g6.r[1].g, fr(st, T.first)); E.op(g6.r[1].parts[1], fr(st, T.firstT));
          E.op(g6.r[2].g, fr(st, T.last - 0.6)); E.op(g6.r[2].parts[1], fr(st, T.last)); E.op(g6.r[2].parts[2], fr(st, T.lastT));
          E.op(g6.r[3].g, fr(st, T.ep4)); E.op(ep4, fr(st, T.ep4)); E.op(g6.r[4].g, fr(st, T.near));
          E.op(g7.g, fr(st, T.real));
          E.op(g7.r[0].parts[1], fr(st, T.real + 0.8)); E.op(g7.r[1].g, fr(st, T.pose)); E.op(g7.r[1].parts[1], fr(st, T.angles)); E.op(g7.r[2].g, fr(st, T.oneByOne)); E.op(g7.r[2].parts[1], fr(st, T.right));
        }
      };
    }
  };
})();
