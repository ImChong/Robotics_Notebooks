// 012 第 2 幕：两个圆——肘只能在交点上。左边同第 1 幕的网格 + UR5，紫色目标 ⊕ 在 (1, 0)（手照第 11 集放在小臂上离肘两格）；右栏一行行写。
// 句 1「关键是肘」肘 ⊙ 鼓一下，「大臂和小臂怎么摆」两节各亮一下；
// 句 2「大臂的长度不会变」肩带着整条胳膊来回摆（肘离肩不变），「离肩的距离」从肩到肘画半径、写 2.24 格，「只能在以肩为中心」
//    琥珀色的圆从肘这一点起描一整圈，「半径二点二四」半径鼓一下；
// 句 3「小臂的长度也不会变」「肘离手」从目标伸出一根 2 格的绿线（小臂），「以手为中心」绿线绕目标转一整圈、描出绿色的圆；
// 句 4「两个条件都要满足」两个圆亮一下，「两个圆的交点」交点处鼓一下，「一上」(1, 2) 出亮黄圈，「一下」(1, −2) 出青圈；
// 句 5「胳膊往上拱」UR5 弯肘、手送到目标（肘上），「胳膊往下拱」肘下的虚影（青）出来，「手都停在同一个点」目标鼓一下；
// 句 6「上下对称」两个肘之间画竖虚线，「肩、肘、手连成一个三角形」三角形（亮黄）一条边一条边画出，「肩和手的连线」这条边描紫，
//    「翻过去」三角形沿 x 轴翻到下面、变青，「三条边的长度都不变」三条边标 2.24、2、1，「另一组解」下面的肘鼓一下。
// 时刻取自配音的逐词时间，见 ../ep012.py。
(function () {
  var E = window.EP012, K = E.K, H = E.HEX;
  var D = Math.PI / 180, A = E.ARM, RX = A.RX;
  var T = {
    key: 0.61, elbow: 1.06, arms: [2.91, 3.36], set: 4.64,
    upper: 6.04, elSh: 7.63, swing: [6.4, 9.6], dist: 8.22, r1: 9.11, only: 10.26, c1: [10.78, 12.3], rad1: 12.02, out2: 13.9,
    dist2: 16.59, stick: 17.25, r2: 18.15, c2: [19.34, 20.8], rad2: 20.57,
    both: 22.98, meet: 24.63, cross: 26.33, upPt: 28.39, dnPt: 28.76, two: 29.1,
    upArm: [31.12, 32.4], upLab: 33.05, dnArm: 34.97, dnLab: 36.04, same: 38.12,
    sym: 41.38, tri: [42.41, 44.35], line: 45.54, flip: [46.59, 47.25], sides: 47.36, other: 49.08
  };
  var END = 50.97;
  var UP = E.ik2(1, 0, true), DN = E.ik2(1, 0, false), A0 = Math.atan2(2, 1);

  E.scenes[1] = {
    title: '两个圆：肘只能在交点上',
    dur: END,
    cues: [
      { at: 0, s: '关键是肘：肘定了，大臂、小臂就都定了' },
      { at: 5.95, s: '大臂长度不变：肘离肩总是 2.24 格 → 以肩为中心的圆' },
      { at: 14.85, s: '小臂长度不变：肘离手总是 2 格 → 以手为中心的圆' },
      { at: 22.89, s: '两个条件都满足：两个圆的交点，一上一下' },
      { at: 30.69, s: '**肘上**、**肘下**两种摆法，手停在同一点' },
      { at: 40.67, s: '上下对称：三角形沿着肩—手的连线翻过去，三条边不变，就是另一组解' }
    ],
    build: function (K) {
      var s = K.sceneSvg('肘的位置定了，大臂和小臂就都定了；大臂长度不变，肘离肩总是 2.24 格，肘只能在以肩为中心、半径 2.24 的圆上；小臂长度不变，肘离手总是 2 格，肘也要在以手 (1, 0) 为中心、半径 2 的圆上；两个条件都满足，肘只能在两个圆的交点上，一上 (1, 2) 一下 (1, −2)；肘在上面叫肘上，肘在下面叫肘下，两种摆法手停在同一点；肩、肘、手连成三角形，沿着肩和手的连线翻过去，三条边长度不变，就是另一组解');

      var under = E.el('g', {}, s);
      var box = [A.OX - 110, A.OY - 214, 440, 420];
      var ghostG = E.el('g', {}, s), ghost = E.arm2(ghostG, A.OX, A.OY, A.S, 0, box);
      var armG = E.el('g', {}, s), arm = E.arm2(armG, A.OX, A.OY, A.S, 0, box), u = arm.u, V = arm.view;
      var pl = E.plane(under, { ox: A.OX, oy: A.OY, u: u, x0: -1, x1: 4, y0: -3, y1: 3 });
      var P = pl.P, O = P(0, 0), TG = P(1, 0), EU = P(1, 2), ED = P(1, -2);
      var top = E.el('g', {}, s);
      function circPath(c, r, a0) { var pts = []; for (var k = 0; k <= 120; k++) { var a = a0 + 2 * Math.PI * k / 120; pts.push([c[0] + r * Math.cos(a), c[1] - r * Math.sin(a)]); } return E.poly(pts); }
      // 两个圆：从肘（上面那个交点）起描
      var c1 = K.paint(E.el('path', { d: circPath(O, E.L1 * u, A0), fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '9 6' }, top), null, E.C1);
      var c2 = K.paint(E.el('path', { d: circPath(TG, E.L2 * u, Math.PI / 2), fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '9 6' }, top), null, E.C2);
      // 句 2 的半径、句 3 的「小臂」
      var r1 = E.vec(top, E.C1, 3.5, { head: 0.1 }), r1Lab = E.halo(E.txt(top, 0, 0, '2.24 格', 21, E.C1, 'middle', 900));
      var mv1 = K.paint(E.el('circle', { r: 7, 'stroke-width': 3 }, top), E.S1, E.C1);
      var stick = K.paint(E.el('path', { fill: 'none', 'stroke-width': 7, 'stroke-linecap': 'round' }, top), null, E.C2);
      var r2Lab = E.halo(E.txt(top, 0, 0, '2 格', 21, E.C2, 'middle', 900));
      var mv2 = K.paint(E.el('circle', { r: 7, 'stroke-width': 3 }, top), E.S1, E.C2);
      // 句 6：三角形（亮黄 → 翻到下面变青）、两个肘之间的竖虚线
      var symL = K.paint(E.el('path', { d: 'M ' + EU[0] + ' ' + EU[1] + ' V ' + ED[1], fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '5 6' }, top), null, E.INK2);
      var triStill = E.el('g', {}, top);   // 留在上面的那个（淡）
      K.paint(E.el('path', { d: E.poly([O, EU, TG], true), 'stroke-width': 2, 'stroke-linejoin': 'round' }, triStill), E.tint(E.UP, 14, 'transparent'), E.UP);
      var triF = E.el('g', {}, top), triIn = E.el('g', {}, triF);
      var tSide1 = K.paint(E.el('path', { d: 'M ' + O[0] + ' ' + O[1] + ' L ' + EU[0] + ' ' + EU[1], fill: 'none', 'stroke-width': 4, 'stroke-linecap': 'round' }, triIn), null, E.UP);
      var tSide2 = K.paint(E.el('path', { d: 'M ' + EU[0] + ' ' + EU[1] + ' L ' + TG[0] + ' ' + TG[1], fill: 'none', 'stroke-width': 4, 'stroke-linecap': 'round' }, triIn), null, E.UP);
      var tFill = K.paint(E.el('path', { d: E.poly([O, EU, TG], true) }, triIn), E.tint(E.UP, 22, 'transparent'), 'none');
      var base = K.paint(E.el('path', { d: 'M ' + (O[0] - 0.5 * u) + ' ' + O[1] + ' H ' + (TG[0] + 1.1 * u), fill: 'none', 'stroke-width': 5, 'stroke-linecap': 'round' }, top), null, E.TGT);
      // 三条边的长度（上、下各一份）
      function sideLabs(sg, col) {
        var g = E.el('g', {}, top), mU = [(O[0] + EU[0]) / 2, (O[1] + (sg > 0 ? EU[1] : ED[1])) / 2], mF = [(EU[0] + TG[0]) / 2, ((sg > 0 ? EU[1] : ED[1]) + TG[1]) / 2];
        E.halo(E.txt(g, mU[0] - 12, mU[1] + 6, '2.24', 20, E.C1, 'end', 900));
        E.halo(E.txt(g, mF[0] + 12, mF[1] + 6, '2', 20, E.C2, 'start', 900));
        if (sg > 0) E.halo(E.txt(g, (O[0] + TG[0]) / 2 + 4, O[1] - 10, '1', 20, E.TGT, 'middle', 900));   // 底边上下共用，只写一次
        return g;
      }
      var labsU = sideLabs(1), labsD = sideLabs(-1);
      // 交点
      var ringU = E.el('g', {}, top), ringD = E.el('g', {}, top);
      K.paint(E.el('circle', { cx: EU[0], cy: EU[1], r: 13, 'stroke-width': 4, fill: 'none' }, ringU), null, E.UP);
      K.paint(E.el('circle', { cx: ED[0], cy: ED[1], r: 13, 'stroke-width': 4, fill: 'none' }, ringD), null, E.DOWN);
      var euLab = E.halo(E.txt(top, EU[0] - 18, EU[1] - 16, '(1, 2)', 22, E.UP, 'end', 900));
      var edLab = E.halo(E.txt(top, ED[0] - 18, ED[1] + 32, '(1, −2)', 22, E.DOWN, 'end', 900));
      var upName = E.halo(E.txt(top, EU[0] + 20, EU[1] - 16, '肘上', 22, E.UP, 'start', 900));
      var dnName = E.halo(E.txt(top, ED[0] + 20, ED[1] + 32, '肘下', 22, E.DOWN, 'start', 900));
      // 目标
      var tgt = E.target(top, TG[0], TG[1], 13, E.TGT);
      var tgLab = E.halo(E.txt(top, TG[0] + 20, TG[1] + 32, '手 (1, 0)', 20, E.TGT, 'start', 900));   // 目标右下：右上要写绿线的 2 格
      // 肘 ⊙（跟着 UR5 的肘走）
      var ring = E.el('g', {}, top);
      K.paint(E.el('circle', { r: 11, 'stroke-width': 3.5 }, ring), E.S1, E.HI); K.paint(E.el('circle', { r: 3.5 }, ring), E.HI);

      /* ══ 右栏 ══ */
      var RK = E.rowKit(), Rg = E.el('g', {}, s), X = RX + 10;
      var gA = E.el('g', {}, Rg);
      var a1 = RK.line(gA, X, 48, [['关键是 ', E.INK2], ['肘', E.HI, 900], [' 的位置', E.INK2]]);
      var a2 = RK.line(gA, X, 84, [['肘定了 → 大臂、小臂都定了', E.INK2]], 21);
      var b1 = RK.line(gA, X, 140, [['① 肘离肩', E.INK, 900], ['：', E.INK, 900], ['2.24 格', E.C1, 900]]);
      var b2 = RK.line(gA, X, 174, [['→ 以肩为中心的圆', E.C1, 700]], 21);
      var c1r = RK.line(gA, X, 230, [['② 肘离', E.INK, 900], ['手', E.TGT, 900], ['：', E.INK, 900], ['2 格', E.C2, 900]]);
      var c2r = RK.line(gA, X, 264, [['→ 以', E.C2, 700], ['手', E.TGT, 700], ['为中心的圆', E.C2, 700]], 21);
      var d1 = RK.line(gA, X, 322, [['① ② 都满足：', E.INK, 900]]);
      var d2 = RK.line(gA, X, 360, [['两个圆的交点', E.HI, 900], ['：一上、一下', E.INK2]]);
      var gB = E.el('g', {}, Rg);
      var e1 = RK.line(gB, X, 58, [['肘上', E.UP, 900], ['：胳膊往上拱', E.INK2]]);
      var e2 = RK.line(gB, X, 98, [['肘下', E.DOWN, 900], ['：胳膊往下拱', E.INK2]]);
      var e3 = RK.line(gB, X, 138, [['手', E.TGT, 900], ['都停在 ', E.INK2], ['(1, 0)', E.TGT, 900]]);
      var f1 = RK.line(gB, X, 210, [['上下对称', E.INK, 900]]);
      var f2 = RK.line(gB, X, 248, [['三角形沿着 ', E.INK2], ['肩—手', E.TGT, 900], [' 的连线翻过去', E.INK2]], 21);
      var f3 = RK.line(gB, X, 288, [['三条边 ', E.INK2], ['2.24', E.C1, 900], ['、', E.INK2], ['2', E.C2, 900], ['、', E.INK2], ['1', E.TGT, 900], [' 都不变', E.INK2]], 21);
      var f4 = RK.line(gB, X, 336, [['→ 另一组解', E.DOWN, 900]], 24);

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) { RK.layout(); laid = true; }
          var f = E.fade, b = E.bump;

          /* ── UR5：句 2 肩带着胳膊来回摆；句 5 弯肘把手送到目标 ── */
          var sw = f(st, T.swing[0], T.swing[1] - T.swing[0]), dA = 0.75 * Math.sin(sw * 2 * Math.PI) * (sw > 0 && sw < 1 ? 1 : 0);
          var up = f(st, T.upArm[0], T.upArm[1] - T.upArm[0]);
          var q1 = A0 + dA, q2 = (0 + dA) * (1 - up) + UP[1] * up;
          arm.pose(q1, q2);
          V.clearTint();
          var dimArm = f(st, T.upper, 0.4) * (1 - f(st, T.upArm[0], 0.4));
          V.tint('all', null, 1, 1 - 0.5 * dimArm);
          V.tint(['upper_arm_link'], E.HI, 0.8 * b(st, T.arms[0], 0.9), 1 - 0.5 * dimArm);
          V.tint(['forearm_link'], E.HI, 0.8 * b(st, T.arms[1], 0.9), 1 - 0.5 * dimArm);
          ghost.pose(DN[0], DN[1]); ghost.view.clearTint(); ghost.view.tint('all', E.DOWN, 0.8, 0.42);
          E.op(ghostG, 0.9 * f(st, T.dnArm, 0.5));
          var e = arm.elbow();
          ring.setAttribute('transform', 'translate(' + e[0].toFixed(1) + ' ' + e[1].toFixed(1) + ') scale(' + (1 + 0.45 * Math.max(b(st, T.elbow, 0.8), b(st, T.key + 0.9, 0.8))).toFixed(3) + ')');
          E.op(ring, 1 - f(st, T.cross, 0.3));

          /* ── 句 2：离肩 2.24 的圆 ── */
          var on2 = f(st, T.dist, 0.3) * (1 - f(st, T.out2, 0.4));
          var c1p = f(st, T.c1[0], T.c1[1] - T.c1[0]), ang1 = A0 + 2 * Math.PI * c1p;
          var m1 = c1p > 0 && c1p < 1 ? [O[0] + E.L1 * u * Math.cos(ang1), O[1] - E.L1 * u * Math.sin(ang1)] : e;
          r1.set(O[0], O[1], m1[0], m1[1]); E.op(r1.g, on2);
          r1.line.style.strokeWidth = 3.5 + 2.5 * b(st, T.rad1, 0.8);
          r1Lab.setAttribute('x', ((O[0] + m1[0]) / 2 - 14).toFixed(1)); r1Lab.setAttribute('y', ((O[1] + m1[1]) / 2).toFixed(1));
          E.op(r1Lab, f(st, T.r1, 0.3) * (1 - f(st, T.out2, 0.4)));
          mv1.setAttribute('cx', m1[0].toFixed(1)); mv1.setAttribute('cy', m1[1].toFixed(1)); E.op(mv1, c1p > 0 && c1p < 1 ? 1 : 0);
          E.drawOn(c1, c1p);
          c1.style.strokeWidth = 3 + 2.5 * Math.max(b(st, T.both, 0.9), b(st, T.meet, 0.9));
          E.op(c1, (c1p > 0 ? 1 : 0) * (1 - 0.6 * f(st, T.upArm[0], 0.4)));

          /* ── 句 3：离手 2 格的圆（绿线绕目标转一圈） ── */
          var c2p = f(st, T.c2[0], T.c2[1] - T.c2[0]), ang2 = Math.PI / 2 + 2 * Math.PI * c2p;
          var m2 = [TG[0] + E.L2 * u * Math.cos(ang2), TG[1] - E.L2 * u * Math.sin(ang2)];
          var sl = f(st, T.stick, 0.4);
          stick.setAttribute('d', 'M ' + TG[0].toFixed(1) + ' ' + TG[1].toFixed(1) + ' L ' + (TG[0] + (m2[0] - TG[0]) * sl).toFixed(1) + ' ' + (TG[1] + (m2[1] - TG[1]) * sl).toFixed(1));
          var on3 = (sl > 0 ? 1 : 0) * (1 - f(st, T.both, 0.4));
          E.op(stick, 0.85 * on3);
          // 「2 格」只在绿线开始转之前标（竖着、标在左边）：跟着转会扫过圆和线
          r2Lab.setAttribute('x', (TG[0] - 14).toFixed(1)); r2Lab.setAttribute('y', (TG[1] - E.L2 * u / 2 + 6).toFixed(1)); r2Lab.setAttribute('text-anchor', 'end'); r2Lab.style.textAnchor = 'end';
          E.op(r2Lab, f(st, T.r2, 0.3) * (1 - f(st, T.c2[0] - 0.2, 0.2))); r2Lab.style.fontSize = (21 * (1 + 0.25 * b(st, T.rad2, 0.8))) + 'px';
          mv2.setAttribute('cx', m2[0].toFixed(1)); mv2.setAttribute('cy', m2[1].toFixed(1)); E.op(mv2, sl >= 1 ? on3 : 0);
          E.drawOn(c2, c2p);
          c2.style.strokeWidth = 3 + 2.5 * Math.max(b(st, T.both, 0.9), b(st, T.meet, 0.9));
          E.op(c2, (c2p > 0 ? 1 : 0) * (1 - 0.6 * f(st, T.upArm[0], 0.4)));

          /* ── 句 4–5：交点、两组解 ── */
          E.pop(ringU, EU[0], EU[1], f(st, T.upPt, 0.35), 0.4); E.pop(ringD, ED[0], ED[1], f(st, T.dnPt, 0.35), 0.4);
          var pulse = b(st, T.cross, 0.9);
          [ringU, ringD].forEach(function (g) { E.glow(g, E.HI, pulse > 0.05 || b(st, T.other, 0.9) > 0.05); });
          E.op(euLab, fr(st, T.upPt) * (1 - f(st, T.sides, 0.3))); E.op(edLab, fr(st, T.dnPt) * (1 - f(st, T.sides, 0.3)));
          E.op(upName, fr(st, T.upLab)); E.op(dnName, fr(st, T.dnLab));
          E.pop(tgt, TG[0], TG[1], 1, 1);
          tgt.setAttribute('transform', 'translate(' + TG[0] + ' ' + TG[1] + ') scale(' + (1 + 0.5 * b(st, T.same, 1.0)) + ') translate(' + (-TG[0]) + ' ' + (-TG[1]) + ')');
          E.op(tgLab, (1 - f(st, T.stick - 0.3, 0.3) + f(st, T.both, 0.3)) * (1 - f(st, T.sym, 0.3)));   // 绿线绕目标转的时候「2 格」会扫过这个标签：先藏起来

          /* ── 句 6：三角形翻过去 ── */
          E.drawOn(symL, f(st, T.sym, 0.5)); E.op(symL, (f(st, T.sym, 0.1) > 0 ? 1 : 0));
          var tp = f(st, T.tri[0], T.tri[1] - T.tri[0]);
          E.drawOn(tSide1, E.clamp(tp * 2, 0, 1)); E.drawOn(tSide2, E.clamp(tp * 2 - 1, 0, 1)); E.op(tFill, f(st, T.tri[1], 0.3));
          E.drawOn(base, f(st, T.line, 0.5)); E.op(base, (f(st, T.line, 0.1) > 0 ? 0.9 : 0));
          var fp = f(st, T.flip[0], T.flip[1] - T.flip[0]), sy = Math.cos(Math.PI * fp);
          triF.setAttribute('transform', 'translate(0 ' + O[1] + ') scale(1 ' + sy.toFixed(4) + ') translate(0 ' + (-O[1]) + ')');
          var colT = fp > 0.5 ? E.DOWN : E.UP;
          K.paint(tSide1, null, colT); K.paint(tSide2, null, colT); K.paint(tFill, E.tint(colT, 22, 'transparent'), 'none');
          E.op(triF, f(st, T.tri[0], 0.1) > 0 ? 1 : 0);
          E.op(triStill, 0.8 * f(st, T.flip[0], 0.3));
          E.op(labsU, fr(st, T.sides)); E.op(labsD, fr(st, T.sides + 0.4));

          /* ── 右栏 ── */
          E.op(gA, 1 - f(st, T.upArm[0] - 0.3, 0.4));
          E.op(a1.g, fr(st, T.key)); E.op(a1.parts[1], fr(st, T.elbow)); E.op(a1.parts[2], fr(st, T.elbow)); E.op(a2.g, fr(st, T.arms[0]));
          E.op(b1.g, fr(st, T.elSh)); E.op(b1.parts[1], fr(st, T.r1)); E.op(b1.parts[2], fr(st, T.r1)); E.op(b2.g, fr(st, T.only));
          E.op(c1r.g, fr(st, T.dist2)); E.op(c1r.parts[2], fr(st, T.r2)); E.op(c1r.parts[3], fr(st, T.r2)); E.op(c2r.g, fr(st, T.c2[0]));
          E.op(d1.g, fr(st, T.both)); E.op(d2.g, fr(st, T.meet)); E.op(d2.parts[1], fr(st, T.upPt));
          E.op(gB, f(st, T.upArm[0], 0.4));
          E.op(e1.g, fr(st, T.upArm[0])); E.op(e2.g, fr(st, T.dnArm - 0.6)); E.op(e3.g, fr(st, T.same));
          E.op(f1.g, fr(st, T.sym)); E.op(f2.g, fr(st, T.line)); E.op(f2.parts[2], fr(st, T.flip[0])); E.op(f3.g, fr(st, T.sides)); E.op(f4.g, fr(st, T.other));
        }
      };
    }
  };
})();
