// 片头主图（800×420）：三句旁白各一段——
// ① 左边第 9–13 集的侧视网格 + UR5（第 13 集第 6 幕：肩先转了 90°，肘 (−2, 1)、手 (−2, 3)），「每个关节的转速」肩、肘旁蓝色小弯箭头，「手的速度」手上紫箭头
//    （转速 (1, −1)：(−3, −2) − (−2, 0) = (−1, −2)，按 E.vHand 算），右上框「第 13 集：转速 × 雅可比 = 手的速度」「每一列 = 关节轴此刻的螺旋轴」；
// ② 「这一集」右边换成问题：「站在底座上看」底座的坐标系亮、写「站在底座上看」，「站在手上看」手的坐标系亮，「雅可比」右边两个 6 × 2（J_s、J_b）中间一个问号，
//    「为什么不一样」「怎么互相换」；
// ③ 「六乘六的矩阵」「伴随变换」右边写 J_b = [Ad_Tbs] J_s，左边一条从底座弯到手的虚线箭头写「伴随变换」，「手此刻的位置和潮向」手亮，「换成另一种」箭头画实。
// 时刻取自配音的逐词时间（../ep014.py 的 [from, to] 等于每句配音实长，分镜时间 = 念到这个词的时刻）。
(function () {
  var E = window.EP014, K = E.K, H = E.HEX;
  var RX = 452, D90 = Math.PI / 2;
  var T = {
    ep13: 0.7, each: 2.11, rate: 2.9, jac: 3.85, hand: 4.93, write: 5.91, eachCol: 7.19, screw: 9.34,
    now: 11.12, same: 12.87, base: 13.94, handV: 15.28, jac2: 17.2, why: 18.03, how: 19.61,
    ans: 21.69, m66: 22.67, adj: 24.4, only: 26.51, pos: 27.62, ori: 28.14, mul: 28.93, other: 30.87
  };

  E.hero = function (K) {
    var s = K.sceneSvg('片头主图：第十三集讲过，每个关节的转速乘上雅可比，就是手的速度；写成运动旋量，每一列就是关节轴此刻的螺旋轴。这一集回答一个问题：同一个速度，站在底座上看、和站在手上看，写出来的雅可比为什么不一样？怎么互相换？答案是一个六乘六的矩阵，叫做伴随变换：它用到的，只有手此刻的位置和朝向，乘上它，就从一种换成另一种');
    var recall = E.recall(s, [['r13hero', 0.55, 5.55], ['r13cols', 5.85, 10.65]]);   // 回顾：第十三集讲过：转速乘雅可比（013 片头）；每一列是螺旋轴（013 第 6 幕）

    var pn = E.panel14(s, { ox: 268, oy: 292 }), P = pn.P, u = pn.u, top = pn.top, O = P(0, 0);
    pn.pose(E.P90[0], E.P90[1]);
    var J = pn.joints(), pE = P(J[1][0], J[1][1]), pH = P(J[2][0], J[2][1]);
    var bF = E.frame2(top, P, 1, '底座', { labAt: [0.75, -0.62], tipOff: E.BASE_TIP }), hF = E.frame2(top, P, 1, '手', { labAt: [0.45, 0.45], labColor: E.V });
    bF.set(0, 0, 0); hF.set(-2, 3, D90);
    var jS = E.joint(top, E.HI), jE = E.joint(top, E.HI), hand = E.dot(top, E.V, 7);
    jS.at(O); jE.at(pE); hand.at(pH);
    var sArc = E.arcArrow(top, O[0], O[1], 0.6 * u, E.OMEGA, 3.5, { head: 12 }), eArc = E.arcArrow(top, pE[0], pE[1], 0.55 * u, E.OMEGA, 3.5, { head: 11 });
    sArc.set(3.45, 4.55); eArc.set(3.6, 2.4);   // 肩逆时针（+1）、肘顺时针（−1），都画在关节左边（右边是大臂）
    var vv = E.gv(top, P, E.V, 5.5, { head: 17 }); vv.line.style.filter = 'drop-shadow(0 0 2px #000)';
    var v1 = E.vHand(J, 1, -1);   // (−1, −2)
    // 视角标签（全片同一个样式，见 00-common.js 的 E.viewTag）：底座旁「底座上看」、手旁「手上看」
    var standB = E.viewTag(top, P(0.4, -1.45)[0], P(0.4, -1.45)[1] - 12, 'base', { anchor: 'start' }).g;
    var standH = E.viewTag(top, P(-2.55, 4.35)[0], P(-2.55, 4.35)[1] - 14, 'hand', { anchor: 'end' }).g;
    // 伴随变换：从底座弯到手的虚线箭头
    var adP = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3.5, 'stroke-dasharray': '8 6' }, top), null, E.HI);
    var a0 = P(0.9, 0.6), a1 = P(-0.9, 3.6), cp = P(1.2, 3.8);
    adP.setAttribute('d', 'M ' + a0[0] + ' ' + a0[1] + ' Q ' + cp[0] + ' ' + cp[1] + ' ' + a1[0] + ' ' + a1[1]);
    var adH = E.vec(top, E.HI, 3.5, { head: 14 });
    var adL = E.halo(E.txt(top, P(1.0, 3.0)[0] + 6, P(1.0, 3.0)[1], '伴随变换', 19, E.HI, 'start', 900));

    /* ── 右 ── */
    var R = E.el('g', {}, s), CX = RX + 165;
    var bA = E.el('g', {}, R);
    K.paint(E.el('rect', { x: RX + 6, y: 16, width: 318, height: 100, rx: 14, 'stroke-width': 3 }, bA), E.tint(E.ACC, 10, 'var(--demo-surface)'), E.ACC);
    E.txt(bA, CX, 48, '第 13 集：转速 × 雅可比 = 手的速度', 18, E.INK, 'middle', 900);
    var bA2 = E.txt(bA, CX, 90, '每一列 = 关节轴此刻的螺旋轴', 19, E.WARN, 'middle', 900);
    var bQ = E.el('g', {}, R);
    E.txt(bQ, CX, 152, '这一集：同一个速度', 20, E.INK2, 'middle', 700);
    var MY = 256;
    var Js = E.matN(bQ, RX + 70, MY, [[0, 0], [0, 0], [1, 1], [0, 1], [0, 2], [0, 0]], { size: 16, dx: 28, dy: 20, color: function (i, j) { return j === 0 ? E.J1 : E.J2; } });
    var Jb = E.matN(bQ, RX + 262, MY, [[0, 0], [0, 0], [1, 1], [-2, 0], [3, 2], [0, 0]], { size: 16, dx: 30, dy: 20, color: function (i, j) { return j === 0 ? E.J1 : E.J2; } });
    var jsT = E.viewTag(bQ, RX + 70, 160, 'base').g, jbT = E.viewTag(bQ, RX + 262, 160, 'hand').g;   // 两个雅可比正上方各挂视角标签
    var qm = E.txt(bQ, RX + 166, MY + 6, '≠ ?', 28, E.ERR, 'middle', 900);
    var q1 = E.txt(bQ, CX, 342, '为什么不一样？怎么互相换？', 21, E.HI, 'middle', 900);
    var gAns = E.el('g', {}, R);
    var fA = E.tex(gAns, RX + 20, 392, '\\textcolor{' + H.violet + '}{J_b}=\\textcolor{' + H.hi + '}{[\\mathrm{Ad}_{T_{bs}}]}\\,J_s', 26, E.INK, 'start', { w: 220, h: 46 });
    var fL = E.txt(gAns, RX + 236, 398, '6 × 6', 21, E.HI, 'start', 900);

    return {
      el: s,
      draw: function (t) {
      recall(t);
        var f = E.fade, b = E.bump, fr = function (a) { return f(t, a, 0.3); };
        pn.pose(E.P90[0], E.P90[1]); pn.view.clearTint();
        E.op(jS, fr(0.2)); E.op(jE, fr(0.2)); E.op(hand, fr(0.2));
        E.glow(jS, E.HI, b(t, T.each, 0.8) > 0.05); E.glow(jE, E.HI, b(t, T.each, 0.8) > 0.05);
        var rOn = E.win(t, T.rate, T.now, 0.3);
        E.op(sArc.g, rOn); E.op(eArc.g, rOn);
        var vg = f(t, T.hand, 0.6); vv.at(J[2][0], J[2][1], J[2][0] + v1[0] * vg, J[2][1] + v1[1] * vg); E.op(vv.g, (t > T.hand ? 1 : 0) * (1 - fr(T.now)));
        E.op(bF.g, fr(0.3)); E.op(hF.g, fr(0.3));
        E.glow(bF.g, E.INK, E.win(t, T.base, T.handV, 0.2) > 0.5); E.glow(hF.g, E.V, E.win(t, T.handV, T.jac2, 0.2) > 0.5 || E.win(t, T.pos, T.mul, 0.2) > 0.5);
        E.op(standB, fr(T.base)); E.op(standH, fr(T.handV));
        // 伴随变换的弯箭头：「伴随变换」出虚线，「换成另一种」箭头头出来
        E.drawOn(adP, f(t, T.adj, 0.8)); adH.set(P(-0.6, 3.75)[0], P(-0.6, 3.75)[1], a1[0], a1[1]); E.op(adH.g, fr(T.other)); E.op(adL, fr(T.adj));
        /* ── 右 ── */
        E.op(bA, fr(T.ep13) * (1 - 0.6 * fr(T.now))); E.op(bA2, fr(T.eachCol)); E.glow(bA2, E.WARN, b(t, T.screw, 1.0) > 0.05);
        E.op(bQ, fr(T.now));
        E.op(jsT, fr(T.base)); E.op(Js.g, fr(T.jac2)); E.op(jbT, fr(T.handV)); E.op(Jb.g, fr(T.jac2)); E.op(qm, fr(T.why));
        E.op(q1, fr(T.why) * (1 - fr(T.ans)));
        E.op(gAns, fr(T.ans)); E.op(fA, fr(T.adj)); E.op(fL, fr(T.m66));
        E.glow(fA, E.HI, b(t, T.mul, 1.2) > 0.05);
      }
    };
  };
})();
