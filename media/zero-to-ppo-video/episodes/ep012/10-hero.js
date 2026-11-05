// 片头主图（800×420）：三句旁白各一段——
// ① 左边第 9、10 集的侧视网格 + UR5（肘在 (1, 2)，小臂水平）。「关节」肩、肘出蓝色的转角 θ₁、θ₂，「指数积公式」右边出上下两个框：
//    每个关节转多少（上）→ 手的位姿（下），「算出」往下的箭头画出来，「手」手腕亮一下，「正向运动学」箭头旁写「正向运动学 · 一个答案」；
// ② 「这一集回答一个问题」往下的箭头变淡、右边长出往上的箭头，「手」网格上出紫色目标 ⊕ (1, 0)，「位置」写坐标，「每个关节」「该转多少」两个转角换成问号，
//    「为什么有时候没有解，有时候不止一组」箭头旁写出这两问（照 011 片尾的预告）；
// ③ 「逆向运动学」往上的箭头旁写「逆向运动学」，「两组」UR5 弯肘把手腕送到目标（肘上，亮黄），同时出肘下的虚影（青），
//    「好几组」「一组都没有」下面出三个小签：2 组、好几组、0 组；「一组都没有」网格右上角出一个够不到的橙色目标、打 ×。
// 时刻取自配音的逐词时间（../ep012.py 的 [from, to] 等于每句配音实长，分镜时间 = 念到这个词的时刻）。
(function () {
  var E = window.EP012, K = E.K, H = E.HEX;
  var D = Math.PI / 180, A = { OX: 92, OY: 205, S: 2.0 }, RX = 452;   // 主图不缩放：UR5 放大到一格约 55 px
  var T = {
    joints: 2.42, turn: 2.77, poe: 3.71, calc: 5.02, hand: 5.3, pose: 5.62, fk: 6.89,
    back: 9.51, tgt: 10.66, where: 11.72, each: 12.4, how: 13.3, why: 15.47, more: 16.7,
    ik: 19.06, two: 21.34, move: [21.34, 22.2], many: 22.03, none: 23.55
  };
  var UP = E.ik2(1, 0, true), DN = E.ik2(1, 0, false);

  E.hero = function (K) {
    var s = K.sceneSvg('片头主图：第十一集讲过，知道每个关节转了多少，用指数积公式就能得到手的位姿，这叫正向运动学，只有一个答案；这一集回答一个问题：知道手要停在什么位置 (1, 0)，每个关节该转多少？为什么有时候没有解，有时候不止一组？这叫逆向运动学，答案可能有两组（肘上、肘下）、好几组，也可能一组都没有（目标太远够不到）');

    /* ── 左：网格 + 两台 UR5（第二台是肘下的虚影） ── */
    var under = E.el('g', {}, s);
    var ghostG = E.el('g', {}, s);
    var box = [A.OX - 90, A.OY - 200, 400, 420];
    var ghost = E.arm2(ghostG, A.OX, A.OY, A.S, 0, box);
    var armG = E.el('g', {}, s);
    var arm = E.arm2(armG, A.OX, A.OY, A.S, 0, box), u = arm.u;
    var P = E.plane(under, { ox: A.OX, oy: A.OY, u: u, x0: -1, x1: 4, y0: -3, y1: 3 }).P;
    var top = E.el('g', {}, s), O = P(0, 0);
    // 两个关节：肩 ⊙（原点）、肘 ⊙ 跟着动；转角弧 + θ₁、θ₂ / 问号
    var j1 = E.el('g', {}, top), j2 = E.el('g', {}, top);
    [j1, j2].forEach(function (g) { K.paint(E.el('circle', { r: 11, 'stroke-width': 3.5 }, g), E.S1, E.HI); K.paint(E.el('circle', { r: 3.5 }, g), E.HI); });
    var arc1 = E.arcArrow(top, O[0], O[1], 0.62 * u, E.OMEGA, 3.5, { head: 12 }), arc2 = E.arcArrow(top, 0, 0, 0.5 * u, E.OMEGA, 3.5, { head: 11 });
    function mathLab(sx) { var n = E.halo(E.txt(top, 0, 0, sx, 26, E.OMEGA, 'middle', 900)); n.style.fontFamily = 'KaTeX_Math, serif'; n.style.fontStyle = 'italic'; return n; }
    var th1 = mathLab('θ₁'), th2 = mathLab('θ₂');
    var q1 = E.halo(E.txt(top, 0, 0, '?', 30, E.HI, 'middle', 900)), q2 = E.halo(E.txt(top, 0, 0, '?', 30, E.HI, 'middle', 900));
    var wDot = K.paint(E.el('circle', { r: 7, 'stroke-width': 3 }, top), E.TGT, E.S1);
    // 目标 ⊕ (1, 0) 与够不到的目标 (4, 2.6)
    var TG = P(1, 0), tgt = E.target(top, TG[0], TG[1], 13, E.TGT);
    var tgtLab = E.halo(E.txt(top, TG[0] + 22, TG[1] + 34, '(1, 0)', 22, E.TGT, 'start', 900));
    var FAR = P(4, 2.6), far = E.target(top, FAR[0], FAR[1], 13, E.ERR), farX = E.cross(top, FAR[0] - 30, FAR[1] - 26, 8, E.ERR, 4);
    var farLab = E.halo(E.txt(top, FAR[0], FAR[1] + 42, '够不到', 20, E.ERR, 'middle', 900));
    var upLab = E.halo(E.txt(top, P(UP[2][0], UP[2][1])[0] - 18, P(UP[2][0], UP[2][1])[1] - 16, '肘上', 22, E.UP, 'end', 900));
    var dnLab = E.halo(E.txt(top, P(DN[2][0], DN[2][1])[0] - 18, P(DN[2][0], DN[2][1])[1] + 30, '肘下', 22, E.DOWN, 'end', 900));

    /* ── 右：上下两个框 + 两支箭头 ── */
    var R = E.el('g', {}, s), CX = RX + 160;
    var top11 = E.txt(R, CX, 48, '第 11 集', 22, E.INK2, 'middle', 700);
    function boxAt(y, l1, c1, l2, c2) {
      var g = E.el('g', {}, R);
      K.paint(E.el('rect', { x: RX + 18, y: y, width: 284, height: 70, rx: 14, 'stroke-width': 3 }, g), E.tint(c1, 12, 'var(--demo-surface)'), c1);
      E.txt(g, CX, y + 32, l1, 26, c1, 'middle', 900);
      E.txt(g, CX, y + 58, l2, 19, c2 || E.INK2, 'middle', 700);
      return g;
    }
    var bA = boxAt(66, '每个关节转多少', E.OMEGA, 'θ₁、θ₂、……');
    var bB = boxAt(268, '手的位姿', E.TGT, '停在什么位置、朝向什么方向');
    var down = E.vec(R, E.OMEGA, 4, { head: 15 }), upA = E.vec(R, E.HI, 4, { head: 15 });
    var fkG = E.el('g', {}, R);
    E.txt(fkG, RX + 92, 184, '指数积公式', 20, E.INK2, 'start', 700);
    var fkName = E.txt(fkG, RX + 92, 214, '正向运动学', 24, E.OMEGA, 'start', 900);
    var fkOne = E.txt(fkG, RX + 92, 246, '一个答案', 20, E.GOOD, 'start', 900);
    var ikG = E.el('g', {}, R);
    var ikName = E.txt(ikG, RX + 254, 196, '逆向运动学', 24, E.HI, 'end', 900);
    // 两问分两行、靠右：写成一行会左伸到往下的箭头边上（RX + 70）
    var ikQ = E.txt(ikG, RX + 254, 228, '没有解？', 20, E.INK2, 'end', 700);
    var ikQ2 = E.txt(ikG, RX + 254, 254, '不止一组？', 20, E.INK2, 'end', 700);
    var chips = [['2 组', E.HI, T.two], ['好几组', E.HI, T.many], ['0 组', E.ERR, T.none]].map(function (c, k) {
      var t = E.tag(R, RX + 18 + k * 100, 360, 84, 40, c[0], c[1], { size: 20, weight: 900, ink: c[1] });
      return { g: t.g, at: c[2], x: RX + 60 + k * 100 };
    });

    return {
      el: s,
      draw: function (t) {
        var f = E.fade, b = E.bump;
        /* ── 左：UR5 ── */
        var mv = f(t, T.move[0], T.move[1] - T.move[0]);
        var a1 = UP[0], a2 = (1 - mv) * 0 + mv * UP[1];
        arm.pose(a1, a2);
        arm.view.clearTint();
        ghost.pose(DN[0], DN[1]);
        ghost.view.clearTint(); ghost.view.tint('all', E.DOWN, 0.8, 0.42);
        E.op(ghostG, 0.9 * f(t, T.two + 0.4, 0.5));
        var e = arm.elbow(), w = arm.hand();
        j1.setAttribute('transform', 'translate(' + O[0].toFixed(1) + ' ' + O[1].toFixed(1) + ')');
        j2.setAttribute('transform', 'translate(' + e[0].toFixed(1) + ' ' + e[1].toFixed(1) + ')');
        var jOn = f(t, T.joints, 0.3);
        E.op(j1, jOn); E.op(j2, jOn);
        var glowJ = Math.max(b(t, T.joints, 0.8), b(t, T.each, 0.8));
        [j1, j2].forEach(function (g) { E.glow(g, E.HI, glowJ > 0.05); });
        // 转角：肩从水平转到大臂，肘从大臂的方向转到小臂（画满转过的角度）
        var pa = f(t, T.turn, 0.6);
        arc1.set(0, Math.max(1e-3, a1 * pa));
        var arc2c = arc2; arc2c.g.setAttribute('transform', 'translate(' + e[0].toFixed(1) + ' ' + e[1].toFixed(1) + ')');
        var rel = a2 - a1; arc2c.set(a1, a1 + (Math.abs(rel) < 1e-3 ? -1e-3 : rel * pa));
        var angOn = f(t, T.turn, 0.2) * (1 - f(t, T.ik + 1.0, 0.4));
        E.op(arc1.g, angOn); E.op(arc2c.g, angOn);
        // 标签放在弧外面：θ₁ 往大臂那边偏（目标 (1, 0) 的十字线在弧的右下），θ₂ 离肘远一点（不压在转角弧和箭头上）
        var l1 = P(1.2 * Math.cos(0.72 * a1), 1.2 * Math.sin(0.72 * a1));
        var mid2 = a1 + rel / 2, l2 = [e[0] + 1.2 * u * Math.cos(mid2), e[1] - 1.2 * u * Math.sin(mid2)];
        th1.setAttribute('x', (l1[0] + 4).toFixed(1)); th1.setAttribute('y', (l1[1] + 9).toFixed(1));
        th2.setAttribute('x', l2[0].toFixed(1)); th2.setAttribute('y', (l2[1] + 9).toFixed(1));
        q1.setAttribute('x', (l1[0] + 4).toFixed(1)); q1.setAttribute('y', (l1[1] + 10).toFixed(1));
        q2.setAttribute('x', l2[0].toFixed(1)); q2.setAttribute('y', (l2[1] + 10).toFixed(1));
        var qOn = f(t, T.each, 0.3) * (1 - f(t, T.two, 0.3));
        E.op(th1, f(t, T.turn + 0.3, 0.3) * (1 - qOn) * angOn); E.op(th2, f(t, T.turn + 0.3, 0.3) * (1 - qOn) * angOn);
        var qb = 1 + 0.35 * b(t, T.how, 0.8);
        [q1, q2].forEach(function (n) { E.op(n, qOn); E.scaleAt(n, parseFloat(n.getAttribute('x')), parseFloat(n.getAttribute('y')) - 10, qb); });
        wDot.setAttribute('cx', w[0].toFixed(1)); wDot.setAttribute('cy', w[1].toFixed(1));
        wDot.setAttribute('r', (7 + 5 * b(t, T.hand, 0.9)).toFixed(2));
        E.op(wDot, f(t, T.hand, 0.3));
        E.pop(tgt, TG[0], TG[1], f(t, T.tgt, 0.35), 0.5);
        E.op(tgtLab, f(t, T.where, 0.3));
        E.op(upLab, f(t, T.move[1], 0.3)); E.op(dnLab, f(t, T.two + 0.6, 0.3));
        E.pop(far, FAR[0], FAR[1], f(t, T.none, 0.35), 0.5);
        E.drawOn(farX, f(t, T.none + 0.3, 0.3)); E.op(farLab, f(t, T.none + 0.3, 0.3));

        /* ── 右 ── */
        E.op(top11, f(t, 0.6, 0.4) * (1 - f(t, T.back, 0.4)));
        E.op(bA, f(t, T.poe, 0.35)); E.op(bB, f(t, T.pose, 0.35));
        var dp = f(t, T.calc, 0.5);
        down.set(RX + 70, 140, RX + 70, 140 + 122 * dp);
        E.op(down.g, (dp > 0 ? 1 : 0) * (1 - 0.7 * f(t, T.back, 0.4)));
        E.op(fkG, f(t, T.poe, 0.3) * (1 - f(t, T.back, 0.4)));   // 倒过来以后正向这组整个淡出，不和逆向那组叠在一起
        E.op(fkName, f(t, T.fk, 0.3)); E.op(fkOne, f(t, T.fk + 0.5, 0.3));
        var upp = f(t, T.back + 0.1, 0.5);
        upA.set(RX + 272, 264, RX + 272, 264 - 122 * upp);
        E.op(upA.g, upp > 0 ? 1 : 0);
        E.op(ikG, f(t, T.back + 0.3, 0.3)); E.op(ikName, f(t, T.ik, 0.3)); E.op(ikQ, f(t, T.why, 0.3)); E.op(ikQ2, f(t, T.more, 0.3));
        chips.forEach(function (c) { E.pop(c.g, c.x, 380, f(t, c.at, 0.35), 0.7); });
      }
    };
  };
})();
