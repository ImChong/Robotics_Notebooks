// 片头主图（800×420）：三句旁白各一段——
// ① 左边斜着看的 UR5（零位：胳膊平伸），「螺旋轴」肘的那根轴（S₃）亮，「转一个角度」小臂绕它往上抬一下又回来（往下会碰到桌面那一侧）；右边第 10 集的流程：螺旋轴 × 角度 → exp → 一个位姿；
// ② 「六个关节」六根关节轴依次出来，「各转一个角度」六个关节一起转到另一个姿势，「手」手上出三根小箭头、打问号；右边换成问句；
// ③ 「正向运动学」右边写 θ₁…θ₆ → 正向运动学 → 手的位姿；「指数积公式」出框，「每个关节一个矩阵指数」六块依次弹出，「按顺序乘起来」整串亮，「所有关节都在零度时、手的位姿」M 出来（片头还没讲零位）；「第七集提过」正向运动学下面写一行小字。
// 配色照「画面配色」：螺旋轴琥珀，角度蓝，位姿亮黄，零位绿；手的三根小箭头 x 红、y 绿、z 蓝。时刻取自配音的逐词时间（../ep011.py）。
(function () {
  var E = window.EP011, K = E.K, H = E.HEX;
  var D = Math.PI / 180, RX = 450, CX = RX + 160;
  var T = {
    ep10: 0.7, axis: 2.44, turn: [3.21, 4.2], ang: 3.67, give: 4.33, mexp: 4.73, step: 6.07, pose: 7.28, out1: 8.6,
    ask: 9.41, arm: 10.67, six: 11.44, each: [12.33, 13.59], hand: 13.66, where: 14.79, dir: 15.37, out2: 17.38,
    joints: 17.98, get: 19.03, pose3: 19.68, ep7: 20.28, fk: 21.88, poe: 24.09, facs: [25.21, 25.52, 25.83, 26.14, 26.45, 26.76], seq: 27.24, mul: 27.84, M: 29.08, hand3: 30.96
  };
  var LAB = [[1, 0, -12, 'middle'], [0, -18, 6, 'end'], [0, -18, 6, 'end'], [1, 6, -12, 'start'], [0, 14, 10, 'start'], [1, 14, 6, 'start']];
  var LEN = [[-0.05, 0.32], [-0.2, 0.2], [-0.18, 0.18], [-0.16, 0.16], [-0.15, 0.15], [-0.14, 0.14]];
  var POSE = { shoulder_pan_joint: 30 * D, shoulder_lift_joint: -70 * D, elbow_joint: 80 * D, wrist_1_joint: -100 * D, wrist_2_joint: -90 * D, wrist_3_joint: 20 * D };

  E.hero = function (K) {
    var s = K.sceneSvg('片头主图：第十集讲过，绕着一根螺旋轴转一个角度，交给矩阵指数，一步得到一个位姿。这一集问：机械臂的六个关节各转一个角度，手最后停在什么位置、朝向什么方向？从关节的角度得到手的位姿，叫正向运动学；答案是指数积公式 T = exp([S₁]θ₁) ⋯ exp([S₆]θ₆) M：每个关节一个矩阵指数，按顺序乘起来，再乘上所有关节都在零度时手的位姿');

    /* ── 左：UR5 ── */
    var g3 = E.el('g', {}, s);
    var R6 = E.ur6(g3, 96, 330, 2.6, 32, 16, [0, 0, 440, 420]), V = R6.view;   // 底座放高：零位时手腕往下垂，原先 388 时被视景框底边切掉（用户反馈「开幕的机器人出界了」）
    var qm = E.halo(E.txt(g3, 0, 0, '?', 36, E.HI, 'middle', 900));

    /* ── 右 ①：第 10 集：螺旋轴 × 角度 → exp → 位姿 ── */
    var r1 = E.el('g', {}, s);
    E.txt(r1, CX, 40, '第 10 集', 22, E.INK2, 'middle', 700);
    var f1 = E.el('g', {}, r1);
    E.txt(f1, CX - 8, 92, '螺旋轴', 28, E.WARN, 'end', 900);
    var f1b = E.txt(f1, CX + 8, 92, '× 角度', 28, E.OMEGA, 'start', 900);
    var a1 = E.vec(r1, E.INK2, 3.5, { head: 13 }), a2 = E.vec(r1, E.INK2, 3.5, { head: 13 });
    var box = E.el('g', {}, r1);
    K.paint(E.el('rect', { x: CX - 150, y: 136, width: 300, height: 78, rx: 16, 'stroke-width': 3.5 }, box), E.tint(E.HI, 12, 'var(--demo-surface)'), E.HI);
    E.tex(box, CX - 114, 188, '\\exp', 40, E.HI, 'start', { w: 90, h: 60 });
    E.txt(box, CX - 34, 186, '= 矩阵指数', 30, E.HI, 'start', 900);
    var oneStep = E.txt(r1, CX + 16, 256, '一步', 22, E.GOOD, 'start', 900);
    var poseT = E.txt(r1, CX, 330, '一个位姿', 34, E.HI, 'middle', 900);

    /* ── 右 ②：问句 ── */
    var r2 = E.el('g', {}, s);
    E.txt(r2, CX, 110, '6 个关节，各转一个角度：', 28, E.INK, 'middle', 900);
    var q1 = E.txt(r2, CX, 200, '手停在什么位置？', 36, E.HI, 'middle', 900);
    var q2 = E.txt(r2, CX, 270, '朝向什么方向？', 36, E.HI, 'middle', 900);

    /* ── 右 ③：正向运动学、指数积公式 ── */
    var r3 = E.el('g', {}, s);
    var fkRow = E.el('g', {}, r3);
    E.tex(fkRow, RX - 6, 52, '\\textcolor{' + H.acc + '}{\\theta_1\\cdots\\theta_6}', 28, E.INK, 'start', { w: 96, h: 50 });
    var fa = E.vec(fkRow, E.INK2, 3.5, { head: 12 }); fa.set(RX + 92, 44, RX + 122, 44);
    var fkName = E.txt(fkRow, RX + 130, 54, '正向运动学', 26, E.HI, 'start', 900);
    var fkEp = E.txt(fkRow, RX + 195, 82, '（第 7 集提过）', 16, E.MUT, 'middle', 700);
    var fb = E.vec(fkRow, E.INK2, 3.5, { head: 12 }); fb.set(RX + 266, 44, RX + 296, 44);
    var fkOut = E.txt(fkRow, RX + 302, 54, '位姿', 26, E.HI, 'start', 900);
    var poeBox = E.el('g', {}, r3);
    K.paint(E.el('rect', { x: CX - 140, y: 96, width: 280, height: 56, rx: 14, 'stroke-width': 3 }, poeBox), E.tint(E.HI, 12, 'var(--demo-surface)'), E.HI);
    E.txt(poeBox, CX, 134, '指数积公式（PoE）', 28, E.HI, 'middle', 900);
    var FY = [212, 270, 328], FX = [[RX + 50, 0], [RX + 168, 0], [RX + 0, 1], [RX + 118, 1], [RX + 236, 1], [RX + 0, 2]];
    var tEq = E.tex(r3, RX, FY[0], 'T=', 26, E.INK, 'start', { w: 44, h: 44 });
    var facs = FX.map(function (q, i) {
      var g = E.el('g', {}, r3), y = FY[q[1]];
      var bx = K.paint(E.el('rect', { x: q[0] - 6, y: y - 28, width: 112, height: 40, rx: 9, 'stroke-width': 2.5 }, g), E.tint(E.WARN, 14, 'var(--demo-surface)'), E.WARN);
      E.tex(g, q[0], y, '\\exp([\\mathcal S_' + (i + 1) + ']\\,\\theta_' + (i + 1) + ')', 19, E.INK, 'start', { w: 104, h: 44 });
      return { g: g, bx: bx, x: q[0], y: y };
    });
    var mG = E.el('g', {}, r3);
    K.paint(E.el('rect', { x: RX + 112, y: FY[2] - 28, width: 40, height: 40, rx: 9, 'stroke-width': 2.5 }, mG), E.tint(E.GOOD, 14, 'var(--demo-surface)'), E.GOOD);
    E.tex(mG, RX + 120, FY[2], 'M', 24, E.INK, 'start', { w: 30, h: 44 });
    // 2026-11-04（观众零基础）：片头还没讲「零位」（第 1 幕才定义），M 旁照旁白写「关节都在 0° 时、手的位姿」，分两行
    var mNote = E.el('g', {}, mG);
    E.txt(mNote, RX + 164, FY[2] - 10, '关节都在 0° 时', 20, E.GOOD, 'start', 900);
    E.txt(mNote, RX + 164, FY[2] + 16, '手的位姿', 20, E.GOOD, 'start', 900);
    var seqL = E.txt(r3, CX, 392, '每个关节一个矩阵指数，按顺序乘起来', 21, E.INK, 'middle', 900);

    return {
      el: s,
      draw: function (t) {
        var f = E.fade, b = E.bump;
        /* ── 左 ── */
        var wig = -35 * D * Math.sin(Math.PI * E.clamp((t - T.turn[0]) / (T.turn[1] - T.turn[0]), 0, 1));
        var p = f(t, T.each[0], T.each[1] - T.each[0]), pose = {};
        E.JN6.forEach(function (n) { pose[n] = POSE[n] * p; });
        pose.elbow_joint += wig;
        V.joints(pose);
        R6.update(LAB, LEN);
        R6.lines.forEach(function (ln, i) {
          var on = i === 2 ? Math.max(f(t, T.axis, 0.3) * (1 - f(t, T.out1, 0.3)), f(t, T.six + 0.3 * i, 0.3)) : f(t, T.six + 0.3 * i, 0.3);
          var pulse = Math.max(i === 2 ? b(t, T.axis, 0.8) : 0, b(t, T.six + 0.3 * i, 0.6), b(t, T.facs[i], 0.6));
          E.op(ln, on); E.op(R6.labs[i], on);
          ln.style.strokeWidth = 3.5 + 3 * pulse; E.glow(ln, E.WARN, pulse > 0.05);
        });
        var hb = Math.max(b(t, T.hand, 0.9), b(t, T.hand3, 0.9));
        E.op(R6.handG, f(t, T.hand, 0.3)); E.glow(R6.handG, E.HI, hb > 0.05);
        var ho = V.project(R6.handAt().o);
        qm.setAttribute('x', (ho[0] + 30).toFixed(1)); qm.setAttribute('y', (ho[1] - 14).toFixed(1));
        E.op(qm, f(t, T.where, 0.3) * (1 - f(t, T.poe, 0.4)));

        /* ── 右 ── */
        E.op(r1, f(t, T.ep10, 0.4) * (1 - f(t, T.out1, 0.4)));
        E.op(f1, f(t, T.axis, 0.3)); E.op(f1b, f(t, T.ang, 0.3));
        var g1 = f(t, T.give, 0.4), g2 = f(t, T.step, 0.4);
        a1.set(CX, 106, CX, 106 + 26 * g1); E.op(a1.g, g1 > 0 ? 1 : 0);
        a2.set(CX, 218, CX, 218 + 70 * g2); E.op(a2.g, g2 > 0 ? 1 : 0);
        E.pop(box, CX, 175, f(t, T.mexp, 0.35), 0.88);
        E.op(oneStep, f(t, T.step, 0.3)); E.pop(poseT, CX, 320, f(t, T.pose, 0.35), 0.9);
        E.op(r2, f(t, T.ask, 0.4) * (1 - f(t, T.out2, 0.4)));
        E.op(q1, f(t, T.where, 0.3)); E.op(q2, f(t, T.dir, 0.3));
        E.op(r3, f(t, T.joints - 0.2, 0.3));
        E.op(fkRow, 1); E.op(fa.g, f(t, T.get, 0.3)); E.op(fkName, f(t, T.fk, 0.3)); E.op(fkEp, f(t, T.fk, 0.3)); E.op(fb.g, f(t, T.fk + 0.3, 0.3)); E.op(fkOut, f(t, T.pose3, 0.3));
        E.pop(poeBox, CX, 124, f(t, T.poe, 0.35), 0.9);
        facs.forEach(function (F, i) {
          E.pop(F.g, F.x + 50, F.y - 8, f(t, T.facs[i], 0.3), 0.85);
          var m = b(t, T.mul + 0.12 * i, 0.6);
          F.bx.style.strokeWidth = 2.5 + 2 * m; E.glow(F.bx, E.HI, m > 0.05);
        });
        E.op(tEq, f(t, T.facs[0], 0.3)); E.op(mG, f(t, T.M, 0.3)); E.op(mNote, f(t, T.M + 0.3, 0.3));
        E.op(seqL, f(t, T.seq, 0.3));
      }
    };
  };
})();
