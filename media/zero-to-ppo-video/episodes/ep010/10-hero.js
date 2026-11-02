// 片头主图（800×420）：四句旁白各一段——
// ⓪（2026-11-02 用户反馈「看完没理解矩阵指数的作用、为什么要学」：先从实际问题讲起）「抓杯子」右上出杯子，「六个关节各转一个角度」
//    肩、肘、腕真的转一下、手伸向杯子，「手会停在什么位置、朝向什么方向」手上打问号、右栏写问题；
//    「每个关节，都是绕一根螺旋轴转」三个关节依次亮一圈，手臂回到原样；
// ① 左边第 9 集的侧视网格 + UR5（肘在 (1, 2)，肘上 ⊙），「运动旋量」「转速」「螺旋轴」右边写「运动旋量 = 转速 × 螺旋轴」，
//    「肘关节的螺旋轴」右边出 S 的六维列向量，「零、零、一、二、负一、零」六个数按念到的顺序弹出；
// ② 9.8 起右边换成问句：「绕着螺旋轴」肘亮一圈，「转了一个角度」小臂绕肘转 90°、标 θ，「停在什么位置」小臂上离肘两格的点旁打问号，「朝向什么方向」那里出两根小箭头；
// ③ 「矩阵指数」右边换成一条竖着的流程：[S] θ →（exp 矩阵指数）→ 4 × 4 的位姿 T，按念到的词依次出来，「位姿」小臂上的两根小箭头鼓一下。
// 2026-11-01 按 005–009 的新配色复查：小臂淡黄，转动和 θ 蓝，小臂上的两根小箭头 x 红、y 绿，位姿里 R 蓝、p 紫。
// 时刻取自配音的逐词时间（../ep010.py 的 [from, to] 等于每句配音实长，分镜时间 = 念到这个词的时刻）。
(function () {
  var E = window.EP010, K = E.K, H = E.HEX;
  var D = Math.PI / 180, A = { OX: 84, OY: 345, S: 2.2 }, RX = 450;   // 主图不缩放：UR5 放大到一格约 60 px
  var T = {
    cup: 1.56, six: 2.77, wig: [3.5, 4.6], hand: 4.44, pos0: 5.33, dir0: 5.84, back0: [7.2, 8.1], out0: 7.6,
    each: 7.87, axes: [9.11, 9.5, 9.9],
    twist: 11.93, speed: 13.12, screw: 14.06, elbow: 14.85, nums: [16.56, 17.04, 17.51, 18.02, 18.47, 19.21], out1: 20.4,
    ask: 21.46, axis: 22.64, turn: [23.35, 24.2], ang: 23.88, where: 25.68, dir: 26.9, out2: 28.9,
    mexp: 29.77, sth: 31.07, times: 31.61, angle: 32.47, give: 33.1, step: 33.87, mat: 35.23, pose: 36.0
  };

  E.hero = function (K) {
    var s = K.sceneSvg('片头主图：机械臂要去抓杯子，先得知道六个关节各转一个角度，手停在什么位置、朝向什么方向；每个关节都是绕一根螺旋轴转；第九集讲过运动旋量等于转速乘以螺旋轴，肘关节的螺旋轴是 (0, 0, 1, 2, −1, 0)；这一集问：绕着螺旋轴转了一个角度，刚体停在什么位置、朝向什么方向？答案是矩阵指数：把螺旋轴乘上转过的角度交给它，一步得到一个四乘四的位姿');

    /* ── 左：网格 + UR5 ── */
    var under = E.el('g', {}, s);
    var arm = E.armAt(s, A.OX, A.OY, A.S, 420);
    var u = arm.aim(1, 2, 0);
    var P = E.plane(under, { ox: A.OX, oy: A.OY, u: u, x0: -1, x1: 4, y0: -1, y1: 5 }).P;
    var V = arm.view, RIGID = ['forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link'], Q = P(1, 2);
    var top = E.el('g', {}, s);
    var ring = K.paint(E.el('circle', { cx: Q[0], cy: Q[1], r: 12, 'stroke-width': 3.5 }, top), E.S1, E.HI);
    var ringDot = K.paint(E.el('circle', { cx: Q[0], cy: Q[1], r: 4 }, top), E.HI);
    var spin = E.arcArrow(top, Q[0], Q[1], 1.05 * u, E.OMEGA, 3.5, { head: 13 });
    var thLab = E.halo(E.txt(top, 0, 0, 'θ', 28, E.OMEGA, 'start', 900));
    thLab.style.fontFamily = 'KaTeX_Math, serif'; thLab.style.fontStyle = 'italic';
    var ax = [E.vec(top, E.X, 4.5, { head: 14 }), E.vec(top, E.Y, 4.5, { head: 14 })];
    var qm = E.halo(E.txt(top, 0, 0, '?', 34, E.HI, 'middle', 900));
    // ⓪ 杯子、手上的问号、三个关节的亮圈；手臂各关节先记下原样（arm.aim 摆好的姿势）
    var V0 = { lift: V.jointValue('shoulder_lift_joint'), elbow: V.jointValue('elbow_joint'), wrist: V.jointValue('wrist_1_joint') };
    var WIG = { lift: 14 * D, elbow: -30 * D, wrist: 22 * D };
    function pose(w, phi) {   // w：⓪ 里关节转了多少（0…1）；phi：肘关节转的角度
      V.joints({ shoulder_lift_joint: V0.lift + WIG.lift * w, elbow_joint: V0.elbow - phi + WIG.elbow * w, wrist_1_joint: V0.wrist + WIG.wrist * w });
    }
    pose(1, 0); var hand1 = arm.tool(); pose(0, 0);
    var cupG = E.el('g', {}, top), CX0 = hand1[0] + 26, CY0 = hand1[1] - 6;   // 杯子在手转过去那里的右边一点
    K.paint(E.el('path', { d: 'M ' + (CX0 - 16) + ' ' + (CY0 - 22) + ' L ' + (CX0 + 16) + ' ' + (CY0 - 22) + ' L ' + (CX0 + 12) + ' ' + (CY0 + 20) + ' L ' + (CX0 - 12) + ' ' + (CY0 + 20) + ' Z',
      'stroke-width': 3, 'stroke-linejoin': 'round' }, cupG), E.tint(E.INK2, 25, 'var(--demo-surface)'), E.INK);
    K.paint(E.el('path', { d: 'M ' + (CX0 + 15) + ' ' + (CY0 - 12) + ' q 14 2 10 14 q -3 8 -12 8', fill: 'none', 'stroke-width': 3 }, cupG), null, E.INK);
    var handQ = E.halo(E.txt(top, 0, 0, '?', 34, E.HI, 'middle', 900));
    var jRings = ['shoulder', 'elbow', 'wrist'].map(function () { return K.paint(E.el('circle', { r: 15, fill: 'none', 'stroke-width': 3.5 }, top), null, E.WARN); });

    /* ── 右 ⓪ ：要抓杯子 ── */
    var r0 = E.el('g', {}, s);
    E.txt(r0, RX + 160, 80, '要去抓杯子：', 30, E.INK, 'middle', 900);
    var r0a = E.txt(r0, RX + 160, 150, '6 个关节各转一个角度', 26, E.INK2, 'middle', 900);
    var r0b = E.txt(r0, RX + 160, 226, '手停在什么位置？', 34, E.HI, 'middle', 900);
    var r0c = E.txt(r0, RX + 160, 290, '朝向什么方向？', 34, E.HI, 'middle', 900);
    var r0d = E.txt(r0, RX + 160, 362, '每个关节：绕一根螺旋轴转', 24, E.WARN, 'middle', 900);

    /* ── 右 ① ：运动旋量 = 转速 × 螺旋轴，S 的六个数 ── */
    var r1 = E.el('g', {}, s);
    var l1 = E.el('g', {}, r1);
    E.txt(l1, RX + 160, 50, '第 9 集', 22, E.INK2, 'middle', 700);
    E.txt(l1, RX + 160, 92, '运动旋量 = 转速 × 螺旋轴', 28, E.INK, 'middle', 900);
    var col = E.colvec(r1, RX + 130, 262, [0, 0, 1, 2, -1, 0], { size: 30, dy: 38, w: 72, colors: [E.WARN, E.WARN, E.WARN, E.WARN, E.WARN, E.WARN] });
    var sLab = E.txt(r1, RX + 82, 272, 'S =', 34, E.WARN, 'end', 900);
    sLab.style.fontFamily = 'KaTeX_Math, serif'; sLab.style.fontStyle = 'italic';
    var sName = E.txt(r1, RX + 184, 258, '肘关节的', 22, E.WARN, 'start', 900);
    var sName2 = E.txt(r1, RX + 184, 290, '螺旋轴', 22, E.WARN, 'start', 900);

    /* ── 右 ② ：问句 ── */
    var r2 = E.el('g', {}, s);
    E.txt(r2, RX + 160, 130, '绕螺旋轴转了 θ：', 30, E.INK, 'middle', 900);
    var q1 = E.txt(r2, RX + 160, 210, '停在什么位置？', 38, E.HI, 'middle', 900);
    var q2 = E.txt(r2, RX + 160, 280, '朝向什么方向？', 38, E.HI, 'middle', 900);

    /* ── 右 ③ ：[S]θ → exp → T ── */
    var r3 = E.el('g', {}, s), CX = RX + 160;
    var f1 = E.el('g', {}, r3);
    E.tex(f1, CX - 40, 66, '[\\textcolor{' + H.warn + '}{\\mathcal S}]\\,\\textcolor{' + H.acc + '}{\\theta}', 40, E.INK, 'start', { w: 110, h: 62 });
    var f1a = E.txt(f1, CX - 52, 62, '螺旋轴', 24, E.WARN, 'end', 900), f1b = E.txt(f1, CX + 66, 62, '× 角度', 24, E.OMEGA, 'start', 900);
    var a1 = E.vec(r3, E.INK2, 3.5, { head: 13 }), a2 = E.vec(r3, E.INK2, 3.5, { head: 13 });
    var box = E.el('g', {}, r3);
    K.paint(E.el('rect', { x: CX - 150, y: 112, width: 300, height: 84, rx: 16, 'stroke-width': 3.5 }, box), E.tint(E.HI, 12, 'var(--demo-surface)'), E.HI);
    // 「exp = 矩阵指数」连着写（用户反馈：exp 和「矩阵指数」中间空一大段，意义不明）
    E.tex(box, CX - 114, 168, '\\exp', 42, E.HI, 'start', { w: 90, h: 62 });
    E.txt(box, CX - 34, 166, '= 矩阵指数', 32, E.HI, 'start', 900);
    var oneStep = E.txt(r3, CX + 16, 236, '一步', 24, E.GOOD, 'start', 900);
    var tG = E.el('g', {}, r3);
    E.tex(tG, CX - 140, 340, 'T=\\begin{bmatrix}\\textcolor{' + H.acc + '}{R} & \\textcolor{' + H.violet + '}{p}\\\\ 0 & 1\\end{bmatrix}', 42, E.INK, 'start', { w: 220, h: 130 });
    E.txt(tG, CX + 100, 324, '4 × 4', 28, E.INK, 'start', 900);
    var tName2 = E.txt(tG, CX + 100, 364, '的位姿', 28, E.HI, 'start', 900);

    return {
      el: s,
      draw: function (t) {
        var f = E.fade, b = E.bump;
        /* ── 左 ── */
        var phi = 90 * D * f(t, T.turn[0], T.turn[1] - T.turn[0]);
        var w = f(t, T.wig[0], T.wig[1] - T.wig[0]) * (1 - f(t, T.back0[0], T.back0[1] - T.back0[0]));
        pose(w, phi); V.clearTint(); V.tint(RIGID, E.BODY, 0.45);
        // ⓪ 杯子、手上的问号、三个关节亮一圈
        E.op(cupG, f(t, T.cup, 0.3) * (1 - f(t, T.out1 - 0.3, 0.4)));
        var hp = arm.tool(); handQ.setAttribute('x', (hp[0] - 4).toFixed(1)); handQ.setAttribute('y', (hp[1] + 44).toFixed(1));
        E.op(handQ, f(t, T.hand, 0.3) * (1 - f(t, T.back0[0], 0.4)));
        [arm.shoulder(), arm.elbow(), arm.wrist()].forEach(function (q, i) {
          jRings[i].setAttribute('cx', q[0].toFixed(1)); jRings[i].setAttribute('cy', q[1].toFixed(1));
          var jb = b(t, T.axes[i], 1.0); E.op(jRings[i], Math.max(f(t, T.six, 0.3) * (1 - f(t, T.dir0 + 0.4, 0.3)), jb)); E.glow(jRings[i], E.WARN, jb > 0.05);
        });
        E.op(ring, f(t, T.elbow, 0.3)); E.op(ringDot, f(t, T.elbow, 0.3));
        ring.setAttribute('r', (12 + 5 * Math.max(b(t, T.elbow, 0.8), b(t, T.axis, 0.8))).toFixed(2));
        E.glow(ring, E.HI, Math.max(b(t, T.elbow, 0.8), b(t, T.axis, 0.8)) > 0.05);
        // 画满转过的角度：从小臂原来的方向到现在的方向（同 009 第 3 幕的修正）；小臂开始转才出现（用户反馈：一开始就显示多余）
        spin.set(0, Math.max(1e-3, phi)); E.op(spin.g, f(t, T.turn[0], 0.2));
        var tl = P(1 + 1.35 * Math.cos(40 * D), 2 + 1.35 * Math.sin(40 * D));
        thLab.setAttribute('x', tl[0].toFixed(1)); thLab.setAttribute('y', (tl[1] + 8).toFixed(1)); E.op(thLab, f(t, T.ang, 0.3));
        // 小臂上离肘两格的点（转完在 (1, 4)）：问号写在它左边，两根小箭头（沿小臂、垂直小臂）
        var pp = P(1 + 2 * Math.cos(phi), 2 + 2 * Math.sin(phi)), dA = [Math.cos(phi), Math.sin(phi)], dB = [-Math.sin(phi), Math.cos(phi)];
        ax[0].set(pp[0], pp[1], pp[0] + dA[0] * 0.95 * u, pp[1] - dA[1] * 0.95 * u);
        ax[1].set(pp[0], pp[1], pp[0] + dB[0] * 0.95 * u, pp[1] - dB[1] * 0.95 * u);
        var ab = b(t, T.pose, 1.0);
        ax.forEach(function (a) { E.op(a.g, f(t, T.dir, 0.3)); a.line.style.strokeWidth = 4.5 + 2.5 * ab; E.glow(a.g, E.HI, ab > 0.05); });
        qm.setAttribute('x', (pp[0] + 0.6 * u).toFixed(1)); qm.setAttribute('y', (pp[1] + 12).toFixed(1));   // 点的右边：左边是 y 轴和往左的小箭头
        E.op(qm, f(t, T.where, 0.3) * (1 - f(t, T.mat, 0.4)));

        /* ── 右 ── */
        E.op(r0, f(t, 0.3, 0.4) * (1 - f(t, T.twist - 0.4, 0.4)));
        E.op(r0a, f(t, T.six, 0.3)); E.op(r0b, f(t, T.pos0, 0.3)); E.op(r0c, f(t, T.dir0, 0.3)); E.op(r0d, f(t, T.each, 0.3));
        E.op(r1, f(t, T.twist - 0.2, 0.4) * (1 - f(t, T.out1, 0.4)));
        E.op(l1, f(t, T.twist, 0.3));
        E.op(col.g, f(t, T.screw, 0.3)); E.op(sLab, f(t, T.elbow, 0.3)); E.op(sName, f(t, T.elbow + 0.4, 0.3)); E.op(sName2, f(t, T.elbow + 0.4, 0.3));
        col.cells.forEach(function (c, i) { var q = col.pos(i), pc = f(t, T.nums[i], 0.3); E.op(c, pc); E.scaleAt(c, q[0], q[1], 1 + 0.5 * (1 - pc)); });
        E.op(r2, f(t, T.ask, 0.4) * (1 - f(t, T.out2, 0.4)));
        E.op(q1, f(t, T.where, 0.3)); E.op(q2, f(t, T.dir, 0.3));
        E.op(r3, f(t, T.mexp - 0.3, 0.3));
        E.op(f1, f(t, T.sth, 0.3)); E.op(f1a, f(t, T.sth, 0.3)); E.op(f1b, f(t, T.angle, 0.3));
        var g1 = f(t, T.give, 0.4), g2 = f(t, T.step, 0.4);
        a1.set(CX, 84, CX, 84 + 26 * g1); E.op(a1.g, g1 > 0 ? 1 : 0);
        a2.set(CX, 200, CX, 200 + 62 * g2); E.op(a2.g, g2 > 0 ? 1 : 0);
        E.pop(box, CX, 154, f(t, T.mexp, 0.35), 0.88);
        E.op(oneStep, f(t, T.step, 0.3));
        E.pop(tG, CX, 330, f(t, T.mat, 0.35), 0.9);
        E.op(tName2, f(t, T.pose, 0.3));
      }
    };
  };
})();
