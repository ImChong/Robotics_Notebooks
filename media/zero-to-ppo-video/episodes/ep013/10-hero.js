// 片头主图（800×420）：三句旁白各一段——
// ① 左边第 9–12 集的侧视网格 + UR5（第 11 集的零位：肘 (1, 2)、手 (3, 2)），「每个关节」肩、肘 ⊙ 亮，「转到一个角度」两个蓝色转角 θ₁、θ₂，
//    「指数积公式」右上框「第 11 集：关节转到的角度 → 手停在的位置」，「手停在什么位置」手亮紫点、写 (3, 2)；
// ② 「这一集」右边换成这一集的问题：「每个关节转得多快」两个关节来回转、转角换成蓝色小弯箭头 θ̇₁、θ̇₂，「手就走得多快」手上出虚线箭头和问号，
//    「往什么方向走」问号鼓一下；
// ③ 「一个矩阵」「雅可比」右边写出 J × θ̇ = 手的速度（雅可比亮黄，转速蓝，手的速度紫），「排成一列」转速那一列框亮，「乘上它」雅可比框亮，
//    「手的速度」手上的紫箭头画实（按转速 (1, −1)：(−2, 1)，第 3 幕）。
// 时刻取自配音的逐词时间（../ep013.py 的 [from, to] 等于每句配音实长，分镜时间 = 念到这个词的时刻）。
(function () {
  var E = window.EP013, K = E.K, H = E.HEX;
  var A = { OX: 96, OY: 330, S: 1.9 }, RX = 452;
  var T = {
    ep11: 0.7, joints: 2.05, angle: 2.74, poe: 3.92, calc: 5.21, hand: 6.17, where: 6.79,
    now: 8.7, q: 9.73, fast: 10.57, rate: 11.27, hfast: 12.22, dir: 13.7,
    ans: 15.94, mat: 16.73, jac: 17.78, col: 18.71, colEnd: 20.54, mul: 20.98, hv: 21.79, hvEnd: 22.46
  };

  E.hero = function (K) {
    var s = K.sceneSvg('片头主图：第十一集讲过，每个关节转到一个角度，用指数积公式就能算出手停在什么位置 (3, 2)；这一集回答一个问题：每个关节转得多快，手就走得多快、往什么方向走？答案是一个矩阵，叫做雅可比：把每个关节的转速排成一列，乘上它，就是手的速度');

    var pn = E.panel(s, { ox: A.OX, oy: A.OY, S: A.S, x0: -1, x1: 4, y0: -1, y1: 5, box: [0, 0, 440, 420] }), P = pn.P, u = pn.u, top = pn.top, O = P(0, 0);
    var jS = E.joint(top, E.HI), jE = E.joint(top, E.HI), hand = E.dot(top, E.V, 7);
    var ang1 = E.arcArrow(top, O[0], O[1], 0.62 * u, E.OMEGA, 3.5, { head: 12 }), ang2 = E.arcArrow(top, 0, 0, 0.5 * u, E.OMEGA, 3.5, { head: 11 });
    function mathLab(sx) { var n = E.halo(E.txt(top, 0, 0, sx, 24, E.OMEGA, 'middle', 900)); n.style.fontFamily = 'KaTeX_Math, serif'; n.style.fontStyle = 'italic'; return n; }
    var th1 = mathLab('θ₁'), th2 = mathLab('θ₂');
    var rt1 = E.tex(top, O[0] - 40, O[1] - 30, '\\dot\\theta_1', 24, E.OMEGA, 'middle', { w: 50, h: 40 }), rt2 = E.tex(top, 0, 0, '\\dot\\theta_2', 24, E.OMEGA, 'middle', { w: 50, h: 40 });
    var hLab = E.halo(E.txt(top, P(3, 2)[0] + 14, P(3, 2)[1] + 34, '(3, 2)', 21, E.V, 'start', 900));
    var qV = E.vec(top, E.V, 4.5, { head: 15, dash: '8 6' }), qM = E.halo(E.txt(top, 0, 0, '?', 34, E.V, 'middle', 900));
    var vv = E.gv(top, P, E.V, 6, { head: 18 });

    /* ── 右：上面一个框（第 11 集），下面这一集的问题和答案 ── */
    var R = E.el('g', {}, s), CX = RX + 165;
    function boxAt(y, h, c) {
      var g = E.el('g', {}, R);
      K.paint(E.el('rect', { x: RX + 6, y: y, width: 318, height: h, rx: 14, 'stroke-width': 3 }, g), E.tint(c, 10, 'var(--demo-surface)'), c);
      return g;
    }
    var bA = boxAt(18, 100, E.ACC);
    E.txt(bA, CX, 50, '第 11 集：指数积公式', 20, E.INK2, 'middle', 700);
    E.txt(bA, CX, 92, '关节转到的角度 → 手停在的位置', 21, E.INK, 'middle', 900);
    var bQ = E.el('g', {}, R);
    E.txt(bQ, CX, 166, '这一集：', 20, E.INK2, 'middle', 700);
    var q1 = E.txt(bQ, CX, 204, '关节转得多快 → 手走得多快？', 23, E.HI, 'middle', 900);
    var q2 = E.txt(bQ, CX, 238, '往什么方向走？', 21, E.V, 'middle', 900);
    // 答案：J × θ̇ = 手的速度（手画的矩阵，数照第 3 幕：[[−2, 0], [3, 2]] × (1, −1) = (−2, 1)）
    var gAns = E.el('g', {}, R), fl = E.flow(RX + 22, 320, 12);
    var Jm = fl.mat(gAns, [[-2, 0], [3, 2]], { size: 26, dx: 48, dy: 36, color: function (i, j) { return j === 0 ? E.J1 : E.J2; } }, 0);
    var Tm = fl.mat(gAns, [[1], [-1]], { size: 26, dy: 36, color: function () { return E.OMEGA; } });
    var eqS = fl.sym(gAns, '=', 26, E.INK);
    var Vm = fl.mat(gAns, [[-2], [1]], { size: 26, dy: 36, color: function () { return E.V; } });
    var lJ = E.txt(gAns, Jm.pos(1, 0)[0] + 24, 392, '雅可比', 21, E.HI, 'middle', 900);
    var lT = E.txt(gAns, Tm.pos(1, 0)[0], 392, '转速', 19, E.OMEGA, 'middle', 900);
    var lV = E.txt(gAns, Vm.pos(1, 0)[0], 392, '手的速度', 19, E.V, 'middle', 900);

    return {
      el: s,
      draw: function (t) {
        var f = E.fade, b = E.bump;
        /* ── 左：UR5 ── */
        var wig = E.win(t, T.rate, T.ans, 0.5), w1 = 0.3 * Math.sin(t * 2.0) * wig, w2 = 0.32 * Math.sin(t * 1.6 + 1) * wig;
        var tr = E.win(t, T.hv, 30, 0.5), ph = 0.3 * Math.sin((t - T.hv) * 1.8) * tr;   // 答案：按 (1, −1) 转，小臂一直水平
        pn.pose(E.ZERO[0] + w1 + ph, E.ZERO[1] + w1 + w2);
        pn.view.clearTint();
        var J = pn.joints(), pE = P(J[1][0], J[1][1]), pH = P(J[2][0], J[2][1]);
        jS.at(O); jE.at(pE); hand.at(pH, 7 + 4 * b(t, T.hand, 0.8));
        E.op(jS, f(t, 0.2, 0.4)); E.op(jE, f(t, 0.2, 0.4));
        var gj = b(t, T.joints, 0.8) > 0.05 || b(t, T.fast, 0.8) > 0.05;
        E.glow(jS, E.HI, gj); E.glow(jE, E.HI, gj);
        E.op(hand, f(t, T.hand, 0.3));
        // ① 转角
        var angOn = f(t, T.angle, 0.4) * (1 - f(t, T.now, 0.3));
        ang1.set(0, Math.max(1e-3, E.ZERO[0] * f(t, T.angle, 0.6)));
        ang2.g.setAttribute('transform', 'translate(' + pE[0].toFixed(1) + ' ' + pE[1].toFixed(1) + ')');
        ang2.set(E.ZERO[0], E.ZERO[0] - Math.max(1e-3, E.ZERO[0] * f(t, T.angle, 0.6)));
        E.op(ang1.g, angOn); E.op(ang2.g, angOn);
        var l1 = P(1.25 * Math.cos(0.5 * E.ZERO[0]), 1.25 * Math.sin(0.5 * E.ZERO[0]));
        th1.setAttribute('x', (l1[0] + 6).toFixed(1)); th1.setAttribute('y', (l1[1] + 8).toFixed(1));
        th2.setAttribute('x', (pE[0] + 0.95 * u).toFixed(1)); th2.setAttribute('y', (pE[1] - 0.45 * u).toFixed(1));
        E.op(th1, angOn); E.op(th2, angOn);
        E.op(hLab, f(t, T.where, 0.3) * (1 - f(t, T.now, 0.3)));
        // ② 转速：两个关节旁的 θ̇
        var rOn = f(t, T.rate, 0.3);
        rt2.setAttribute('transform', 'translate(' + (pE[0] - 52).toFixed(1) + ' ' + (pE[1] - 16).toFixed(1) + ')');   // 肘的左边：上方一格是紫箭头的尖
        E.op(rt1, rOn); E.op(rt2, rOn);
        var qOn = f(t, T.hfast, 0.3) * (1 - f(t, T.hv, 0.3));
        qV.set(pH[0], pH[1], pH[0] - 1.0 * u, pH[1] - 1.4 * u); E.op(qV.g, qOn);
        qM.setAttribute('x', (pH[0] - 1.3 * u).toFixed(1)); qM.setAttribute('y', (pH[1] - 1.6 * u).toFixed(1));
        E.op(qM, qOn * (0.65 + 0.35 * b(t, T.dir, 0.8)));
        // ③ 手的速度 (−2, 1)
        var vg = f(t, T.hv, 0.6), ve = [-2 * vg, 1 * vg];
        vv.at(J[2][0], J[2][1], J[2][0] + ve[0], J[2][1] + ve[1]); E.op(vv.g, vg > 0 ? 1 : 0);

        /* ── 右 ── */
        E.op(bA, f(t, T.ep11, 0.4) * (1 - f(t, T.ans - 0.2, 0.4) * 0.6));
        E.glow(bA, E.ACC, b(t, T.poe, 1.0) > 0.05);
        E.op(bQ, f(t, T.now, 0.4)); E.op(q1, f(t, T.fast, 0.3)); E.op(q2, f(t, T.dir, 0.3));
        E.op(gAns, f(t, T.ans, 0.3));
        E.op(Jm.g, f(t, T.mat, 0.3)); E.op(lJ, f(t, T.jac, 0.3)); E.glow(lJ, E.HI, b(t, T.jac, 1.0) > 0.05);
        E.op(Tm.g, f(t, T.col, 0.3)); E.op(lT, f(t, T.col, 0.3));
        Tm.hl('c', 0, 0, 1, 0, E.OMEGA, E.win(t, T.col, T.mul, 0.3));
        Jm.hl('j', 0, 0, 1, 1, E.HI, E.win(t, T.mul, T.hv, 0.3) * 0.7);
        E.op(eqS, f(t, T.mul, 0.3)); E.op(Vm.g, f(t, T.hv, 0.3)); E.op(lV, f(t, T.hv, 0.3));
        Vm.hl('v', 0, 0, 1, 0, E.V, E.win(t, T.hv, 30, 0.3) * 0.7);
      }
    };
  };
})();
