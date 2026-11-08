// 013 第 3 幕：两个关节一起转——两份加起来，就是雅可比矩阵乘转速（《Modern Robotics》式 (5.1)(5.2)：v_tip = J1 θ̇1 + J2 θ̇2 = J(θ) θ̇）。
// 左边网格 + UR5（零位），手上留着第 2 幕的两支箭头（肩那一列亮黄 (−2, 3)、肘那一列青 (0, 2)）；右栏前半逐行算、后半写成矩阵乘向量。
// 句 1「肩的转速是一」肩旁蓝色逆时针小弯箭头 θ̇₁ = 1，「肘的转速是负一」肘旁顺时针小弯箭头 θ̇₂ = −1；
// 句 2「两份加起来」「肩的那支箭头」「肘的那支箭头」两支箭头各亮一下；
// 句 3「乘以负一」青箭头翻过来，变成 (0, −2)；句 4「加起来」青箭头挪到亮黄箭头的尖上（头尾相接），紫色的和 (−2, 1) 从手画出来，「往左」「往上」两段分量；
// 句 5「验算」两个关节按 (1, −1) 一起转（只朝紫箭头的方向）：小臂一直水平，肘和手上各一支紫箭头（此刻的速度），一模一样；
// 句 6 停在零位：「从肩关节指向肘」白箭头画到 (1, 2)，「转九十度」绕原点转到 (−2, 1)，「第一列取一份」红 (0, 1)，「第二列取两份」绿 (−2, 0)，
//    「加起来」平移到肘上：和手上的紫箭头平行、一样长；
// 句 7「竖着写」右栏两支箭头各写成一列，「一左一右放在一起」拼成 2 × 2 的矩阵（第一列亮黄、第二列青）；
// 句 8「两个转速」右边写一列 (1, −1)（蓝），「乘以这一列」「第一列取几份」「第二列取几份」逐个框亮，「再加起来」= (−2, 1)；
// 句 9「雅可比」矩阵下面写「雅可比 J」，「第几列」「只让第几个关节」写在下面；
// 句 10「转速翻一倍」转速变成 (2, −2)、结果变成 (−4, 2)，网格上紫箭头拉长一倍；「直接相加」「一次矩阵乘法」写在下面。
// 时刻取自配音的逐词时间，见 ../ep013.py。
(function () {
  var E = window.EP013, K = E.K, H = E.HEX;
  var RX = E.ARM.RX, X = RX + 10, D90 = Math.PI / 2;
  var T = {
    both: 0.7, r1: 2.5, r1v: 3.45, r2: 3.85, r2v: 4.74, cw: 6.25, rev: 7.01, out0: 8.9,
    hv: 9.28, sum: 10.53, a1: 11.75, t1: 13.05, a2: 14.6, t2: 15.93,
    sh: 18.44, c23: 19.36, x1: 20.69, still: 21.5, el: 23.45, c02: 24.32, xm1: 25.32, flip: [25.4, 26.2], c0m2: 26.31,
    add: 28.68, addX: 29.4, resX: 30.38, addY: 31.28, resY: 32.31, slide: [28.7, 29.6], left: 34.0, up: 35.16, out1: 36.9,
    chk: 37.36, shCCW: 38.35, elCW: 39.89, same: 41.5, fore: 42.25, keep: 43.36, flat: 44.27, hand: 45.34, alike: 46.14, prep: [37.25, 38.15], wig: [38.35, 47.6],
    elRot: 48.48, from: 49.66, c12: 51.39, turn: [52.19, 53.1], take1: 53.14, n01: 54.4, take2: 55.2, n20: 56.52, sumE: 57.58, n21: 58.27, moveE: [58.3, 59.1], same2: 59.22, out2: 61.6,
    vert: 61.96, side: 63.8, mat22: 66.0, col1: 67.39, col2: 68.54,
    rates: 70.88, colR: 71.8, hv2: 73.62, eqM: 74.59, timesC: 75.95, ep3: 76.96, mv: 78.2, take1b: 79.79, take2b: 80.86, add2: 82.03,
    name: 84.0, jac: 85.0, nth: 86.67, only: 88.09, per: 89.57, hvel: 91.15,
    dbl: 93.13, dblV: 94.38, sumUp: 96.0, direct: 97.4, once: 99.15
  };
  var END = 102.35;

  E.scenes[2] = {
    title: '一起转：两份加起来，就是雅可比',
    dur: END,
    cues: [
      { at: 0, s: '两个关节一起转：肩的**转速** $1$，肘的**转速** $-1$（每秒顺时针、反着转 $1$ 弧度）' },
      { at: 9.18, s: '**手的速度** = 两份加起来：肩那支 × 肩的**转速** + 肘那支 × 肘的**转速**' },
      { at: 18.34, s: '肩这份 $(-2,\\,3)\\times 1=(-2,\\,3)$；肘这份 $(0,\\,2)\\times(-1)=(0,\\,-2)$' },
      { at: 28.58, s: '加起来：$(-2+0,\\,3+(-2))=(-2,\\,1)$：手每秒往左 $2$ 格、往上 $1$ 格' },
      { at: 37.26, s: '验算：肩逆时针、肘顺时针，一样快——小臂的朝向不变，手和肘走得一模一样' },
      { at: 48.38, s: '肘绕着肩转：$(1,\\,2)$ 转 $90^\\circ$：$1\\times(0,\\,1)+2\\times(-1,\\,0)=(-2,\\,1)$，正好就是**手的速度** ✓' },
      { at: 61.86, s: '两支箭头竖着写，一左一右放在一起：$2\\times 2$ 的矩阵，第 $1$ 列是肩的，第 $2$ 列是肘的' },
      { at: 70.78, s: '**转速**也排成一列：**手的速度** = 矩阵 × 这一列——第 3 集的**按列拿份**' },
      { at: 83.9, s: '这个矩阵就叫**雅可比** $J$：第几列 = 只让第几个关节每秒转 $1$ 弧度时，**手的速度**' },
      { at: 93.03, s: '**转速**翻一倍，**手的速度**也翻一倍；各自那份直接相加——中间只隔着一次矩阵乘法' }
    ],
    build: function (K) {
      var s = K.sceneSvg('两个关节一起转：肩的转速是 1，肘的转速是 −1（每秒顺时针、反着转一弧度）；手的速度是两份加起来：肩的那支箭头乘以肩的转速，肘的那支箭头乘以肘的转速；(−2, 3) × 1 = (−2, 3)，(0, 2) × (−1) = (0, −2)，加起来 (−2, 1)，手每秒往左走 2 格、往上走 1 格。验算：肩逆时针转、肘顺时针转、一样快，小臂的朝向一直不变，手和肘走得一模一样；肘绕着肩转，从肩关节指向肘 (1, 2)，转九十度 1 × (0, 1) + 2 × (−1, 0) = (−2, 1)，正好就是手的速度。把两支箭头竖着写、一左一右放在一起，就是 2 × 2 的矩阵 [[−2, 0], [3, 2]]；两个转速也竖着排成一列 (1, −1)；手的速度等于这个矩阵乘以这一列，第三集的按列拿份；这个矩阵就叫做雅可比，它的第几列就是只让第几个关节每秒转一弧度时手的速度；转速翻一倍，手的速度也翻一倍，几个关节一起转各自那份直接相加，中间只隔着一次矩阵乘法');

      var pn = E.panel(s), P = pn.P, u = pn.u, top = pn.top, O = P(0, 0);
      var jS = E.joint(top, E.HI), jE = E.joint(top, E.HI);
      var arcS = E.arcArrow(top, O[0], O[1], 0.62 * u, E.OMEGA, 3.5, { head: 12 }), arcE = E.arcArrow(top, 0, 0, 0.5 * u, E.OMEGA, 3.5, { head: 11 });
      arcS.set(-0.5, 1.0); arcE.set(1.3, -0.3);
      var thS = E.tex(top, O[0] - 64, O[1] - 30, '\\dot\\theta_1=1', 22, E.OMEGA, 'middle', { w: 90, h: 36 });
      var thE = E.tex(top, 0, 0, '\\dot\\theta_2=-1', 22, E.OMEGA, 'middle', { w: 100, h: 36 });
      var hand = E.dot(top, E.V, 7);
      // 两列（头尾相接）+ 紫色的和
      var c1 = E.gv(top, P, E.J1, 5.5, { head: 17 }), c2 = E.gv(top, P, E.J2, 5.5, { head: 17 }), vv = E.gv(top, P, E.V, 6, { head: 18 });
      var c1L = E.halo(E.txt(top, 0, 0, '(−2, 3)', 20, E.J1, 'start', 900)), c2L = E.halo(E.txt(top, 0, 0, '(0, 2)', 20, E.J2, 'start', 900));
      var vL = E.halo(E.txt(top, 0, 0, '(−2, 1)', 21, E.V, 'end', 900));
      var legU = K.paint(E.el('path', { d: 'M ' + P(3, 2)[0] + ' ' + P(3, 2)[1] + ' V ' + P(3, 3)[1], fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '6 5' }, top), null, E.Y);
      var legL = K.paint(E.el('path', { d: 'M ' + P(3, 3)[0] + ' ' + P(3, 3)[1] + ' H ' + P(1, 3)[0], fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '6 5' }, top), null, E.X);
      // 句 5：肘、手上此刻的速度
      var vE = E.gv(top, P, E.V, 5, { head: 15 }), vH = E.gv(top, P, E.V, 5, { head: 15 });
      var flatL = E.halo(E.txt(top, 0, 0, '一直水平', 18, E.INK, 'middle', 900));
      // 句 6：从肩指向肘、转 90°、红绿两段、平移到肘上
      var dSh = E.gv(top, P, E.INK2, 3, { head: 12, dash: '6 6' });
      var dA = E.gv(top, P, E.INK, 4.5, { head: 15 });
      var red = E.gv(top, P, E.X, 5, { head: 14 }), grn = E.gv(top, P, E.Y, 5, { head: 14 });
      var turnArc = E.arcArrow(top, O[0], O[1], 0.95 * u, E.OMEGA, 3, { head: 11 });
      var vE2 = E.gv(top, P, E.V, 5.5, { head: 16 });
      var eqMark = E.halo(E.txt(top, P(0, 3.6)[0], P(0, 3.6)[1], '一样 ✓', 20, E.GOOD, 'middle', 900));

      /* ══ 右栏 ══ */
      var RK = E.rowKit(), Rg = E.el('g', {}, s);
      function grp() { return E.el('g', {}, Rg); }
      var gA = grp();
      var a0 = RK.line(gA, X, 40, [['③ 两个关节一起转', E.INK, 900]], 24);
      var a1 = E.symRow(gA, RK, X, 80, '\\dot\\theta_1=1', E.OMEGA, 72, [['：肩，逆时针', E.INK2]], 21);
      var a2 = E.symRow(gA, RK, X, 116, '\\dot\\theta_2=-1', E.OMEGA, 88, [['：肘，顺时针（反着转）', E.INK2]], 21);
      var a3 = RK.line(gA, X, 162, [['手的速度', E.V, 900], [' = 两份加起来', E.INK2]], 22);
      var a4 = E.tex(gA, X, 202, '\\textcolor{' + H.hi + '}{\\text{肩那支}}\\times\\textcolor{' + H.acc + '}{\\dot\\theta_1}+\\textcolor{' + H.j2 + '}{\\text{肘那支}}\\times\\textcolor{' + H.acc + '}{\\dot\\theta_2}', 23, E.INK, 'start', { w: 340, h: 40 });
      var a5 = RK.line(gA, X, 250, [['(−2, 3)', E.J1, 900], [' × 1 = ', E.INK2], ['(−2, 3)', E.J1, 900]], 22);
      var a6 = RK.line(gA, X, 286, [['(0, 2)', E.J2, 900], [' × (−1) = ', E.INK2], ['(0, −2)', E.J2, 900]], 22);
      var a7 = RK.line(gA, X, 326, [['加起来：', E.INK2], ['(−2 + 0, 3 + (−2))', E.INK, 900]], 21);
      var a8 = RK.line(gA, X, 362, [['= ', E.INK2], ['(−2, 1)', E.V, 900], ['：往左 2 格、往上 1 格', E.INK2]], 21);
      var gB = grp();
      var b0 = RK.line(gB, X, 40, [['验算', E.INK, 900]], 24);
      var b1 = RK.line(gB, X, 78, [['肩逆时针、肘顺时针，一样快', E.INK2]], 21);
      var b2 = RK.line(gB, X, 112, [['→ 小臂的朝向不变，', E.INK2], ['一直水平', E.INK, 900]], 21);
      var b3 = RK.line(gB, X, 146, [['→ 手和肘，走得', E.INK2], ['一模一样', E.INK, 900]], 21);
      var b4 = RK.line(gB, X, 196, [['肘绕着肩转：', E.INK, 900], ['从肩关节指向肘 ', E.INK2], ['(1, 2)', E.HI, 900]], 20);
      var b5 = RK.line(gB, X, 234, [['转 90°：', E.OMEGA, 900], ['1 × ', E.INK, 900], ['(0, 1)', E.X, 900], [' + ', E.INK2], ['2 × ', E.INK, 900], ['(−1, 0)', E.Y, 900]], 19);
      var b6 = RK.line(gB, X, 270, [['= ', E.INK2], ['(0, 1)', E.X, 900], [' + ', E.INK2], ['(−2, 0)', E.Y, 900], [' = ', E.INK2], ['(−2, 1)', E.V, 900]], 20);
      var b7 = RK.line(gB, X, 312, [['= 手的速度 ✓', E.GOOD, 900]], 22);
      // 后半：矩阵乘向量
      var gC = grp(), MY = 120, fl = E.flow(X + 4, MY, 14);
      var Jm = fl.mat(gC, [[-2, 0], [3, 2]], { size: 28, dx: 52, dy: 40, color: function (i, j) { return j === 0 ? E.J1 : E.J2; } }, 0);
      var Tm = fl.mat(gC, [[1], [-1]], { size: 28, dy: 40, wide: [[-2], [-2]], color: function () { return E.OMEGA; } });
      var eqS = fl.sym(gC, '=', 28, E.INK);
      var Vm = fl.mat(gC, [[-2], [1]], { size: 28, dy: 40, wide: [[-4], [-4]], color: function () { return E.V; } });
      var labJ = E.txt(gC, Jm.pos(1, 0)[0] + 28, MY + 76, '雅可比 J', 21, E.HI, 'middle', 900);
      var labT2 = E.txt(gC, Tm.pos(1, 0)[0], MY + 76, '转速', 19, E.OMEGA, 'middle', 900);
      var labV = E.txt(gC, Vm.pos(1, 0)[0], MY + 76, '手的速度', 19, E.V, 'middle', 900);
      // 句 7：两列先分开竖着写，再并到一起（两个手画的列向量，位置从分开挪到矩阵的两列上）
      var colA = E.colvec(gC, 0, MY, [-2, 3], { size: 28, dy: 40, colors: [E.J1, E.J1], bracket: E.J1 });
      var colB = E.colvec(gC, 0, MY, [0, 2], { size: 28, dy: 40, colors: [E.J2, E.J2], bracket: E.J2 });
      var cA = E.txt(gC, 0, MY + 70, '肩那支', 18, E.J1, 'middle', 900), cB = E.txt(gC, 0, MY + 70, '肘那支', 18, E.J2, 'middle', 900);
      var c1r = RK.line(gC, X, 238, [['按列拿份（第 3 集）：', E.INK2]], 20);
      var c2r = RK.line(gC, X, 272, [['1 × ', E.OMEGA, 900], ['第 1 列', E.J1, 900], [' + ', E.INK2], ['(−1) × ', E.OMEGA, 900], ['第 2 列', E.J2, 900]], 21);
      var d1r = RK.line(gC, X, 238, [['第几列', E.HI, 900], [' = 只让第几个关节', E.INK2]], 21);
      var d2r = RK.line(gC, X, 272, [['每秒转 1 弧度时，', E.INK2], ['手的速度', E.V, 900]], 21);
      var e1r = RK.line(gC, X, 322, [['转速 × 2 → 手的速度 × 2', E.INK, 900]], 21);
      var e2r = RK.line(gC, X, 358, [['各自那份直接相加：一次', E.INK2], ['矩阵乘法', E.HI, 900]], 20);

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }
      function rot(v, a) { return [v[0] * Math.cos(a) - v[1] * Math.sin(a), v[0] * Math.sin(a) + v[1] * Math.cos(a)]; }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) { RK.layout(); laid = true; }
          var f = E.fade, b = E.bump;
          /* ── UR5：句 5 按 (1, −1) 一起转，小臂一直水平。只朝紫箭头的方向转（2026-11-07 用户反馈：原先来回摆，往回摆的半程和箭头方向相反）：
             「验算一下」先把胳膊顺时针挪开 0.3 弧度（这时手上的箭头都已收起），「肩，逆时针转」起一路逆时针转回零位 ── */
          var phi = -0.3 * f(st, T.prep[0], T.prep[1] - T.prep[0]) * (1 - f(st, T.wig[0], T.wig[1] - T.wig[0]));
          pn.pose(E.ZERO[0] + phi, 0);
          pn.view.clearTint();
          var J = pn.joints(), pS = P(0, 0), pE = P(J[1][0], J[1][1]), pH = P(J[2][0], J[2][1]);
          jS.at(pS); jE.at(pE); hand.at(pH);
          arcE.g.setAttribute('transform', 'translate(' + pE[0].toFixed(1) + ' ' + pE[1].toFixed(1) + ')');
          var rOn = 1 - fr(st, T.out1);
          E.op(arcS.g, fr(st, T.r1) * rOn); E.op(thS, fr(st, T.r1v) * (1 - fr(st, T.add)));   // 转速的字用完就收：后面箭头挪动、「一直水平」都会压到
          E.op(arcE.g, fr(st, T.r2) * rOn);
          thE.setAttribute('transform', 'translate(' + (pE[0] + 46).toFixed(1) + ' ' + (pE[1] + 34).toFixed(1) + ')');   // 肘的右下：左上是紫色的 (−2, 1)
          E.op(thE, fr(st, T.r2v) * (1 - fr(st, T.add)));
          E.glow(jS, E.HI, b(st, T.r1, 0.8) > 0.05 || b(st, T.shCCW, 0.8) > 0.05); E.glow(jE, E.HI, b(st, T.r2, 0.8) > 0.05 || b(st, T.elCW, 0.8) > 0.05);

          /* ── 句 1–4：两列、翻转、头尾相接、和 ── */
          var fl2 = f(st, T.flip[0], T.flip[1] - T.flip[0]), k2 = 1 - 2 * fl2;              // 肘那支 × (−1)：2 → −2
          var sl = f(st, T.slide[0], T.slide[1] - T.slide[0]);                                // 挪到亮黄箭头的尖上
          var colOn = 1 - fr(st, T.out1);
          c1.at(3, 2, 1, 5); E.op(c1.g, colOn + E.win(st, T.vert, T.out2 + 50, 0.3) * 0);
          var bx = 3 + (1 - 3) * sl, by = 2 + (5 - 2) * sl;
          c2.at(bx, by, bx, by + 2 * k2); E.op(c2.g, colOn);
          E.glow(c1.line, E.J1, b(st, T.a1, 0.9) > 0.05 || b(st, T.sh, 0.9) > 0.05); E.glow(c2.line, E.J2, b(st, T.a2, 0.9) > 0.05 || b(st, T.el, 0.9) > 0.05);
          c1L.setAttribute('x', (P(1, 5)[0] - 10).toFixed(1)); c1L.setAttribute('y', (P(1, 5)[1] - 14).toFixed(1)); E.op(c1L, colOn * (1 - fr(st, T.add)));
          c2L.textContent = fl2 > 0.5 ? '(0, −2)' : '(0, 2)';
          var c2tip = P(bx, by + 2 * k2);
          c2L.setAttribute('x', (c2tip[0] + 12).toFixed(1)); c2L.setAttribute('y', (c2tip[1] + (k2 < 0 ? 16 : 4)).toFixed(1)); E.op(c2L, colOn * (1 - fr(st, T.slide[0])));   // 挪到亮黄箭头尖上以后不写：和紫色的 (−2, 1) 挤在一起
          var vg = f(st, T.resX, 1.0);
          vv.at(3, 2, 3 - 2 * vg, 2 + 1 * vg); E.op(vv.g, (vg > 0 ? 1 : 0) * colOn);
          vL.setAttribute('x', (P(1, 3)[0] - 12).toFixed(1)); vL.setAttribute('y', (P(1, 3)[1] + 6).toFixed(1)); E.op(vL, fr(st, T.resY) * colOn);
          E.drawOn(legU, f(st, T.up, 0.4)); E.drawOn(legL, f(st, T.left, 0.5));
          if (st > T.out1 - 0.3) { E.op(legU, colOn); E.op(legL, colOn); }

          /* ── 句 5：此刻的速度（按 (1, −1)：肘、手都是 rot90(肘的位置)） ── */
          var on5 = E.win(st, T.same, T.out2 - 1.5, 0.4);
          var ve = rot(J[1], D90);
          vE.at(J[1][0], J[1][1], J[1][0] + ve[0], J[1][1] + ve[1]); vH.at(J[2][0], J[2][1], J[2][0] + ve[0], J[2][1] + ve[1]);
          E.op(vE.g, on5 * fr(st, T.hand)); E.op(vH.g, on5 * fr(st, T.hand));
          flatL.setAttribute('x', ((pE[0] + pH[0]) / 2).toFixed(1)); flatL.setAttribute('y', (pE[1] + 30).toFixed(1)); E.op(flatL, E.win(st, T.flat, T.elRot, 0.3));

          /* ── 句 6：从肩指向肘、转 90°、按列拿份、平移到肘上 ── */
          var on6 = E.win(st, T.from, T.out2, 0.3);
          var grow = f(st, T.from, 0.9), turn = f(st, T.turn[0], T.turn[1] - T.turn[0]), mvE = f(st, T.moveE[0], T.moveE[1] - T.moveE[0]);
          var dv = rot([1, 2], D90 * turn);
          dA.at(0, 0, dv[0] * grow, dv[1] * grow); E.op(dA.g, (grow > 0 ? 1 : 0) * (1 - fr(st, T.n21)) * on6);
          dSh.at(0, 0, 1, 2); E.op(dSh.g, (turn > 0.02 ? 0.7 : 0) * on6 * (1 - fr(st, T.sumE)));
          turnArc.set(Math.atan2(2, 1), Math.atan2(2, 1) + D90 * Math.max(turn, 0.01)); E.op(turnArc.g, (turn > 0.01 ? 1 : 0) * (1 - fr(st, T.take1)) * on6);
          red.at(0, 0, 0, 1 * f(st, T.take1, 0.6)); E.op(red.g, (st > T.take1 ? 1 : 0) * (1 - fr(st, T.sumE)) * on6);
          grn.at(0, 1, -2 * f(st, T.take2, 0.8), 1); E.op(grn.g, (st > T.take2 ? 1 : 0) * (1 - fr(st, T.sumE)) * on6);
          vE2.at(1 * mvE, 2 * mvE, 1 * mvE - 2, 2 * mvE + 1); E.op(vE2.g, fr(st, T.n21) * on6);
          vv.g.style.display = '';
          if (st > T.wig[1]) { vH.at(3, 2, 1, 3); E.op(vH.g, Math.max(on5 * fr(st, T.hand), E.win(st, T.n21, T.out2, 0.3))); }
          E.op(eqMark, E.win(st, T.same2, T.out2, 0.3));

          /* ── 右栏 ── */
          E.op(gA, 1 - fr(st, T.chk - 0.3));
          E.op(a0.g, fr(st, T.both)); E.op(a1.g, fr(st, T.r1)); E.op(a2.g, fr(st, T.r2));
          E.op(a3.g, fr(st, T.hv)); E.op(a3.parts[1], fr(st, T.sum));
          E.op(a4, fr(st, T.a1));
          E.op(a5.g, fr(st, T.sh)); E.op(a5.parts[1], fr(st, T.x1)); E.op(a5.parts[2], fr(st, T.still));
          E.op(a6.g, fr(st, T.el)); E.op(a6.parts[1], fr(st, T.xm1)); E.op(a6.parts[2], fr(st, T.c0m2));
          E.op(a7.g, fr(st, T.add)); E.op(a7.parts[1], fr(st, T.addX));
          E.op(a8.g, fr(st, T.resX)); E.op(a8.parts[2], fr(st, T.left));
          E.op(gB, E.win(st, T.chk - 0.2, T.out2, 0.3));
          E.op(b0.g, fr(st, T.chk)); E.op(b1.g, fr(st, T.shCCW)); E.op(b2.g, fr(st, T.fore)); E.op(b2.parts[1], fr(st, T.keep));
          E.op(b3.g, fr(st, T.hand)); E.op(b3.parts[1], fr(st, T.alike));
          E.op(b4.g, fr(st, T.elRot)); E.op(b4.parts[1], fr(st, T.from)); E.op(b4.parts[2], fr(st, T.c12));
          E.op(b5.g, fr(st, T.turn[0])); E.op(b5.parts[1], fr(st, T.take1)); E.op(b5.parts[2], fr(st, T.take1)); E.op(b5.parts[3], fr(st, T.take2)); E.op(b5.parts[4], fr(st, T.take2)); E.op(b5.parts[5], fr(st, T.take2));
          E.op(b6.g, fr(st, T.n01)); E.op(b6.parts[2], fr(st, T.n20)); E.op(b6.parts[3], fr(st, T.n20)); E.op(b6.parts[4], fr(st, T.n21)); E.op(b6.parts[5], fr(st, T.n21));
          E.op(b7.g, fr(st, T.same2));
          /* 后半：矩阵 */
          E.op(gC, fr(st, T.vert - 0.2));
          var join = f(st, T.side, 0.9);
          var A0 = [X + 40, MY], B0 = [X + 150, MY];
          var ax = A0[0] + (Jm.pos(0, 0)[0] - A0[0]) * join, bx2 = B0[0] + (Jm.pos(0, 1)[0] - B0[0]) * join;
          colA.g.setAttribute('transform', 'translate(' + ax.toFixed(1) + ' 0)'); colB.g.setAttribute('transform', 'translate(' + bx2.toFixed(1) + ' 0)');
          var sep = 1 - fr(st, T.mat22);
          E.op(colA.g, fr(st, T.vert) * sep); E.op(colB.g, fr(st, T.vert + 0.4) * sep);
          cA.setAttribute('x', (ax).toFixed(1)); cB.setAttribute('x', (bx2).toFixed(1)); E.op(cA, fr(st, T.vert) * (1 - fr(st, T.side))); E.op(cB, fr(st, T.vert + 0.4) * (1 - fr(st, T.side)));
          E.op(Jm.g, fr(st, T.mat22));
          Jm.hl('j1', 0, 0, 1, 0, E.J1, E.win(st, T.col1, T.col2, 0.3) + E.win(st, T.take1b, T.take2b, 0.3) + E.win(st, T.nth, T.dbl, 0.3) * 0.6);
          Jm.hl('j2', 0, 1, 1, 1, E.J2, E.win(st, T.col2, T.rates, 0.3) + E.win(st, T.take2b, T.add2, 0.3));
          E.op(Tm.g, fr(st, T.colR)); E.op(labT2, fr(st, T.colR));
          Tm.hl('t1', 0, 0, 0, 0, E.OMEGA, E.win(st, T.take1b, T.take2b, 0.3)); Tm.hl('t2', 1, 0, 1, 0, E.OMEGA, E.win(st, T.take2b, T.add2, 0.3));
          E.op(eqS, fr(st, T.eqM)); E.op(Vm.g, fr(st, T.add2)); E.op(labV, fr(st, T.add2));
          Vm.hl('v', 0, 0, 1, 0, E.V, E.win(st, T.add2, T.name, 0.3) + E.win(st, T.dblV, T.sumUp, 0.3));
          E.op(labJ, fr(st, T.jac)); E.glow(labJ, E.HI, b(st, T.jac, 1.0) > 0.05);
          // 句 10：转速翻一倍
          var dbl = st >= T.dbl + 0.2, dblV = st >= T.dblV + 0.2;
          Tm.set(dbl ? [[2], [-2]] : [[1], [-1]]); Vm.set(dblV ? [[-4], [2]] : [[-2], [1]]);
          Tm.hl('t3', 0, 0, 1, 0, E.OMEGA, E.win(st, T.dbl, T.sumUp, 0.3));
          E.op(c1r.g, E.win(st, T.ep3, T.name, 0.3)); E.op(c2r.g, E.win(st, T.take1b, T.name, 0.3)); E.op(c2r.parts[2], fr(st, T.take2b)); E.op(c2r.parts[3], fr(st, T.take2b)); E.op(c2r.parts[4], fr(st, T.take2b));
          E.op(d1r.g, fr(st, T.nth)); E.op(d1r.parts[1], fr(st, T.only)); E.op(d2r.g, fr(st, T.per)); E.op(d2r.parts[1], fr(st, T.hvel));
          E.op(e1r.g, fr(st, T.dbl)); E.op(e2r.g, fr(st, T.direct)); E.op(e2r.parts[1], fr(st, T.once));
          // 左边：后半留着零位的手和两支箭头（头尾相接）+ 紫色的和；翻一倍时只拉长紫箭头
          var on7 = fr(st, T.vert);
          if (st > T.out2) {
            c1.at(3, 2, 1, 5); c2.at(1, 5, 1, 3); E.op(c1.g, on7 * (1 - fr(st, T.dbl))); E.op(c2.g, on7 * (1 - fr(st, T.dbl)));
            c2L.textContent = '(0, −2)'; c2L.setAttribute('x', (P(1, 3)[0] + 12).toFixed(1)); c2L.setAttribute('y', (P(1, 3)[1] + 16).toFixed(1));
            E.op(c1L, on7 * (1 - fr(st, T.dbl))); E.op(c2L, 0);
            c1L.setAttribute('x', (P(1, 5)[0] - 10).toFixed(1)); c1L.setAttribute('y', (P(1, 5)[1] - 14).toFixed(1));
            var g2 = f(st, T.dblV, 0.8);
            vv.at(3, 2, 3 - 2 * (1 + g2), 2 + (1 + g2)); E.op(vv.g, fr(st, T.add2));
            vL.textContent = g2 > 0.5 ? '(−4, 2)' : '(−2, 1)';
            var vt = P(3 - 2 * (1 + g2), 2 + (1 + g2));
            vL.setAttribute('x', (vt[0] + 4).toFixed(1)); vL.setAttribute('y', (vt[1] - 14).toFixed(1)); vL.style.textAnchor = 'middle';
            E.op(vL, fr(st, T.add2));
            E.op(vH.g, E.win(st, T.n21, T.out2, 0.3));   // 接着淡出（原先在这里一下清成 0）
          } else vL.style.textAnchor = 'end';
        }
      };
    }
  };
})();
