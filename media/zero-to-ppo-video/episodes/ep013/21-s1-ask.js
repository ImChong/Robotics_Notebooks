// 013 第 1 幕：关节转多快，手走多快——把问题摆出来。左边第 9–12 集的侧视网格 + UR5（第 11 集的零位：肘在 (1, 2)，小臂水平，手在 (3, 2)），右栏一行行写。
// 句 1「网格」网格淡入，「肩关节在原点」「肘在一、二」⊙ 和坐标，「手」小臂上离肘两格亮紫点、量出 2 格，「三、二」写坐标；
// 句 2「只看肩和肘」两个 ⊙ 亮，「关节转得多快」两个关节来回转，「转速」两个关节旁出蓝色小弯箭头 + θ̇₁、θ̇₂，「逆时针」「顺时针」右栏各配一个小弯箭头写 + / −；
// 句 3「手的速度」「一支箭头」手上出紫色箭头（举例：往右 1 格、往上 2 格），「往右」红色虚线、「往上」绿色虚线分量，「第三集的向量」右栏写；
// 句 4「肩每秒转一些」「肘也每秒转一些」两个关节一起来回转，「手往什么方向走」手上出虚线箭头 + 问号；
// 句 5「先一个一个来」右栏写 ① 只转肩、② 只转肘，「别的关节都不动」。
// 右栏一行分几段出时，「、」「→」「：」这类符号跟它后面那一段一起出（同 012）。时刻取自配音的逐词时间，见 ../ep013.py。
(function () {
  var E = window.EP013, K = E.K, H = E.HEX;
  var RX = E.ARM.RX, X = RX + 10;
  var T = {
    grid: 1.7, sh: 2.4, origin: 3.1, el: 3.76, e12: 4.13, hand: 5.01, ep11: 5.44, fore: 6.43, two: 7.58, c32: 9.03, out1: 10.6,
    only: 10.98, shJ: 11.39, elJ: 11.8, joints: 12.04, fast: 13.31, ep9: 14.64, rate: 16.05, perSec: 16.86, rad: 17.98,
    ccw: 18.92, plus: 20.33, cw: 20.95, minus: 22.55, out2: 23.8,
    hv: 24.17, arrow: 25.25, right: 26.76, rightN: 27.35, up: 27.94, upN: 28.53, ep3: 29.45, vec: 30.19, out3: 31.4,
    q: 31.81, shQ: 32.67, elQ: 34.14, dirQ: 35.7, farQ: 37.11, out4: 38.95,
    one: 39.33, each: 40.62, onlyOne: 41.04, others: 42.39
  };
  var END = 44.64;

  E.scenes[0] = {
    title: '关节转多快，手走多快',
    dur: END,
    cues: [
      { at: 0, s: '第 9 集的网格：肩关节在原点，肘在 (1, 2)；手（第 11 集）在小臂上、离肘 2 格：(3, 2)' },
      { at: 10.88, s: '只看肩、肘 2 个关节。**转速**（第 9 集）：每秒转多少弧度；逆时针记成正的，顺时针记成负的' },
      { at: 24.07, s: '**手的速度**：一支箭头——每秒往右几格、往上几格（第 3 集的向量）' },
      { at: 31.71, s: '肩、肘各有一个**转速**：手往什么方向走、每秒走多远？' },
      { at: 39.23, s: '先一个一个来：每次只转一个关节，别的关节不动' }
    ],
    build: function (K) {
      var s = K.sceneSvg('第九集的网格：肩关节在原点，肘在 (1, 2)；手照第十一集在小臂上离肘两格的地方 (3, 2)。只看肩和肘两个关节；关节转得多快叫转速，每秒转多少弧度，逆时针记成正的、顺时针记成负的。手的速度是一支箭头：每秒往右走几格、往上走几格，就像第三集的向量。问题：肩每秒转一些，肘也每秒转一些，手往什么方向走、每秒走多远？先一个一个来：每次只让一个关节转，别的关节都不动');

      var pn = E.panel(s), P = pn.P, u = pn.u, top = pn.top;
      var O = P(0, 0);
      var jS = E.joint(top, E.HI), jE = E.joint(top, E.HI);
      var arcS = E.arcArrow(top, 0, 0, 0.62 * u, E.OMEGA, 3.5, { head: 12 }), arcE = E.arcArrow(top, 0, 0, 0.5 * u, E.OMEGA, 3.5, { head: 11 });
      var thS = E.tex(top, 0, 0, '\\dot\\theta_1', 24, E.OMEGA, 'middle', { w: 50, h: 40 }), thE = E.tex(top, 0, 0, '\\dot\\theta_2', 24, E.OMEGA, 'middle', { w: 50, h: 40 });
      var hand = E.dot(top, E.V, 7);
      var oLab = E.halo(E.txt(top, O[0] - 14, O[1] + 30, '(0, 0)', 20, E.INK, 'end', 900));
      var Q = P(1, 2), eLab = E.halo(E.txt(top, Q[0] - 18, Q[1] - 16, '(1, 2)', 22, E.HI, 'end', 900));
      var H32 = P(3, 2), hLab = E.halo(E.txt(top, H32[0] + 16, H32[1] + 34, '手 (3, 2)', 21, E.V, 'start', 900));
      var dim = K.paint(E.el('path', { d: 'M ' + Q[0] + ' ' + (Q[1] - 16) + ' L ' + H32[0] + ' ' + (H32[1] - 16) + ' M ' + Q[0] + ' ' + (Q[1] - 23) + ' V ' + (Q[1] - 9) +
        ' M ' + H32[0] + ' ' + (H32[1] - 23) + ' V ' + (H32[1] - 9), fill: 'none', 'stroke-width': 2.5 }, top), null, E.GOOD);
      var dimL = E.halo(E.txt(top, (Q[0] + H32[0]) / 2, Q[1] - 26, '2 格', 20, E.GOOD, 'middle', 900));
      // 句 3：举例的手速度 (1, 2)：往右 1（红虚线）、往上 2（绿虚线），紫箭头
      var exR = K.paint(E.el('path', { d: 'M ' + P(3, 4)[0] + ' ' + P(3, 4)[1] + ' H ' + P(4, 4)[0], fill: 'none', 'stroke-width': 3.5, 'stroke-dasharray': '7 5' }, top), null, E.X);
      var exU = K.paint(E.el('path', { d: 'M ' + H32[0] + ' ' + H32[1] + ' V ' + P(3, 4)[1], fill: 'none', 'stroke-width': 3.5, 'stroke-dasharray': '7 5' }, top), null, E.Y);
      var exV = E.gv(top, P, E.V, 5, { head: 16 });
      var exRL = E.halo(E.txt(top, P(3.5, 4)[0], P(3.5, 4)[1] - 12, '往右 1', 18, E.X, 'middle', 900));
      var exUL = E.halo(E.txt(top, P(3, 3)[0] - 10, P(3, 3)[1] + 6, '往上 2', 18, E.Y, 'end', 900));
      // 句 4：问号箭头
      var qV = E.vec(top, E.V, 4.5, { head: 15, dash: '8 6' });
      var qM = E.halo(E.txt(top, 0, 0, '?', 34, E.V, 'middle', 900));

      /* ══ 右栏 ══ */
      var RK = E.rowKit(), Rg = E.el('g', {}, s);
      function grp() { return E.el('g', {}, Rg); }
      var g1 = grp();
      var a1 = RK.line(g1, X, 60, [['第 9 集的网格', E.INK, 900]], 24);
      var a2 = RK.line(g1, X, 108, [['肩关节', E.HI, 900], ['：原点 ', E.INK2], ['(0, 0)', E.INK, 900]]);
      var a3 = RK.line(g1, X, 148, [['肘', E.HI, 900], ['：', E.INK2], ['(1, 2)', E.HI, 900]]);
      var a4 = RK.line(g1, X, 200, [['手', E.V, 900], ['（第 11 集）', E.MUT, 700], ['：小臂上', E.INK2]], 22);
      var a5 = RK.line(g1, X, 238, [['离肘 ', E.INK2], ['2 格', E.GOOD, 900], [' → ', E.INK2], ['(3, 2)', E.V, 900]], 22);
      var g2 = grp();
      var b1 = RK.line(g2, X, 52, [['只看 ', E.INK2], ['肩', E.HI, 900], ['、', E.INK2], ['肘', E.HI, 900], [' 2 个关节', E.INK2]]);
      var b2 = RK.line(g2, X, 104, [['关节转得多快', E.INK, 900]], 23);
      var b3 = E.symRow(g2, RK, X, 146, '\\dot\\theta', E.OMEGA, 22, [['转速', E.OMEGA, 900], ['（第 9 集）', E.MUT, 700]], 22);
      var b4 = RK.line(g2, X, 186, [['= 每秒转多少弧度', E.INK2]], 22);
      var b5 = RK.line(g2, X, 252, [['逆时针', E.INK, 900], ['：记成正的 ', E.INK2], ['+', E.GOOD, 900]], 22);
      var b6 = RK.line(g2, X, 310, [['顺时针', E.INK, 900], ['：记成负的 ', E.INK2], ['−', E.ERR, 900]], 22);
      var icoP = E.arcArrow(g2, X + 300, 244, 16, E.OMEGA, 3, { head: 9 }), icoM = E.arcArrow(g2, X + 300, 302, 16, E.OMEGA, 3, { head: 9 });
      icoP.set(-1.2, 2.6); icoM.set(2.6, -1.2);
      var g3 = grp();
      var c1 = RK.line(g3, X, 60, [['手的速度', E.V, 900], ['：一支箭头', E.INK2]], 24);
      var c2 = RK.line(g3, X, 112, [['每秒 ', E.INK2], ['往右', E.X, 900], [' 走几格', E.INK2]]);
      var c3 = RK.line(g3, X, 152, [['每秒 ', E.INK2], ['往上', E.Y, 900], [' 走几格', E.INK2]]);
      var c4 = RK.line(g3, X, 200, [['就像第 3 集的', E.INK2], ['向量', E.INK, 900]], 22);
      var c5 = RK.line(g3, X, 236, [['（图上：往右 1 格、往上 2 格）', E.MUT, 700]], 18);
      var g4 = grp();
      var d1 = RK.line(g4, X, 56, [['问题：', E.INK, 900]], 24);
      var d2 = E.symRow(g4, RK, X, 100, '\\dot\\theta_1', E.OMEGA, 34, [['：', E.INK2], ['肩', E.HI, 900], ['每秒转一些', E.INK2]]);
      var d3 = E.symRow(g4, RK, X, 144, '\\dot\\theta_2', E.OMEGA, 34, [['：', E.INK2], ['肘', E.HI, 900], ['也每秒转一些', E.INK2]]);
      var d4 = RK.line(g4, X, 206, [['→ ', E.INK2], ['手', E.V, 900], ['往什么方向走？', E.INK, 900]], 23);
      var d5 = RK.line(g4, X, 246, [['　 每秒走多远？', E.INK, 900]], 23);
      var g5 = grp();
      var e1 = RK.line(g5, X, 60, [['先一个一个来：', E.INK, 900]], 22);
      var e2 = RK.line(g5, X, 108, [['① 只转', E.INK2], ['肩', E.HI, 900], ['　② 只转', E.INK2], ['肘', E.HI, 900]], 21);
      var e3 = RK.line(g5, X, 148, [['别的关节都不动', E.INK2]], 21);

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) { RK.layout(); laid = true; }
          var f = E.fade, b = E.bump;
          /* ── UR5：句 2「关节转得多快」两个关节来回转，句 4 再转；别的时候停在零位 ── */
          var wig = Math.max(E.win(st, T.fast, T.ep9 + 1.2, 0.6), E.win(st, T.shQ, T.out4, 0.6));
          var w1 = 0.28 * Math.sin(t * 2.0) * wig, w2 = 0.32 * Math.sin(t * 1.6 + 1) * wig;
          pn.pose(E.ZERO[0] + w1, E.ZERO[1] + w1 + w2);
          pn.view.clearTint();
          var J = pn.joints(), pS = P(0, 0), pE = P(J[1][0], J[1][1]), pH = P(J[2][0], J[2][1]);
          E.op(pn.under, f(st, T.grid, 0.5));
          jS.at(pS); jE.at(pE);
          E.op(jS, fr(st, T.sh)); E.op(jE, fr(st, T.el));
          E.glow(jS, E.HI, b(st, T.shJ, 0.8) > 0.05); E.glow(jE, E.HI, b(st, T.elJ, 0.8) > 0.05);
          E.op(oLab, fr(st, T.origin) * (1 - fr(st, T.out1))); E.op(eLab, fr(st, T.e12) * (1 - fr(st, T.out1)));
          hand.at(pH, 7 + 4 * Math.max(b(st, T.hand, 0.8), b(st, T.hv, 0.8)));
          E.op(hand, fr(st, T.hand));
          E.drawOn(dim, f(st, T.two, 0.5)); E.op(dimL, fr(st, T.two) * (1 - fr(st, T.out1))); if (st > T.out1) E.op(dim, 1 - fr(st, T.out1));
          E.op(hLab, fr(st, T.c32) * (1 - fr(st, T.out1)));
          // 转速：两个关节旁的小弯箭头（来回转时跟着转的方向翻）
          var rOn = fr(st, T.rate) * (1 - fr(st, T.out2));
          arcS.g.setAttribute('transform', 'translate(' + pS[0].toFixed(1) + ' ' + pS[1].toFixed(1) + ')');
          arcE.g.setAttribute('transform', 'translate(' + pE[0].toFixed(1) + ' ' + pE[1].toFixed(1) + ')');
          arcS.set(-0.5, 0.9); arcE.set(-0.2, 1.1);
          E.op(arcS.g, rOn); E.op(arcE.g, rOn);
          thS.setAttribute('transform', 'translate(' + (pS[0] - 44).toFixed(1) + ' ' + (pS[1] - 30).toFixed(1) + ')');
          thE.setAttribute('transform', 'translate(' + (pE[0] - 2).toFixed(1) + ' ' + (pE[1] - 52).toFixed(1) + ')');
          E.op(thS, rOn); E.op(thE, rOn);
          // 句 3：举例的手速度
          var exOn = 1 - fr(st, T.out3);
          exV.at(3, 2, 4, 4); E.op(exV.g, fr(st, T.arrow) * exOn);
          E.drawOn(exR, f(st, T.right, 0.5)); E.drawOn(exU, f(st, T.up, 0.5));
          if (st > T.out3) { E.op(exR, exOn); E.op(exU, exOn); }
          E.op(exRL, fr(st, T.right) * exOn); E.op(exUL, fr(st, T.up) * exOn);
          // 句 4：问号箭头跟着手
          var qOn = fr(st, T.dirQ) * (1 - fr(st, T.out4));
          qV.set(pH[0], pH[1], pH[0] - 0.9 * u, pH[1] - 1.5 * u); E.op(qV.g, qOn);
          qM.setAttribute('x', (pH[0] - 1.25 * u).toFixed(1)); qM.setAttribute('y', (pH[1] - 1.7 * u).toFixed(1));
          E.op(qM, qOn * (0.6 + 0.4 * Math.max(b(st, T.dirQ + 0.3, 0.8), b(st, T.farQ, 0.8))));

          /* ── 右栏 ── */
          E.op(g1, 1 - fr(st, T.out1));
          E.op(a1.g, fr(st, T.grid)); E.op(a2.g, fr(st, T.sh)); E.op(a2.parts[2], fr(st, T.origin)); E.op(a3.g, fr(st, T.el)); E.op(a3.parts[1], fr(st, T.e12)); E.op(a3.parts[2], fr(st, T.e12));
          E.op(a4.g, fr(st, T.hand)); E.op(a4.parts[1], fr(st, T.ep11)); E.op(a4.parts[2], fr(st, T.fore));
          E.op(a5.g, fr(st, T.two)); E.op(a5.parts[2], fr(st, T.c32)); E.op(a5.parts[3], fr(st, T.c32));
          E.op(g2, f(st, T.only - 0.2, 0.3) * (1 - fr(st, T.out2)));
          E.op(b1.g, fr(st, T.only)); E.op(b1.parts[1], fr(st, T.shJ)); E.op(b1.parts[2], fr(st, T.elJ)); E.op(b1.parts[3], fr(st, T.elJ)); E.op(b1.parts[4], fr(st, T.joints));
          E.op(b2.g, fr(st, T.fast)); E.op(b3.g, fr(st, T.rate)); E.op(b4.g, fr(st, T.perSec));
          E.op(b5.g, fr(st, T.ccw)); E.op(b5.parts[1], fr(st, T.plus)); E.op(b5.parts[2], fr(st, T.plus)); E.op(icoP.g, fr(st, T.ccw));
          E.op(b6.g, fr(st, T.cw)); E.op(b6.parts[1], fr(st, T.minus)); E.op(b6.parts[2], fr(st, T.minus)); E.op(icoM.g, fr(st, T.cw));
          E.op(g3, f(st, T.hv - 0.2, 0.3) * (1 - fr(st, T.out3)));
          E.op(c1.g, fr(st, T.hv)); E.op(c1.parts[1], fr(st, T.arrow)); E.op(c2.g, fr(st, T.right)); E.op(c3.g, fr(st, T.up)); E.op(c4.g, fr(st, T.ep3)); E.op(c4.parts[1], fr(st, T.vec)); E.op(c5.g, fr(st, T.vec + 0.3));
          E.op(g4, f(st, T.q - 0.2, 0.3) * (1 - fr(st, T.out4)));
          E.op(d1.g, fr(st, T.q)); E.op(d2.g, fr(st, T.shQ)); E.op(d3.g, fr(st, T.elQ)); E.op(d4.g, fr(st, T.dirQ)); E.op(d5.g, fr(st, T.farQ));
          E.op(g5, fr(st, T.one - 0.1));
          E.op(e1.g, fr(st, T.one)); E.op(e2.g, fr(st, T.onlyOne)); E.op(e2.parts[2], fr(st, T.onlyOne + 0.5)); E.op(e2.parts[3], fr(st, T.onlyOne + 0.5)); E.op(e3.g, fr(st, T.others));
          // 句 5：两个关节分别亮一下（① 肩、② 肘）
          if (st > T.one) { E.glow(jS, E.HI, b(st, T.onlyOne, 0.8) > 0.05); E.glow(jE, E.HI, b(st, T.onlyOne + 0.5, 0.8) > 0.05); }
        }
      };
    }
  };
})();
