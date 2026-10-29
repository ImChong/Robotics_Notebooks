// 010 第 1 幕：直接乘角度——走成了直线。左边是 009 的侧视网格 + UR5（肩关节为原点，肘在 (1, 2)，小臂水平），右栏一行行写要点。
// 句 1「肘在一、二」肘上 ⊙、写 (1, 2)，「小臂水平」小臂上色，「逆时针往上抬」肘旁出逆时针的小弯箭头；
// 句 2「转了九十度」右栏写问题，「一点」小臂上 (3, 2) 亮出一点，「离肘两格」肘到这一点量出 2 格，「三、二」写 (3, 2)；
// 句 3「朝上走」这一点长出朝上的速度箭头，「一点五七弧度」「弧的长度」「半径乘角度」「三点一四格」右栏一行行写出，
//    「这段弧」它要走的四分之一圆描成灰色虚线；
// 句 4「一直朝上走」从这一点往上画一条红色虚线（3.14 格），「三、五点一四」终点写 (3, 5.14)；
//    「小臂转完九十度」小臂绕肘转到竖直，这一点沿圆弧走到 (1, 4)，「一、四」写 (1, 4)，直线终点打红 ×；
// 句 5「前进的方向也跟着转」圆弧上 0°、30°、60°、90° 四处依次长出速度箭头（朝上 → 朝左），
//    「只按一开始的方向走」直线鼓一下，「冲出了圆」整个圆描成淡虚线、直线终点的 × 鼓一下；
// 句 6「一段圆弧」圆弧换成绿色实线、亮一下，「不能拿速度直接乘角度」右栏出结论框。
// 时刻取自配音的逐词时间（../ep010.py 的 [from, to] 等于每句配音实长，分镜时间 = 念到这个词的时刻）。
(function () {
  var E = window.EP010, K = E.K;
  var D = Math.PI / 180, A = E.ARM, RX = A.RX;
  var T = {
    q: [2.42, 2.79], flat: 3.56, spin: 4.9,
    ask: 7.49, pt: 11.36, two: 12.19, ptLab: 13.19, out1: 14.8,
    up: 15.84, rad: 16.95, arc: 19.31, len: 19.71, rule: 20.29, calc: 21.84, res: 23.38,
    line: [25.77, 27.35], end: 28.23, turn: [30.25, 31.6], real: 33.57,
    dirs: [35.97, 36.88, 37.66, 38.91], up2: 38.13, left: 39.68, only: 40.65, straight: 43.26, outC: 43.89,
    arcOk: 46.48, no: 51.51
  };
  var END = 54.34, R0 = 2, LEN = Math.PI / 2 * R0;

  E.scenes[0] = {
    title: '直接乘角度：走成了直线',
    dur: END,
    cues: [
      { at: 0, s: '回到第 9 集：肘在 (1, 2)，小臂绕着肘逆时针转' },
      { at: 6.94, s: '小臂转 90°：离肘 2 格的点 (3, 2) 停在什么位置？' },
      { at: 15.02, s: '90° ≈ 1.57 弧度；弧长 = 半径 × 角度 = 2 × 1.57 ≈ 3.14 格' },
      { at: 25.39, s: '一直朝上走：停在 (3, 5.14)；其实停在 (1, 4)' },
      { at: 35.38, s: '前进的方向跟着转：只按一开始的方向走，就冲出了圆' },
      { at: 45.84, s: '真正的路线是圆弧：位置**不能**拿速度直接乘角度' }
    ],
    build: function (K) {
      var s = K.sceneSvg('机械臂小臂绕肘关节逆时针转九十度：小臂上离肘两格的点 (3, 2) 一开始朝上走，九十度约 1.57 弧度，弧长等于半径乘角度约 3.14 格；要是一直朝上走 3.14 格会停在 (3, 5.14)，其实小臂转完是竖着的，这一点停在 (1, 4)；因为前进的方向跟着转，从朝上慢慢变成朝左，只按一开始的方向走就走成直线冲出了圆；真正的路线是一段圆弧，位置不能拿速度直接乘角度');

      var under = E.el('g', {}, s);
      var arm = E.armAt(s, A.OX, A.OY, A.S, 330);
      var u = arm.aim(1, 2, 0);
      if (Math.abs(u - A.U) > 0.3) console.error('ep010 grid unit ' + u + ' vs E.ARM.U ' + A.U);
      var pl = E.plane(under, { ox: A.OX, oy: A.OY, u: u, x0: -1, x1: 4, y0: -1, y1: 5 });
      var P = pl.P, V = arm.view, RIGID = ['forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link'];
      var top = E.el('g', {}, s), Q = P(1, 2);
      function onArc(a) { return P(1 + R0 * Math.cos(a), 2 + R0 * Math.sin(a)); }
      function arcD(a0, a1) { var pts = []; for (var k = 0; k <= 40; k++) pts.push(onArc(a0 + (a1 - a0) * k / 40)); return E.poly(pts); }

      // 整个圆（句 5「冲出了圆」）、要走的四分之一圆（灰虚线 → 绿实线）
      var circle = K.paint(E.el('circle', { cx: Q[0], cy: Q[1], r: R0 * u, fill: 'none', 'stroke-width': 2, 'stroke-dasharray': '6 7' }, top), null, E.MUT);
      var arcGrey = K.paint(E.el('path', { d: arcD(0, Math.PI / 2), fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '8 7' }, top), null, E.INK2);
      var arcOk = K.paint(E.el('path', { d: arcD(0, Math.PI / 2), fill: 'none', 'stroke-width': 5, 'stroke-linecap': 'round' }, top), null, E.GOOD);
      // 直线：从 (3, 2) 一直朝上 3.14 格
      var S0 = P(3, 2), S1 = P(3, 2 + LEN);
      var line = K.paint(E.el('path', { d: 'M ' + S0[0].toFixed(1) + ' ' + S0[1].toFixed(1) + ' L ' + S1[0].toFixed(1) + ' ' + S1[1].toFixed(1), fill: 'none', 'stroke-width': 4, 'stroke-dasharray': '9 6' }, top), null, E.BAD);
      var endDot = K.paint(E.el('circle', { cx: S1[0], cy: S1[1], r: 7, 'stroke-width': 3 }, top), E.S1, E.BAD);
      var endLab = E.halo(E.txt(top, S1[0] + 16, S1[1] + 8, '(3, 5.14)', 22, E.BAD, 'start', 900));
      var endX = E.cross(top, S1[0] - 22, S1[1] - 2, 8, E.BAD, 4);
      // 肘：⊙ + (1, 2) + 小弯箭头
      var outOf = E.el('g', {}, top);
      K.paint(E.el('circle', { cx: Q[0], cy: Q[1], r: 12, 'stroke-width': 3.5 }, outOf), E.S1, E.HI);
      K.paint(E.el('circle', { cx: Q[0], cy: Q[1], r: 4 }, outOf), E.HI);
      var qLab = E.halo(E.txt(top, Q[0] - 18, Q[1] - 16, '(1, 2)', 22, E.HI, 'end', 900));
      var spin = E.arcArrow(top, Q[0], Q[1], 30, E.VIOLET, 3.5, { head: 12 }); spin.set(-60 * D, 40 * D);
      // 两格的量尺（小臂上方）
      var dimG = E.el('g', {}, top), dy = -0.42 * u;
      K.paint(E.el('path', { d: 'M ' + Q[0] + ' ' + (Q[1] + dy - 8) + ' V ' + (Q[1] + dy + 8) + ' M ' + Q[0] + ' ' + (Q[1] + dy) + ' H ' + S0[0] + ' M ' + S0[0] + ' ' + (Q[1] + dy - 8) + ' V ' + (Q[1] + dy + 8),
        fill: 'none', 'stroke-width': 2.5 }, dimG), null, E.INK);
      E.halo(E.txt(dimG, (Q[0] + S0[0]) / 2, Q[1] + dy - 10, '2 格', 20, E.INK, 'middle', 900));
      // 小臂上的那一点（跟着小臂转）+ 速度箭头 + 标签
      var vel = E.vec(top, E.ACC, 4.5, { head: 14 });
      var pDot = K.paint(E.el('circle', { r: 7.5, 'stroke-width': 3 }, top), E.ACC, E.S1);
      var pLab = E.halo(E.txt(top, S0[0] - 12, S0[1] + 34, '(3, 2)', 22, E.ACC, 'end', 900));   // 点的左下：右下挂着 UR5 的手
      var realLab = E.halo(E.txt(top, 0, 0, '(1, 4)', 24, E.GOOD, 'end', 900));
      // 句 5：圆弧上四处的速度箭头
      var dirs = [0, 30, 60, 90].map(function () { return E.vec(top, E.ACC, 4, { head: 13 }); });

      /* ══ 右栏 ══ */
      var RK = E.rowKit(), Rg = E.el('g', {}, s), X = RX + 10;
      var g1 = E.el('g', {}, Rg);
      var a1 = RK.line(g1, X, 60, [['第 9 集：肘在 ', E.INK2], ['(1, 2)', E.HI, 900]]);
      var a2 = RK.line(g1, X, 102, [['小臂绕着肘，', E.INK2], ['逆时针转', E.VIOLET, 900]]);
      var a3 = RK.line(g1, X, 164, [['问：转 90° 以后，', E.HI, 900]], 24);
      var a4 = RK.line(g1, X, 202, [['　　停在什么位置？', E.HI, 900]], 24);
      var a5 = RK.line(g1, X, 262, [['看小臂上一点：', E.INK2], ['(3, 2)', E.ACC, 900]]);
      var a6 = RK.line(g1, X, 302, [['离肘 2 格', E.INK, 900]]);
      var g2 = E.el('g', {}, Rg);
      var b1 = RK.line(g2, X, 56, [['刚开始：', E.INK2], ['朝上走', E.ACC, 900]]);
      var b2 = RK.line(g2, X, 98, [['90° ≈ ', E.INK], ['1.57 弧度', E.VIOLET, 900]]);
      var b3 = RK.line(g2, X, 140, [['弧长 = 半径 × 角度', E.INK, 900]]);
      var b4 = RK.line(g2, X, 182, [['　　 = 2 × 1.57 ≈ ', E.INK], ['3.14 格', E.ACC, 900]]);
      var g3 = E.el('g', {}, Rg);
      var c1 = RK.line(g3, X, 250, [['一直朝上走 3.14 格：', E.INK2]]);
      var c2 = RK.line(g3, X, 290, [['→ ', E.INK2], ['(3, 5.14)', E.BAD, 900]]);
      var c2x = E.cross(c2.g, X + 150, 282, 9, E.BAD, 4);
      var c3 = RK.line(g3, X, 340, [['其实停在：', E.INK2], ['(1, 4)', E.GOOD, 900]]);
      var c3ok = E.check(c3.g, X + 190, 322, 22, E.GOOD, 4);
      var g4 = E.el('g', {}, Rg);
      var d1 = RK.line(g4, X, 56, [['前进的方向跟着转：', E.INK, 900]]);
      var d2 = RK.line(g4, X, 96, [['朝上', E.ACC, 900], [' → ', E.INK2], ['朝左', E.ACC, 900]]);
      var d3 = RK.line(g4, X, 150, [['只按一开始的方向走：', E.INK2]]);
      var d4 = RK.line(g4, X, 190, [['走成直线，', E.BAD, 900], ['冲出了圆', E.BAD, 900]]);
      var d5 = RK.line(g4, X, 244, [['真正的路线：', E.INK2], ['圆弧', E.GOOD, 900]]);
      var box = E.el('g', {}, g4);
      K.paint(E.el('rect', { x: RX, y: 282, width: 330, height: 96, rx: 14, 'stroke-width': 3 }, box), E.tint(E.HI, 8, 'var(--demo-surface)'), E.HI);
      E.txt(box, RX + 165, 322, '转完停在什么位置，', 23, E.INK, 'middle', 900);
      E.txt(box, RX + 165, 360, '不能拿速度直接乘角度', 23, E.BAD, 'middle', 900);

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) { RK.layout(); laid = true; }
          var f = E.fade, b = E.bump;

          /* ── 小臂：句 4 绕肘转到竖直 ── */
          var phi = 90 * D * f(st, T.turn[0], T.turn[1] - T.turn[0]);
          arm.bend(phi);
          V.clearTint();
          var ft = f(st, T.flat, 0.4);
          if (ft > 0) V.tint(RIGID, E.VIOLET, 0.45 * ft);

          /* ── 肘 ── */
          E.op(outOf, f(st, T.q[0], 0.3)); E.op(qLab, f(st, T.q[1], 0.3));
          E.op(spin.g, f(st, T.spin, 0.3) * (1 - f(st, T.pt - 0.4, 0.4)));   // 「(3, 2)」出来前收掉：它写在弯箭头的右下

          /* ── 小臂上的一点 ── */
          var pp = onArc(phi);
          pDot.setAttribute('cx', pp[0].toFixed(1)); pDot.setAttribute('cy', pp[1].toFixed(1));
          pDot.setAttribute('r', (7.5 + 4 * Math.max(b(st, T.pt, 0.8), b(st, T.real, 0.8))).toFixed(2));
          E.op(pDot, f(st, T.pt, 0.3));
          E.op(dimG, f(st, T.two, 0.3) * (1 - f(st, T.up - 0.4, 0.3)));
          E.op(pLab, f(st, T.ptLab, 0.3) * (1 - f(st, T.line[0], 0.3)));
          // 速度箭头：长 1 格，垂直于「肘 → 这一点」，跟着转；句 5 起收掉（换成圆弧上的四支）
          var tg = [-Math.sin(phi), Math.cos(phi)], tip = [pp[0] + tg[0] * u, pp[1] - tg[1] * u];
          vel.set(pp[0], pp[1], tip[0], tip[1]);
          E.op(vel.g, f(st, T.up, 0.3) * (1 - f(st, T.dirs[0] - 0.4, 0.3)));
          // 终点 (1, 4)：写在点的左上（正左是句 5 朝左的速度箭头，右上是手腕；压着 y 轴的那一段有底色描边）
          realLab.setAttribute('x', (pp[0] - 22).toFixed(1)); realLab.setAttribute('y', (pp[1] - 16).toFixed(1));
          E.op(realLab, f(st, T.real, 0.3));

          /* ── 圆弧、直线 ── */
          E.op(arcGrey, f(st, T.arc, 0.4) * (1 - f(st, T.arcOk, 0.4)));
          E.drawOn(arcOk, f(st, T.arcOk, 0.8));
          var ab = b(st, T.arcOk + 0.8, 1.0);
          arcOk.style.strokeWidth = 5 + 3 * ab; E.glow(arcOk, E.GOOD, ab > 0.05);
          E.drawOn(line, f(st, T.line[0], T.line[1] - T.line[0]));
          var lb = Math.max(b(st, T.only, 0.9), b(st, T.straight, 0.9));
          line.style.strokeWidth = 4 + 3 * lb; E.glow(line, E.BAD, lb > 0.05);
          E.op(endDot, f(st, T.line[1] - 0.1, 0.2)); E.op(endLab, f(st, T.end, 0.3));
          E.drawOn(endX, f(st, T.real + 0.3, 0.3));
          endX.style.strokeWidth = 4 + 3 * b(st, T.outC + 0.3, 0.8);
          E.op(circle, f(st, T.outC, 0.5) * 0.9);

          /* ── 句 5：圆弧上四支速度箭头（朝上 → 朝左） ── */
          dirs.forEach(function (a, i) {
            var ang = [0, 30, 60, 90][i] * D, c = onArc(ang), d = [-Math.sin(ang), Math.cos(ang)];
            a.set(c[0], c[1], c[0] + d[0] * 0.9 * u, c[1] - d[1] * 0.9 * u);
            var hi = i === 0 ? b(st, T.up2, 0.8) : i === 3 ? b(st, T.left, 0.8) : 0;
            a.line.style.strokeWidth = 4 + 3 * hi;
            E.op(a.g, f(st, T.dirs[i], 0.3) * (1 - f(st, T.arcOk - 0.2, 0.4)));
          });

          /* ── 右栏 ── */
          E.op(g1, 1 - f(st, T.out1, 0.4));
          E.op(a1.g, fr(st, T.q[0])); E.op(a2.g, fr(st, T.spin));
          E.op(a3.g, fr(st, T.ask)); E.op(a4.g, fr(st, T.ask));
          E.op(a5.g, fr(st, T.pt)); E.op(a5.parts[1], fr(st, T.ptLab)); E.op(a6.g, fr(st, T.two));
          E.op(g2, f(st, T.out1 + 0.2, 0.4) * (1 - f(st, T.dirs[0] - 0.6, 0.4)));
          E.op(b1.g, fr(st, T.up)); E.op(b2.g, fr(st, T.rad)); E.op(b3.g, fr(st, T.rule));
          E.op(b4.g, fr(st, T.calc)); E.op(b4.parts[1], fr(st, T.res));
          E.op(g3, f(st, T.line[0], 0.4) * (1 - f(st, T.dirs[0] - 0.6, 0.4)));
          E.op(c1.g, 1); E.op(c2.g, fr(st, T.end)); E.drawOn(c2x, f(st, T.real + 0.3, 0.3));
          E.op(c3.g, fr(st, T.real)); E.drawOn(c3ok, f(st, T.real + 0.3, 0.3));
          E.op(g4, f(st, T.dirs[0] - 0.2, 0.4));
          E.op(d1.g, 1); E.op(d2.parts[0], fr(st, T.up2)); E.op(d2.parts[1], fr(st, T.left)); E.op(d2.parts[2], fr(st, T.left));
          E.op(d3.g, fr(st, T.only)); E.op(d4.parts[0], fr(st, T.straight)); E.op(d4.parts[1], fr(st, T.outC));
          E.op(d5.g, fr(st, T.arcOk));
          E.pop(box, RX + 165, 330, f(st, T.no, 0.35), 0.9);
        }
      };
    }
  };
})();
