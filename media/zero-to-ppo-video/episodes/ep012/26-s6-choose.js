// 012 第 6 幕：多组解——怎么选。左边先是第 2 幕的网格 + 两台 UR5（肘上、肘下的虚影），句 3 换成侧视的 G1 抬腿，句 4 回到 UR5；右栏一行行写。
// 句 1「解有好几组」两台都亮，「只能摆一个姿势」「选谁」右栏写；
// 句 2「装在桌子上」底座下面画出桌面（底座底面比肩低 0.089 m ≈ 0.47 格，UR5 URDF），「肘下这一组」虚影亮，「伸到桌面下面去」肘下的大臂描橙，
//    「会撞上」碰到桌面的地方打 ×，「不要」虚影淡掉；
// 句 3「超出关节限位」「第八集」换成侧视 G1 抬起近侧的腿（膝弯 70°，在 −5°～165° 里），「膝盖反过来弯的那一组」把膝盖沿髋—踝连线翻过去、
//    画成橙色虚线（膝 −70°），「超出限位」「用不了」打 ×，右栏写 −5°～165°（G1 URDF：left_knee_joint −0.087267～2.8798 rad）；
// 句 4「挑一组离现在的姿势最近的」回到 UR5，目标从 (1, 0) 挪到 (1.2, 0.3)：肘上这组肩只多转 13°、肘多转 7°（亮黄）；
//    「胳膊却要整个翻过去」换成肘下那组要肩转 −112°、肘转 300°（橙，虚影从现在的姿势翻过去）；
// 2026-11-05 参照 011：手照第 11 集放在小臂上离肘两格，目标和角度跟着改（Python 重算：肩 63.43° → 76.82° / −48.75°，肘 −153.43° → −146.63° / +146.63°）。
// 句 5「公式写不出来」「第十五集」「用雅可比，一步一步往目标挪」手腕一小步一小步走到目标（虚线折线）。
// 时刻取自配音的逐词时间，见 ../ep012.py。
(function () {
  var E = window.EP012, K = E.K, H = E.HEX;
  var D = Math.PI / 180, A = E.ARM, RX = A.RX;
  var T = {
    many: 0.59, one: 1.9, who: 3.6,
    drop: 5.3, table: 7.72, dnSol: 8.73, under: [9.84, 11.35], hit: 11.77, no: 12.61, out2: 13.8,
    limit: 14.08, ep8: 15.95, knee: 18.08, range: 18.64, neg: 19.12, rev: [21.7, 23.09], over: 23.45, cant: 24.47, out3: 25.9,
    rest: 26.15, pick: 27.03, near: 28.49, move: [27.6, 28.9], small: 29.36, flip: [30.96, 32.3],
    formula: 33.72, ep15: 35.97, jac: 37.1, steps: [38.09, 39.6]
  };
  var END = 40.58;
  var UP = E.ik2(1, 0, true), DN = E.ik2(1, 0, false), UP2 = E.ik2(1.2, 0.3, true), DN2 = E.ik2(1.2, 0.3, false);
  var TABLE = -0.089159 / (0.425 / Math.sqrt(5));   // 桌面（底座底面）在肩下面多少格

  E.scenes[5] = {
    title: '多组解：怎么选',
    dur: END,
    cues: [
      { at: 0, s: '解有好几组，机器人一次只能摆一个姿势：选谁？' },
      { at: 4.94, s: '先去掉做不到的：装在桌子上，**肘下**这组大臂伸到桌面下，会撞上' },
      { at: 13.99, s: '超出关节限位的也不要：G1 的膝只能在 −5°～165°（第 8 集），膝盖反过来弯的那组**用不了**' },
      { at: 26.06, s: '剩下的，挑离现在姿势最近的一组：不然手只挪一点，胳膊却要整个翻过去' },
      { at: 33.62, s: '公式写不出来：第 15 集，用雅可比一步一步往目标挪' }
    ],
    build: function (K) {
      var s = K.sceneSvg('解有好几组，机器人一次只能摆一个姿势；先去掉做不到的：机械臂装在桌子上，肘下这一组大臂要伸到桌面下面，会撞上；超出关节限位的也不要：人形机器人的膝盖只能在 −5° 到 165° 之间弯，膝盖反过来弯的那一组用不了；剩下的挑离现在的姿势最近的一组：目标从 (1, 0) 挪到 (1.2, 0.3)，肘上这组肩只转 13°、肘转 7°，换成肘下要肩转 −112°、肘转 300°，胳膊整个翻过去；公式写不出来时，第十五集讲用雅可比一步一步往目标挪');

      /* ── 左 A：网格 + 两台 UR5 + 桌面 ── */
      var armPanel = E.el('g', {}, s);
      var under = E.el('g', {}, armPanel);
      var box = [A.OX - 110, A.OY - 214, 440, 420];
      var ghostG = E.el('g', {}, armPanel), ghost = E.arm2(ghostG, A.OX, A.OY, A.S, 0, box);
      var armG = E.el('g', {}, armPanel), arm = E.arm2(armG, A.OX, A.OY, A.S, 0, box), u = arm.u, V = arm.view;
      var pl = E.plane(under, { ox: A.OX, oy: A.OY, u: u, x0: -1, x1: 4, y0: -3, y1: 3 });
      var P = pl.P, O = P(0, 0), TG = P(1, 0), TG2 = P(1.2, 0.3);
      var top = E.el('g', {}, armPanel);
      var tableG = E.el('g', {}, top), TY = P(0, TABLE)[1];
      K.paint(E.el('rect', { x: P(-1.4, 0)[0], y: TY, width: 5.8 * u, height: 0.55 * u, rx: 4, 'stroke-width': 2.5 }, tableG), E.tint('#a0784f', 45, 'var(--demo-surface)'), '#a0784f');
      E.halo(E.txt(tableG, P(3.2, 0)[0], TY + 0.55 * u + 22, '桌面', 20, '#c49a6c', 'middle', 900));
      var hitX = E.cross(top, P(0.24, TABLE)[0], TY, 9, E.ERR, 4.5);
      var badArm = K.paint(E.el('path', { d: 'M ' + O[0] + ' ' + O[1] + ' L ' + P(1, -2)[0] + ' ' + P(1, -2)[1], fill: 'none', 'stroke-width': 7, 'stroke-linecap': 'round', 'stroke-dasharray': '10 7' }, top), null, E.ERR);
      var tgt = E.target(top, 0, 0, 12, E.TGT);
      var tg2Lab = E.halo(E.txt(top, TG2[0] + 18, TG2[1] + 30, '(1.2, 0.3)', 20, E.TGT, 'start', 900));   // 点的右下：右上是一步一步挪过来的折线
      var oldT = E.target(top, TG[0], TG[1], 9, E.MUT);
      // 句 5：一步一步挪过去的折线（示意：每步离目标近一点）
      var STEPM = [[3, 2], [2.6, 1.5], [2.1, 1.0], [1.65, 0.62], [1.38, 0.4], [1.2, 0.3]];
      var stepPts = STEPM.map(function (p) { return P(p[0], p[1]); });
      var stepPath = K.paint(E.el('path', { d: E.poly(stepPts), fill: 'none', 'stroke-width': 3.5, 'stroke-dasharray': '6 5' }, top), null, E.OMEGA);
      var stepDots = stepPts.map(function (p) { return K.paint(E.el('circle', { cx: p[0], cy: p[1], r: 5 }, top), E.OMEGA); });

      /* ── 左 B：侧视 G1 抬起近侧的腿 ── */
      var g1Panel = E.el('g', {}, s);
      var ground = K.paint(E.el('line', { x1: 40, y1: 408, x2: 400, y2: 408, 'stroke-width': 3 }, g1Panel), null, E.BORDER);
      // 只画腰以下、放大（同 001 第 4 幕的腿部特写：box 是 g1Side 的本地坐标，上缘渐隐）
      // 整个人画出来（视景框切掉上半身会被 check_layout 报「机器人伸出视景框」），放大到 1.55 倍、脚底在 408
      var g1 = window.Robot3D.g1Side(g1Panel, 0, { res: 3 }), SC = 1.55;
      g1.place({ x: 200, scale: SC, dy: 408 - 388 });
      function sv(p) { return [g1.state.x + SC * p[0], g1.state.dy + 388 + SC * (p[1] - 388)]; }
      var KA = 50, KB = -20;                                 // 大腿往前抬 50°、小腿往后 20°：膝弯 70°
      g1.legs([KA, KB], [-4, 0]);
      var Pg = g1.pts(), hip = sv(Pg.hip), kneeP = sv(Pg.knee), ank = sv(Pg.ankle);
      // 膝盖沿髋—踝连线翻过去
      var dx = ank[0] - hip[0], dy2 = ank[1] - hip[1], L = Math.hypot(dx, dy2), ux = dx / L, uy = dy2 / L;
      var kx = kneeP[0] - hip[0], ky = kneeP[1] - hip[1], pr = kx * ux + ky * uy;
      var kneeM = [hip[0] + 2 * pr * ux - kx, hip[1] + 2 * pr * uy - ky];
      var revLeg = K.paint(E.el('path', { d: E.poly([hip, kneeM, ank]), fill: 'none', 'stroke-width': 8, 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-dasharray': '11 7' }, g1Panel), null, E.ERR);
      var revKnee = K.paint(E.el('circle', { cx: kneeM[0], cy: kneeM[1], r: 8, 'stroke-width': 3 }, g1Panel), E.S1, E.ERR);
      var revX = E.cross(g1Panel, kneeM[0] + 22, kneeM[1] - 18, 9, E.ERR, 4.5);
      var revLab = E.halo(E.txt(g1Panel, kneeM[0] + 36, kneeM[1] + 8, '膝 −70°', 20, E.ERR, 'start', 900));
      var okKnee = K.paint(E.el('circle', { cx: kneeP[0], cy: kneeP[1], r: 8, 'stroke-width': 3 }, g1Panel), E.S1, E.GOOD);
      var okLab = E.halo(E.txt(g1Panel, kneeP[0] + 16, kneeP[1] - 12, '膝 70°', 20, E.GOOD, 'start', 900));
      var ankT = E.target(g1Panel, ank[0], ank[1], 10, E.TGT);

      /* ══ 右栏 ══ */
      var RK = E.rowKit(), Rg = E.el('g', {}, s), X = RX + 10;
      var g1r = E.el('g', {}, Rg);
      var a1 = RK.line(g1r, X, 40, [['好几组解，一次只能摆一个：', E.INK2]], 20);
      var a2 = RK.line(g1r, X, 76, [['选谁？', E.HI, 900]], 25);
      var b1 = RK.line(g1r, X, 130, [['① 先去掉', E.INK, 900], ['做不到的', E.ERR, 900]], 21);
      var b2 = RK.line(g1r, X, 166, [['肘下', E.DOWN, 900], ['：大臂伸到桌面下，撞上', E.INK2]], 20);
      var b3 = RK.line(g1r, X, 206, [['超出关节限位的也不要', E.INK2]], 20);
      var b4 = RK.line(g1r, X, 242, [['G1 的膝：', E.INK2], ['−5° ～ 165°', E.GOOD, 900], ['（第 8 集）', E.MUT, 700]], 20);
      var b5 = RK.line(g1r, X, 278, [['膝盖反过来弯：', E.INK2], ['用不了', E.ERR, 900]], 20);
      var c1 = RK.line(g1r, X, 330, [['② 挑离现在', E.INK, 900], ['最近', E.HI, 900], ['的一组', E.INK, 900]], 21);
      var c2 = RK.line(g1r, X, 366, [['肘上', E.UP, 900], ['：肩转 13°、肘转 7°', E.INK2]], 20);
      var c3 = RK.line(g1r, X, 400, [['换肘下', E.DOWN, 900], ['：肩转 −112°、肘转 300°', E.ERR, 900]], 20);
      var g2r = E.el('g', {}, Rg);
      var d1 = RK.line(g2r, X, 140, [['公式写不出来？', E.INK, 900]], 23);
      var d2 = RK.line(g2r, X, 182, [['第 15 集：用', E.INK2], ['雅可比', E.OMEGA, 900]], 21);
      var d3 = RK.line(g2r, X, 218, [['一步一步往目标挪', E.INK2]], 21);

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }
      function lerp(a, b, k) { return a + (b - a) * k; }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) { RK.layout(); laid = true; }
          var f = E.fade, b = E.bump;
          var bOn = f(st, T.ep8 - 0.6, 0.4) * (1 - f(st, T.out3, 0.4));
          E.op(armPanel, 1 - bOn); E.op(g1Panel, bOn);
          /* ── UR5：句 4 目标挪了，先走近的那组，再示意「翻过去」 ── */
          var mv = f(st, T.move[0], T.move[1] - T.move[0]), fl = f(st, T.flip[0], T.flip[1] - T.flip[0]);
          var q1 = lerp(UP[0], UP2[0], mv), q2 = lerp(UP[1], UP2[1], mv);
          // 句 5：先回到第 9–11 集的摆法（手在 (3, 2)），再按折线上的点一步一步走到目标（每一步都是肘上那组）
          if (st >= T.formula) {
            var back = f(st, T.formula, 0.8), P0 = [Math.atan2(2, 1), 0];
            q1 = lerp(UP2[0], P0[0], back); q2 = lerp(UP2[1], P0[1], back);
            if (st >= T.steps[0]) { var kS = Math.min(STEPM.length - 1, Math.floor(f(st, T.steps[0], T.steps[1] - T.steps[0]) * (STEPM.length - 1) + 1e-6)); var sq = E.ik2(STEPM[kS][0], STEPM[kS][1], true); q1 = sq[0]; q2 = sq[1]; }
          }
          arm.pose(q1, q2); V.clearTint();
          // 虚影：句 1–2 是肘下那组；句 4「整个翻过去」从肘上转到肘下（新目标），肩、肘各转一大圈
          var gq1 = DN[0], gq2 = DN[1];
          if (st >= T.rest) { gq1 = lerp(UP[0], DN2[0], fl); gq2 = lerp(UP[1] - UP[0], DN2[1] - DN2[0], fl) + gq1; }   // 从现在的姿势（目标 (1, 0) 的肘上）翻到新目标的肘下：肘从 −153.4° 转到 +146.6°
          ghost.pose(gq1, gq2); ghost.view.clearTint();
          ghost.view.tint('all', st >= T.rest ? E.ERR : E.DOWN, 0.8, 0.45);
          var gOn = st < T.rest ? 0.9 * (1 - 0.75 * f(st, T.no, 0.4)) * (1 + 0.0 * b(st, T.dnSol, 0.8)) : 0.85 * f(st, T.flip[0] - 0.2, 0.3) * (1 - f(st, T.formula, 0.4));
          E.op(ghostG, gOn);
          E.glow(ghostG, E.DOWN, b(st, T.dnSol, 1.0) > 0.05);
          // 桌面、撞上
          E.op(tableG, fr(st, T.table) * (1 - f(st, T.rest, 0.4)));
          E.drawOn(badArm, f(st, T.under[0], T.under[1] - T.under[0])); E.op(badArm, (f(st, T.under[0], 0.1) > 0 ? 1 : 0) * (1 - f(st, T.no + 0.3, 0.4)));
          E.drawOn(hitX, f(st, T.hit, 0.3)); E.op(hitX, (f(st, T.hit, 0.1) > 0 ? 1 : 0) * (1 - f(st, T.rest, 0.4)));
          // 目标
          var tq = st < T.move[0] ? TG : [lerp(TG[0], TG2[0], mv), lerp(TG[1], TG2[1], mv)];
          tgt.setAttribute('transform', 'translate(' + tq[0].toFixed(1) + ' ' + tq[1].toFixed(1) + ')');
          E.op(tg2Lab, fr(st, T.move[1]) * (1 - f(st, T.formula, 0.3) * 0));
          E.op(oldT, f(st, T.move[0], 0.3) * (1 - f(st, T.formula, 0.3)));
          // 句 5：一步一步
          var sp = f(st, T.steps[0], T.steps[1] - T.steps[0]);
          E.drawOn(stepPath, sp); E.op(stepPath, sp > 0 ? 1 : 0);
          stepDots.forEach(function (c, k) { E.op(c, sp >= k / (stepDots.length - 1) - 1e-3 && sp > 0 ? 1 : 0); });

          /* ── G1 ── */
          E.op(revLeg, f(st, T.rev[0], 0.5)); E.op(revKnee, f(st, T.rev[0], 0.5)); E.op(revLab, fr(st, T.rev[1]));
          E.drawOn(revX, f(st, T.over, 0.3)); E.op(revX, f(st, T.over, 0.1) > 0 ? 1 : 0);
          revLeg.style.strokeWidth = 8 + 3 * b(st, T.cant, 0.8);
          E.op(okKnee, fr(st, T.knee)); E.op(okLab, fr(st, T.range + 1.0));
          E.op(ankT, fr(st, T.knee));

          /* ── 右栏 ── */
          E.op(g1r, 1 - f(st, T.formula, 0.4));
          E.op(a1.g, fr(st, T.many)); E.op(a2.g, fr(st, T.who));
          E.op(b1.g, fr(st, T.drop)); E.op(b2.g, fr(st, T.dnSol)); E.op(b3.g, fr(st, T.limit)); E.op(b4.g, fr(st, T.knee)); E.op(b4.parts[1], fr(st, T.neg)); E.op(b4.parts[2], fr(st, T.neg));
          E.op(b5.g, fr(st, T.rev[0])); E.op(b5.parts[1], fr(st, T.cant));
          E.op(c1.g, fr(st, T.pick)); E.op(c2.g, fr(st, T.small)); E.op(c3.g, fr(st, T.flip[0]));
          E.op(g2r, f(st, T.formula, 0.4));
          E.op(d1.g, fr(st, T.formula)); E.op(d2.g, fr(st, T.ep15)); E.op(d2.parts[1], fr(st, T.jac)); E.op(d3.g, fr(st, T.steps[0]));
        }
      };
    }
  };
})();
