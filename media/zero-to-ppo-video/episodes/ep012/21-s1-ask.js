// 012 第 1 幕：倒过来问——已知手腕的位置，求关节角。左边 UR5（先斜着看，句 3「从侧面看」转到正侧视、网格淡入），右栏一行行写。
// 句 1「正向运动学」右栏写，「每个关节」六个关节亮出圆点，「角度定了」关节转到位、停住，「手」「停在什么位置」手腕亮紫点，「朝向」手上出三根小箭头，
//    「只有一个答案」右栏打勾；
// 句 2「逆向运动学」「反过来」右栏写，「手」「要到的位姿」出紫色目标 ⊕，「求每个关节的角度」关节圆点换成问号，「不一定只有一个」右栏写；
// 句 3「从侧面看」镜头从斜视转到正侧视，「只看肩和肘」只留肩、肘两个圆点，其余连杆变淡，「只管手腕」手腕亮紫点，
//    「手的朝向」「交给手腕上的关节」手腕那几节亮一下，右栏写「第 5 幕」；
// 句 4「第九集的网格」网格淡入，「肩关节在原点」「肘在一、二」标出 (0, 0)、(1, 2)，「勾股定理」画出直角三角形：往右 1（红）、往上 2（绿），
//    「一的平方」「加二的平方」「是五」右栏逐项写，「五开平方」「约二点二四格」大臂旁写 2.24 格（琥珀）；
// 句 5「零点四二五米」「除以二点二四」「一格约零点一九米」右栏写，「小臂」「零点三九二米」「除以零点一九」「约二点零六格」小臂旁写 2.06 格（绿）；
// 句 6「手腕停在肩的正前方一点五格」网格上出目标 ⊕ (1.5, 0)、量出 1.5 格，「和肩一样高」水平虚线，「一点五、零」写坐标，「肩和肘，各该转多少」两个关节出问号。
// 时刻取自配音的逐词时间，见 ../ep012.py。
(function () {
  var E = window.EP012, K = E.K, H = E.HEX;
  var D = Math.PI / 180, A = E.ARM, RX = A.RX;
  var T = {
    fk: 1.09, joints: 2.23, ang: 2.93, fixed: 3.27, hand: 3.8, where: 4.21, ori: 5.1, one: 6.18,
    ik: 8.13, rev: 8.95, hand2: 10.56, pose2: 11.26, solve: 11.86, joints2: 12.09, ans: 14.56, notOne: 15.04, out1: 16.9,
    simple: 17.95, side: [19.16, 20.4], only: 21.22, sh: 21.62, el: 21.95, wrist: 23.45, ori3: 25.54, wj: 26.98, later: 28.25, out3: 30.1,
    grid: 30.86, origin: 32.93, elbow: 33.6, py: 36.18, sq1: 37.06, sq2: 37.38, five: 38.65, upper: 39.27, sqrt: 40.17, len: 41.33,
    m425: 45.12, div: 46.54, cell: 47.8, fore: 49.41, m392: 49.8, div2: 51.35, fl: 52.62, out5: 54.6,
    now: 54.89, tw: 55.81, front: 56.94, d15: 57.65, level: 58.62, c15: 60.12, how: 61.75, ask: 62.71
  };
  var END = 64.63, C = 0.425 / Math.sqrt(5);   // 一格多少米
  var POSE0 = [Math.atan2(2, 1), 0];            // 第 9、10 集的摆法：肘在 (1, 2)，小臂水平

  E.scenes[0] = {
    title: '倒过来问：已知手腕的位置，求关节角',
    dur: END,
    cues: [
      { at: 0, s: '**正向运动学**：关节角定了，手的位姿只有一个答案' },
      { at: 8.04, s: '**逆向运动学**反过来：给出手要到的位姿，求关节角——答案不一定只有一个' },
      { at: 17.38, s: '先看最简单的：从侧面看，只看肩、肘 2 个关节，只管**手腕**的位置' },
      { at: 30.36, s: '第 9 集的网格：肘在 (1, 2)，1² + 2² = 5，大臂 = √5 ≈ 2.24 格' },
      { at: 43.49, s: '大臂 0.425 m ÷ 2.24 ≈ 0.19 m / 格；小臂 0.392 m ÷ 0.19 ≈ 2.06 格' },
      { at: 54.79, s: '目标：**手腕**停在 (1.5, 0)。肩、肘各转多少？' }
    ],
    build: function (K) {
      var s = K.sceneSvg('正向运动学：每个关节的角度定了，手停在什么位置、朝向什么方向只有一个答案；逆向运动学反过来，给出手要到的位姿求每个关节的角度，答案不一定只有一个；先看最简单的情况：从侧面看机械臂，只看肩和肘两个关节，只管手腕的位置，手的朝向交给手腕上的关节；第九集的网格里肩关节在原点、肘在 (1, 2)，1² + 2² = 5，大臂是 √5 ≈ 2.24 格；大臂 0.425 m ÷ 2.24 ≈ 0.19 m 一格，小臂 0.392 m ÷ 0.19 ≈ 2.06 格；目标：手腕停在肩的正前方 1.5 格、和肩一样高，也就是 (1.5, 0)，肩和肘各该转多少？');

      var under = E.el('g', {}, s);
      var armG = E.el('g', {}, s);
      var arm = E.arm2(armG, A.OX, A.OY, A.S, 0, [A.OX - 110, A.OY - 214, 440, 420]), u = arm.u, V = arm.view;
      var pl = E.plane(under, { ox: A.OX, oy: A.OY, u: u, x0: -1, x1: 4, y0: -3, y1: 3 });
      var P = pl.P, O = P(0, 0), Q = P(1, 2);
      // 斜视：同一个原点，镜头绕竖直轴转过来（句 3 转回正侧视 az 0、el 0）
      V.camera({ az: 34, el: 14 });
      var top = E.el('g', {}, s);
      // 六个关节的圆点（句 1）→ 句 3 只留肩、肘
      var LINKS = ['shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link'];
      var jd = LINKS.map(function () { var g = E.el('g', {}, top); K.paint(E.el('circle', { r: 8, 'stroke-width': 3 }, g), E.S1, E.HI); return g; });
      var QK = [0, 2, 4];   // 斜着看时手腕三个关节挤在一起：问号只打在底座、肘、手腕中间那个上，不叠成一团
      var jq = LINKS.map(function () { return E.halo(E.txt(top, 0, 0, '?', 24, E.HI, 'middle', 900)); });
      var wDot = K.paint(E.el('circle', { r: 7, 'stroke-width': 3 }, top), E.TGT, E.S1);
      var ori = [E.X, E.Y, E.Z].map(function (c) { return E.vec(top, c, 4, { head: 12 }); });
      // 目标（句 2 斜着看的、句 6 网格上的）
      var tg2 = E.target(top, 0, 0, 12, E.TGT);
      var TG = P(1.5, 0), tg6 = E.target(top, TG[0], TG[1], 13, E.TGT);
      var tgLab = E.halo(E.txt(top, TG[0] + 20, TG[1] + 34, '(1.5, 0)', 22, E.TGT, 'start', 900));
      // 句 4：直角三角形（往右 1 红、往上 2 绿、大臂斜边琥珀）
      var tri = E.el('g', {}, top), P10 = P(1, 0);
      var tA = K.paint(E.el('path', { d: 'M ' + O[0] + ' ' + O[1] + ' L ' + P10[0] + ' ' + P10[1], fill: 'none', 'stroke-width': 4, 'stroke-linecap': 'round' }, tri), null, E.X);
      var tB = K.paint(E.el('path', { d: 'M ' + P10[0] + ' ' + P10[1] + ' L ' + Q[0] + ' ' + Q[1], fill: 'none', 'stroke-width': 4, 'stroke-linecap': 'round' }, tri), null, E.Y);
      var tC = K.paint(E.el('path', { d: 'M ' + O[0] + ' ' + O[1] + ' L ' + Q[0] + ' ' + Q[1], fill: 'none', 'stroke-width': 4, 'stroke-dasharray': '8 6' }, tri), null, E.C1);
      var tAl = E.halo(E.txt(tri, (O[0] + P10[0]) / 2, O[1] + 26, '1', 22, E.X, 'middle', 900));
      var tBl = E.halo(E.txt(tri, P10[0] + 14, (P10[1] + Q[1]) / 2 + 8, '2', 22, E.Y, 'start', 900));
      var corner = K.paint(E.el('path', { d: 'M ' + (P10[0] - 10) + ' ' + P10[1] + ' V ' + (P10[1] - 10) + ' H ' + P10[0], fill: 'none', 'stroke-width': 2 }, tri), null, E.INK2);
      var upLen = E.halo(E.txt(top, (O[0] + Q[0]) / 2 - 14, (O[1] + Q[1]) / 2 + 2, '2.24 格', 21, E.C1, 'end', 900));
      var oLab = E.halo(E.txt(top, O[0] - 14, O[1] + 30, '(0, 0)', 20, E.INK, 'end', 900));
      var qLab = E.halo(E.txt(top, Q[0] - 16, Q[1] - 14, '(1, 2)', 22, E.HI, 'end', 900));
      var foreLen = E.halo(E.txt(top, 0, 0, '2.06 格', 21, E.C2, 'middle', 900));
      // 句 6：正前方 1.5 格的量尺、和肩一样高的水平虚线
      var dim = E.el('g', {}, top), dy = 0.62 * u;
      K.paint(E.el('path', { d: 'M ' + O[0] + ' ' + (O[1] + dy - 7) + ' V ' + (O[1] + dy + 7) + ' M ' + O[0] + ' ' + (O[1] + dy) + ' H ' + TG[0] + ' M ' + TG[0] + ' ' + (O[1] + dy - 7) + ' V ' + (O[1] + dy + 7),
        fill: 'none', 'stroke-width': 2.5 }, dim), null, E.INK);
      E.halo(E.txt(dim, (O[0] + TG[0]) / 2, O[1] + dy + 26, '1.5 格', 20, E.INK, 'middle', 900));
      var level = K.paint(E.el('path', { d: 'M ' + (O[0] - 0.8 * u) + ' ' + O[1] + ' H ' + (TG[0] + 1.2 * u), fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '7 6' }, top), null, E.TGT);

      /* ══ 右栏 ══ */
      var RK = E.rowKit(), Rg = E.el('g', {}, s), X = RX + 10;
      function grp() { return E.el('g', {}, Rg); }
      var g1 = grp();
      var a1 = RK.line(g1, X, 64, [['正向运动学', E.OMEGA, 900]], 26);
      var a2 = RK.line(g1, X, 104, [['关节角', E.OMEGA, 900], [' → ', E.INK2], ['手的位姿', E.TGT, 900]]);
      var a3 = RK.line(g1, X, 144, [['只有一个答案', E.GOOD, 900]]);
      var a3ok = E.check(a3.g, X + 150, 126, 22, E.GOOD, 4);
      var a4 = RK.line(g1, X, 226, [['逆向运动学', E.HI, 900]], 26);
      var a5 = RK.line(g1, X, 266, [['手的位姿', E.TGT, 900], [' → ', E.INK2], ['关节角', E.OMEGA, 900], [' ？', E.HI, 900]]);
      var a6 = RK.line(g1, X, 306, [['答案', E.INK2], ['不一定只有一个', E.HI, 900]]);
      var g3 = grp();
      var c1 = RK.line(g3, X, 64, [['先看最简单的：', E.INK, 900]]);
      var c2 = RK.line(g3, X, 112, [['从侧面看', E.INK2], ['，只看 ', E.INK2], ['肩', E.HI, 900], ['、', E.INK2], ['肘', E.HI, 900], [' 两个关节', E.INK2]], 21);
      var c3 = RK.line(g3, X, 156, [['只管 ', E.INK2], ['手腕', E.TGT, 900], [' 停在什么位置', E.INK2]]);
      var c4 = RK.line(g3, X, 222, [['手的朝向', E.INK, 900], [' → 手腕上的关节', E.INK2]]);
      var c5 = RK.line(g3, X, 258, [['（第 5 幕再说）', E.MUT, 700]], 20);
      var g4 = grp();
      var d1 = RK.line(g4, X, 54, [['第 9 集的网格：', E.INK2], ['肘在 ', E.INK2], ['(1, 2)', E.HI, 900]]);
      var d2 = RK.line(g4, X, 96, [['勾股定理（第 5 集）：', E.INK2]], 21);
      var d3 = RK.line(g4, X, 134, [['1²', E.X, 900], [' + ', E.INK2], ['2²', E.Y, 900], [' = 1 + 4 = ', E.INK2], ['5', E.INK, 900]]);
      var d4 = RK.line(g4, X, 174, [['大臂 = √5 ≈  ', E.INK2], ['2.24 格', E.C1, 900]]);
      var d5 = RK.line(g4, X, 232, [['大臂 0.425 m', E.INK, 900]], 21);
      var d6 = RK.line(g4, X, 268, [['0.425 ÷ 2.24 ≈  ', E.INK2], ['0.19 m / 格', E.INK, 900]], 21);
      var d7 = RK.line(g4, X, 322, [['小臂 0.392 m', E.INK, 900]], 21);
      var d8 = RK.line(g4, X, 358, [['0.392 ÷ 0.19 ≈  ', E.INK2], ['2.06 格', E.C2, 900]], 21);
      var g6 = grp();
      var e1 = RK.line(g6, X, 80, [['目标：', E.INK2], ['手腕', E.TGT, 900], ['停在', E.INK2]]);
      var e2 = RK.line(g6, X, 122, [['肩的正前方 ', E.INK2], ['1.5 格', E.INK, 900]]);
      var e3 = RK.line(g6, X, 162, [['和肩一样高', E.INK, 900]]);
      var e4 = RK.line(g6, X, 206, [['→ ', E.INK2], ['(1.5, 0)', E.TGT, 900]], 26);
      var e5 = RK.line(g6, X, 284, [['肩', E.HI, 900], ['、', E.INK2], ['肘', E.HI, 900], ['，各该转多少？', E.HI, 900]], 25);

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }
      function lerp(a, b, k) { return a + (b - a) * k; }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) { RK.layout(); laid = true; }
          var f = E.fade, b = E.bump;

          /* ── UR5：句 1 关节转到位；句 3 镜头转到正侧视 ── */
          var sk = f(st, T.side[0], T.side[1] - T.side[0]);
          V.camera({ az: lerp(34, 0, sk), el: lerp(14, 0, sk) });
          var wig = 1 - f(st, T.ang, T.fixed - T.ang + 0.2);   // 一开始几个关节在动，「角度定了」停住
          var w1 = 0.35 * Math.sin(t * 2.1) * wig, w2 = 0.3 * Math.sin(t * 1.7 + 1) * wig;
          arm.pose(POSE0[0] + w1, POSE0[1] + w2);
          V.joints({ shoulder_pan_joint: 0.25 * Math.sin(t * 1.3) * wig });
          V.clearTint();
          var dim3 = f(st, T.only, 0.4);
          V.tint(['shoulder_link', 'wrist_2_link', 'wrist_3_link'], null, 1, 1 - 0.55 * dim3);
          V.tint(['wrist_1_link', 'wrist_2_link', 'wrist_3_link'], E.HI, 0.7 * b(st, T.wj, 1.0), 1 - 0.55 * dim3 * (1 - b(st, T.wj, 1.0)));
          var pts = LINKS.map(function (l) { return V.at(l); });
          jd.forEach(function (g, k) {
            g.setAttribute('transform', 'translate(' + pts[k][0].toFixed(1) + ' ' + pts[k][1].toFixed(1) + ')');
            var keep = k === 1 || k === 2;
            E.op(g, f(st, T.joints, 0.3) * (keep ? 1 : 1 - f(st, T.only, 0.3)));
            E.glow(g, E.HI, b(st, keep ? (k === 1 ? T.sh : T.el) : T.joints, 0.8) > 0.05);
          });
          jq.forEach(function (n, k) {
            var early = QK.indexOf(k) >= 0 ? f(st, T.joints2, 0.3) * (1 - f(st, T.out1, 0.3)) : 0, on = (k === 1 || k === 2) ? Math.max(early, f(st, T.how, 0.3)) : early;
            n.setAttribute('x', (pts[k][0] + 18).toFixed(1)); n.setAttribute('y', (pts[k][1] - 10).toFixed(1));
            E.op(n, on);
          });
          var w = arm.wrist();
          wDot.setAttribute('cx', w[0].toFixed(1)); wDot.setAttribute('cy', w[1].toFixed(1));
          wDot.setAttribute('r', (7 + 4 * Math.max(b(st, T.where, 0.8), b(st, T.wrist, 0.8))).toFixed(2));
          E.op(wDot, Math.max(f(st, T.hand, 0.3) * (1 - f(st, T.out1, 0.3)), f(st, T.wrist, 0.3) * (1 - f(st, T.tw, 0.3))));
          // 手的朝向：tool0 的三根小箭头（句 1）
          var tp = V.at('tool0');
          [[0.14, 0, 0], [0, 0.14, 0], [0, 0, 0.14]].forEach(function (d, k) { var q = V.at('tool0', d); ori[k].set(tp[0], tp[1], q[0], q[1]); E.op(ori[k].g, f(st, T.ori, 0.3) * (1 - f(st, T.ik, 0.3))); });
          // 句 2 的目标：网格上 (1.5, 0) 那一点在三维里的位置（手腕所在的竖直面）
          var sh = V.world('upper_arm_link'), wy = V.world('wrist_1_link').y;
          var g2p = V.project([sh.x + 1.5 * C, wy, sh.z]);
          tg2.setAttribute('transform', 'translate(' + g2p[0].toFixed(1) + ' ' + g2p[1].toFixed(1) + ')');
          var p2 = f(st, T.pose2, 0.35) * (1 - f(st, T.out1, 0.3));
          E.op(tg2, p2);

          /* ── 句 4–6：网格 ── */
          E.op(under, f(st, T.grid, 0.5));
          E.op(oLab, fr(st, T.origin) * (1 - f(st, T.out5, 0.3))); E.op(qLab, fr(st, T.elbow) * (1 - f(st, T.out5, 0.3)));
          E.op(tri, f(st, T.py, 0.3) * (1 - f(st, T.out5, 0.3)));
          E.drawOn(tA, f(st, T.sq1, 0.4)); E.drawOn(tB, f(st, T.sq2, 0.4)); E.drawOn(tC, f(st, T.sqrt, 0.5));
          E.op(tAl, fr(st, T.sq1)); E.op(tBl, fr(st, T.sq2)); E.op(corner, fr(st, T.sq2 + 0.3));
          E.op(upLen, fr(st, T.len) * (1 - f(st, T.out5, 0.3)));
          var e = arm.elbow();
          foreLen.setAttribute('x', ((e[0] + w[0]) / 2).toFixed(1)); foreLen.setAttribute('y', (e[1] - 16).toFixed(1));
          E.op(foreLen, fr(st, T.fl) * (1 - f(st, T.out5, 0.3)));
          E.pop(tg6, TG[0], TG[1], f(st, T.tw, 0.35), 0.5);
          E.op(dim, fr(st, T.d15)); E.drawOn(level, f(st, T.level, 0.6)); E.op(level, (f(st, T.level, 0.1) > 0 ? 0.8 : 0));
          E.op(tgLab, fr(st, T.c15));

          /* ── 右栏 ── */
          E.op(g1, 1 - f(st, T.out1, 0.4));
          E.op(a1.g, fr(st, T.fk)); E.op(a2.g, fr(st, T.joints)); E.op(a2.parts[2], fr(st, T.hand)); E.op(a3.g, fr(st, T.one)); E.drawOn(a3ok, f(st, T.one + 0.4, 0.3));
          E.op(a4.g, fr(st, T.ik)); E.op(a5.g, fr(st, T.rev)); E.op(a5.parts[2], fr(st, T.solve)); E.op(a5.parts[3], fr(st, T.joints2)); E.op(a6.g, fr(st, T.ans)); E.op(a6.parts[1], fr(st, T.notOne));
          E.op(g3, f(st, T.simple - 0.3, 0.3) * (1 - f(st, T.out3, 0.4)));
          E.op(c1.g, fr(st, T.simple)); E.op(c2.g, fr(st, T.side[0])); E.op(c2.parts[2], fr(st, T.sh)); E.op(c2.parts[4], fr(st, T.el)); E.op(c2.parts[5], fr(st, T.el + 0.2));   // 21 号字：23 号写全会超出右栏
          E.op(c3.g, fr(st, T.wrist)); E.op(c4.g, fr(st, T.ori3)); E.op(c4.parts[1], fr(st, T.wj)); E.op(c5.g, fr(st, T.later));
          E.op(g4, f(st, T.grid - 0.3, 0.3) * (1 - f(st, T.out5, 0.4)));
          E.op(d1.g, fr(st, T.grid)); E.op(d1.parts[2], fr(st, T.elbow)); E.op(d2.g, fr(st, T.py)); E.op(d3.g, fr(st, T.sq1));
          E.op(d3.parts[2], fr(st, T.sq2)); E.op(d3.parts[3], fr(st, T.five)); E.op(d3.parts[4], fr(st, T.five));
          E.op(d4.g, fr(st, T.sqrt)); E.op(d4.parts[1], fr(st, T.len));
          E.op(d5.g, fr(st, T.m425)); E.op(d6.g, fr(st, T.div)); E.op(d6.parts[1], fr(st, T.cell));
          E.op(d7.g, fr(st, T.m392)); E.op(d8.g, fr(st, T.div2)); E.op(d8.parts[1], fr(st, T.fl));
          E.op(g6, f(st, T.now, 0.3));
          E.op(e1.g, fr(st, T.tw)); E.op(e2.g, fr(st, T.front)); E.op(e3.g, fr(st, T.level)); E.op(e4.g, fr(st, T.c15)); E.op(e5.g, fr(st, T.how)); E.op(e5.parts[3], fr(st, T.ask));
        }
      };
    }
  };
})();
