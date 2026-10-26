// 010 第 3 幕：手算——转九十度后的位姿。左边同第 1 幕的网格 + UR5（往左、往下多放两格），右栏上方是 3 × 3 的位姿矩阵 T，下方一行行写算式。
// 句 1 右栏「不用真的分无穷多步」「现成的公式」「是一个位姿」；
// 句 2「三乘三的矩阵」T 出来（上两排先是问号，最下面一排 0、0、1），「左边两列是朝向」框紫、写「朝向」，「右边一列是位置」框蓝、写「位置」；
// 句 3「小臂转了九十度」小臂绕肘转到竖直，小臂上 (3, 2) 那一点沿圆弧走到 (1, 4)，「第三集那个转九十度的矩阵」左边两列填 0、1 / −1、0；
// 句 4「轴上的点」「肘就在这根轴上」肘亮一圈，「还在原地」右栏写；
// 句 5「先转」肘绕原点转九十度到 (−2, 1)（灰虚线弧 + 空心点），「负二、一」写出，「再平移」(−2, 1) → (1, 2) 画蓝箭头，「三、一」第三列填 3、1；
// 句 6「拼起来」T 亮一下，「三、二后面补一」「乘上它」「一、四」右栏写验算，左边 (1, 4) 鼓一下、写出；
// 句 7「位置这一列」第三列框鼓一下，「透明板上、原来在原点的那一点」原点亮出蓝点，「绕着肘」「转到了三、一」它沿圆弧走到 (3, 1)；
// 句 8「v 直接乘角度」从原点沿 (2, −1) 画红虚线到 (3.14, −1.57)，「走歪了」打红 ×；
// 句 9 UR5 淡出：「扫地机」(3, 1) 出现车头朝上的扫地机（第 7 集），「原点」原点出一台车头朝右的，「绕着一、二，转了九十度」它绕肘转到 (3, 1)，和前一台叠在一起。
// 时刻取自配音的逐词时间，见 ../ep010.py。
(function () {
  var E = window.EP010, K = E.K, H = E.HEX;
  var D = Math.PI / 180, A = E.ARM, RX = A.RX;
  var T = {
    nope: 0.91, ready: 3.11, pose: 4.82, out0: 6.3,
    mat: 9.56, rot: 10.89, pos: 12.34,
    simple: 14.78, turn: [16.49, 17.53], ep3: 18.2, nums: [19.06, 19.29, 19.72, 19.98], out2: 21.2,
    posRow: 21.5, axisPt: 23.04, elbow: 23.97, stay: 25.29,
    first: 27.57, qq: 28.3, rArc: [29.08, 30.4], neg: 31.18, shift: 32.53, back: [33.51, 34.3], p31: 36.54, p31b: 37.07,
    glue: 38.47, check: 40.77, p321: 41.83, mult: 43.49, res: 44.78, out6: 46.6,
    col: 47.62, plate: 49.6, origin: 50.93, travel: [52.1, 53.6], lab31: 53.51,
    vth: 55.68, vline: [56.71, 57.8], vlab: 58.2, wrong: 60.13,
    armOut: 62.19, ep7: 63.1, vac: 64.84, up: 66.95, ghost: 68.68, around: [70.94, 73.03]
  };
  var END = 74.02;

  E.scenes[2] = {
    title: '手算：转九十度后的位姿',
    dur: END,
    cues: [
      { at: 0, s: '不用真的分无穷多步：现成的公式，算出来是一个**位姿**' },
      { at: 6.48, s: '平面上的**位姿**写成 3 × 3：左边两列是**朝向**，右边一列是**位置**' },
      { at: 14.69, s: '**朝向**：小臂转了 90°，就是第 3 集那个转 90° 的矩阵' },
      { at: 21.41, s: '**位置**：肘就在轴上，转完还在原地' },
      { at: 27.48, s: '先转：(1, 2) → (−2, 1)；再平移回到 (1, 2)：**位置** = (3, 1)' },
      { at: 38.38, s: '验算：T × (3, 2, 1) = (1, 4, 1)，正好落在 (1, 4)' },
      { at: 46.85, s: '(3, 1) 是透明板上原来在原点的那一点，绕肘转过去的地方' },
      { at: 55.3, s: '拿 v 直接乘角度：(3.14, −1.57)，一样走歪了' },
      { at: 62.09, s: '第 7 集的扫地机：停在 (3, 1)、车头朝上 = 绕 (1, 2) 转 90°' }
    ],
    build: function (K) {
      var s = K.sceneSvg('矩阵指数算出来是一个位姿：平面上写成三乘三的矩阵，左边两列是朝向，右边一列是位置；小臂转了九十度，朝向就是第三集转九十度的矩阵；位置这一列让肘转完还在原地：(1, 2) 先转成 (−2, 1)，再平移回 (1, 2)，位置是 (3, 1)；验算 T 乘 (3, 2, 1) 得 (1, 4, 1)；(3, 1) 是透明板上原来在原点的点绕肘转过去的地方，拿 v 直接乘角度得 (3.14, −1.57)，走歪了；这个矩阵正是第七集扫地机停在 (3, 1)、车头朝上的位姿');

      var under = E.el('g', {}, s);
      var armG = E.el('g', {}, s);
      var arm = E.armAt(armG, A.OX, A.OY, A.S, 330);
      var u = arm.aim(1, 2, 0);
      var pl = E.plane(under, { ox: A.OX, oy: A.OY, u: u, x0: -2, x1: 4, y0: -2, y1: 5 });
      var P = pl.P, V = arm.view, RIGID = ['forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link'];
      var top = E.el('g', {}, s), Q = P(1, 2), O = P(0, 0);
      function around(c, p0, a) { var d = [p0[0] - c[0], p0[1] - c[1]], cs = Math.cos(a), sn = Math.sin(a); return [c[0] + cs * d[0] - sn * d[1], c[1] + sn * d[0] + cs * d[1]]; }
      function arcPts(c, p0, a0, a1) { var pts = []; for (var k = 0; k <= 40; k++) { var q = around(c, p0, a0 + (a1 - a0) * k / 40); pts.push(P(q[0], q[1])); } return pts; }

      // 小臂上的点：(3, 2) → (1, 4)
      var pArc = K.paint(E.el('path', { d: E.poly(arcPts([1, 2], [3, 2], 0, Math.PI / 2)), fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '7 6' }, top), null, E.ACC);
      var pDot = K.paint(E.el('circle', { r: 7, 'stroke-width': 3 }, top), E.ACC, E.S1);
      var p14 = P(1, 4);
      var p14Lab = E.halo(E.txt(top, p14[0] - 14, p14[1] + 8, '(1, 4)', 24, E.GOOD, 'end', 900));   // 点的左边：上面是手腕
      // 句 5：肘绕原点转 90° → (−2, 1)，再平移回 (1, 2)
      var eArc = K.paint(E.el('path', { d: E.poly(arcPts([0, 0], [1, 2], 0, Math.PI / 2)), fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '7 6' }, top), null, E.VIOLET);
      var R12 = P(-2, 1);
      var rDot = K.paint(E.el('circle', { cx: R12[0], cy: R12[1], r: 7, 'stroke-width': 3 }, top), E.S1, E.VIOLET);
      var rLab = E.halo(E.txt(top, R12[0] - 6, R12[1] + 30, '(−2, 1)', 22, E.VIOLET, 'start', 900));
      var shift = E.vec(top, E.ACC, 4.5, { head: 15 });
      // 句 7：透明板上原来在原点的点：(0, 0) → (3, 1)
      var oArc = K.paint(E.el('path', { d: E.poly(arcPts([1, 2], [0, 0], 0, Math.PI / 2)), fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '7 6' }, top), null, E.ACC);
      var oDot = K.paint(E.el('circle', { r: 7.5, 'stroke-width': 3 }, top), E.ACC, E.S1);
      var p31 = P(3, 1);
      var oLab = E.halo(E.txt(top, p31[0] + 14, p31[1] + 8, '(3, 1)', 24, E.ACC, 'start', 900));
      // 句 8：v × 角度，从原点沿 (2, −1) 走直线
      var vEnd = P(2 * Math.PI / 2, -Math.PI / 2);
      var vLine = K.paint(E.el('path', { d: 'M ' + O[0] + ' ' + O[1] + ' L ' + vEnd[0].toFixed(1) + ' ' + vEnd[1].toFixed(1), fill: 'none', 'stroke-width': 4, 'stroke-dasharray': '9 6' }, top), null, E.BAD);
      var vDot = K.paint(E.el('circle', { cx: vEnd[0], cy: vEnd[1], r: 7, 'stroke-width': 3 }, top), E.S1, E.BAD);
      var vLab = E.halo(E.txt(top, vEnd[0] + 14, vEnd[1] + 8, '(3.14, −1.57)', 22, E.BAD, 'start', 900));
      var vX = E.cross(top, vEnd[0] - 4, vEnd[1] - 26, 8, E.BAD, 4);
      // 肘：⊙ + (1, 2)
      var outOf = E.el('g', {}, top);
      var ring = K.paint(E.el('circle', { cx: Q[0], cy: Q[1], r: 12, 'stroke-width': 3.5 }, outOf), E.S1, E.HI);
      K.paint(E.el('circle', { cx: Q[0], cy: Q[1], r: 4 }, outOf), E.HI);
      var qLab = E.halo(E.txt(top, Q[0] + 18, Q[1] + 36, '(1, 2)', 22, E.HI, 'start', 900));   // 离肘的圈远一点：圈鼓起来半径到 17
      // 句 9：两台扫地机
      var vacs = E.el('g', {}, top);
      var vacA = E.vac(vacs, P, 0.46), vacB = E.vac(vacs, P, 0.46);
      vacA.set(3, 1, Math.PI / 2);

      /* ══ 右栏：T + 算式 ══ */
      var RK = E.rowKit(), Rg = E.el('g', {}, s), X = RX + 10, MX = RX + 172, MY = 112;
      var g0 = E.el('g', {}, Rg);
      var a1 = RK.line(g0, X, 96, [['不用真的分无穷多步', E.INK, 900]]);
      var a2 = RK.line(g0, X, 140, [['有', E.INK2], ['现成的公式', E.GOOD, 900]]);
      var a2ok = E.check(a2.g, X + 150, 122, 22, E.GOOD, 4);
      var a3 = E.el('g', {}, g0);
      E.tex(a3, X, 204, '\\textcolor{' + H.hi + '}{\\exp([\\mathcal S]\\,\\theta)}=', 28, E.INK, 'start', { w: 170, h: 50 });
      E.txt(a3, X + 176, 204, '一个位姿 T', 24, E.HI, 'start', 900);
      var mG = E.el('g', {}, Rg);
      E.tex(mG, X + 22, MY + 10, 'T=', 30, E.INK, 'start', { w: 60, h: 50 });
      var M = E.matN(mG, MX, MY, [['?', '?', '?'], ['?', '?', '?'], [0, 0, 1]], { size: 28, dx: 56, dy: 40,
        color: function (i, j) { return i === 2 ? E.MUT : j < 2 ? E.VIOLET : E.ACC; } });
      M.cells.forEach(function (row) { row.forEach(function (n) { n.__col = n.style.fill; }); });   // 填数前先写灰色问号
      var rotLab = E.txt(mG, MX - 28, MY + 92, '朝向', 22, E.VIOLET, 'middle', 900);
      var posLab = E.txt(mG, MX + 56, MY + 92, '位置', 22, E.ACC, 'middle', 900);
      var ROWS = 232;
      function rows(lines) {   // 一组算式：lines = [[y 偏移, parts, size], …]
        var g = E.el('g', {}, Rg);
        return { g: g, r: lines.map(function (l) { return RK.line(g, X, ROWS + l[0], l[1], l[2] || 22); }) };
      }
      var b3 = rows([[10, [['转了 90°：', E.INK2], ['第 3 集的矩阵', E.VIOLET, 900]]]]);
      var b4 = rows([[10, [['位置：用轴上的点', E.INK, 900]]], [52, [['肘就在轴上，转完还在 ', E.INK2], ['(1, 2)', E.HI, 900]]]]);
      var b5 = rows([[10, [['先转：', E.INK2], ['(1, 2)', E.HI, 900], [' → ', E.INK2], ['(−2, 1)', E.VIOLET, 900]]],
        [52, [['再平移，回到 ', E.INK2], ['(1, 2)', E.HI, 900], ['：', E.INK2]]],
        [96, [['位置 = ', E.ACC, 900], ['(1, 2) − (−2, 1)', E.INK, 900]]],
        [138, [['　　 = ', E.ACC, 900], ['(3, 1)', E.ACC, 900]]]]);
      var b6 = rows([[10, [['验算：', E.INK2], ['T', E.INK, 900], [' × ', E.INK2], ['(3, 2, 1)', E.ACC, 900]]],
        [52, [['　　 = ', E.INK2], ['(1, 4, 1)', E.GOOD, 900]]]]);
      var b6ok = E.check(b6.g, X + 190, ROWS + 34, 22, E.GOOD, 4);
      var b7 = rows([[10, [['(3, 1)', E.ACC, 900], ['：透明板上', E.INK, 900]]], [50, [['原来在原点的那一点', E.INK, 900]]],
        [92, [['绕着肘，转到了 ', E.INK2], ['(3, 1)', E.ACC, 900]]]]);
      var b8 = rows([[138, [['v', E.ACC, 900], [' × 角度 = ', E.INK2], ['(3.14, −1.57)', E.BAD, 900]]]]);
      var b8x = E.cross(b8.g, X + 300, ROWS + 130, 9, E.BAD, 4);
      var b9 = rows([[10, [['第 7 集：扫地机停在 ', E.INK2], ['(3, 1)', E.ACC, 900]]], [50, [['车头朝上，就是这个位姿', E.INK2]]],
        [100, [['= 绕 ', E.INK, 900], ['(1, 2)', E.HI, 900], [' 转 90°', E.VIOLET, 900]]]]);

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) { RK.layout(); laid = true; }
          var f = E.fade, b = E.bump;

          /* ── 小臂：句 3 转到竖直；句 9 淡出 ── */
          var phi = 90 * D * f(st, T.turn[0], T.turn[1] - T.turn[0]);
          arm.bend(phi);
          V.clearTint(); V.tint(RIGID, E.VIOLET, 0.45);
          E.op(armG, 1 - f(st, T.armOut, 0.5));
          var pp = around([1, 2], [3, 2], phi), ps = P(pp[0], pp[1]);
          pDot.setAttribute('cx', ps[0].toFixed(1)); pDot.setAttribute('cy', ps[1].toFixed(1));
          pDot.setAttribute('r', (7 + 4 * b(st, T.res, 0.9)).toFixed(2));
          E.op(pDot, 1 - f(st, T.armOut, 0.4));
          E.drawOn(pArc, f(st, T.turn[0], T.turn[1] - T.turn[0]));
          E.op(pArc, (f(st, T.turn[0], 0.1) > 0 ? 0.8 : 0) * (1 - f(st, T.out6, 0.4)));
          E.op(p14Lab, f(st, T.res, 0.3) * (1 - f(st, T.armOut, 0.4)));

          /* ── 肘 ── */
          E.op(outOf, 1); E.op(qLab, 1 - f(st, T.vac - 0.4, 0.4));   // 扫地机往左那根小箭头的尖在 (2, 1)，紧挨着「(1, 2)」
          ring.setAttribute('r', (12 + 5 * Math.max(b(st, T.elbow, 0.8), b(st, T.back[1], 0.8))).toFixed(2));
          E.glow(ring, E.HI, b(st, T.elbow, 0.8) > 0.05);

          /* ── 句 5：(1, 2) → (−2, 1) → (1, 2) ── */
          var on5 = 1 - f(st, T.glue + 0.6, 0.4);
          E.drawOn(eArc, f(st, T.rArc[0], T.rArc[1] - T.rArc[0])); E.op(eArc, f(st, T.rArc[0], 0.1) > 0 ? on5 : 0);
          E.op(rDot, f(st, T.rArc[1] - 0.1, 0.2) * on5); E.op(rLab, f(st, T.neg, 0.3) * on5);
          var sp = f(st, T.back[0], T.back[1] - T.back[0]);
          shift.set(R12[0], R12[1], R12[0] + (Q[0] - 14 - R12[0]) * sp, R12[1] + (Q[1] + 4 - R12[1]) * sp);
          E.op(shift.g, (sp > 0 ? 1 : 0) * on5);

          /* ── 句 7：原点那一点 → (3, 1) ── */
          var tv = f(st, T.travel[0], T.travel[1] - T.travel[0]), oq = around([1, 2], [0, 0], Math.PI / 2 * tv), os = P(oq[0], oq[1]);
          oDot.setAttribute('cx', os[0].toFixed(1)); oDot.setAttribute('cy', os[1].toFixed(1));
          oDot.setAttribute('r', (7.5 + 4 * b(st, T.origin, 0.8)).toFixed(2));
          E.op(oDot, f(st, T.plate, 0.3) * (1 - f(st, T.vac - 0.4, 0.4)));
          E.drawOn(oArc, tv); E.op(oArc, (tv > 0 ? 0.85 : 0) * (1 - f(st, T.armOut, 0.4) + 0.7 * f(st, T.ghost, 0.4)));   // 句 9 扫地机沿它走
          E.op(oLab, f(st, T.lab31, 0.3) * (1 - f(st, T.vac - 0.4, 0.4)));

          /* ── 句 8：v × 角度 ── */
          var on8 = 1 - f(st, T.armOut, 0.4);
          E.drawOn(vLine, f(st, T.vline[0], T.vline[1] - T.vline[0])); E.op(vLine, (f(st, T.vline[0], 0.1) > 0 ? 1 : 0) * on8);
          E.op(vDot, f(st, T.vline[1] - 0.1, 0.2) * on8); E.op(vLab, f(st, T.vlab, 0.3) * on8);
          E.drawOn(vX, f(st, T.wrong, 0.3)); E.op(vX, (f(st, T.wrong, 0.1) > 0 ? 1 : 0) * on8);

          /* ── 句 9：扫地机 ── */
          E.op(vacA.g, f(st, T.vac, 0.4));
          var ga = f(st, T.around[0], T.around[1] - T.around[0]), vq = around([1, 2], [0, 0], Math.PI / 2 * ga);
          vacB.set(vq[0], vq[1], Math.PI / 2 * ga);
          E.op(vacB.g, f(st, T.ghost, 0.4) * (0.75 - 0.75 * f(st, T.around[1], 0.4)));

          /* ── 右栏 ── */
          E.op(g0, 1 - f(st, T.out0, 0.4));
          E.op(a1.g, fr(st, T.nope)); E.op(a2.g, fr(st, T.ready)); E.drawOn(a2ok, f(st, T.ready + 0.3, 0.3)); E.op(a3, fr(st, T.pose));
          E.pop(mG, MX, MY, f(st, T.mat, 0.35), 0.9);
          M.hl('rot', 0, 0, 1, 1, E.VIOLET, f(st, T.rot, 0.3) * (1 - f(st, T.out2, 0.4)) + 0.6 * b(st, T.simple, 0.9));
          M.hl('pos', 0, 2, 1, 2, E.ACC, f(st, T.pos, 0.3) * (1 - f(st, T.ep3, 0.4)) + f(st, T.posRow, 0.3) * (1 - f(st, T.glue, 0.4)) + b(st, T.col, 0.9));
          M.hl('all', 0, 0, 2, 2, E.HI, b(st, T.glue, 1.0));
          E.op(rotLab, f(st, T.rot, 0.3)); E.op(posLab, f(st, T.pos, 0.3));
          var cells = [[0, 0, '0', T.nums[0]], [1, 0, '1', T.nums[1]], [0, 1, '−1', T.nums[2]], [1, 1, '0', T.nums[3]], [0, 2, '3', T.p31], [1, 2, '1', T.p31b]];
          cells.forEach(function (c) {
            var n = M.cells[c[0]][c[1]], q = M.pos(c[0], c[1]), p = f(st, c[3], 0.3);
            n.textContent = st >= c[3] ? c[2] : '?';
            E.op(n, st >= c[3] ? p : 1); E.scaleAt(n, q[0], q[1], st >= c[3] ? 1 + 0.2 * (1 - p) : 1);
            n.style.fill = st >= c[3] ? n.__col : E.MUT;
          });
          E.op(b3.g, fr(st, T.ep3) * (1 - f(st, T.out2, 0.4)));
          E.op(b4.g, fr(st, T.posRow) * (1 - f(st, T.first - 0.3, 0.3)));
          E.op(b4.r[1].g, fr(st, T.elbow)); E.op(b4.r[1].parts[1], fr(st, T.stay));
          E.op(b5.g, fr(st, T.first) * (1 - f(st, T.glue + 0.4, 0.3)));
          E.op(b5.r[0].parts[1], fr(st, T.qq)); E.op(b5.r[0].parts[2], fr(st, T.neg)); E.op(b5.r[0].parts[3], fr(st, T.neg));
          E.op(b5.r[1].g, fr(st, T.shift)); E.op(b5.r[2].g, fr(st, T.back[1])); E.op(b5.r[3].g, fr(st, T.p31));
          E.op(b6.g, fr(st, T.check) * (1 - f(st, T.out6, 0.4)));
          E.op(b6.r[0].parts[3], fr(st, T.p321)); E.op(b6.r[0].parts[1], fr(st, T.mult)); E.op(b6.r[0].parts[2], fr(st, T.mult));
          E.op(b6.r[1].g, fr(st, T.res)); E.drawOn(b6ok, f(st, T.res + 0.3, 0.3));
          E.op(b7.g, fr(st, T.col) * (1 - f(st, T.armOut, 0.4)));
          E.op(b7.r[0].parts[1], fr(st, T.plate)); E.op(b7.r[1].g, fr(st, T.plate)); E.op(b7.r[2].g, fr(st, T.travel[0]));
          E.op(b8.g, fr(st, T.vth) * (1 - f(st, T.armOut, 0.4))); E.op(b8.r[0].parts[2], fr(st, T.vlab)); E.drawOn(b8x, f(st, T.wrong, 0.3));
          E.op(b9.g, fr(st, T.ep7));
          E.op(b9.r[0].parts[1], fr(st, T.vac)); E.op(b9.r[1].g, fr(st, T.up)); E.op(b9.r[2].g, fr(st, T.around[0]));
        }
      };
    }
  };
})();
