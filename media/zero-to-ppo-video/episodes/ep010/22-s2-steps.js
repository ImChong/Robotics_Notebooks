// 010 第 2 幕：切成无穷多小步——矩阵指数。左边同第 1 幕的网格（肘在 (1, 2)，小臂上的点从 (3, 2) 出发，要走的四分之一圆描成灰虚线），
// 右栏一行行写算式；中段左边换成复利的条形图；最后换回网格。
// 2026-11-01 按 009 的用户反馈「全篇的计算都展开讲不要跳过」「纯公式推导没有画面观众会看不懂」重做：分两步的每一步都算出来、在网格上画出来，
// 式子里的符号在屏幕上逐个说明；配色同第 1 幕（转动蓝、点紫、走歪的橙，往右 / 往左一段红、往上一段绿）。
// 句 1「每走一小步」「停下来」「这时候的方向」「再走下一步」右栏三行；
// 句 2「分两步」「每步转四十五度」四分之一圈上 45° 处标一道，「一点五七除以二」「零点七八五」右栏写出；
// 句 3「离肘两格」量出 2 格，「朝上走」这一点出朝上的小箭头，「二乘零点七八五」「一点五七格」右栏写出、网格上画第一段（琥珀），「三、三点五七」写出；
// 句 4「右边两格、上边一点五七格」肘到拐点画虚线（黄），旁边写 (2, 1.57)，「垂直」拐点标直角，「转九十度」虚线平移到拐点、绕拐点逆时针转 90°（蓝弧），
//    「往左一点五七」红的一段往左、「往上二」绿的一段往上，头尾相接（段长写在右栏，同色）；
// 句 5「各乘零点七八五」红绿两段缩成 0.785 倍，「一点二三」「一点五七」写出，「走过去」画第二段（琥珀），「三减一点二三」「三点五七加上一点五七」右栏写出；
// 句 6「一点七七、五点一四」终点写出，「折了一次」拐点鼓一下，「离圆近了一些」终点和第 1 幕的 (3, 5.14) 各连一条到 (1, 4) 的灰虚线；
// 句 7「四步、八步」两条折线依次画出，「交给电脑」右栏写，「贴着圆走」灰虚线亮一下；
// 句 8「一百步」画第五条，「一、四点零二」写出，「无穷多步」绿色圆弧画出，「一、四」打勾；
// 句 9「乘同一个矩阵」右栏出一步的矩阵，「乘上几次」出 n 次方，「无穷多步」n → ∞，「矩阵指数」写出 exp([S]θ)，下面逐个说明 I、[S]、θ、n；
// 句 10–12 复利：「存一块钱」第一条先有 1 块，「一加一」长到 2；「十二个月」「乘上一加十二分之一」「乘十二次」第二条一格一格长 12 次到 2.61，
//    「分得越细」第三条长到 2.71，「二点七一八」画出竖虚线，「自然常数」写 e；
// 句 13「几次方，叫指数函数」右栏写 eˣ，「同一个做法」两行对照，左边换回网格，「一步一步滚起来」一个点沿圆一小步一小步走，
//    「不是钱，是位姿」点上长出两根小箭头（位置 + 朝向）。
// 时刻取自配音的逐词时间，见 ../ep010.py。
(function () {
  var E = window.EP010, K = E.K, H = E.HEX;
  var D = Math.PI / 180, A = E.ARM, RX = A.RX;
  var T = {
    each: 1.97, stop: 3.08, dir: 3.71, next: 4.89, out0: 6.7,
    two: 7.26, deg: 8.12, div: 9.15, h: 10.85,
    first: 13.34, r2: 13.93, upA: 14.81, mul1: 15.47, len1: 17.23, seg1: [17.23, 18.3], k1: 18.71,
    second: 21.33, legs: 22.9, legsB: 23.83, perp: 25.3, perp2: 26.63, turn: [27.43, 28.2], left: 28.52, up2: 29.78,
    scale: [32.04, 33.3], r123: 33.63, r157: 34.66, go: [35.93, 36.8], sub: 36.8, sub2: 38.12, add: 39.33, add2: 41.44,
    stopAt: 43.89, kink: 45.66, near: 46.46, out1: 48.3,
    n4: 48.83, n8: 49.4, pc: 51.92, hug: 54.18, n100: 56.83, at100: 57.97, inf: 61.45, arc: 62.56, at: 64.33,
    one: 66.28, same: 67.85, times: 69.52, nInf: 71.04, name: 73.27, legend: [67.0, 68.0, 69.0, 70.4],
    out2: 75.6, why: 76.06, e0: 77.8, coin: 78.74, rate: 79.89, once: 81.2, onePlus: 82.55, two2: 83.31,
    m12: 85.15, perM: 86.88, k12: [89.43, 90.8], v12: 90.94, fine: [92.87, 93.7], lim: 93.99, eNum: 96.23,
    pow: 98.4, fn: 99.89, same2: 100.91, back: 101.1, roll: [103.06, 105.6], money: 106.74, pose: 107.46
  };
  var END = 108.91, TH = Math.PI / 2, NS = [1, 2, 4, 8, 100], HH = TH / 2;
  var COLS = [E.ERR, E.WARN, E.INK2, E.INK, E.GOOD];

  // 切成 n 步：d ← d + (θ / n) · 转 90°(d)，d 是相对肘的位置；返回各拐点（格）
  function steps(n) {
    var d = [2, 0], h = TH / n, pts = [[3, 2]];
    for (var k = 0; k < n; k++) { d = [d[0] - h * d[1], d[1] + h * d[0]]; pts.push([1 + d[0], 2 + d[1]]); }
    return pts;
  }

  E.scenes[1] = {
    title: '切成无穷多小步：矩阵指数',
    dur: END,
    cues: [
      { at: 0, s: '分成几小步：每走一小步就停下来，按这时候的方向再走下一步' },
      { at: 6.84, s: '分 2 步、每步 45°：1.57 ÷ 2 ≈ 0.785 弧度' },
      { at: 13.25, s: '第一步：朝上走 2 × 0.785 ≈ 1.57 格，到 (3, 3.57)' },
      { at: 21.24, s: '第二步：离肘 (2, 1.57)，转 90° 是往左 1.57、往上 2' },
      { at: 31.37, s: '各乘 0.785：往左 1.23、往上 1.57；3 − 1.23 = 1.77，3.57 + 1.57 = 5.14' },
      { at: 43.42, s: '停在 (1.77, 5.14)：折了一次，离圆近了一些' },
      { at: 48.46, s: '分得越细越贴着圆走：100 步停在 (1, 4.02)，无穷多步停在 (1, 4)' },
      { at: 66.19, s: '每一小步乘同一个矩阵；无穷多步乘出来的，叫**矩阵指数**' },
      { at: 75.74, s: '名字来自自然常数 e：存 1 块，年利率 100%，一年算 1 次：1 + 1 = 2 块' },
      { at: 85.06, s: '12 个月利滚利：每月乘 (1 + 1/12)，乘 12 次是 2.61；越分越细，接近 2.718' },
      { at: 98.04, s: 'eˣ 叫指数函数；**矩阵指数**是同一个做法，滚起来的是**位姿**' }
    ],
    build: function (K) {
      var s = K.sceneSvg('小臂上的点绕肘转九十度：分成几小步，每走一小步就按这时候的方向再走下一步；分两步每步转 45°，约 0.785 弧度：第一步朝上走 2 × 0.785 ≈ 1.57 格到 (3, 3.57)；第二步离肘 (2, 1.57)，前进方向和连线垂直，转 90° 是往左 1.57、往上 2，各乘 0.785 得 1.23 和 1.57，3 − 1.23 = 1.77、3.57 + 1.57 = 5.14，停在 (1.77, 5.14)；分四步、八步越来越贴着圆，一百步停在 (1, 4.02)，无穷多步正好停在 (1, 4)；每一小步乘同一个矩阵，无穷多步乘出来的叫矩阵指数；名字来自自然常数 e：存一块钱年利率百分之百，一年算一次利息是两块，分十二个月利滚利是 2.61，越分越接近 2.718；矩阵指数是同一个做法，滚起来的是位姿');

      /* ══ 左：网格 + 折线 ══ */
      // 这一幕没有 UR5：网格放大到一格 60 px，只画 x 0…4、y 0…6（要走的路都在肘的右上方，分两步那几段要画得开）
      var gridG = E.el('g', {}, s), u = 60;
      var pl = E.plane(gridG, { ox: 74, oy: 392, u: u, x0: 0, x1: 4, y0: 0, y1: 6 });
      var P = pl.P, Q = P(1, 2);
      function onArc(a) { return P(1 + 2 * Math.cos(a), 2 + 2 * Math.sin(a)); }
      var pts40 = []; for (var k = 0; k <= 40; k++) pts40.push(onArc(TH * k / 40));
      var arcGrey = K.paint(E.el('path', { d: E.poly(pts40), fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '8 7' }, gridG), null, E.INK2);
      var radius = K.paint(E.el('path', { d: E.poly([Q, P(3, 2)]), fill: 'none', 'stroke-width': 5, 'stroke-linecap': 'round' }, gridG), null, E.BODY);
      radius.style.opacity = 0.5;
      var tick45 = K.paint(E.el('path', { d: E.poly([onArc(HH), P(1 + 2.35 * Math.cos(HH), 2 + 2.35 * Math.sin(HH))]), fill: 'none', 'stroke-width': 3 }, gridG), null, E.OMEGA);
      var tick45L = E.halo(E.txt(gridG, P(1 + 1.55 * Math.cos(HH), 2 + 1.55 * Math.sin(HH))[0], P(1 + 1.55 * Math.cos(HH), 2 + 1.55 * Math.sin(HH))[1] + 7, '45°', 20, E.OMEGA, 'middle', 900));   // 写在弧里面：弧外是 x = 3 的竖线
      var paths = NS.map(function (n, i) {
        var p = K.paint(E.el('path', { d: E.poly(steps(n).map(function (q) { return P(q[0], q[1]); })), fill: 'none', 'stroke-width': n > 50 ? 3.5 : 4,
          'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, gridG), null, COLS[i]);
        if (i === 0) p.setAttribute('stroke-dasharray', '9 6');
        return p;
      });
      var arcOk = K.paint(E.el('path', { d: E.poly(pts40), fill: 'none', 'stroke-width': 5, 'stroke-linecap': 'round' }, gridG), null, E.GOOD);
      // 分两步：第一段、第二段（琥珀）分开画，好跟着旁白
      var c2 = steps(2), K1 = P(c2[1][0], c2[1][1]), K2 = P(c2[2][0], c2[2][1]), S0 = P(3, 2);
      var segA = K.paint(E.el('path', { d: E.poly([S0, K1]), fill: 'none', 'stroke-width': 5, 'stroke-linecap': 'round' }, gridG), null, E.WARN);
      var segB = K.paint(E.el('path', { d: E.poly([K1, K2]), fill: 'none', 'stroke-width': 5, 'stroke-linecap': 'round' }, gridG), null, E.WARN);
      var upA = E.vec(gridG, E.VEL, 4, { head: 13 }); upA.set(S0[0], S0[1], S0[0], S0[1] - 0.9 * u);
      var dimG = E.el('g', {}, gridG), dy = -0.42 * u;
      K.paint(E.el('path', { d: 'M ' + Q[0] + ' ' + (Q[1] - dy - 8) + ' V ' + (Q[1] - dy + 8) + ' M ' + Q[0] + ' ' + (Q[1] - dy) + ' H ' + S0[0] + ' M ' + S0[0] + ' ' + (Q[1] - dy - 8) + ' V ' + (Q[1] - dy + 8),
        fill: 'none', 'stroke-width': 2.5 }, dimG), null, E.INK);
      E.halo(E.txt(dimG, (Q[0] + S0[0]) / 2, Q[1] - dy + 26, '2 格', 20, E.INK, 'middle', 900));   // 量尺画在小臂下面：上面要画第一段
      var k1Dot = K.paint(E.el('circle', { cx: K1[0], cy: K1[1], r: 6.5, 'stroke-width': 2.5 }, gridG), E.S1, E.WARN);
      var k1Lab = E.halo(E.txt(gridG, K1[0] + 14, K1[1] + 8, '(3, 3.57)', 20, E.WARN, 'start', 900));
      // 第二步：肘 → 拐点的虚线（黄）、(2, 1.57)、直角、平移到拐点并转 90°（蓝弧）、红绿两段
      var spoke = E.vec(gridG, E.HI, 3, { head: 12, dash: '6 5' });
      var spokeLab = E.halo(E.txt(gridG, P(1.75, 2.75)[0], P(1.75, 2.75)[1], '(2, 1.57)', 20, E.HI, 'end', 900));   // 虚线的左上：右下是 x = 3 那条竖线
      var corner = K.paint(E.el('path', { fill: 'none', 'stroke-width': 2.5 }, gridG), null, E.INK);
      var spoke2 = E.vec(gridG, E.HI, 3, { head: 12, dash: '6 5' });      // 平移到拐点的那一份，绕拐点转 90°
      var turnArc = E.arcArrow(gridG, K1[0], K1[1], 0.62 * u, E.OMEGA, 3, { head: 11 });
      var legR = E.vec(gridG, E.X, 4.5, { head: 13 }), legG = E.vec(gridG, E.Y, 4.5, { head: 13 });
      var legRL = E.halo(E.txt(gridG, 0, 0, '', 20, E.X, 'middle', 900)), legGL = E.halo(E.txt(gridG, 0, 0, '', 20, E.Y, 'start', 900));
      var k2Dot = K.paint(E.el('circle', { cx: K2[0], cy: K2[1], r: 6.5, 'stroke-width': 2.5 }, gridG), E.S1, E.WARN);
      var k2Lab = E.halo(E.txt(gridG, K2[0] - 12, K2[1] - 12, '(1.77, 5.14)', 20, E.WARN, 'end', 900));
      var nearG = E.el('g', {}, gridG), endOk = P(1, 4);
      [P(3, 2 + TH * 2), K2].forEach(function (a) { K.paint(E.el('path', { d: E.poly([a, endOk]), fill: 'none', 'stroke-width': 2.2, 'stroke-dasharray': '4 5' }, nearG), null, E.INK2); });
      // 肘、出发点、终点标签
      var outOf = E.el('g', {}, gridG);
      K.paint(E.el('circle', { cx: Q[0], cy: Q[1], r: 12, 'stroke-width': 3.5 }, outOf), E.S1, E.HI);
      K.paint(E.el('circle', { cx: Q[0], cy: Q[1], r: 4 }, outOf), E.HI);
      K.paint(E.el('circle', { cx: S0[0], cy: S0[1], r: 7, 'stroke-width': 3 }, gridG), E.VEL, E.S1);
      var lab100 = E.halo(E.txt(gridG, endOk[0] - 22, endOk[1] - 40, '(1, 4.02)', 20, E.GOOD, 'end', 900));
      var labOk = E.halo(E.txt(gridG, endOk[0] - 22, endOk[1] - 16, '(1, 4)', 24, E.GOOD, 'end', 900));
      // 句 13：沿圆一小步一小步走的点 + 两根小箭头（位姿）
      var rollG = E.el('g', {}, gridG);
      var rAx = [E.vec(rollG, E.X, 4.5, { head: 13 }), E.vec(rollG, E.Y, 4.5, { head: 13 })];
      var rDot = K.paint(E.el('circle', { r: 8, 'stroke-width': 3 }, rollG), E.VEL, E.S1);

      /* ══ 左：复利条形图 ══ */
      var money = E.el('g', {}, s), X0 = 64, UX = 112, BH = 34, BY = [112, 212, 312];
      var bars = BY.map(function (y, i) {
        var g = E.el('g', {}, money);
        var base = K.paint(E.el('rect', { x: X0, y: y, width: UX, height: BH, rx: 6, 'stroke-width': 2 }, g), E.tint(E.INK2, 30, 'var(--demo-surface)'), E.INK2);
        var grow = K.paint(E.el('rect', { x: X0 + UX, y: y, width: 0, height: BH, rx: 6, 'stroke-width': 2 }, g), E.tint(E.HI, 30, 'var(--demo-surface)'), E.HI);
        var ticks = E.el('g', {}, g);
        var head = E.txt(g, X0, y - 12, ['一年算 1 次利息', '12 个月利滚利', '天天利滚利（365 次）'][i], 20, E.INK2, 'start', 700);
        var val = E.txt(g, 0, y + BH / 2 + 8, '', 22, E.HI, 'start', 900);
        return { g: g, base: base, grow: grow, ticks: ticks, head: head, val: val, y: y };
      });
      var one = E.txt(money, X0 + UX / 2, BY[0] + BH / 2 + 8, '1 块', 20, E.INK, 'middle', 900);
      for (var m = 1; m < 12; m++) {
        var v = Math.pow(1 + 1 / 12, m);
        K.paint(E.el('line', { x1: X0 + UX * v, x2: X0 + UX * v, y1: BY[1] + 4, y2: BY[1] + BH - 4, 'stroke-width': 1.5 }, bars[1].ticks), null, E.HI);
      }
      var eX = X0 + UX * Math.E;
      var limLine = K.paint(E.el('path', { d: 'M ' + eX.toFixed(1) + ' 84 V 362', fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '7 6' }, money), null, E.GOOD);
      var limLab = E.txt(money, eX - 10, 74, '2.718…', 22, E.GOOD, 'end', 900);   // 竖虚线顶上：底下是刻度，右边是右栏
      var eLab = E.txt(money, eX + 10, 76, 'e', 30, E.HI, 'start', 900);
      eLab.style.fontFamily = 'KaTeX_Math, serif'; eLab.style.fontStyle = 'italic';
      var scaleG = E.el('g', {}, money);
      K.paint(E.el('line', { x1: X0, x2: X0 + UX * 3, y1: 368, y2: 368, 'stroke-width': 2 }, scaleG), null, E.MUT);
      [0, 1, 2, 3].forEach(function (k) {
        K.paint(E.el('line', { x1: X0 + UX * k, x2: X0 + UX * k, y1: 362, y2: 374, 'stroke-width': 2 }, scaleG), null, E.MUT);
        if (k !== 0) E.txt(scaleG, X0 + UX * k, 396, k + ' 块', 18, E.MUT, 'middle', 700);
      });

      /* ══ 右栏 ══ */
      var RK = E.rowKit(), Rg = E.el('g', {}, s), X = RX + 10;
      var t0 = E.el('g', {}, Rg);
      var z1 = RK.line(t0, X, 80, [['分成几小步：', E.INK, 900]], 24);
      var z2 = RK.line(t0, X, 130, [['每走一小步 → ', E.INK2], ['停下来', E.WARN, 900]]);
      var z3 = RK.line(t0, X, 176, [['按这时的方向 → ', E.INK2], ['再走一步', E.WARN, 900]]);
      var t1 = E.el('g', {}, Rg), SZ = 20;
      var w1 = RK.line(t1, X, 40, [['分 2 步：', E.INK, 900], ['每步转 45°', E.OMEGA, 900]], SZ);
      var w2 = RK.line(t1, X, 74, [['1.57 ÷ 2 ≈ ', E.INK2], ['0.785 弧度', E.OMEGA, 900]], SZ);
      var w3 = RK.line(t1, X, 116, [['① ', E.WARN, 900], ['2 × 0.785 ≈ 1.57', E.INK, 900], [' → ', E.INK2], ['(3, 3.57)', E.WARN, 900]], SZ);
      var w4 = RK.line(t1, X, 158, [['② 离肘 ', E.WARN, 900], ['(2, 1.57)', E.HI, 900], ['，转 90°：', E.OMEGA, 900]], SZ);
      var w5 = RK.line(t1, X, 192, [['　 往左 ', E.INK2], ['1.57', E.X, 900], ['、往上 ', E.INK2], ['2', E.Y, 900]], SZ);
      var w6 = RK.line(t1, X, 226, [['　 × 0.785 → ', E.INK2], ['1.23', E.X, 900], [' 和 ', E.INK2], ['1.57', E.Y, 900]], SZ);
      var w7 = RK.line(t1, X, 268, [['3 − ', E.INK, 900], ['1.23', E.X, 900], [' = 1.77', E.INK, 900]], SZ);
      var w8 = RK.line(t1, X, 302, [['3.57 + ', E.INK, 900], ['1.57', E.Y, 900], [' = 5.14', E.INK, 900]], SZ);
      var w9 = RK.line(t1, X, 344, [['→ ', E.INK2], ['(1.77, 5.14)', E.WARN, 900], ['：折了一次', E.INK2]], 22);
      var t2 = E.el('g', {}, Rg);
      var r0 = RK.line(t2, X, 46, [['分几步', E.INK2, 700], ['　→　', E.MUT], ['停在', E.INK2, 700]], 21);
      var LABS = ['(3, 5.14)', '(1.77, 5.14)', '(1.20, 4.66)', '(1.05, 4.33)', '(1.00, 4.02)'];
      var rowsN = NS.map(function (n, i) {
        return RK.line(t2, X, 84 + 34 * i, [[n + ' 步', COLS[i], 900], ['　→　', E.MUT], [LABS[i], COLS[i], 900]], 21);
      });
      var pcRow = RK.line(t2, X, 84 + 34 * 5 + 6, [['4 步起：交给电脑一步一步算', E.INK2, 700]], 19);
      var rowInf = RK.line(t2, X, 84 + 34 * 6 + 14, [['无穷多步', E.GOOD, 900], ['　→　', E.MUT], ['(1, 4)', E.GOOD, 900]], 22);
      var infOk = E.check(rowInf.g, X + 230, 84 + 34 * 6 - 4, 20, E.GOOD, 4);
      var t3 = E.el('g', {}, Rg);
      var f1 = RK.line(t3, X, 44, [['每一小步：乘同一个矩阵', E.INK, 900]], 21);
      var f2 = E.el('g', {}, t3);
      E.tex(f2, X, 100, '\\left(I+\\tfrac{[\\mathcal S]\\,\\theta}{n}\\right)^{n}', 26, E.INK, 'start', { w: 170, h: 56 });
      var f3 = E.el('g', {}, t3);
      E.txt(f3, X + 186, 102, '分几步，乘几次', 19, E.INK2, 'start', 700);
      var f4 = E.el('g', {}, t3);
      K.paint(E.el('rect', { x: RX, y: 128, width: 340, height: 96, rx: 14, 'stroke-width': 3 }, f4), E.tint(E.HI, 10, 'var(--demo-surface)'), E.HI);
      E.tex(f4, X + 4, 166, 'n\\to\\infty:\\ \\textcolor{' + H.hi + '}{\\exp\\big([\\mathcal S]\\,\\theta\\big)}', 26, E.INK, 'start', { w: 240, h: 50 });
      var fName = E.txt(f4, RX + 170, 206, '矩阵指数', 24, E.HI, 'middle', 900);
      var lg = [['I', '：单位矩阵（第 5 集）', E.INK], ['[S]', '：螺旋轴排成的矩阵', E.WARN], ['θ', '：转过的角度', E.OMEGA], ['n', '：分几步', E.INK2]].map(function (r, i) {
        var g = E.el('g', {}, t3), y = 258 + 36 * i;
        var sym = E.txt(g, X + 30, y, r[0], 22, r[2], 'middle', 900);
        sym.style.fontFamily = 'KaTeX_Math, serif'; sym.style.fontStyle = r[0] === '[S]' ? 'normal' : 'italic';
        E.txt(g, X + 54, y, r[1], 20, E.INK, 'start', 700);
        return g;
      });
      var t4 = E.el('g', {}, Rg);
      var m1 = RK.line(t4, X, 50, [['为什么叫「指数」？', E.HI, 900]], 24);
      var m2 = RK.line(t4, X, 92, [['来自', E.INK2], ['自然常数 e', E.HI, 900]], 22);
      var m3 = RK.line(t4, X, 134, [['存 1 块，年利率 100%', E.INK, 900]], 22);
      var m4 = RK.line(t4, X, 176, [['一年算 1 次：', E.INK2], ['1 + 1 = 2', E.HI, 900]], 21);
      var m5 = RK.line(t4, X, 222, [['12 个月：每月 × ', E.INK2], ['(1 + 1/12)', E.HI, 900]], 21);
      var m6 = RK.line(t4, X, 258, [['乘 12 次 → ', E.INK2], ['2.61', E.HI, 900]], 21);
      var m7 = RK.line(t4, X, 300, [['365 次 → ', E.INK2], ['2.71', E.HI, 900]], 21);
      var m8 = RK.line(t4, X, 344, [['越分越细 → ', E.INK2], ['2.718… = e', E.GOOD, 900]], 22);
      var t5 = E.el('g', {}, Rg);
      var x1 = E.el('g', {}, t5);
      E.tex(x1, X, 66, 'e^{x}', 34, E.HI, 'start', { w: 60, h: 56 });
      E.txt(x1, X + 64, 66, '：自然常数的 x 次方，叫指数函数', 19, E.INK, 'start', 700);
      var x2 = E.el('g', {}, t5);
      E.txt(x2, X, 126, '同一个做法：', 22, E.INK, 'start', 900);
      E.tex(x2, X, 182, '\\left(1+\\tfrac{x}{n}\\right)^{n}\\to e^{x}', 26, E.INK, 'start', { w: 260, h: 50 });
      E.tex(x2, X, 246, '\\left(I+\\tfrac{[\\mathcal S]\\theta}{n}\\right)^{n}\\to \\textcolor{' + H.hi + '}{\\exp([\\mathcal S]\\theta)}', 26, E.INK, 'start', { w: 330, h: 50 });
      var x3 = RK.line(t5, X, 316, [['滚起来的不是钱，', E.INK2, 700]], 22);
      var x3x = E.cross(x3.g, X + 196, 308, 9, E.ERR, 4);
      var x4 = RK.line(t5, X, 356, [['是', E.INK2, 700], ['位姿', E.HI, 900]], 24);
      var x4ok = E.check(x4.g, X + 80, 338, 22, E.GOOD, 4);

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }
      function setLeg(vec, lab, a, b, txt, side) {
        vec.set(a[0], a[1], b[0], b[1]);
        lab.textContent = txt;
        var mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2;
        if (side === 'above') { lab.setAttribute('x', mx.toFixed(1)); lab.setAttribute('y', (my - 10).toFixed(1)); }
        else { lab.setAttribute('x', (mx + 10).toFixed(1)); lab.setAttribute('y', (my + 7).toFixed(1)); }
      }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) { RK.layout(); laid = true; }
          var f = E.fade, b = E.bump;

          /* ── 左：网格 ↔ 复利 ── */
          E.op(gridG, Math.min(1, (1 - f(st, T.out2, 0.4)) + f(st, T.back, 0.5)));
          E.op(money, f(st, T.out2 + 0.3, 0.4) * (1 - f(st, T.back - 0.3, 0.4)));
          var back = st > T.back;

          /* ── 分两步（句 2–6） ── */
          var two = 1 - f(st, T.out1, 0.4);
          E.op(tick45, f(st, T.deg, 0.3) * (1 - f(st, T.first, 0.3))); E.op(tick45L, f(st, T.deg, 0.3) * (1 - f(st, T.first, 0.3)));
          E.op(dimG, f(st, T.r2, 0.3) * (1 - f(st, T.upA + 0.6, 0.3)));
          E.op(upA.g, f(st, T.upA, 0.3) * (1 - f(st, T.seg1[1], 0.3)));
          E.drawOn(segA, f(st, T.seg1[0], T.seg1[1] - T.seg1[0])); E.op(segA, (f(st, T.seg1[0], 0.1) > 0 ? 1 : 0) * two);
          E.op(k1Dot, f(st, T.k1, 0.3) * two); k1Dot.setAttribute('r', (6.5 + 4 * b(st, T.kink, 0.8)).toFixed(2));
          E.op(k1Lab, f(st, T.k1, 0.3) * (1 - f(st, T.legs - 0.3, 0.3)));
          // 肘 → 拐点的虚线（黄），旁边写 (2, 1.57)
          var sp = f(st, T.legs, 0.5);
          spoke.set(Q[0], Q[1], Q[0] + (K1[0] - Q[0]) * sp, Q[1] + (K1[1] - Q[1]) * sp);
          var stepOff = 1 - f(st, T.go[0], 0.4);
          E.op(spoke.g, (sp > 0 ? 1 : 0) * stepOff); E.op(spokeLab, f(st, T.legsB, 0.3) * stepOff);
          // 直角：拐点处，「连线方向（指回肘）」和「前进方向」
          var dd = [c2[1][0] - 1, c2[1][1] - 2], dl = Math.hypot(dd[0], dd[1]), ud = [-dd[0] / dl, -dd[1] / dl], nd = [-dd[1] / dl, dd[0] / dl], mm = 0.28;
          var kq = c2[1];
          corner.setAttribute('d', E.poly([P(kq[0] + ud[0] * mm, kq[1] + ud[1] * mm), P(kq[0] + (ud[0] + nd[0]) * mm, kq[1] + (ud[1] + nd[1]) * mm), P(kq[0] + nd[0] * mm, kq[1] + nd[1] * mm)]));
          E.op(corner, f(st, T.perp2, 0.3) * stepOff);
          // 平移到拐点的那一份（尾巴在拐点），绕拐点转 90°（蓝弧）
          var tr = f(st, T.turn[0], T.turn[1] - T.turn[0]), ang = Math.atan2(dd[1], dd[0]) + TH * tr;
          var sp2 = P(kq[0] + dl * Math.cos(ang), kq[1] + dl * Math.sin(ang));
          spoke2.set(K1[0], K1[1], sp2[0], sp2[1]);
          E.op(spoke2.g, f(st, T.perp, 0.3) * (1 - f(st, T.left, 0.3)));
          turnArc.set(Math.atan2(dd[1], dd[0]) + 4 * D, Math.max(Math.atan2(dd[1], dd[0]) + 5 * D, ang - 4 * D));
          E.op(turnArc.g, f(st, T.turn[0], 0.2) * (1 - f(st, T.left + 0.6, 0.3)));
          // 红（往左）、绿（往上）两段：先是整份，句 5 缩成 0.785 倍，头尾相接
          var sc = 1 - (1 - HH) * f(st, T.scale[0], T.scale[1] - T.scale[0]);
          var lx = -dd[1] * sc, ly = dd[0] * sc, a0 = kq, a1 = [kq[0] + lx * f(st, T.left, 0.4), kq[1]], a2 = [a1[0], kq[1] + ly * f(st, T.up2, 0.4)];
          setLeg(legR, legRL, P(a0[0], a0[1]), P(a1[0], a1[1]), '往左 ' + (st >= T.scale[1] - 0.2 ? '1.23' : '1.57'), 'above');
          setLeg(legG, legGL, P(a1[0], a1[1]), P(a2[0], a2[1]), '往上 ' + (st >= T.scale[1] - 0.2 ? '1.57' : '2'), 'right');
          var legOn = 1 - f(st, T.stopAt, 0.4);
          E.op(legR.g, (f(st, T.left, 0.1) > 0 ? 1 : 0) * legOn); E.op(legG.g, (f(st, T.up2, 0.1) > 0 ? 1 : 0) * legOn);
          E.op(legRL, 0); E.op(legGL, 0);   // 红绿两段的长度只写在右栏（同色）：小网格上写在段旁会压到圆弧和别的段
          E.drawOn(segB, f(st, T.go[0], T.go[1] - T.go[0])); E.op(segB, (f(st, T.go[0], 0.1) > 0 ? 1 : 0) * two);
          E.op(k2Dot, f(st, T.stopAt, 0.3) * two); E.op(k2Lab, f(st, T.stopAt, 0.3) * two);
          E.op(nearG, f(st, T.near, 0.4) * two);

          /* ── 折线 4 / 8 / 100（句 7–8） ── */
          var drawT = [null, null, T.n4, T.n8, T.n100];
          paths.forEach(function (p, i) {
            if (i === 1) { E.op(p, 0); return; }                            // 分两步那条由上面的两段画
            var on = i === 0 ? 1 : f(st, drawT[i], i === 4 ? 0.8 : 0.5);
            if (i > 0) E.drawOn(p, on);
            var later = [2, 3, 4].map(function (j) { return j > i ? f(st, drawT[j], 0.4) : 0; }).reduce(function (x, y) { return Math.max(x, y); }, 0);
            var dim = 1 - 0.6 * Math.max(later, f(st, T.arc, 0.5));
            p.style.opacity = (i === 0 ? 0.85 : 1) * dim * (1 - f(st, T.out2, 0.3)) * (on > 0 ? 1 : 0);   // 没画出来的折线不留圆头线帽的点
            if (back) p.style.opacity = i === 4 ? 0.45 : 0;
          });
          segA.style.opacity = +segA.style.opacity * (1 - 0.6 * f(st, T.n4, 0.4)); segB.style.opacity = +segB.style.opacity * (1 - 0.6 * f(st, T.n4, 0.4));
          if (st > T.out1) { E.op(segA, (1 - 0.6 * f(st, T.n4, 0.4)) * (1 - 0.6 * f(st, T.arc, 0.5)) * (1 - f(st, T.out2, 0.3)) * (back ? 0 : 1)); E.op(segB, +segA.style.opacity); }
          E.op(arcGrey, (1 - f(st, T.arc, 0.4)) * (1 - f(st, T.out2, 0.3)));
          var hb = b(st, T.hug, 1.0);
          arcGrey.style.strokeWidth = 3 + 2.5 * hb; E.glow(arcGrey, E.INK2, hb > 0.05);
          E.drawOn(arcOk, f(st, T.arc, 0.9));
          E.op(lab100, f(st, T.at100, 0.3) * (1 - f(st, T.at - 0.4, 0.3)));
          E.op(labOk, f(st, T.at, 0.3) * (1 - f(st, T.out2, 0.3)));
          // 句 13：一小步一小步滚
          var rk = Math.min(16, Math.floor(16 * E.clamp((st - T.roll[0]) / (T.roll[1] - T.roll[0]), 0, 1)));
          var a16 = TH * rk / 16, rp = onArc(a16);
          rDot.setAttribute('cx', rp[0].toFixed(1)); rDot.setAttribute('cy', rp[1].toFixed(1));
          var ro = [Math.cos(a16), Math.sin(a16)], rt = [-Math.sin(a16), Math.cos(a16)];
          rAx[0].set(rp[0], rp[1], rp[0] + ro[0] * 0.9 * u, rp[1] - ro[1] * 0.9 * u);
          rAx[1].set(rp[0], rp[1], rp[0] + rt[0] * 0.9 * u, rp[1] - rt[1] * 0.9 * u);
          E.op(rollG, f(st, T.roll[0] - 0.4, 0.3));
          rAx.forEach(function (a) { E.op(a.g, f(st, T.pose, 0.3)); });
          E.glow(rDot, E.HI, b(st, T.pose, 1) > 0.05);

          /* ── 左：复利 ── */
          E.op(bars[0].g, f(st, T.coin, 0.3)); E.op(one, f(st, T.coin, 0.3));
          var g0 = f(st, T.onePlus, 1.0);
          bars[0].grow.setAttribute('width', (UX * g0).toFixed(1));
          bars[0].val.textContent = '2'; bars[0].val.setAttribute('x', (X0 + UX * (1 + g0) + 10).toFixed(1)); E.op(bars[0].val, f(st, T.two2, 0.3));
          E.op(bars[1].g, f(st, T.m12, 0.3));
          var k12 = Math.floor(12 * E.clamp((st - T.k12[0]) / (T.k12[1] - T.k12[0]), 0, 1) + 1e-9), v12 = Math.pow(1 + 1 / 12, k12);
          bars[1].grow.setAttribute('width', (UX * (v12 - 1)).toFixed(1));
          Array.prototype.forEach.call(bars[1].ticks.children, function (ln, j) { E.op(ln, j + 1 < k12 ? 0.8 : 0); });
          bars[1].val.textContent = '2.61'; bars[1].val.setAttribute('x', (eX + 12).toFixed(1)); E.op(bars[1].val, f(st, T.v12, 0.3));   // 条尾离竖虚线只有 12 px：数写到虚线右边
          E.op(bars[2].g, f(st, T.fine[0] - 0.2, 0.3));
          var g3 = f(st, T.fine[0], T.fine[1] - T.fine[0]), v365 = Math.pow(1 + 1 / 365, 365);
          bars[2].grow.setAttribute('width', (UX * (v365 - 1) * g3).toFixed(1));
          bars[2].val.textContent = '2.71'; bars[2].val.setAttribute('x', (eX + 12).toFixed(1)); E.op(bars[2].val, f(st, T.fine[1], 0.3));
          E.drawOn(limLine, f(st, T.lim, 0.5)); E.op(limLab, f(st, T.lim, 0.3));
          E.pop(eLab, eX + 18, 66, f(st, T.eNum, 0.35), 0.6);
          E.op(scaleG, f(st, T.coin, 0.3));

          /* ── 右栏 ── */
          E.op(t0, 1 - f(st, T.out0, 0.4));
          E.op(z1.g, fr(st, 0.6)); E.op(z2.g, fr(st, T.each)); E.op(z2.parts[1], fr(st, T.stop)); E.op(z3.g, fr(st, T.dir)); E.op(z3.parts[1], fr(st, T.next));
          E.op(t1, f(st, T.out0 + 0.2, 0.4) * (1 - f(st, T.out1, 0.4)));
          E.op(w1.g, fr(st, T.two)); E.op(w1.parts[1], fr(st, T.deg)); E.op(w2.g, fr(st, T.div)); E.op(w2.parts[1], fr(st, T.h));
          E.op(w3.g, fr(st, T.mul1)); E.op(w3.parts[2], fr(st, T.k1)); E.op(w3.parts[3], fr(st, T.k1));
          E.op(w4.g, fr(st, T.legs)); E.op(w4.parts[2], fr(st, T.turn[0]));
          E.op(w5.g, fr(st, T.left)); E.op(w5.parts[2], fr(st, T.up2)); E.op(w5.parts[3], fr(st, T.up2));
          E.op(w6.g, fr(st, T.scale[0])); E.op(w6.parts[1], fr(st, T.r123)); E.op(w6.parts[2], fr(st, T.r157)); E.op(w6.parts[3], fr(st, T.r157));
          E.op(w7.g, fr(st, T.sub)); E.op(w7.parts[2], fr(st, T.sub2));
          E.op(w8.g, fr(st, T.add)); E.op(w8.parts[2], fr(st, T.add2));
          E.op(w9.g, fr(st, T.stopAt)); E.op(w9.parts[2], fr(st, T.kink));
          E.op(t2, f(st, T.out1 + 0.2, 0.4) * (1 - f(st, T.one - 0.3, 0.4)));
          E.op(r0.g, 1);
          var rowT = [0, 0, T.n4, T.n8, T.at100];
          rowsN.forEach(function (r, i) { E.op(r.g, i < 2 ? 1 : fr(st, rowT[i])); });
          E.op(pcRow.g, fr(st, T.pc));
          E.op(rowInf.g, fr(st, T.inf)); E.op(rowInf.parts[2], fr(st, T.at)); E.drawOn(infOk, f(st, T.at + 0.3, 0.3));
          E.op(t3, f(st, T.one - 0.1, 0.4) * (1 - f(st, T.out2, 0.4)));
          E.op(f1.g, fr(st, T.one)); E.op(f2, fr(st, T.same)); E.op(f3, fr(st, T.times)); E.pop(f4, RX + 170, 176, f(st, T.nInf, 0.35), 0.9);
          E.op(fName, fr(st, T.name));
          lg.forEach(function (g, i) { E.op(g, fr(st, T.legend[i])); });
          E.op(t4, f(st, T.out2 + 0.3, 0.4) * (1 - f(st, T.pow - 0.3, 0.4)));
          E.op(m1.g, fr(st, T.why)); E.op(m2.g, fr(st, T.e0)); E.op(m3.g, fr(st, T.coin)); E.op(m4.g, fr(st, T.once)); E.op(m4.parts[1], fr(st, T.onePlus));
          E.op(m5.g, fr(st, T.m12)); E.op(m5.parts[1], fr(st, T.perM)); E.op(m6.g, fr(st, T.k12[0])); E.op(m6.parts[1], fr(st, T.v12));
          E.op(m7.g, fr(st, T.fine[1])); E.op(m8.g, fr(st, T.lim));
          E.op(t5, f(st, T.pow, 0.4));
          E.op(x1, fr(st, T.pow)); E.op(x2, fr(st, T.same2));
          E.op(x3.g, fr(st, T.money)); E.drawOn(x3x, f(st, T.money + 0.2, 0.3));
          E.op(x4.g, fr(st, T.pose)); E.drawOn(x4ok, f(st, T.pose + 0.2, 0.3));
        }
      };
    }
  };
})();
