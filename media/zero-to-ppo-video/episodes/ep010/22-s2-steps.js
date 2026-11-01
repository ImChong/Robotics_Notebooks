// 010 第 2 幕：切成无穷多小步——矩阵指数。左边同第 1 幕的网格（肘在 (1, 2)，小臂上的点从 (3, 2) 出发，要走的四分之一圆描成灰虚线），
// 右栏一行行写算式；中段左边换成复利的条形图；最后换回网格。
// 2026-11-01 按 009 的用户反馈「全篇的计算都展开讲不要跳过」「纯公式推导没有画面观众会看不懂」重做：分两步的每一步都算出来、在网格上画出来，
// 式子里的符号在屏幕上逐个说明；配色同第 1 幕（转动蓝、点紫、走歪的橙）。
// 同日再按用户反馈「计算步骤需要展开，不要直接给结果」：两步里的加减、转 90°（照第 3 集按列拿份：第一列红、第二列绿，同第 3 幕）、
// 两段各乘 0.785 都逐项算；分四步也逐步算（每步 1.57 ÷ 4 ≈ 0.39，数留两位小数，照算到 (1.22, 4.65)，电脑精确算是 (1.20, 4.66)），
// 网格上一段一段画；八步、一百步说明交给电脑；复利把头两个月的乘法算出来。
// 句 1「每走一小步」「停下来」「这时候的方向」「再走下一步」右栏三行；
// 句 2「分两步」「每步转四十五度」四分之一圈上 45° 处标一道，「一点五七除以二」「零点七八五」右栏写出；
// 句 3「离肘两格」量出 2 格，「朝上走」出朝上的小箭头，「二乘零点七八五」「一点五七格」右栏写出、网格上画第一段（琥珀），「二加一点五七」「三点五七」写出；
// 句 4「离肘多远」肘到拐点画虚线（黄），「三减一」「三点五七减二」右栏写出，「右边两格、上边一点五七格」虚线旁写 (2, 1.57)；
// 句 5「垂直」拐点标直角，「转九十度」虚线平移到拐点、绕拐点逆时针转 90°（蓝弧画满 90°），「第一列拿两份」红的一段往上 2，
//    「第二列拿一点五七份」绿的一段接在红的尖上往左 1.57（头尾相接）；
// 句 6「加起来」右栏写出 (−1.57, 2)，「各乘零点七八五」红绿两段缩成 0.785 倍，「约一点五七」「约一点二三」右栏写出；
// 句 7「走过去」画第二段（琥珀），「三减一点二三」「三点五七加上一点五七」右栏写出；
// 句 8「一点七七、五点一四」终点写出，「折了一次」拐点鼓一下，「离圆近了一些」终点和第 1 幕的 (3, 5.14) 各连一条到 (1, 4) 的灰虚线；
// 句 9「分四步」右栏换成四步的算式，「二十二点五度」「一点五七除以四」「零点三九」写出，「每一步都照第二步那样」写出做法；
// 句 10–13 每一步：肘到这一步起点画虚线（黄），「转九十度」虚线平移到起点、转 90°，「乘零点三九」缩成 0.39 倍，「加上去」画这一段（灰白）、标出新点；
// 句 14「加上肘」右栏逐项加，「停在」写 (1.22, 4.65)；
// 句 15「分八步」画八步的折线，「交给电脑」右栏写，「八步停在」写出，「贴着圆走」灰虚线亮一下；
// 句 16「一百步」画第五条，「一、四点零二」写出，「无穷多步」绿色圆弧画出，「一、四」打勾；
// 句 17「乘同一个矩阵」右栏出一步的矩阵，「乘上几次」出 n 次方，「无穷多步」n → ∞，「矩阵指数」写出 exp([S]θ)，下面逐个说明 I、[S]、θ、n；
// 句 18–21 复利（2026-11-01 按一手资料核对：不说「高中学过」，名字说成「从指数函数来」）：「存一块钱」第一条先有 1 块，「一加一」长到 2；
//    「十二个月」第二条出来，「一加十二分之一」「一点零八三」右栏写出，「第一个月后」「第二个月」各长一格、右栏写出乘法，
//    「乘满十二次」再一格一格长到 2.61；「按天算」第三条长到 2.71，「二点七一八」画出竖虚线，「自然常数」写 e；
// 句 22「几次方，叫指数函数」右栏写 eˣ，「矩阵指数的名字」左边换回网格，「同一个做法」两行对照，「一步一步滚起来」一个点沿圆一小步一小步走，
//    「不是钱，是位姿」点上长出两根小箭头（位置 + 朝向）。
// 时刻取自配音的逐词时间，见 ../ep010.py。
(function () {
  var E = window.EP010, K = E.K, H = E.HEX;
  var A = E.ARM, RX = A.RX;
  var T = {
    each: 1.97, stop: 3.08, dir: 3.71, next: 4.89, out0: 6.7,
    two: 7.26, deg: 8.12, div: 9.15, h: 10.85,
    first: 13.34, r2: 13.96, upA: 14.81, mul1: 15.42, len1: 17.03, seg1: [17.03, 18.1], add1: 18.29, add1v: 19.67, k1: 20.96,
    second: 23.51, legs: 24.8, subA: 26.22, subAv: 26.93, subB: 27.41, subBv: 28.68, legsB: 29.64,
    perp: 33.04, perp2: 34.27, turn: [36.43, 37.3], mat: 37.5, col1: 38.72, col1v: 39.94, col2: 40.9, col2v: 42.65,
    sum: 45.26, sumV: 46.17, scale: [48.05, 49.6], r157: 49.81, r157v: 51.47, r123: 52.49, r123v: 54.78,
    go: [56.71, 57.5], sub: 57.57, sub2: 59.07, add: 60.07, add2: 62.29,
    stopAt: 64.17, kink: 66.32, near: 67.12, out1: 68.9,
    n4: 69.21, d22: 70.49, d4: 71.95, d4v: 73.63, recipe: 75.39,
    st4: [
      { start: 82.12, rot: 83.35, rotV: 84.42, mul: 85.3, mulV: 86.73, add: 88.14, addX: 88.89, addY: 89.53, addV: 89.53 },
      { start: 91.53, rot: 92.15, rotV: 93.2, mul: 94.78, mulV: 96.02, add: 97.94, addX: 98.72, addY: 100.74, addV: 103.03 },
      { start: 105.0, rot: 105.72, rotV: 106.76, mul: 108.47, mulV: 109.98, add: 111.99, addX: 112.76, addY: 115.55, addV: 117.36 },
      { start: 119.31, rot: 120.07, rotV: 121.19, mul: 123.1, mulV: 124.39, add: 126.46, addX: 127.19, addY: 130.01, addV: 132.28 }
    ],
    elb: 134.27, ex: 135.86, exV: 137.07, ey: 138.24, eyV: 139.4, at4: 140.84, near4: 143.26, out4: 144.9,
    n8: 145.22, pc: 148.76, at8: 150.57, hug: 153.15,
    n100: 155.99, at100: 157.17, inf: 160.36, arc: 161.79, at: 163.29,
    one: 165.69, same: 167.26, times: 168.93, nInf: 170.45, name: 172.68, legend: [166.41, 167.41, 168.41, 169.81],
    out2: 175.0, why: 175.51, e0: 176.53, coin: 177.89, rate: 179.01, once: 180.33, onePlus: 181.68, two2: 182.43,
    m12: 184.24, cmp: 186.25, perM: 187.84, r1083: 189.62, mo1: 191.73, mo1v: 194.11, mo2: 195.23, mo2v: 197.4,
    k12: [198.39, 199.66], v12: 200.23, day: 202.17, fine: [202.93, 204.4], d365: 204.57, lim: 206.87, eNum: 208.65,
    pow: 210.83, fn: 212.23, back: 212.89, same2: 215.32, roll: [216.37, 219.04], money: 220.22, pose: 220.98
  };
  var END = 222.43, TH = Math.PI / 2, NS = [1, 2, 4, 8, 100], HH = TH / 2;
  var COLS = [E.ERR, E.WARN, E.INK2, E.INK, E.GOOD];
  // 分四步照算（每步 × 0.39，数留两位小数）：离肘的位置 d0 … d4
  var D4 = [[2, 0], [2, 0.78], [1.7, 1.56], [1.09, 2.22], [0.22, 2.65]], F4 = 0.39;

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
      { at: 13.25, s: '第一步：2 × 0.785 ≈ 1.57，2 + 1.57 = 3.57，走到 (3, 3.57)' },
      { at: 23.42, s: '第二步：离肘 3 − 1 = 2，3.57 − 2 = 1.57' },
      { at: 32.95, s: '转 90°：第一列拿 2 份 (0, 2)，第二列拿 1.57 份 (−1.57, 0)' },
      { at: 45.17, s: '加起来 (−1.57, 2)；各乘 0.785：2 × 0.785 ≈ 1.57，1.57 × 0.785 ≈ 1.23' },
      { at: 56.62, s: '走过去：3 − 1.23 = 1.77，3.57 + 1.57 = 5.14' },
      { at: 64.08, s: '停在 (1.77, 5.14)：折了一次，离圆近了一些' },
      { at: 69.12, s: '分 4 步：1.57 ÷ 4 ≈ 0.39；每步：离肘的位置转 90°，× 0.39，加上去' },
      { at: 82.03, s: '① (2, 0) 转 90° 是 (0, 2)，× 0.39 = (0, 0.78)，加上去是 (2, 0.78)' },
      { at: 91.44, s: '② (−0.78, 2) × 0.39 = (−0.3, 0.78)；2 − 0.3 = 1.7，0.78 + 0.78 = 1.56' },
      { at: 104.9, s: '③ (−1.56, 1.7) × 0.39 = (−0.61, 0.66)；1.7 − 0.61 = 1.09，1.56 + 0.66 = 2.22' },
      { at: 119.21, s: '④ (−2.22, 1.09) × 0.39 = (−0.87, 0.43)；1.09 − 0.87 = 0.22，2.22 + 0.43 = 2.65' },
      { at: 134.18, s: '加上肘：1 + 0.22 = 1.22，2 + 2.65 = 4.65，停在 (1.22, 4.65)' },
      { at: 145.13, s: '8 步、100 步：同样的算法交给电脑算；8 步停在 (1.05, 4.33)' },
      { at: 155.9, s: '100 步停在 (1, 4.02)，无穷多步正好停在 (1, 4)' },
      { at: 165.6, s: '每一小步乘同一个矩阵；无穷多步乘出来的，叫**矩阵指数**' },
      { at: 175.15, s: '名字要从自然常数 e 说起：存 1 块，年利率 100%，一年算 1 次：1 + 1 = 2 块' },
      { at: 184.15, s: '12 个月利滚利（复利）：每月乘 (1 + 1/12) ≈ 1.083' },
      { at: 191.64, s: '1 × 1.083 = 1.083，再乘 1.083 ≈ 1.17；乘满 12 次是 2.61' },
      { at: 202.08, s: '按天算乘 365 次约 2.71；越分越细，接近 2.718 = e' },
      { at: 210.46, s: 'eˣ 叫指数函数，**矩阵指数**的名字从这来：同一个做法，滚起来的是**位姿**' }
    ],
    build: function (K) {
      var s = K.sceneSvg('小臂上的点绕肘转九十度：分成几小步，每走一小步就按这时候的方向再走下一步；分两步每步转 45°，约 0.785 弧度：第一步朝上走 2 × 0.785 ≈ 1.57 格，2 + 1.57 = 3.57，到 (3, 3.57)；第二步离肘 3 − 1 = 2、3.57 − 2 = 1.57，前进方向和连线垂直，把 (2, 1.57) 转 90°：第一列拿两份 (0, 2)、第二列拿 1.57 份 (−1.57, 0)，加起来 (−1.57, 2)，各乘 0.785 得 1.57 和 1.23，3 − 1.23 = 1.77、3.57 + 1.57 = 5.14，停在 (1.77, 5.14)；分四步每步约 0.39 弧度，每步把离肘的位置转 90°、乘 0.39、加上去，四步算到 (0.22, 2.65)，加上肘得 (1.22, 4.65)；八步、一百步交给电脑算，一百步停在 (1, 4.02)，无穷多步正好停在 (1, 4)；每一小步乘同一个矩阵，无穷多步乘出来的叫矩阵指数；名字要从自然常数 e 说起：存一块钱年利率百分之百，一年算一次利息是两块，分十二个月利滚利（复利）每月乘 1.083，1 × 1.083 = 1.083、再乘 1.083 ≈ 1.17，乘满十二次是 2.61，按天算是 2.71，越分越接近 2.718；e 的几次方叫指数函数，矩阵指数的名字从这来，是同一个做法，滚起来的是位姿');

      /* ══ 左：网格 + 折线 ══ */
      // 这一幕没有 UR5：网格放大到一格 60 px，只画 x 0…4、y 0…6（要走的路都在肘的右上方，分两步那几段要画得开）
      var gridG = E.el('g', {}, s), u = 60;
      var pl = E.plane(gridG, { ox: 74, oy: 400, u: u, x0: 0, x1: 4, y0: 0, y1: 6 });
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
      // 转 90° 按列拿份（同第 3 幕）：第一列 (0, 1) 拿 2 份 → 红，往上；第二列 (−1, 0) 拿 1.57 份 → 绿，接在红的尖上往左
      var legR = E.vec(gridG, E.X, 4.5, { head: 13 }), legG = E.vec(gridG, E.Y, 4.5, { head: 13 });
      var k2Dot = K.paint(E.el('circle', { cx: K2[0], cy: K2[1], r: 6.5, 'stroke-width': 2.5 }, gridG), E.S1, E.WARN);
      var k2Lab = E.halo(E.txt(gridG, K2[0] - 12, K2[1] - 12, '(1.77, 5.14)', 20, E.WARN, 'end', 900));
      var nearG = E.el('g', {}, gridG), endOk = P(1, 4);
      [P(3, 2 + TH * 2), K2].forEach(function (a) { K.paint(E.el('path', { d: E.poly([a, endOk]), fill: 'none', 'stroke-width': 2.2, 'stroke-dasharray': '4 5' }, nearG), null, E.INK2); });
      // 分四步：照算的四段（灰白）、肘到这一步起点的虚线（黄）、转 90° 再缩成 0.39 倍的那一份（黄虚线）、各步的新点
      var V4 = D4.map(function (d) { return P(1 + d[0], 2 + d[1]); });
      var seg4 = [0, 1, 2, 3].map(function (k) {
        return K.paint(E.el('path', { d: E.poly([V4[k], V4[k + 1]]), fill: 'none', 'stroke-width': 4.5, 'stroke-linecap': 'round' }, gridG), null, E.INK2);
      });
      var dot4 = [1, 2, 3, 4].map(function (k) { return K.paint(E.el('circle', { cx: V4[k][0], cy: V4[k][1], r: 5.5, 'stroke-width': 2.5 }, gridG), E.S1, E.INK2); });
      var spoke4 = E.vec(gridG, E.HI, 3, { head: 11, dash: '6 5' });
      var turn4 = E.vec(gridG, E.HI, 3, { head: 11, dash: '6 5' });
      var turn4Arc = E.arcArrow(gridG, 0, 0, 0.45 * u, E.OMEGA, 3, { head: 10 });
      var lab4 = E.halo(E.txt(gridG, V4[4][0] - 14, V4[4][1] - 14, '(1.22, 4.65)', 20, E.INK2, 'end', 900));
      // 肘、出发点、终点标签
      var outOf = E.el('g', {}, gridG);
      K.paint(E.el('circle', { cx: Q[0], cy: Q[1], r: 12, 'stroke-width': 3.5 }, outOf), E.S1, E.HI);
      K.paint(E.el('circle', { cx: Q[0], cy: Q[1], r: 4 }, outOf), E.HI);
      K.paint(E.el('circle', { cx: S0[0], cy: S0[1], r: 7, 'stroke-width': 3 }, gridG), E.VEL, E.S1);
      var lab100 = E.halo(E.txt(gridG, endOk[0] - 22, endOk[1] - 40, '(1, 4.02)', 20, E.GOOD, 'end', 900));
      var labOk = E.halo(E.txt(gridG, endOk[0] - 22, endOk[1] - 16, '(1, 4)', 24, E.GOOD, 'end', 900));
      // 句 22：沿圆一小步一小步走的点 + 两根小箭头（位姿）
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
        var head = E.txt(g, X0, y - 12, ['一年算 1 次利息', '12 个月利滚利（复利）', '天天利滚利（365 次）'][i], 20, E.INK2, 'start', 700);
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
      // 分两步（逐项算）
      var t1 = E.el('g', {}, Rg), SZ = 19;
      var w1 = RK.line(t1, X, 36, [['分 2 步：', E.INK, 900], ['每步转 45°', E.OMEGA, 900]], SZ);
      var w2 = RK.line(t1, X, 66, [['1.57 ÷ 2 ≈ ', E.INK2], ['0.785 弧度', E.OMEGA, 900]], SZ);
      var w3 = RK.line(t1, X, 100, [['① ', E.WARN, 900], ['2 × 0.785 ≈ 1.57', E.INK, 900]], SZ);
      var w3b = RK.line(t1, X, 130, [['　 2 + 1.57 = 3.57', E.INK, 900], [' → ', E.INK2], ['(3, 3.57)', E.WARN, 900]], SZ);
      var w4 = RK.line(t1, X, 164, [['② 离肘：', E.WARN, 900], ['3 − 1 = 2', E.INK, 900], ['，', E.INK2], ['3.57 − 2 = 1.57', E.INK, 900]], SZ);
      var w5 = RK.line(t1, X, 196, [['转 90°：', E.OMEGA, 900], ['2 × (0, 1)', E.X, 900], [' + ', E.INK2], ['1.57 × (−1, 0)', E.Y, 900]], SZ);
      var w5b = RK.line(t1, X, 228, [['　= ', E.INK2], ['(0, 2)', E.X, 900], [' + ', E.INK2], ['(−1.57, 0)', E.Y, 900], [' = (−1.57, 2)', E.INK, 900]], SZ);
      var w6 = RK.line(t1, X, 262, [['× 0.785：', E.INK2], ['2 → 1.57', E.X, 900], ['，', E.INK2], ['1.57 → 1.23', E.Y, 900]], SZ);
      var w7 = RK.line(t1, X, 296, [['3 − ', E.INK, 900], ['1.23', E.Y, 900], [' = 1.77', E.INK, 900]], SZ);
      var w8 = RK.line(t1, X, 326, [['3.57 + ', E.INK, 900], ['1.57', E.X, 900], [' = 5.14', E.INK, 900]], SZ);
      var w9 = RK.line(t1, X, 360, [['→ ', E.INK2], ['(1.77, 5.14)', E.WARN, 900], ['：折了一次', E.INK2]], 21);
      // 分四步（逐步算）
      var t1b = E.el('g', {}, Rg), S4 = 17;
      var q1 = RK.line(t1b, X, 34, [['分 4 步：', E.INK, 900], ['1.57 ÷ 4 ≈ ', E.INK2], ['0.39 弧度', E.OMEGA, 900]], 19);
      var q2 = RK.line(t1b, X, 64, [['每步：离肘的位置转 90° → × 0.39 → 加上去', E.INK2, 700]], S4);
      var Q4 = [
        [['① (0, 2)，× 0.39 = (0, 0.78)'], ['　 2 + 0 = 2，0 + 0.78 = 0.78']],
        [['② (−0.78, 2)，× 0.39 = (−0.3, 0.78)'], ['　 2 − 0.3 = 1.7，0.78 + 0.78 = 1.56']],
        [['③ (−1.56, 1.7)，× 0.39 = (−0.61, 0.66)'], ['　 1.7 − 0.61 = 1.09，1.56 + 0.66 = 2.22']],
        [['④ (−2.22, 1.09)，× 0.39 = (−0.87, 0.43)'], ['　 1.09 − 0.87 = 0.22，2.22 + 0.43 = 2.65']]
      ].map(function (r, k) {
        var y = 96 + 58 * k;
        var a = RK.line(t1b, X, y, [[r[0][0].split('，× ')[0] + '，', E.INK, 900], ['× ' + r[0][0].split('，× ')[1], E.INK, 900]], S4);
        var bb = RK.line(t1b, X, y + 26, [[r[1][0].split('，')[0] + '，', E.INK, 900], [r[1][0].split('，')[1], E.INK, 900]], S4);
        return { a: a, b: bb };
      });
      var q7 = RK.line(t1b, X, 334, [['+ 肘：', E.HI, 900], ['1 + 0.22 = 1.22', E.INK, 900], ['，', E.INK2], ['2 + 2.65 = 4.65', E.INK, 900]], S4);
      var q8 = RK.line(t1b, X, 366, [['→ ', E.INK2], ['(1.22, 4.65)', E.INK, 900], ['：又近了一些', E.INK2]], 20);
      var t2 = E.el('g', {}, Rg);
      var r0 = RK.line(t2, X, 46, [['分几步', E.INK2, 700], ['　→　', E.MUT], ['停在', E.INK2, 700]], 21);
      var LABS = ['(3, 5.14)', '(1.77, 5.14)', '(1.22, 4.65)', '(1.05, 4.33)', '(1.00, 4.02)'];
      var rowsN = NS.map(function (n, i) {
        return RK.line(t2, X, 84 + 34 * i, [[n + ' 步', COLS[i], 900], ['　→　', E.MUT], [LABS[i], COLS[i], 900]], 21);
      });
      var pcRow = RK.line(t2, X, 84 + 34 * 5 + 6, [['8 步起：同样的算法交给电脑算', E.INK2, 700]], 19);
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
      var m1 = RK.line(t4, X, 46, [['为什么叫「指数」？', E.HI, 900]], 24);
      var m2 = RK.line(t4, X, 84, [['先说', E.INK2], ['自然常数 e', E.HI, 900]], 21);
      var m3 = RK.line(t4, X, 120, [['存 1 块，年利率 100%', E.INK, 900]], 20);
      var m4 = RK.line(t4, X, 156, [['一年算 1 次：', E.INK2], ['1 + 1 = 2', E.HI, 900]], 20);
      var m5 = RK.line(t4, X, 196, [['12 个月：每月 × ', E.INK2], ['(1 + 1/12)', E.HI, 900], [' ≈ 1.083', E.INK, 900]], 19);
      var m5b = RK.line(t4, X, 230, [['1 × 1.083 = 1.083', E.INK, 900], ['，', E.INK2], ['× 1.083 ≈ 1.17', E.INK, 900]], 19);
      var m6 = RK.line(t4, X, 264, [['乘满 12 次 → ', E.INK2], ['2.61', E.HI, 900]], 20);
      var m7 = RK.line(t4, X, 302, [['按天：乘 365 次 → ', E.INK2], ['2.71', E.HI, 900]], 20);
      var m8 = RK.line(t4, X, 342, [['越分越细 → ', E.INK2], ['2.718… = e', E.GOOD, 900]], 21);
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

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) { RK.layout(); laid = true; }
          var f = E.fade, b = E.bump;

          /* ── 左：网格 ↔ 复利 ── */
          E.op(gridG, Math.min(1, (1 - f(st, T.out2, 0.4)) + f(st, T.back, 0.5)));
          E.op(money, f(st, T.out2 + 0.3, 0.4) * (1 - f(st, T.back - 0.3, 0.4)));
          var back = st > T.back;

          /* ── 分两步（句 2–8） ── */
          var two = 1 - f(st, T.out1, 0.4);
          E.op(tick45, f(st, T.deg, 0.3) * (1 - f(st, T.first, 0.3))); E.op(tick45L, f(st, T.deg, 0.3) * (1 - f(st, T.first, 0.3)));
          E.op(dimG, f(st, T.r2, 0.3) * (1 - f(st, T.upA + 0.6, 0.3)));
          E.op(upA.g, f(st, T.upA, 0.3) * (1 - f(st, T.seg1[1], 0.3)));
          E.drawOn(segA, f(st, T.seg1[0], T.seg1[1] - T.seg1[0])); E.op(segA, (f(st, T.seg1[0], 0.1) > 0 ? 1 : 0) * two);
          E.op(k1Dot, f(st, T.add1v, 0.3) * two); k1Dot.setAttribute('r', (6.5 + 4 * b(st, T.kink, 0.8)).toFixed(2));
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
          E.op(spoke2.g, f(st, T.perp, 0.3) * (1 - f(st, T.col1, 0.3)));
          turnArc.set(Math.atan2(dd[1], dd[0]), Math.max(Math.atan2(dd[1], dd[0]) + 1e-3, ang));   // 画满 90°：从虚线原来的方向到转完的方向（同 009 第 3 幕的修正）
          E.op(turnArc.g, f(st, T.turn[0], 0.2) * (1 - f(st, T.col1 + 0.6, 0.3)));
          // 红（第一列 2 份：往上）、绿（第二列 1.57 份：往左）两段：先是整份，句 6 缩成 0.785 倍，头尾相接
          var sc = 1 - (1 - HH) * f(st, T.scale[0], T.scale[1] - T.scale[0]);
          var a0 = kq, a1 = [kq[0], kq[1] + dd[0] * sc * f(st, T.col1, 0.5)], a2 = [a1[0] - dd[1] * sc * f(st, T.col2, 0.5), a1[1]];
          var A0 = P(a0[0], a0[1]), A1 = P(a1[0], a1[1]), A2 = P(a2[0], a2[1]);
          legR.set(A0[0], A0[1], A1[0], A1[1]); legG.set(A1[0], A1[1], A2[0], A2[1]);
          var legOn = 1 - f(st, T.stopAt, 0.4);
          E.op(legR.g, (f(st, T.col1, 0.1) > 0 ? 1 : 0) * legOn); E.op(legG.g, (f(st, T.col2, 0.1) > 0 ? 1 : 0) * legOn);
          E.drawOn(segB, f(st, T.go[0], T.go[1] - T.go[0])); E.op(segB, (f(st, T.go[0], 0.1) > 0 ? 1 : 0) * two);
          E.op(k2Dot, f(st, T.stopAt, 0.3) * two); E.op(k2Lab, f(st, T.stopAt, 0.3) * two);
          E.op(nearG, f(st, T.near, 0.4) * two);

          /* ── 分四步（句 9–14）：一步一步照算 ── */
          var cur = -1;
          T.st4.forEach(function (S, k) { if (st >= S.start - 0.3) cur = k; });
          seg4.forEach(function (p, k) {
            var S = T.st4[k], on = f(st, S.add, 0.6);
            E.drawOn(p, on);
            var dimLater = 1 - 0.6 * Math.max(f(st, T.n8, 0.4), f(st, T.arc, 0.5));
            p.style.opacity = (on > 0 ? 1 : 0) * dimLater * (1 - f(st, T.out2, 0.3)) * (back ? 0 : 1);
          });
          dot4.forEach(function (c, k) { E.op(c, f(st, T.st4[k].addV, 0.3) * (1 - f(st, T.out4, 0.4))); });
          if (cur >= 0 && st < T.out4) {
            var S = T.st4[cur], d = D4[cur], vs = V4[cur], len = Math.hypot(d[0], d[1]);
            var stepOn = f(st, S.start - 0.3, 0.3) * (1 - f(st, S.add + 0.5, 0.3));
            spoke4.set(Q[0], Q[1], vs[0], vs[1]); E.op(spoke4.g, stepOn);
            var a = Math.atan2(d[1], d[0]) + TH * f(st, S.rot, 0.8), ln = len * (1 - (1 - F4) * f(st, S.mul, 0.6));
            var tip = P(1 + d[0] + ln * Math.cos(a), 2 + d[1] + ln * Math.sin(a));
            turn4.set(vs[0], vs[1], tip[0], tip[1]); E.op(turn4.g, f(st, S.rot - 0.2, 0.3) * (1 - f(st, S.add + 0.3, 0.3)));
            E.op(turn4Arc.g, f(st, S.rot, 0.2) * (1 - f(st, S.mul, 0.3)));
            // 蓝弧画满转过的角度：圆心在这一步的起点
            turn4Arc.g.setAttribute('transform', 'translate(' + vs[0].toFixed(1) + ' ' + vs[1].toFixed(1) + ')');
            turn4Arc.set(Math.atan2(d[1], d[0]), Math.max(Math.atan2(d[1], d[0]) + 1e-3, a));
          } else { E.op(spoke4.g, 0); E.op(turn4.g, 0); E.op(turn4Arc.g, 0); }
          E.op(lab4, f(st, T.at4, 0.3) * (1 - f(st, T.out4, 0.4)));

          /* ── 折线 1 / 8 / 100（句 15–16） ── */
          var drawT = [null, null, null, T.n8, T.n100];
          paths.forEach(function (p, i) {
            if (i === 1 || i === 2) { E.op(p, 0); return; }                 // 分两步、分四步由上面的分段画
            var on = i === 0 ? 1 : f(st, drawT[i], i === 4 ? 0.8 : 0.5);
            if (i > 0) E.drawOn(p, on);
            var later = [3, 4].map(function (j) { return j > i ? f(st, drawT[j], 0.4) : 0; }).reduce(function (x, y) { return Math.max(x, y); }, 0);
            if (i === 0) later = Math.max(later, f(st, T.n4, 0.4));
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
          // 句 22：一小步一小步滚
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
          // 第一个月、第二个月各长一格（旁白算出来的那两次），之后「乘满十二次」再一格一格长到 12 格
          var k12 = (st >= T.mo1v ? 1 : 0) + (st >= T.mo2v ? 1 : 0) + Math.floor(10 * E.clamp((st - T.k12[0]) / (T.k12[1] - T.k12[0]), 0, 1) + 1e-9);
          var v12 = Math.pow(1 + 1 / 12, k12);
          bars[1].grow.setAttribute('width', (UX * (v12 - 1)).toFixed(1));
          Array.prototype.forEach.call(bars[1].ticks.children, function (ln, j) { E.op(ln, j + 1 < k12 ? 0.8 : 0); });
          bars[1].val.textContent = '2.61'; bars[1].val.setAttribute('x', (eX + 12).toFixed(1)); E.op(bars[1].val, f(st, T.v12, 0.3));   // 条尾离竖虚线只有 12 px：数写到虚线右边
          E.op(bars[2].g, f(st, T.day - 0.2, 0.3));
          var g3 = f(st, T.fine[0], T.fine[1] - T.fine[0]), v365 = Math.pow(1 + 1 / 365, 365);
          bars[2].grow.setAttribute('width', (UX * (v365 - 1) * g3).toFixed(1));
          bars[2].val.textContent = '2.71'; bars[2].val.setAttribute('x', (eX + 12).toFixed(1)); E.op(bars[2].val, f(st, T.d365, 0.3));
          E.drawOn(limLine, f(st, T.lim, 0.5)); E.op(limLab, f(st, T.lim, 0.3));
          E.pop(eLab, eX + 18, 66, f(st, T.eNum, 0.35), 0.6);
          E.op(scaleG, f(st, T.coin, 0.3));

          /* ── 右栏 ── */
          E.op(t0, 1 - f(st, T.out0, 0.4));
          E.op(z1.g, fr(st, 0.6)); E.op(z2.g, fr(st, T.each)); E.op(z2.parts[1], fr(st, T.stop)); E.op(z3.g, fr(st, T.dir)); E.op(z3.parts[1], fr(st, T.next));
          E.op(t1, f(st, T.out0 + 0.2, 0.4) * (1 - f(st, T.out1, 0.4)));
          E.op(w1.g, fr(st, T.two)); E.op(w1.parts[1], fr(st, T.deg)); E.op(w2.g, fr(st, T.div)); E.op(w2.parts[1], fr(st, T.h));
          E.op(w3.g, fr(st, T.mul1)); E.op(w3b.g, fr(st, T.add1)); E.op(w3b.parts[1], fr(st, T.k1)); E.op(w3b.parts[2], fr(st, T.k1));
          E.op(w4.g, fr(st, T.subA)); E.op(w4.parts[2], fr(st, T.subB)); E.op(w4.parts[3], fr(st, T.subB));
          E.op(w5.g, fr(st, T.mat)); E.op(w5.parts[1], fr(st, T.col1)); E.op(w5.parts[2], fr(st, T.col2)); E.op(w5.parts[3], fr(st, T.col2));
          E.op(w5b.g, fr(st, T.col1v)); E.op(w5b.parts[2], fr(st, T.col2v)); E.op(w5b.parts[3], fr(st, T.col2v)); E.op(w5b.parts[4], fr(st, T.sumV));
          E.op(w6.g, fr(st, T.scale[0])); E.op(w6.parts[1], fr(st, T.r157)); E.op(w6.parts[2], fr(st, T.r123)); E.op(w6.parts[3], fr(st, T.r123));
          E.op(w7.g, fr(st, T.sub)); E.op(w7.parts[2], fr(st, T.sub2));
          E.op(w8.g, fr(st, T.add)); E.op(w8.parts[2], fr(st, T.add2));
          E.op(w9.g, fr(st, T.stopAt)); E.op(w9.parts[2], fr(st, T.kink));
          E.op(t1b, f(st, T.out1 + 0.2, 0.4) * (1 - f(st, T.out4, 0.4)));
          E.op(q1.g, fr(st, T.n4)); E.op(q1.parts[1], fr(st, T.d4)); E.op(q1.parts[2], fr(st, T.d4v)); E.op(q2.g, fr(st, T.recipe));
          Q4.forEach(function (r, k) {
            var S = T.st4[k];
            E.op(r.a.g, fr(st, S.rot)); E.op(r.a.parts[1], fr(st, S.mul));
            E.op(r.b.g, fr(st, S.addX)); E.op(r.b.parts[1], fr(st, S.addY));
          });
          E.op(q7.g, fr(st, T.elb)); E.op(q7.parts[1], fr(st, T.ex)); E.op(q7.parts[2], fr(st, T.ey)); E.op(q7.parts[3], fr(st, T.ey));
          E.op(q8.g, fr(st, T.at4)); E.op(q8.parts[2], fr(st, T.near4));
          E.op(t2, f(st, T.out4 + 0.2, 0.4) * (1 - f(st, T.one - 0.3, 0.4)));
          E.op(r0.g, 1);
          var rowT = [0, 0, 0, T.at8, T.at100];
          rowsN.forEach(function (r, i) { E.op(r.g, i < 3 ? 1 : fr(st, rowT[i])); });
          E.op(pcRow.g, fr(st, T.pc));
          E.op(rowInf.g, fr(st, T.inf)); E.op(rowInf.parts[2], fr(st, T.at)); E.drawOn(infOk, f(st, T.at + 0.3, 0.3));
          E.op(t3, f(st, T.one - 0.1, 0.4) * (1 - f(st, T.out2, 0.4)));
          E.op(f1.g, fr(st, T.one)); E.op(f2, fr(st, T.same)); E.op(f3, fr(st, T.times)); E.pop(f4, RX + 170, 176, f(st, T.nInf, 0.35), 0.9);
          E.op(fName, fr(st, T.name));
          lg.forEach(function (g, i) { E.op(g, fr(st, T.legend[i])); });
          E.op(t4, f(st, T.out2 + 0.3, 0.4) * (1 - f(st, T.pow - 0.3, 0.4)));
          E.op(m1.g, fr(st, T.why)); E.op(m2.g, fr(st, T.e0)); E.op(m3.g, fr(st, T.coin)); E.op(m4.g, fr(st, T.once)); E.op(m4.parts[1], fr(st, T.onePlus));
          E.op(m5.g, fr(st, T.m12)); E.op(m5.parts[1], fr(st, T.perM)); E.op(m5.parts[2], fr(st, T.r1083));
          E.op(m5b.g, fr(st, T.mo1)); E.op(m5b.parts[1], fr(st, T.mo2)); E.op(m5b.parts[2], fr(st, T.mo2));
          E.op(m6.g, fr(st, T.k12[0])); E.op(m6.parts[1], fr(st, T.v12));
          E.op(m7.g, fr(st, T.day)); E.op(m7.parts[1], fr(st, T.d365)); E.op(m8.g, fr(st, T.lim));
          E.op(t5, f(st, T.pow, 0.4));
          E.op(x1, fr(st, T.pow)); E.op(x2, fr(st, T.same2));
          E.op(x3.g, fr(st, T.money)); E.drawOn(x3x, f(st, T.money + 0.2, 0.3));
          E.op(x4.g, fr(st, T.pose)); E.drawOn(x4ok, f(st, T.pose + 0.2, 0.3));
        }
      };
    }
  };
})();
