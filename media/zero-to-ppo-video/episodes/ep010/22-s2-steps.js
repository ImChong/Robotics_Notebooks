// 010 第 2 幕：切成无穷多小步——矩阵指数。左边同第 1 幕的网格（肘在 (1, 2)，小臂上的点从 (3, 2) 出发，要走的四分之一圆描成灰虚线），
// 右栏是「分几步 → 停在」的表；中段左边换成复利的条形图，右栏换成 (1 + 1/n)ⁿ；最后换回网格。
// 句 1「每走一小步」画第一段（朝上 1.57 格），「停下来」拐点亮一下，「按这时候的方向」肘到拐点画虚线、拐点长出垂直于它的方向箭头，「再走下一步」画第二段；
// 句 2「分两步」「每步转四十五度」右栏写出，「折了一次」拐点鼓一下，「离圆近了一些」终点写 (1.77, 5.14)；
// 句 3「四步」「八步」两条折线依次画出，「贴着圆走」灰虚线亮一下，「一百步」画第五条，「一、四点零二」表里依次写出各行；
// 句 4「无穷多步」绿色圆弧画出、表里写「无穷多 → (1, 4)」，「一、四」终点打勾，「一步一步乘出来」右栏出 (I + [S]θ / n)ⁿ，「矩阵指数」写出 exp([S]θ)；
// 句 5–6 复利：「存一块钱」第一条先有 1 块，「一年算一次利息」长到 2；「十二个月利滚利」第二条一格一格长 12 次到 2.61，
//    「分得越细」第三条长到 2.71，「二点七一八」画出竖虚线，「自然常数」写 e；
// 句 7「几次方，叫指数函数」右栏写 eˣ，「同一个做法」两行对照，左边换回网格，「一步一步滚起来」一个点沿圆一小步一小步走，
//    「不是钱，是位姿」点上长出两根小箭头（位置 + 朝向）。
// 时刻取自配音的逐词时间，见 ../ep010.py。
(function () {
  var E = window.EP010, K = E.K, H = E.HEX;
  var A = E.ARM, RX = A.RX;
  var T = {
    s1: [2.14, 2.9], stop: 3.08, dir: 3.71, s2: [4.89, 5.6],
    two: 7.17, deg: 8.07, kink: 9.09, near: 9.88,
    n4: 12.25, n8: 12.79, hug: 13.8, n100: 15.84, at100: 16.97, rows: [7.17, 9.88, 12.25, 12.79, 16.97],
    inf: 21.13, arc: 22.23, at: 24.01, mult: 25.62, name: 27.55,
    out1: 29.6, why: 30.35, e0: 32.09, coin: 33.06, rate: 34.22, once: 35.59, two2: 37.08,
    m12: [38.71, 40.4], v12: 40.69, fine: [41.74, 42.6], lim: 42.82, eNum: 44.87,
    pow: 47.04, same: 49.55, back: 49.7, roll: [51.7, 54.24], money: 55.38, pose: 56.24
  };
  var END = 57.55, TH = Math.PI / 2, NS = [1, 2, 4, 8, 100];
  var COLS = [E.BAD, E.WARN, E.VIOLET, E.ACC, E.GOOD];

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
      { at: 6.84, s: '分 2 步、每步 45°：折了一次，离圆近了一些' },
      { at: 11.88, s: '分得越细，越贴着圆走：100 步停在 (1, 4.02)' },
      { at: 20.54, s: '无穷多步正好走在圆上：这样一步一步乘出来的，叫**矩阵指数**' },
      { at: 30.02, s: '名字来自自然常数 e：存 1 块，年利率 100%，一年算一次是 2 块' },
      { at: 38.62, s: '12 个月利滚利是 2.61；分得越细，越接近 2.718，这就是 e' },
      { at: 46.68, s: 'eˣ 叫指数函数；**矩阵指数**是同一个做法，滚起来的是**位姿**' }
    ],
    build: function (K) {
      var s = K.sceneSvg('小臂上的点绕肘转九十度：分成几小步，每走一小步就按这时候的方向再走下一步；分两步折一次停在 (1.77, 5.14)，分四步、八步越来越贴着圆，分一百步停在 (1, 4.02)，分成无穷多步正好走在圆上停在 (1, 4)，这样一步一步乘出来的结果叫矩阵指数；名字来自自然常数 e：存一块钱、年利率百分之百，一年算一次利息是两块，分十二个月利滚利是 2.61，分得越细越接近 2.718；矩阵指数是同一个做法，滚起来的是位姿');

      /* ══ 左：网格 + 折线 ══ */
      var gridG = E.el('g', {}, s), u = A.U;
      var pl = E.plane(gridG, { ox: A.OX, oy: A.OY, u: u, x0: -1, x1: 4, y0: -1, y1: 5 });
      var P = pl.P, Q = P(1, 2);
      function onArc(a) { return P(1 + 2 * Math.cos(a), 2 + 2 * Math.sin(a)); }
      var pts40 = []; for (var k = 0; k <= 40; k++) pts40.push(onArc(TH * k / 40));
      var arcGrey = K.paint(E.el('path', { d: E.poly(pts40), fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '8 7' }, gridG), null, E.INK2);
      var radius = K.paint(E.el('path', { d: E.poly([Q, P(3, 2)]), fill: 'none', 'stroke-width': 4, 'stroke-linecap': 'round' }, gridG), null, E.VIOLET);
      radius.style.opacity = 0.55;
      var paths = NS.map(function (n, i) {
        var p = K.paint(E.el('path', { d: E.poly(steps(n).map(function (q) { return P(q[0], q[1]); })), fill: 'none', 'stroke-width': n > 50 ? 3.5 : 4,
          'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, gridG), null, COLS[i]);
        if (i === 0) p.setAttribute('stroke-dasharray', '9 6');
        return p;
      });
      var arcOk = K.paint(E.el('path', { d: E.poly(pts40), fill: 'none', 'stroke-width': 5, 'stroke-linecap': 'round' }, gridG), null, E.GOOD);
      // 分两步：拐点、肘到拐点的虚线、拐点上的新方向
      var c2 = steps(2), K1 = P(c2[1][0], c2[1][1]);
      var spoke = K.paint(E.el('path', { d: E.poly([Q, K1]), fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '6 5' }, gridG), null, E.HI);
      var kDot = K.paint(E.el('circle', { cx: K1[0], cy: K1[1], r: 6, 'stroke-width': 2.5 }, gridG), E.S1, E.WARN);
      var newDir = E.vec(gridG, E.ACC, 4, { head: 13 });
      var dd = [c2[1][0] - 1, c2[1][1] - 2], dl = Math.hypot(dd[0], dd[1]), nd = [-dd[1] / dl, dd[0] / dl];
      newDir.set(K1[0], K1[1], K1[0] + nd[0] * 0.95 * u, K1[1] - nd[1] * 0.95 * u);
      // 肘、出发点、终点标签
      var outOf = E.el('g', {}, gridG);
      K.paint(E.el('circle', { cx: Q[0], cy: Q[1], r: 12, 'stroke-width': 3.5 }, outOf), E.S1, E.HI);
      K.paint(E.el('circle', { cx: Q[0], cy: Q[1], r: 4 }, outOf), E.HI);
      var start = P(3, 2);
      K.paint(E.el('circle', { cx: start[0], cy: start[1], r: 7, 'stroke-width': 3 }, gridG), E.ACC, E.S1);
      var e2 = P(c2[2][0], c2[2][1]);
      var lab2 = E.halo(E.txt(gridG, e2[0] + 12, e2[1] - 10, '(1.77, 5.14)', 20, E.WARN, 'start', 900));
      var endOk = P(1, 4);
      var labOk = E.halo(E.txt(gridG, endOk[0] - 14, endOk[1] + 8, '(1, 4)', 24, E.GOOD, 'end', 900));
      // 句 7：沿圆一小步一小步走的点 + 两根小箭头（位姿）
      var rollG = E.el('g', {}, gridG);
      var rAx = [E.vec(rollG, E.GOOD, 4.5, { head: 13 }), E.vec(rollG, E.WARN, 4.5, { head: 13 })];
      var rDot = K.paint(E.el('circle', { r: 8, 'stroke-width': 3 }, rollG), E.ACC, E.S1);

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
      var t1 = E.el('g', {}, Rg);
      var r0 = RK.line(t1, X, 46, [['分几步', E.INK2, 700], ['　→　', E.MUT], ['停在', E.INK2, 700]], 21);
      var LABS = ['(3, 5.14)', '(1.77, 5.14)', '(1.20, 4.66)', '(1.05, 4.33)', '(1.00, 4.02)'];
      var rowsN = NS.map(function (n, i) {
        return RK.line(t1, X, 84 + 34 * i, [[n + ' 步', COLS[i], 900], ['　→　', E.MUT], [LABS[i], COLS[i], 900]], 21);
      });
      var rowInf = RK.line(t1, X, 84 + 34 * 5 + 4, [['无穷多步', E.GOOD, 900], ['　→　', E.MUT], ['(1, 4)', E.GOOD, 900]], 22);
      var infOk = E.check(rowInf.g, X + 230, 240, 20, E.GOOD, 4);
      var hdr2 = RK.line(t1, X, 152, [['每步转 45°', E.WARN, 900]], 20);   // 写在表的下一行：「4 步」那一行出来前收掉
      var fm = E.el('g', {}, t1);
      E.txt(fm, X, 306, '一步一步乘：', 20, E.INK2, 'start', 700);
      E.tex(fm, X + 126, 306, '\\left(I+\\tfrac{[\\mathcal S]\\,\\theta}{n}\\right)^{n}', 26, E.INK, 'start', { w: 210, h: 50 });
      var fmName = E.el('g', {}, t1);
      K.paint(E.el('rect', { x: RX, y: 334, width: 340, height: 64, rx: 14, 'stroke-width': 3 }, fmName), E.tint(E.HI, 10, 'var(--demo-surface)'), E.HI);
      E.tex(fmName, X + 8, 375, '\\to\\ \\textcolor{' + H.hi + '}{\\exp\\big([\\mathcal S]\\,\\theta\\big)}', 28, E.INK, 'start', { w: 200, h: 50 });
      E.txt(fmName, RX + 326, 376, '矩阵指数', 24, E.HI, 'end', 900);
      // 复利
      var t2 = E.el('g', {}, Rg);
      var w1 = RK.line(t2, X, 50, [['为什么叫「指数」？', E.HI, 900]], 24);
      var w2 = RK.line(t2, X, 92, [['来自', E.INK2], ['自然常数 e', E.HI, 900]], 22);
      var w3 = RK.line(t2, X, 134, [['存 1 块，年利率 100%', E.INK, 900]], 22);
      var w4 = E.el('g', {}, t2);
      E.txt(w4, X, 188, '一年算 n 次：', 20, E.INK2, 'start', 700);
      E.tex(w4, X + 128, 188, '\\left(1+\\tfrac{1}{n}\\right)^{n}', 26, E.INK, 'start', { w: 140, h: 50 });
      var w5 = RK.line(t2, X, 236, [['n = 1：', E.INK2], ['2', E.HI, 900]], 21);
      var w6 = RK.line(t2, X, 270, [['n = 12：', E.INK2], ['2.61', E.HI, 900]], 21);
      var w7 = RK.line(t2, X, 304, [['n = 365：', E.INK2], ['2.71', E.HI, 900]], 21);
      var w8 = RK.line(t2, X, 344, [['n 越大 → ', E.INK2], ['2.718… = e', E.GOOD, 900]], 22);
      // 句 7：对照
      var t3 = E.el('g', {}, Rg);
      var x1 = E.el('g', {}, t3);
      E.tex(x1, X, 66, 'e^{x}', 34, E.HI, 'start', { w: 60, h: 56 });
      E.txt(x1, X + 64, 66, '：自然常数的 x 次方，叫指数函数', 19, E.INK, 'start', 700);
      var x2 = E.el('g', {}, t3);
      E.txt(x2, X, 126, '同一个做法：', 22, E.INK, 'start', 900);
      E.tex(x2, X, 182, '\\left(1+\\tfrac{x}{n}\\right)^{n}\\to e^{x}', 26, E.INK, 'start', { w: 260, h: 50 });
      E.tex(x2, X, 246, '\\left(I+\\tfrac{[\\mathcal S]\\theta}{n}\\right)^{n}\\to \\textcolor{' + H.hi + '}{\\exp([\\mathcal S]\\theta)}', 26, E.INK, 'start', { w: 330, h: 50 });
      var x3 = RK.line(t3, X, 316, [['滚起来的不是钱，', E.INK2, 700]], 22);
      var x3x = E.cross(x3.g, X + 196, 308, 9, E.BAD, 4);
      var x4 = RK.line(t3, X, 356, [['是', E.INK2, 700], ['位姿', E.HI, 900]], 24);
      var x4ok = E.check(x4.g, X + 80, 338, 22, E.GOOD, 4);

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) { RK.layout(); laid = true; }
          var f = E.fade, b = E.bump;

          /* ── 左：网格 ↔ 复利 ── */
          var gridOn = (1 - f(st, T.out1, 0.4)) + f(st, T.back, 0.5);
          E.op(gridG, Math.min(1, gridOn));
          E.op(money, f(st, T.out1 + 0.3, 0.4) * (1 - f(st, T.back - 0.3, 0.4)));
          E.op(outOf, 1);
          // 折线：n = 1 一开始就在（第 1 幕的直线）；后来的画出来，前面的变淡
          var drawT = [null, T.s1[0], T.n4, T.n8, T.n100];
          paths.forEach(function (p, i) {
            var on;
            if (i === 0) on = 1;
            else if (i === 1) {   // 分两步：句 1 一段一段画
              var a = f(st, T.s1[0], T.s1[1] - T.s1[0]) * 0.5, c = f(st, T.s2[0], T.s2[1] - T.s2[0]) * 0.5;
              on = a + c;
            } else on = f(st, drawT[i], i === 4 ? 0.8 : 0.5);
            E.drawOn(p, on);
            var later = NS.map(function (n, j) { return j > i && j > 1 ? f(st, drawT[j], 0.4) : 0; }).reduce(function (x, y) { return Math.max(x, y); }, 0);
            var dim = 1 - 0.6 * Math.max(later, f(st, T.arc, 0.5));
            p.style.opacity = (i === 0 ? 0.9 : 1) * dim * (1 - f(st, T.out1, 0.3)) + 0.0;
            if (st > T.back) p.style.opacity = i === 4 ? 0.45 : 0;
          });
          E.op(arcGrey, (1 - f(st, T.arc, 0.4)) * (1 - f(st, T.out1, 0.3)));
          arcGrey.style.strokeWidth = 3 + 2.5 * b(st, T.hug, 1.0); E.glow(arcGrey, E.INK2, b(st, T.hug, 1.0) > 0.05);
          E.drawOn(arcOk, f(st, T.arc, 0.9));
          var kb = b(st, T.kink, 0.8);
          E.op(kDot, f(st, T.stop, 0.3) * (1 - f(st, T.n4, 0.4))); kDot.setAttribute('r', (6 + 5 * Math.max(b(st, T.stop, 0.7), kb)).toFixed(2));
          E.op(spoke, f(st, T.dir, 0.3) * (1 - f(st, T.s2[1] + 0.6, 0.4)));
          E.op(newDir.g, f(st, T.dir + 0.3, 0.3) * (1 - f(st, T.s2[1] + 0.6, 0.4)));
          E.op(lab2, f(st, T.near, 0.3) * (1 - f(st, T.n4, 0.4)));
          E.op(labOk, f(st, T.at, 0.3) * (1 - f(st, T.out1, 0.3)) + (st > T.back ? 0 : 0));
          // 句 7：一小步一小步滚
          var rk = f(st, T.roll[0], 0) > 0 ? Math.min(16, Math.floor(16 * E.clamp((st - T.roll[0]) / (T.roll[1] - T.roll[0]), 0, 1))) : 0;
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
          var g0 = f(st, T.once, 1.2);
          bars[0].grow.setAttribute('width', (UX * g0).toFixed(1));
          bars[0].val.textContent = '2'; bars[0].val.setAttribute('x', (X0 + UX * (1 + g0) + 10).toFixed(1)); E.op(bars[0].val, f(st, T.two2, 0.3));
          E.op(bars[1].g, f(st, T.m12[0] - 0.2, 0.3));
          var k12 = Math.floor(12 * E.clamp((st - T.m12[0]) / (T.m12[1] - T.m12[0]), 0, 1) + 1e-9), v12 = Math.pow(1 + 1 / 12, k12);
          bars[1].grow.setAttribute('width', (UX * (v12 - 1)).toFixed(1));
          Array.prototype.forEach.call(bars[1].ticks.children, function (ln, j) { E.op(ln, j + 1 < k12 ? 0.8 : 0); });
          bars[1].val.textContent = '2.61'; bars[1].val.setAttribute('x', (eX + 12).toFixed(1));   // 2.61 的条尾离竖虚线只有 12 px：数写到虚线右边 E.op(bars[1].val, f(st, T.v12, 0.3));
          E.op(bars[2].g, f(st, T.fine[0] - 0.2, 0.3));
          var g3 = f(st, T.fine[0], T.fine[1] - T.fine[0]), v365 = Math.pow(1 + 1 / 365, 365);
          bars[2].grow.setAttribute('width', (UX * (v365 - 1) * g3).toFixed(1));
          bars[2].val.textContent = '2.71'; bars[2].val.setAttribute('x', (eX + 12).toFixed(1)); E.op(bars[2].val, f(st, T.fine[1], 0.3));
          E.drawOn(limLine, f(st, T.lim, 0.5)); E.op(limLab, f(st, T.lim, 0.3));
          E.pop(eLab, eX + 18, 66, f(st, T.eNum, 0.35), 0.6);
          E.op(scaleG, f(st, T.coin, 0.3));

          /* ── 右栏 ── */
          E.op(t1, 1 - f(st, T.out1, 0.4));
          E.op(r0.g, fr(st, T.s1[0]));
          rowsN.forEach(function (r, i) { E.op(r.g, i === 0 ? fr(st, T.s1[0]) : fr(st, T.rows[i])); });
          E.op(hdr2.g, fr(st, T.deg) * (1 - f(st, T.n4 - 0.5, 0.3)));
          E.op(rowInf.g, fr(st, T.inf)); E.op(rowInf.parts[2], fr(st, T.at)); E.drawOn(infOk, f(st, T.at + 0.3, 0.3));
          E.op(fm, fr(st, T.mult)); E.pop(fmName, RX + 170, 366, f(st, T.name, 0.35), 0.9);
          E.op(t2, f(st, T.out1 + 0.3, 0.4) * (1 - f(st, T.pow - 0.3, 0.4)));
          E.op(w1.g, fr(st, T.why)); E.op(w2.g, fr(st, T.e0)); E.op(w3.g, fr(st, T.coin)); E.op(w3.g, fr(st, T.coin) * 1);
          E.op(w4, fr(st, T.once)); E.op(w5.g, fr(st, T.two2)); E.op(w6.g, fr(st, T.v12)); E.op(w7.g, fr(st, T.fine[1])); E.op(w8.g, fr(st, T.lim));
          E.op(t3, f(st, T.pow, 0.4));
          E.op(x1, fr(st, T.pow)); E.op(x2, fr(st, T.same));
          E.op(x3.g, fr(st, T.money)); E.drawOn(x3x, f(st, T.money + 0.2, 0.3));
          E.op(x4.g, fr(st, T.pose)); E.drawOn(x4ok, f(st, T.pose + 0.2, 0.3));
        }
      };
    }
  };
})();
