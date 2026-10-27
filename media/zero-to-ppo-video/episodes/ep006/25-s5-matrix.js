// 006 第 5 幕：旋转矩阵——九个数，六个等式。
// 句 1 左上出第 2 幕那个朝向（RPY (20°, 25°, 45°)：侧歪 20°、前倾 25°、转身 45°）的旋转矩阵，「好处」右边三行依次打勾：接着做 = 相乘、乘向量得位置、没有万向锁；
// 句 2「冗余」右边换成「代价：冗余」，左下出三维小坐标架；「三列长度都是一」三根小箭头依次标 1、矩阵三列依次框亮，「三个等式」；
//    「每两列互相垂直」三对小箭头之间依次画直角、矩阵两列一起框亮，「又是三个等式」；
// 句 3「九个数，六个等式」写出 9 − 6 = 3，「三个自由度」接上，「多出来的六个数」「负担」红字；
// 句 4「算久了会走样」左边换成二维网格（从上往下看），右边出四十五度的矩阵：「小数只留两位」0.7071… 收成 0.71，「每一列的长度」算出 ≈ 1.004，「千分之四」；
// 句 5「连着乘一百次」两根小箭头每乘一次转 45°、长一点，箭头尖留下一圈圈往外的点，计数到 100，「一点五」长度变红，「格子大了一半」虚线画出原来的格子，「不是旋转」红牌；
// 句 6 右边一行行写「电脑留十几位小数，走样也慢得多」「一秒算几百上千次，算久了照样会走样」，网格变成又斜又长的样子（夸大画），
//    「扶正」「长度拉回一」两根小箭头缩回单位圆，「两两垂直」往上那根转到和往右那根垂直；
// 句 7 左右两块：四元数「除以自己的长度」一步扶正；矩阵「六个等式要守」六块小牌依次出来，「麻烦得多」。
// 时刻取自配音的逐词时间（分镜时间 = 念到这个词的时刻），见 ../ep006.py。
(function () {
  var E = window.EP006, K = E.K, H = E.HEX;
  var D = Math.PI / 180;
  var T = {
    mat: 0.9, goodHead: 2.61, goods: [4.42, 6.89, 10.54],
    cost: 13.02, nine: 13.91, len: [16.3, 16.65, 17.0], eq3a: 18.02, perp: [19.04, 19.4, 19.75], eq3b: 20.59,
    tally: 22.44, six: 23.29, free: 24.24, dof: 27.18, extra: 28.57, burden: 30.22,
    to2d: 31.6, m45: 34.64, round: 36.06, r71: 37.65, colLen: 38.72, l1004: 40.36, permil: 42.05,
    mul: [44.27, 2.8], mulTxt: 43.82, worse: 45.05, l15: 46.47, bigger: 48.1, notRot: 49.64,
    pc: 51.53, small: 53.35, many: 55.14, accum: 57.06, skew: [57.06, 1.0], fix: 60.6, unit: [62.28, 0.8], perpFix: [63.56, 0.8],
    quat: 65.5, oneRule: 67.34, divide: 68.56, fixed: 70.23, six2: 71.36, hard: 73.27
  };
  var EUL = E.zyx(45 * D, 25 * D, 20 * D);                    // 第 2 幕的朝向
  var MX = 186, MY = 86, GZ = [150, 296];
  var OX = 196, OY = 214, U = 64;                             // 二维网格（x、y 都是 −2…2）
  var A71 = [0.71, -0.71, 0.71, 0.71], S = Math.hypot(0.71, 0.71);   // 只留两位小数的 45° 矩阵，每列长 ≈ 1.0041
  var SKEW = [1.18, -0.3, 0.14, 1.1];                         // 「积累的误差」夸大画：两列又长又不垂直

  function bump(st, a, d) { var u = K.seg(st, a, a + d); return u > 0 && u < 1 ? Math.sin(Math.PI * u) : 0; }
  function lerp(a, b, p) { return a.map(function (x, i) { return x + (b[i] - x) * p; }); }
  function rot2(a) { return [Math.cos(a), -Math.sin(a), Math.sin(a), Math.cos(a)]; }

  E.scenes[4] = {
    title: '旋转矩阵：九个数，六个等式',
    dur: 75.1,
    cues: [
      { at: 0, s: '**旋转矩阵**：接着做 = 相乘；乘向量就得到位置；没有万向锁' },
      { at: 12.26, s: '代价：**冗余**。三列长度为 1 → 3 个等式；两两垂直 → 3 个等式' },
      { at: 22.34, s: '9 个数 − 6 个等式 = 3 个自由度：多出来的 6 个数是负担' },
      { at: 31.46, s: '45° 的矩阵只留两位小数：0.71，每一列长 1.004' },
      { at: 43.58, s: '连乘 100 次：长度变成 1.5，已经不是旋转' },
      { at: 51.43, s: '算久了照样会走样：隔一阵子要把矩阵**扶正**' },
      { at: 65.4, s: '四元数：除以长度就扶正；矩阵 6 个等式，扶正麻烦得多' }
    ],
    build: function (K) {
      var s = K.sceneSvg('旋转矩阵的好处：两个转动接着做直接相乘，乘上向量就知道转到了什么位置，没有万向锁。代价是冗余：九个数要守六个等式（三列长度为一、两两垂直），能自由变的只剩三个。算久了会走样：四十五度的矩阵只留两位小数写成 0.71，每列长 1.004，连乘一百次长度变成 1.5，已经不是旋转。误差会慢慢积累，要隔一阵子把矩阵扶正：长度拉回一、两两垂直；四元数除以长度就扶正，矩阵要守六个等式，扶正麻烦得多');

      /* ══ 句 1–3：矩阵 + 小坐标架 + 右边的列表 ══ */
      var A = E.el('g', {}, s);
      var m3 = E.grid(A, MX, MY, EUL, { size: 26, dx: 94, dy: 38, pad: 46, colors: [E.X, E.Y, E.Z] });
      var cap = E.txt(A, MX, 166, '侧歪 20°、前倾 25°、转身 45°', 19, E.INK2, 'middle', 700);
      var gzG = E.el('g', {}, A);
      var proj = E.view3(128, 24);
      var gz = E.gizmo(gzG, GZ[0], GZ[1], 118, proj, { w: 5, head: 15, lab: 22 });
      gz.set(EUL);
      var ones = [0, 1, 2].map(function (j) {
        // 「1」写在箭头中段的一侧（沿画面里的法向挪开 16 px），不压箭头
        var o0 = gz.P([0, 0, 0]), tp = gz.P(E.col3(EUL, j).map(function (x) { return x * 0.55; }));
        var dx = tp[0] - o0[0], dy = tp[1] - o0[1], dl = Math.hypot(dx, dy) || 1, nx = -dy / dl, ny = dx / dl;
        if (ny > 0) { nx = -nx; ny = -ny; }
        var n = E.halo(E.txt(gzG, tp[0] + 16 * nx, tp[1] + 16 * ny + 8, '1', 22, [E.X, E.Y, E.Z][j], 'middle', 900));
        return n;
      });
      var corners = [[0, 1], [0, 2], [1, 2]].map(function (pr) {
        var a = E.col3(EUL, pr[0]).map(function (x) { return x * 0.16; }), b = E.col3(EUL, pr[1]).map(function (x) { return x * 0.16; });
        var p1 = gz.P(a), p2 = gz.P([a[0] + b[0], a[1] + b[1], a[2] + b[2]]), p3 = gz.P(b);
        return K.paint(E.el('path', { d: 'M ' + p1.join(' ') + ' L ' + p2.join(' ') + ' L ' + p3.join(' '), fill: 'none', 'stroke-width': 3 }, gzG), null, E.HI);
      });
      // 右：好处
      var RX = 390;
      var goodG = E.el('g', {}, s);
      E.txt(goodG, RX, 46, '好处：前几集都见过', 24, E.INK, 'start', 900);
      var goods = ['两个转动接着做：直接相乘（第 3 集）', '乘上一个向量：就知道转到了什么位置', '没有万向锁'].map(function (t1, i) {
        var g = E.el('g', {}, goodG), y = 96 + 58 * i;
        var ck = E.check(g, RX, y - 16, 22, E.GOOD, 4);
        E.txt(g, RX + 36, y, t1, 22, E.INK, 'start', 700);
        return { g: g, ck: ck };
      });
      // 右：代价——6 个等式
      var costG = E.el('g', {}, s);
      E.txt(costG, RX, 46, '代价：冗余', 26, E.ERR, 'start', 900);
      var nineT = E.txt(costG, RX + 150, 46, '9 个数要守上一集的规矩', 21, E.INK2, 'start', 700);
      function chipRow(y, label, chips, color) {
        var g = E.el('g', {}, costG);
        E.txt(g, RX, y + 27, label, 21, E.INK, 'start', 900);
        var cs = chips.map(function (c, i) {
          var cg = E.el('g', {}, g), x = RX + 120 + 96 * i;
          E.tag(cg, x, y, 88, 38, c, color, { size: 19, ink: E.INK, weight: 900, rx: 10 });
          return { g: cg, cx: x + 44 };
        });
        return { g: g, cs: cs, y: y };
      }
      var rowLen = chipRow(78, '长度 = 1', ['第 1 列', '第 2 列', '第 3 列'], E.GOOD);
      var rowPerp = chipRow(132, '两两垂直', ['1 ⊥ 2', '1 ⊥ 3', '2 ⊥ 3'], E.HI);
      var tal = E.tex(costG, RX + 4, 222, '', 28, E.INK, 'start', { w: 400, h: 50 });
      var dofT = E.txt(costG, RX + 4, 282, '= 转动的 3 个自由度', 22, E.GOOD, 'start', 900);
      var burden = E.txt(costG, RX + 4, 330, '多出来的 6 个数：负担', 22, E.ERR, 'start', 900);

      /* ══ 句 4–6：二维网格（从上往下看）+ 右边的数 ══ */
      var B = E.el('g', {}, s);
      var pl = E.plane(B, { ox: OX, oy: OY, u: U, x0: -2, x1: 2, y0: -2, y1: 2, labels: false });
      E.op(pl.warp.g, 0.55);
      var P = pl.P;
      var unit = K.paint(E.el('circle', { cx: OX, cy: OY, r: U, fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '6 6' }, B), null, E.HI);
      unit.style.strokeOpacity = 0.7;
      var trail = E.el('g', {}, B), dots = [];
      for (var k = 1; k <= 100; k++) {
        var r = Math.pow(S, k), a = 45 * D * k, p = P(r * Math.cos(a), r * Math.sin(a));
        dots.push(K.paint(E.el('circle', { cx: p[0].toFixed(1), cy: p[1].toFixed(1), r: 3 }, trail), E.X));
      }
      var ghostSq = K.paint(E.el('path', { fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '6 5' }, B), null, E.INK2);
      var sq = K.paint(E.el('path', { 'stroke-width': 2.5 }, B), E.tint(E.ERR, 26, 'var(--demo-surface)'), E.ERR);
      var e1 = E.vec(B, E.X, 5, { head: 15 }), e2 = E.vec(B, E.Y, 5, { head: 15 });
      var cornerFix = K.paint(E.el('path', { fill: 'none', 'stroke-width': 2.5 }, B), null, E.INK);
      var lenLab = E.halo(E.txt(B, 0, 0, '', 20, E.INK, 'middle', 900));
      var big = E.halo(E.txt(B, OX, OY + 2.5 * U - 2, '格子大了一半', 21, E.ERR, 'middle', 900));
      var exag = E.halo(E.txt(B, OX, OY + 2.5 * U - 2, '积累的误差（夸大画）', 20, E.INK2, 'middle', 700));
      // 右：45° 的矩阵、每列长度、连乘计数
      var RB = 420;
      var r1 = E.el('g', {}, s);
      E.txt(r1, RB, 40, '45° 的矩阵，小数只留两位：', 22, E.INK, 'start', 900);
      var m2 = E.grid(r1, 560, 104, [['0.7071…', '−0.7071…'], ['0.7071…', '0.7071…']], { size: 26, dx: 118, dy: 40, pad: 66, colors: [E.X, E.Y] });
      // 不用根号（KaTeX 的根号是一条很长、靠裁剪显示的 path）：先写长度的平方，再写长度，同 005 的 a² + a² = 1
      var lenL = E.txt(r1, RB, 190, '长度的平方：', 21, E.INK, 'start', 900);
      var lenT = E.tex(r1, RB + 128, 182, '0.71^2+0.71^2\\approx 1.008', 24, E.INK, 'start', { w: 250, h: 44 });
      var len2 = E.txt(r1, RB, 228, '每一列长 ≈ 1.004', 22, E.WARN, 'start', 900);
      var permil = E.txt(r1, RB, 264, '只多了千分之四', 21, E.INK2, 'start', 700);
      var cnt = E.txt(r1, RB, 304, '', 24, E.INK, 'start', 900);
      var lenNow = E.txt(r1, RB, 340, '', 24, E.INK, 'start', 900);
      var notRot = E.tag(r1, RB, 358, 200, 40, '已经不是旋转了', E.ERR, { size: 22, ink: E.ERR, weight: 900, rx: 12, sw: 3 });
      // 右：电脑、积累、扶正
      var r2 = E.el('g', {}, s);
      var pcL = [E.txt(r2, RB, 52, '电脑留十几位小数：0.70710678…', 21, E.INK, 'start', 700),
        E.txt(r2, RB, 92, '走样也慢得多', 21, E.GOOD, 'start', 900),
        E.txt(r2, RB, 148, '可一秒要算几百上千次', 21, E.INK, 'start', 700),
        E.txt(r2, RB, 188, '算久了，照样会走样', 21, E.ERR, 'start', 900)];
      var fixT = E.txt(r2, RB, 250, '隔一阵子，把矩阵扶正：', 22, E.HI, 'start', 900);
      var fixSteps = ['① 长度拉回 1', '② 两两垂直'].map(function (t1, i) { return E.txt(r2, RB + 24, 292 + 40 * i, t1, 22, E.GOOD, 'start', 900); });

      /* ══ 句 7：四元数 vs 矩阵 ══ */
      var C = E.el('g', {}, s);
      var lp = E.card(C, 14, 20, 370, 380, '四元数：1 条规矩', E.ACC, 24);
      var rp = E.card(C, 404, 20, 382, 380, '旋转矩阵：6 个等式', E.VIOLET, 24);
      var qa = E.tex(C, 199, 120, '(0.72,\\ 0,\\ 0,\\ 0.72)', 26, E.INK, 'middle', { w: 330, h: 44 });
      var qdiv = E.tex(C, 199, 186, '\\div\\ \\text{长度}\\ 1.02', 26, E.INK2, 'middle', { w: 330, h: 44 });
      var qb = E.tex(C, 199, 252, '(0.71,\\ 0,\\ 0,\\ 0.71)', 26, E.INK, 'middle', { w: 330, h: 44 });
      var qok = E.el('g', {}, C);
      E.check(qok, 128, 304, 22, E.GOOD, 4); E.txt(qok, 162, 324, '一步就扶正了', 22, E.GOOD, 'start', 900);
      var eqs = ['第 1 列长 1', '第 2 列长 1', '第 3 列长 1', '1 ⊥ 2', '1 ⊥ 3', '2 ⊥ 3'].map(function (t1, i) {
        var g = E.el('g', {}, C), x = 424 + 120 * (i % 3), y = 90 + 60 * Math.floor(i / 3);
        E.tag(g, x, y, 110, 42, t1, i < 3 ? E.GOOD : E.HI, { size: 18, ink: E.INK, weight: 900, rx: 10 });
        return { g: g, cx: x + 55, cy: y + 21 };
      });
      var hard = E.txt(C, 595, 280, '六个都要同时守住', 22, E.INK, 'middle', 900);
      var hard2 = E.txt(C, 595, 324, '扶正麻烦得多', 24, E.ERR, 'middle', 900);

      function sqPath(m) {
        var pts = [[0, 0], [m[0], m[2]], [m[0] + m[1], m[2] + m[3]], [m[1], m[3]]].map(function (q) { return P(q[0], q[1]); });
        return E.ptsPath(pts, true);
      }

      return {
        el: s,
        draw: function (st) {
          var f = E.fade;

          /* ══ 句 1–3 ══ */
          var aOn = f(st, T.mat, 0.4) * (1 - f(st, T.to2d, 0.5));
          E.op(A, aOn);
          E.op(cap, f(st, T.mat + 0.4, 0.4));
          E.op(gzG, f(st, T.cost, 0.4));
          [0, 1, 2].forEach(function (j) {
            var lenOn = f(st, T.len[j], 0.3);
            E.op(ones[j], lenOn);
            var pj = [0, 1, 2].filter(function (pi) { return [[0, 1], [0, 2], [1, 2]][pi].indexOf(j) >= 0; });
            var colOn = Math.max(bump(st, T.len[j], 0.9), Math.max.apply(null, pj.map(function (pi) { return bump(st, T.perp[pi], 0.9); })));
            m3.colBox(j, [E.X, E.Y, E.Z][j], colOn);
            E.op(corners[j], f(st, T.perp[j], 0.3));
          });
          E.op(goodG, f(st, T.goodHead, 0.3) * (1 - f(st, T.cost - 0.2, 0.3)));
          goods.forEach(function (g, i) { E.op(g.g, f(st, T.goods[i], 0.3)); E.drawOn(g.ck, f(st, T.goods[i] + 0.15, 0.35)); });
          E.op(costG, f(st, T.cost, 0.3) * (1 - f(st, T.to2d, 0.5)));
          E.op(nineT, f(st, T.nine, 0.3));
          [rowLen, rowPerp].forEach(function (rw, ri) {
            E.op(rw.g, f(st, ri ? T.perp[0] : T.len[0], 0.3));
            rw.cs.forEach(function (c, i) { E.pop(c.g, c.cx, rw.y + 19, f(st, (ri ? T.perp : T.len)[i], 0.3), 0.7); });
          });
          var TL = '9\\ \\text{个数}-6\\ \\text{个等式}';
          if (st >= T.free) TL += '=\\textcolor{' + H.good + '}{3}';
          tal.setTex(st < T.six ? '9\\ \\text{个数}' : TL);
          E.op(tal, f(st, T.tally, 0.3));
          E.op(dofT, f(st, T.dof, 0.3)); E.op(burden, f(st, T.extra, 0.3));
          burden.style.fontSize = (22 + 4 * bump(st, T.burden, 0.8)) + 'px';

          /* ══ 句 4–6：二维网格 ══ */
          var bOn = f(st, T.to2d, 0.5) * (1 - f(st, T.quat - 0.1, 0.4));
          E.op(B, bOn);
          var kk = Math.round(100 * K.seg(st, T.mul[0], T.mul[0] + T.mul[1]));
          var M;
          if (st < T.mul[0]) M = [1, 0, 0, 1];
          else if (st < T.skew[0]) M = rot2(45 * D * kk).map(function (x) { return x * Math.pow(S, kk); });
          else {
            var end = rot2(Math.PI).map(function (x) { return x * Math.pow(S, 100); });
            M = lerp(end, SKEW, f(st, T.skew[0], T.skew[1]));
            // 扶正：先把两列的长度拉回 1，再把第二列转到和第一列垂直
            var c1 = [M[0], M[2]], c2 = [M[1], M[3]], u1 = f(st, T.unit[0], T.unit[1]), u2 = f(st, T.perpFix[0], T.perpFix[1]);
            var L1 = Math.hypot(c1[0], c1[1]), L2 = Math.hypot(c2[0], c2[1]);
            c1 = c1.map(function (x) { return x / (1 + (L1 - 1) * u1) ; }); c2 = c2.map(function (x) { return x / (1 + (L2 - 1) * u1); });
            var a1 = Math.atan2(c1[1], c1[0]), a2 = Math.atan2(c2[1], c2[0]), tgt = a1 + Math.PI / 2;
            var da = Math.atan2(Math.sin(tgt - a2), Math.cos(tgt - a2)), L2n = Math.hypot(c2[0], c2[1]);
            var a2n = a2 + da * u2;
            c2 = [L2n * Math.cos(a2n), L2n * Math.sin(a2n)];
            M = [c1[0], c2[0], c1[1], c2[1]];
          }
          pl.warp.set(M);
          var o = P(0, 0), q1 = P(M[0], M[2]), q2 = P(M[1], M[3]);
          e1.set(o[0], o[1], q1[0], q1[1]); e2.set(o[0], o[1], q2[0], q2[1]);
          dots.forEach(function (d, i) { E.op(d, (i < kk ? 0.85 : 0) * (1 - f(st, T.skew[0], 0.5))); });
          var Lc = Math.hypot(M[0], M[2]) || 1;
          sq.setAttribute('d', sqPath(M)); ghostSq.setAttribute('d', sqPath(M.map(function (x) { return x / Lc; })));   // 虚线：同一个朝向、边长 1 的格子
          var bigOn = f(st, T.bigger, 0.3) * (1 - f(st, T.skew[0], 0.4));
          E.op(sq, Math.max(bigOn, f(st, T.skew[0], 0.4) * (1 - f(st, T.perpFix[0] + 0.8, 0.4))) * 0.9);
          E.op(ghostSq, bigOn); E.op(big, bigOn);
          E.op(exag, f(st, T.skew[0] + 0.3, 0.4) * (1 - f(st, T.unit[0], 0.4)));
          var L = Math.hypot(M[0], M[2]);
          lenLab.textContent = E.num(L, 2);
          var lp2 = P(M[0] * 0.55 + 0.25 * -M[2] / (L || 1), M[2] * 0.55 + 0.25 * M[0] / (L || 1));
          lenLab.setAttribute('x', lp2[0].toFixed(1)); lenLab.setAttribute('y', (lp2[1] + 7).toFixed(1));
          lenLab.style.fill = L > 1.2 ? E.ERR : '';
          E.op(lenLab, f(st, T.mulTxt, 0.3) * (1 - f(st, T.skew[0], 0.4)));
          // 扶正以后：直角记号
          var cf = f(st, T.perpFix[0] + T.perpFix[1], 0.3);
          var k1 = [M[0] * 0.2, M[2] * 0.2], k2 = [M[1] * 0.2, M[3] * 0.2];
          cornerFix.setAttribute('d', 'M ' + P(k1[0], k1[1]).join(' ') + ' L ' + P(k1[0] + k2[0], k1[1] + k2[1]).join(' ') + ' L ' + P(k2[0], k2[1]).join(' '));
          E.op(cornerFix, cf);
          unit.style.strokeWidth = 2.5 + 2.5 * bump(st, T.unit[0], T.unit[1] + 0.4);

          // 右：45° 矩阵、长度、计数
          E.op(r1, f(st, T.m45, 0.3) * (1 - f(st, T.pc - 0.2, 0.3)));
          m2.set(st < T.round + 0.6 ? [['0.7071…', '−0.7071…'], ['0.7071…', '0.7071…']] : [['0.71', '−0.71'], ['0.71', '0.71']]);
          [0, 1, 2, 3].forEach(function (c) { m2.cellBox(c, E.HI, bump(st, T.round + 0.4, 0.9) + bump(st, T.r71, 0.7)); });
          E.op(lenT, f(st, T.colLen, 0.3)); E.op(lenL, f(st, T.colLen, 0.3)); E.op(len2, f(st, T.l1004, 0.3)); E.op(permil, f(st, T.permil, 0.3));
          cnt.textContent = '连乘了 ' + kk + ' 次';
          E.op(cnt, f(st, T.mulTxt, 0.3));
          lenNow.textContent = '每一列长：' + E.num(Math.pow(S, kk), kk >= 100 ? 1 : 2);
          lenNow.style.fill = kk >= 100 ? E.ERR : '';
          E.op(lenNow, f(st, T.mulTxt, 0.3));
          E.scaleAt(lenNow, RB, 332, 1 + 0.15 * bump(st, T.l15, 0.8));
          E.pop(notRot.g, RB + 100, 378, f(st, T.notRot, 0.3), 0.8);
          // 右：电脑、积累、扶正
          E.op(r2, f(st, T.pc, 0.3) * (1 - f(st, T.quat - 0.1, 0.4)));
          [T.pc, T.small, T.many, T.accum].forEach(function (a, i) { E.op(pcL[i], f(st, a, 0.3)); });
          E.op(fixT, f(st, T.fix - 0.8, 0.3));
          E.op(fixSteps[0], f(st, T.unit[0], 0.3)); E.op(fixSteps[1], f(st, T.perpFix[0], 0.3));

          /* ══ 句 7 ══ */
          E.op(C, f(st, T.quat, 0.4));
          E.op(lp.g, f(st, T.quat, 0.3)); E.op(rp.g, f(st, T.six2 - 0.3, 0.3));
          E.op(qa, f(st, T.quat + 0.3, 0.3)); E.op(qdiv, f(st, T.divide, 0.3)); E.op(qb, f(st, T.divide + 0.6, 0.3));
          E.op(qok, f(st, T.fixed, 0.3));
          eqs.forEach(function (q, i) { E.pop(q.g, q.cx, q.cy, f(st, T.six2 + 0.22 * i, 0.3), 0.7); });
          E.op(hard, f(st, T.six2 + 1.4, 0.3)); E.op(hard2, f(st, T.hard, 0.3));
        }
      };
    }
  };
})();
