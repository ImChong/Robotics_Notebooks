// 005 第 4 幕：直接平均——跳出了旋转矩阵（路线 §L0 自测题 1）。左边一张从上往下看的地面网格：x 是机器人原来的「前」、y 是「左」；右栏放算式、G1 与示意图。
// 句 1「机器人」右栏出 G1，「朝前」网格上两根小箭头标「0°：朝前」，「转到朝左」G1 原地往左转 90°，网格上同时有一组虚线小箭头从 0° 跟着转过去、
//    描出 0°→90° 的弧，转到了标「90°：朝左」；「中间那一刻」弧鼓一下，45° 处一个问号；
// 句 2 G1 退场，右栏「两个矩阵」出 0° 和 90° 的矩阵，「对应位置」四对数依次一起框亮，「除以二」出「÷ 2 =」，结果矩阵四个数按念到的顺序写出来；
// 句 3「画出来」网格从原样变成平均出来的矩阵：两根小箭头转到 45°（虚线标 45°），「长度」「零点七一」两根上写橙色 0.71，「格子」两根小箭头围成的小方块（红）和原来的小方块（虚线）比，「缩小」鼓一下；
// 句 4 右栏算式收起、结果矩阵挪到顶上，「第一条破了」「航列式」「零点二五」「零点五」「第二条也破了」三行依次写出，方块里写「面积 0.5」；
// 句 5「不是旋转」右栏顶上橙牌，「身子」右栏出 G1（按 45° 摆，后面垫一台正常的淡影子），「摆」G1 横向压成 0.71 倍，「压瘦」写「压瘦了一圈」；
// 句 6 网格回到原样，「朝前」标 0°，「朝后」虚线小箭头从 0° 转过 180°、描出半圈弧，转到了标「180°：朝后」；右栏出 0° 和 180° 的两个矩阵，「一平均」结果全是 0，
//    「所有的点」网格交点上亮起一片点，「缩到了原点」网格、点和小箭头一起缩成原点一个点；
// 句 7 网格回来，「箭头尖」往右那根沿单位圆从 0° 转到 90°，「圆上」圆亮一下；「直接平均」画橙色虚线弦，「直线」一个橙点沿弦走过去，
//    「离开了圆」弦和圆弧之间的月牙涂橙；「三维」右栏出示意图：一个弯曲的球面（特殊正交群），两个朝向之间的蓝色弧沿着面走、橙色虚线弦穿进里面，「跳出」橙叉；
// 句 8「相加」网格变成两个矩阵直接相加的结果（两根小箭头长 1.41），右栏「R₁ + R₂」一行打叉；「接着做」网格回到原样，往右那根先转 45°、再转 45°，
//    右栏「R₂ R₁」一行打勾，「第三集」出「第 3 集：矩阵相乘 = 接着做」。
// 时刻取自配音的逐词时间（分镜时间 = 念到这个词的时刻），见 ../ep005.py。
(function () {
  var E = window.EP005, K = E.K, H = E.HEX;
  var D = Math.PI / 180;
  var OX = 200, OY = 214, U = 82;                           // 网格：x、y 都是 −2…2
  var MX = 622;
  var I = [1, 0, 0, 1], R90 = [0, -1, 1, 0], R180 = [-1, 0, 0, -1], AVG = [0.5, -0.5, 0.5, 0.5], Z = [0, 0, 0, 0], SUM = [1, -1, 1, 1];
  var T = {
    // turn：G1 和虚线小箭头一起转 90°，转到了才标「90°：朝左」；f180：「朝后」起虚线小箭头转 180°，「一百八十度」转到
    robot: 2.62, front: 3.26, turn: [3.87, 1.2], mid: 7.24, ask: 8.38,
    out1: 10.75, mats: 11.89, pairs: 12.51, div: 13.8, res: [14.85, 16.76, 15.61, 17.54],   // res 按矩阵里 [左上, 右上, 左下, 右下] 的顺序
    draw: [19.4, 1.6], half: 21.73, len: 22.76, l71: 23.99, grid: 24.9, shrink: 25.7,
    up: [27.0, 0.6], b1: 28.53, det: [29.56, 30.17, 30.96, 32.17], b2: 32.93,
    notRot: 35.49, body: 36.73, squash: [38.26, 1.0], thin: 39.59,
    back: [41.5, 0.7], f0: 44.01, f180: [44.43, 1.4], mats2: 45.44, avg2: 46.75, zeros: [47.39, 47.6, 47.99, 48.46], dots: 48.87, collapse: [49.61, 0.9],
    restore: [51.8, 0.7], sweep: [53.53, 1.0], onCircle: 54.55, chord: 55.13, walk: [56.16, 1.2], leave: 57.81, so3: 59.8, jump: 61.88,
    add: 65.48, toSum: [66.6, 0.8], addX: 68.52, toI: [69.9, 0.5], steps: [70.68, 71.4], mul: 72.1, ep3: 72.78
  };

  function lerp4(A, B, p) { return A.map(function (a, i) { return a + (B[i] - a) * p; }); }
  function bump(st, a, d) { var u = K.seg(st, a, a + d); return u > 0 && u < 1 ? Math.sin(Math.PI * u) : 0; }
  function matAt(st) {
    var f = E.fade;
    if (st < T.back[0]) return lerp4(I, AVG, f(st, T.draw[0], T.draw[1]));
    if (st < T.collapse[0]) return lerp4(AVG, I, f(st, T.back[0], T.back[1]));
    if (st < T.restore[0]) return lerp4(I, Z, f(st, T.collapse[0], T.collapse[1]));
    if (st < T.sweep[0]) return lerp4(Z, I, f(st, T.restore[0], T.restore[1]));
    if (st < T.toSum[0]) return I;
    if (st < T.toI[0]) return lerp4(I, SUM, f(st, T.toSum[0], T.toSum[1]));
    if (st < T.steps[0]) return lerp4(SUM, I, f(st, T.toI[0], T.toI[1]));
    return E.rot(45 * D * (f(st, T.steps[0], 0.5) + f(st, T.steps[1], 0.5)));   // 先转 45°，再转 45°
  }

  E.scenes[3] = {
    title: '直接平均：跳出了旋转矩阵',
    dur: 74.64,
    cues: [
      { at: 0, s: '机器人从朝前（0°）转到朝左（90°）：中间的朝向是什么？' },
      { at: 10.7, s: '直接平均：两个矩阵对应位置相加，再除以 2' },
      { at: 19.2, s: '小箭头转了一半，可长度缩成 0.71，格子也缩小了' },
      { at: 27.34, s: '长度不是 1，行列式 0.5：两条规矩都破了' },
      { at: 34.95, s: '已经**不是旋转**：身子按它摆，机器人被压瘦一圈' },
      { at: 41.36, s: '朝前和朝后一平均：全是 0，所有的点缩到原点' },
      { at: 51.68, s: '箭头尖在圆上，平均却走圆里的直线 → **跳出**旋转矩阵' },
      { at: 64.66, s: '错在相加：两个转动接着做，要用乘法（第 3 集）' }
    ],
    build: function (K) {
      var s = K.sceneSvg('机器人从朝前转到朝左，零度和九十度的旋转矩阵直接平均，得到每列 (0.5, 0.5) 和 (−0.5, 0.5) 的矩阵：小箭头转了一半但长度缩成 0.71，格子缩小，行列式 0.5，两条规矩都破了，已经不是旋转，身子按它摆机器人被压瘦；朝前和朝后一平均全是零，所有点缩到原点；箭头尖都在圆上，直接平均走的是圆里面的直线，三维也一样会跳出特殊正交群；错在相加，转动接着做要用乘法');

      /* ══ 左：地面网格（从上往下看） ══ */
      var pl = E.plane(s, { ox: OX, oy: OY, u: U, x0: -2, x1: 2, y0: -2, y1: 2, labels: false });
      var P = pl.P, box = pl.box;
      E.halo(E.txt(s, box[2] + 22, OY + 8, 'x 前', 22, E.INK2, 'start', 700));
      E.halo(E.txt(s, OX + 12, box[1] - 4, 'y 左', 22, E.INK2, 'start', 700));
      E.op(pl.warp.g, 0.6);
      var unit = K.paint(E.el('circle', { cx: OX, cy: OY, r: U, fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '6 6' }, s), null, E.HI);
      var lens = K.paint(E.el('path', {}, s), E.tint(E.ERR, 40, 'var(--demo-surface)'));
      var chord = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '8 6' }, s), null, E.ERR);
      var walker = K.paint(E.el('circle', { r: 8 }, s), E.ERR);
      var sweepArc = K.paint(E.el('path', { fill: 'none', 'stroke-width': 4 }, s), null, E.Z);   // 箭头尖沿圆转：绕 z 轴的转动，蓝
      var ghostSq = K.paint(E.el('path', { fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '6 5' }, s), null, E.INK2);
      var sq = K.paint(E.el('path', { 'stroke-width': 2.5 }, s), E.tint(E.ERR, 30, 'var(--demo-surface)'), E.ERR);
      var area = E.halo(E.txt(s, 0, 0, '面积 0.5', 20, E.ERR, 'middle', 900));
      // 0° / 90° / 180° 的虚线小箭头与标签、中间的弧和问号、45° 虚线
      var ghostA = [E.vec(s, E.X, 3.5, { head: 13, dash: '6 5' }), E.vec(s, E.Y, 3.5, { head: 13, dash: '6 5' })];
      var midArc = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '4 5' }, s), null, E.Z);   // 0° → 90° 的转动：蓝
      var ask = E.halo(E.txt(s, 0, 0, '?', 40, E.HI, 'middle', 900));
      var ray45 = K.paint(E.el('path', { fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '6 6' }, s), null, E.HI);
      var lab45 = E.halo(E.txt(s, 0, 0, '45°', 22, E.HI, 'middle', 900));
      var labs = [E.halo(E.txt(s, 0, 0, '0°：朝前', 21, E.X, 'start', 900)), E.halo(E.txt(s, 0, 0, '', 21, E.X, 'middle', 900))];
      var dots = [];
      for (var x = -2; x <= 2; x++) for (var y = -2; y <= 2; y++) dots.push({ p: [x, y], el: K.paint(E.el('circle', { r: 4.5 }, s), E.INK) });
      var e1 = E.vec(s, E.X, 5, { head: 16 }), e2 = E.vec(s, E.Y, 5, { head: 16 });
      var lens71 = [E.ERR, E.ERR].map(function () { return E.halo(E.txt(s, 0, 0, '0.71', 21, E.ERR, 'middle', 900)); });
      var dot0 = K.paint(E.el('circle', { cx: OX, cy: OY, r: 7 }, s), E.ERR);

      /* ══ 右栏 ══ */
      // G1（句 1 转身；句 5 按平均出来的矩阵摆：后面垫一台正常的淡影子）
      var gRobot = E.el('g', {}, s);
      var shadow = E.g1(gRobot, MX, 378, 238, { az: 135, el: 20 });
      var g1 = E.g1(gRobot, MX, 378, 238, { az: 135, el: 20 });
      var fr = E.frame3(gRobot, 4.5);
      var thin = E.halo(E.txt(gRobot, MX, 414, '压瘦了一圈', 24, E.ERR, 'middle', 900));
      var turnLab = E.halo(E.txt(gRobot, MX, 414, '', 22, E.INK, 'middle', 900));
      // 两个矩阵相加、除以二
      var rowA = E.el('g', {}, s);
      var mA = E.mat(rowA, 530, 80, I, { size: 26, dx: 52, dy: 38, pad: 34, cw: 44, colors: [E.X, E.Y, E.X, E.Y] });
      var mB = E.mat(rowA, 706, 80, R90, { size: 26, dx: 52, dy: 38, pad: 34, cw: 44, colors: [E.X, E.Y, E.X, E.Y] });
      E.txt(rowA, 618, 92, '+', 36, E.INK, 'middle', 700);
      E.txt(rowA, 530, 146, '0°：朝前', 20, E.INK2, 'middle', 700);
      var labB = E.txt(rowA, 706, 146, '90°：朝左', 20, E.INK2, 'middle', 700);
      var rowB = E.el('g', {}, s), rowBin = E.el('g', {}, rowB);
      E.txt(rowBin, 476, 260, '÷ 2 =', 30, E.INK, 'start', 700);
      var mR = E.mat(rowBin, 668, 250, AVG, { size: 28, dx: 82, dy: 42, pad: 44, cw: 72, colors: [E.X, E.Y, E.X, E.Y] });
      // 句 4 三行
      var badRows = E.el('g', {}, s);
      function badRow(y, txt1, color) {
        var g = E.el('g', {}, badRows);
        var x = E.cross(g, 480, y - 8, 9, E.ERR, 4);
        var t = E.txt(g, 500, y, txt1, 22, color || E.INK, 'start', 900);
        return { g: g, x: x, t: t };
      }
      var bad1 = badRow(196, '规矩一：长度 0.71，不是 1');
      var detLab = E.txt(badRows, 476, 252, '行列式 =', 22, E.INK, 'start', 900);
      var detEq = E.tex(badRows, 572, 246, '', 26, E.INK, 'start', { w: 230, h: 44 });
      var bad2 = badRow(304, '规矩二：行列式 0.5，不是 1');
      var notRot = E.tag(s, 512, 14, 220, 44, '不是旋转', E.ERR, { size: 24, ink: E.ERR, weight: 900, rx: 12, sw: 3 });
      // 句 7：两行 + 弯曲的球面示意图
      var why = E.el('g', {}, s);
      var whyA = E.txt(why, 470, 54, '旋转：箭头尖都在圆上', 22, E.GOOD, 'start', 900);
      var whyB = E.txt(why, 470, 94, '平均：走圆里面的直线', 22, E.ERR, 'start', 900);
      var sph = E.el('g', {}, why);
      K.paint(E.el('circle', { cx: MX, cy: 258, r: 92, 'stroke-width': 2.5 }, sph), E.tint(E.VIOLET, 16, 'var(--demo-surface)'), E.VIOLET);
      K.paint(E.el('ellipse', { cx: MX, cy: 300, rx: 80, ry: 20, fill: 'none', 'stroke-width': 1.8, 'stroke-dasharray': '6 6' }, sph), null, E.VIOLET);
      var sArc = K.paint(E.el('path', { d: 'M 566 236 Q 622 166 678 236', fill: 'none', 'stroke-width': 4 }, sph), null, E.ACC);
      var sChord = K.paint(E.el('path', { d: 'M 566 236 L 678 236', fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '8 6' }, sph), null, E.ERR);
      [[566, 'R₁'], [678, 'R₂']].forEach(function (d) {
        K.paint(E.el('circle', { cx: d[0], cy: 236, r: 7 }, sph), E.INK);
        E.halo(E.txt(sph, d[0] + (d[0] < MX ? -16 : 16), 228, d[1], 22, E.INK, 'middle', 900));
      });
      E.txt(sph, MX, 384, '特殊正交群：弯的，不是平的', 21, E.VIOLET, 'middle', 900);
      var jumpG = E.el('g', {}, why);
      E.cross(jumpG, MX, 236, 10, E.ERR, 4.5);
      E.halo(E.txt(jumpG, MX, 270, '跳出', 22, E.ERR, 'middle', 900));
      // 句 8：相加 vs 相乘
      var ops = E.el('g', {}, s);
      var opAdd = E.tex(ops, 500, 86, 'R_1+R_2', 32, E.INK, 'start', { w: 160, h: 50 });
      var opAddX = E.el('g', {}, ops); E.cross(opAddX, 482, 74, 10, E.ERR, 4.5);
      var opAddT = E.txt(ops, 630, 84, '一般不是旋转', 22, E.ERR, 'start', 900);
      var opMul = E.tex(ops, 500, 166, 'R_2\\,R_1', 32, E.INK, 'start', { w: 160, h: 50 });
      var opMulOk = E.check(ops, 470, 146, 22, E.GOOD, 4);
      var opMulT = E.txt(ops, 630, 164, '还是旋转', 22, E.GOOD, 'start', 900);
      var ep3 = E.tag(ops, 470, 222, 320, 42, '第 3 集：矩阵相乘 = 接着做', E.VIOLET, { size: 21, ink: E.INK, weight: 900, rx: 12 });

      function at(m, p) { var q = E.apply(m, p); return P(q[0], q[1]); }
      function sqPath(m) { return 'M ' + [[0, 0], [1, 0], [1, 1], [0, 1]].map(function (p) { var q = at(m, p); return q[0].toFixed(1) + ' ' + q[1].toFixed(1); }).join(' L ') + ' Z'; }
      function arcD(a0, a1, r) {
        var p = P(r * Math.cos(a0), r * Math.sin(a0)), q = P(r * Math.cos(a1), r * Math.sin(a1));
        return 'M ' + p[0].toFixed(1) + ' ' + p[1].toFixed(1) + ' A ' + (r * U) + ' ' + (r * U) + ' 0 ' + (a1 - a0 > Math.PI ? 1 : 0) + ' 0 ' + q[0].toFixed(1) + ' ' + q[1].toFixed(1);
      }

      return {
        el: s,
        draw: function (st, t) {
          var f = E.fade, M = matAt(st), o = P(0, 0);
          pl.warp.set(M);
          var q1 = at(M, [1, 0]), q2 = at(M, [0, 1]);
          e1.set(o[0], o[1], q1[0], q1[1]); e2.set(o[0], o[1], q2[0], q2[1]);
          var collapsed = st >= T.collapse[0] + 0.5 && st < T.restore[0] + 0.3;
          E.op(dot0, collapsed ? f(st, T.collapse[0] + 0.5, 0.3) : 0);
          E.op(unit, 0.55 + 0.45 * bump(st, T.onCircle, 1.0));
          E.glow(unit, E.HI, bump(st, T.onCircle, 1.0) > 0.05);

          /* 句 1：0° / 90° 两组、弧和问号；句 6：0° / 180° */
          var phase2 = st >= T.back[0];
          // 虚线小箭头从 0° 转过去（句 1 和 G1 一起转 90°，句 6 转 180°），句 6 留到「所有的点」再收
          var turnW = phase2 ? T.f180 : T.turn, gEnd = turnW[0] + turnW[1];
          var gAng = (phase2 ? Math.PI : Math.PI / 2) * f(st, turnW[0], turnW[1]);
          var ga = E.rot(gAng), g1p = at(ga, [1, 0]), g2p = at(ga, [0, 1]);
          ghostA[0].set(o[0], o[1], g1p[0], g1p[1]); ghostA[1].set(o[0], o[1], g2p[0], g2p[1]);
          var out2 = 1 - f(st, T.dots - 0.3, 0.4);
          var ghostOn = phase2 ? f(st, turnW[0], 0.2) * out2 : f(st, turnW[0], 0.2) * (1 - f(st, T.shrink + 1.2, 0.5)) * 0.8;
          ghostA.forEach(function (g) { E.op(g.g, ghostOn); });
          labs[0].setAttribute('x', P(1.06, 0)[0]); labs[0].setAttribute('y', P(0, -0.36)[1]);   // 圆外面
          E.op(labs[0], phase2 ? f(st, T.f0, 0.3) * out2 : f(st, T.front, 0.3) * (1 - f(st, T.draw[0], 0.4)));
          labs[1].textContent = phase2 ? '180°：朝后' : '90°：朝左';
          var lp = phase2 ? P(-1.06, -0.36) : P(-0.72, 1.2);   // 90° 的写在 y 轴左边，右上方留给弧和问号
          labs[1].setAttribute('x', lp[0]); labs[1].setAttribute('y', lp[1]); labs[1].setAttribute('text-anchor', phase2 ? 'end' : 'middle');
          E.op(labs[1], phase2 ? f(st, gEnd, 0.3) * out2 : f(st, gEnd, 0.3) * (1 - f(st, T.draw[0], 0.4)));
          midArc.setAttribute('d', gAng > 0.01 ? arcD(0, gAng, 1.32) : '');   // 跟着虚线小箭头描出来
          E.op(midArc, f(st, turnW[0], 0.2) * (phase2 ? out2 : 1 - f(st, T.draw[0], 0.4)));
          midArc.style.strokeWidth = 3 + 2 * bump(st, T.mid, 0.8);
          var ap = P(1.24, 1.24);
          ask.setAttribute('x', ap[0]); ask.setAttribute('y', ap[1] + 14);
          E.pop(ask, ap[0], ap[1], f(st, T.ask, 0.3) * (1 - f(st, T.draw[0], 0.4)), 0.6);

          /* 句 3：45° 虚线、0.71、方块 */
          var r45 = P(1.6 * Math.cos(Math.PI / 4), 1.6 * Math.sin(Math.PI / 4));
          ray45.setAttribute('d', 'M ' + o[0] + ' ' + o[1] + ' L ' + r45[0].toFixed(1) + ' ' + r45[1].toFixed(1));
          lab45.setAttribute('x', r45[0] + 10); lab45.setAttribute('y', r45[1] - 6);
          var on3 = f(st, T.half, 0.3) * (1 - f(st, T.back[0], 0.4));
          E.op(ray45, on3); E.op(lab45, on3);
          [[0.5, 0.5], [-0.5, 0.5]].forEach(function (v, i) {   // 「0.71」写在两个箭头尖外面一点
            var tp = P(i ? -1.32 : 1.32, 0.62);   // 圆外、45° 虚线下面、原来的小方块外面
            lens71[i].setAttribute('x', tp[0].toFixed(1)); lens71[i].setAttribute('y', (tp[1] + 8).toFixed(1));
            E.op(lens71[i], f(st, T.l71, 0.3) * (1 - f(st, T.back[0], 0.4)));
          });
          var lb = bump(st, T.len, 0.7);
          e1.line.style.strokeWidth = e2.line.style.strokeWidth = 5 + 3 * lb;
          sq.setAttribute('d', sqPath(M)); ghostSq.setAttribute('d', sqPath(I));
          var sqOn = f(st, T.grid, 0.4) * (1 - f(st, T.back[0], 0.4));
          E.op(sq, sqOn * (0.85 + 0.15 * bump(st, T.shrink, 0.8))); E.op(ghostSq, sqOn);
          var ac = P(-1.1, 1.35);   // 方块太小写不下：写在左上方圆外
          area.setAttribute('x', ac[0].toFixed(1)); area.setAttribute('y', (ac[1] + 7).toFixed(1));
          E.op(area, f(st, T.det[0], 0.3) * (1 - f(st, T.back[0], 0.4)));

          /* 句 6：所有的点缩到原点 */
          dots.forEach(function (d) {
            var q = at(M, d.p);
            d.el.setAttribute('cx', q[0].toFixed(1)); d.el.setAttribute('cy', q[1].toFixed(1));
            E.op(d.el, f(st, T.dots, 0.4) * (1 - f(st, T.restore[0], 0.3)));
          });

          /* 句 7：往右那根沿圆转一段（蓝：绕 z 轴转）、弦、橙点、月牙 */
          var sw = f(st, T.sweep[0], T.sweep[1]);
          sweepArc.setAttribute('d', sw > 0.01 ? arcD(0, Math.PI / 2 * sw, 1) : '');
          E.op(sweepArc, f(st, T.sweep[0], 0.2) * (1 - f(st, T.add - 0.3, 0.4)));
          var c0 = P(1, 0), c1 = P(0, 1);
          chord.setAttribute('d', 'M ' + c0[0] + ' ' + c0[1] + ' L ' + c1[0] + ' ' + c1[1]);
          E.drawOn(chord, f(st, T.chord, 0.6) * (1 - f(st, T.add - 0.3, 0.4) > 0.01 ? 1 : 0));
          E.op(chord, f(st, T.chord, 0.1) * (1 - f(st, T.add - 0.3, 0.4)));
          var wk = f(st, T.walk[0], T.walk[1]);
          walker.setAttribute('cx', (c0[0] + (c1[0] - c0[0]) * wk).toFixed(1)); walker.setAttribute('cy', (c0[1] + (c1[1] - c0[1]) * wk).toFixed(1));
          E.op(walker, f(st, T.walk[0], 0.2) * (1 - f(st, T.add - 0.3, 0.4)));
          lens.setAttribute('d', arcD(0, Math.PI / 2, 1) + ' L ' + c0[0] + ' ' + c0[1] + ' Z');
          E.op(lens, 0.8 * f(st, T.leave, 0.5) * (1 - f(st, T.add - 0.3, 0.4)));

          /* ══ 右栏 ══ */
          // G1：句 1 转身（2.62 出现，10.75 退场）；句 5 按平均的矩阵摆，横向压成 0.71 倍（36.73 出现，41.4 退场）
          var phase5 = st >= T.notRot;
          var yaw = phase5 ? 45 * D : 90 * D * f(st, T.turn[0], T.turn[1]);
          var sq5 = phase5 ? 1 + (Math.cos(45 * D) - 1) * f(st, T.squash[0], T.squash[1]) : 1;
          g1.set(yaw, sq5); shadow.set(yaw, 1);
          g1.view.clearTint();
          if (phase5) g1.tint('all', E.ERR, 0.5 * f(st, T.thin, 0.5));
          fr.set(g1.frame(0.5));
          E.op(shadow.g, phase5 ? 0.28 * f(st, T.squash[0], 0.4) : 0);
          var robotOn = phase5 ? f(st, T.body, 0.4) * (1 - f(st, T.back[0], 0.4)) : f(st, T.robot, 0.4) * (1 - f(st, T.out1, 0.4));
          E.op(gRobot, robotOn);
          E.op(thin, phase5 ? f(st, T.thin, 0.3) : 0);
          turnLab.textContent = st < T.turn[0] + T.turn[1] / 2 ? '朝前' : '朝左';
          E.op(turnLab, phase5 ? 0 : f(st, T.front, 0.3));

          // 两个矩阵 + 除以二（句 2；句 6 换成 0° 和 180°）
          mB.set(phase2 ? R180 : R90);
          labB.textContent = phase2 ? '180°：朝后' : '90°：朝左';
          var aOn = phase2 ? f(st, T.mats2, 0.35) * (1 - f(st, T.restore[0], 0.4)) : f(st, T.mats, 0.35) * (1 - f(st, T.up[0], 0.3));
          E.op(rowA, aOn);
          for (var k = 0; k < 4; k++) {
            var pk = phase2 ? 0 : bump(st, T.pairs + 0.27 * k, 0.5);
            mA.cellBox(k, E.HI, pk); mB.cellBox(k, E.HI, pk);
          }
          // 结果矩阵：句 2 在下面，句 4 挪到顶上；句 6 换成全零
          var upP = phase2 ? 0 : f(st, T.up[0], T.up[1]);
          rowBin.setAttribute('transform', 'translate(0 ' + (-160 * upP).toFixed(1) + ')');
          mR.set(phase2 ? Z : AVG);
          var resT = phase2 ? T.zeros : T.res;
          mR.cells.forEach(function (c, k) { E.op(c, f(st, resT[k], 0.25)); });
          var bOn = phase2 ? f(st, T.avg2, 0.3) * (1 - f(st, T.restore[0], 0.4)) : f(st, T.div, 0.3) * (1 - f(st, T.notRot, 0.4));
          E.op(rowB, bOn);
          mR.brackets.forEach(function (b) { E.op(b, phase2 ? f(st, T.zeros[0], 0.25) : f(st, T.res[0], 0.25)); });

          // 句 4：两条规矩都破了
          E.op(badRows, 1 - f(st, T.notRot, 0.4));
          E.op(bad1.g, f(st, T.b1, 0.3)); E.drawOn(bad1.x, f(st, T.b1 + 0.39, 0.35));
          var DE = '';
          if (st >= T.det[1]) DE = '0.25';
          if (st >= T.det[2]) DE += '+0.25';
          if (st >= T.det[3]) DE += '=\\textcolor{' + E.HEX.err + '}{0.5}';
          detEq.setTex(DE || '\\,');
          E.op(detLab, f(st, T.det[0], 0.3)); E.op(detEq, f(st, T.det[1], 0.3));
          E.op(bad2.g, f(st, T.b2, 0.3)); E.drawOn(bad2.x, f(st, T.b2 + 0.59, 0.35));
          E.pop(notRot.g, 622, 36, f(st, T.notRot, 0.35) * (1 - f(st, T.back[0], 0.4)), 0.85);

          // 句 7：两行 + 球面示意图
          E.op(why, f(st, T.onCircle - 0.2, 0.3) * (1 - f(st, T.add - 0.3, 0.4)));
          E.op(whyA, f(st, T.onCircle - 0.2, 0.3)); E.op(whyB, f(st, T.walk[0], 0.3));
          E.op(sph, f(st, T.so3, 0.4));
          E.drawOn(sArc, f(st, T.so3 + 0.4, 0.6)); E.drawOn(sChord, f(st, T.so3 + 1.0, 0.6));
          E.pop(jumpG, MX, 248, f(st, T.jump, 0.3), 0.6);

          // 句 8：相加 vs 相乘
          E.op(ops, f(st, T.add, 0.3));
          E.op(opAdd, f(st, T.add, 0.3)); E.drawOn(opAddX.firstChild, f(st, T.addX, 0.35)); E.op(opAddT, f(st, T.addX, 0.3));
          E.op(opMul, f(st, T.steps[0] - 0.3, 0.3)); E.drawOn(opMulOk, f(st, T.mul, 0.35)); E.op(opMulT, f(st, T.mul, 0.3));
          E.pop(ep3.g, 630, 243, f(st, T.ep3, 0.35), 0.85);
        }
      };
    }
  };
})();
