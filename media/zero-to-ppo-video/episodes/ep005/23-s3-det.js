// 005 第 3 幕：规矩二——行列式等于 1（不照镜子），最后给两条规矩起名：特殊正交群 SO(3)。左边网格（x、y 都是 −3…3），右栏放矩阵和算式。
// 句 1「照镜子」y 轴上画一面镜子，「左右一翻」网格、两根小箭头和一面亮黄小旗（F 形，不对称）左右翻过去；右栏出镜子的矩阵，
//    「长度」「角度」两行打勾，「拦不住」写出「规矩一拦不住它」；
// 句 2「小旗」鼓一下，「不管怎么转」原来那面小旗的虚线影子绕原点转一整圈，怎么转都和翻过去的小旗对不上，「这个样子」红叉「转不出来」；
// 句 3 小旗退场、网格回到原样，右栏顶上出「行列式」；「小方块」两根小箭头围成的小方块涂亮黄、写「面积 1」，方块里一支逆时针的小弯箭头；
//    「变换以后」网格横着拉长到 2 倍，「面积」「几倍」方块写「面积 2」，右栏一行「拉长 2 倍 → 行列式 2」；
//    「翻了面」网格变成镜子（中途压扁再翻过去），方块翻到背面变成红色，小弯箭头变成顺时针，「负号」右栏一行「镜子：翻面 → 行列式 −1」；
// 句 4 网格转回 36.87° 的旋转；右栏换成字母矩阵 [[a, b], [c, d]]，「左上」「右下」两个数框亮、连一条绿对角线，「减去」「右上」「左下」框亮、连红对角线，
//    下面一行一段段写出「ad − bc」；
// 句 5 字母换成旋转矩阵的数，「零点八乘零点八」「减去」「负零点六乘零点六」「等于一」第二行写出算式，「面积不变」方块写「面积 1」，「没翻面」小弯箭头亮一下；
// 句 6「镜子」网格翻成镜子，矩阵换成镜子的数，第三行写出「(−1)×1 − 0×0 = −1」，「第二条规矩」顶上改成「规矩二：行列式 = 1」，两行各打勾、打叉；
// 句 7「三维」换成斜着看的小坐标架（慢慢转），右栏「正交：每列长度 1、两两垂直」「特殊：行列式 = 1」两行，
//    「特殊正交群」大字 SO(3)，三个字母下面写特殊 / 正交 / 三维，念到「正交」「特殊」各自和上面那行一起亮。
// 时刻取自配音的逐词时间（分镜时间 = 念到这个词的时刻），见 ../ep005.py。
(function () {
  var E = window.EP005, K = E.K, H = E.HEX;
  var D = Math.PI / 180;
  var OX = 222, OY = 216, U = 60;                           // 网格：x、y 都是 −3…3
  var MX = 622, EX = 466;
  var I = [1, 0, 0, 1], MIR = [-1, 0, 0, 1], S = [2, 0, 0, 1], Rm = [0.8, -0.6, 0.6, 0.8];
  var FLAG = [[[0.7, 0.4], [0.7, 2.2], [1.6, 2.2]], [[0.7, 1.3], [1.35, 1.3]]];   // 离开 x 轴，不压往右那根小箭头
  var T = {
    mirror: 2.26, flip: [3.33, 0.9], len: 3.78, ang: 4.77, rule1: 6.27,
    flag: 8.25, spin: [9.49, 11.2], no: 11.35,
    reset: [12.9, 0.7], det: 14.15, arrows: 15.31, square: 16.42, toS: [17.87, 0.8], area: 19.38, row1: 20.32,
    toM: [21.63, 1.0], row2: 22.82,
    toR: [24.4, 0.9], abcd: 25.09, a: 26.51, d: 27.47, minus: 28.35, b: 28.71, c: 29.56,
    nums: 31.8, r: [32.47, 33.18, 33.94, 34.46, 35.21, 36.25], area1: 36.72, noflip: 37.99,
    toMir: [39.57, 0.9], m: [40.46, 41.24, 41.65, 42.99], rule2: 44.43, marks: 46.22,
    to3: 47.98, rows3: [49.89, 50.21], so3: 54.16, named: 54.66, ortho: 55.57, special: 57.18
  };

  function lerp4(A, B, p) { return A.map(function (a, i) { return a + (B[i] - a) * p; }); }
  // 这一刻网格的矩阵 M 与可见度 vis。「左右一翻」「翻了面」「镜子」三处要演给人看，照旧逐元素插值（中途压扁再翻过去）；
  // 镜子 → 原样、镜子 → 旋转这两处只是换下一段的起点，不演变形：淡出、换矩阵、再淡入（逐元素插值会在中途把整张网格压成一条，闪一下）
  function matAt(st) {
    var f = E.fade;
    function swap(A, B, w) { var p = K.seg(st, w[0], w[0] + w[1]); return { M: p < 0.5 ? A : B, vis: Math.abs(1 - 2 * p) }; }
    if (st < T.reset[0]) return { M: lerp4(I, MIR, f(st, T.flip[0], T.flip[1])), vis: 1 };
    if (st < T.toS[0]) return swap(MIR, I, T.reset);
    if (st < T.toM[0]) return { M: lerp4(I, S, f(st, T.toS[0], T.toS[1])), vis: 1 };
    if (st < T.toR[0]) return { M: lerp4(S, MIR, f(st, T.toM[0], T.toM[1])), vis: 1 };
    if (st < T.toMir[0]) return swap(MIR, Rm, T.toR);
    return { M: lerp4(Rm, MIR, f(st, T.toMir[0], T.toMir[1])), vis: 1 };
  }
  function bump(st, a, d) { var u = K.seg(st, a, a + d); return u > 0 && u < 1 ? Math.sin(Math.PI * u) : 0; }

  E.scenes[2] = {
    title: '规矩二：行列式等于 1',
    dur: 59.3,
    cues: [
      { at: 0, s: '照镜子：长度、角度都不变，规矩一拦不住它' },
      { at: 7.85, s: '可小旗反了过来：怎么转都转不出这个样子' },
      { at: 12.79, s: '**行列式**：小方块的面积变成几倍；翻了面带负号' },
      { at: 24.31, s: '2×2 的行列式 = 左上 × 右下 − 右上 × 左下' },
      { at: 31.25, s: '旋转：$0.8\\times0.8-(-0.6)\\times0.6=1$，面积不变、没翻面' },
      { at: 39.48, s: '镜子是 −1，旋转是 1　所以**规矩二：行列式 = 1**' },
      { at: 47.88, s: '两条都守的 3×3 矩阵，合在一起叫**特殊正交群**' }
    ],
    build: function (K) {
      var s = K.sceneSvg('照镜子左右一翻，长度和角度都不变，第一条规矩拦不住，但小旗反了过来，怎么转都转不出；行列式是两根小箭头围成的小方块面积变成几倍，翻面带负号；二乘二矩阵的行列式是左上乘右下减去右上乘左下；旋转矩阵的行列式是 1，镜子是 −1；所以第二条规矩是行列式等于 1；三维里满足两条规矩的 3×3 矩阵合在一起叫特殊正交群 SO(3)');

      /* ══ 二维部分 ══ */
      var flat = E.el('g', {}, s);
      var pl = E.plane(flat, { ox: OX, oy: OY, u: U, x0: -3, x1: 3, y0: -3, y1: 3 });
      var P = pl.P;
      E.op(pl.warp.g, 0.6);
      var mirror = E.el('g', {}, flat);
      K.paint(E.el('rect', { x: OX - 4, y: OY - 3 * U, width: 8, height: 6 * U, rx: 3 }, mirror), E.tint(E.ACC, 45, 'var(--demo-surface)'), E.ACC);
      E.halo(E.txt(mirror, OX + 12, OY - 3 * U + 22, '镜子', 22, E.ACC, 'start', 900));
      // 方块（正面亮黄、背面红）+ 方块里的小弯箭头 + 面积
      var sq = E.el('path', { 'stroke-width': 2.5 }, flat);
      var turnArc = E.el('path', { fill: 'none', 'stroke-width': 3, 'stroke-linecap': 'round' }, flat), turnHead = E.el('path', {}, flat);
      var areaLab = E.halo(E.txt(flat, 0, 0, '', 20, E.INK, 'middle', 900));
      var ghost = FLAG.map(function () {
        return K.paint(E.el('path', { fill: 'none', 'stroke-width': 5, 'stroke-dasharray': '7 6', 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, flat), null, E.HI);
      });
      var flag = FLAG.map(function () {
        return K.paint(E.el('path', { fill: 'none', 'stroke-width': 7, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, flat), null, E.HI);
      });
      var e1 = E.vec(flat, E.GOOD, 5, { head: 16 }), e2 = E.vec(flat, E.WARN, 5, { head: 16 });

      /* 右栏 ①（句 1、2）：镜子 */
      var r1 = E.el('g', {}, flat);
      var mirTag = E.tag(r1, 472, 14, 300, 42, '镜子：左右一翻', E.ACC, { size: 22, ink: E.INK, weight: 900, rx: 12 });
      var mMir = E.mat(r1, MX, 116, MIR, { size: 30, dx: 84, dy: 44, pad: 46, cw: 72, colors: [E.GOOD, E.WARN, E.GOOD, E.WARN] });
      var okRows = ['长度不变', '角度不变'].map(function (t, i) {
        var g = E.el('g', {}, r1), y = 214 + 40 * i;
        return { g: g, mark: E.check(g, EX + 10, y - 16, 22, E.GOOD, 4), text: E.txt(g, EX + 44, y + 3, t, 24, E.INK, 'start', 700) };
      });
      var pass = E.txt(r1, EX + 10, 304, '规矩一拦不住它', 24, E.WARN, 'start', 900);
      var noG = E.el('g', {}, r1);
      E.cross(noG, EX + 21, 345, 10, E.BAD, 4.5);
      E.txt(noG, EX + 44, 353, '小旗反了，转不出来', 24, E.BAD, 'start', 900);

      /* 右栏 ②（句 3–6）：行列式 */
      var r2 = E.el('g', {}, flat);
      var detTag = E.tag(r2, 452, 14, 340, 44, '', E.ACC, { size: 22, ink: E.INK, weight: 900, rx: 12, sw: 2.5 });
      var rowA = E.txt(r2, EX, 118, '拉长 2 倍：面积 ×2 → 行列式 2', 22, E.INK, 'start', 700);
      var rowB = E.txt(r2, EX, 166, '镜子：翻了面 → 行列式 −1', 22, E.BAD, 'start', 700);
      // 两条对角线垫在矩阵底下（数字框把线遮住，只在框和框之间露出来）
      var diag = [E.GOOD, E.BAD].map(function (c) { var l = K.paint(E.el('path', { 'stroke-width': 3, 'stroke-linecap': 'round' }, r2), null, c); l.style.strokeOpacity = 0.7; return l; });
      var mAbc = E.mat(r2, MX, 120, ['a', 'b', 'c', 'd'], { size: 32, dx: 90, dy: 50, pad: 48, cw: 70, colors: [E.GOOD, E.WARN, E.GOOD, E.WARN] });
      var detLab = E.txt(r2, EX, 240, '行列式 =', 24, E.INK, 'start', 900);
      var eq = [[EX + 108, 232], [EX, 290], [EX, 348]].map(function (p) { return E.tex(r2, p[0], p[1], '', 28, E.INK, 'start', { w: 330, h: 48 }); });
      var marks = [E.check(r2, 0, 0, 22, E.GOOD, 4), E.cross(r2, 0, 0, 10, E.BAD, 4.5)];   // 打在第二、三行左边

      /* ══ 三维：小坐标架 + SO(3) ══ */
      var solid = E.el('g', {}, s);
      var gz = E.gizmo(solid, 190, 250, 118, E.view3(128, 24));
      var rows3 = [{ s: '正交：每列长度 1、两两垂直', c: E.GOOD }, { s: '特殊：行列式 = 1', c: E.ACC }].map(function (d, i) {
        var g = E.el('g', {}, solid), y = 52 + 50 * i;
        var tg = E.tag(g, 436, y - 22, 356, 44, d.s, d.c, { size: 22, ink: E.INK, weight: 900, rx: 12 });
        return { g: g, tg: tg, y: y };
      });
      var so = E.el('g', {}, solid);
      var LET = [{ s: 'S', x: 552, c: E.ACC, lab: '特殊' }, { s: 'O', x: 616, c: E.GOOD, lab: '正交' }, { s: '(3)', x: 700, c: E.INK, lab: '三维' }].map(function (d) {
        var big = E.txt(so, d.x, 236, d.s, 64, d.c, 'middle', 900);
        var lab = E.txt(so, d.x, 280, d.lab, 22, d.c, 'middle', 900);
        return { big: big, lab: lab, d: d };
      });
      var group = E.txt(so, 620, 346, '特殊正交群', 32, E.VIOLET, 'middle', 900);
      var groupSub = E.txt(so, 620, 384, '所有三维旋转矩阵', 22, E.INK2, 'middle', 700);

      function at(m, p) { var q = E.apply(m, p); return P(q[0], q[1]); }
      function poly(m, pts) { return pts.map(function (p, k) { var q = at(m, p); return (k ? 'L ' : 'M ') + q[0].toFixed(1) + ' ' + q[1].toFixed(1); }).join(' '); }
      var ARC = [];
      // 方块里的小弯箭头：逆时针从 −60° 转到 160°，半径 0.21；箭头尖停在方块左上部，整个三角离四条边（两根小箭头压在边上）都留出空，不被挡住
      for (var k = 0; k <= 24; k++) { var aa = (-60 + 220 * k / 24) * D; ARC.push([0.5 + 0.21 * Math.cos(aa), 0.5 + 0.21 * Math.sin(aa)]); }

      return {
        el: s,
        draw: function (st, t) {
          var f = E.fade, MV = matAt(st), M = MV.M, vis = MV.vis;
          var two = f(st, T.to3, 0.5);
          E.op(flat, 1 - two);
          pl.warp.set(M);
          // 翻面途中面积因子（行列式）接近 0，格线全挤到一条线上：按 |行列式| 把紫色网格淡掉，不让它闪成一根实心的条
          E.op(pl.warp.g, (0.6 * vis * E.clamp(Math.abs(M[0] * M[3] - M[1] * M[2]) / 0.4, 0, 1)).toFixed(3));
          var o = P(0, 0), q1 = at(M, [1, 0]), q2 = at(M, [0, 1]);
          e1.set(o[0], o[1], q1[0], q1[1]); e2.set(o[0], o[1], q2[0], q2[1]);
          E.op(e1.g, vis); E.op(e2.g, vis);
          var ba = bump(st, T.arrows, 0.7);
          e1.line.style.strokeWidth = e2.line.style.strokeWidth = 5 + 3 * ba;

          /* 镜子、小旗 */
          E.op(mirror, f(st, T.mirror, 0.4) * (1 - f(st, T.reset[0], 0.4)));
          flag.forEach(function (p, i) { p.setAttribute('d', poly(M, FLAG[i])); E.op(p, f(st, 0.3, 0.4) * (1 - f(st, T.reset[0], 0.4))); });
          var fb = bump(st, T.flag, 0.7);
          flag.forEach(function (p) { p.style.strokeWidth = 7 + 3 * fb; E.glow(p, E.HI, fb > 0.05); });
          var gA = 2 * Math.PI * f(st, T.spin[0], T.spin[1] - T.spin[0]);
          ghost.forEach(function (p, i) { p.setAttribute('d', poly(E.rot(gA), FLAG[i])); E.op(p, f(st, T.spin[0] - 0.3, 0.3) * (1 - f(st, T.reset[0], 0.4)) * 0.85); });

          /* 方块：det > 0 正面亮黄，det < 0 背面红；小弯箭头跟着变换（镜子里变成顺时针）；面积 */
          var det = M[0] * M[3] - M[1] * M[2], front = det >= 0;
          sq.setAttribute('d', poly(M, [[0, 0], [1, 0], [1, 1], [0, 1]]) + ' Z');
          K.paint(sq, E.tint(front ? E.HI : E.BAD, 30, 'var(--demo-surface)'), front ? E.HI : E.BAD);
          var sqOn = f(st, T.square, 0.4) * vis;
          E.op(sq, sqOn);
          turnArc.setAttribute('d', poly(M, ARC));
          var a = at(M, ARC[ARC.length - 2]), b = at(M, ARC[ARC.length - 1]), an = Math.atan2(b[1] - a[1], b[0] - a[0]), L = 10, W = 6;
          turnHead.setAttribute('d', 'M ' + (b[0] + 3 * Math.cos(an)).toFixed(1) + ' ' + (b[1] + 3 * Math.sin(an)).toFixed(1) +
            ' L ' + (b[0] - L * Math.cos(an) + W * Math.sin(an)).toFixed(1) + ' ' + (b[1] - L * Math.sin(an) - W * Math.cos(an)).toFixed(1) +
            ' L ' + (b[0] - L * Math.cos(an) - W * Math.sin(an)).toFixed(1) + ' ' + (b[1] - L * Math.sin(an) + W * Math.cos(an)).toFixed(1) + ' Z');
          var tc = front ? E.INK : E.BAD;
          K.paint(turnArc, null, tc); K.paint(turnHead, tc);
          var tb = bump(st, T.noflip, 0.9);
          turnArc.style.strokeWidth = 3 + 2 * tb; E.glow(turnArc, E.HI, tb > 0.05);
          E.op(turnArc, sqOn); E.op(turnHead, sqOn);
          // 面积写在方块对角（两根小箭头之和）再往外一点，不压两根小箭头
          var far = E.apply(M, [1, 1]), fl = Math.hypot(far[0], far[1]) || 1, fp = P(far[0] + 0.42 * far[0] / fl, far[1] + 0.42 * far[1] / fl);
          areaLab.textContent = '面积 ' + E.num(Math.abs(det), 1);
          areaLab.setAttribute('x', fp[0].toFixed(1)); areaLab.setAttribute('y', (fp[1] + 7).toFixed(1));
          var areaOn = st < T.toR[0] ? f(st, T.square + 0.4, 0.3) * (Math.abs(det) > 0.5 ? 1 : 0) : f(st, T.area1, 0.3) * (Math.abs(det) > 0.5 ? 1 : 0);
          E.op(areaLab, areaOn * vis);
          areaLab.style.fill = front ? E.INK : E.BAD;
          var ab = bump(st, T.area, 0.8);
          areaLab.style.fontSize = (20 + 4 * ab) + 'px';

          /* 右栏 ①：镜子 */
          var out1 = 1 - f(st, T.reset[0], 0.4);
          E.op(r1, out1);
          E.pop(mirTag.g, 622, 35, f(st, T.mirror, 0.35), 0.85);
          E.op(mMir.g, f(st, T.flip[0], 0.4));
          okRows.forEach(function (r, i) { var p = f(st, i ? T.ang : T.len, 0.35); E.op(r.g, Math.min(1, p * 2)); E.drawOn(r.mark, p); });
          E.op(pass, f(st, T.rule1, 0.35));
          E.op(noG, f(st, T.no, 0.35));

          /* 右栏 ②：行列式 */
          E.op(r2, f(st, T.det - 0.1, 0.3));
          detTag.text.textContent = st < T.rule2 ? '行列式' : '规矩二：行列式 = 1';
          var rb = bump(st, T.rule2, 0.8);
          detTag.rect.style.strokeWidth = 2.5 + 2 * rb; E.glow(detTag.rect, E.ACC, rb > 0.05);
          E.pop(detTag.g, 622, 36, f(st, T.det, 0.35), 0.85);
          var rowsOut = 1 - f(st, T.toR[0] - 0.2, 0.3);
          E.op(rowA, f(st, T.row1, 0.35) * rowsOut); E.op(rowB, f(st, T.row2, 0.35) * rowsOut);
          // 字母矩阵 → 旋转矩阵的数 → 镜子的数
          var cells = st < T.nums ? ['a', 'b', 'c', 'd'] : st < T.toMir[0] ? Rm : MIR;
          mAbc.set(cells);
          E.op(mAbc.g, f(st, T.abcd, 0.35));
          var hiA = f(st, T.a, 0.25), hiD = f(st, T.d, 0.25), hiB = f(st, T.b, 0.25), hiC = f(st, T.c, 0.25);
          mAbc.cellBox(0, E.GOOD, hiA); mAbc.cellBox(3, E.GOOD, hiD); mAbc.cellBox(1, E.BAD, hiB); mAbc.cellBox(2, E.BAD, hiC);
          [[0, 3, hiD], [1, 2, hiC]].forEach(function (d, i) {   // 只连两个框之间的空隙（框的角到角），不穿过数字
            var p = mAbc.pos(d[0]), q = mAbc.pos(d[1]), u = d[2], sx = q[0] > p[0] ? 1 : -1;
            p = [p[0] + sx * 35, p[1] + 17.6]; q = [q[0] - sx * 35, q[1] - 17.6];
            diag[i].setAttribute('d', 'M ' + p[0] + ' ' + p[1] + ' L ' + (p[0] + (q[0] - p[0]) * u).toFixed(1) + ' ' + (p[1] + (q[1] - p[1]) * u).toFixed(1));
            E.op(diag[i], u > 0 ? 0.85 : 0);
          });
          var g = H.good, w = H.warn, bad = H.bad;
          function c(x, hex) { return '\\textcolor{' + hex + '}{' + x + '}'; }
          // 第一行：ad − bc
          var L1 = c('a', g) + c('d', w);
          if (st >= T.minus) L1 += '-';
          if (st >= T.c) L1 += c('b', w) + c('c', g);
          eq[0].setTex(st < T.a ? '\\,' : st < T.d ? c('a', g) : L1);
          E.op(eq[0], f(st, T.a, 0.3)); E.op(detLab, f(st, T.a, 0.3));
          // 第二行：旋转矩阵
          var L2 = '';
          if (st >= T.r[0]) L2 = c('0.8', g) + '\\times' + c('0.8', w);
          if (st >= T.r[2]) L2 += '-';
          if (st >= T.r[3]) L2 += '(' + c('-0.6', w) + ')\\times' + c('0.6', g);
          if (st >= T.r[5]) L2 += '=1';
          eq[1].setTex(L2 || '\\,');
          E.op(eq[1], f(st, T.r[0], 0.3));
          // 第三行：镜子
          var L3 = '';
          if (st >= T.m[0]) L3 = '(' + c('-1', g) + ')\\times' + c('1', w);
          if (st >= T.m[1]) L3 += '-' + c('0', w) + '\\times' + c('0', g);
          if (st >= T.m[3]) L3 += '=' + c('-1', bad);
          eq[2].setTex(L3 || '\\,');
          E.op(eq[2], f(st, T.m[0], 0.3));
          marks[0].setAttribute('transform', 'translate(' + (EX - 36) + ' 272)'); E.drawOn(marks[0], f(st, T.marks, 0.4));
          marks[1].setAttribute('transform', 'translate(' + (EX - 22) + ' 340)'); E.drawOn(marks[1], f(st, T.marks + 0.25, 0.4));

          /* 三维 */
          E.op(solid, f(st, T.to3 + 0.2, 0.5));
          gz.set(E.rot3([0.3, 0.25, 1], (25 + 12 * Math.sin(2 * Math.PI * t / 9)) * D));
          rows3.forEach(function (r, i) {
            E.pop(r.g, 614, r.y, f(st, T.rows3[i], 0.35), 0.85);
            var hb = bump(st, i ? T.special : T.ortho, 1.0);
            r.tg.rect.style.strokeWidth = 2 + 2 * hb; E.glow(r.tg.rect, i ? E.ACC : E.GOOD, hb > 0.05);
          });
          E.op(so, f(st, T.so3 - 0.1, 0.4));
          LET.forEach(function (l, i) {
            var hb = i === 0 ? bump(st, T.special, 1.0) : i === 1 ? bump(st, T.ortho, 1.0) : 0;
            E.scaleAt(l.big, l.d.x, 214, 1 + 0.15 * hb);
            E.op(l.lab, f(st, i === 2 ? T.named : i === 1 ? T.ortho : T.special, 0.3));
          });
          E.op(group, f(st, T.so3, 0.4)); E.op(groupSub, f(st, T.named + 0.3, 0.4));
        }
      };
    }
  };
})();
