// 004 第 4 幕：两个变换接着做，顺序不能换。记号同 003 第 5 幕：R 转 90°，S = [[2, 0], [0, 1]] 横着拉长到 2 倍；手的那个点 p = (3, 2)。
// 左边交互窗口，右边上方放矩阵 / 写法对照 / 结论，下方网格（x −4…6、y −1…6，原点在右边，p 一开始就标在 (3, 2)）。
// ① 「横着拉长到两倍」打 S（2 亮黄）；右上出 S 的矩阵，网格横着拉一下再回来；「叫它 S」框住 S。
// ② 「S 艾特 R 艾特 p」逐字打出；「离 p 最近的先做」下一排标 ①（R 下面）、②（S 下面）和一支从 p 往左的箭头、「离 p 最近的先做」；
//    「和纸上的写法一样」右上换成「纸上 S R p」对「代码 S @ R @ p」。
// ③ 「回车」出 array([-4,  3])（蓝）；网格上蓝色路线：(3, 2) 绕原点转到 (−2, 3)，再横着拉到 (−4, 3)，标 (−4, 3)。
// ④ 「换个位置」打出 R @ S @ p，上下两句里的 S 都框亮黄、R 都框紫，一眼看出两个字母换了位置；「得到」出 array([-2,  6])（亮黄，同 003 第 5 幕的第二条路线）；
//    亮黄路线：(3, 2) 横着拉到 (6, 2)，再转到 (−2, 6)，标 (−2, 6)；「落点完全不同」两个终点各圈一下，右上出 (−4, 3) ≠ (−2, 6)。
// ⑤ 「顺序」右上出「顺序要自己写对」。
// 时刻取自配音的逐词时间（分镜时间 ≈ 念到的时刻），见 ../ep004.py。
(function () {
  var E = window.EP004, K = E.K, H = E.HEX;
  var RX = 466, RW = 318, RC = RX + RW / 2;
  var S_LINE = 3.05, STRETCH = [3.97, 4.72], NAME = 5.57, KEYS = [8.85, 8.9, 9.04, 9.1, 9.42, 9.45, 9.57, 9.6, 9.87];
  var NEAR = 10.58, FIRST = 11.46, NEAR_OUT = 14.3, PAPER = 12.34, SAME = 13.25, PAPER_OUT = 14.5;
  var OUT_A = 14.76, PATH_A = [15.0, 16.4], LBL_A = 15.75;
  var SWAP = [18.59, 19.4], OUT_B = 20.67, PATH_B = [20.8, 22.0], LBL_B = 21.3, DIFF = 22.04, NEQ = 22.6, ORDER = 25.9;
  var OX = 582, OY = 384, U = 30;                           // 网格原点、一格多少 px

  E.scenes[3] = {
    title: '两个变换接着做：顺序不能换',
    dur: 28.45,
    cues: [
      { at: 0, s: '横着拉长到 2 倍的矩阵：`S`' },
      { at: 7.13, s: '`S @ R @ p`：离 p 最近的**先做**，和纸上一样' },
      { at: 14.67, s: '先转再拉：(−4, 3)' },
      { at: 17.62, s: '`R @ S @ p`：先拉再转，落到 (−2, 6)' },
      { at: 24.32, s: '电脑算得再快，**顺序**也要自己写对' }
    ],
    build: function (K) {
      var s = K.sceneSvg('再写一个横着拉长到 2 倍的矩阵 S；先转再拉长写成 S @ R @ p，离 p 最近的先做，结果 (−4, 3)；换成 R @ S @ p 就是先拉长再转，落到 (−2, 6)，落点完全不同，顺序要自己写对');

      /* ── 左：交互窗口。「>>> S @ R @ p」里 S、R 在第 4、8 列 ── */
      var py = E.code(s, 10, 14, 440, 392, { title: 'Python', size: 25 });
      py.line('>>> {n.S:S} = np.array([[{h:2}, 0],', S_LINE, { dur: 1.5 });
      py.line('...               [0, 1]])', S_LINE + 1.55, { dur: 0.6 });
      var L_A = py.line('>>> S @ R @ {a:p}', KEYS[0], { keys: KEYS });
      py.line('array([{a:-4},  {a:3}])', OUT_A, { out: true });
      var L_B = py.line('>>> R @ S @ {h:p}', SWAP[0], { dur: 0.5 });
      py.line('array([{h:-2},  {h:6}])', OUT_B, { out: true });
      // 「离 p 最近的先做」：标在 S @ R @ p 下面一排（输出出来之前收掉）
      var near = E.el('g', {}, s);
      var nArrow = E.arrow(near, [[0, 0], [1, 0]], E.HI, 3, { head: 11 });
      var n1 = E.txt(near, 0, 0, '①', 22, E.HI, 'middle', 900), n2 = E.txt(near, 0, 0, '②', 22, E.HI, 'middle', 900);
      var nT = E.txt(near, 0, 0, '离 p 最近的先做', 22, E.HI, 'start', 900);

      /* ── 右上：S 的矩阵 / 纸上 vs 代码 / 结论 ── */
      var sG = E.el('g', {}, s);
      var mS = E.mat(sG, RC + 34, 104, [2, 0, 0, 1], { size: 34, colors: [H.hi, E.INK, E.INK, E.INK] });
      var sLbl = E.txt(sG, mS.box[0] - 12, 116, 'S =', 34, E.VIOLET, 'end', 900); sLbl.style.fontFamily = 'KaTeX_Main, serif';
      var cmp = E.el('g', {}, s);
      E.txt(cmp, RX + 20, 78, '纸上', 22, E.INK2, 'start', 700);
      E.tex(cmp, RX + 200, 70, 'S\\,R\\,p', 34, E.INK, 'middle', { w: 160, h: 60 });
      E.txt(cmp, RX + 20, 138, '代码', 22, E.INK2, 'start', 700);
      var cc = E.txt(cmp, RX + 200, 140, 'S @ R @ p', 28, E.INK, 'middle', 700); cc.style.fontFamily = E.MONO;
      var cmpOk = E.el('g', {}, cmp);
      E.check(cmpOk, RX + 286, 92, 24, E.GOOD, 4);
      var neq = E.tex(s, RC, 104, '\\textcolor{' + H.acc + '}{(-4,\\,3)}\\;\\textcolor{' + H.bad + '}{\\neq}\\;\\textcolor{' + H.hi + '}{(-2,\\,6)}', 34, E.INK, 'middle', { w: 310, h: 70 });
      var order = E.el('g', {}, s);
      K.paint(E.el('rect', { x: RX + 24, y: 60, width: RW - 48, height: 64, rx: 32, 'stroke-width': 3 }, order), E.tint(E.HI, 14, 'var(--demo-surface)'), E.HI);
      E.txt(order, RC, 102, '顺序要自己写对', 28, E.HI, 'middle', 900);

      /* ── 右下：网格与两条路线 ── */
      var grid = E.el('g', {}, s);
      var pl = E.plane(grid, { ox: OX, oy: OY, u: U, x0: -4, x1: 6, y0: -1, y1: 6 });
      var P = pl.P;
      function arc(a, b) { var r = Math.hypot(a[0], a[1]) * U, q = P(b[0], b[1]); return ' A ' + r.toFixed(1) + ' ' + r.toFixed(1) + ' 0 0 0 ' + q[0].toFixed(1) + ' ' + q[1].toFixed(1); }
      function M(a) { var q = P(a[0], a[1]); return 'M ' + q[0].toFixed(1) + ' ' + q[1].toFixed(1); }
      function L(a) { var q = P(a[0], a[1]); return ' L ' + q[0].toFixed(1) + ' ' + q[1].toFixed(1); }
      var pathA = K.paint(E.el('path', { d: M([3, 2]) + arc([3, 2], [-2, 3]) + L([-4, 3]), fill: 'none', 'stroke-width': 4, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, grid), null, E.ACC);
      var pathB = K.paint(E.el('path', { d: M([3, 2]) + L([6, 2]) + arc([6, 2], [-2, 6]), fill: 'none', 'stroke-width': 4, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, grid), null, E.HI);
      var endA = K.paint(E.el('circle', { cx: P(-4, 3)[0], cy: P(-4, 3)[1], r: 7 }, grid), E.ACC);
      var endB = K.paint(E.el('circle', { cx: P(-2, 6)[0], cy: P(-2, 6)[1], r: 7 }, grid), E.HI);
      var ringA = K.paint(E.el('circle', { cx: P(-4, 3)[0], cy: P(-4, 3)[1], r: 16, fill: 'none', 'stroke-width': 3 }, grid), null, E.ACC);
      var ringB = K.paint(E.el('circle', { cx: P(-2, 6)[0], cy: P(-2, 6)[1], r: 16, fill: 'none', 'stroke-width': 3 }, grid), null, E.HI);
      var lblA = E.txt(grid, P(-4, 3)[0] + 4, P(-4, 3)[1] + 34, '(−4, 3)', 22, E.ACC, 'start', 900);
      var lblB = E.txt(grid, P(-2, 6)[0], P(-2, 6)[1] + 44, '(−2, 6)', 22, E.HI, 'middle', 900);
      E.halo(lblA); E.halo(lblB);
      var pDot = K.paint(E.el('circle', { cx: P(3, 2)[0], cy: P(3, 2)[1], r: 7 }, grid), E.ACC);
      var pLbl = E.txt(grid, P(3, 2)[0] + 4, P(3, 2)[1] + 30, 'p', 24, E.ACC, 'middle', 700);
      pLbl.style.fontFamily = 'KaTeX_Main, serif'; pLbl.style.fontStyle = 'italic';

      return {
        el: s,
        draw: function (st, t) {
          var fade = E.fade, win = E.win;
          py.draw(st, t);
          py.box('S', E.HI, win(st, NAME, NAME + 1.0, 0.25));

          /* 「离 p 最近的先做」 */
          var yA = py.y(L_A) + py.lh, nv = win(st, NEAR, NEAR_OUT, 0.3);   // 标在下一排：箭头、①②、一句话
          var ax0 = py.x(12) + py.cw / 2, ax1 = py.x(4) + py.cw / 2 + 12, ay = yA - 16;
          nArrow.line.setAttribute('d', 'M ' + ax0 + ' ' + ay + ' L ' + (ax0 + (ax1 - ax0) * fade(st, NEAR, 1.0)).toFixed(1) + ' ' + ay);
          var hx = ax0 + (ax1 - ax0) * fade(st, NEAR, 1.0);
          nArrow.head.setAttribute('d', 'M ' + (hx - 11).toFixed(1) + ' ' + ay + ' L ' + hx.toFixed(1) + ' ' + (ay - 6.6) + ' L ' + hx.toFixed(1) + ' ' + (ay + 6.6) + ' Z');
          n1.setAttribute('x', py.x(8) + py.cw / 2); n1.setAttribute('y', yA + 10);
          n2.setAttribute('x', py.x(4) + py.cw / 2); n2.setAttribute('y', yA + 10);
          nT.setAttribute('x', py.x(14)); nT.setAttribute('y', yA - 6);
          E.op(near, nv);
          E.op(n1, fade(st, FIRST, 0.3)); E.op(n2, fade(st, FIRST + 0.35, 0.3)); E.op(nT, fade(st, NEAR + 0.2, 0.3));

          /* 换位：上下两句里的 S 框亮黄、R 框紫（S @ R @ p 里 S、R 在第 4、8 列，R @ S @ p 里反过来） */
          var sw = win(st, SWAP[0] + 0.3, OUT_B + 0.6, 0.25);
          py.boxCol(L_A, 4, 1, E.HI, sw); py.boxCol(L_A, 8, 1, E.VIOLET, sw);
          py.boxCol(L_B, 8, 1, E.HI, sw); py.boxCol(L_B, 4, 1, E.VIOLET, sw);

          /* 右上 */
          E.pop(sG, RC, 104, fade(st, STRETCH[0] - 0.5, 0.35) * (1 - fade(st, PAPER - 0.4, 0.3)), 0.85);
          E.op(cmp, win(st, PAPER, PAPER_OUT + 0.6, 0.3));
          E.pop(cmpOk, RX + 298, 104, fade(st, SAME, 0.3), 0.5);
          E.pop(order, RC, 92, fade(st, ORDER, 0.35), 0.85);

          /* 网格：「拉长」时横着拉一下再回来 */
          E.op(grid, fade(st, 0.1, 0.5));
          var sw = win(st, STRETCH[0], STRETCH[1] + 0.9, 0.5);
          pl.warp.set(E.lerpM([2, 0, 0, 1], sw));
          E.op(pl.warp.g, 0.55 * sw);
          E.drawOn(pathA, fade(st, PATH_A[0], PATH_A[1] - PATH_A[0]));
          E.drawOn(pathB, fade(st, PATH_B[0], PATH_B[1] - PATH_B[0]));
          E.op(endA, fade(st, PATH_A[1] - 0.1, 0.2)); E.op(endB, fade(st, PATH_B[1] - 0.1, 0.2));
          E.op(lblA, fade(st, LBL_A, 0.3)); E.op(lblB, fade(st, LBL_B, 0.3));
          var ring = K.seg(st, DIFF, DIFF + 0.8);
          [ringA, ringB].forEach(function (r) { r.setAttribute('r', (12 + 10 * Math.sin(Math.PI * Math.min(ring, 0.5))).toFixed(1)); E.op(r, ring > 0 ? 1 : 0); });
          E.op(neq, win(st, NEQ, ORDER - 0.4, 0.3));
          E.op(pDot, 1); E.op(pLbl, 1);
        }
      };
    }
  };
})();
