// 004 第 2 幕：矩阵一个横排一个横排地写。左边交互窗口，右边说明区。
// ① 「手的位置」右边出「手的位置」+ [3; 2]（蓝，同 003）；「写成一个数组」打 p = np.array([3, 2])，「叫它 p」框住 p。
// ② 「转九十度的矩阵」右边换成 R = [[0, −1], [1, 0]]（第一列绿、第二列琥珀，同 003）；「两列」两列各框一圈，「零、一」「负一、零」依次亮。
// ③ 「一个横排」打 R 的第一排、矩阵上面一排垫底色；「一个横排」打第二排（续行「...」，0 对齐 −1 的 1）、下面一排垫底色；
//    「最外层是方括号」框住代码最外层的 [ 和 ]、矩阵括号亮黄；「上下两个横排」框住代码里的两个小列表。
// ④ 「上面一排」「下面一排」依次亮（代码与矩阵一起）；「绿色的第一列」矩阵第一列框绿，「拆到」矩阵里上面的 0、下面的 1 先后放大闪一下，
//    代码里第一排、第二排的第一个数同时框绿（不让数字飞过去：飞的路上会压过上面几句代码）。
// ⑤ 「R、方括号、冒号、逗号、零」逐字打 R[:, 0]，右下方放大写一遍：「冒号是每个横排都要」冒号下标字、矩阵两排都亮，
//    「零是第一列」0 下标字、矩阵第一列框绿；「编号从零开始」矩阵两列头上标 0、1。
// ⑥ 「结果」出 array([0, 1])（绿）；右下换成小网格：「往右的那根小箭头」灰色 (1, 0)，「转完」转到 (0, 1) 变绿，标 (0, 1)。
// 时刻取自配音的逐词时间（分镜时间 ≈ 念到的时刻），见 ../ep004.py。
(function () {
  var E = window.EP004, K = E.K, H = E.HEX;
  var RX = 466, RW = 318, RC = RX + RW / 2;                 // 右边说明区（左边代码窗口宽 440）
  var MX = 668, MY = 176;                                   // 右边矩阵中心
  var HAND = 3.37, P_LINE = 5.21, P_NAME = 6.73, P_OUT = 8.3;
  var MAT = 9.04, TWO = 10.39, COL1 = [11.11, 11.71], COL2 = [11.95, 12.72];
  var ROW_T = [14.99, 16.72], ROW_B = [16.91, 17.62], OUTER = [18.4, 19.85], INNER = [19.95, 22.15];
  var UP = [22.82, 24.59], DOWN = [24.69, 26.44], GREEN = 26.76, FLY = [[27.87, 28.57], [28.37, 29.07]], GREEN_OUT = 30.54;
  var KEYS = [33.01, 33.46, 34.24, 34.86, 35.24, 35.53, 35.79], ZOOM = 33.01, COLON = 36.29, ZERO = 38.12, IDX = 38.98, ZOOM_OUT = 41.04;
  var RES = 41.34, ARROW = 43.38, TURN = [45.1, 46.14];

  E.scenes[1] = {
    title: '矩阵：一个横排一个横排地写',
    dur: 47.14,
    cues: [
      { at: 0, s: '手的位置写成数组：`p = np.array([3, 2])`' },
      { at: 7.92, s: '转 90° 的**矩阵** R：两列是 (0, 1)、(−1, 0)' },
      { at: 13.8, s: 'NumPy 按**横排**写：外层方括号里装两个横排' },
      { at: 22.73, s: '第一列拆到了每一排的第一个位置' },
      { at: 30.94, s: '`R[:, 0]`：冒号 = 每个横排都要，0 = 第一列' },
      { at: 41.24, s: '结果 `[0, 1]`：往右的小箭头转完以后的位置' }
    ],
    build: function (K) {
      var s = K.sceneSvg('把手的位置写成数组 p = np.array([3, 2])；转 90 度的矩阵 R 两列是 (0, 1) 和 (−1, 0)，写进 NumPy 时一个横排一个横排地写：最外层是方括号，里面装着上下两个横排，第一列拆到每一排的第一个位置；R[:, 0] 取出第一列，结果 [0, 1] 是往右的小箭头转完以后的位置');

      /* ── 左：交互窗口。R 的两排（列号见下）：第一排「>>> R = np.array([[0, -1],」最外层 [ 在第 17 列，小列表 [0, -1] 占 18–24 列；
             第二排「...」后补空格让 [ 也在第 18 列，1 对齐 0、0 对齐 −1 的 1，最外层 ] 在第 25 列 ── */
      var py = E.code(s, 10, 14, 440, 392, { title: 'Python', size: 25 });
      py.line('>>> {n.p:p} = np.array([{a:3}, {a:2}])', P_LINE, { dur: 1.1 });
      var L_R1 = py.line('>>> R = np.array([[{g.r0:0}, {w:-1}],', ROW_T[0], { dur: ROW_T[1] - ROW_T[0] });
      var L_R2 = py.line('...               [{g.r1:1},  {w:0}]])', ROW_B[0], { dur: ROW_B[1] - ROW_B[0] });
      py.line('>>> R[:, {g:0}]', KEYS[0], { keys: KEYS });
      py.line('array([{g:0}, {g:1}])', RES, { out: true });

      /* ── 右 ①：手的位置 ── */
      var pCard = E.el('g', {}, s);
      E.txt(pCard, RC, 70, '手的位置', 26, E.ACC, 'middle', 900);
      E.tex(pCard, RC, 190, 'p=' + E.col([3, 2], H.acc), 48, E.INK, 'middle', { w: 260, h: 170 });

      /* ── 右 ②–⑤：矩阵 ── */
      var matG = E.el('g', {}, s);
      var M = E.mat(matG, MX, MY, [0, -1, 1, 0], { size: 42, colors: [H.good, H.warn, H.good, H.warn] });
      var rLbl = E.txt(matG, M.box[0] - 14, MY + 14, 'R =', 40, E.VIOLET, 'end', 900);
      rLbl.style.fontFamily = 'KaTeX_Main, serif';
      var idxG = E.el('g', {}, matG);
      [0, 1].forEach(function (c) {
        var t = E.txt(idxG, MX + (c ? 1 : -1) * M.dx / 2, M.box[1] - 14, String(c), 24, E.HI, 'middle', 700);
        t.style.fontFamily = E.MONO;
      });

      /* ── 右 ⑤：放大的 R[:, 0] 与两条注解 ── */
      var zoomG = E.el('g', {}, s);
      var ZX = RX + 50, ZY = 330, ZS = 40, ZCW = ZS * 0.6;  // 「R[:, 0]」左缘、基线、字号、字宽
      var zt = E.txt(zoomG, ZX, ZY, '', ZS, E.INK, 'start', 700);
      zt.style.fontFamily = E.MONO; zt.setAttribute('xml:space', 'preserve'); zt.style.whiteSpace = 'pre';
      zt.innerHTML = '';
      ['R[', ':', ', ', '0', ']'].forEach(function (part, k) {
        var sp = E.el('tspan', {}, zt); sp.textContent = part; sp.style.fill = k === 1 ? E.HI : k === 3 ? E.GOOD : E.INK;
      });
      function colX(c) { return ZX + ZCW * c; }
      var cLbl = E.txt(zoomG, colX(2.5), ZY + 50, '每个横排都要', 22, E.HI, 'middle', 900);
      var zLbl = E.txt(zoomG, colX(5.5) + 30, ZY + 50, '第一列', 22, E.GOOD, 'start', 900);
      var cTick = E.arrow(zoomG, [[colX(2.5), ZY + 28], [colX(2.5), ZY + 10]], E.HI, 2.5, { head: 9 });
      var zTick = E.arrow(zoomG, [[colX(5.5) + 26, ZY + 40], [colX(5.5) + 4, ZY + 14]], E.GOOD, 2.5, { head: 9 });

      /* ── 右 ⑥：小网格，往右的一格转到 (0, 1) ── */
      var miniG = E.el('g', {}, s);
      var mp = E.plane(miniG, { ox: RC, oy: 398, u: 112, x0: -1, x1: 1, y0: 0, y1: 1, labels: false });
      var trace = E.el('path', { d: 'M ' + (RC + 112) + ' 398 A 112 112 0 0 0 ' + RC + ' 286', fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '6 6' }, miniG);
      K.paint(trace, null, E.HI);
      var unit = E.vec(miniG, E.INK2, 6, { head: 18 });
      var rLbl0 = E.txt(miniG, RC + 56, 370, '往右一格', 22, E.INK, 'middle', 700);
      var uLbl = E.txt(miniG, 0, 0, '(0, 1)', 24, E.GOOD, 'start', 900);

      return {
        el: s,
        draw: function (st, t) {
          var fade = E.fade, win = E.win;
          py.draw(st, t);
          py.box('p', E.HI, win(st, P_NAME, P_NAME + 1.0, 0.25));

          /* ① 手的位置卡：「手的位置」出，矩阵出来前收掉 */
          E.pop(pCard, RC, 150, fade(st, HAND, 0.35) * (1 - fade(st, P_OUT, 0.4)), 0.85);

          /* ② 矩阵 */
          E.pop(matG, MX, MY, fade(st, MAT, 0.35), 0.85);
          var c1 = Math.max(win(st, TWO, COL2[1], 0.25) * (st < COL2[0] ? 1 : 0.35), win(st, COL1[0], COL1[1], 0.2),
            win(st, GREEN, GREEN_OUT, 0.3), win(st, ZERO, ZOOM_OUT, 0.3), win(st, RES, 47.14, 0.3));
          var c2 = Math.max(win(st, TWO, COL1[0], 0.25), win(st, COL2[0], COL2[1], 0.2)) * (st >= COL1[0] && st < COL2[0] ? 0.35 : 1);
          M.colBox(0, E.GOOD, c1);
          M.colBox(1, E.WARN, c2);

          /* ③④ 横排：代码与矩阵一起亮 */
          var top = Math.max(win(st, ROW_T[0], ROW_B[0], 0.2), win(st, UP[0], UP[1], 0.2), win(st, COLON, ZERO, 0.25));
          var bot = Math.max(win(st, ROW_B[0], OUTER[0], 0.2), win(st, DOWN[0], DOWN[1], 0.2), win(st, COLON, ZERO, 0.25));
          M.rowBand(0, E.VIOLET, top); M.rowBand(1, E.VIOLET, bot);
          py.hl(L_R1, E.VIOLET, Math.max(win(st, ROW_T[0], ROW_B[0], 0.2), win(st, UP[0], UP[1], 0.2)));
          py.hl(L_R2, E.VIOLET, Math.max(win(st, ROW_B[0] + 0.2, OUTER[0], 0.2), win(st, DOWN[0], DOWN[1], 0.2)));
          var outer = win(st, OUTER[0], OUTER[1], 0.25);
          py.boxCol(L_R1, 17, 1, E.HI, outer); py.boxCol(L_R2, 25, 1, E.HI, outer);
          M.brackets.forEach(function (b) { b.style.stroke = outer > 0.5 ? E.HI : E.INK; });
          var inner = win(st, INNER[0], INNER[1], 0.25);
          py.boxCol(L_R1, 18, 7, E.VIOLET, Math.max(inner, win(st, UP[0], UP[1], 0.2)));
          py.boxCol(L_R2, 18, 7, E.VIOLET, Math.max(inner, win(st, DOWN[0], DOWN[1], 0.2)));

          /* ④ 矩阵里上面的 0、下面的 1 先后放大闪一下，代码里第一排、第二排的第一个数（第 19 列）同时框绿 */
          [[L_R1, 0], [L_R2, 2]].forEach(function (q, k) {
            var f = FLY[k], u = K.seg(st, f[0], f[1]), b = u > 0 && u < 1 ? Math.sin(Math.PI * u) : 0;
            M.cells[q[1]].setAttribute('font-size', (42 * (1 + 0.3 * b)).toFixed(1));
            py.boxCol(q[0], 19, 1, E.GOOD, win(st, f[0] + 0.15, GREEN_OUT, 0.25));
          });

          /* ⑤ R[:, 0]：放大写一遍，冒号、0 各标一句；「编号从零开始」两列头上标 0、1 */
          var zg = win(st, ZOOM, ZOOM_OUT, 0.35);
          E.op(zoomG, zg);
          var shown = KEYS.filter(function (k) { return st >= k; }).length;   // 跟着交互窗口一个字一个字出：R [ : , 空格 0 ]
          var parts = zt.childNodes, upto = [2, 3, 5, 6, 7];                  // 每段在 KEYS 里结束于第几个字
          ['R[', ':', ', ', '0', ']'].forEach(function (part, k) {
            var start = k ? upto[k - 1] : 0, n = Math.max(0, Math.min(part.length, shown - start));
            parts[k].textContent = part.slice(0, n) + ' '.repeat(part.length - n);
          });
          E.op(cLbl, fade(st, COLON + 0.3, 0.3)); E.op(cTick.g, fade(st, COLON, 0.3));
          E.op(zLbl, fade(st, ZERO + 0.3, 0.3)); E.op(zTick.g, fade(st, ZERO, 0.3));
          E.op(idxG, win(st, IDX, ZOOM_OUT, 0.3));

          /* ⑥ 小网格：往右的一格转到 (0, 1) */
          E.op(miniG, fade(st, ZOOM_OUT + 0.2, 0.4));
          var a = Math.PI / 2 * fade(st, TURN[0], TURN[1] - TURN[0]), o = mp.P(0, 0), e = mp.P(Math.cos(a), Math.sin(a));
          unit.set(o[0], o[1], o[0] + (e[0] - o[0]) * fade(st, ARROW, 0.5), o[1] + (e[1] - o[1]) * fade(st, ARROW, 0.5));
          unit.paint(st >= TURN[1] - 0.3 ? E.GOOD : E.INK2);
          E.drawOn(trace, fade(st, TURN[0], TURN[1] - TURN[0])); E.op(trace, 0.7 * fade(st, TURN[0], 0.2));
          E.op(rLbl0, E.win(st, ARROW + 0.3, TURN[0], 0.3));
          uLbl.setAttribute('x', (mp.P(0, 1)[0] + 16).toFixed(1)); uLbl.setAttribute('y', (mp.P(0, 1)[1] + 10).toFixed(1));
          E.op(uLbl, fade(st, TURN[1], 0.3));
        }
      };
    }
  };
})();
