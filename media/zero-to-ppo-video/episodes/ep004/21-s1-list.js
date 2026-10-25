// 004 第 1 幕：列表不是向量，要用 NumPy 数组。左边 Python 交互窗口，右边说明区。
// ① 「列表」a = [2, 1]（2、1 用 003 第 2 幕扫地机第一段的绿）；「两个列表」b = [1, 2]（琥珀）；「相加」a + b → [2, 1, 1, 2]。
//    「二、一」右边上方先出「向量相加：」[2;1] + [1;2] = [3;3]（003 第 2 幕的结果）；下方两组小方块（绿 2 1、琥珀 1 2），「接在一起」琥珀那组滑过来贴上，「四个数」底下标 1–4，「不是向量相加」打红 ×。
// ② 「乘二」2 * a → [2, 1, 2, 1]：右上换成 2 × [2;1] = [4;2]，「重复」绿方块抄一份滑到右边，打红 ×。
// ③ 「NumPy」右边出「NumPy · 专门处理数组的库」；「终端」右边弹出小终端，打 pip install numpy，「装上它」✓；
//    交互窗口翻到新的一页：「导入」打 import numpy as np，「短名字」框住 np、下面标「短名字」。
// ④ 「交给 NumPy」a = np.array([2, 1])、b = np.array([1, 2])，「再加一次」a + b → array([3, 3])（蓝）；右边出网格：
//    绿 (2, 1) 接琥珀 (1, 2)、蓝 (3, 3)，右上 [2;1] + [1;2] = [3;3]；「二加一是三」上面一排亮、竖虚线落到 x = 3，「一加二也是三」下面一排亮、横虚线落到 y = 3。
// ⑤ 「乘二」2 * a → array([4, 2])：右上换成 2 × [2;1] = [4;2]，琥珀、蓝退淡，绿箭头拉到 (4, 2)；「方向不变」过原点的虚线；「两倍长」两段等长的刻度。
// 代码输出都用 NumPy 2.5.3 实跑过（见 ../ep004.py）。时刻取自配音的逐词时间（分镜时间 ≈ 念到的时刻）。
(function () {
  var E = window.EP004, K = E.K, H = E.HEX, DIM = '#5c5c64';
  var RX = 466, RW = 318, RC = RX + RW / 2;                 // 右边说明区（左边代码窗口宽 440）
  var CP = 62, CX = RX + 16, CY = 196, GAP = 56;             // 列表小方块：间距、第一个方块左上角、两组之间多空出的距离
  var WANT = 4.5, OUT_A = 8.34, JOIN = [9.56, 9.91], FOUR = 10.95, NOT = 11.63;
  var TIMES = 13.96, OUT_B = 14.96, DUP = [15.59, 16.0], NOT_B = 16.25, PART_A_OUT = 17.3;
  var NP = 18.49, TERM = 20.42, PIP = [20.81, 21.6], OK = 21.6, IMPORT = 22.26, NICK = 24.49, NICK_OUT = 26.3;
  var GRID = 26.6, ADD = 29.67, ROW1 = [31.26, 32.4], ROW2 = [32.51, 33.6], SCALE = 34.65, GROW = 36.56, DIR = 37.11, TWICE = 38.9;

  // 一排小方块（列表里的数）：返回 { g, cells }，(x, y) 是第一个方块左上角
  function cells(parent, x, y, nums, color) {
    var g = E.el('g', {}, parent), out = [];
    nums.forEach(function (n, i) {
      var c = E.el('g', {}, g);
      K.paint(E.el('rect', { x: x + i * CP, y: y, width: 54, height: 58, rx: 9, 'stroke-width': 2.5 }, c), E.tint(color, 16, 'var(--demo-surface)'), color);
      var t = E.txt(c, x + i * CP + 27, y + 40, String(n), 30, color, 'middle', 700);
      t.style.fontFamily = E.MONO;
      out.push(c);
    });
    return { g: g, cells: out };
  }

  E.scenes[0] = {
    title: '列表不是向量：要用 NumPy 数组',
    dur: 40.36,
    cues: [
      { at: 0, s: 'Python 自带的**列表**也能装下一串数' },
      { at: 6.62, s: '列表相加是**接起来**：`[2, 1, 1, 2]`' },
      { at: 13.87, s: '列表乘 2 是把它重复一遍' },
      { at: 17.49, s: '装上 NumPy：`pip install numpy`，再 `import numpy as np`' },
      { at: 26.44, s: '`np.array` 是数组：对应的数分别相加' },
      { at: 34.55, s: '乘 2：每个数都乘 2，方向不变、两倍长' }
    ],
    build: function (K) {
      var s = K.sceneSvg('Python 的列表相加是把两个列表接在一起，乘 2 是重复一遍，都不是向量运算；装上 NumPy、导入并起短名字 np 以后，把列表变成数组，相加就是对应的数分别相加，乘 2 就是每个数都乘 2');

      /* ── 左：交互窗口（第 0 页列表、第 1 页 NumPy） ── */
      var py = E.code(s, 10, 14, 440, 392, { title: 'Python', size: 25 });
      py.line('>>> a = [{g:2, 1}]', 3.3, { dur: 1.5 });
      py.line('>>> b = [{w:1, 2}]', 6.96, { dur: 0.6 });
      py.line('>>> a + b', 7.67, { dur: 0.4 });
      py.line('{g:[2, 1}, {w:1, 2]}', OUT_A, { out: true });
      py.line('>>> 2 * a', TIMES, { dur: 0.4 });
      py.line('{g:[2, 1, 2, 1]}', OUT_B, { out: true });
      var L_IMP = py.line('>>> import numpy as {h.np:np}', IMPORT, { dur: 1.0, page: 1 });
      py.line('>>> a = np.array([{g:2, 1}])', 26.6, { dur: 1.4, page: 1 });
      py.line('>>> b = np.array([{w:1, 2}])', 28.1, { dur: 0.7, page: 1 });
      py.line('>>> a + b', 28.85, { dur: 0.3, page: 1 });
      var L_SUM = py.line('array([{a:3, 3}])', ADD, { out: true, page: 1 });
      py.line('>>> 2 * a', SCALE, { dur: 0.35, page: 1 });
      var L_TWO = py.line('array([{g:4, 2}])', 35.8, { out: true, page: 1 });
      var nickLbl = E.txt(s, 0, 0, '↑ 短名字', 20, E.HI, 'middle', 700);

      /* ── 右 ①②：「应该是」的算式 + 列表方块 ── */
      var partA = E.el('g', {}, s);
      var wantLbl = E.txt(partA, RX + 4, 40, '向量相加：', 22, E.INK2, 'start', 700);
      var want = E.tex(partA, RC + 10, 104, E.col([2, 1], H.good) + '+' + E.col([1, 2], H.warn) + '=' + E.col([3, 3], H.acc), 34, E.INK, 'middle', { w: 300, h: 120 });
      var A = cells(partA, CX, CY, [2, 1], E.GOOD);
      var B = cells(partA, CX + 2 * CP + GAP, CY, [1, 2], E.WARN);
      var D = cells(partA, CX, CY, [2, 1], E.GOOD);        // 「重复」抄出来的那一份
      var idx = E.el('g', {}, partA);
      [0, 1, 2, 3].forEach(function (i) { E.txt(idx, CX + 27 + i * CP, CY + 86, String(i + 1), 20, E.MUT, 'middle', 700); });
      var badG = E.el('g', {}, partA);
      E.cross(badG, RX + 18, 344, 11, E.BAD, 4);
      var badT = E.txt(badG, RX + 40, 352, '四个数，不是向量相加', 24, E.BAD, 'start', 900);

      /* ── 右 ③：NumPy 卡 + 小终端 ── */
      var npG = E.el('g', {}, s);
      var npT = E.txt(npG, RC, 92, 'NumPy', 46, E.HI, 'middle', 700); npT.style.fontFamily = E.MONO;
      E.txt(npG, RC, 136, '专门处理数组的库', 24, E.INK2, 'middle', 700);
      var termG = E.el('g', {}, s);
      var term = E.code(termG, RX, 186, RW, 132, { title: '终端', size: 22 });
      term.line('$ pip install numpy', PIP[0], { dur: PIP[1] - PIP[0] });
      var okG = E.el('g', {}, termG);
      E.check(okG, RX + 18, 284, 22, E.GOOD, 4);
      E.txt(okG, RX + 50, 300, '装好了', 22, E.GOOD, 'start', 900);

      /* ── 右 ④⑤：网格 + 算式 ── */
      var partC = E.el('g', {}, s);
      var pl = E.plane(partC, { ox: 520, oy: 362, u: 46, x0: -1, x1: 5, y0: -1, y1: 4 });
      var P = pl.P;
      var sumTex = E.tex(partC, RC + 6, 84, '', 34, E.INK, 'middle', { w: 320, h: 120 });
      var guide = E.el('g', {}, partC);
      var gx = E.el('path', { d: 'M ' + P(3, 3)[0] + ' ' + P(3, 3)[1] + ' V ' + P(3, 0)[1], fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '6 5' }, guide);
      var gy = E.el('path', { d: 'M ' + P(3, 3)[0] + ' ' + P(3, 3)[1] + ' H ' + P(0, 3)[0], fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '6 5' }, guide);
      K.paint(gx, null, E.HI); K.paint(gy, null, E.HI);
      var gx3 = E.txt(guide, P(3, 0)[0], P(3, 0)[1] + 26, '3', 22, E.HI, 'middle', 900);
      var gy3 = E.txt(guide, P(0, 3)[0] - 14, P(0, 3)[1] + 8, '3', 22, E.HI, 'end', 900);
      var dirLine = E.el('path', { d: 'M ' + P(-1, -0.5)[0] + ' ' + P(-1, -0.5)[1] + ' L ' + P(5, 2.5)[0] + ' ' + P(5, 2.5)[1], fill: 'none', 'stroke-width': 2, 'stroke-dasharray': '4 6' }, partC);
      K.paint(dirLine, null, E.HI);
      var green = E.vec(partC, E.GOOD, 5, { head: 16 }), amber = E.vec(partC, E.WARN, 5, { head: 16 }), blue = E.vec(partC, E.ACC, 5, { head: 16 });
      // 「两倍长」：箭头右下方两段等长的刻度（各占一倍）
      var twice = E.el('g', {}, partC), nx = 1 / Math.sqrt(5), ny = 2 / Math.sqrt(5), off = 16;
      [[0, 2], [2, 4]].forEach(function (seg, k) {
        var a = P(seg[0], seg[0] / 2), b = P(seg[1], seg[1] / 2);
        var p0 = [a[0] + nx * off, a[1] + ny * off], p1 = [b[0] + nx * off, b[1] + ny * off];
        K.paint(E.el('path', { d: 'M ' + (p0[0] + 3).toFixed(1) + ' ' + (p0[1] - 1.5).toFixed(1) + ' L ' + (p1[0] - 3).toFixed(1) + ' ' + (p1[1] + 1.5).toFixed(1), fill: 'none', 'stroke-width': 3, 'stroke-linecap': 'round' }, twice), null, E.HI);
        [p0, p1].forEach(function (q) { K.paint(E.el('path', { d: 'M ' + (q[0] - nx * 7).toFixed(1) + ' ' + (q[1] - ny * 7).toFixed(1) + ' L ' + (q[0] + nx * 7).toFixed(1) + ' ' + (q[1] + ny * 7).toFixed(1), fill: 'none', 'stroke-width': 2.5 }, twice), null, E.HI); });
      });
      E.txt(twice, P(3.4, 1.7)[0] + 40, P(3.4, 1.7)[1] + 56, '两倍长', 22, E.HI, 'middle', 900);

      return {
        el: s,
        draw: function (st, t) {
          var fade = E.fade;
          py.draw(st, t);

          /* 「短名字」：框住 np，下面一排标字（下一句开始前收掉） */
          var nk = E.win(st, NICK, NICK_OUT, 0.3);
          py.box('np', E.HI, nk);
          var npSeg = py.seg('np');
          nickLbl.setAttribute('x', (py.x(npSeg.col) + py.cw).toFixed(1));
          nickLbl.setAttribute('y', (py.y(L_IMP) + 36).toFixed(1));
          E.op(nickLbl, nk);
          // 「二加一」「一加二」时 array([3, 3]) 那一句垫底色
          py.hl(L_SUM, E.HI, Math.max(E.win(st, ROW1[0], ROW1[1], 0.2), E.win(st, ROW2[0], ROW2[1], 0.2)) * 0.9);
          py.hl(L_TWO, E.GOOD, E.win(st, 35.8, 38.0, 0.3) * 0.8);

          /* ①② 列表 */
          E.op(partA, 1 - fade(st, PART_A_OUT, 0.4));
          var scal = st >= TIMES - 0.2;
          var wa = scal ? fade(st, TIMES, 0.3) : fade(st, WANT, 0.3) * (1 - fade(st, TIMES - 0.3, 0.2));
          wantLbl.textContent = scal ? '数乘：' : '向量相加：';
          want.setTex(scal ? '\\textcolor{' + H.hi + '}{2}\\times' + E.col([2, 1], H.good) + '=' + E.col([4, 2], H.good)
            : E.col([2, 1], H.good) + '+' + E.col([1, 2], H.warn) + '=' + E.col([3, 3], H.acc));
          E.op(wantLbl, wa); E.op(want, wa);
          // 方块：① 8.34 两组分开出现，9.56 琥珀滑过来贴上；② 13.96 只剩绿组，15.59 抄一份滑到右边
          var pa = fade(st, OUT_A, 0.3) * (1 - fade(st, TIMES - 0.3, 0.25));
          var pb = fade(st, TIMES, 0.3);
          E.op(A.g, Math.max(pa, pb));
          var j = fade(st, JOIN[0], JOIN[1] - JOIN[0]);
          B.g.setAttribute('transform', 'translate(' + (-GAP * j).toFixed(1) + ' 0)');
          E.op(B.g, pa);
          var dp = fade(st, DUP[0], DUP[1] - DUP[0]);
          D.g.setAttribute('transform', 'translate(' + (2 * CP * dp).toFixed(1) + ' ' + (-36 * Math.sin(Math.PI * dp)).toFixed(1) + ')');
          E.op(D.g, pb * (dp > 0 ? 1 : 0));
          E.op(idx, Math.max(fade(st, FOUR, 0.3) * pa, fade(st, NOT_B - 0.3, 0.3) * pb));
          badT.textContent = scal ? '只是重复了一遍' : '四个数，不是向量相加';
          E.op(badG, Math.max(fade(st, NOT, 0.3) * pa, fade(st, NOT_B, 0.3) * pb));

          /* ③ NumPy 卡、小终端 */
          var pc = fade(st, NP, 0.4) * (1 - fade(st, GRID - 0.6, 0.4));
          E.pop(npG, RC, 110, pc, 0.85);
          var tg = fade(st, TERM, 0.3) * (1 - fade(st, GRID - 0.6, 0.4));
          E.op(termG, tg);
          term.draw(st, t);
          E.pop(okG, RX + 60, 292, fade(st, OK, 0.3), 0.6);

          /* ④⑤ 网格 */
          E.op(partC, fade(st, GRID, 0.5));
          var sc = st >= SCALE;
          var r1 = st >= ROW1[0] && st < ROW1[1], r2 = st >= ROW2[0] && st < ROW2[1];
          sumTex.setTex(sc ? '\\textcolor{' + H.hi + '}{2}\\times' + E.col([2, 1], H.good) + '=' + E.col([4, 2], H.good)
            : E.col([2, 1], r2 ? [DIM, H.good] : r1 ? [H.good, DIM] : H.good) + '+' + E.col([1, 2], r2 ? [DIM, H.warn] : r1 ? [H.warn, DIM] : H.warn) +
              '=' + E.col([3, 3], r2 ? [DIM, H.acc] : r1 ? [H.acc, DIM] : H.acc));
          E.op(sumTex, sc ? fade(st, SCALE, 0.3) : fade(st, ADD, 0.4) * (1 - fade(st, SCALE - 0.3, 0.2)));
          var g0 = P(0, 0), g1 = P(2, 1), g2 = P(3, 3);
          var pg = fade(st, ADD, 0.5), pam = fade(st, ADD + 0.4, 0.5), pbl = fade(st, 30.67, 0.6);
          var grow = fade(st, GROW, 0.5);
          var ge = P(2 + 2 * grow, 1 + grow);
          green.set(g0[0], g0[1], g0[0] + (ge[0] - g0[0]) * pg, g0[1] + (ge[1] - g0[1]) * pg); E.op(green.g, pg > 0 ? 1 : 0);
          var dimC = 1 - 0.8 * fade(st, SCALE, 0.4);
          amber.set(g1[0], g1[1], g1[0] + (g2[0] - g1[0]) * pam, g1[1] + (g2[1] - g1[1]) * pam); E.op(amber.g, (pam > 0 ? 1 : 0) * dimC);
          blue.set(g0[0], g0[1], g0[0] + (g2[0] - g0[0]) * pbl, g0[1] + (g2[1] - g0[1]) * pbl); E.op(blue.g, (pbl > 0 ? 1 : 0) * dimC);
          E.drawOn(gx, fade(st, ROW1[0], 0.4)); E.op(gx, E.win(st, ROW1[0], ROW1[1] + 0.6, 0.2)); E.op(gx3, E.win(st, ROW1[0] + 0.3, ROW1[1] + 0.6, 0.2));
          E.drawOn(gy, fade(st, ROW2[0], 0.4)); E.op(gy, E.win(st, ROW2[0], ROW2[1] + 0.6, 0.2)); E.op(gy3, E.win(st, ROW2[0] + 0.3, ROW2[1] + 0.6, 0.2));
          E.drawOn(dirLine, fade(st, DIR, 0.6));
          E.op(twice, fade(st, TWICE, 0.4));
        }
      };
    }
  };
})();
