// 004 第 5 幕：存成文件，跑一遍。左边先是编辑器 hand.py（前几幕的几句整理成 11 句），三维时换成交互窗口；右边说明区。
// ① 「这几句」hand.py 一句句贴进来；「存成」右下出一个文件图标「hand.py」，「文件」标题栏的 hand.py 变亮黄。
// ② 「终端」右上弹出终端，「用 Python 跑它」打 python3 hand.py；「三个结果」三行输出依次出来（[-2  3]、[-4  3] 蓝，[-2  6] 亮黄），
//    hand.py 里对应的三句 print 同时垫底色。
// ③ 「三维」左边换成交互窗口：「数组里写三个数」p = np.array([3, 2, 1])，「矩阵写三个横排」R 写成三排（绕 z 轴转 90°，第三列 0、0、1），
//    「乘法」R @ p → array([-2,  3,  1])，「一点都不用改」R @ p 框亮黄、旁边写「和二维一样」；
//    右边一个三维小坐标架（x 右、y 上、z 斜着朝向观众），p = (3, 2, 1) 一支蓝箭头，「乘法」时绕 z 轴转 90° 到 (−2, 3, 1)，箭头尖画出转过的虚线弧。
// ④ 「后面的课里」右边换成三张卡：旋转矩阵（3×3，第 5 集）、位姿（4×4，第 7 集）、雅可比（第 13 集），念到哪张弹哪张；
//    「都是这样的数组」下面的框写「都是 np.array」，「艾特」再写「都用 @ 相乘」。
// 终端输出与三维结果都用 NumPy 2.5.3 实跑过（见 ../ep004.py）。时刻取自配音的逐词时间（分镜时间 ≈ 念到的时刻）。
(function () {
  var E = window.EP004, K = E.K;
  var RX = 466, RW = 318, RC = RX + RW / 2;
  var PASTE = 1.10, SAVE = 1.68, FILE = 2.71, TERM = 4.05, RUN = [4.65, 5.55], OUTS = [6.13, 6.41, 6.73];
  var D3 = 8.75, P3 = 9.84, R3 = [11.22, 11.6, 11.95], MUL = 12.71, RES = 13.07, SAME = 13.57, TURN = [12.8, 13.9];
  var LATER = 15.67, CARDS = [16.73, 17.75, 19.22], ARR = 20.78, AT = 22.55;

  E.scenes[4] = {
    title: '存成文件，跑一遍',
    dur: 24.21,
    cues: [
      { at: 0, s: '把这几句存成一个 Python 文件：`hand.py`' },
      { at: 3.96, s: '`python3 hand.py`：三个结果一次打印出来' },
      { at: 8.66, s: '三维：三个数、三个横排，`R @ p` 不用改' },
      { at: 15.57, s: '旋转、位姿、雅可比：都是数组，都用 `@` 相乘' }
    ],
    build: function (K) {
      var s = K.sceneSvg('把这几句存成 Python 文件 hand.py，在终端里用 python3 运行，三个结果 [-2 3]、[-4 3]、[-2 6] 一次打印出来；三维也一样：数组写三个数、矩阵写三个横排，R @ p 一点都不用改；后面的旋转矩阵、4×4 位姿、雅可比在代码里都是这样的数组，都用 @ 相乘');

      /* ── 左 ①②：编辑器 hand.py（整段贴进来，不画光标） ── */
      var edG = E.el('g', {}, s);
      var ed = E.code(edG, 10, 14, 440, 392, { title: 'hand.py', size: 22, lh: 30, cursor: false });
      var SRC = [
        'import numpy as np',
        '',
        'p = np.array([{a:3}, {a:2}])',
        'R = np.array([[{g:0}, {w:-1}],',
        '              [{g:1},  {w:0}]])',
        'S = np.array([[{h:2}, 0],',
        '              [0, 1]])',
        '',
        'print(R @ p)',
        'print(S @ R @ p)',
        'print(R @ S @ p)'
      ];
      var L_PRINT = SRC.map(function (src, k) { return ed.line(src, PASTE + 0.05 * k, { dur: 0 }); }).slice(8);

      /* ── 左 ③：三维的交互窗口 ── */
      var pyG = E.el('g', {}, s);
      var py = E.code(pyG, 10, 14, 440, 392, { title: 'Python', size: 23 });
      py.line('>>> p = np.array([{a:3}, {a:2}, {a:1}])', P3, { dur: 0.9 });
      py.line('>>> R = np.array([[{g:0}, {w:-1}, 0],', R3[0], { dur: 0.36 });
      py.line('...               [{g:1},  {w:0}, 0],', R3[1], { dur: 0.33 });
      py.line('...               [{g:0},  {w:0}, 1]])', R3[2], { dur: 0.36 });
      var L_MUL = py.line('>>> {h.mul:R @ p}', MUL, { dur: 0.3 });
      py.line('array([{a:-2},  {a:3},  {a:1}])', RES, { out: true });
      var sameT = E.txt(pyG, 0, 0, '← 和二维一样', 22, E.HI, 'start', 900);

      /* ── 右 ①：文件图标 ── */
      var fileG = E.el('g', {}, s);
      var FX = RC - 34, FY = 270;
      K.paint(E.el('path', { d: 'M ' + FX + ' ' + FY + ' H ' + (FX + 48) + ' L ' + (FX + 68) + ' ' + (FY + 20) + ' V ' + (FY + 86) + ' H ' + FX + ' Z', 'stroke-width': 2.5, 'stroke-linejoin': 'round' }, fileG), E.S2, E.INK2);
      K.paint(E.el('path', { d: 'M ' + (FX + 48) + ' ' + FY + ' V ' + (FY + 20) + ' H ' + (FX + 68), fill: 'none', 'stroke-width': 2.5, 'stroke-linejoin': 'round' }, fileG), null, E.INK2);
      [0, 1, 2].forEach(function (k) { K.paint(E.el('line', { x1: FX + 12, x2: FX + 56 - 12 * (k % 2), y1: FY + 40 + 13 * k, y2: FY + 40 + 13 * k, 'stroke-width': 3, 'stroke-linecap': 'round' }, fileG), null, E.MUT); });
      var fileT = E.txt(fileG, RC, FY + 118, 'hand.py', 24, E.HI, 'middle', 700); fileT.style.fontFamily = E.MONO;

      /* ── 右 ②：终端 ── */
      var termG = E.el('g', {}, s);
      var term = E.code(termG, RX, 22, RW, 196, { title: '终端', size: 22 });
      term.line('$ python3 hand.py', RUN[0], { dur: RUN[1] - RUN[0] });
      term.line('{a:[-2  3]}', OUTS[0], { out: true });
      term.line('{a:[-4  3]}', OUTS[1], { out: true });
      term.line('{h:[-2  6]}', OUTS[2], { out: true });

      /* ── 右 ③：三维小坐标架（x 右、y 上、z 斜着朝向观众：每一格往左下 (−0.6, 0.45)） ── */
      var d3G = E.el('g', {}, s);
      var OX = 616, OY = 318, U = 38;
      function P3d(x, y, z) { return [OX + U * (x - 0.6 * z), OY - U * (y - 0.45 * z)]; }
      [[[0, 0, 0], [3.6, 0, 0], 'x'], [[0, 0, 0], [0, 3.6, 0], 'y'], [[0, 0, 0], [0, 0, 2.4], 'z']].forEach(function (ax) {
        var a = P3d.apply(null, ax[0]), b = P3d.apply(null, ax[1]);
        var v = E.vec(d3G, E.INK2, 2.5, { head: 12 }); v.set(a[0], a[1], b[0], b[1]);
        var lp = ax[2] === 'x' ? [b[0] + 8, b[1] + 8] : ax[2] === 'y' ? [b[0] + 12, b[1] + 6] : [b[0] - 12, b[1] + 20];
        E.txt(d3G, lp[0], lp[1], ax[2], 24, E.INK2, ax[2] === 'z' ? 'end' : 'start', 700);
      });
      var trace = E.el('path', { fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '6 6' }, d3G); K.paint(trace, null, E.HI);
      var drop = E.el('path', { fill: 'none', 'stroke-width': 2, 'stroke-dasharray': '4 5' }, d3G); K.paint(drop, null, E.MUT);
      var pv = E.vec(d3G, E.ACC, 5, { head: 16 });
      var p3Lbl = E.txt(d3G, 0, 0, '', 22, E.ACC, 'middle', 900); E.halo(p3Lbl);
      var zLbl = E.txt(d3G, OX - 70, OY + 82, '绕 z 轴转 90°', 20, E.HI, 'middle', 900);

      /* ── 右 ④：三张卡 + 结论框 ── */
      var later = E.el('g', {}, s);
      function card(y, title, sub, ep) {
        var g = E.el('g', {}, later);
        K.paint(E.el('rect', { x: RX + 4, y: y, width: RW - 8, height: 80, rx: 14, 'stroke-width': 2.5 }, g), E.S1, E.VIOLET);
        E.txt(g, RX + 22, y + 36, title, 26, E.VIOLET, 'start', 900);
        E.txt(g, RX + 22, y + 66, sub, 20, E.INK2, 'start', 700);
        E.tag(g, RX + RW - 104, y + 14, 86, 30, ep, E.VIOLET, { size: 18, ink: E.VIOLET, weight: 700 });
        return g;
      }
      var cards = [card(22, '旋转矩阵', '朝向：3×3', '第 5 集'), card(110, '位姿', '朝向 + 位置：4×4', '第 7 集'), card(198, '雅可比', '关节速度 → 手的速度', '第 13 集')];
      var box = E.el('g', {}, later);
      K.paint(E.el('rect', { x: RX + 4, y: 294, width: RW - 8, height: 108, rx: 14, 'stroke-width': 2.5 }, box), E.tint(E.HI, 10, 'var(--demo-surface)'), E.HI);
      function mixed(y, parts) {
        var t = E.txt(box, RC, y, '', 26, E.INK, 'middle', 900);
        t.setAttribute('xml:space', 'preserve'); t.style.whiteSpace = 'pre';
        parts.forEach(function (q) { var sp = E.el('tspan', {}, t); sp.textContent = q[0]; if (q[1]) { sp.style.fontFamily = E.MONO; sp.style.fill = E.HI; } });
        return t;
      }
      var row1 = mixed(338, [['都是 '], ['np.array', 1]]), row2 = mixed(384, [['都用 '], ['@', 1], [' 相乘']]);

      return {
        el: s,
        draw: function (st, t) {
          var fade = E.fade;
          /* 左：编辑器 → 三维交互窗口 */
          ed.draw(st, t);
          E.op(edG, 1 - fade(st, D3 - 0.2, 0.4));
          ed.title.style.fill = st >= FILE ? E.HI : E.INK2;
          var hc = [E.ACC, E.ACC, E.HI];
          L_PRINT.forEach(function (i, k) { ed.hl(i, hc[k], fade(st, OUTS[k], 0.25)); });
          py.draw(st, t);
          E.op(pyG, fade(st, D3, 0.4) * (1 - 0.45 * fade(st, LATER, 0.4)));
          py.box('mul', E.HI, fade(st, SAME, 0.3));
          sameT.setAttribute('x', (py.x(10) + 6).toFixed(1)); sameT.setAttribute('y', (py.y(L_MUL) - 2).toFixed(1));
          E.op(sameT, fade(st, SAME + 0.3, 0.3));

          /* 右 ①② */
          E.pop(fileG, RC, FY + 50, fade(st, SAVE, 0.35) * (1 - fade(st, D3 - 0.2, 0.4)), 0.7);
          E.op(termG, fade(st, TERM, 0.3) * (1 - fade(st, D3 - 0.2, 0.4)));
          term.draw(st, t);

          /* 右 ③ 三维：p 绕 z 轴转 90° */
          E.op(d3G, fade(st, D3 + 0.1, 0.4) * (1 - fade(st, LATER, 0.4)));
          var th = Math.PI / 2 * fade(st, TURN[0], TURN[1] - TURN[0]);
          var x = 3 * Math.cos(th) - 2 * Math.sin(th), y = 3 * Math.sin(th) + 2 * Math.cos(th);
          var o = P3d(0, 0, 0), tip = P3d(x, y, 1), foot = P3d(x, y, 0);
          var pp = fade(st, P3 + 0.4, 0.6);
          pv.set(o[0], o[1], o[0] + (tip[0] - o[0]) * pp, o[1] + (tip[1] - o[1]) * pp);
          drop.setAttribute('d', 'M ' + tip[0].toFixed(1) + ' ' + tip[1].toFixed(1) + ' L ' + foot[0].toFixed(1) + ' ' + foot[1].toFixed(1));
          E.op(drop, fade(st, P3 + 1.0, 0.3));
          var arc = [];
          for (var k = 0; k <= 24; k++) { var a = th * k / 24; arc.push(P3d(3 * Math.cos(a) - 2 * Math.sin(a), 3 * Math.sin(a) + 2 * Math.cos(a), 1)); }
          trace.setAttribute('d', th > 0.01 ? K.polyPath(arc) : '');
          var done = st >= TURN[1];
          p3Lbl.textContent = done ? '(−2, 3, 1)' : '(3, 2, 1)';
          p3Lbl.setAttribute('x', (tip[0] + (done ? -6 : 10)).toFixed(1)); p3Lbl.setAttribute('y', (tip[1] - 16).toFixed(1));
          E.op(p3Lbl, done ? fade(st, TURN[1], 0.3) : fade(st, P3 + 1.0, 0.3) * (1 - fade(st, TURN[0], 0.2)));
          E.op(zLbl, fade(st, MUL, 0.3));

          /* 右 ④ 三张卡 + 结论 */
          E.op(later, fade(st, LATER, 0.4));
          cards.forEach(function (c, k) { E.pop(c, RC, 62 + 88 * k, fade(st, CARDS[k], 0.3), 0.85); });
          E.op(box, fade(st, ARR - 0.2, 0.3));
          E.op(row1, fade(st, ARR, 0.3)); E.op(row2, fade(st, AT, 0.3));
        }
      };
    }
  };
})();
