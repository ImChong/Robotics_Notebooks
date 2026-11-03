// 011 第 6 幕：写成代码（路线 §L1 推荐做什么「用 PoE 公式手写 2–3 自由度机械臂的 FKinSpace」）。左边编辑器 fk.py，右上终端，右下说明。
// 代码照 Modern Robotics 官方 Python 库（modern_robotics 1.1.1，core.py 的 FKinSpace：T = M，从最后一个关节往前，T = MatrixExp6(VecTose3(S_i θ_i)) · T）
// 手写成 fk_space（螺旋轴一行一个，免得讲转置）；输出都按 NumPy 2.5.3 + modern_robotics 1.1.1 实跑结果写（见 ../ep011.py）。
// 句 1「导入 Numb Pie」「导入两个函数」逐字打出前两行；句 2「乘上角度的螺旋轴，排成四乘四」框 VecTose3（S[i] × theta[i] 先乘好，VecTose3 只负责排成 4 × 4），「算矩阵指数」框 MatrixExp6。
// 句 3「三样输入」打 def 一行，M、S、theta 依次框亮。句 4「等于零位时、手的位姿」T = M；「从最后一个关节往前」for 一行；
// 「算矩阵指数，乘到左边」两行，「@ T」框亮。句 5「教材库里」右下写 FKinSpace 也是这样写的。
// 句 6–7 第 2 页（函数照原样，接着写例子）：M 四行、S 两行；「圆周率除以二」theta 一行，框 np.pi / 2、右下写 π ÷ 2 ≈ 1.57。
// 句 8「运行它」print 一行，终端 python3 fk.py，四行输出；「负四、一」最后一列的 −4、1 框紫，「x 轴朝左」「y 轴朝下」前两列框红、绿。
// 句 9「负零」框住 −0.，右下写 −2.2 × 10⁻¹⁶。句 10 第 3 页：例子换成教材例 4.5 的 UR5（M、六根螺旋轴、六个角度）。
// 句 11 再跑：位置 [0.095 0.109 0.988] 框紫，「和教材上的结果一样」右下打勾；句 12 朝向三行框蓝，右下写「第 6 集：原地往左转 90°」。
// 2026-11-03（用户反馈）：句 10 先 M 后 S（原先 S 先打，最后一排在窗口滚上来之前出了窗口下沿）；句 11–13 讲打印那两句：方括号里逗号前选横排、
// 后选列（`:3` 框亮黄、列的切片框紫 / 蓝），右下画 4 × 4 小格子，上面、左边写编号 0–3，选到的格子上色，「位置」（第 4 列前 3 个）紫、「朝向」（左上 3 × 3）蓝；
// 「保留几位小数」框 .round(3) / .round(2)。
// 时刻取自配音的逐词时间，见 ../ep011.py。
(function () {
  var E = window.EP011, K = E.K, H = E.HEX;
  var T = {
    imp: 3.63, lib: 4.72, two: 7.26,
    vec: 10.52, mul: 9.62, x44: 11.7, exp: 13.27,
    def: 15.67, three: 16.4, M: 17.41, S: 20.19, th: 21.72,
    tm: 23.36, eq: 24.4, loop: 26.6, back: 28.05, calc: 28.84, left: 30.76, ret: 31.85, order: 32.13,
    book: 34.99, same: 38.23,
    ex: 40.35, m33: 42.95, z: 44.53, m44: 45.66, s6: 46.5, nums: 47.81,
    two90: 49.55, rad: 52.22, pi: 53.53, half: 54.08, v157: 55.04,
    run: 57.12, print: 57.89, res: 58.3, p4: 59.88, p1: 60.54, cx: 61.02, cy: 61.98, same2: 62.95,
    neg0: 65.2, tiny: 68.33, round: 70.15,
    // 句 10 起（2026-11-03）：先 M 后 S（和代码顺序一致，原先 S 先打、窗口还没滚到，最后一排出了窗口下沿）；句 11–13 讲打印那两句在取什么
    ur: 73.49, urM: 76.11, urS: 78.19, book2: 79.57, ang: 81.5, angs: [82.17, 82.67, 83.87, 84.36, 84.91, 85.86],
    last2: 87.31, pr: 87.98, part: 88.99, brk: 90.43, rowsel: 91.38, rowW: 92.4, colsel: 93.04, colW: 93.78,
    c3: 95.41, head3: 96.37, three: 96.96, lone3: 97.68, idx3: 98.79, fourth: 100.69, py0: 101.49, zero0: 102.46,
    l1: 104.79, right: 105.68, front3: 106.65, posW: 108.3, keep3: 108.93, l2: 110.37, tl: 111.32, t33: 111.99, rotW: 113.06, keep2: 113.71,
    run2: 116.02, res2: 117.02, p95: 118.22, p109: 119.46, p988: 120.7, same3: 122.25,
    rot: 124.84, ep6: 126.0, left90: 127.28
  };
  var END = 130.3;

  E.scenes[5] = {
    title: '写成代码',
    dur: END,
    cues: [
      { at: 0, s: '把**指数积公式**写成代码：导入 NumPy，再从教材配套的 Python 库导入两个函数' },
      { at: 9.08, s: '`VecTose3`：乘上角度的螺旋轴，排成 4 × 4；`MatrixExp6`：算**矩阵指数**' },
      { at: 15.57, s: '三样输入：零位时手的**位姿** `M`、每个关节的**螺旋轴** `S`、每个关节的**角度** `theta`' },
      { at: 23.26, s: '`T = M`，再从最后一个关节往前，一个一个算**矩阵指数**、乘到左边：第 4 幕的顺序' },
      { at: 34.94, s: '教材库里算正向运动学的 `FKinSpace`，也是这样写的' },
      { at: 40.3, s: '这一集的例子：平面的 3 × 3 补一根 z 轴，写成 4 × 4；两根**螺旋轴**各 6 个数' },
      { at: 49.5, s: '两个**角度**都是 90°：代码里用弧度，`np.pi / 2` ≈ 1.57' },
      { at: 57.07, s: '运行：手的**位置** (−4, 1)，x 轴朝左，y 轴朝下，和手算的一样' },
      { at: 65.15, s: '`-0.` 就是 0：电脑算出的是一个非常接近 0 的负数，保留两位小数成了 −0' },
      { at: 73.49, s: '换成教材例 4.5 的六关节机械臂：M、六根**螺旋轴**，θ = (0, −90°, 0, 0, 90°, 0)' },
      { at: 87.21, s: '最后两句打印只取 4 × 4 的一部分：方括号里，逗号前面选横排，逗号后面选列' },
      { at: 95.31, s: '`:3` 从头取 3 个；单独的 `3` 是编号 3，也就是第 4 个（Python 的编号从 0 开始）' },
      { at: 104.32, s: '`T[:3, 3]`：最右边一列的前 3 个数 = 手的**位置**；`T[:3, :3]`：左上角 3 × 3 = **朝向**' },
      { at: 115.92, s: '同一个函数：手停在 (0.095, 0.109, 0.988) 米，和教材上的结果一样' },
      { at: 124.74, s: '**朝向**正是第 6 集「原地往左转 90°」的那个矩阵' }
    ],
    build: function (K) {
      var s = K.sceneSvg('把指数积公式写成代码 fk.py：导入 numpy，从教材配套的 modern_robotics 库导入 MatrixExp6、VecTose3；def fk_space(M, S, theta): T = M，从最后一个关节往前，se3 = VecTose3(S[i] * theta[i])，T = MatrixExp6(se3) @ T，return T。这一集的例子 M = [[1, 0, 0, 3], [0, 1, 0, 2], [0, 0, 1, 0], [0, 0, 0, 1]]，两根螺旋轴，两个角度 np.pi / 2，打印出来手在 (−4, 1)，x 轴朝左、y 轴朝下，和手算一样；−0. 是一个非常接近零的负数。换成教材例 4.5 的 UR5，角度 (0, −π/2, 0, 0, π/2, 0)，手停在 (0.095, 0.109, 0.988) 米，和教材一样，朝向是第六集原地往左转九十度的矩阵');

      /* ── 左：编辑器 ── */
      var edG = E.el('g', {}, s);
      var ed = E.code(edG, 10, 14, 540, 392, { title: 'fk.py', size: 18, lh: 26 });
      var FUN = [
        ['import numpy as np', T.imp, 0.7],
        ['from modern_robotics import {h.exp:MatrixExp6}, {w.vec:VecTose3}', T.lib + 0.2, 2.6],
        ['', T.two + 0.3, 0],
        ['def fk_space({g.M:M}, {w.S:S}, {a.th:theta}):', T.def, 1.3],
        ['    T = M', T.tm, 0.7],
        ['    for i in {i.rev:reversed}(range(len(theta))):', T.loop, 1.4],
        ['        se3 = VecTose3(S[i] * theta[i])', T.calc, 0.9],
        ['        T = {h.left:MatrixExp6(se3) @ T}', T.calc + 1.0, 0.9],
        ['    return T', T.ret, 0.4]
      ];
      // 第 1 页：函数；第 2、3 页：函数原样（一下子出来）+ 例子
      FUN.forEach(function (l) { ed.line(l[0], l[1], { dur: l[2], page: 0 }); });
      function funPage(pg, at) { FUN.forEach(function (l) { ed.line(l[0].replace(/\{(\w)\.\w+:/g, '{$1:'), at, { dur: 0, page: pg }); }); ed.line('', at, { dur: 0, page: pg }); }
      funPage(1, T.ex);
      ed.line('M = np.array([[{x:1}, {y:0}, 0, {v:3}],', T.m33, { dur: 0.6, page: 1 });
      ed.line('              [{x:0}, {y:1}, 0, {v:2}],', T.m33 + 0.6, { dur: 0.5, page: 1 });
      ed.line('              [0, 0, {z.z:1}, 0],', T.z, { dur: 0.5, page: 1 });
      ed.line('              [0, 0, 0, 1]])', T.m44, { dur: 0.5, page: 1 });
      ed.line('S = np.array([[{w:0, 0, 1, 0, 0, 0}],', T.s6, { dur: 0.6, page: 1 });
      ed.line('              [{w:0, 0, 1, 2, -1, 0}]])', T.nums, { dur: 0.7, page: 1 });
      ed.line('theta = [{a.pi:np.pi / 2}, np.pi / 2]', T.rad, { dur: 1.2, page: 1 });
      var L_PRINT = ed.line('print(fk_space(M, S, theta).round(2))', T.run, { dur: 0.7, page: 1 });
      funPage(2, T.ur);
      ed.line('M = np.array([[-1, 0, 0, {v:0.817}],', T.urM, { dur: 0.3, page: 2 });
      ed.line('              [0, 0, 1, {v:0.191}],', T.urM + 0.35, { dur: 0.3, page: 2 });
      ed.line('              [0, 1, 0, {v:-0.006}],', T.urM + 0.7, { dur: 0.3, page: 2 });
      ed.line('              [0, 0, 0, 1]])', T.urM + 1.05, { dur: 0.3, page: 2 });
      ['S = np.array([[{w:0, 0, 1, 0, 0, 0}],', '              [{w:0, 1, 0, -0.089, 0, 0}],', '              [{w:0, 1, 0, -0.089, 0, 0.425}],',
        '              [{w:0, 1, 0, -0.089, 0, 0.817}],', '              [{w:0, 0, -1, -0.109, 0.817, 0}],', '              [{w:0, 1, 0, 0.006, 0, 0.817}]])'].forEach(function (l, k) {
        ed.line(l, T.urS + 0.3 * k, { dur: 0.25, page: 2 });
      });
      ed.line('theta = [{a:0}, {a:-np.pi / 2}, {a:0}, {a:0}, {a:np.pi / 2}, {a:0}]', T.ang, { dur: 4.4, page: 2 });
      ed.line('T = fk_space(M, S, theta)', T.last2, { dur: 0.5, page: 2 });
      // 打印两句：横排的切片亮黄、列的切片按含义上色（位置紫、朝向蓝），保留几位小数灰
      var L_P1 = ed.line('print(T[{h.r1::3}, {v.c1:3}]{i.d1:.round(3)})', T.pr, { dur: 0.45, page: 2 });
      var L_P2 = ed.line('print(T[{h.r2::3}, {a.c2::3}]{i.d2:.round(2)})', T.pr + 0.5, { dur: 0.45, page: 2 });

      /* ── 右上：终端 ── */
      var RX = 562, RW = 228;
      var termG = E.el('g', {}, s);
      var term = E.code(termG, RX, 14, RW, 222, { title: '终端', size: 18, lh: 25 });
      term.line('$ python3 fk.py', T.run + 0.3, { dur: 0.6 });
      var O1 = [term.line('[[{x.o00:-1.} {y:-0.}  0. {v.o03:-4.}]', T.res, { out: true }),
        term.line(' [ {x:0.} {y:-1.}  0.  {v.o13:1.}]', T.res, { out: true }),
        term.line(' [ 0.  0.  1.  0.]', T.res, { out: true }),
        term.line(' [ 0.  0.  0.  1.]]', T.res, { out: true })];
      term.line('$ python3 fk.py', T.run2 + 0.6, { dur: 0.5 });
      var O2 = [term.line('[{v:0.095 0.109 0.988}]', T.res2, { out: true }),
        term.line('[[-0. -1.  0.]', T.res2 + 0.1, { out: true }),
        term.line(' [ 1.  0.  0.]', T.res2 + 0.1, { out: true }),
        term.line(' [-0.  0.  1.]]', T.res2 + 0.1, { out: true })];

      /* ── 右下：说明 ── */
      var RK = E.rowKit(), NX = RX + 2;
      function rows(lines) { var g = E.el('g', {}, s); return { g: g, r: lines.map(function (l) { return RK.line(g, NX, 268 + l[0], l[1], l[2] || 18); }) }; }
      var n1 = rows([[0, [['VecTose3', E.WARN, 700]]], [26, [['螺旋轴 × 角度 → 4 × 4', E.INK2]]], [62, [['MatrixExp6', E.HI, 700]]], [88, [['算矩阵指数', E.INK2]]]]);
      n1.r[0].parts[0].style.fontFamily = E.MONO; n1.r[2].parts[0].style.fontFamily = E.MONO;
      var n2 = rows([[0, [['M', E.GOOD, 700], ['：零位时、手的位姿', E.INK2]]], [30, [['S', E.WARN, 700], ['：螺旋轴，一行一个', E.INK2]]], [60, [['theta', E.ACC, 700], ['：各关节的角度', E.INK2]]]]);
      [0, 1, 2].forEach(function (k) { n2.r[k].parts[0].style.fontFamily = E.MONO; });
      var n3 = rows([[0, [['从最后一个关节往前', E.INK, 900]]], [28, [['一个个乘到左边', E.INK, 900]]], [60, [['= 第 4 幕的顺序', E.GOOD, 900]]]]);
      var n4 = rows([[0, [['教材库的 ', E.INK2], ['FKinSpace', E.HI, 700]]], [28, [['也是这样写的', E.INK2]]]]);
      n4.r[0].parts[1].style.fontFamily = E.MONO;
      var n5 = rows([[0, [['3 × 3', E.INK2], [' 补 z 轴 → ', E.INK2], ['4 × 4', E.INK, 900]]], [30, [['螺旋轴：各 6 个数', E.WARN, 900]]],
        [64, [['90° = π ÷ 2 ≈ 1.57', E.ACC, 900]]], [92, [['代码里角度用弧度', E.INK2]]]]);
      var n6 = rows([[0, [['手在 ', E.INK2], ['(−4, 1)', E.VEL, 900]]], [28, [['x 轴', E.X, 900], ['朝左，', E.INK2], ['y 轴', E.Y, 900], ['朝下', E.INK2]]], [62, [['和手算的一样 ✓', E.GOOD, 900]]]]);
      var n7 = rows([[0, [['-0.', E.HI, 700], [' 就是 0', E.INK2]]], [28, [['电脑算出 −2.2 × 10⁻¹⁶', E.INK2]]], [56, [['保留两位小数 → −0', E.INK2]]]]);
      n7.r[0].parts[0].style.fontFamily = E.MONO;
      /* ── 右下：4 × 4 的小格子，讲切片（句 11–13）：上面、左边写编号 0–3，取到的格子上色 ── */
      var gM = E.el('g', {}, s), GX = RX + 40, GY = 290, CW = 24;
      var cells = [];
      for (var i = 0; i < 4; i++) { cells.push([]); for (var j = 0; j < 4; j++) cells[i].push(K.paint(E.el('rect', { x: GX + CW * j + 1, y: GY + CW * i + 1, width: CW - 2, height: CW - 2, rx: 3 }, gM), E.S2)); }
      var colIdx = [0, 1, 2, 3].map(function (j) { return E.txt(gM, GX + CW * j + CW / 2, GY - 7, String(j), 18, E.MUT, 'middle', 700); });
      var rowIdx = [0, 1, 2, 3].map(function (i) { return E.txt(gM, GX - 9, GY + CW * i + CW / 2 + 6, String(i), 18, E.MUT, 'end', 700); });
      var gLab1 = E.txt(gM, GX + 4 * CW + 10, GY + 1.5 * CW + 6, '位置', 18, E.VEL, 'start', 900);
      var gLab2 = E.txt(gM, GX + 4 * CW + 10, GY + 3 * CW + 22, '朝向', 18, E.OMEGA, 'start', 900);
      var gCap = E.txt(gM, GX - 20, GY + 4 * CW + 22, '编号从 0 开始', 18, E.INK2, 'start', 700);
      var n8 = rows([[0, [['教材例 4.5：UR5', E.GOOD, 900]]], [28, [['θ = (0, −90°, 0, 0, 90°, 0)', E.ACC, 900]]]]);
      var n9 = rows([[0, [['手：', E.INK2], ['(0.095, 0.109, 0.988)', E.VEL, 900]]], [28, [['和教材上的一样 ✓', E.GOOD, 900]]],
        [62, [['朝向：第 6 集', E.INK2]]], [88, [['原地往左转 90°', E.OMEGA, 900]]]]);

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) { RK.layout(); laid = true; }
          var f = E.fade, b = E.bump;
          ed.draw(st, t); term.draw(st, t);
          // 编辑器里的框
          var pg0 = st < T.ex;
          if (pg0) {
            ed.box('vec', E.WARN, f(st, T.vec, 0.3) * (1 - f(st, T.def, 0.3)));
            ed.box('exp', E.HI, f(st, T.exp, 0.3) * (1 - f(st, T.def, 0.3)));
            ed.box('M', E.GOOD, f(st, T.M, 0.3) * (1 - f(st, T.tm, 0.3)));
            ed.box('S', E.WARN, f(st, T.S, 0.3) * (1 - f(st, T.tm, 0.3)));
            ed.box('th', E.ACC, f(st, T.th, 0.3) * (1 - f(st, T.tm, 0.3)));
            ed.box('rev', E.GOOD, f(st, T.back, 0.3) * (1 - f(st, T.book, 0.3)));   // 绿：第 4 幕的顺序（右下「= 第 4 幕的顺序」同色；原先灰框像「不用」）
            ed.box('left', E.HI, f(st, T.left, 0.3) * (1 - f(st, T.book, 0.3)));
          } else {
            ['vec', 'exp', 'M', 'S', 'th', 'rev', 'left'].forEach(function (k) { ed.box(k, E.HI, 0); });
          }
          if (st >= T.ex && st < T.ur) {
            ed.box('z', E.Z, b(st, T.z, 1.0));
            ed.box('pi', E.ACC, f(st, T.pi, 0.3) * (1 - f(st, T.run, 0.3)));
          } else { ed.box('z', E.Z, 0); ed.box('pi', E.ACC, 0); }
          // 终端里的框
          term.box('o03', E.VEL, Math.max(f(st, T.p4, 0.3), 0) * (1 - f(st, T.neg0, 0.3)));
          term.box('o13', E.VEL, f(st, T.p1, 0.3) * (1 - f(st, T.neg0, 0.3)));
          term.boxCol(O1[0], 2, 3, E.X, f(st, T.cx, 0.3) * (1 - f(st, T.neg0, 0.3)));
          term.boxCol(O1[1], 2, 3, E.X, f(st, T.cx, 0.3) * (1 - f(st, T.neg0, 0.3)));
          term.boxCol(O1[0], 6, 3, E.Y, Math.max(f(st, T.cy, 0.3) * (1 - f(st, T.neg0, 0.3)), f(st, T.neg0, 0.3) * (1 - f(st, T.ur, 0.3))));
          term.boxCol(O1[1], 6, 3, E.Y, f(st, T.cy, 0.3) * (1 - f(st, T.neg0, 0.3)));
          term.boxCol(O2[0], 1, 17, E.VEL, f(st, T.p95, 0.3) * (1 - f(st, T.rot, 0.3)));
          [1, 2, 3].forEach(function (k) { term.boxCol(O2[k], 1, 13, E.OMEGA, f(st, T.rot, 0.3)); });
          // 说明
          E.op(n1.g, fr(st, T.vec) * (1 - f(st, T.def - 0.3, 0.3))); E.op(n1.r[2].g, fr(st, T.exp)); E.op(n1.r[3].g, fr(st, T.exp));
          E.op(n2.g, fr(st, T.three) * (1 - f(st, T.tm - 0.3, 0.3))); E.op(n2.r[0].g, fr(st, T.M)); E.op(n2.r[1].g, fr(st, T.S)); E.op(n2.r[2].g, fr(st, T.th));
          E.op(n3.g, fr(st, T.back) * (1 - f(st, T.book - 0.3, 0.3))); E.op(n3.r[1].g, fr(st, T.left)); E.op(n3.r[2].g, fr(st, T.order));
          E.op(n4.g, fr(st, T.book) * (1 - f(st, T.ex - 0.3, 0.3))); E.op(n4.r[1].g, fr(st, T.same));
          E.op(n5.g, fr(st, T.m33) * (1 - f(st, T.run - 0.3, 0.3))); E.op(n5.r[0].parts[1], fr(st, T.z)); E.op(n5.r[0].parts[2], fr(st, T.m44));
          E.op(n5.r[1].g, fr(st, T.s6)); E.op(n5.r[2].g, fr(st, T.pi)); E.op(n5.r[3].g, fr(st, T.rad));
          E.op(n6.g, fr(st, T.p4) * (1 - f(st, T.neg0 - 0.3, 0.3))); E.op(n6.r[1].g, fr(st, T.cx)); E.op(n6.r[1].parts[2], fr(st, T.cy)); E.op(n6.r[1].parts[3], fr(st, T.cy));
          E.op(n6.r[2].g, fr(st, T.same2));
          E.op(n7.g, fr(st, T.neg0) * (1 - f(st, T.ur - 0.3, 0.3))); E.op(n7.r[1].g, fr(st, T.tiny)); E.op(n7.r[2].g, fr(st, T.round));
          E.op(n8.g, fr(st, T.ur) * (1 - f(st, T.last2 - 0.3, 0.3))); E.op(n8.r[1].g, fr(st, T.ang));
          // 句 11–13：小格子
          E.op(gM, fr(st, T.part) * (1 - f(st, T.run2 - 0.3, 0.3)));
          var selR = f(st, T.rowW, 0.3) * (1 - f(st, T.l1, 0.3)) + f(st, T.head3, 0.3) * (1 - f(st, T.lone3, 0.3));
          var selC = f(st, T.colW, 0.3) * (1 - f(st, T.c3, 0.3)) + f(st, T.idx3, 0.3) * (1 - f(st, T.l1, 0.3));
          var pos = f(st, T.right, 0.3), rotA = f(st, T.tl, 0.3);   // 位置那一列讲完也留着：第二句讲朝向时，两块一起看
          cells.forEach(function (row, i) { row.forEach(function (c, j) {
            var col = E.S2, a = 0;
            if (i < 3 && j === 3 && pos > 0.01) { col = E.VEL; a = pos; }
            else if (i < 3 && j < 3 && rotA > 0.01) { col = E.OMEGA; a = rotA; }
            else if (i < 3 && selR > 0.01) { col = E.HI; a = 0.7 * selR; }
            if (j === 3 && selC > 0.01 && !(pos > 0.01)) { col = E.VEL; a = Math.max(a, 0.8 * selC); }
            K.paint(c, a > 0.01 ? E.tint(col, Math.round(25 + 50 * a), 'var(--demo-surface-2)') : E.S2);
          }); });
          colIdx.forEach(function (n, j) { n.style.fill = (j === 3 && st >= T.idx3 && st < T.l1) ? E.VEL : (j === 0 && st >= T.zero0 && st < T.l1) ? E.HI : E.MUT; });
          rowIdx.forEach(function (n, i) { n.style.fill = (i < 3 && st >= T.head3 && st < T.lone3) ? E.HI : E.MUT; });
          E.op(gLab1, fr(st, T.posW)); E.op(gLab2, fr(st, T.rotW)); E.op(gCap, fr(st, T.py0) * (1 - f(st, T.l1, 0.3)));
          // 代码里的框：方括号整段、横排切片、列切片、保留几位小数；整句垫底色
          if (st >= T.last2 && st < T.run2) {
            ed.hl(L_P1, E.VEL, f(st, T.l1, 0.3) * (1 - f(st, T.l2, 0.3)));
            ed.hl(L_P2, E.OMEGA, f(st, T.l2, 0.3) * (1 - f(st, T.run2 - 0.3, 0.3)));
          }
          ed.box('r1', E.HI, st < T.run2 ? f(st, T.rowsel, 0.3) * (1 - f(st, T.colsel, 0.3)) + f(st, T.c3, 0.3) * (1 - f(st, T.lone3, 0.3)) : 0);
          ed.box('c1', E.VEL, st < T.run2 ? f(st, T.colsel, 0.3) * (1 - f(st, T.c3, 0.3)) + f(st, T.lone3, 0.3) * (1 - f(st, T.l1, 0.3)) + f(st, T.right, 0.3) * (1 - f(st, T.keep3, 0.3)) : 0);
          ed.box('d1', E.INK2, st < T.run2 ? f(st, T.keep3, 0.3) * (1 - f(st, T.l2, 0.3)) : 0);
          ed.box('r2', E.HI, st < T.run2 ? f(st, T.tl, 0.3) * (1 - f(st, T.keep2, 0.3)) : 0);
          ed.box('c2', E.OMEGA, st < T.run2 ? f(st, T.tl, 0.3) * (1 - f(st, T.keep2, 0.3)) : 0);
          ed.box('d2', E.INK2, st < T.run2 ? f(st, T.keep2, 0.3) * (1 - f(st, T.run2 - 0.3, 0.3)) : 0);
          E.op(n9.g, fr(st, T.p95)); E.op(n9.r[1].g, fr(st, T.same3)); E.op(n9.r[2].g, fr(st, T.rot)); E.op(n9.r[3].g, fr(st, T.left90));
        }
      };
    }
  };
})();
