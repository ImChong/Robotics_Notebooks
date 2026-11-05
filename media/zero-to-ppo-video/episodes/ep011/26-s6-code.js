// 011 第 6 幕：写成代码（路线 §L1 推荐做什么「用 PoE 公式手写 2–3 自由度机械臂的 FKinSpace」）。左边编辑器 fk.py，右上终端，右边说明。
// 代码照 Modern Robotics 官方 Python 库（modern_robotics 1.1.1，core.py 的 FKinSpace：T = M，从最后一个关节往前，T = MatrixExp6(VecTose3(S_i θ_i)) · T）
// 手写成 fk_space（螺旋轴一排一根，免得讲转置）；输出都按 NumPy 2.5.3 + modern_robotics 1.1.1 实跑结果写（见 ../ep011.py）。
// 2026-11-04（「观众完全零基础」：004 没教 Python 语法）重排：每种写法当场讲清在做什么，讲到哪段、编辑器里哪段垫底色，右边同步写说明。
//   第 1 页（函数）：终端先收起，右边整列写说明——import / as、from … import、函数是什么、两个导入的函数、def 和起的名字、括号里三样输入、
//   冒号下面缩进的几句（「缩进」时函数体五句一起打出来，每句开头的四个空格垫底色）、等号是「存进」（第 4 集「叫它 p」）、
//   for 那一句由里往外：len(theta) → 2、range → 编号 0、1、reversed → 1、0、i 依次取 1、0；S[i] * theta[i]（第 4 集：数组乘上一个数）、
//   VecTose3 → 4 × 4 存进 se3；MatrixExp6(se3) @ T 乘到左边、存回 T；return T 交回结果；从最后一个关节往前 = 第 4 幕的顺序。
//   第 2 页（这一集的例子）：终端出来，说明在右下——3 × 3 补 z 轴、螺旋轴一根一排；方括号 = 第 4 集的列表、弧度（第 9 集）、
//   np.pi = 圆周率 π ≈ 3.14 = 半圈；最后一句：三样输入放进括号交给函数、.round(2)、print；输出数后面的点 = 带小数的数（-4. 就是 −4）；
//   -0. 是电脑算出的 −0.00000000000000022 保留两位小数。
//   第 3 页（UR5）：T = fk_space(...) 先存进 T，再分两句打印；切片讲法同 11-03（4 × 4 小格子，上面、左边写编号 0–3）。
// 2026-11-05（用户反馈「说列了出来，就要给观众列出来」）：删掉「Modern Robotics 的库里，算正向运动学的函数，也是这样写的」
//   （屏幕上没给观众看官方代码；要给看又得讲 range 三个参数、np.dot、[:, i]）；结尾把书上例 4.5 的结果、第 6 集的矩阵摆出来对照。
// 时刻取自配音的逐词时间，见 ../ep011.py。
(function () {
  var E = window.EP011, K = E.K, H = E.HEX;
  var T = {
    // 第 1 页：函数
    four: 4.27, imp: 5.05, npW: 5.41, asnp: 6.05,
    from: 8.81, lib: 9.05, impt: 11.5, two: 11.89,
    fn: 13.81, named: 15.23, input: 16.61, calc0: 17.6, back0: 18.57,
    these: 20.45, vec: 22.04, mul: 22.19, x44: 24.34, exp: 25.87,
    defL: 28.88, defW: 30.46, name: 32.95,
    three: 35.33, M: 37.86, S: 40.69, th: 42.32,
    colon: 43.96, indent: 45.3, body: 44.94, job: 46.37,
    tm: 49.32, eq: 53.03, notEq: 54.35, store: 57.04, ep4: 58.58,
    loop: 62.96, back: 65.1, ln: 66.94, ln2: 69.71,
    rg: 71.78, rg2: 73.11, zero: 74.42, rv: 75.96, rv2: 77.08, rv3: 77.89,
    whole: 80.29, iseq: 81.47, each: 83.41, two2: 84.51,
    calc: 87.86, si: 88.63, ti: 90.39, ep4b: 91.62, star: 92.83, vt: 95.52, x44b: 97.41,
    calc2: 99.15, me: 99.89, me2: 100.89, at: 102.32, left: 103.07, storeB: 104.54,
    done: 106.89, ret: 109.29, order: 111.02, ep4m: 112.62,
    // 第 -3.65 页：这一集的例子
    ex: 115.57, m33: 117.65, z: 119, m44: 120.35, s6: 121.71, nums: 122.76, row1: 123.61,
    two90: 125.59, brk: 127.45, list: 129.9, rad: 130.81, ep9: 132.48,
    npi: 134.77, pi: 135.9, p314: 136.79, half: 138.12, div2: 139.35, v157: 140.27, is90: 141.19,
    last: 143.42, call: 144.4, fnc: 146.22, round: 148.64, prn: 150.14,
    run: 152.1, res: 153.09, p4: 154.04, p1: 154.7, cx: 155.2, cy: 156.18, same2: 157.16,
    dot: 159.41, dotW: 160.56, dec: 162.13, dec0: 163.32, m4: 164.71,
    neg0: 167.56, tiny: 169.05, near: 170.69, round0: 172.51,
    // 第 -2.65 页：UR5
    ur: 175.9, urM: 178.34, urS: 180.36, book2: 181.64, ang: 183.88,
    store2: 189.33, fnRes: 190.39, keepW: 191.77, pr: 192.46,
    part: 195.62, ep4c: 197.25, brk2: 198.53, rowsel: 199.45, rowW: 200.46, colsel: 201.09, colW: 201.82,
    c3: 203.45, head3: 204.41, lone3: 205.72, idx3: 206.83, fourth: 208.73, py0: 209.53, zero0: 210.5,
    l1: 212.84, right: 213.91, front3: 214.69, posW: 216.34, keep3: 216.97, l2: 218.41, tl: 219.36, t33: 220.03, rotW: 221.1, keep2: 221.75,
    run2: 224.06, res2: 224.92, p95: 226.26, same3: 230.18,
    rot: 233.46, ep6: 234.62, left90: 235.9
  };
  var END = 238.92;

  E.scenes[5] = {
    title: '写成代码',
    dur: END,
    cues: [
      { at: 0, s: '把**指数积公式**写成代码。第 1 句：照第 4 集导入 NumPy，起个短名字 `np`' },
      { at: 8.03, s: '第 2 句：`from` 库 `import` 函数——从《Modern Robotics》配套的 Python 库，导入 2 个函数' },
      { at: 13.71, s: '函数：一段起了名字的代码。给它输入，它算好结果，交回来' },
      { at: 20.35, s: '`VecTose3`：乘上角度的螺旋轴，排成 4 × 4；`MatrixExp6`：算**矩阵指数**' },
      { at: 28.23, s: '自己写一个函数：`def` 表示要写函数，`fk_space` 是起的名字' },
      { at: 35.23, s: '括号里 3 样输入：零位时手的**位姿** `M`、每个关节的**螺旋轴** `S`、每个关节的**角度** `theta`' },
      { at: 43.86, s: '冒号下面、往里缩进的几句：都是这个函数要做的事' },
      { at: 49.22, s: '`T = M`：代码里的等号不是「两边相等」，是把右边存进左边的名字（第 4 集「叫它 p」）' },
      { at: 62.86, s: '由里往外：`len(theta)` 一共几个角度 → 2；`range` 排出编号 0、1；`reversed` 倒过来 → 1、0' },
      { at: 80.19, s: '`for i in …`：编号 i 依次是 1、0，每个编号把下面缩进的 2 句做一遍' },
      { at: 87.76, s: '`S[i] * theta[i]`：这根**螺旋轴**的每个数都乘上**角度**（第 4 集），交给 `VecTose3` 排成 4 × 4' },
      { at: 99.05, s: '`MatrixExp6(se3) @ T`：算出**矩阵指数**，乘到结果左边，再存回 `T`' },
      { at: 106.79, s: '`return T`：把结果交回去。从最后一个关节往前，正是第 4 幕的顺序' },
      { at: 115.47, s: '这一集的例子：平面的 3 × 3 补一根 z 轴，写成 4 × 4；**螺旋轴**一根一排，各 6 个数' },
      { at: 125.49, s: '两个**角度**写在方括号里：第 4 集的列表；代码里角度用弧度（第 9 集）' },
      { at: 134.67, s: '`np.pi`：NumPy 存好的圆周率 π ≈ 3.14，是半圈的弧度；除以 2 ≈ 1.57 = 90°' },
      { at: 143.32, s: '最后一句：3 样输入放进括号交给函数，`.round(2)` 保留 2 位小数，`print` 打印出来' },
      { at: 152, s: '运行：手的**位置** (−4, 1)，x 轴朝左，y 轴朝下，和手算的一样' },
      { at: 159.31, s: '数后面的小点：带小数的数，小数部分是 0。`-4.` 就是 −4' },
      { at: 167.46, s: '`-0.` 就是 0：电脑算出的是一个非常接近 0 的负数，保留两位小数成了 −0' },
      { at: 175.8, s: '换成《Modern Robotics》例 4.5 的六关节机械臂：M、六根**螺旋轴**，θ = (0, −90°, 0, 0, 90°, 0)' },
      { at: 189.23, s: '`T = fk_space(M, S, theta)`：先把结果存进 `T`，再分 2 句打印' },
      { at: 194.67, s: '这 2 句只取 4 × 4 的一部分：方括号里，逗号前面选横排，逗号后面选列（第 4 集）' },
      { at: 203.35, s: '`:3` 从头取 3 个；单独的 `3` 是编号 3，也就是第 4 个（Python 的编号从 0 开始）' },
      { at: 212.36, s: '`T[:3, 3]`：最右边一列的前 3 个数 = 手的**位置**；`T[:3, :3]`：左上角 3 × 3 = **朝向**' },
      { at: 223.96, s: '同一个函数：手停在 (0.095, 0.109, 0.988) 米，和《Modern Robotics》书上的结果一样' },
      { at: 233.36, s: '**朝向**正是第 6 集「原地往左转 90°」的那个矩阵' }
    ],
    build: function (K) {
      var s = K.sceneSvg('把指数积公式写成代码 fk.py，每种写法当场讲清：import numpy as np 导入 NumPy、起短名字 np（第 4 集）；from 库 import 函数，从《Modern Robotics》配套的 modern_robotics 库导入 MatrixExp6、VecTose3；函数是一段起了名字的代码，给它输入，它算好结果交回来；def fk_space(M, S, theta): 自己写函数，括号里三样输入，冒号下面缩进的几句是函数要做的事；T = M，等号是把右边存进左边；for i in reversed(range(len(theta))): 编号依次是 1、0；se3 = VecTose3(S[i] * theta[i])；T = MatrixExp6(se3) @ T；return T。例子：M 补 z 轴写成 4 × 4，两根螺旋轴，theta = [np.pi / 2, np.pi / 2]（列表、弧度、圆周率），print(fk_space(M, S, theta).round(2)) 打印出手在 (−4, 1)；-4. 是带小数的数；−0. 是非常接近零的负数。换成《Modern Robotics》例 4.5 的 UR5，T = fk_space(M, S, theta)，T[:3, 3] 是位置 (0.095, 0.109, 0.988)，T[:3, :3] 是朝向，原地往左转九十度');

      /* ── 左：编辑器 ── */
      var edG = E.el('g', {}, s);
      var ed = E.code(edG, 10, 14, 540, 392, { title: 'fk.py', size: 18, lh: 26 });
      // 第 1 页：函数。函数体五句在「缩进」时一起打出来，之后一句一句讲
      var B0 = T.body, FUN = [
        ['{n.imp:import} {n.npm:numpy} {n.asnp:as np}', T.imp, 0.7],
        ['{n.from:from} {n.mrl:modern_robotics} {n.impt:import} {h.exp:MatrixExp6}, {w.vec:VecTose3}', T.from, 2.6],
        ['', T.two + 0.3, 0],
        ['{n.def:def} {n.name:fk_space}({g.M:M}, {w.S:S}, {a.th:theta}){n.colon::}', T.defL, 1.3],
        ['    {n.tm:T = M}', B0, 0.35],
        ['    for {n.i:i} in {n.rev:reversed}({n.rg:range}({n.ln:len(theta)})):', B0 + 0.4, 0.7],
        ['        {n.se3:se3} = {n.vt:VecTose3}({n.si:S[i]} {n.star:*} {n.ti:theta[i]})', B0 + 1.15, 0.6],
        ['        {n.tl:T} = {h.left:MatrixExp6(se3) @ T}', B0 + 1.8, 0.5],
        ['    {n.ret:return T}', B0 + 2.35, 0.3]
      ];
      var FL = FUN.map(function (l) { return ed.line(l[0], l[1], { dur: l[2], page: 0 }); });
      // 第 2、3 页：函数原样（一下子出来，去掉段名）+ 例子
      function funPage(pg, at) { FUN.forEach(function (l) { ed.line(l[0].replace(/\{(\w)\.\w+:/g, '{$1:'), at, { dur: 0, page: pg }); }); ed.line('', at, { dur: 0, page: pg }); }
      funPage(1, T.ex);
      ed.line('M = np.array([[{x:1}, {y:0}, 0, {v:3}],', T.m33, { dur: 0.6, page: 1 });
      ed.line('              [{x:0}, {y:1}, 0, {v:2}],', T.m33 + 0.6, { dur: 0.5, page: 1 });
      ed.line('              [0, 0, {z.z:1}, 0],', T.z, { dur: 0.5, page: 1 });
      ed.line('              [0, 0, 0, 1]])', T.m44, { dur: 0.5, page: 1 });
      var L_S1 = ed.line('S = np.array([[{w:0, 0, 1, 0, 0, 0}],', T.s6, { dur: 0.6, page: 1 });
      var L_S2 = ed.line('              [{w:0, 0, 1, 2, -1, 0}]])', T.nums, { dur: 0.7, page: 1 });
      var L_TH = ed.line('theta = [{a.pi:np.pi / 2}, np.pi / 2]', T.two90, { dur: 1.2, page: 1 });
      var L_PRINT = ed.line('{n.prn:print}({n.call:fk_space(M, S, theta)}{n.rd:.round(2)})', T.last, { dur: 0.7, page: 1 });
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
      var L_T2 = ed.line('T = fk_space(M, S, theta)', T.store2 + 0.2, { dur: 0.5, page: 2 });
      // 打印两句：横排的切片亮黄、列的切片按含义上色（位置紫、朝向蓝），保留几位小数灰
      var L_P1 = ed.line('print(T[{h.r1::3}, {v.c1:3}]{i.d1:.round(3)})', T.pr + 0.2, { dur: 0.45, page: 2 });
      var L_P2 = ed.line('print(T[{h.r2::3}, {a.c2::3}]{i.d2:.round(2)})', T.pr + 0.7, { dur: 0.45, page: 2 });

      /* ── 右上：终端（第 1 页讲函数时收起，右边整列写说明） ── */
      var RX = 562, RW = 228;
      var termG = E.el('g', {}, s);
      var term = E.code(termG, RX, 14, RW, 222, { title: '终端', size: 18, lh: 25 });
      term.line('$ python3 fk.py', T.run + 0.3, { dur: 0.6 });
      var O1 = [term.line('[[{x.o00:-1.} {y:-0.}  0. {v.o03:-4.}]', T.res, { out: true }),
        term.line(' [ {x:0.} {y:-1.}  0.  {v.o13:1.}]', T.res, { out: true }),
        term.line(' [ 0.  0.  1.  0.]', T.res, { out: true }),
        term.line(' [ 0.  0.  0.  1.]]', T.res, { out: true })];
      term.line('$ python3 fk.py', T.run2 + 0.3, { dur: 0.5 });
      var O2 = [term.line('[{v:0.095 0.109 0.988}]', T.res2, { out: true }),
        term.line('[[-0. -1.  0.]', T.res2 + 0.1, { out: true }),
        term.line(' [ 1.  0.  0.]', T.res2 + 0.1, { out: true }),
        term.line(' [-0.  0.  1.]]', T.res2 + 0.1, { out: true })];

      /* ── 右边：说明。parts = [字, 颜色, 粗细, 等宽]；第 1 页从上面排，第 2、3 页排在终端下面 ── */
      var RK = E.rowKit(), NX = RX + 2;
      function rows(y0, lines) {
        var g = E.el('g', {}, s);
        return { g: g, r: lines.map(function (l) {
          var row = RK.line(g, NX, y0 + l[0], l[1].map(function (p) { return [p[0], p[1], p[2]]; }), l[2] || 18);
          l[1].forEach(function (p, k) { if (p[3]) row.parts[k].style.fontFamily = E.MONO; });
          return row;
        }) };
      }
      var TOP = 44, BOT = 268;
      var nImp = rows(TOP, [[0, [['第 1 句', E.HI, 900]], 20], [34, [['import', E.HI, 700, 1], ['：导入', E.INK2]]], [64, [['numpy', E.INK, 700, 1], ['：库的名字', E.INK2]]],
        [94, [['as np', E.INK, 700, 1], ['：起个短名字', E.INK2]]], [124, [['（第 4 集讲过）', E.MUT, 700]], 17]]);
      var nFrom = rows(TOP, [[0, [['第 2 句', E.HI, 900]], 20], [34, [['from', E.HI, 700, 1], ['：从哪个库', E.INK2]]], [62, [['（Modern Robotics 的库）', E.MUT, 700]], 16],
        [94, [['import', E.HI, 700, 1], ['：导入哪几个', E.INK2]]], [124, [['导入 2 个函数', E.INK, 900]]]]);
      var nFn = rows(TOP, [[0, [['函数', E.HI, 900]], 22], [36, [['起了名字的一段代码', E.INK2]]], [72, [['给它输入', E.INK, 900]]], [102, [['→ 它算好结果', E.INK, 900]]], [132, [['→ 交回来', E.INK, 900]]]]);
      var nVec = rows(TOP, [[0, [['VecTose3', E.WARN, 700, 1]]], [28, [['螺旋轴 × 角度 → 4 × 4', E.INK2]]], [72, [['MatrixExp6', E.HI, 700, 1]]], [100, [['算矩阵指数', E.INK2]]]]);
      var nDef = rows(TOP, [[0, [['def', E.HI, 700, 1], ['：要写一个函数', E.INK2]]], [34, [['fk_space', E.INK, 700, 1], ['：起的名字', E.INK2]]]]);
      var nIn = rows(TOP, [[0, [['括号里：3 样输入', E.INK, 900]]], [34, [['M', E.GOOD, 700, 1], ['：零位时手的位姿', E.INK2]]], [64, [['S', E.WARN, 700, 1], ['：螺旋轴，一排一根', E.INK2]]],
        [94, [['theta', E.ACC, 700, 1], ['：各关节的角度', E.INK2]]]]);
      var nBody = rows(TOP, [[0, [['冒号 ', E.INK2], [':', E.HI, 700, 1], [' 下面', E.INK2]]], [30, [['往里缩进的几句', E.INK, 900]]], [60, [['= 这个函数要做的事', E.INK2]]]]);
      var nEq = rows(TOP, [[0, [['T = M', E.INK, 700, 1], ['：结果先是 M', E.INK2]]], [36, [['等号 ', E.INK2], ['=', E.HI, 700, 1]]], [64, [['不是「两边相等」', E.ERR, 900]]],
        [92, [['是把右边，存进左边', E.INK, 900]]], [122, [['（第 4 集「叫它 p」）', E.MUT, 700]], 17]]);
      var nLoop = rows(TOP, [[0, [['从最后一个往前取', E.INK, 900]]], [32, [['len(theta)', E.INK, 700, 1], [' → 2 个', E.INK2]]], [62, [['range(2)', E.INK, 700, 1], [' → 编号 0、1', E.INK2]]],
        [92, [['reversed', E.INK, 700, 1], [' → 1、0', E.INK2]]], [130, [['for i in …', E.HI, 700, 1]]], [158, [['i 依次是 1、0', E.INK, 900]]],
        [188, [['每个 i：缩进的 2 句', E.INK2]]], [214, [['做一遍', E.INK2]]]]);
      var nCalc = rows(TOP, [[0, [['S[i]', E.WARN, 700, 1], ['：这根螺旋轴', E.INK2]]], [30, [['theta[i]', E.ACC, 700, 1], ['：它的角度', E.INK2]]],
        [60, [['*', E.INK, 700, 1], ['：每个数都乘上', E.INK2]]], [88, [['（第 4 集：数组乘一个数）', E.MUT, 700]], 16],
        [124, [['VecTose3', E.WARN, 700, 1], [' → 4 × 4', E.INK2]]], [154, [['存进 ', E.INK2], ['se3', E.INK, 700, 1]]]]);
      var nExp = rows(TOP, [[0, [['MatrixExp6(se3)', E.HI, 700, 1]]], [28, [['：算出矩阵指数', E.INK2]]], [62, [['@ T', E.INK, 700, 1], ['：乘到结果左边', E.INK2]]],
        [90, [['（@：第 4 集）', E.MUT, 700]], 16], [122, [['T =', E.INK, 700, 1], [' …：存回 T', E.INK2]]]]);
      var nRet = rows(TOP, [[0, [['return T', E.HI, 700, 1]]], [28, [['：把结果交回去', E.INK2]]], [66, [['从最后一个关节往前', E.INK, 900]]], [96, [['= 第 4 幕的顺序', E.GOOD, 900]]]]);
      // 第 2 页
      var n5 = rows(BOT, [[0, [['3 × 3', E.INK2], [' 补 z 轴 → ', E.INK2], ['4 × 4', E.INK, 900]]], [30, [['螺旋轴：各 6 个数', E.WARN, 900]]], [58, [['一根一排', E.INK2]]]]);
      var nPi = rows(BOT, [[0, [['[ ]', E.INK, 700, 1], [' = 列表（第 4 集）', E.INK2]]], [28, [['角度用弧度（第 9 集）', E.INK2]]],
        [60, [['np.pi', E.ACC, 700, 1], [' = 圆周率 π', E.INK2]]], [88, [['≈ 3.14 = 半圈', E.INK2]]], [120, [['90° = π ÷ 2 ≈ 1.57', E.ACC, 900]]]]);
      var nCall = rows(BOT, [[0, [['fk_space(M, S, theta)', E.INK, 700, 1]], 17], [28, [['：3 样输入交给函数', E.INK2]]], [60, [['.round(2)', E.INK, 700, 1], ['：留 2 位小数', E.INK2]]],
        [90, [['print', E.HI, 700, 1], ['：打印出来', E.INK2]]]]);
      var n6 = rows(BOT, [[0, [['手在 ', E.INK2], ['(−4, 1)', E.VEL, 900]]], [28, [['x 轴', E.X, 900], ['朝左，', E.INK2], ['y 轴', E.Y, 900], ['朝下', E.INK2]]], [62, [['和手算的一样 ✓', E.GOOD, 900]]]]);
      var nDot = rows(BOT, [[0, [['-4.', E.HI, 700, 1], ['：数后面的小点', E.INK2]]], [28, [['= 带小数的数', E.INK2]]], [56, [['小数部分是 0', E.INK2]]], [90, [['-4.', E.INK, 700, 1], [' 就是 −4', E.INK, 900]]]]);
      var n7 = rows(BOT, [[0, [['-0.', E.HI, 700, 1], [' 就是 0', E.INK2]]], [28, [['电脑算出的是', E.INK2]]], [54, [['−0.00000000000000022', E.INK2]], 16], [84, [['保留两位小数 → −0', E.INK2]]]]);
      /* ── 第 3 页：4 × 4 的小格子，讲切片：上面、左边写编号 0–3，取到的格子上色 ── */
      var gM = E.el('g', {}, s), GX = RX + 40, GY = 290, CW = 24;
      var cells = [];
      for (var i = 0; i < 4; i++) { cells.push([]); for (var j = 0; j < 4; j++) cells[i].push(K.paint(E.el('rect', { x: GX + CW * j + 1, y: GY + CW * i + 1, width: CW - 2, height: CW - 2, rx: 3 }, gM), E.S2)); }
      var colIdx = [0, 1, 2, 3].map(function (j) { return E.txt(gM, GX + CW * j + CW / 2, GY - 7, String(j), 18, E.MUT, 'middle', 700); });
      var rowIdx = [0, 1, 2, 3].map(function (i) { return E.txt(gM, GX - 9, GY + CW * i + CW / 2 + 6, String(i), 18, E.MUT, 'end', 700); });
      var gLab1 = E.txt(gM, GX + 4 * CW + 10, GY + 1.5 * CW + 6, '位置', 18, E.VEL, 'start', 900);
      var gLab2 = E.txt(gM, GX + 4 * CW + 10, GY + 3 * CW + 22, '朝向', 18, E.OMEGA, 'start', 900);
      var gCap = E.txt(gM, GX - 20, GY + 4 * CW + 22, '编号从 0 开始', 18, E.INK2, 'start', 700);
      var n8 = rows(BOT, [[0, [['《Modern Robotics》例 4.5', E.GOOD, 900]], 17], [28, [['θ = (0, −90°, 0, 0, 90°, 0)', E.ACC, 900]]]]);
      var nKeep = rows(BOT, [[0, [['结果先存进 ', E.INK2], ['T', E.INK, 700, 1]]], [30, [['再分 2 句打印', E.INK2]]]]);
      // 2026-11-05（用户反馈：说「和书上一样」「正是第 6 集的那个矩阵」，就把书上的结果、第 6 集的矩阵摆出来对照）
      var n9 = rows(BOT, [[0, [['电脑：', E.INK2], ['(0.095, 0.109, 0.988)', E.VEL, 900]]], [30, [['《Modern Robotics》例 4.5：', E.GOOD, 900]], 16],
        [56, [['(0.095, 0.109, 0.988) ✓', E.GOOD, 900]]]]);
      var n10 = rows(BOT, [[0, [['第 6 集：原地往左转 90°', E.OMEGA, 900]], 18], [128, [['和终端里的一样（−0 就是 0）', E.INK2]], 16]]);
      var R6m = E.matN(n10.g, NX + 70, BOT + 62, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], { size: 18, dx: 34, dy: 22,
        color: function (i, j) { return j === 0 ? E.X : j === 1 ? E.Y : E.Z; } });

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }
      function win(st, a, b) { return E.fade(st, a, 0.3) * (1 - E.fade(st, b - 0.3, 0.3)); }
      // 一组说明：从 a 出来、到 b 收起；第 k 排在 at[k] 出来
      function show(n, st, a, b, at) { E.op(n.g, win(st, a, b)); (at || []).forEach(function (t0, k) { if (t0 !== null) E.op(n.r[k].g, fr(st, t0)); }); }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) { RK.layout(); laid = true; }
          var f = E.fade;
          ed.draw(st, t); term.draw(st, t);
          E.op(termG, f(st, T.ex - 0.3, 0.3));

          /* 第 1 页：编辑器里的底色（讲到哪段垫哪段） */
          var pg0 = st < T.ex;
          function bx(id, col, a, b) { ed.box(id, col, pg0 ? win(st, a, b) : 0); }
          bx('imp', E.HI, T.imp, T.from); bx('npm', E.HI, T.npW, T.from); bx('asnp', E.HI, T.asnp, T.from);
          bx('from', E.HI, T.from, T.fn); bx('impt', E.HI, T.impt, T.fn);
          ed.box('vec', E.WARN, pg0 ? Math.max(win(st, T.two, T.fn), win(st, T.vec, T.defL)) : 0);
          ed.box('exp', E.HI, pg0 ? Math.max(win(st, T.two, T.fn), win(st, T.exp, T.defL)) : 0);
          bx('def', E.HI, T.defW, T.three); bx('name', E.HI, T.name, T.three);
          bx('M', E.GOOD, T.M, T.colon); bx('S', E.WARN, T.S, T.colon); bx('th', E.ACC, T.th, T.colon);
          bx('colon', E.HI, T.colon, T.tm);
          [4, 5, 6, 7, 8].forEach(function (k) { ed.boxCol(FL[k], 0, 4, E.HI, pg0 ? win(st, T.indent, T.tm) : 0, 'ind' + k); });
          if (pg0) {
            ed.hl(FL[4], E.HI, win(st, T.tm, T.loop));
            ed.hl(FL[5], E.HI, win(st, T.loop, T.calc));
            ed.hl(FL[6], E.HI, win(st, T.calc, T.calc2));
            ed.hl(FL[7], E.HI, win(st, T.calc2, T.done));
            ed.hl(FL[8], E.HI, win(st, T.done, T.order));
          }
          bx('tm', E.HI, T.eq, T.loop);
          bx('ln', E.HI, T.ln, T.rg); bx('rg', E.HI, T.rg, T.rv); bx('i', E.HI, T.iseq, T.calc);
          bx('si', E.WARN, T.si, T.vt); bx('ti', E.ACC, T.ti, T.vt); bx('star', E.HI, T.star, T.vt); bx('vt', E.WARN, T.vt, T.calc2); bx('se3', E.HI, T.x44b, T.calc2);
          bx('left', E.HI, T.me, T.storeB); bx('tl', E.HI, T.storeB, T.done);
          bx('ret', E.HI, T.ret, T.order);
          var revOrd = st >= T.order - 0.3;   // 讲 reversed 时亮黄；讲「第 4 幕的顺序」时换绿
          ed.box('rev', revOrd ? E.GOOD : E.HI, pg0 ? (revOrd ? win(st, T.order, T.ex) : win(st, T.rv, T.whole)) : 0);

          /* 第 2 页 */
          var pg1 = st >= T.ex && st < T.ur;
          ed.box('z', E.Z, pg1 ? E.bump(st, T.z, 1.0) : 0);
          if (pg1) { ed.hl(L_S1, E.WARN, win(st, T.row1, T.two90)); ed.hl(L_S2, E.WARN, win(st, T.row1, T.two90)); }
          ed.boxCol(L_TH, 8, 22, E.HI, pg1 ? win(st, T.brk, T.rad) : 0, 'list');
          ed.box('pi', E.ACC, pg1 ? win(st, T.pi, T.last) : 0);
          ed.box('call', E.HI, pg1 ? win(st, T.call, T.round) : 0);
          ed.box('rd', E.HI, pg1 ? win(st, T.round, T.prn) : 0);
          ed.box('prn', E.HI, pg1 ? win(st, T.prn, T.run) : 0);
          // 终端里的框
          term.box('o03', E.VEL, win(st, T.p4, T.dot));
          term.box('o13', E.VEL, win(st, T.p1, T.dot));
          term.boxCol(O1[0], 2, 3, E.X, win(st, T.cx, T.dot));
          term.boxCol(O1[1], 2, 3, E.X, win(st, T.cx, T.dot));
          term.boxCol(O1[0], 6, 3, E.Y, Math.max(win(st, T.cy, T.dot), win(st, T.neg0, T.ur)));
          term.boxCol(O1[1], 6, 3, E.Y, win(st, T.cy, T.dot));
          term.boxCol(O1[0], 14, 3, E.HI, win(st, T.dec, T.neg0), 'm4');   // 「负四点」整个数
          term.boxCol(O1[0], 16, 1, E.HI, win(st, T.dotW, T.neg0), 'dot4'); term.boxCol(O1[1], 16, 1, E.HI, win(st, T.dotW, T.neg0), 'dot1');   // 数后面的小点
          term.boxCol(O2[0], 1, 17, E.VEL, win(st, T.p95, T.rot));
          [1, 2, 3].forEach(function (k) { term.boxCol(O2[k], 1, 13, E.OMEGA, fr(st, T.rot)); });

          /* 说明 */
          show(nImp, st, T.four - 0.3, T.from, [T.imp, T.imp, T.npW, T.asnp, T.asnp + 0.3]);
          show(nFrom, st, T.from - 0.3, T.fn, [T.from, T.from, T.lib, T.impt, T.two]);
          show(nFn, st, T.fn, T.these, [T.fn, T.named, T.input, T.calc0, T.back0]);
          show(nVec, st, T.vec, T.defL, [T.vec, T.vec, T.exp, T.exp]);
          show(nDef, st, T.defW, T.three, [T.defW, T.name]);
          show(nIn, st, T.three, T.colon, [T.three, T.M, T.S, T.th]);
          show(nBody, st, T.colon, T.tm, [T.colon, T.indent, T.job]);
          show(nEq, st, T.tm, T.loop, [T.tm, T.eq, T.notEq, T.store, T.ep4]);
          show(nLoop, st, T.back, T.calc, [T.back, T.ln, T.rg, T.rv, T.whole, T.iseq, T.each, T.each]);
          E.op(nLoop.r[1].parts[1], fr(st, T.ln2)); E.op(nLoop.r[2].parts[1], fr(st, T.rg2)); E.op(nLoop.r[3].parts[1], fr(st, T.rv3));
          show(nCalc, st, T.si, T.calc2, [T.si, T.ti, T.star, T.star, T.vt, T.x44b]);
          show(nExp, st, T.me, T.done, [T.me, T.me2, T.at, T.at, T.storeB]);
          E.op(nExp.r[2].parts[1], fr(st, T.left));
          show(nRet, st, T.ret, T.ex, [T.ret, T.ret, T.order, T.ep4m]);
          show(n5, st, T.m33, T.two90, [T.m33, T.nums, T.row1]);
          E.op(n5.r[0].parts[1], fr(st, T.z)); E.op(n5.r[0].parts[2], fr(st, T.m44));
          show(nPi, st, T.brk, T.last, [T.brk, T.rad, T.pi, T.p314, T.div2]);
          E.op(nPi.r[0].parts[1], fr(st, T.list)); E.op(nPi.r[3].g, fr(st, T.p314)); E.op(nPi.r[4].g, fr(st, T.div2));
          show(nCall, st, T.call, T.run, [T.call, T.fnc, T.round, T.prn]);
          show(n6, st, T.p4, T.dot, [T.p4, T.cx, T.same2]);
          E.op(n6.r[1].parts[2], fr(st, T.cy)); E.op(n6.r[1].parts[3], fr(st, T.cy));
          show(nDot, st, T.dotW, T.neg0, [T.dotW, T.dec, T.dec0, T.m4]);
          show(n7, st, T.neg0, T.ur, [T.neg0, T.tiny, T.near, T.round0]);
          show(n8, st, T.ur, T.store2, [T.ur, T.ang]);
          show(nKeep, st, T.fnRes, T.part, [T.keepW, T.pr]);
          if (st >= T.ur) ed.hl(L_T2, E.HI, win(st, T.fnRes, T.part));

          /* 第 3 页：小格子 */
          E.op(gM, win(st, T.part, T.run2));
          var selR = win(st, T.rowW, T.colsel) + win(st, T.head3, T.lone3);
          var selC = win(st, T.colW, T.c3) + win(st, T.idx3, T.l1);
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
          E.op(gLab1, fr(st, T.posW)); E.op(gLab2, fr(st, T.rotW)); E.op(gCap, win(st, T.py0, T.l1));
          // 打印两句：整句垫底色；方括号整段、横排切片、列切片、保留几位小数
          if (st >= T.ur && st < T.run2) {
            ed.hl(L_P1, E.VEL, win(st, T.l1, T.l2));
            ed.hl(L_P2, E.OMEGA, win(st, T.l2, T.run2));
          }
          ed.box('r1', E.HI, st < T.run2 ? win(st, T.rowsel, T.colsel) + win(st, T.c3, T.lone3) : 0);
          ed.box('c1', E.VEL, st < T.run2 ? win(st, T.colsel, T.c3) + win(st, T.lone3, T.l1) + win(st, T.right, T.keep3) : 0);
          ed.box('d1', E.INK2, st < T.run2 ? win(st, T.keep3, T.l2) : 0);
          ed.box('r2', E.HI, st < T.run2 ? win(st, T.tl, T.keep2) : 0);
          ed.box('c2', E.OMEGA, st < T.run2 ? win(st, T.tl, T.keep2) : 0);
          ed.box('d2', E.INK2, st < T.run2 ? win(st, T.keep2, T.run2) : 0);
          E.op(n9.g, win(st, T.p95, T.rot)); E.op(n9.r[1].g, fr(st, T.same3)); E.op(n9.r[2].g, fr(st, T.same3));
          E.op(n10.g, fr(st, T.rot)); E.op(n10.r[0].g, fr(st, T.ep6)); E.op(R6m.g, fr(st, T.left90)); E.op(n10.r[1].g, fr(st, T.left90 + 0.6));
        }
      };
    }
  };
})();
