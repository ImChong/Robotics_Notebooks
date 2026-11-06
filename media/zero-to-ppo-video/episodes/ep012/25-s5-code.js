// 012 第 5 幕：写成代码（2026-11-06 用户要求「逆运动学也加一个写代码示意的部分」；同日用户要求「代码示意参考正运动学」「所有信息确保都有一手资料支撑」：
// 后半段接第 11 集照《Modern Robotics》官方库 FKinSpace 写的 fk_space 验算，零位取胳膊伸直平放在 x 轴上，两组角度放回去手都在 (1, 0)）。照 011 第 6 幕：左边编辑器 ik.py，右边说明，运行时右上出终端。
// 手放在 x 轴上 (d, 0)（第 3 幕 d = 1、第 4 幕 d = 5），代码就是第 3、4 幕的手算：两式相减求 x，y² = 5 − x²，y² < 0 没有解，开根号给肘上，
// atan2 算方向的角度、degrees 换成多少度，肘下正负反过来。只用 Python 自带的 math 库（sqrt、atan2、degrees）和内置 round；
// 输出按 Python 3.13 实跑：print(ik(1)) → [[63.4, -153.4], [-63.4, 153.4]]，print(ik(5)) → []（见 ../ep012.py）。
// 011 讲过的写法（from … import、def、括号里的输入、冒号下面缩进、等号「存进」、return、print、方括号 = 第 4 集的列表）注明第 11 集；
// 新写法当场讲：* 乘号、/ 除号、if（如果）、< 小于、[] 空的方括号、round(数, 1) 保留 1 位小数、外面的方括号装着两组。
// 代码按念到的词一段一段打出来（keys：每个字各自的时刻）；讲到哪段、编辑器里哪段垫底色，右边同步写说明。
// 时刻取自配音的逐词时间，见 ../ep012.py。
(function () {
  var E = window.EP012, K = E.K;
  var T = {
    code: 2.79, arm0: 3.66, xax: 5.35, right: 6.11, dW: 6.48, up0: 7.03, ep3: 8.33, ep4: 9.9,
    l1: 12.5, imp: 14.23, math: 16.97, lib: 17.87, three: 18.93,
    sqrtW: 21.41, atanW: 22.82, angle: 25.08, degW: 26.27, rad: 28.2, deg: 28.79,
    defW: 32.53, name: 33.67, inp: 34.6, dIn: 36.37,
    xl: 39.43, elbowX: 40.13, eqs: 41.52, sub: 43.31, rhs: 44.53, lhs: 46.14, rem: 47.05,
    add: 51.53, div: 52.83, getx: 54.26, star: 56.12, slash: 57.5,
    yyL: 59.94, yy2: 60.66, back3: 63.43, name2: 65.52,
    newW: 68.28, ifW: 69.19, ifMean: 70.74, cond: 71.78, lt: 73.9,
    okC: 76.59, colon: 77.92, indent: 78.88, ret0: 80.42, empty: 80.86, none: 82.62, ep4b: 84.26,
    notC: 87.85, down: 89.33, sq: 89.98, pos: 91.82, up: 94.01,
    a1W: 96.06, args: 97.14, fnA: 98.8, order: 100.49, rad2: 102.63, deg2: 103.75,
    a2W: 106.04, fore: 106.98, dx: 108, dy: 109.34, ang2: 110.5, minus: 112.06, same3: 113.6,
    retW: 115.81, upG: 117.9, dnG: 119.4, sign: 121.01, outer: 123.16,
    rnd: 127.71, rnd1: 129.57, oneD: 131.23,
    prL: 133.9, one5: 134.9, prn: 137.12,
    run: 138.94, r1: 139.69, two: 140.94, upA: 141.93, dnA: 144.9, same: 147.83,
    r5: 150.42, empty2: 151.8, none2: 153.72, same4: 155,
    // 第 2 页：接 011 的 fk_space 验算
    chk: 157.06, fkW: 158.26, back: 161.79, look: 162.55, at10: 164.3,
    page2: 165.99, npW: 168.04, mrW: 169.09, two2: 170.39, fkFn: 171.29, same0: 173.99,
    zero: 175.99, straight: 177.08, flat: 177.96, handM: 179.24, sq5: 181.12, approx: 181.72, far: 182.87,
    from0: 185.9, shAng: 187.39, xAng: 188.48, elAng: 190.66, rel: 191.72, match: 194.15,
    sW: 197.35, s1: 198.42, s1b: 199.32, s2: 200.49, alg: 202.22, v3: 203.81,
    unp: 207.45, unp2: 208.92, sep: 210.61, upS: 211.9, dnS: 212.86,
    rad: 214.87, radW: 215.85, fkC: 216.87, store: 218.86,
    last: 221.3, ep11s: 222.3, brk: 223.68, rowSel: 224.63, colSel: 226.37, hpos: 227.62, two3: 228.68, rr: 229.79, rd2: 231.11, both: 232.57,
    run2: 234.64, up2: 235.28, dn2: 235.96, hand10: 236.98, neg0: 238.68,
    why: 240.89, off: 242.95, milli: 245.68, rd3: 247.1
  };
  var END = 250.29;

  // 每个字的出现时刻：spans = [[子串, 第几处（从 0 起）, t0, t1], …]，子串的字在 t0…t1 之间匀速打出；没列到的字在 dflt 出现
  function keysOf(s, spans, dflt) {
    var plain = s.replace(/\{\w(?:\.\w+)?:([^}]*)\}/g, '$1').replace(/^(>>> |\.\.\. |\$ )/, ''), ks = [];
    for (var i = 0; i < plain.length; i++) ks.push(dflt);
    spans.forEach(function (sp) {
      var at = -1;
      for (var n = 0; n <= sp[1]; n++) at = plain.indexOf(sp[0], at + 1);
      if (at < 0) throw new Error('keysOf: 找不到「' + sp[0] + '」');
      for (var k = 0; k < sp[0].length; k++) ks[at + k] = sp[2] + (sp[3] - sp[2]) * (k + 0.5) / sp[0].length;
    });
    return ks;
  }
  function colOf(s, sub, nth) {
    var plain = s.replace(/\{\w(?:\.\w+)?:([^}]*)\}/g, '$1'), at = -1;
    for (var n = 0; n <= (nth || 0); n++) at = plain.indexOf(sub, at + 1);
    return at;
  }

  E.scenes[4] = {
    title: '写成代码',
    dur: END,
    cues: [
      { at: 0, s: '把手算写成代码：手放在 x 轴上 $(d,\\,0)$，第 3 幕 $d=1$，第 4 幕 $d=5$' },
      { at: 12.41, s: '第 1 句：`from` 库 `import` 函数（第 11 集）——Python 自带的数学库 `math`，导入 3 个函数' },
      { at: 20.71, s: '`sqrt` 开根号；`atan2` 算方向的**角度**（第 3 幕）；`degrees` 把弧度换成多少度（第 9 集）' },
      { at: 30.84, s: '`def` 写一个函数（第 11 集），名字 `ik`，输入 $d$：手往右的格数' },
      { at: 38.71, s: '两个式子相减：右边 $5-4=1$，左边 $2dx-d^2$' },
      { at: 50.95, s: '$x=(5-4+d^2)\\div 2d$；代码里乘号是 `*`，除号是 `/`' },
      { at: 59.85, s: '`yy` 存 $y^2=5-x^2$（第 3 幕放回第 1 个式子）' },
      { at: 67.67, s: '`if`：如果。条件 `yy < 0`，`<` 读作小于' },
      { at: 76.5, s: '条件成立才做缩进的那句：`return []` 交回空的方括号——**没有**解（第 4 幕）' },
      { at: 87.76, s: '不成立就往下：`sqrt(yy)` **开根号**，给正的那个，就是**肘上**' },
      { at: 95.97, s: '大臂的**角度**：`atan2(y, x)` 先写往上、再写往右，算出弧度，`degrees` 换成多少度' },
      { at: 105.95, s: '肘的**角度**：小臂方向（往右 $d-x$，往上 $-y$）的角度，减去大臂的角度' },
      { at: 115.72, s: '交回两组：**肘上** `[a1, a2]`，**肘下**正负反过来 `[-a1, -a2]`' },
      { at: 126.54, s: '`round(…, 1)`：保留 1 位小数' },
      { at: 133.81, s: '把 1 和 5 分别交给函数，`print` 打印出来' },
      { at: 138.85, s: '手在 $(1,\\,0)$：**肘上** $63.4^\\circ$、$-153.4^\\circ$，**肘下** $-63.4^\\circ$、$153.4^\\circ$——和第 3 幕手算的一样' },
      { at: 150.32, s: '手在 $(5,\\,0)$：`[]`，一组解也**没有**——和第 4 幕一样' },
      { at: 156.97, s: '验算：把两组**角度**放回第 11 集的**正向运动学**，看手是不是停在 $(1,\\,0)$' },
      { at: 165.99, s: '导入 NumPy 和 `modern_robotics` 的 2 个函数；`fk_space` 原样用第 11 集的（照《Modern Robotics》官方库 `FKinSpace` 写的）' },
      { at: 175.9, s: '这次的零位：胳膊伸直平放在 x 轴上，手在 $\\sqrt5+2\\approx 4.24$ 格（第 4 幕最远的那一点）' },
      { at: 185.81, s: '从这个零位算起：肩转的就是大臂和 x 轴的夹角，肘转的就是小臂相对大臂转了多少——正好是 `ik` 算出的两个**角度**' },
      { at: 197.26, s: '螺旋轴：肩在原点（同第 11 集）；肘在 $(\\sqrt5,\\,0)$，照第 9 集算，后 3 个数是 $(0,\\,-\\sqrt5,\\,0)$' },
      { at: 207.36, s: '`up, down = ik(1)`：左边 2 个名字，交回的 2 组分别存进去' },
      { at: 214.78, s: '`np.radians` 把度换回弧度，交给 `fk_space`，结果存进 `T1`、`T2`' },
      { at: 221.21, s: '`T1[:2, 3]`：位置那一列的前 2 个数（往右、往上），保留 2 位小数，2 个一起打印' },
      { at: 234.55, s: '**肘上**、**肘下**：手都停在 $(1,\\,0)$（$-0.$ 就是 $0$）✓' },
      { at: 240.79, s: '角度只保留了 1 位小数，位置离 $(1,\\,0)$ 约 $0.001$；保留 2 位小数就是 $(1,\\,0)$' }
    ],
    build: function (K) {
      var s = K.sceneSvg('把手算写成代码 ik.py：手放在 x 轴上 (d, 0)；from math import sqrt, atan2, degrees 导入开根号、算方向角度、弧度换成多少度三个函数；def ik(d) 写一个函数；x = (5 − 4 + d·d) / (2·d) 是两式相减的结果，* 是乘号、/ 是除号；yy = 5 − x·x 是 y 的平方；if yy < 0: return [] 平方不会是负的，没有解；y = sqrt(yy) 开平方给正的、就是肘上；a1 = degrees(atan2(y, x)) 大臂的角度，atan2 先写往上再写往右；a2 = degrees(atan2(−y, d − x)) − a1 肘的角度；交回 [[a1, a2], [−a1, −a2]] 肘上、肘下两组，round 保留 1 位小数；print(ik(1)) 打印 [[63.4, -153.4], [-63.4, 153.4]]，和第 3 幕手算的一样；print(ik(5)) 打印 []，没有解，和第 4 幕一样');

      /* ── 句 1（2026-11-06 用户要求「说手放在 x 轴上时要显示一下对应的机械臂画面」）：左边先画第 3、4 幕的网格和 UR5，
         「x 轴上」x 轴亮一下，「往右 d 格，往上零格」x 轴上出目标 (d, 0)、量出 d 格，「第三幕的一、零」目标挪到 (1, 0)、UR5 摆成肘上那组，
         「第四幕的五、零」目标挪到 (5, 0)、UR5 伸直也只到 4.24 格、剩下一段橙色打 ×；第 2 句起换成编辑器 ── */
      var AOX = 70, AOY = 214;
      var armPanel = E.el('g', {}, s);
      var under = E.el('g', {}, armPanel);
      var armG = E.el('g', {}, armPanel);
      var arm = E.arm2(armG, AOX, AOY, E.ARM.S, 0, [10, 14, 440, 392]), au = arm.u;
      var pl = E.plane(under, { ox: AOX, oy: AOY, u: au, x0: -1, x1: 6, y0: -3, y1: 3 }), P = pl.P, O = P(0, 0);
      var atop = E.el('g', {}, armPanel);
      var xHi = K.paint(E.el('path', { d: 'M ' + P(-1, 0)[0] + ' ' + O[1] + ' H ' + P(6, 0)[0], fill: 'none', 'stroke-width': 7, 'stroke-linecap': 'round' }, atop), null, E.TGT);
      var dDim = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3, 'stroke-linecap': 'round' }, atop), null, E.X);
      var dLab = E.halo(E.txt(atop, 0, 0, 'd 格', 20, E.X, 'middle', 900));
      // 到不了 (5, 0)：伸直的胳膊盖住了 x 轴上这一段，画到量尺下面：最远 4.24 到 5 之间橙色、打 ×，写「到不了」
      var GY2 = O[1] + 1.6 * au, gapG = E.el('g', {}, atop);
      K.paint(E.el('path', { d: 'M ' + P(E.L1 + E.L2, 0)[0] + ' ' + (GY2 - 8) + ' V ' + (GY2 + 8) + ' M ' + O[0] + ' ' + GY2 + ' H ' + P(E.L1 + E.L2, 0)[0], fill: 'none', 'stroke-width': 2.5 }, gapG), null, E.INK2);
      E.halo(E.txt(gapG, (O[0] + P(E.L1 + E.L2, 0)[0]) / 2, GY2 + 24, '伸直最远 4.24 格', 18, E.INK2, 'middle', 900));
      K.paint(E.el('path', { d: 'M ' + P(E.L1 + E.L2, 0)[0] + ' ' + GY2 + ' H ' + P(5, 0)[0], fill: 'none', 'stroke-width': 6, 'stroke-linecap': 'round' }, gapG), null, E.ERR);
      E.cross(gapG, P(4.62, 0)[0], GY2 - 18, 7, E.ERR, 4);
      E.halo(E.txt(gapG, P(5, 0)[0] + 12, GY2 + 6, '到不了', 18, E.ERR, 'start', 900));
      var tgA = E.target(atop, 0, 0, 11, E.TGT), tgA2 = E.target(atop, 0, 0, 11, E.ERR);
      var tLab = E.halo(E.txt(atop, 0, 0, '(d, 0)', 20, E.TGT, 'middle', 900));
      var upLab = E.halo(E.txt(atop, 0, 0, '往上 0 格', 18, E.Y, 'middle', 900));
      var UPA = E.ik2(1, 0, true), Z0 = [Math.atan2(2, 1), 0];

      /* ── 左：编辑器（15 句，一屏放得下；第 2 句起出来） ── */
      var edG = E.el('g', {}, s);
      var ed = E.code(edG, 10, 14, 440, 392, { title: 'ik.py', size: 16, lh: 22 });
      function lineIn(e, str, spans, dflt) { return e.line(str, dflt, { keys: keysOf(str, spans, dflt) }); }
      function line(str, spans, dflt) { return lineIn(ed, str, spans, dflt); }
      var S0 = '{n.from:from} {n.math:math} {h.imp:import} {w.sqrt:sqrt}, {a.atan:atan2}, {a.deg:degrees}';
      var L0 = line(S0, [['from math import', 0, T.imp, T.imp + 1.0], [' sqrt, atan2, degrees', 0, T.three, T.three + 0.8]], T.imp);
      ed.line('', T.three + 0.9, { dur: 0 });
      var S2 = '{h.def:def} {n.nm:ik}({v.din:d}){n.c0::}';
      var L2 = line(S2, [['def ', 0, T.defW, T.defW + 0.3], ['ik', 0, T.name, T.name + 0.25], ['(', 0, T.inp, T.inp + 0.1], ['d', 1, T.dIn, T.dIn + 0.1], ['):', 0, T.dIn + 0.2, T.dIn + 0.4]], T.defW);
      var S3 = '    {h.xl:x} = (5 - 4 + d {n.st1:*} d) {n.sl:/} (2 {n.st2:*} d)';
      var L3 = line(S3, [['    x = ', 0, T.elbowX, T.elbowX + 0.3], ['(5 - 4 + d * d)', 0, T.add, T.add + 0.9], [' / (2 * d)', 0, T.div, T.div + 0.9]], T.elbowX);
      var S4 = '    {h.yy:yy} = 5 - x * x';
      var L4 = line(S4, [['    yy = 5 - x * x', 0, T.yy2, T.yy2 + 1.0]], T.yy2);
      var S5 = '    {h.if:if} {n.c1:yy} {h.lt:<} {n.c2:0}{n.c3::}';
      var L5 = line(S5, [['if', 0, T.ifW, T.ifW + 0.25], [' yy < 0:', 0, T.cond, T.cond + 1.2]], T.ifW);
      var S6 = '        {b.r0:return []}';
      var L6 = line(S6, [['return []', 0, T.ret0, T.ret0 + 0.9]], T.ret0);
      var S7 = '    y = {w.sq:sqrt(yy)}';
      var L7 = line(S7, [['    y = sqrt(yy)', 0, T.sq, T.sq + 0.8]], T.sq);
      var S8 = '    {h.a1:a1} = {a.dg1:degrees(}{a.at1:atan2(y, x)}{a.dg1e:)}';
      var L8 = line(S8, [['    a1 = ', 0, T.a1W, T.a1W + 0.3], ['atan2(y, x)', 0, T.args, T.args + 1.5], ['degrees(', 0, T.deg2, T.deg2 + 0.5], [')', 1, T.deg2 + 0.55, T.deg2 + 0.6]], T.a1W);
      var S9 = '    {h.a2:a2} = {a.dg2:degrees(}{a.at2:atan2(-y, d - x)}{a.dg2e:)}{n.m1: - a1}';
      var L9 = line(S9, [['    a2 = ', 0, T.a2W, T.a2W + 0.3], ['atan2(-y, d - x)', 0, T.dx, T.dx + 2.0], ['degrees(', 0, T.ang2, T.ang2 + 0.45], [')', 1, T.ang2 + 0.5, T.ang2 + 0.55], [' - a1', 0, T.minus, T.minus + 0.5]], T.a2W);
      var S10 = '    return [{h.up:[round(a1, 1), round(a2, 1)]},';
      // 交回两组：整句一次打出来，讲到「保留小数」再给 round(…, 1) 垫底色（原先等到那时才补上 round(、, 1)，之前代码里空着一大块）
      var L10 = line(S10, [['    return [', 0, T.retW, T.retW + 0.5], ['[round(a1, 1), round(a2, 1)],', 0, T.upG, T.upG + 1.0]], T.upG);
      var S11 = '            {c.dn:[round(-a1, 1), round(-a2, 1)]}]';
      var L11 = line(S11, [['            [round(-a1, 1), round(-a2, 1)]]', 0, T.dnG, T.dnG + 1.1]], T.dnG);
      ed.line('', T.prL, { dur: 0 });
      var S13 = 'print({n.ik1:ik(1)})', S14 = 'print({n.ik5:ik(5)})';
      var L13 = line(S13, [['ik(1)', 0, T.one5 + 0.1, T.one5 + 0.4], ['print(', 0, T.prn, T.prn + 0.3], [')', 1, T.prn + 0.35, T.prn + 0.4]], T.one5);
      var L14 = line(S14, [['ik(5)', 0, T.one5 + 0.5, T.one5 + 0.8], ['print(', 0, T.prn + 0.45, T.prn + 0.75], [')', 1, T.prn + 0.8, T.prn + 0.85]], T.one5 + 0.5);

      /* ── 第 2 页：接着写验算（同一个 ik.py 往下写；最长一句 48 个字，换 14 号字的编辑器；20 句，最后滚上去 2 排） ── */
      var ed2G = E.el('g', {}, s);
      var ed2 = E.code(ed2G, 10, 14, 440, 392, { title: 'ik.py（接着写）', size: 14, lh: 19 });
      function l2(str, spans, dflt) { return lineIn(ed2, str, spans, dflt); }
      var P0 = l2('{h.imp2:import} {n.np:numpy as np}', [['import numpy as np', 0, T.npW, T.npW + 0.6]], T.npW);
      var P1 = l2('{h.from2:from} {n.mr:modern_robotics} {h.imp3:import} {h.exp:MatrixExp6}, {w.vec:VecTose3}', [['from modern_robotics import MatrixExp6, VecTose3', 0, T.mrW, T.mrW + 1.6]], T.mrW);
      ed2.line('', T.two2 + 0.6, { dur: 0 });
      var FUN = ['def fk_space(M, S, theta):', '    T = M', '    for i in reversed(range(len(theta))):', '        se3 = VecTose3(S[i] * theta[i])', '        T = MatrixExp6(se3) @ T', '    return T'];
      var PF = FUN.map(function (l) { return ed2.line(l, T.fkFn, { dur: 0 }); });   // 第 11 集写的函数：原样一下子出来
      ed2.line('', T.fkFn, { dur: 0 });
      var SM0 = 'M = np.array([[1, 0, 0, {h.m03:np.sqrt(5) + 2}],';
      var PM = [l2(SM0, [['M = np.array([[1, 0, 0, ', 0, T.zero, T.zero + 0.6], ['np.sqrt(5) + 2', 0, T.sq5, T.sq5 + 0.9], ['],', 0, T.approx, T.approx + 0.1]], T.zero),
        l2('              [0, 1, 0, 0],', [], T.approx + 0.2), l2('              [0, 0, 1, 0],', [], T.approx + 0.3), l2('              [0, 0, 0, 1]])', [], T.approx + 0.4)];
      var PS = [l2('S = np.array([[{w.s1:0, 0, 1, 0, 0, 0}],', [['S = np.array([[', 0, T.sW, T.sW + 0.4], ['0, 0, 1, 0, 0, 0', 0, T.s1, T.s1 + 0.6], ['],', 0, T.s1 + 0.7, T.s1 + 0.75]], T.sW),
        l2('              [{w.s2:0, 0, 1, 0, }{h.v3:-np.sqrt(5)}{w.s2b:, 0}]])', [['              [0, 0, 1, ', 0, T.s2, T.s2 + 0.5], ['0, -np.sqrt(5), 0', 0, T.v3, T.v3 + 1.4], [']])', 0, T.v3 + 1.5, T.v3 + 1.6]], T.s2)];
      var SU = '{h.ud:up, down} = {n.ik1b:ik(1)}';
      var PU = l2(SU, [['up, down', 0, T.unp, T.unp + 0.5], [' = ik(1)', 0, T.unp2, T.unp2 + 0.5]], T.unp);
      var ST1 = 'T1 = {n.fk1:fk_space}(M, S, {a.r1:np.radians}(up))', ST2 = 'T2 = {n.fk2:fk_space}(M, S, {a.r2:np.radians}(down))';
      var PT1 = l2(ST1, [['T1 = fk_space(M, S, np.radians(up))', 0, T.rad, T.rad + 1.0]], T.rad);
      var PT2 = l2(ST2, [['T2 = fk_space(M, S, np.radians(down))', 0, T.rad + 1.0, T.rad + 2.0]], T.rad + 1.0);
      var SP = 'print(T1{h.sl1:[:2, 3]}{i.rd1:.round(2)}, T2{h.sl2:[:2, 3]}{i.rd2:.round(2)})';
      var PP = l2(SP, [['print(T1', 0, T.last, T.last + 0.4], ['[:2, 3]', 0, T.hpos, T.hpos + 0.5], ['.round(2)', 0, T.rd2, T.rd2 + 0.4],
        [', T2[:2, 3].round(2))', 0, T.both, T.both + 0.9]], T.last);

      /* ── 右上：终端（运行时出来） ── */
      var RX = 460, RW = 330;
      var termG = E.el('g', {}, s);
      var term = E.code(termG, RX, 14, RW, 140, { title: '终端', size: 16, lh: 22 });   // 4 排：第 2 页运行时 1 句命令 + 3 句输出
      term.line('$ python3 ik.py', T.run + 0.2, { dur: 0.5 });
      var O1 = term.line('[{h.o1:[63.4, -153.4]}, {c.o2:[-63.4, 153.4]}]', T.r1 - 0.2, { out: true });
      var O2 = term.line('{b.o3:[]}', T.r1 - 0.2, { out: true });
      // 第 2 页：同一个文件再运行一次，三句打印都出来
      term.line('$ python3 ik.py', T.run2 + 0.1, { dur: 0.4, page: 1 });
      term.line('[[63.4, -153.4], [-63.4, 153.4]]', T.up2 - 0.25, { out: true, page: 1 });
      term.line('[]', T.up2 - 0.25, { out: true, page: 1 });
      term.line('{h.q1:[ 1. -0.]} {c.q2:[1. 0.]}', T.up2 - 0.25, { out: true, page: 1 });

      /* ── 右边：说明。parts = [字, 颜色, 粗细, 等宽]；运行前从上面排，运行后排在终端下面 ── */
      var RK = E.rowKit(), NX = RX + 2;
      function rows(y0, ls) {
        var g = E.el('g', {}, s);
        return { g: g, r: ls.map(function (l) {
          if (l[3] === 'tex') return { g: E.tex(g, NX, y0 + l[0], l[1], l[2] || 19, E.INK2, 'start', { w: 326, h: 34 }), parts: [] };   // 整行公式（KaTeX，根号用 \\sqrt）
          var row = RK.line(g, NX, y0 + l[0], l[1].map(function (p) { return [p[0], p[1], p[2]]; }), l[2] || 18);
          l[1].forEach(function (p, k) { if (p[3]) row.parts[k].style.fontFamily = E.MONO; });
          return row;
        }) };
      }
      var TOP = 44, BOT = 186, TOP2 = 232;   // TOP2：第 2 页右上有小图时，说明排在图下面
      var n1 = rows(TOP, [[0, [['写成代码', E.HI, 900]], 22], [40, [['手放在 x 轴上：', E.INK2], ['(d, 0)', E.TGT, 900]]],
        [74, [['第 3 幕：', E.INK2], ['d = 1', E.TGT, 900]]], [104, [['第 4 幕：', E.INK2], ['d = 5', E.TGT, 900]]]]);
      var n2 = rows(TOP, [[0, [['第 1 句（第 11 集）', E.HI, 900]], 20], [34, [['from math', E.INK, 700, 1], ['：Python 自带的', E.INK2]], 17],
        [60, [['数学库', E.INK2]], 17], [92, [['import', E.HI, 700, 1], ['：导入 3 个函数', E.INK2]], 17],
        [130, [['sqrt', E.WARN, 700, 1], ['：开根号', E.INK2]]], [160, [['atan2', E.ACC, 700, 1], ['：方向的角度（第 3 幕）', E.INK2]], 17],
        [190, [['degrees', E.ACC, 700, 1], ['：弧度 → 多少度', E.INK2]], 17], [218, [['（弧度：第 9 集）', E.MUT, 700]], 16]]);
      var n3 = rows(TOP, [[0, [['def', E.HI, 700, 1], ['：写一个函数', E.INK2]]], [28, [['（第 11 集）', E.MUT, 700]], 16],
        [62, [['ik', E.INK, 700, 1], ['：起的名字', E.INK2]]], [94, [['括号里的输入 ', E.INK2], ['d', E.TGT, 700, 1]]], [124, [['= 手往右的格数', E.INK2]]]]);
      var n4 = rows(TOP, [[0, [['第 1 句：肘往右几格 x', E.INK, 900]], 19], [36, [['x² + y² = 5', E.INK, 700]]], [64, [['(x − d)² + y² = 4', E.INK, 700]]],
        [98, [['相减：右边 ', E.INK2], ['5 − 4 = 1', E.INK, 900]]], [128, [['左边拆开：', E.INK2], ['2dx − d²', E.INK, 900]]],
        [164, [['→ x = (5 − 4 + d²) ÷ 2d', E.HI, 900]]], [204, [['*', E.HI, 700, 1], ['：乘号　', E.INK2], ['/', E.HI, 700, 1], ['：除号', E.INK2]]]]);
      var n5 = rows(TOP, [[0, [['第 2 句：', E.INK, 900], ['y² = 5 − x²', E.INK, 900]], 19], [32, [['（第 3 幕：放回第 1 个式子）', E.MUT, 700]], 16],
        [66, [['yy', E.HI, 700, 1], ['：名字写成两个 y', E.INK2]]]]);
      var n6 = rows(TOP, [[0, [['新写法：', E.INK, 900], ['if', E.HI, 700, 1], [' 如果', E.INK2]], 19], [34, [['条件：', E.INK2], ['yy < 0', E.INK, 700, 1]]],
        [64, [['<', E.HI, 700, 1], ['：小于', E.INK2]]], [100, [['成立 → 做冒号下面', E.INK2]]], [128, [['往里缩进的那句', E.INK2]]],
        [164, [['return []', E.ERR, 700, 1], ['：交回空的', E.INK2]], 17], [190, [['方括号', E.INK2]], 17], [222, [['= 一组解也没有（第 4 幕）', E.ERR, 900]], 17]]);
      var n7 = rows(TOP, [[0, [['不成立 → 往下', E.INK2]]], [34, [['sqrt(yy)', E.WARN, 700, 1], ['：开根号 → y', E.INK2]]],
        [64, [['根号给正的那个 = ', E.INK2], ['肘上', E.UP, 900]]]]);
      var n8 = rows(TOP, [[0, [['a1', E.HI, 700, 1], ['：大臂的角度', E.INK2]]], [34, [['atan2(y, x)', E.ACC, 700, 1]]],
        [62, [['先写往上 y，再写往右 x', E.INK, 900]], 17], [98, [['算出弧度 → ', E.INK2], ['degrees', E.ACC, 700, 1]], 17], [126, [['换成多少度', E.INK2]], 17]]);
      var n9 = rows(TOP, [[0, [['a2', E.HI, 700, 1], ['：肘的角度', E.INK2]]], [34, [['小臂：往右 ', E.INK2], ['d − x', E.INK, 900], ['，往上 ', E.INK2], ['−y', E.INK, 900]], 17],
        [64, [['这个方向的角度 − a1', E.INK2]], 17], [94, [['（第 3 幕：−90 − 63.4）', E.MUT, 700]], 16]]);
      var n10 = rows(TOP, [[0, [['交回两组', E.INK, 900]]], [34, [['肘上：', E.UP, 900], ['[a1, a2]', E.INK, 700, 1]]],
        [64, [['肘下：', E.DOWN, 900], ['[-a1, -a2]', E.INK, 700, 1]]], [92, [['（正负反过来）', E.MUT, 700]], 16],
        [124, [['外面的 [ ]：装着两组', E.INK2]], 17], [160, [['round(…, 1)', E.INK, 700, 1]]], [188, [['：保留 1 位小数', E.INK2]], 17]]);
      var n11 = rows(TOP, [[0, [['ik(1)', E.INK, 700, 1], ['：手在 (1, 0)', E.INK2]]], [30, [['ik(5)', E.INK, 700, 1], ['：手在 (5, 0)', E.INK2]]],
        [62, [['print', E.HI, 700, 1], ['：打印出来', E.INK2]]]]);
      // 第 2 页的说明
      var v1 = rows(BOT, [[0, [['验算：', E.INK, 900], ['正向运动学', E.ACC, 900], ['（第 11 集）', E.MUT, 700]]], [32, [['2 组角度放回去', E.INK2]]],
        [62, [['→ 手停在 ', E.INK2], ['(1, 0)', E.TGT, 900], [' 吗？', E.INK2]]]]);
      var v2 = rows(TOP, [[0, [['照第 11 集导入：', E.INK, 900]]], [32, [['numpy', E.INK, 700, 1], [' → np', E.INK2]]], [62, [['modern_robotics', E.INK, 700, 1], ['：2 个函数', E.INK2]], 17],
        [100, [['fk_space', E.HI, 700, 1], ['：第 11 集写的，原样', E.INK2]], 17], [128, [['照《Modern Robotics》官方库', E.MUT, 700]], 16], [152, [['FKinSpace 写的', E.MUT, 700]], 16]]);
      var v3 = rows(TOP2, [[0, [['这次的零位：', E.INK, 900]]], [32, [['胳膊伸直，平放在 x 轴上', E.INK2]]], [64, '\\textcolor{' + E.HEX.good + '}{M}\\text{：手在 }\\sqrt5+2\\approx 4.24\\text{ 格}', 19, 'tex'],
        [104, '\\texttt{np.sqrt(5)}\\text{：}\\sqrt5', 19, 'tex'], [146, [['= 第 4 幕最远的那一点', E.INK2]], 17]]);
      var v4 = rows(TOP2, [[0, [['从这个零位算起：', E.INK, 900]]], [34, [['肩转的', E.INK2], ['角度', E.ACC, 900]]], [60, [['= 大臂和 x 轴的夹角', E.INK2]], 17],
        [94, [['肘转的', E.INK2], ['角度', E.ACC, 900]]], [120, [['= 小臂相对大臂转了多少', E.INK2]], 17], [156, [['= ik 算出的两个角度 ✓', E.GOOD, 900]], 17]]);
      var v5 = rows(TOP2, [[0, [['S', E.WARN, 700, 1], ['：螺旋轴，一根一排', E.INK2]]], [34, [['肩：在原点（同第 11 集）', E.INK2]], 17],
        [64, '\\text{肘：在 }(\\sqrt5,\\,0)', 19, 'tex'], [96, '\\text{后 3 个数：}(0,\\,-\\sqrt5,\\,0)', 19, 'tex'], [122, [['（第 9 集的算法）', E.MUT, 700]], 16]]);
      var v6 = rows(TOP2, [[0, [['up, down', E.HI, 700, 1], [' = ik(1)', E.INK, 700, 1]]], [32, [['左边 2 个名字', E.INK2]]], [60, [['交回的 2 组分别存进去', E.INK2]], 17],
        [92, [['up', E.UP, 700, 1], ['：肘上　', E.INK2], ['down', E.DOWN, 700, 1], ['：肘下', E.INK2]]]]);
      var v7 = rows(TOP2, [[0, [['np.radians', E.ACC, 700, 1], ['：度 → 弧度', E.INK2]]], [32, [['交给 ', E.INK2], ['fk_space', E.HI, 700, 1]]],
        [62, [['结果存进 ', E.INK2], ['T1', E.INK, 700, 1], ['、', E.INK2], ['T2', E.INK, 700, 1]]]]);
      var v8 = rows(TOP2, [[0, [['T1[:2, 3]', E.HI, 700, 1]]], [28, [['逗号前选横排、后选列', E.INK2]], 17], [52, [['（第 11 集）', E.MUT, 700]], 16],
        [80, [[':2', E.HI, 700, 1], ['：前 2 个　', E.INK2], ['3', E.TGT, 700, 1], ['：位置那一列', E.INK2]], 17], [106, [['= 手往右、往上', E.TGT, 900]], 17],
        [134, [['.round(2)', E.INK2, 700, 1], ['：保留 2 位小数', E.INK2]], 17], [158, [['2 个一起打印', E.INK2]], 17]]);
      var v9 = rows(BOT, [[0, [['肘上 ', E.UP, 900], ['[ 1. -0.]', E.INK, 700, 1]]], [30, [['肘下 ', E.DOWN, 900], ['[1. 0.]', E.INK, 700, 1]]],
        [62, [['手都停在 (1, 0) ✓', E.GOOD, 900]]], [90, [['-0.', E.INK, 700, 1], [' 就是 0（第 11 集）', E.INK2]], 17]]);
      var v10 = rows(BOT, [[124, [['角度只留了 1 位小数', E.INK2]], 17], [150, [['→ 位置离 (1, 0) 约 0.001', E.INK2]], 17], [176, [['留 2 位小数：(1, 0)', E.INK2]], 17]]);
      var n12 = rows(BOT, [[0, [['手在 (1, 0)：2 组', E.INK, 900]]], [32, [['肘上 63.4°、−153.4°', E.UP, 900]]], [62, [['肘下 −63.4°、153.4°', E.DOWN, 900]]],
        [92, [['= 第 3 幕手算的 ✓', E.GOOD, 900]]], [136, [['手在 (5, 0)：', E.INK, 900], ['[]', E.ERR, 700, 1]]], [166, [['一组解也没有 = 第 4 幕 ✓', E.GOOD, 900]], 17]]);

      /* ── 第 2 页的机械臂画面（2026-11-06 用户要求「Modern Robotics 的例子说的时候也要显示机械臂的画面」）：
         右上小图（「这次的零位」起到运行前）：零位伸直、手在 √5 + 2；「肩转的角度」「肘转的角度」转到肘上那组、画出两个转角；
         「两根螺旋轴」回到零位、标出肩 (0, 0)、肘 (√5, 0)。运行出结果时，左边换成大图：肘上、肘下两台 UR5，手都停在 (1, 0) ── */
      var DOX = 500, DOY = 118;
      var dG = E.el('g', {}, s), dUnder = E.el('g', {}, dG), dArmG = E.el('g', {}, dG);
      var dArm = E.arm2(dArmG, DOX, DOY, E.ARM.S * 34 / 46.08, 0, [466, 18, 324, 196]), du = dArm.u;
      var dPl = E.plane(dUnder, { ox: DOX, oy: DOY, u: du, x0: -1, x1: 5, y0: -2, y1: 2 }), DP = dPl.P, DO = DP(0, 0);
      var dTop = E.el('g', {}, dG);
      var dHand = K.paint(E.el('circle', { r: 6, 'stroke-width': 2.5 }, dTop), E.TGT, E.S1);
      var dHandLab = E.tex(dTop, DP(E.L1 + E.L2, 0)[0], DO[1] + 30, '\\textcolor{' + E.HEX.violet + '}{\\sqrt5+2}', 18, null, 'middle', { w: 90, h: 30 });
      var dArc1 = E.arcArrow(dTop, DO[0], DO[1], 0.75 * du, E.OMEGA, 3, { head: 9 }), dArc2 = null;
      var dExt = K.paint(E.el('path', { fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '5 5' }, dTop), null, E.INK2);
      var dA1 = E.halo(E.txt(dTop, 0, 0, 'a1', 17, E.OMEGA, 'start', 900)), dA2 = E.halo(E.txt(dTop, 0, 0, 'a2', 17, E.OMEGA, 'start', 900));
      dA1.style.fontFamily = E.MONO; dA2.style.fontFamily = E.MONO;
      var dArcE = E.arcArrow(dTop, 0, 0, 0.55 * du, E.OMEGA, 3, { head: 9 });
      function axisMark(x, y) { var g = E.el('g', {}, dTop); K.paint(E.el('circle', { cx: x, cy: y, r: 9, 'stroke-width': 3 }, g), E.S1, E.WARN); K.paint(E.el('circle', { cx: x, cy: y, r: 3 }, g), E.WARN); return g; }
      var dS1 = axisMark(DO[0], DO[1]), dS2 = axisMark(DP(E.L1, 0)[0], DO[1]);
      var dS1L = E.tex(dTop, DO[0] - 4, DO[1] - 20, '\\textcolor{' + E.HEX.warn + '}{(0,\\,0)}', 16, null, 'middle', { w: 70, h: 26 });
      var dS2L = E.tex(dTop, DP(E.L1, 0)[0], DO[1] - 20, '\\textcolor{' + E.HEX.warn + '}{(\\sqrt5,\\,0)}', 16, null, 'middle', { w: 80, h: 26 });
      var UPA2 = E.ik2(1, 0, true), DNA2 = E.ik2(1, 0, false);
      // 运行后的大图
      var bigG2 = E.el('g', {}, s), bUnder = E.el('g', {}, bigG2), bGhostG = E.el('g', {}, bigG2), bArmG = E.el('g', {}, bigG2);
      var BOX2 = [10, 14, 440, 392], BOX_ = 110, BOY = 214;
      var bGhost = E.arm2(bGhostG, BOX_, BOY, E.ARM.S, 0, BOX2), bArm = E.arm2(bArmG, BOX_, BOY, E.ARM.S, 0, BOX2), bu = bArm.u;
      var bPl = E.plane(bUnder, { ox: BOX_, oy: BOY, u: bu, x0: -1, x1: 5, y0: -3, y1: 3 }), BP = bPl.P;
      bArm.pose(UPA2[0], UPA2[1]); bGhost.pose(DNA2[0], DNA2[1]);
      bGhost.view.tint('all', E.DOWN, 0.8, 0.55);
      var bTop = E.el('g', {}, bigG2);
      var bTg = E.target(bTop, BP(1, 0)[0], BP(1, 0)[1], 12, E.TGT);
      var bTgL = E.halo(E.txt(bTop, BP(1, 0)[0] + 22, BP(1, 0)[1] - 14, '(1, 0)', 20, E.TGT, 'start', 900));
      var bUpL = E.halo(E.txt(bTop, BP(1, 2)[0] + 18, BP(1, 2)[1] - 18, '肘上 [ 1. -0.]', 20, E.UP, 'start', 900));
      var bDnL = E.halo(E.txt(bTop, BP(1, -2)[0] + 18, BP(1, -2)[1] + 30, '肘下 [1. 0.]', 20, E.DOWN, 'start', 900));
      var bTitle = E.halo(E.txt(bTop, 30, 44, 'fk_space 验算：手都停在 (1, 0)', 20, E.INK, 'start', 900));

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }
      function win(st, a, b) { return E.fade(st, a, 0.3) * (1 - E.fade(st, b - 0.3, 0.3)); }
      function show(n, st, a, b, at) { E.op(n.g, win(st, a, b)); (at || []).forEach(function (t0, k) { if (t0 !== null) E.op(n.r[k].g, fr(st, t0)); }); }
      function bc(i, str, sub, nth, len, col, p, key) { ed.boxCol(i, colOf(str, sub, nth), len || sub.length, col, p, key); }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) { RK.layout(); laid = true; }
          ed.draw(st, t); ed2.draw(st, t); term.draw(st, t);
          var pg2 = E.fade(st, T.page2 - 0.2, 0.4), ed1 = E.fade(st, 12.1, 0.4);
          var big2 = E.fade(st, T.up2 - 0.2, 0.4);
          E.op(edG, ed1 * (1 - pg2)); E.op(ed2G, pg2 * (1 - big2)); E.op(bigG2, big2);
          if (big2 > 0) { E.op(bUpL, E.fade(st, T.up2, 0.3)); E.op(bDnL, E.fade(st, T.dn2, 0.3)); }
          /* 第 2 页右上小图 */
          var dOn = E.fade(st, T.zero - 0.3, 0.4) * (1 - E.fade(st, T.run2 - 0.4, 0.3));
          E.op(dG, dOn);
          if (dOn > 0) {
            var kUp = E.fade(st, T.shAng, 1.2) * (1 - E.fade(st, T.sW, 0.8));   // 讲角度时转到肘上那组，讲螺旋轴时回到零位
            var qa = UPA2[0] * kUp, qb = UPA2[1] * kUp;
            dArm.pose(qa, qb);
            var hp2 = dArm.hand(); dHand.setAttribute('cx', hp2[0].toFixed(1)); dHand.setAttribute('cy', hp2[1].toFixed(1));
            E.op(dHand, fr(st, T.handM)); E.op(dHandLab, fr(st, T.sq5) * (1 - E.fade(st, T.shAng - 0.3, 0.3)));
            var angOn = 1 - E.fade(st, T.sW - 0.3, 0.3);
            dArc1.set(0, qa); E.op(dArc1.g, fr(st, T.xAng) * angOn);
            var ae = qa * 0.5, ar = 0.75 * du + 14;
            dA1.setAttribute('x', (DO[0] + ar * Math.cos(ae)).toFixed(1)); dA1.setAttribute('y', (DO[1] - ar * Math.sin(ae) + 6).toFixed(1));
            dA1.textContent = st >= T.match ? '63.4°' : 'a1'; E.op(dA1, fr(st, T.xAng) * angOn);
            var ep = dArm.elbow(), L = 1.1 * du;
            dExt.setAttribute('d', 'M ' + ep[0].toFixed(1) + ' ' + ep[1].toFixed(1) + ' L ' + (ep[0] + L * Math.cos(qa)).toFixed(1) + ' ' + (ep[1] - L * Math.sin(qa)).toFixed(1));
            E.op(dExt, fr(st, T.elAng) * angOn);
            dArcE.g.setAttribute('transform', 'translate(' + ep[0].toFixed(1) + ' ' + ep[1].toFixed(1) + ')');
            dArcE.set(qa, qb); E.op(dArcE.g, fr(st, T.rel) * angOn);
            var am = (qa + qb) / 2, r2 = 0.55 * du + 16;
            dA2.setAttribute('x', (ep[0] + r2 * Math.cos(am) + 4).toFixed(1)); dA2.setAttribute('y', (ep[1] - r2 * Math.sin(am) + 6).toFixed(1));
            dA2.textContent = st >= T.match ? '−153.4°' : 'a2'; E.op(dA2, fr(st, T.rel) * angOn);
            E.op(dS1, fr(st, T.s1)); E.op(dS1L, fr(st, T.s1)); E.op(dS2, fr(st, T.s2)); E.op(dS2L, fr(st, T.s2));
          }
          /* 句 1：网格和 UR5 */
          var aOn = E.fade(st, T.arm0, 0.4) * (1 - ed1);
          E.op(armPanel, aOn);
          if (aOn > 0) {
            var k3 = E.fade(st, T.ep3, 0.8), k4 = E.fade(st, T.ep4, 0.8);
            arm.pose(Z0[0] + (UPA[0] - Z0[0]) * k3 + (0 - UPA[0]) * k4, Z0[1] + (UPA[1] - Z0[1]) * k3 + (0 - UPA[1]) * k4);
            E.op(xHi, 0.55 * E.bump(st, T.xax, 1.2));
            var dx = 3 + (1 - 3) * k3 + (5 - 1) * k4, tp = P(dx, 0);   // 目标：先是一般的 d（画在 3 格），再挪到 1、5
            tgA.setAttribute('transform', 'translate(' + tp[0].toFixed(1) + ' ' + tp[1].toFixed(1) + ')'); tgA2.setAttribute('transform', tgA.getAttribute('transform'));
            E.op(tgA, fr(st, T.right) * (1 - k4)); E.op(tgA2, k4);
            tLab.textContent = k4 > 0 ? '(5, 0)' : k3 > 0 ? '(1, 0)' : '(d, 0)';   // 一开始挪就换成要去的那一点
            tLab.style.fill = k4 > 0 ? E.ERR : E.TGT;
            tLab.setAttribute('x', tp[0].toFixed(1)); tLab.setAttribute('y', (tp[1] - 22).toFixed(1));
            E.op(tLab, fr(st, T.right));
            dDim.setAttribute('d', 'M ' + O[0] + ' ' + (O[1] + 0.62 * au) + ' H ' + tp[0].toFixed(1) + ' M ' + O[0] + ' ' + (O[1] + 0.62 * au - 7) + ' V ' + (O[1] + 0.62 * au + 7) +
              ' M ' + tp[0].toFixed(1) + ' ' + (O[1] + 0.62 * au - 7) + ' V ' + (O[1] + 0.62 * au + 7));
            dLab.textContent = k4 > 0 ? 'd = 5' : k3 > 0 ? 'd = 1' : 'd 格';
            dLab.setAttribute('x', ((O[0] + tp[0]) / 2).toFixed(1)); dLab.setAttribute('y', (O[1] + 0.62 * au + 26).toFixed(1));
            E.op(dDim, fr(st, T.right)); E.op(dLab, fr(st, T.dW));
            upLab.setAttribute('x', tp[0].toFixed(1)); upLab.setAttribute('y', (tp[1] - 48).toFixed(1));
            E.op(upLab, E.fade(st, T.up0, 0.3) * (1 - E.fade(st, T.ep3, 0.3)));
            E.op(gapG, E.fade(st, T.ep4 + 0.8, 0.3));
          }
          // 终端：第 1 次运行到第 2 页开写前；第 2 页写代码时收起（右边写说明），第 2 次运行再出来
          E.op(termG, E.fade(st, T.run - 0.3, 0.3) * (1 - E.fade(st, T.page2 - 0.3, 0.3)) + E.fade(st, T.run2 - 0.3, 0.3));

          /* 编辑器里的底色：讲到哪句、哪段垫哪段 */
          ed.box('from', E.HI, win(st, T.math, T.three)); ed.box('math', E.HI, win(st, T.math, T.three));
          ed.box('imp', E.HI, win(st, T.three, T.sqrtW));
          ed.box('sqrt', E.WARN, win(st, T.sqrtW, T.atanW)); ed.box('atan', E.ACC, win(st, T.atanW, T.degW)); ed.box('deg', E.ACC, win(st, T.degW, 30.84));
          ed.box('def', E.HI, win(st, T.defW, T.name)); ed.box('nm', E.HI, win(st, T.name, T.inp)); ed.box('din', E.TGT, win(st, T.inp, 38.71));
          ed.hl(L3, E.HI, win(st, T.elbowX, 59.85));
          ed.box('st1', E.HI, win(st, T.star, T.slash)); ed.box('st2', E.HI, win(st, T.star, T.slash)); ed.box('sl', E.HI, win(st, T.slash, 59.85));
          ed.hl(L4, E.HI, win(st, T.yy2, 67.67)); ed.box('yy', E.HI, win(st, T.name2, 67.67));
          ed.hl(L5, E.HI, win(st, T.ifW, 76.5)); ed.box('if', E.HI, win(st, T.ifW, T.cond)); ed.box('lt', E.HI, win(st, T.lt, 76.5));
          ed.box('c3', E.HI, win(st, T.colon, T.indent));
          ed.boxCol(L6, 0, 8, E.HI, win(st, T.indent, T.ret0), 'ind6');
          ed.hl(L6, E.ERR, win(st, T.ret0, 87.76));
          ed.hl(L7, E.HI, win(st, T.sq, 95.97)); ed.box('sq', E.WARN, win(st, T.sq, T.up));
          ed.hl(L8, E.HI, win(st, T.a1W, 105.95)); ed.box('at1', E.ACC, win(st, T.fnA, T.order));
          bc(L8, S8, 'y, x', 0, 4, E.HI, win(st, T.order, T.rad2), 'yx');
          ed.box('dg1', E.ACC, win(st, T.deg2, 105.95)); ed.box('dg1e', E.ACC, win(st, T.deg2, 105.95));
          ed.hl(L9, E.HI, win(st, T.a2W, 115.72)); bc(L9, S9, '-y, d - x', 0, 9, E.HI, win(st, T.dx, T.ang2), 'vec');
          ed.box('at2', E.ACC, win(st, T.ang2, T.minus)); ed.box('m1', E.HI, win(st, T.minus, 115.72));
          ed.box('up', E.UP, win(st, T.upG, T.rnd)); ed.box('dn', E.DOWN, win(st, T.dnG, T.rnd));
          bc(L11, S11, '-', 0, 1, E.HI, win(st, T.sign, T.outer), 'neg1'); bc(L11, S11, '-', 1, 1, E.HI, win(st, T.sign, T.outer), 'neg2');
          bc(L10, S10, '[', 0, 1, E.HI, win(st, T.outer, 126.54), 'o1'); bc(L11, S11, ']', 1, 1, E.HI, win(st, T.outer, 126.54), 'o2');
          ['round(', ', 1)'].forEach(function (sub, k) {
            [0, 1].forEach(function (nth) {
              var p = k === 0 ? win(st, T.rnd, 133.81) : win(st, T.rnd1, 133.81);
              bc(L10, S10, sub, nth, sub.length, E.HI, p, 'r10' + k + nth); bc(L11, S11, sub, nth, sub.length, E.HI, p, 'r11' + k + nth);
            });
          });
          ed.box('ik1', E.TGT, win(st, T.one5, T.run)); ed.box('ik5', E.TGT, win(st, T.one5 + 0.5, T.run));
          bc(L13, S13, 'print', 0, 5, E.HI, win(st, T.prn, T.run), 'p13'); bc(L14, S14, 'print', 0, 5, E.HI, win(st, T.prn, T.run), 'p14');
          // 运行后：两句打印和终端里对应的输出一起亮
          ed.hl(L13, E.UP, win(st, T.r1, T.r5)); ed.hl(L14, E.ERR, fr(st, T.r5));
          term.box('o1', E.UP, win(st, T.upA, T.r5)); term.box('o2', E.DOWN, win(st, T.dnA, T.r5)); term.box('o3', E.ERR, fr(st, T.empty2));

          /* 说明 */
          show(n1, st, T.code, 12.41, [T.code, T.xax, T.ep3, T.ep4]);
          E.op(n1.r[1].parts[1], fr(st, T.dW));
          show(n2, st, T.l1, 30.84, [T.l1, T.math, T.math, T.three, T.sqrtW, T.atanW, T.degW, T.rad]);
          show(n3, st, T.defW, 38.71, [T.defW, T.defW, T.name, T.inp, T.dIn]);
          show(n4, st, T.xl, 59.85, [T.xl, T.eqs, T.eqs, T.rhs, T.lhs, T.getx, T.star]);
          E.op(n4.r[4].parts[1], fr(st, T.rem)); E.op(n4.r[6].parts[2], fr(st, T.slash)); E.op(n4.r[6].parts[3], fr(st, T.slash));
          show(n5, st, T.yyL, 67.67, [T.yy2, T.back3, T.name2]);
          show(n6, st, T.newW, 87.76, [T.ifW, T.cond, T.lt, T.okC, T.indent, T.ret0, T.empty, T.none]);
          E.op(n6.r[0].parts[2], fr(st, T.ifMean));
          show(n7, st, T.notC, 95.97, [T.notC, T.sq, T.pos]);
          E.op(n7.r[2].parts[1], fr(st, T.up));
          show(n8, st, T.a1W, 105.95, [T.a1W, T.args, T.order, T.rad2, T.deg2]);
          E.op(n8.r[3].parts[1], fr(st, T.deg2));
          show(n9, st, T.a2W, 115.72, [T.a2W, T.fore, T.ang2, T.same3]);
          E.op(n9.r[1].parts[1], fr(st, T.dx)); E.op(n9.r[1].parts[2], fr(st, T.dy)); E.op(n9.r[1].parts[3], fr(st, T.dy)); E.op(n9.r[2].g, fr(st, T.ang2));
          show(n10, st, T.retW, 133.81, [T.retW, T.upG, T.dnG, T.sign, T.outer, T.rnd, T.rnd1]);
          show(n11, st, T.one5, T.run, [T.one5, T.one5 + 0.5, T.prn]);
          show(n12, st, T.r1, T.chk, [T.r1, T.upA, T.dnA, T.same, T.r5, T.none2]);
          show(v1, st, T.chk, T.page2, [T.chk, T.back, T.look]);
          E.op(v1.r[2].parts[1], fr(st, T.at10)); E.op(v1.r[2].parts[2], fr(st, T.at10));
          show(v2, st, T.page2, T.zero, [T.npW, T.npW, T.mrW, T.fkFn, T.same0, T.same0]);
          show(v3, st, T.zero, T.from0, [T.zero, T.straight, T.handM, T.sq5, T.far]);
          show(v4, st, T.from0, T.sW, [T.from0, T.shAng, T.xAng, T.elAng, T.rel, T.match]);
          show(v5, st, T.sW, T.unp, [T.sW, T.s1, T.s2, T.v3, T.alg]);
          show(v6, st, T.unp, T.rad, [T.unp, T.unp, T.unp2, T.upS]);
          E.op(v6.r[3].parts[2], fr(st, T.dnS)); E.op(v6.r[3].parts[3], fr(st, T.dnS));
          show(v7, st, T.rad, T.last, [T.radW, T.fkC, T.store]);
          show(v8, st, T.last, T.run2, [T.hpos, T.brk, T.ep11s, T.two3, T.rr, T.rd2, T.both]);
          show(v9, st, T.up2, END + 1, [T.up2, T.dn2, T.hand10, T.neg0]);
          show(v10, st, T.why, END + 1, [T.why, T.off, T.rd3]);
          /* 第 2 页编辑器里的底色 */
          ed2.box('imp2', E.HI, win(st, T.npW, T.mrW)); ed2.box('np', E.HI, win(st, T.npW, T.mrW));
          ed2.box('mr', E.HI, win(st, T.mrW, T.two2)); ed2.box('exp', E.HI, win(st, T.two2, T.fkFn)); ed2.box('vec', E.WARN, win(st, T.two2, T.fkFn));
          PF.forEach(function (i) { ed2.hl(i, E.HI, win(st, T.fkFn, T.zero)); });
          PM.forEach(function (i) { ed2.hl(i, E.GOOD, win(st, T.zero, T.from0)); });
          ed2.box('m03', E.TGT, win(st, T.sq5, T.from0));
          PS.forEach(function (i) { ed2.hl(i, E.WARN, win(st, T.sW, T.unp)); });
          ed2.box('s1', E.WARN, win(st, T.s1, T.s2)); ed2.box('s2', E.WARN, win(st, T.s2, T.v3)); ed2.box('s2b', E.WARN, win(st, T.s2, T.v3)); ed2.box('v3', E.HI, win(st, T.v3, T.unp));
          ed2.hl(PU, E.HI, win(st, T.unp, T.rad)); ed2.box('ud', E.HI, win(st, T.unp, T.unp2)); ed2.box('ik1b', E.HI, win(st, T.unp2, T.rad));
          ed2.hl(PT1, E.HI, win(st, T.rad, T.last)); ed2.hl(PT2, E.HI, win(st, T.rad, T.last));
          ed2.box('r1', E.ACC, win(st, T.radW, T.fkC)); ed2.box('r2', E.ACC, win(st, T.radW, T.fkC));
          ed2.box('fk1', E.HI, win(st, T.fkC, T.store)); ed2.box('fk2', E.HI, win(st, T.fkC, T.store));
          ed2.hl(PP, E.HI, win(st, T.last, T.run2));
          ed2.box('sl1', E.TGT, win(st, T.brk, T.rd2)); ed2.box('sl2', E.TGT, win(st, T.both + 0.9, T.run2));
          ed2.box('rd1', E.INK2, win(st, T.rd2, T.both)); ed2.box('rd2', E.INK2, win(st, T.both + 0.9, T.run2));
          term.box('q1', E.UP, win(st, T.up2, T.why)); term.box('q2', E.DOWN, win(st, T.dn2, T.why));
          E.op(n12.r[4].parts[1], fr(st, T.empty2)); E.op(n12.r[0].g, fr(st, T.two));
        }
      };
    }
  };
})();
