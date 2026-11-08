// 013 第 5 幕：写成代码（照 011 第 6 幕、012 第 5 幕：左边编辑器 jacobian.py，右边说明，运行时右上出终端）。
// 前半：转 90° 的矩阵 R（第 3、9 集；按横排写，第 4 集）、肩 / 肘 / 手的位置数组，c1 = R @ (hand - shoulder)、c2 = R @ (hand - elbow)（小括号里的先算），
//   np.column_stack 把两列并排拼成雅可比 J（NumPy 文档：Stack 1-D arrays as columns into a 2-D array），theta_dot = (1, −1)，打印 J 和 J @ theta_dot。
// 后半（同一个文件接着写）：照第 11 集导入 modern_robotics 的 MatrixExp6、VecTose3，fk_space 原样（照官方库 FKinSpace 写的），零位 M 和两根螺旋轴 S 照第 11 集
//   （第 11 集的零位正好是现在这个姿势：肘 (1, 2)、手 (3, 2)）；dt = 0.001 秒，theta = theta_dot * dt 是这一小会儿转过的角度，p0、p1 是开始时和
//   0.001 秒以后手的位置（[:2, 3] 取往右、往上两个数，第 11 集），(p1 - p0) / dt 是走过的距离除以时间，保留 2 位小数打印。
// 输出按 Python 3.13 + NumPy 2.5.3 + modern_robotics 1.1.1 实跑（见 ../ep013.py）：[[-2  0] [ 3  2]]、[-2  1]、[-2.  1.]；不保留小数 [-2.00049967  0.99899983]。
// 代码按念到的词一段一段打出来（keys：每个字各自的时刻）；讲到哪段、编辑器里哪段垫底色，右边同步写说明。时刻取自配音的逐词时间，见 ../ep013.py。
(function () {
  var E = window.EP013, K = E.K, H = E.HEX;
  var T = {
    code: 0.7, l1: 2.9, ep4: 3.55, imp: 4.59, np: 4.95, short: 5.59,
    rW: 7.67, rEp4: 9.14, rows: 10.23, r1: 12.31, r2: 13.74,
    sh: 16.11, el: 16.65, hd: 17.26, arr: 18.22, n00: 19.85, n12: 20.91, n32: 22.06,
    c1: 24.16, sub1: 24.91, from1: 25.98, paren: 28.18, at1: 29.57, turn1: 30.5,
    c2: 33.56, sub2: 34.54, from2: 35.63, turn2: 37.73,
    fn: 40.05, cols: 41.5, l2r: 43.0, stack: 44.47, jac: 45.77, J: 46.92,
    td: 48.82, td1: 49.78, tdArr: 50.74,
    last2: 53.3, pJ: 54.32, pJv: 55.46, times: 56.74, atW: 57.73,
    run: 59.55, jOut: 60.41, rows2: 61.84, j1: 62.72, j2: 63.86, same3: 64.94, hv: 66.12, v21: 66.95,
    chk: 69.26, fk: 70.51, verify: 71.67, vel: 72.58, dist: 73.18, div: 74.65,
    after: 77.02, mrW: 78.01, two: 79.13, fkFn: 80.59, same0: 83.21,
    zero: 85.26, sW: 85.74, ep11: 86.84, z11: 88.38, now: 89.55,
    dt: 92.45, milli: 93.83, little: 95.42, ang: 96.34, prod: 98.07,
    start: 100.86, later: 101.68, hpos: 103.25, fkC: 104.15, brk: 106.71, xy: 107.26, ep11b: 109.24,
    minus: 111.67, distW: 113.26, divW: 114.66, rd: 115.79, prn: 117.16,
    run2: 119.1, r21: 119.88, sameJ: 120.97,
    noRound: 123.98, raw: 125.07, raw2: 126.88, shorter: 128.25, closer: 129.57
  };
  var END = 131.48;

  // 每个字的出现时刻（取自 012）：spans = [[子串, 第几处（从 0 起）, t0, t1], …]，子串的字在 t0…t1 之间匀速打出；没列到的字在 dflt 出现
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

  E.scenes[4] = {
    title: '写成代码',
    dur: END,
    cues: [
      { at: 0, s: '写成代码：第 $1$ 句照第 4 集，`import numpy as np`' },
      { at: 7.57, s: '转 $90^\\circ$ 的矩阵 `R`：照第 4 集一个横排一个横排地写，第 $1$ 排 $0,\\,-1$，第 $2$ 排 $1,\\,0$' },
      { at: 16.01, s: '肩、肘、手的位置各存成一个数组：$(0,\\,0)$、$(1,\\,2)$、$(3,\\,2)$' },
      { at: 24.06, s: '第 $1$ 列 `c1`：`hand - shoulder` 从肩关节指向手（小括号里的先算），再 `R @` 转 $90^\\circ$' },
      { at: 33.46, s: '第 $2$ 列 `c2`：`hand - elbow` 从肘关节指向手，再转 $90^\\circ$' },
      { at: 39.95, s: '`np.column_stack`：把几个数组当成列，从左到右并排拼成矩阵——这就是**雅可比**，存进 `J`' },
      { at: 48.72, s: '`theta_dot`：$2$ 个**转速**，$1$ 和 $-1$' },
      { at: 53.2, s: '最后 $2$ 句：打印**雅可比**；打印**雅可比** `@` **转速**' },
      { at: 59.45, s: '运行：**雅可比**印成 $2$ 个横排 $-2,\\,0$ 和 $3,\\,2$（同第 3 幕）；**手的速度** $(-2,\\,1)$' },
      { at: 69.16, s: '用第 11 集的**正向运动学**验算：速度 = 走过的距离 ÷ 用的时间' },
      { at: 76.92, s: '接着写：导入 `modern_robotics` 的 $2$ 个函数；`fk_space` 原样用第 11 集的（照《Modern Robotics》官方库 `FKinSpace` 写的）' },
      { at: 85.16, s: '零位 `M` 和 $2$ 根**螺旋轴** `S` 照第 11 集写：那一集的零位，正好就是现在这个姿势' },
      { at: 92.35, s: '`dt = 0.001`：千分之一秒；`theta = theta_dot * dt`：这一小会儿转过的角度 = **转速** × 时间' },
      { at: 100.76, s: '`p0`、`p1`：开始时、$0.001$ 秒以后手的位置；`[:2, 3]` 取往右、往上 $2$ 个数（第 11 集）' },
      { at: 111.57, s: '`(p1 - p0) / dt`：走过的距离 ÷ 时间；`.round(2)` 保留 $2$ 位小数' },
      { at: 119.0, s: '运行：$(-2,\\,1)$，和**雅可比**算出来的一样 ✓' },
      { at: 123.88, s: '不保留小数：$-2.0005,\\,0.999$——时间取得越短，越接近' }
    ],
    build: function (K) {
      var s = K.sceneSvg('把雅可比写成代码 jacobian.py：import numpy as np（第四集）；R = np.array([[0, -1], [1, 0]]) 转九十度的矩阵，按横排写；shoulder、elbow、hand 三个数组 (0, 0)、(1, 2)、(3, 2)；c1 = R @ (hand - shoulder)，小括号里先算，从肩关节指向手再转九十度；c2 = R @ (hand - elbow)；J = np.column_stack([c1, c2]) 把两个数组当成列从左到右并排拼成矩阵，这就是雅可比；theta_dot = np.array([1, -1]) 两个转速；print(J)、print(J @ theta_dot)；运行打印 [[-2 0] [3 2]] 和 [-2 1]。验算：接着写，from modern_robotics import MatrixExp6, VecTose3，第十一集的 fk_space 原样拿过来，零位 M 和两根螺旋轴 S 照第十一集；dt = 0.001 千分之一秒，theta = theta_dot * dt 这一小会儿转过的角度；p0、p1 是开始时和千分之一秒以后手的位置，[:2, 3] 取往右、往上两个数；print(((p1 - p0) / dt).round(2)) 走过的距离除以时间，保留两位小数；运行打印 [-2. 1.]，和雅可比算出来的一样；不保留小数是 [-2.00049967 0.99899983]，时间取得越短越接近');

      /* ── 左：编辑器（第 1 页 13 句） ── */
      var edG = E.el('g', {}, s);
      var ed = E.code(edG, 10, 14, 440, 392, { title: 'jacobian.py', size: 17, lh: 24 });
      function line(e, str, spans, dflt) { return e.line(str, dflt, { keys: keysOf(str, spans, dflt) }); }
      var L0 = line(ed, '{h.imp:import} {n.npn:numpy as np}', [['import numpy as np', 0, T.imp, T.short + 0.4]], T.imp);
      ed.line('', T.short + 0.5, { dur: 0 });
      var S1 = '{h.R:R} = np.array([[{x.r00:0}, {y.r01:-1}], [{x.r10:1}, {y.r11:0}]])';
      var L1 = line(ed, S1, [['R = np.array(', 0, T.rW, T.rW + 0.9], ['[[0, -1],', 0, T.r1, T.r1 + 0.9], [' [1, 0]])', 0, T.r2, T.r2 + 0.9]], T.rW);
      var L2 = line(ed, '{h.sh:shoulder} = np.array([{v.p0:0, 0}])', [['shoulder = np.array([', 0, T.sh, T.sh + 0.5], ['0, 0])', 0, T.n00, T.n00 + 0.6]], T.sh);
      var L3 = line(ed, '{h.el:elbow} = np.array([{v.p1:1, 2}])', [['elbow = np.array([', 0, T.el, T.el + 0.5], ['1, 2])', 0, T.n12, T.n12 + 0.6]], T.el);
      var L4 = line(ed, '{h.hd:hand} = np.array([{v.p2:3, 2}])', [['hand = np.array([', 0, T.hd, T.hd + 0.5], ['3, 2])', 0, T.n32, T.n32 + 0.6]], T.hd);
      ed.line('', T.n32 + 0.7, { dur: 0 });
      var L5 = line(ed, '{h.c1:c1} = {a.at1:R @} {n.d1:(hand - shoulder)}', [['c1 = ', 0, T.c1, T.c1 + 0.3], ['(hand - shoulder)', 0, T.sub1, T.sub1 + 0.8], ['R @ ', 0, T.at1, T.at1 + 0.4]], T.c1);
      var L6 = line(ed, '{c.c2:c2} = {a.at2:R @} {n.d2:(hand - elbow)}', [['c2 = ', 0, T.c2, T.c2 + 0.3], ['(hand - elbow)', 0, T.sub2, T.sub2 + 0.7], ['R @ ', 0, T.turn2, T.turn2 + 0.4]], T.c2);
      var L7 = line(ed, '{h.J:J} = {a.cs:np.column_stack}({n.cc:[c1, c2]})', [['np.column_stack(', 0, T.fn, T.fn + 0.8], ['[c1, c2]', 0, T.cols, T.cols + 1.0], [')', 0, T.stack, T.stack + 0.1], ['J = ', 0, T.J, T.J + 0.3]], T.fn);
      var L8 = line(ed, '{a.td:theta_dot} = np.array([{a.tv:1, -1}])', [['theta_dot = ', 0, T.td, T.td + 0.5], ['np.array([1, -1])', 0, T.td1, T.tdArr + 0.8]], T.td);
      ed.line('', T.tdArr + 0.9, { dur: 0 });
      var L9 = line(ed, '{h.p1:print(J)}', [['print(J)', 0, T.pJ, T.pJ + 0.5]], T.pJ);
      var L10 = line(ed, 'print({h.jt:J @ theta_dot})', [['print(J', 0, T.pJv, T.pJv + 0.5], [' @ ', 0, T.atW, T.atW + 0.2], ['theta_dot)', 0, T.times, T.times + 0.6]], T.pJv);

      /* ── 第 2 页：接着写验算（14 号字，20 句，最后滚上去 2 排） ── */
      var ed2G = E.el('g', {}, s);
      var ed2 = E.code(ed2G, 10, 14, 440, 392, { title: 'jacobian.py（接着写）', size: 14, lh: 19 });
      var P1 = line(ed2, '{h.from2:from} {n.mr:modern_robotics} {h.imp3:import} {n.fns:MatrixExp6, VecTose3}', [['from modern_robotics import MatrixExp6, VecTose3', 0, T.mrW, T.two + 0.7]], T.mrW);
      ed2.line('', T.fkFn, { dur: 0 });
      var FUN = ['def fk_space(M, S, theta):', '    T = M', '    for i in reversed(range(len(theta))):', '        se3 = VecTose3(S[i] * theta[i])', '        T = MatrixExp6(se3) @ T', '    return T'];
      var PF = FUN.map(function (l) { return ed2.line(l, T.fkFn, { dur: 0 }); });   // 第 11 集写的函数：原样一下子出来
      ed2.line('', T.fkFn, { dur: 0 });
      var PM = [line(ed2, 'M = np.array([[1, 0, 0, {v.m03:3}],', [], T.zero), line(ed2, '              [0, 1, 0, {v.m13:2}],', [], T.zero + 0.1),
        line(ed2, '              [0, 0, 1, 0],', [], T.zero + 0.2), line(ed2, '              [0, 0, 0, 1]])', [], T.zero + 0.3)];
      var PS = [line(ed2, 'S = np.array([[{w.s1:0, 0, 1, 0, 0, 0}],', [], T.sW), line(ed2, '              [{w.s2:0, 0, 1, 2, -1, 0}]])', [], T.sW + 0.15)];
      var PD = line(ed2, '{h.dt:dt} = {n.dtv:0.001}', [['dt = 0.001', 0, T.milli, T.milli + 0.8]], T.milli);
      var PT = line(ed2, '{h.th:theta} = {a.thv:theta_dot * dt}', [['theta = ', 0, T.ang, T.ang + 0.5], ['theta_dot * dt', 0, T.prod, T.prod + 1.0]], T.ang);
      var SP0 = '{h.q0:p0} = {n.f0:fk_space}(M, S, {n.z0:[0, 0]}){v.c0:[:2, 3]}';
      var PP0 = line(ed2, SP0, [['p0 = fk_space(M, S, [0, 0])', 0, T.start, T.start + 0.9], ['[:2, 3]', 0, T.brk, T.brk + 0.5]], T.start);
      var SP1 = '{h.q1:p1} = {n.f1:fk_space}(M, S, {a.z1:theta}){v.c1:[:2, 3]}';
      var PP1 = line(ed2, SP1, [['p1 = fk_space(M, S, theta)', 0, T.later, T.later + 0.9], ['[:2, 3]', 0, T.brk + 0.5, T.brk + 1.0]], T.later);
      var PPR = line(ed2, 'print({h.dd:((p1 - p0) / dt)}{i.rd:.round(2)})', [['print(', 0, T.minus, T.minus + 0.2], ['((p1 - p0)', 0, T.minus + 0.2, T.minus + 1.0],
        [' / dt)', 0, T.divW, T.divW + 0.4], ['.round(2)', 0, T.rd, T.rd + 0.5], [')', 3, T.prn, T.prn + 0.1]], T.minus);

      /* ── 右上：终端（运行时出来） ── */
      var RX = 460, RW = 330;
      var termG = E.el('g', {}, s);
      var term = E.code(termG, RX, 14, RW, 162, { title: '终端', size: 16, lh: 22 });   // 5 排：第 2 次运行 1 句命令 + 4 句输出（原先 4 排，命令出来 0.27 s 就被挤上去，闪一下）
      term.line('$ python3 jacobian.py', T.run + 0.2, { dur: 0.5 });
      var O1 = term.line('{h.o1:[[-2}  {c.o2:0]}', T.jOut, { out: true }), O2 = term.line(' {h.o3:[ 3}  {c.o4:2]]}', T.jOut, { out: true });
      var O3 = term.line('{v.o5:[-2  1]}', T.jOut, { out: true });
      term.line('', T.after, { dur: 0, page: 1, row: 0 });   // 终端收起时就换到第 2 页：原先第 2 次运行淡入的那 0.3 s 先闪出上一次的输出
      term.line('$ python3 jacobian.py', T.run2 + 0.1, { dur: 0.4, page: 1, row: 0 });
      term.line('[[-2  0]', T.r21 - 0.3, { out: true, page: 1 }); term.line(' [ 3  2]]', T.r21 - 0.3, { out: true, page: 1 });
      term.line('[-2  1]', T.r21 - 0.3, { out: true, page: 1 });
      // 第 2 次运行：4 句输出，窗口放 4 排（命令那排滚上去）
      var O4 = term.line('{v.o6:[-2.  1.]}', T.r21 - 0.3, { out: true, page: 1 });

      /* ── 右边：说明。parts = [字, 颜色, 粗细, 等宽]；运行前从上面排，运行后排在终端下面 ── */
      var RK = E.rowKit(), NX = RX + 2;
      function rows(y0, ls) {
        var g = E.el('g', {}, s);
        return { g: g, r: ls.map(function (l) {
          var row = RK.line(g, NX, y0 + l[0], l[1].map(function (p) { return [p[0], p[1], p[2]]; }), l[2] || 18);
          l[1].forEach(function (p, k) { if (p[3]) row.parts[k].style.fontFamily = E.MONO; });
          return row;
        }) };
      }
      var TOP = 44, BOT = 200, TOP2 = 210;
      var n1 = rows(TOP, [[0, [['写成代码', E.HI, 900]], 22], [40, [['第 1 句（第 4 集）：', E.INK2]]], [70, [['import numpy as np', E.INK, 700, 1]], 17],
        [100, [['导入 NumPy，起个短名字 np', E.INK2]], 17]]);
      var n2 = rows(TOP, [[0, [['R', E.HI, 700, 1], ['：转 90° 的矩阵（第 3、9 集）', E.INK2]], 18], [32, [['一个横排一个横排地写（第 4 集）', E.INK2]], 17],
        [66, [['第 1 排 ', E.INK2], ['0, −1', E.INK, 900]]], [96, [['第 2 排 ', E.INK2], ['1, 0', E.INK, 900]]], [126, [['（两列：', E.MUT, 700], ['(0, 1)', E.X, 900], [' 和 ', E.MUT, 700], ['(−1, 0)', E.Y, 900], ['）', E.MUT, 700]], 17]]);
      var n3 = rows(TOP, [[0, [['肩、肘、手的位置', E.INK, 900]], 20], [32, [['各存成一个数组：', E.INK2]]],
        [64, [['shoulder', E.HI, 700, 1], [' (0, 0)', E.INK2]]], [94, [['elbow', E.HI, 700, 1], [' (1, 2)', E.INK2]]], [124, [['hand', E.V, 700, 1], [' (3, 2)', E.INK2]]]]);
      var n4 = rows(TOP, [[0, [['第 1 列 ', E.INK, 900], ['c1', E.J1, 700, 1]], 20], [34, [['hand - shoulder', E.INK, 700, 1]], 17], [60, [['= 从肩关节指向手', E.INK2]], 17],
        [92, [['小括号里的先算', E.INK, 900]], 17], [124, [['R @', E.ACC, 700, 1], [' 乘上 R：转 90°', E.INK2]], 17], [150, [['（@：矩阵乘法，第 4 集）', E.MUT, 700]], 16]]);
      var n5 = rows(TOP, [[0, [['第 2 列 ', E.INK, 900], ['c2', E.J2, 700, 1]], 20], [34, [['hand - elbow', E.INK, 700, 1]], 17], [60, [['= 从肘关节指向手', E.INK2]], 17],
        [92, [['R @', E.ACC, 700, 1], [' 再转 90°', E.INK2]], 17]]);
      var n6 = rows(TOP, [[0, [['np.column_stack', E.ACC, 700, 1]], 18], [30, [['NumPy 的函数：', E.INK2]], 17], [58, [['把几个数组当成列', E.INK, 900]], 17],
        [86, [['从左到右并排，拼成矩阵', E.INK, 900]], 17], [124, [['→ 雅可比，存进 ', E.HI, 900], ['J', E.HI, 700, 1]], 19]]);
      var n7 = rows(TOP, [[0, [['theta_dot', E.ACC, 700, 1], ['：2 个转速', E.INK2]], 19], [32, [['肩 1，肘 −1', E.OMEGA, 900]], 19]]);
      var n8 = rows(TOP, [[0, [['最后 2 句', E.INK, 900]], 20], [34, [['print(J)', E.HI, 700, 1], ['：打印雅可比', E.INK2]], 17],
        [64, [['print(J @ theta_dot)', E.HI, 700, 1]], 17], [90, [['= 雅可比 @ 转速', E.INK2]], 17]]);
      var n9 = rows(BOT, [[0, [['雅可比：2 个横排', E.INK, 900]], 19], [30, [['−2, 0', E.INK, 900], [' 和 ', E.INK2], ['3, 2', E.INK, 900]], 19], [60, [['= 第 3 幕的 ✓', E.GOOD, 900]], 18],
        [100, [['手的速度 ', E.V, 900], ['[-2  1]', E.INK, 700, 1]], 19], [130, [['= (−2, 1) ✓', E.GOOD, 900]], 18]]);
      var v1 = rows(BOT, [[0, [['验算：', E.INK, 900], ['正向运动学', E.ACC, 900], ['（第 11 集）', E.MUT, 700]], 19],
        [34, [['速度 = 走过的距离 ÷ 用的时间', E.INK, 900]], 18]]);
      var v2 = rows(TOP, [[0, [['接着写：', E.INK, 900]], 19], [32, [['modern_robotics', E.INK, 700, 1], ['：2 个函数', E.INK2]], 17],
        [64, [['fk_space', E.HI, 700, 1], ['：第 11 集写的，原样', E.INK2]], 17], [92, [['照《Modern Robotics》官方库', E.MUT, 700]], 16], [116, [['FKinSpace 写的', E.MUT, 700]], 16]]);
      var v3 = rows(TOP2, [[0, [['M', E.GOOD, 700, 1], ['、', E.INK2], ['S', E.WARN, 700, 1], ['：照第 11 集写', E.INK2]], 19], [32, [['那一集的零位 = 现在这个姿势', E.INK, 900]], 17]]);
      var v4 = rows(TOP2, [[0, [['dt = 0.001', E.HI, 700, 1], ['：千分之一秒', E.INK2]], 18], [34, [['theta = theta_dot * dt', E.ACC, 700, 1]], 16],
        [60, [['这一小会儿转过的角度', E.INK2]], 17], [86, [['= 转速 × 时间', E.OMEGA, 900]], 17]]);
      var v5 = rows(TOP2, [[0, [['p0', E.HI, 700, 1], ['：开始时手的位置', E.INK2]], 17], [28, [['p1', E.HI, 700, 1], ['：0.001 秒以后', E.INK2]], 17],
        [58, [['[:2, 3]', E.VIOLET, 700, 1], ['：往右、往上 2 个数', E.INK2]], 17], [84, [['（第 11 集）', E.MUT, 700]], 16]]);
      var v6 = rows(TOP2, [[0, [['(p1 - p0) / dt', E.HI, 700, 1]], 17], [28, [['= 走过的距离 ÷ 时间', E.INK2]], 17], [58, [['.round(2)', E.INK2, 700, 1], ['：保留 2 位小数', E.INK2]], 17]]);
      var v7 = rows(BOT, [[0, [['[-2.  1.]', E.V, 700, 1], [' = (−2, 1)', E.V, 900]], 19], [32, [['= 雅可比算出来的 ✓', E.GOOD, 900]], 18]]);
      var v8 = rows(BOT, [[82, [['不保留小数：', E.INK2]], 17], [110, [['[-2.00049967  0.99899983]', E.INK, 700, 1]], 16], [140, [['时间取得越短，就越接近', E.INK, 900]], 17]]);

      /* ── 右上小图（「那一集的零位」到第 2 次运行前）：零位的 UR5，肘 (1, 2)、手 (3, 2)；「这一小会儿」两个关节按 (1, −1) 转一点点 ── */
      var DOX = 540, DOY = 160;
      var dG = E.el('g', {}, s), dUnder = E.el('g', {}, dG), dArmG = E.el('g', {}, dG);
      var dArm = E.arm2(dArmG, DOX, DOY, E.ARM.S * 30 / 46, 0, [466, 18, 324, 186]), du = dArm.u;
      var dPl = E.plane(dUnder, { ox: DOX, oy: DOY, u: du, x0: -2, x1: 5, y0: -1, y1: 4 }), DP = dPl.P;
      var dTop = E.el('g', {}, dG);
      var dHand = E.dot(dTop, E.V, 5.5), dEl = E.joint(dTop, E.HI, 8), dSh = E.joint(dTop, E.HI, 8);
      var dEL = E.halo(E.txt(dTop, DP(1, 2)[0] - 12, DP(1, 2)[1] - 10, '(1, 2)', 15, E.HI, 'end', 900));
      var dHL = E.halo(E.txt(dTop, DP(3, 2)[0] + 4, DP(3, 2)[1] + 26, '(3, 2)', 15, E.V, 'start', 900));
      var dV = E.gv(dTop, DP, E.V, 4, { head: 12 });

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }
      function win(st, a, b) { return E.fade(st, a, 0.3) * (1 - E.fade(st, b - 0.3, 0.3)); }
      function show(n, st, a, b, at) { E.op(n.g, win(st, a, b)); (at || []).forEach(function (t0, k) { if (t0 !== null) E.op(n.r[k].g, fr(st, t0)); }); }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) { RK.layout(); laid = true; }
          ed.draw(st, t); ed2.draw(st, t); term.draw(st, t);
          var pg2 = E.fade(st, T.after - 0.2, 0.4);
          E.op(edG, E.fade(st, 0, 0.4) * (1 - pg2)); E.op(ed2G, pg2);
          // 终端：第 1 次运行到验算开写前；第 2 页写代码时收起，第 2 次运行再出来
          E.op(termG, E.fade(st, T.run - 0.3, 0.3) * (1 - E.fade(st, T.after - 0.3, 0.3)) + E.fade(st, T.run2 - 0.3, 0.3));
          /* 右上小图 */
          var dOn = E.fade(st, T.z11 - 0.3, 0.4) * (1 - E.fade(st, T.run2 - 0.9, 0.3));
          E.op(dG, dOn);
          if (dOn > 0) {
            var k = E.fade(st, T.little, 0.8) * (1 - E.fade(st, T.start, 0.6)) * 0.35;   // 示意：按 (1, −1) 转一点（真的只转 0.001 弧度，看不出来）
            dArm.pose(E.ZERO[0] + k, 0);
            var e = [E.L1 * Math.cos(E.ZERO[0] + k), E.L1 * Math.sin(E.ZERO[0] + k)];
            dSh.at(DP(0, 0)); dEl.at(DP(e[0], e[1])); dHand.at(DP(e[0] + 2, e[1]));
            E.op(dEL, 1 - E.fade(st, T.little, 0.3)); E.op(dHL, 1 - E.fade(st, T.little, 0.3));
            dV.at(3, 2, 1, 3); E.op(dV.g, E.win(st, T.distW, T.run2, 0.3));
          }

          /* 编辑器里的底色：讲到哪句、哪段垫哪段 */
          ed.hl(L0, E.HI, win(st, T.imp, T.rW));
          ed.hl(L1, E.HI, win(st, T.rW, T.sh)); ed.box('r00', E.X, win(st, T.r1, T.sh)); ed.box('r10', E.X, win(st, T.r2, T.sh));
          ed.box('r01', E.Y, win(st, T.r1, T.sh)); ed.box('r11', E.Y, win(st, T.r2, T.sh));
          ed.box('p0', E.V, win(st, T.n00, T.c1)); ed.box('p1', E.V, win(st, T.n12, T.c1)); ed.box('p2', E.V, win(st, T.n32, T.c1));
          ed.hl(L5, E.J1, win(st, T.c1, T.c2)); ed.box('d1', E.INK, win(st, T.sub1, T.at1)); ed.box('at1', E.ACC, win(st, T.at1, T.c2));
          ed.hl(L6, E.J2, win(st, T.c2, T.fn)); ed.box('d2', E.INK, win(st, T.sub2, T.turn2)); ed.box('at2', E.ACC, win(st, T.turn2, T.fn));
          ed.hl(L7, E.HI, win(st, T.fn, T.td)); ed.box('cs', E.ACC, win(st, T.fn, T.jac)); ed.box('cc', E.INK, win(st, T.cols, T.jac)); ed.box('J', E.HI, win(st, T.jac, T.td));
          ed.hl(L8, E.OMEGA, win(st, T.td, T.last2));
          ed.hl(L9, E.HI, win(st, T.pJ, T.run)); ed.hl(L10, E.HI, win(st, T.pJv, T.run));
          ed.hl(L9, E.HI, win(st, T.jOut, T.hv)); ed.hl(L10, E.V, win(st, T.hv, T.chk));
          term.box('o1', E.J1, win(st, T.j1, T.hv)); term.box('o3', E.J1, win(st, T.j2, T.hv)); term.box('o2', E.J2, win(st, T.j1, T.hv)); term.box('o4', E.J2, win(st, T.j2, T.hv));
          term.box('o5', E.V, win(st, T.v21, T.chk));
          /* 第 2 页 */
          ed2.box('fns', E.HI, win(st, T.two, T.fkFn));
          PF.forEach(function (i) { ed2.hl(i, E.HI, win(st, T.fkFn, T.zero)); });
          PM.forEach(function (i) { ed2.hl(i, E.GOOD, win(st, T.zero, T.dt)); });
          PS.forEach(function (i) { ed2.hl(i, E.WARN, win(st, T.sW, T.dt)); });
          ed2.hl(PD, E.HI, win(st, T.milli, T.start)); ed2.hl(PT, E.OMEGA, win(st, T.ang, T.start)); ed2.box('thv', E.OMEGA, win(st, T.prod, T.start));
          ed2.hl(PP0, E.HI, win(st, T.start, T.minus)); ed2.hl(PP1, E.HI, win(st, T.later, T.minus));
          ed2.box('c0', E.VIOLET, win(st, T.brk, T.minus)); ed2.box('c1', E.VIOLET, win(st, T.brk + 0.5, T.minus));
          ed2.hl(PPR, E.HI, win(st, T.minus, T.run2)); ed2.box('dd', E.V, win(st, T.minus, T.rd)); ed2.box('rd', E.INK2, win(st, T.rd, T.run2));
          ed2.hl(PPR, E.V, win(st, T.r21, END + 1));
          term.box('o6', E.V, fr(st, T.r21));

          /* 说明 */
          show(n1, st, T.code, T.rW, [T.code, T.l1, T.imp, T.np]);
          show(n2, st, T.rW, T.sh, [T.rW, T.rows, T.r1, T.r2, T.r2 + 0.6]);
          show(n3, st, T.sh, T.c1, [T.sh, T.arr, T.n00, T.n12, T.n32]);
          show(n4, st, T.c1, T.c2, [T.c1, T.sub1, T.from1, T.paren, T.at1, T.turn1]);
          show(n5, st, T.c2, T.fn, [T.c2, T.sub2, T.from2, T.turn2]);
          show(n6, st, T.fn, T.td, [T.fn, T.fn, T.cols, T.l2r, T.jac]);
          show(n7, st, T.td, T.last2, [T.td, T.td1]);
          show(n8, st, T.last2, T.run - 0.35, [T.last2, T.pJ, T.pJv, T.times]);   // 终端出来之前收掉（同一块地方）
          show(n9, st, T.jOut, T.chk, [T.jOut, T.j1, T.same3, T.hv, T.v21]);
          show(v1, st, T.chk, T.after, [T.chk, T.vel]);
          show(v2, st, T.after, T.zero, [T.after, T.mrW, T.fkFn, T.same0, T.same0]);
          show(v3, st, T.zero, T.dt, [T.zero, T.z11]);
          show(v4, st, T.dt, T.start, [T.milli, T.ang, T.little, T.prod]);
          show(v5, st, T.start, T.minus, [T.start, T.later, T.xy, T.ep11b]);
          show(v6, st, T.minus, T.run2, [T.minus, T.distW, T.rd]);
          show(v7, st, T.r21, END + 1, [T.r21, T.sameJ]);
          show(v8, st, T.noRound, END + 1, [T.noRound, T.raw, T.shorter]);
        }
      };
    }
  };
})();
