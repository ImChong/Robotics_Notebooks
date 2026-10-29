// 006 第 2 幕：欧拉角——三个角，按顺序转。每次绕身子自己的一根轴（轴跟着身子一起动）：先偏航（绕 z）、再俯仰（绕身子的 y）、最后横滚（绕身子的 x），
// 即 Z-Y-X 欧拉角；同样三个角绕不动的轴、反过来先横滚再俯仰后偏航，摆出来一模一样，叫固定角（X-Y-Z 固定角，URDF 的 rpy 就这么定义）——
// 两种转法是同一个矩阵 Rz·Ry·Rx（Lynch & Park《Modern Robotics》附录 B.1、B.2）；ROS 的 URDF 文件按 (横滚, 俯仰, 偏航) 写这三个数（urdfdom pose.hpp、tf2 setRPY）。
// 左边 G1（身子绕骨盆转，不画地面），右边三行按转的顺序「① 偏航 / ② 俯仰 / ③ 横滚」，底下放例子和「顺序不能换」。
// 句 1「欧拉角」右边出标题「按顺序转三次」，「三次」三行的编号 ①②③ 依次出来，「身子自己的一根轴」「转一个角度」各鼓一下；
// 句 2「偏航」第一行写出名字和「绕竖着的轴」，「竖着的轴」骨盆上画一根蓝色虚线（z）、绕它一圈，「原地转身」G1 转身 45°，行尾的数从 0° 数到 45°；
// 句 3「俯仰」第二行，「身子左右方向的轴」沿身子此刻的 y 画绿虚线和一圈，「往前倾」G1 前倾 25°；
// 句 4「横滚」第三行，「身子前后方向的轴」沿身子此刻的 x 画红虚线和一圈，「测歪」G1 侧歪 20°；
// 句 5「身子自己的轴」「跟着身子」骨盆上三根小箭头鼓一下，「机器人里最常用」底下写出「偏航 → 俯仰 → 横滚」，念到哪个哪个行亮；
// 句 6「同样三个角」G1 回到原样，标题换成「固定角：绕不动的轴，③ → ② → ①」，底下出欧拉角 / 固定角两行对照，「绕不动的轴」画世界的三根轴（虚线），
//    「横滚」「俯仰」「偏航」G1 依次绕不动的 x、y、z 转（正在转的那根轴亮、画一圈），「一模一样」打勾，「ROS 的机器人模型文件」底下换成「ROS 的 URDF 文件这样写」RPY = Roll 横滚 · Pitch 俯仰 · Yaw 偏航；
// 句 7「写成三个数」底下出 (20°, 25°, 45°)，念到哪个数（再念到「测歪」「前倾」「转身」）哪个数对应的行一起亮；
// 句 8「直观」「只用三个数」底下两块绿牌依次打勾；
// 句 9「欧拉角这三次转动」底下换成两台小 G1：左边先绕竖轴转 90° 再绕（身子的）左右轴转 45°，右边反过来，「第三集」顶上出小牌「欧拉角：顺序不能换」，
//    「转动接着做」左边那台按顺序做，「换了顺序」右边那台做，「结果」中间打「≠」，「固定角也是这样」小牌换成「固定角：顺序也不能换」，
//    「欧拉角和固定角各有十二种」顶上写出「各 12 种：写清楚是哪一种」；
// 句 10「这十二种」两台小 G1 和左边的 G1 收起（后面三句用整个画面排表），「第一次」写出数法，「三乘二乘二」写出 3 × 2 × 2 = 12；
// 句 11 右边两块：左「三根轴都不一样」、右「头尾是同一根轴」，「六种」各写出一列欧拉角（x-y-z … z-y-x / x-y-x … z-y-z），「z、y、x」「z、y、z」各框住最后一种
//    （相邻两次绕的轴不同就行，第一根和第三根可以相同：Modern Robotics 附录 B.1.2）；
// 句 12「固定角」每行补上「= 固定角」（欧拉角的顺序反过来），「欧拉角 z、y、x」「固定角 x、y、z」两格依次框住，「头尾」框住 z-y-z = z-y-z，「轴的顺序不变」右块等号亮一下
//    （绕不动的轴转是左乘、绕身子的轴转是右乘：Modern Robotics 式 (3.23)(3.24)，所以欧拉角 a-b-c = 固定角 c-b-a，12 种都验算过）。
// 时刻取自配音的逐词时间（分镜时间 = 念到这个词的时刻），见 ../ep006.py。
(function () {
  var E = window.EP006, K = E.K;
  var D = Math.PI / 180;
  // 按轴排的数组下标 i：0 = x 横滚，1 = y 俯仰，2 = z 偏航；三行按转的顺序排：行 r 对应轴 ROWAX[r]
  var T = {
    head: 0.1, nums: 3.27, axisB: 4.48, angB: 6.36,
    rows: [7.76, 11.96, 16.59], axis: [18.04, 13.24, 9.03], spins: [20.2, 14.9, 10.4], spinD: 1.0,   // axis / spins 按轴：x、y、z
    body: [22.94, 24.21], ord: 26.06, ordN: [28.94, 29.73, 30.68],
    fix0: 31.99, fixAx: 33.69, fx: [35.42, 36.21, 37.15], fxD: 0.6, same: 38.29, fixName: 39.73,
    rpy: [40.89, 43.24, 43.88, 44.38],
    triple: 47.21, flash: [48.53, 49.17, 49.85], easy: 50.65, flash2: [51.85, 52.95, 54.22],
    good: [56.33, 57.27], best: 59.19,
    order: 61.9, ep3: 65.05, runA: 66.15, runB: 67.24, runD: 0.45, neq: 68.19, fixToo: 69.41, conv: 72.53,
    cnt: 78.6, rule: 80.26, mult: 85.28, r1: [88.3, 89.7, 91.34], r2: [92.73, 94.55, 95.62],
    fixCol: 100.87, exE: 103.41, exF: 105.46, pal: 107.43, palRev: 109.12
  };
  var ANG = [20 * D, 25 * D, 45 * D];                         // (横滚, 俯仰, 偏航)
  var GX = 170, GY = 362, GH = 290;
  var RX = 372, ROW = [{ y: 52 }, { y: 120 }, { y: 188 }];
  var COLS = [E.X, E.Y, E.Z];                                  // 横滚绕 x（红）、俯仰绕 y（绿）、偏航绕 z（蓝）
  var ROWAX = [2, 1, 0];                                       // 三行：① 偏航（z）② 俯仰（y）③ 横滚（x）
  var AXES = [[1, 0, 0], [0, 1, 0], [0, 0, 1]], PLANES = [[[0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0]]];
  var PL = [[1, 2], [0, 2], [0, 1]];   // 绕第 i 根轴转的那一圈，在另外两根轴张成的平面里
  // 欧拉角：绕身子的第 i 根轴转之前，身子的朝向（先偏航、再俯仰、最后横滚）
  var BEFORE = [E.mul3(E.Rz(ANG[2]), E.Ry(ANG[1])), E.Rz(ANG[2]), [[1, 0, 0], [0, 1, 0], [0, 0, 1]]];

  function bump(st, a, d) { var u = K.seg(st, a, a + d); return u > 0 && u < 1 ? Math.sin(Math.PI * u) : 0; }

  E.scenes[1] = {
    title: '欧拉角：三个角，按顺序转',
    dur: 111.62,
    cues: [
      { at: 0, s: '**欧拉角**：拆成按顺序的三次转动，每次绕身子自己的一根轴、转一个角度' },
      { at: 7.66, s: '① 偏航：绕竖着的轴（z），原地转身' },
      { at: 11.86, s: '② 俯仰：绕身子左右方向的轴（y），往前倾' },
      { at: 16.49, s: '③ 横滚：绕身子前后方向的轴（x），往一边侧歪' },
      { at: 21.53, s: '绕的都是身子自己的轴；最常用的顺序：偏航 → 俯仰 → 横滚' },
      { at: 31.99, s: '**固定角**：绕不动的轴，反过来先横滚、再俯仰、最后偏航，摆出来一样' },
      { at: 47.11, s: '(20°, 25°, 45°)：侧歪 20°，前倾 25°，转身 45°' },
      { at: 56.23, s: '好处：**直观**，又只用 3 个数' },
      { at: 61.03, s: '欧拉角三次转动的顺序不能换，固定角也一样；两种各 12 种，要写清楚是哪一种' },
      { at: 78.5, s: '数法：第一次 3 根轴选 1 根，之后每次和上一次不同：3 × 2 × 2 = 12' },
      { at: 88.2, s: '三根轴都不一样 6 种（如 z-y-x），头尾是同一根轴 6 种（如 z-y-z）' },
      { at: 97.68, s: '欧拉角 a-b-c = 固定角 c-b-a，摆出来一样' }
    ],
    build: function (K) {
      var s = K.sceneSvg('欧拉角把一个朝向拆成按顺序的三次转动，每次绕身子自己的一根轴：先偏航，绕竖着的轴原地转身 45 度；再俯仰，绕身子左右方向的轴前倾 25 度；最后横滚，绕身子前后方向的轴侧歪 20 度，机器人里最常用这个顺序。同样三个角绕不动的轴、反过来先横滚再俯仰后偏航，摆出来一模一样，叫固定角，ROS 的 URDF 文件就按横滚、俯仰、偏航的顺序写这三个数。写成 (20°, 25°, 45°)：直观，只用三个数。欧拉角三次转动的顺序不能换：先绕竖轴转 90 度再绕身子的左右轴转 45 度，和反过来，结果不一样，固定角也一样；欧拉角和固定角各有 12 种，要写清楚是哪一种：第一次在 x、y、z 里选一根，之后每次和上一次不同，3 × 2 × 2 = 12；三根轴都不一样的 6 种，比如 z-y-x，头尾是同一根轴的 6 种，比如 z-y-z；每一种欧拉角把顺序反过来就是一种固定角，摆出来一样，比如欧拉角 z-y-x 就是固定角 x-y-z');

      /* ══ 左：G1 + 转轴（虚线）和绕轴的一圈 ══ */
      var rob = E.el('g', {}, s);
      var back = E.el('g', {}, rob);
      var g1 = E.g1(rob, GX, GY, GH, { az: 100, el: 18, w: 230, up: 20 });
      var axG = E.el('g', {}, rob), front = E.el('g', {}, rob);
      var fr = E.frame3(rob, 5);
      // 欧拉角：沿身子自己的轴（转之前身子的朝向里取，每步不同）；固定角：世界的 x、y、z
      var euLines = COLS.map(function (c) { return K.paint(E.el('path', { fill: 'none', 'stroke-width': 3.5, 'stroke-dasharray': '10 7', 'stroke-linecap': 'round' }, axG), null, c); });
      var euRings = COLS.map(function (c) { return E.ring3(back, front, c, 4); });
      var axLines = COLS.map(function (c) { return K.paint(E.el('path', { fill: 'none', 'stroke-width': 3.5, 'stroke-dasharray': '10 7', 'stroke-linecap': 'round' }, axG), null, c); });
      var rings = COLS.map(function (c) { return E.ring3(back, front, c, 4); });
      var P = function (v) { return g1.project(v); };
      P.toward = E.view3(100, 18).toward;
      function seg(path, ax) {
        var p = P(ax.map(function (x) { return -0.62 * x; })), q = P(ax.map(function (x) { return 0.62 * x; }));
        path.setAttribute('d', 'M ' + p[0].toFixed(1) + ' ' + p[1].toFixed(1) + ' L ' + q[0].toFixed(1) + ' ' + q[1].toFixed(1));
      }
      [0, 1, 2].forEach(function (i) {
        seg(euLines[i], E.col3(BEFORE[i], i));
        euRings[i].set([0, 0, 0], E.col3(BEFORE[i], PL[i][0]), E.col3(BEFORE[i], PL[i][1]), 0.26, P, P);
        seg(axLines[i], AXES[i]);
        rings[i].set([0, 0, 0], PLANES[i][0], PLANES[i][1], 0.26, P, P);
      });

      /* ══ 右：标题 + 三行（按转的顺序） ══ */
      var head = E.txt(s, RX, 30, '', 24, E.INK, 'start', 900);
      var rows = [
        { name: '偏航', how: '绕竖着的轴 z：转身' },
        { name: '俯仰', how: '绕身子左右的轴 y：前倾' },
        { name: '横滚', how: '绕身子前后的轴 x：侧歪' }
      ].map(function (d, r) {
        var y = ROW[r].y, c = COLS[ROWAX[r]], g = E.el('g', {}, s);
        var box = K.paint(E.el('rect', { x: RX, y: y, width: 418, height: 58, rx: 12, 'stroke-width': 2.5 }, g), E.tint(c, 8, 'var(--demo-surface)'), c);
        var badge = E.el('g', {}, s);
        K.paint(E.el('circle', { cx: RX + 26, cy: y + 29, r: 16 }, badge), c);
        E.txt(badge, RX + 26, y + 36, String(r + 1), 20, 'var(--bg)', 'middle', 900);
        var nm = E.txt(g, RX + 54, y + 38, d.name, 26, c, 'start', 900);
        var how = E.txt(g, RX + 116, y + 37, d.how, 20, E.INK2, 'start', 700);
        var val = E.txt(g, RX + 404, y + 39, '0°', 28, E.INK, 'end', 900);
        return { g: g, box: box, badge: badge, nm: nm, how: how, val: val, y: y, ax: ROWAX[r] };
      });

      /* ══ 右下：一行写轴的顺序（名字用那根轴的颜色） ══ */
      function seqLine(parent, y, lead, names) {
        var t = E.txt(parent, RX, y, '', 20, E.INK2, 'start', 700);
        function sp(str, col, w) { var n = document.createElementNS('http://www.w3.org/2000/svg', 'tspan'); n.textContent = str; if (col) n.style.fill = col; if (w) n.style.fontWeight = w; t.appendChild(n); }
        sp(lead[0], E.INK, 900); sp(lead[1]);
        names.forEach(function (k, j) { if (j) sp(' → '); sp(['横滚', '俯仰', '偏航'][k], COLS[k], 900); });
        return t;
      }
      // 句 5：机器人里最常用的顺序
      var ordG = E.el('g', {}, s);
      E.txt(ordG, RX, 284, '机器人里最常用的顺序：', 21, E.INK, 'start', 900);
      var ordSeq = seqLine(ordG, 326, ['', ''], [2, 1, 0]);
      ordSeq.style.fontSize = '24px';
      // 句 6：欧拉角 / 固定角对照
      var cmp = E.el('g', {}, s);
      var cmpA = seqLine(cmp, 282, ['欧拉角', '（轴跟着身子）：'], [2, 1, 0]);
      var cmpB = seqLine(cmp, 322, ['固定角', '（轴不动）：'], [0, 1, 2]);
      var cmpOk = E.el('g', {}, cmp);
      E.check(cmpOk, RX + 2, 352, 20, E.GOOD, 4);
      E.txt(cmpOk, RX + 36, 370, '摆出来一模一样', 21, E.GOOD, 'start', 900);
      // 句 6「ROS 的机器人模型文件」：URDF 里 rpy 三个数的顺序（每个英文首字母用那根轴的颜色）
      var rpy = E.el('g', {}, s);
      E.txt(rpy, RX, 284, 'ROS 的 URDF 文件这样写：', 21, E.INK, 'start', 900);
      var rpyParts = [['R', 'oll 横滚'], ['P', 'itch 俯仰'], ['Y', 'aw 偏航']].map(function (d, i) {
        var x = RX + 6 + 140 * i;
        var t = E.txt(rpy, x, 330, '', 22, E.INK2, 'start', 700);
        var b = document.createElementNS('http://www.w3.org/2000/svg', 'tspan'); b.textContent = d[0]; b.style.fill = COLS[i]; b.style.fontWeight = 900; b.style.fontSize = '28px';
        var r = document.createElementNS('http://www.w3.org/2000/svg', 'tspan'); r.textContent = d[1];
        t.appendChild(b); t.appendChild(r);
        return { t: t, x: x };
      });

      /* ══ 右下：例子、好处 ══ */
      var tri = E.el('g', {}, s);
      var TXS = [462, 546, 630];
      [[420, '('], [500, ','], [584, ','], [670, ')']].forEach(function (d) { E.txt(tri, d[0], 300, d[1], 34, E.INK, 'middle', 700); });
      var triN = ['20°', '25°', '45°'].map(function (v, i) { return E.txt(tri, TXS[i], 300, v, 34, COLS[i], 'middle', 900); });
      var easy = E.txt(tri, 556, 344, '人一看就懂', 22, E.INK2, 'middle', 700);
      var goods = ['直观', '只用 3 个数'].map(function (t, i) {
        var g = E.el('g', {}, s), x = 400 + 196 * i;
        var tg = E.tag(g, x, 368, 180, 42, '', E.GOOD, { size: 22, ink: E.GOOD, weight: 900, rx: 12 });
        tg.text.textContent = t;
        tg.text.setAttribute('x', x + 104);
        var ck = E.check(g, x + 16, 380, 20, E.GOOD, 4);
        return { g: g, tg: tg, ck: ck, cx: x + 90 };
      });

      /* ══ 右下 ②：顺序不能换——两台小 G1（都绕身子自己的轴转） ══ */
      var ord = E.el('g', {}, s);
      var mini = [{ x: 470, lab: ['先绕竖轴转 90°', '再绕左右轴转 45°'] }, { x: 690, lab: ['先绕左右轴转 45°', '再绕竖轴转 90°'] }].map(function (d) {
        var g = E.el('g', {}, ord);
        var r = E.g1(g, d.x, 392, 200, { az: 135, el: 20, w: 200, up: 10 });   // 斜着看：往左倒、往前倒都看得出来
        var f3 = E.frame3(g, 4);
        var l1 = E.txt(g, d.x, 80, d.lab[0], 20, E.INK, 'middle', 900), l2 = E.txt(g, d.x, 106, d.lab[1], 20, E.INK, 'middle', 900);
        return { g: g, r: r, f3: f3, l1: l1, l2: l2, x: d.x };
      });
      var neq = E.txt(ord, 580, 300, '≠', 52, E.ERR, 'middle', 900);
      var ep3 = E.tag(ord, 478, 8, 204, 36, '欧拉角：顺序不能换', E.VIOLET, { size: 19, ink: E.INK, weight: 900, rx: 10 });
      var conv = E.halo(E.txt(ord, 580, 34, '欧拉角、固定角各 12 种：写清楚是哪一种', 20, E.HI, 'middle', 900));

      /* ══ 右 ③：12 种怎么数、是哪 12 种，每种欧拉角对应哪种固定角（每个字母用那根轴的颜色） ══ */
      var cnt = E.el('g', {}, s);   // 用整个画面（左边的 G1 收起），居中在 400
      var cConv = E.halo(E.txt(cnt, 400, 34, '欧拉角、固定角各 12 种：写清楚是哪一种', 20, E.HI, 'middle', 900));
      var cRule = E.txt(cnt, 400, 78, '第一次 3 选 1，之后每次和上一次不同', 22, E.INK, 'middle', 900);
      var cMult = E.txt(cnt, 400, 122, '3 × 2 × 2 = 12 种', 30, E.HI, 'middle', 900);
      var AXC = { x: E.X, y: E.Y, z: E.Z };
      function seqTxt(parent, x, y, q) {
        var t = E.txt(parent, x, y, '', 26, E.INK2, 'middle', 900);
        q.split('').forEach(function (ch, j) {
          if (j) { var d = document.createElementNS('http://www.w3.org/2000/svg', 'tspan'); d.textContent = '-'; t.appendChild(d); }
          var n = document.createElementNS('http://www.w3.org/2000/svg', 'tspan'); n.textContent = ch; n.style.fill = AXC[ch]; t.appendChild(n);
        });
        return t;
      }
      // 一块 6 行：欧拉角 = 固定角（同一个转法，轴的顺序反过来）；最后一行（z-y-x / z-y-z）两格各带一个框
      function block(cx, title, list) {
        var g = E.el('g', {}, cnt), EX = cx - 72, FX = cx + 72, Y0 = 232, DY = 34;
        var tt = E.txt(g, cx, 166, title, 22, E.INK2, 'middle', 900);
        var hE = E.txt(g, EX, 198, '欧拉角', 19, E.WARN, 'middle', 900), hF = E.txt(g, FX, 198, '固定角', 19, E.INK, 'middle', 900);
        function box(x) { return K.paint(E.el('rect', { x: x - 46, y: Y0 + 5 * DY - 26, width: 92, height: 35, rx: 8, 'stroke-width': 2.5 }, g), E.tint(E.HI, 18, 'var(--demo-surface)'), E.HI); }
        var hlE = box(EX), hlF = box(FX);
        var rows = list.map(function (q, k) {
          var y = Y0 + DY * k;
          return { e: seqTxt(g, EX, y, q), eq: E.txt(g, cx, y, '=', 24, E.MUT, 'middle', 900), fx: seqTxt(g, FX, y, q.split('').reverse().join('')) };
        });
        return { g: g, tt: tt, hE: hE, hF: hF, hlE: hlE, hlF: hlF, rows: rows };
      }
      var blkA = block(210, '三根轴都不一样：6 种', ['xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx']);
      var blkB = block(590, '头尾是同一根轴：6 种', ['xyx', 'xzx', 'yxy', 'yzy', 'zxz', 'zyz']);

      return {
        el: s,
        draw: function (st) {
          var f = E.fade;

          /* G1：欧拉角——偏航 → 俯仰 → 横滚（绕身子的轴）；「同样三个角」回到原样，按固定角再做一遍——横滚 → 俯仰 → 偏航（绕不动的轴）。
             两遍的式子都是 Rz(偏航)·Ry(俯仰)·Rx(横滚)：欧拉角是偏航先走完（绕身子的轴转是右乘），固定角是横滚先走完（绕不动的轴转是左乘） */
          var fixed = st >= T.fix0, back0 = 1 - f(st, T.fix0 + 0.1, 0.8);
          var a = [0, 1, 2].map(function (i) {
            var p = fixed ? (st < T.fx[0] ? back0 : f(st, T.fx[i], T.fxD)) : f(st, T.spins[i], T.spinD);
            return ANG[i] * p;
          });
          g1.setR(E.zyx(a[2], a[1], a[0]));
          fr.set(g1.frame(0.42));
          // 「身子自己的轴」「跟着身子」：骨盆上三根小箭头鼓一下
          var bodyB = Math.max(bump(st, T.body[0], 1.0), bump(st, T.body[1], 1.0));
          [0, 1, 2].forEach(function (i) {
            // 欧拉角那一遍：身子此刻的那根轴（虚线 + 一圈），下一根出来时收掉
            var nextAx = i === 2 ? T.axis[1] : i === 1 ? T.axis[0] : T.body[0];
            var euOn = f(st, T.axis[i], 0.4) * (1 - f(st, nextAx, 0.4));
            E.op(euLines[i], euOn); euRings[i].op(euOn);
            // 固定角那一遍：「绕不动的轴」三根世界的轴淡淡亮着，正在转的那根全亮、带一圈
            var fxNext = i < 2 ? T.fx[i + 1] : T.same + 0.6;
            var stepOn = f(st, T.fx[i] - 0.3, 0.3) * (1 - f(st, fxNext, 0.3));
            var allOn = 0.45 * f(st, T.fixAx, 0.4) * (1 - f(st, T.same + 0.6, 0.4));
            E.op(axLines[i], Math.max(allOn, stepOn)); rings[i].op(stepOn);
            fr.arrows[i].line.style.strokeWidth = 5 + 3 * Math.max(bodyB, euOn * f(st, T.spins[i] - 0.2, 0.2) * (1 - f(st, T.spins[i] + T.spinD, 0.3)));
          });

          /* 右：标题、三行 */
          head.textContent = st < T.fix0 + 0.2 ? '按顺序转三次：' : st < T.triple ? '固定角：绕不动的轴，③ → ② → ①' : '按顺序转三次：';
          var dip = [T.fix0 + 0.2, T.triple].reduce(function (m, x) { return Math.max(m, bump(st, x - 0.2, 0.4)); }, 0);
          E.op(head, f(st, T.head, 0.4) * (1 - f(st, T.order, 0.4)) * (1 - 0.9 * dip));
          rows.forEach(function (r, k) {
            var i = r.ax;
            var bb = bump(st, T.axisB, 0.6) + bump(st, T.angB, 0.6);
            E.scaleAt(r.badge, RX + 26, r.y + 29, f(st, T.nums + 0.18 * k, 0.3) * (1 + 0.25 * bb));
            E.op(r.g, f(st, T.rows[k], 0.35) * (1 - f(st, T.order, 0.4)));
            E.op(r.badge, f(st, T.nums + 0.18 * k, 0.3) * (1 - f(st, T.order, 0.4)));
            r.val.textContent = Math.round(a[i] / D) + '°';
            var fl = bump(st, T.ordN[k], 0.7) + bump(st, T.fx[i] - 0.1, 0.8) + bump(st, T.rpy[i + 1], 0.7) + bump(st, T.flash[i], 0.8) + bump(st, T.flash2[i], 0.8);
            r.box.style.strokeWidth = 2.5 + 3 * Math.min(1, fl);
            E.glow(r.box, COLS[i], fl > 0.05);
          });

          /* 右下：句 5 顺序、句 6 对照、RPY */
          E.op(ordG, f(st, T.ord, 0.35) * (1 - f(st, T.fix0, 0.3)));
          var cmpOut = 1 - f(st, T.rpy[0] - 0.4, 0.3);   // 几块说明前一块淡完、后一块才出来，不同时出现
          E.op(cmpA, f(st, T.fix0 + 0.35, 0.3) * cmpOut); E.op(cmpB, f(st, T.fixAx, 0.3) * cmpOut);
          E.op(cmpOk, f(st, T.same, 0.3) * cmpOut);
          E.scaleAt(cmpB, RX, 316, 1 + 0.06 * bump(st, T.fixName, 0.8));   // 「固定角」念到时这一行鼓一下
          E.op(rpy, f(st, T.rpy[0], 0.35) * (1 - f(st, T.triple - 0.35, 0.3)));
          rpyParts.forEach(function (p, i) {
            p.t.style.fill = f(st, T.rpy[i + 1], 0.3) > 0.5 ? E.INK : '';
            E.scaleAt(p.t, p.x + 40, 322, 1 + 0.12 * bump(st, T.rpy[i + 1], 0.7));
          });

          /* 右下：(20°, 25°, 45°)、人一看就懂、两块绿牌；「顺序不能换」起换成两台小 G1 */
          var ex = f(st, T.triple, 0.35) * (1 - f(st, T.order, 0.4));
          E.op(tri, ex);
          triN.forEach(function (n, i) { E.scaleAt(n, TXS[i], 290, 1 + 0.14 * Math.max(bump(st, T.flash[i], 0.8), bump(st, T.flash2[i], 0.8))); });   // 鼓一下不碰到两边的逗号
          E.op(easy, f(st, T.easy, 0.3));
          goods.forEach(function (g, i) {
            E.pop(g.g, g.cx, 389, f(st, T.good[i], 0.3) * (1 - f(st, T.order, 0.4)), 0.85);
            E.drawOn(g.ck, f(st, T.good[i] + 0.2, 0.35));
            g.tg.rect.style.strokeWidth = 2 + 2.5 * bump(st, T.best, 0.9);
          });

          // 两台小 G1，都绕身子自己的轴转（后做的右乘）：左边先 Rz(90°) 再 Ry(45°) → Rz·Ry；右边先 Ry(45°) 再 Rz(90°) → Ry·Rz
          E.op(ord, f(st, T.order, 0.4));
          var A1 = f(st, T.runA, T.runD), A2 = f(st, T.runA + T.runD + 0.1, T.runD);
          var B1 = f(st, T.runB, T.runD), B2 = f(st, T.runB + T.runD + 0.1, T.runD);
          mini[0].r.setR(E.mul3(E.Rz(90 * D * A1), E.Ry(45 * D * A2)));
          mini[1].r.setR(E.mul3(E.Ry(45 * D * B1), E.Rz(90 * D * B2)));
          mini.forEach(function (m, i) {
            m.f3.set(m.r.frame(0.36));
            var a1 = i ? B1 : A1, a2 = i ? B2 : A2, s0 = i ? T.runB : T.runA;
            m.l1.style.fill = st >= s0 && a1 < 1 ? 'var(--hi)' : '';
            m.l2.style.fill = a1 >= 1 && a2 > 0 && a2 < 1 ? 'var(--hi)' : '';
          });
          E.pop(neq, 580, 284, f(st, T.neq, 0.3) * (1 - f(st, T.cnt, 0.4)), 0.5);
          mini.forEach(function (m) { E.op(m.g, 1 - f(st, T.cnt, 0.4)); });   // 句 10：两台小 G1、左边的 G1 收起，整个画面换上 12 种
          E.op(rob, 1 - f(st, T.cnt, 0.4));
          E.op(cnt, f(st, T.cnt + 0.4, 0.3));
          E.op(cRule, f(st, T.rule, 0.3)); E.op(cMult, f(st, T.mult, 0.3));
          // 句 11 先出欧拉角一列，句 12「固定角」每行补上「= 固定角」
          [[blkA, T.r1, T.exF], [blkB, T.r2, T.pal]].forEach(function (b, n) {
            var B = b[0], tt = b[1];
            E.op(B.tt, f(st, tt[0], 0.3)); E.op(B.hE, f(st, tt[1], 0.3)); E.op(B.hF, f(st, T.fixCol, 0.3));
            B.rows.forEach(function (r, k) {
              E.op(r.e, f(st, tt[1] + 0.08 * k, 0.25));
              var fo = f(st, T.fixCol + 0.08 * k, 0.25);
              E.op(r.eq, fo); E.op(r.fx, fo);
              r.eq.style.fill = n && bump(st, T.palRev, 1.2) > 0.3 ? 'var(--hi)' : '';   // 「轴的顺序不变」：右块等号亮一下
            });
            E.op(B.hlE, f(st, tt[2], 0.3));
            E.op(B.hlF, f(st, b[2], 0.3));
          });
          // 「固定角也是这样」小牌换字，换字时先淡下去再亮回来
          ep3.text.textContent = st < T.fixToo ? '欧拉角：顺序不能换' : '固定角：顺序也不能换';
          E.pop(ep3.g, 580, 26, f(st, T.ep3, 0.3) * (1 - f(st, T.conv - 0.3, 0.3)) * (1 - 0.9 * bump(st, T.fixToo - 0.2, 0.4)), 0.85);
          E.op(conv, f(st, T.conv, 0.35) * (1 - f(st, T.cnt, 0.4)));   // 句 10 起换成居中的那一行（cConv）
        }
      };
    }
  };
})();
