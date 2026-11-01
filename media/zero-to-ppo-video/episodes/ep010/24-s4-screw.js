// 010 第 4 幕：三维——边转边移的位姿。左边是斜着看的三维小场景（z 朝上）：竖直的螺旋轴（琥珀虚线）穿过原点，离轴 0.8 格的小方块（淡黄）带着自己的
// 三根小箭头（x 红、y 绿、z 蓝）；右栏上方是 4 × 4 的 T（第一、二、三列红、绿、蓝，位置一列紫），下方一行行写。
// 2026-11-01 按 009 的用户反馈重做：公式出来前先把符号逐个对一遍（「第五幕开头解释一下矩阵中各参数是什么」），转 1/4 圈的三列逐列说、逐列填。
// 句 1「四乘四的位姿」T 出来（先是问号，最下面一排 0、0、0、1），「左上角三乘三是朝向」框蓝，「右边一列是位置」框紫；
// 句 2「符号对一遍」右栏一行行写 ŝ、q、h、θ、I，同时左边指出来：「轴的方向」ŝ 箭头，「轴上的一个点」原点亮一下，「螺距」轴旁往上的绿箭头，
//    「转过的角度」轴上的小弯箭头 θ；
// 句 3「绕着螺旋轴的方向」轴亮一下，「罗德里格斯公式」右栏出公式，「正弦和余弦」下一行写；
// 句 4「位置分两份」右栏出 p = (I − R)q + θhŝ：「轴上的点转完不动」原点亮一下，「角度，乘上螺距」轴旁的绿箭头鼓一下；
// 句 5「拧螺丝」画出一圈多（450°）的螺旋线，原点出 x、y 两根细箭头，「竖直向上」「穿过原点」一行，「每转一圈，往上走一格」一行，量出 1 格；
// 句 6「螺距是转一弧度，往上走多少」右栏写 h 的意思，「一整圈」「六点二八弧度」「一除以六点二八」「零点一六格」逐个写出，「1 格」鼓一下
//    （2026-11-01 按一手资料改：螺距是沿轴走的速度 ÷ 转的速度，即转 1 弧度走多少，《Modern Robotics》图 3.19、例 4.20；
//    Murray–Li–Sastry 式 (2.40) 上方 h := d/θ。原先说「螺距是每转一圈往上走一格」，和位置那份「角度 × 螺距」对不上）；
// 句 7「转四分之一圈」方块沿螺旋线走 90°，「x 轴转到零、一、零」方块的红箭头鼓一下、T 第一列填数，y、z 同样逐列填，「这三列就是朝向」左上框亮；
// 句 8「轴上的点是零、零、零」「第一份乘出来是零」（2026-11-01 用户反馈「计算步骤需要展开」）「角度，乘上螺距」「一点五七，乘上零点一六」「零点二五」T 右边一列填 0、0、0.25，「一格的四分之一」写出；
// 句 9「转一整圈」方块接着走到 360°（T 跟着变），「正弦是零」「一减余弦也是零」右栏写出，「只剩单位矩阵」左上角框亮，
//    「六点二八，乘上零点一六」右栏写出，「一格」右边一列框亮；
// 句 10「不管转多少」方块接着走到 450°（T 跟着变），「每一列的长度是一、两两垂直」「航列式是一」右栏两行打勾。
// 三维按远近遮挡（同 006、009）：方块走到转轴后面时画在轴后面。时刻取自配音的逐词时间，见 ../ep010.py。
(function () {
  var E = window.EP010, K = E.K, H = E.HEX;
  var D = Math.PI / 180, RX = E.ARM.RX;
  var T = {
    mat: 3.36, rot: 4.52, pos: 6.32,
    sym: 8.59, sHat: 10.66, unit: 12.02, q: 13.67, h: 14.95, th: 15.64, I: 17.27, out1: 19.7,
    axis: 21.33, turn: 22.5, ang: 23.18, rod: 25.69, rodName: 27.2, sc: 28.66, out2: 31.3,
    two: 31.58, part1: 32.73, still: 34.93, part2: 35.82, hpart: 37.35, out3: 39.7,
    ex: 40.03, up: 42.07, orig: 43.04, pitch: 44.1, one: 45.65,
    hDef: 47.13, rev: 49.78, rev2: 50.65, hDiv: 51.95, h16: 53.74,
    q1: [55.89, 56.95], deg: 57.4, cx: [58.17, 58.95], cy: [60.21, 61.01], cz: [62.51, 63.45], three: 65.65,
    pos2: 68.03, qz: 69.79, p0: 71.88, p0v: 72.87, perTurn: 73.29, mul: 75.28, p25: 77.56, quarter: 78.39,
    full: [81.21, 82.35], sin0: 82.2, cos0: 83.19, rodI: 84.69, eye: 86.34, back3: 87.37, p1: 89.04, pmul: 89.7, p1v: 92.05,
    any: [95.31, 98.34], rule1: 99.61, rule1b: 101.99, rule2: 102.74, rule2b: 103.57
  };
  var END = 104.66, HP = 1 / (2 * Math.PI), RAD = 0.8, CUBE = 0.17;
  var proj = E.view3(28, 18), WX = 205, WY = 352, L = 150;
  function S3(v) { var q = proj(v); return [WX + L * q[0], WY - L * q[1]]; }
  function th(st) {
    return Math.PI / 2 * E.fade(st, T.q1[0], T.q1[1] - T.q1[0]) + 1.5 * Math.PI * E.fade(st, T.full[0], T.full[1] - T.full[0]) +
      Math.PI / 2 * E.fade(st, T.any[0], T.any[1] - T.any[0]);
  }

  E.scenes[3] = {
    title: '三维：边转边移的位姿',
    dur: END,
    cues: [
      { at: 0, s: '三维也一样：**矩阵指数**给出 4 × 4 的**位姿**，左上角 3 × 3 是**朝向**，右边一列是**位置**' },
      { at: 8.5, s: '符号：ŝ 轴的方向（长度 1），q 轴上一点，h **螺距**，θ 转过的**角度**，I 单位矩阵' },
      { at: 19.9, s: '**朝向**：绕**螺旋轴**的方向转过这个**角度**，罗德里格斯公式只用到 sin、cos' },
      { at: 31.49, s: '**位置**两份：轴上的点转完不动 + **角度** × **螺距**（沿着轴走）' },
      { at: 39.94, s: '拧螺丝：**螺旋轴**竖直向上、穿过原点，每转一圈往上走 1 格' },
      { at: 47.04, s: '**螺距**：转 1 弧度往上走多少；一整圈 ≈ 6.28 弧度，1 ÷ 6.28 ≈ 0.16 格' },
      { at: 55.8, s: '转 ¼ 圈 = 90°：x 轴 → (0, 1, 0)，y 轴 → (−1, 0, 0)，z 轴不动 (0, 0, 1)' },
      { at: 67.94, s: '**位置**：q = (0, 0, 0)，第一份 = 0；第二份 = **角度** × **螺距** = 1.57 × 0.16 ≈ 0.25' },
      { at: 81.12, s: '转一整圈：sin 360° = 0，1 − cos 360° = 0 → **朝向** = 单位矩阵；**位置** 6.28 × 0.16 ≈ 1 格' },
      { at: 94.85, s: '不管转多少：每列长度 1、两两垂直，行列式是 1' }
    ],
    build: function (K) {
      var s = K.sceneSvg('三维里矩阵指数给出四乘四的位姿：左上角三乘三是朝向，右边一列是位置；符号：ŝ 轴的方向（长度为一），q 轴上一点，h 螺距，θ 转过的角度，I 单位矩阵；朝向是绕螺旋轴方向转过这个角度，用罗德里格斯公式算；位置分两份：让轴上的点转完不动，再加上角度乘螺距；例子：螺旋轴竖直向上穿过原点，每转一圈往上走一格，螺距是转一弧度走多少，一除以 6.28 约 0.16 格；转四分之一圈，x 轴转到 (0, 1, 0)，y 轴转到 (−1, 0, 0)，z 轴不动，位置第一份：轴上的点 q 是原点，(I − R) 乘零是零；第二份是角度乘螺距 1.57 × 0.16 ≈ 0.25，正好一格的四分之一；转一整圈 sin 360° = 0、1 − cos 360° = 0，罗德里格斯公式只剩单位矩阵，位置 6.28 × 0.16 ≈ 1，往上一格；算出来的朝向总守着正交、行列式为一两条规矩');

      /* ══ 左：三维 ══ */
      var g3 = E.el('g', {}, s);
      var ground = K.paint(E.el('path', { fill: 'none', 'stroke-width': 1.8, 'stroke-dasharray': '5 6' }, g3), null, E.MUT);
      var gp = []; for (var k = 0; k <= 48; k++) { var a = 2 * Math.PI * k / 48; gp.push(S3([RAD * Math.cos(a), RAD * Math.sin(a), 0])); }
      ground.setAttribute('d', E.poly(gp, true));
      var hBack = K.paint(E.el('path', { fill: 'none', 'stroke-width': 2.2, 'stroke-dasharray': '4 6' }, g3), null, E.VEL);
      hBack.style.opacity = 0.55;
      var behind = E.el('g', {}, g3);                                   // 方块在转轴后面时挂到这里
      var a0 = S3([0, 0, -0.12]), a1 = S3([0, 0, 1.95]);
      var axis = K.paint(E.el('path', { d: E.poly([a0, a1]), fill: 'none', 'stroke-width': 3.5, 'stroke-dasharray': '9 7' }, g3), null, E.WARN);
      var sHat = E.vec(g3, E.WARN, 5, { head: 16 }), sh0 = S3([0, 0, 1.55]), sh1 = S3([0, 0, 2.02]);
      sHat.set(sh0[0], sh0[1], sh1[0], sh1[1]);
      var sHatLab = E.halo(E.txt(g3, sh1[0] + 14, sh1[1] + 18, 'ŝ', 26, E.WARN, 'start', 900));
      var axLab = E.halo(E.txt(g3, a1[0] - 16, a1[1] + 14, '螺旋轴', 22, E.WARN, 'end', 900));   // 轴顶左边：方块最高走到 1.25 格
      var hFront = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3.2 }, g3), null, E.VEL);
      var O = S3([0, 0, 0]);
      // 原点上 x、y 两根细箭头（句 5 起）：对照方块的小箭头转到了哪
      var wx = E.vec(g3, E.X, 2.5, { head: 10 }), wy = E.vec(g3, E.Y, 2.5, { head: 10 }), wx1 = S3([0.5, 0, 0]), wy1 = S3([0, 0.5, 0]);
      wx.set(O[0], O[1], wx1[0], wx1[1]); wy.set(O[0], O[1], wy1[0], wy1[1]);
      var wxL = E.halo(E.txt(g3, wx1[0] + 4, wx1[1] + 22, 'x', 20, E.X, 'middle', 900)), wyL = E.halo(E.txt(g3, wy1[0] + 12, wy1[1] + 4, 'y', 20, E.Y, 'start', 900));
      var oDot = K.paint(E.el('circle', { cx: O[0], cy: O[1], r: 6.5, 'stroke-width': 2.5 }, g3), E.HI, E.S1);
      var oLab = E.halo(E.txt(g3, O[0] - 14, O[1] + 28, 'q：原点', 20, E.HI, 'end', 900));
      // 转过的角度：绕轴的小弯箭头（轴上 1.3 格高处一圈椭圆的一段，蓝）
      var spin = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3.5 }, g3), null, E.OMEGA), spinHead = K.paint(E.el('path', {}, g3), E.OMEGA);
      var thLab = E.halo(E.txt(g3, 0, 0, 'θ', 26, E.OMEGA, 'start', 900));
      thLab.style.fontFamily = 'KaTeX_Math, serif'; thLab.style.fontStyle = 'italic';
      // 螺距：沿轴往上的箭头（绿）
      var rise = E.vec(g3, E.GOOD, 5, { head: 15 });
      var riseLab = E.halo(E.txt(g3, 0, 0, 'h：螺距', 20, E.GOOD, 'end', 900));
      // 句 5：一圈后上 1 格的量尺（离方块远一点：方块起步时它的 x 小箭头朝右伸出 0.36 格）
      var dimG = E.el('g', {}, g3), d0 = S3([RAD, 0, 0]), d1 = S3([RAD, 0, 1]);
      K.paint(E.el('path', { d: 'M ' + (d0[0] + 46) + ' ' + d0[1].toFixed(1) + ' H ' + (d0[0] + 58) + ' M ' + (d0[0] + 52) + ' ' + d0[1].toFixed(1) + ' V ' + d1[1].toFixed(1) + ' M ' + (d1[0] + 46) + ' ' + d1[1].toFixed(1) + ' H ' + (d1[0] + 58),
        fill: 'none', 'stroke-width': 2.5 }, dimG), null, E.GOOD);
      var dimLab = E.halo(E.txt(dimG, d0[0] + 66, (d0[1] + d1[1]) / 2 + 8, '1 格', 22, E.GOOD, 'start', 900));
      dimLab.__x = d0[0] + 80; dimLab.__y = (d0[1] + d1[1]) / 2;
      E.halo(E.txt(dimG, d0[0] + 66, (d0[1] + d1[1]) / 2 + 36, '（一圈）', 18, E.INK2, 'start', 700));
      // 小方块（淡黄）+ 自己的三根小箭头（x 红、y 绿、z 蓝）
      var blockG = E.el('g', {}, g3);
      var block = K.paint(E.el('path', { 'stroke-width': 2.5, 'stroke-linejoin': 'round' }, blockG), E.tint(E.BODY, 40, 'var(--demo-surface)'), E.BODY);
      var giz = E.gizmo(blockG, WX, WY, L, proj, { w: 3.5 });
      giz.labels.forEach(function (n) { n.style.display = 'none'; });   // 方块小，字母会压在方块和箭头上：三根小箭头照「画面配色」上色

      /* ══ 右栏：T + 公式 ══ */
      var RK = E.rowKit(), Rg = E.el('g', {}, s), X = RX + 10, MX = RX + 196, MY = 96;
      var mG = E.el('g', {}, Rg);
      E.tex(mG, X, MY + 10, 'T=', 30, E.INK, 'start', { w: 60, h: 50 });
      var Q4 = [['?', '?', '?', '?'], ['?', '?', '?', '?'], ['?', '?', '?', '?'], [0, 0, 0, 1]];
      var M = E.matN(mG, MX, MY, Q4, { size: 22, dx: 64, dy: 33, wide: [['−0.71', '−0.71', '0', '1.25'], ['0', '0', '0', '0'], ['0', '0', '1', '0'], [0, 0, 0, 1]],
        color: function (i, j) { return i === 3 ? E.MUT : [E.X, E.Y, E.Z, E.VEL][j]; } });
      M.cells.forEach(function (row) { row.forEach(function (n) { n.__col = n.style.fill; }); });
      var rotLab = E.txt(mG, M.pos(0, 1)[0], MY + 104, '朝向', 22, E.OMEGA, 'middle', 900);
      var posLab = E.txt(mG, M.pos(0, 3)[0], MY + 104, '位置', 22, E.VEL, 'middle', 900);
      var ROWS = 236, SZ = 20;
      var g1 = E.el('g', {}, Rg);
      var lg = [['ŝ', '：轴的方向，长度为 1', E.WARN, T.sHat], ['q', '：轴上的一个点', E.HI, T.q], ['h', '：螺距', E.GOOD, T.h], ['θ', '：转过的角度', E.OMEGA, T.th],
        ['I', '：单位矩阵（第 5 集）', E.INK, T.I]].map(function (r, i) {
        var g = E.el('g', {}, g1), y = ROWS - 4 + 34 * i;
        var sym = E.txt(g, X + 16, y, r[0], 22, r[2], 'middle', 900);
        sym.style.fontFamily = 'KaTeX_Math, serif'; sym.style.fontStyle = r[0] === 'ŝ' || r[0] === 'I' ? 'normal' : 'italic';
        E.txt(g, X + 32, y, r[1], SZ, E.INK, 'start', 700);
        return { g: g, at: r[3] };
      });
      var g2 = E.el('g', {}, Rg);
      var b1 = RK.line(g2, X, ROWS, [['朝向：绕 ', E.INK2], ['ŝ', E.WARN, 900], [' 转过角度 ', E.INK2], ['θ', E.OMEGA, 900]]);
      var b2 = E.el('g', {}, g2);
      E.tex(b2, X, ROWS + 56, 'R=I+\\textcolor{' + H.acc + '}{\\sin\\theta}\\,[\\hat s]+(1-\\textcolor{' + H.acc + '}{\\cos\\theta})\\,[\\hat s]^2', 21, E.INK, 'start', { w: 320, h: 44 });
      var b3 = RK.line(g2, X, ROWS + 104, [['罗德里格斯公式', E.OMEGA, 900]], 22);
      var b4 = RK.line(g2, X, ROWS + 140, [['只用到 ', E.INK2], ['sin、cos', E.OMEGA, 900]], 21);
      var b4b = RK.line(g2, X, ROWS + 170, [['[ŝ]：ŝ 的三个数带正负号排成 3 × 3', E.MUT, 700]], 17);
      var g3r = E.el('g', {}, Rg);
      var c1 = E.el('g', {}, g3r);
      E.tex(c1, X, ROWS + 16, 'p=\\textcolor{' + H.hi + '}{(I-R)\\,q}+\\textcolor{' + H.good + '}{\\theta\\,h\\,\\hat s}', 28, E.VEL, 'start', { w: 300, h: 56 });
      var c2 = RK.line(g3r, X, ROWS + 74, [['① ', E.HI, 900], ['轴上的点 q 转完不动', E.INK, 900]], 21);
      var c3 = RK.line(g3r, X, ROWS + 112, [['② ', E.GOOD, 900], ['沿着轴走：角度 × 螺距', E.INK, 900]], 21);
      var c4 = RK.line(g3r, X, ROWS + 150, [['原书写作 G(θ)v，展开就是这两份', E.MUT, 700]], 17);
      var g4 = E.el('g', {}, Rg);
      var e1 = RK.line(g4, X, ROWS + 6, [['ŝ', E.WARN, 900], ['：竖直向上；', E.INK, 900], ['q', E.HI, 900], ['：原点', E.INK, 900]], 21);
      var e3 = RK.line(g4, X, ROWS + 42, [['每转一圈，往上 ', E.INK, 900], ['1 格', E.GOOD, 900]], 21);
      // 螺距按原书定义是「沿轴走的速度 ÷ 转的速度」，即转 1 弧度走多少（《Modern Robotics》图 3.19、例 4.20 的 0.1 m/rad）
      var e4 = RK.line(g4, X, ROWS + 78, [['h', E.GOOD, 900], [' 螺距：转 1 弧度，往上走多少', E.INK, 900]], 21);
      var e5 = RK.line(g4, X, ROWS + 114, [['一整圈 ≈ ', E.INK2], ['6.28 弧度', E.OMEGA, 900]], 21);
      var e6 = RK.line(g4, X, ROWS + 150, [['h = 1 ÷ 6.28 ≈ ', E.INK2], ['0.16 格', E.GOOD, 900]], 21);
      var g4b = E.el('g', {}, Rg);
      var q1 = RK.line(g4b, X, ROWS + 6, [['转 ¼ 圈 = ', E.INK2], ['90°', E.OMEGA, 900], ['：', E.INK2]], 21);
      var q2 = RK.line(g4b, X, ROWS + 40, [['x 轴 → ', E.X, 900], ['(0, 1, 0)', E.X, 900]], 21);
      var q3 = RK.line(g4b, X, ROWS + 72, [['y 轴 → ', E.Y, 900], ['(−1, 0, 0)', E.Y, 900]], 21);
      var q4 = RK.line(g4b, X, ROWS + 104, [['z 轴不动：', E.Z, 900], ['(0, 0, 1)', E.Z, 900]], 21);
      var q5 = RK.line(g4b, X, ROWS + 142, [['这 3 列就是', E.INK2], ['朝向', E.OMEGA, 900]], 21);
      var g4c = E.el('g', {}, Rg);
      // 2026-11-01 用户反馈「计算步骤需要展开，不要直接给结果」：第一份写出 q = 0 再乘
      var p0r = RK.line(g4c, X, ROWS + 6, [['位置：轴过原点，', E.INK2], ['q = (0, 0, 0)', E.HI, 900]], 20);
      var p1 = RK.line(g4c, X, ROWS + 40, [['第一份：(I − R) × 0 = ', E.INK2], ['0', E.HI, 900]], 20);
      var p2 = RK.line(g4c, X, ROWS + 74, [['第二份 = ', E.INK2], ['角度', E.OMEGA, 900], [' × ', E.INK2], ['螺距', E.GOOD, 900]], 20);
      var p3 = RK.line(g4c, X, ROWS + 108, [['= 1.57 × 0.16 ≈ ', E.INK2], ['0.25', E.GOOD, 900]], 20);
      var p4 = RK.line(g4c, X, ROWS + 142, [['正好是 1 格的 ¼', E.INK2]], 20);
      var g4d = E.el('g', {}, Rg);
      // 转一整圈代进罗德里格斯公式：sin、1 − cos 都是 0，只剩 I；位置 6.28 × 0.16 ≈ 1
      var u1 = RK.line(g4d, X, ROWS + 6, [['转一整圈：', E.INK2], ['sin 360° = 0', E.OMEGA, 900]], 20);
      var u1b = RK.line(g4d, X, ROWS + 38, [['　　　　　', E.INK2], ['1 − cos 360° = 0', E.OMEGA, 900]], 20);
      var u2 = RK.line(g4d, X, ROWS + 72, [['R = I + 0 + 0 = ', E.INK2], ['单位矩阵', E.OMEGA, 900]], 20);
      var u2b = RK.line(g4d, X, ROWS + 106, [['3 根轴都回到原样', E.INK2]], 20);
      var u3 = RK.line(g4d, X, ROWS + 140, [['位置：6.28 × 0.16 ≈ ', E.INK2], ['1', E.VEL, 900], ['，上 1 格', E.INK2]], 20);
      var g5 = E.el('g', {}, Rg);
      var f1 = RK.line(g5, X, ROWS + 6, [['不管转多少，朝向都守规矩：', E.INK, 900]], 22);
      var f2 = RK.line(g5, X, ROWS + 50, [['① 每一列长度 1、两两垂直', E.INK, 900]], 21);
      var f2ok = E.check(f2.g, X + 286, ROWS + 32, 22, E.GOOD, 4);
      var f3 = RK.line(g5, X, ROWS + 92, [['② 行列式 = 1', E.INK, 900]], 21);
      var f3ok = E.check(f3.g, X + 150, ROWS + 74, 22, E.GOOD, 4);

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) { RK.layout(); laid = true; }
          var f = E.fade, b = E.bump, ang = th(st);

          /* ── 螺旋线：一圈多（450°）；句 5 起画出 ── */
          var hp = f(st, T.ex, 1.6), thMax = 2.5 * Math.PI * hp;
          var fr_ = '', bk = '', prev = null, n = Math.max(2, Math.ceil(thMax / 0.05));
          for (var k = 0; k <= n; k++) {
            var aa = thMax * k / n, p = S3([RAD * Math.cos(aa), RAD * Math.sin(aa), HP * aa]), isF = proj.depth([Math.cos(aa), Math.sin(aa), 0]) < 0, sg = p[0].toFixed(1) + ' ' + p[1].toFixed(1);
            if (isF) fr_ += (prev === true ? ' L ' : ' M ') + sg; else bk += (prev === false ? ' L ' : ' M ') + sg;
            if (prev !== null && prev !== isF) { if (isF) bk += ' L ' + sg; else fr_ += ' L ' + sg; }
            prev = isF;
          }
          hFront.setAttribute('d', thMax > 0.02 ? fr_ : ''); hBack.setAttribute('d', thMax > 0.02 ? bk : '');
          E.op(ground, 0.5);
          var ab = Math.max(b(st, T.axis, 0.9), b(st, T.up, 0.9), b(st, T.sHat, 0.9));
          axis.style.strokeWidth = 3.5 + 2.5 * ab; E.glow(axis, E.WARN, ab > 0.05);
          E.op(sHat.g, f(st, T.sHat, 0.3)); E.op(sHatLab, f(st, T.sHat, 0.3)); E.op(axLab, f(st, T.sHat, 0.3));
          sHat.line.style.strokeWidth = 5 + 3 * b(st, T.unit, 0.9);
          oDot.setAttribute('r', (6.5 + 4 * Math.max(b(st, T.q, 0.8), b(st, T.still, 0.8), b(st, T.orig, 0.8))).toFixed(2));
          E.op(oLab, f(st, T.q, 0.3));
          E.op(dimG, f(st, T.one, 0.3) * (1 - f(st, T.q1[0], 0.4)));
          E.scaleAt(dimLab, dimLab.__x, dimLab.__y, 1 + 0.25 * Math.max(b(st, T.hDiv, 0.9), b(st, T.one, 0.9)));
          var won = f(st, T.ex + 0.4, 0.4);
          [wx.g, wy.g].forEach(function (e) { E.op(e, won); }); E.op(wxL, 0); E.op(wyL, 0);   // 不写 x、y：字会压在地上的圆和螺旋线上，红、绿照「画面配色」认
          wx.line.style.strokeWidth = 2.5 + 2 * b(st, T.cx[0], 0.9); wy.line.style.strokeWidth = 2.5 + 2 * b(st, T.cy[0], 0.9);

          /* ── θ：小弯箭头（轴上 1.3 格高处一圈椭圆的 −30° → 70°） ── */
          var sp = f(st, T.th, 0.6), z0 = 1.3, r0 = 0.42, s0 = -30 * D, s1 = s0 + 100 * D * sp;
          var spts = []; for (k = 0; k <= 24; k++) { var q = s0 + (s1 - s0) * k / 24; spts.push(S3([r0 * Math.cos(q), r0 * Math.sin(q), z0])); }
          spin.setAttribute('d', sp > 0.02 ? E.poly(spts) : '');
          var tip = spts[24], pre = spts[22], tx = tip[0] - pre[0], ty = tip[1] - pre[1], tl = Math.hypot(tx, ty) || 1;
          tx /= tl; ty /= tl;
          spinHead.setAttribute('d', sp > 0.02 ? 'M ' + (tip[0] + tx * 6).toFixed(1) + ' ' + (tip[1] + ty * 6).toFixed(1) + ' L ' + (tip[0] - tx * 8 - ty * 7).toFixed(1) + ' ' + (tip[1] - ty * 8 + tx * 7).toFixed(1) +
            ' L ' + (tip[0] - tx * 8 + ty * 7).toFixed(1) + ' ' + (tip[1] - ty * 8 - tx * 7).toFixed(1) + ' Z' : '');
          var son = f(st, T.th, 0.3) * (1 - f(st, T.ex, 0.4));
          E.op(spin, son); E.op(spinHead, son);
          var tlp = S3([0.62, -0.2, z0]); thLab.setAttribute('x', tlp[0].toFixed(1)); thLab.setAttribute('y', (tlp[1] + 8).toFixed(1));
          E.op(thLab, f(st, T.th, 0.3) * (1 - f(st, T.ex, 0.4)));
          var tb = b(st, T.ang, 0.9); spin.style.strokeWidth = 3.5 + 2 * tb;
          // 螺距 h：沿轴往上（画在轴左边一点）
          var rp = f(st, T.h, 0.5), r0p = S3([-0.18, 0.12, 0.35]), r1p = S3([-0.18, 0.12, 0.35 + 0.7 * rp]);
          rise.set(r0p[0], r0p[1], r1p[0], r1p[1]); E.op(rise.g, (rp > 0 ? 1 : 0) * (1 - f(st, T.ex, 0.4)));
          rise.line.style.strokeWidth = 5 + 3 * b(st, T.hpart, 0.9);
          var rl = S3([-0.18, 0.12, 0.7]); riseLab.setAttribute('x', (rl[0] - 14).toFixed(1)); riseLab.setAttribute('y', (rl[1] + 6).toFixed(1));
          E.op(riseLab, f(st, T.h, 0.3) * (1 - f(st, T.ex, 0.4)));

          /* ── 小方块：中心 = R (0.8, 0, 0) + (0, 0, h θ)，朝向 R = 绕竖直轴转 θ；走到轴后面时画在轴后面 ── */
          var R = E.rot3([0, 0, 1], ang), c = E.mv3(R, [RAD, 0, 0]); c[2] += HP * ang;
          var cs = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(function (w) { var v = E.mv3(R, [w[0] * CUBE, w[1] * CUBE, 0]); return [c[0] + v[0], c[1] + v[1], c[2]]; });
          var topF = cs.map(function (p) { return S3([p[0], p[1], p[2] + CUBE]); }), botF = cs.map(function (p) { return S3([p[0], p[1], p[2] - CUBE]); });
          var d = '';
          for (k = 0; k < 4; k++) {
            var j = (k + 1) % 4, mid = [(cs[k][0] + cs[j][0]) / 2 - c[0], (cs[k][1] + cs[j][1]) / 2 - c[1], 0];
            if (proj.depth(mid) < 0) d += ' M ' + [topF[k], topF[j], botF[j], botF[k]].map(function (p) { return p[0].toFixed(1) + ' ' + p[1].toFixed(1); }).join(' L ') + ' Z';
          }
          d += ' M ' + topF.map(function (p) { return p[0].toFixed(1) + ' ' + p[1].toFixed(1); }).join(' L ') + ' Z';
          block.setAttribute('d', d);
          giz.set(R.map(function (row) { return row.map(function (x) { return 0.36 * x; }); }), [c[0], c[1], c[2] + CUBE]);
          var isBehind = proj.depth([c[0], c[1], 0]) > 0.05;
          if (isBehind && blockG.parentNode !== behind) behind.appendChild(blockG);
          if (!isBehind && blockG.parentNode !== g3) g3.appendChild(blockG);
          [[T.cx[0], 0], [T.cy[0], 1], [T.cz[0], 2]].forEach(function (cc) { giz.arrows[cc[1]].line.style.strokeWidth = 3.5 + 3 * b(st, cc[0], 1.0); });

          /* ── 右栏：T（转 ¼ 圈时逐列填；之后跟着方块变） ── */
          E.pop(mG, MX, MY, f(st, T.mat, 0.35), 0.9);
          var Tm = [[R[0][0], R[0][1], R[0][2], 0], [R[1][0], R[1][1], R[1][2], 0], [R[2][0], R[2][1], R[2][2], HP * ang]];
          var colAt = [T.cx[1], T.cy[1], T.cz[1]];
          for (var i = 0; i < 3; i++) for (j = 0; j < 4; j++) {
            var at = j < 3 ? colAt[j] : i < 2 ? T.p0v : T.p25;
            var cell = M.cells[i][j], on = st >= at, pq = M.pos(i, j), pc = f(st, at, 0.3);
            cell.textContent = on ? E.num(Math.abs(Tm[i][j]) < 5e-3 ? 0 : Tm[i][j]) : '?';
            cell.style.fill = on ? cell.__col : E.MUT;
            E.scaleAt(cell, pq[0], pq[1], on ? 1 + 0.2 * (1 - pc) : 1); E.op(cell, on ? pc : 1);
          }
          M.hl('rot', 0, 0, 2, 2, E.OMEGA, f(st, T.rot, 0.3) * (1 - f(st, T.sym, 0.4)) + b(st, T.three, 1.0) + b(st, T.eye, 1.0) + 0.6 * b(st, T.rule1, 1.0));
          M.hl('pos', 0, 3, 2, 3, E.VEL, f(st, T.pos, 0.3) * (1 - f(st, T.sym, 0.4)) + f(st, T.two, 0.3) * (1 - f(st, T.out3, 0.4)) + b(st, T.p25, 0.9) + b(st, T.p1v, 1.0));
          E.op(rotLab, f(st, T.rot, 0.3)); E.op(posLab, f(st, T.pos, 0.3));

          /* ── 右栏：符号、公式、例子 ── */
          E.op(g1, fr(st, T.sym) * (1 - f(st, T.out1, 0.4)));
          lg.forEach(function (r) { E.op(r.g, fr(st, r.at)); });
          E.op(g2, fr(st, T.out1 + 0.2) * (1 - f(st, T.out2, 0.4)));
          E.op(b1.g, 1); E.op(b1.parts[1], fr(st, T.axis)); E.op(b1.parts[2], fr(st, T.turn)); E.op(b1.parts[3], fr(st, T.ang));
          E.op(b2, fr(st, T.rod)); E.op(b3.g, fr(st, T.rodName)); E.op(b4.g, fr(st, T.sc)); E.op(b4b.g, fr(st, T.sc + 0.4));
          E.op(g3r, fr(st, T.two) * (1 - f(st, T.out3, 0.4)));
          E.op(c1, 1); E.op(c2.g, fr(st, T.part1)); E.op(c3.g, fr(st, T.part2)); E.op(c4.g, fr(st, T.hpart + 0.4));
          E.op(g4, fr(st, T.ex) * (1 - f(st, T.q1[0] - 0.3, 0.3)));
          E.op(e1.g, fr(st, T.up)); E.op(e1.parts[2], fr(st, T.orig)); E.op(e1.parts[3], fr(st, T.orig)); E.op(e3.g, fr(st, T.pitch)); E.op(e3.parts[1], fr(st, T.one));
          E.op(e4.g, fr(st, T.hDef)); E.op(e5.g, fr(st, T.rev)); E.op(e5.parts[1], fr(st, T.rev2)); E.op(e6.g, fr(st, T.hDiv)); E.op(e6.parts[1], fr(st, T.h16));
          E.op(g4b, fr(st, T.q1[0]) * (1 - f(st, T.pos2 - 0.3, 0.3)));
          E.op(q1.g, 1); E.op(q1.parts[1], fr(st, T.deg));
          E.op(q2.g, fr(st, T.cx[0])); E.op(q2.parts[1], fr(st, T.cx[1])); E.op(q3.g, fr(st, T.cy[0])); E.op(q3.parts[1], fr(st, T.cy[1]));
          E.op(q4.g, fr(st, T.cz[0])); E.op(q4.parts[1], fr(st, T.cz[1])); E.op(q5.g, fr(st, T.three));
          E.op(g4c, fr(st, T.pos2) * (1 - f(st, T.full[0] - 0.3, 0.3)));
          E.op(p0r.g, fr(st, T.pos2)); E.op(p0r.parts[1], fr(st, T.qz)); E.op(p1.g, fr(st, T.p0)); E.op(p1.parts[1], fr(st, T.p0v)); E.op(p2.g, fr(st, T.perTurn)); E.op(p3.g, fr(st, T.mul)); E.op(p3.parts[1], fr(st, T.p25)); E.op(p4.g, fr(st, T.quarter));
          E.op(g4d, fr(st, T.full[0]) * (1 - f(st, T.any[0] - 0.3, 0.3)));
          E.op(u1.g, fr(st, T.sin0)); E.op(u1b.g, fr(st, T.cos0)); E.op(u2.g, fr(st, T.rodI)); E.op(u2.parts[1], fr(st, T.eye));
          E.op(u2b.g, fr(st, T.back3)); E.op(u3.g, fr(st, T.pmul)); E.op(u3.parts[1], fr(st, T.p1v)); E.op(u3.parts[2], fr(st, T.p1v));
          E.op(g5, fr(st, T.any[0]));
          E.op(f1.g, 1); E.op(f2.g, fr(st, T.rule1)); E.drawOn(f2ok, f(st, T.rule1b, 0.3)); E.op(f3.g, fr(st, T.rule2)); E.drawOn(f3ok, f(st, T.rule2b, 0.3));
        }
      };
    }
  };
})();
