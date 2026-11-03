// 011 第 7 幕：比 D-H 参数好在哪（§L1 自测题 3；《Modern Robotics》式 (4.4)、4.1.1 节、附录 C、C.6 节、4.5 节）。
// 左边先是前几幕的网格 + UR5（零位），句 7 起换成斜着看的 UR5（第 2、3、4 根轴平行，例 4.5）；右栏一组组换。
// 句 1「第三集、第七集」「每过一个关节，就是一次变换」；「给每一节连杆，都摆一个坐标系」{0}、{1}、{2}、{b} 依次出来（MR 图 4.1 的摆法：
//    连杆坐标系放在关节上、x 沿着连杆）。句 2「相邻两节之间，一个四乘四」右栏出 T = T₀₁(θ₁) T₁₂(θ₂) T₂b，「一节一节乘到手」网格上三段箭头依次画。
// 句 3「严格的规矩」「德纳维特、哈滕伯格参数」右栏写 D-H 参数，「四个数」写 a、α、d、φ（附录 C）。
// 句 4「不用给每一节摆坐标系」{1}、{2} 打橙 × 淡掉；「零位时、手的位姿」{b} 鼓一下写 M；「螺旋轴」肩、肘出 ⊙ 和 S₁、S₂；「随便选」{0}、{b} 各晃一下。
// 句 5–6 右栏写螺旋轴就是关节轴、转动 / 移动一样写；雅可比的每一列 = 此刻的螺旋轴（附录 C.6）。
// 句 7「几乎平行」右栏写；「这台机械臂的第二、三、四根轴」左边换成斜着看的 UR5，S₂、S₃、S₄ 亮、其余变淡。句 8 数字的代价。
// 句 8–13（2026-11-03 用户要求「D-H 参数过一个实例体现不好用的地方」，附录 C.1、C.6）：左边换成从上往下看的第 2、3 根轴（相距画 80 px = 0.425 米），
// 平行时画几条公垂线（无数条，挑中间那条）；第 3 根轴绕转点歪过去（角度画大了，正好在画面里相交），交点出橙点；弧长 / 半径用紫色虚线弧标出；
// D-H 原点（亮黄点）从挑的那条公垂线上端搬到交点，歪向另一边再搬到另一边；右栏两组算式。句 14（代价）左边换回 UR5。
// 时刻取自配音的逐词时间，见 ../ep011.py。
(function () {
  var E = window.EP011, K = E.K, H = E.HEX;
  var D = Math.PI / 180, A = E.ARM, RX = A.RX, X = RX + 10;
  var T = {
    ep37: 0.7, each: 2.67, xf: 4.48, seq: 5.11, common: 6.85, put: 8.36, frames: [8.36, 9.01, 9.73, 10.23],
    adj: 12.06, m44: 13.8, ang: 15.34, base: 16.36, hops: [16.94, 17.3, 17.69],
    rule: 21.26, strict: 21.59, dh: 22.91, dh2: 24.55, four: 26.97, outA: 28.4,
    poe: 28.72, noLink: 30.07, only: 31.77, M: 32.95, S: 34.84, free: 35.67, freeW: [37.44, 38.6],
    axis: 39.63, joint: 40.49, see: 41.22, rp: 43.12, same: 45.04,
    jac: 47.51, col: 50.23, now: 52.59, ep13: 55.04,
    flaw: 57.22, par: 60.24, tilt: 61.38, big: 63.62, hard: 64.2, ur: 65.59, s234: 66.74,
    // 句 8–13（2026-11-03 用户要求「D-H 参数过一个实例体现不好用的地方」）：第 2、3 根轴相距 0.425 米、本来平行，第 3 根轴在两轴共同的平面里歪 0.1°
    ex: 70.94, ax23: 71.79, dist: 73.5, par2: 75.11, normals: 75.78, inf: 76.5, pick: 77.29,
    assem: 80.26, third: 81.41, plane: 83.42, tiltA: [84.24, 85.3], deg01: 84.49, far: 86.45, meet: 87.32,
    howfar: 88.96, ep10: 90.0, arc: 91.09, rad: 94.37, r57: 95.5, div: 97.3, r175: 99.66,
    small: 102.68, d425: 103.8, arcEq: 105.28, radius: 106.32, r243: 110.06,
    rule2: 112.39, a0: 114.4, origin: 115.7, moveO: [116.44, 117.6], far243: 118.54, other: [120.08, 121.4], opp: 121.52,
    poeE: 125.17, s3: 126.72, six6: 127.84, max: 128.52, v175: 129.57, small2: 131.15, jump: 133.59, hard2: 134.5,
    cost: 136.75, six: 139.88, dh3: 140.7, three: 142.64, plus: 143.32, four2: 144.88, least: 145.84, many: 147.18
  };
  var END = 149.84;
  var LAB = [[1, 0, -12, 'middle'], [0, -18, 6, 'end'], [0, -18, 6, 'end'], [1, 6, -12, 'start'], [0, 14, 10, 'start'], [1, 14, 6, 'start']];
  var LEN = [[-0.05, 0.32], [-0.2, 0.2], [-0.18, 0.18], [-0.16, 0.16], [-0.15, 0.15], [-0.14, 0.14]];

  E.scenes[6] = {
    title: '比 D-H 参数好在哪',
    dur: END,
    cues: [
      { at: 0, s: '第 3、7 集：每过一个关节就是一次变换。常见的做法：给每一节连杆摆一个坐标系' },
      { at: 11.96, s: '相邻两节之间一个 4 × 4，跟着关节的**角度**变：T = T₀₁ T₁₂ T₂b，从底座一节一节乘到手' },
      { at: 19.44, s: '摆坐标系的严格规矩：D-H 参数（Denavit–Hartenberg），相邻两节只用 4 个数' },
      { at: 28.62, s: '**指数积公式**不用给每一节摆坐标系：只要**零位**时手的**位姿**和各关节的**螺旋轴**' },
      { at: 39.53, s: '**螺旋轴**就是关节轴，看着就能写；转动关节、移动关节同一种写法' },
      { at: 47.41, s: '和雅可比一套说法：雅可比的每一列，就是关节轴此刻的**螺旋轴**（第 13 集）' },
      { at: 57.12, s: 'D-H 的毛病：相邻关节轴几乎平行时，轴歪一点点参数就大变、难测准；UR5 第 2、3、4 根轴就平行' },
      { at: 70.84, s: '例子：第 2、3 根轴相距 0.425 米、平行，公垂线有无数条，挑一条就行' },
      { at: 80.16, s: '装配时第 3 根轴在两轴共同的平面里歪了 0.1°：两根轴会在很远的地方相交' },
      { at: 88.86, s: '弧长 = 半径 × **角度**（第 10 集），角度用弧度：0.1 ÷ 57.3 ≈ 0.00175 弧度' },
      { at: 102.58, s: '角度很小，0.425 米约等于这段弧长：半径 ≈ 0.425 ÷ 0.00175 ≈ 243 米' },
      { at: 112.29, s: 'D-H：连杆长度 0.425 → 0，坐标系原点搬到 243 米外的交点；歪向另一边，跑到另一边 243 米外' },
      { at: 125.07, s: '**指数积公式**：第 3 根**螺旋轴**的 6 个数最多只变约 0.00175。D-H 参数大起大落，很难测准' },
      { at: 136.65, s: '代价是数字多：每根**螺旋轴** 6 个数；D-H 每个关节 3 + 1 = 4 个，最少，很多资料还在用' }
    ],
    build: function (K) {
      var s = K.sceneSvg('比 D-H 参数好在哪：常见的做法是给每一节连杆摆一个坐标系，相邻两节之间一个四乘四，T = T₀₁(θ₁) T₁₂(θ₂) T₂b，从底座一节一节乘到手；怎么摆有严格的规矩，叫 D-H 参数（Denavit–Hartenberg），相邻两节只用四个数 a、α、d、φ。指数积公式不用给每一节摆坐标系，只要零位时手的位姿和每个关节的螺旋轴，底座和手的坐标系可以随便选；螺旋轴就是关节轴，看着就能写，转动、移动关节同一种写法；雅可比的每一列就是关节轴此刻的螺旋轴。D-H 的毛病：相邻关节轴几乎平行时轴歪一点点参数就大变、难测准，UR5 第二、三、四根轴就平行。代价：每根螺旋轴六个数，D-H 每个关节三个加关节角共四个，最少，很多资料还在用');

      /* ── 左 ①：网格 + 两关节的胳膊 ── */
      var gL = E.el('g', {}, s);
      var under = E.el('g', {}, gL), armG = E.el('g', {}, gL);
      var arm = E.arm2(armG, 320), u = arm.u, V = arm.view;
      var P = E.plane(under, { ox: A.OX, oy: A.OY, u: u, x0: A.X0, x1: A.X1, y0: A.Y0, y1: A.Y1 }).P;
      var RIGID = ['forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link'];
      var top = E.el('g', {}, gL);
      // 连杆坐标系：{0} 灰、{1} {2} {b} 红绿
      var e1 = [1 / Math.sqrt(5), 2 / Math.sqrt(5)];
      function frame(x, y, ang, L, cx, cy, lab, ldx, ldy) {
        var g = E.el('g', {}, top), c = P(x, y);
        var ax = E.vec(g, cx, 4, { head: 12 }), ay = E.vec(g, cy, 4, { head: 12 });
        var t = E.halo(E.txt(g, c[0] + ldx, c[1] + ldy, lab, 20, E.INK, 'middle', 900));
        function set(a) {
          var px = P(x + L * Math.cos(a), y + L * Math.sin(a)), py = P(x - L * Math.sin(a), y + L * Math.cos(a));
          ax.set(c[0], c[1], px[0], px[1]); ay.set(c[0], c[1], py[0], py[1]);
        }
        set(ang);
        return { g: g, set: set, lab: t, c: c, a0: ang };
      }
      var F0 = frame(0, 0, 0, 0.9, E.INK2, E.INK2, '{0}', 34, 30);   // 右下：左下是 T₀₁ 的弯箭头（线压字检查）
      var F1 = frame(0, 0, Math.atan2(2, 1), 0.75, E.X, E.Y, '{1}', -36, -10);
      var F2 = frame(1, 2, 0, 0.75, E.X, E.Y, '{2}', -14, -40);
      var Fb = frame(3, 2, 0, 0.75, E.X, E.Y, '{b}', 30, -30);
      var x1 = E.cross(top, F1.c[0] + 12, F1.c[1] - 16, 8, E.ERR, 4), x2 = E.cross(top, F2.c[0] + 14, F2.c[1] - 14, 8, E.ERR, 4);   // 叉在坐标系两根箭头中间，不压标签
      var mLab = E.halo(E.txt(top, Fb.c[0] + 34, Fb.c[1] + 26, 'M', 22, E.GOOD, 'start', 900)); mLab.style.fontFamily = 'KaTeX_Math, serif'; mLab.style.fontStyle = 'italic';
      // 一节一节：三段弯箭头
      function hop(a, b, bend) {
        var pa = P(a[0], a[1]), pb = P(b[0], b[1]), mx = (pa[0] + pb[0]) / 2 + bend[0], my = (pa[1] + pb[1]) / 2 + bend[1];
        return K.paint(E.el('path', { d: 'M ' + pa[0].toFixed(1) + ' ' + pa[1].toFixed(1) + ' Q ' + mx.toFixed(1) + ' ' + my.toFixed(1) + ' ' + pb[0].toFixed(1) + ' ' + pb[1].toFixed(1), fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '6 5' }, top), null, E.HI);
      }
      var hops = [hop([-0.4, -0.5], [-0.3, 0.4], [-26, 0]), hop([0.2, 0.6], [0.8, 2.4], [-40, -6]), hop([1.3, 2.5], [2.8, 2.5], [0, -34])];
      var hopL = [E.halo(E.txt(top, P(-0.9, 0)[0], P(-0.9, 0)[1] + 6, 'T₀₁', 18, E.HI, 'end', 900)),
        E.halo(E.txt(top, P(-0.4, 1.7)[0], P(-0.4, 1.7)[1], 'T₁₂', 18, E.HI, 'end', 900)),
        E.halo(E.txt(top, P(2.05, 3.2)[0], P(2.05, 3.2)[1], 'T₂b', 18, E.HI, 'middle', 900))];
      // 螺旋轴 ⊙
      function joint(x, y, lab, dx, dy, anchor) {
        var g = E.el('g', {}, top), c = P(x, y);
        K.paint(E.el('circle', { cx: c[0], cy: c[1], r: 12, 'stroke-width': 3.5 }, g), E.S1, E.WARN);
        K.paint(E.el('circle', { cx: c[0], cy: c[1], r: 4 }, g), E.WARN);
        E.halo(E.txt(g, c[0] + dx, c[1] + dy, lab, 20, E.WARN, anchor || 'start', 900));
        return g;
      }
      var jS = joint(0, 0, 'S₁', -16, 30, 'end'), jE = joint(1, 2, 'S₂', 16, 30);   // S₁ 写在左下：右下是 {0}

      /* ── 左 ②：斜着看的 UR5 ── */
      var g3 = E.el('g', {}, s);
      var R6 = E.ur6(g3, 100, 352, 2.25, 32, 16, [0, 0, 440, 420]);
      R6.update(LAB, LEN);
      E.op(R6.handG, 0);
      var parL = E.halo(E.txt(g3, 60, 120, '第 2、3、4 根轴：平行', 22, E.WARN, 'start', 900));

      /* ── 左 ③：从上往下看第 2、3 根轴（句 8–13）。两轴相距画 80 px = 0.425 米；歪的角度画夸张了（25°，正好在画面里相交） ── */
      var gS = E.el('g', {}, s), YA = 170, YB = 250, PX = 230, XL = 40, XR = 420, PHI = Math.atan(80 / 170);
      E.txt(gS, XL, 70, '从上往下看：第 2、3 根轴', 19, E.INK2, 'start', 700);
      K.paint(E.el('path', { d: 'M ' + XL + ' ' + YA + ' H ' + XR, fill: 'none', 'stroke-width': 3.5, 'stroke-dasharray': '9 6' }, gS), null, E.WARN);
      E.txt(gS, XL - 8, YA + 7, 'S₂', 20, E.WARN, 'end', 900);
      var s3L = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3.5, 'stroke-dasharray': '9 6' }, gS), null, E.WARN);
      var s3T = E.txt(gS, 0, 0, 'S₃', 20, E.WARN, 'end', 900);
      var norms = [110, 170, 290, 350].map(function (x) { return K.paint(E.el('path', { d: 'M ' + x + ' ' + YA + ' V ' + YB, fill: 'none', 'stroke-width': 2 }, gS), null, E.MUT); });
      var chosen = K.paint(E.el('path', { d: 'M ' + PX + ' ' + YA + ' V ' + YB, fill: 'none', 'stroke-width': 3 }, gS), null, E.INK);
      var chosenL = E.halo(E.txt(gS, PX + 10, (YA + YB) / 2 + 7, '0.425 米', 19, E.INK, 'start', 900));
      var normL = E.txt(gS, PX, YB + 52, '公垂线：无数条，挑一条', 19, E.INK2, 'middle', 900);
      var tiltArc = E.arcArrow(gS, PX, YB, 46, E.OMEGA, 3, { head: 10 });
      var tiltL = E.halo(E.txt(gS, PX + 52, YB + 26, '0.1°（角度画大了）', 18, E.OMEGA, 'start', 900));
      var meetDot = K.paint(E.el('circle', { r: 7, 'stroke-width': 3 }, gS), E.S1, E.ERR);
      var meetL = E.halo(E.txt(gS, 0, 0, '相交', 19, E.ERR, 'middle', 900));
      var radArc = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '5 4' }, gS), null, E.VEL);
      var radL = E.halo(E.txt(gS, (PX + 400) / 2, YA - 12, '半径 ≈ 243 米', 19, E.VEL, 'middle', 900));
      var arcL = E.halo(E.txt(gS, PX - 30, (YA + YB) / 2 + 7, '弧长 ≈ 0.425 米', 18, E.VEL, 'end', 900));
      var oDot = K.paint(E.el('circle', { r: 7 }, gS), E.HI);
      var oL = E.halo(E.txt(gS, 0, 0, 'D-H 原点', 18, E.HI, 'middle', 900));

      /* ══ 右栏 ══ */
      var RK = E.rowKit();
      function rows(y0, lines) { var g = E.el('g', {}, s); return { g: g, r: lines.map(function (l) { return RK.line(g, X, y0 + l[0], l[1], l[2] || 20); }) }; }
      var gA = rows(36, [[0, [['第 3、7 集：', E.INK2], ['每过一个关节', E.INK, 900]]], [32, [['就是一次变换，按顺序乘起来', E.INK2]]],
        [76, [['常见的做法：', E.INK2], ['给每一节连杆', E.INK, 900]]], [108, [['摆一个坐标系', E.INK, 900]]]]);
      var fG = E.el('g', {}, gA.g);
      E.tex(fG, X, 206, 'T=T_{01}(\\textcolor{' + H.acc + '}{\\theta_1})\\,T_{12}(\\textcolor{' + H.acc + '}{\\theta_2})\\,T_{2b}', 26, E.INK, 'start', { w: 330, h: 54 });
      var fNote = RK.line(gA.g, X, 250, [['相邻两节一个 4 × 4，跟着', E.INK2], ['角度', E.OMEGA, 900], ['变', E.INK2]], 19);
      var fNote2 = RK.line(gA.g, X, 282, [['从底座，一节一节乘到手', E.INK2]], 19);
      var dG = rows(330, [[0, [['怎么摆：', E.INK2], ['D-H 参数', E.HI, 900]]], [28, [['（Denavit–Hartenberg）', E.INK2]], 17],
        [60, [['相邻两节 ', E.INK2], ['4 个数', E.INK, 900], ['：a、α、d、φ', E.INK2]], 19]]);
      dG.g.setAttribute('transform', 'translate(0 -10)');
      var gB = rows(36, [[0, [['指数积公式', E.HI, 900], ['：不用连杆坐标系', E.INK, 900]], 21],
        [36, [['只要：零位时手的位姿 ', E.INK2], ['M', E.GOOD, 900]]], [66, [['　　　各关节的', E.INK2], ['螺旋轴', E.WARN, 900]]],
        [100, [['底座、手的坐标系：', E.INK2], ['随便选', E.GOOD, 900]]],
        [148, [['螺旋轴 = 关节轴', E.WARN, 900], ['：看着就能写', E.INK2]]], [180, [['转动、移动关节：', E.INK2], ['同一种写法', E.INK, 900]]],
        [228, [['和雅可比一套说法：', E.INK2]]], [258, [['雅可比的每一列 = ', E.INK2], ['此刻的螺旋轴', E.WARN, 900]]], [288, [['（雅可比：第 13 集）', E.MUT, 700]], 17]]);
      var gC = rows(40, [[0, [['D-H 参数的毛病：', E.ERR, 900]], 21], [36, [['相邻关节轴', E.INK2], ['几乎平行', E.INK, 900], ['时', E.INK2]]],
        [66, [['轴歪一点点 → 参数大变', E.INK2]]], [96, [['很难测准', E.ERR, 900]]], [136, [['UR5：第 2、3、4 根轴就平行', E.WARN, 900]], 19]]);
      var gE1 = rows(40, [[0, [['举个例子：', E.INK2], ['第 2、3 根轴', E.WARN, 900]]], [34, [['相距 ', E.INK2], ['0.425 米', E.INK, 900], ['，平行', E.INK2]]],
        [68, [['公垂线有无数条：挑一条', E.INK2]]], [110, [['第 3 根轴歪了 ', E.INK2], ['0.1°', E.OMEGA, 900]]], [144, [['→ 在很远的地方相交', E.ERR, 900]]],
        [186, [['弧长 = 半径 × 角度', E.INK, 900], ['（第 10 集）', E.INK2]]], [220, [['0.1 ÷ 57.3 ≈ ', E.INK2], ['0.00175', E.OMEGA, 900], [' 弧度', E.INK2]]],
        [262, [['半径 ≈ 0.425 ÷ 0.00175 ≈ ', E.INK2], ['243 米', E.VEL, 900]]]]);
      var gE2 = rows(40, [[0, [['按 D-H 的规矩：', E.INK2]]], [34, [['连杆长度 a：0.425 → ', E.INK2], ['0', E.ERR, 900]]],
        [68, [['原点搬到交点：', E.INK2], ['243 米外', E.ERR, 900]]], [102, [['歪向另一边：', E.INK2], ['另一边 243 米外', E.ERR, 900]]],
        [148, [['指数积公式：第 3 根螺旋轴', E.HI, 900]]], [182, [['第 1 个数：0 → ', E.INK2], ['0.00175', E.GOOD, 900]]],
        [214, [['其余几乎不变：最多变约 0.00175', E.INK2]]], [256, [['那套参数却大起大落 → ', E.INK2], ['很难测准', E.ERR, 900]]]]);
      var gD = rows(222, [[0, [['代价：', E.INK2], ['用的数字多一些', E.INK, 900]], 21], [36, [['螺旋轴：每根 ', E.INK2], ['6 个数', E.WARN, 900]]],
        [66, [['D-H：每关节 3 个 + 关节角 = ', E.INK2], ['4 个', E.HI, 900]]], [104, [['数最少，很多资料还在用', E.INK2]], 19]]);

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }

      return {
        el: s,
        draw: function (st, t) {
          if (s.isConnected) RK.layout();   // 每帧量：「≈」是后备字体，第一次量的时候字宽还不对，后面的数压上来（线压字检查）
          var f = E.fade, b = E.bump;
          /* ── 左 ① ── */
          var onL = 1 - f(st, T.ur - 0.6, 0.6);
          E.op(gL, onL);
          arm.pose(0, 0); V.clearTint(); V.tint(RIGID, E.BODY, 0.45);
          [F0, F1, F2, Fb].forEach(function (F, k) { E.op(F.g, fr(st, T.frames[k])); });
          var gone = f(st, T.noLink + 0.3, 0.6);
          E.op(F1.g, fr(st, T.frames[1]) * (1 - gone)); E.op(F2.g, fr(st, T.frames[2]) * (1 - gone));
          var cx = f(st, T.noLink, 0.3) * (1 - f(st, T.noLink + 0.9, 0.3));
          E.drawOn(x1, cx > 0 ? 1 : 0); E.drawOn(x2, cx > 0 ? 1 : 0); E.op(x1, cx); E.op(x2, cx);
          var w = Math.sin(Math.PI * E.clamp((st - T.freeW[0]) / (T.freeW[1] - T.freeW[0]), 0, 1)) * 25 * D;
          F0.set(w); Fb.set(-w);
          var bm = b(st, T.M, 0.9);
          E.op(mLab, fr(st, T.M)); E.glow(Fb.g, E.HI, bm > 0.05);
          hops.forEach(function (h, k) { E.drawOn(h, f(st, T.hops[k], 0.4)); E.op(h, (f(st, T.hops[k], 0.1) > 0 ? 1 : 0) * (1 - f(st, T.noLink, 0.4))); E.op(hopL[k], fr(st, T.hops[k] + 0.2) * (1 - f(st, T.noLink, 0.4))); });
          E.op(jS, fr(st, T.S)); E.op(jE, fr(st, T.S + 0.2));
          var ja = Math.max(b(st, T.axis, 0.9), b(st, T.joint, 0.9));
          [jS, jE].forEach(function (g) { E.glow(g, E.WARN, ja > 0.05); });

          /* ── 左 ② ── */
          E.op(g3, Math.min(1, f(st, T.ur - 0.6, 0.6) * (1 - f(st, T.ex - 0.4, 0.4)) + f(st, T.cost - 0.2, 0.5)));
          /* ── 左 ③：两根轴 ── */
          E.op(gS, f(st, T.ex - 0.2, 0.4) * (1 - f(st, T.cost - 0.6, 0.4)));
          var th = PHI * (f(st, T.tiltA[0], T.tiltA[1] - T.tiltA[0]) - 2 * f(st, T.other[0], T.other[1] - T.other[0]));
          var tn = Math.tan(th);
          s3L.setAttribute('d', 'M ' + XL + ' ' + (YB + (PX - XL) * tn).toFixed(1) + ' L ' + XR + ' ' + (YB - (XR - PX) * tn).toFixed(1));
          // 歪向另一边时左端翘到 S₂ 旁边：S₃ 改写在右端下方（线压字检查）
          if (th >= 0) { s3T.setAttribute('x', XL - 8); s3T.setAttribute('y', (YB + (PX - XL) * tn + 7).toFixed(1)); s3T.style.textAnchor = 'end'; }
          else { s3T.setAttribute('x', XR); s3T.setAttribute('y', (YB - (XR - PX) * tn + 26).toFixed(1)); s3T.style.textAnchor = 'end'; }
          var npar = 1 - f(st, T.tiltA[0], 0.4);
          norms.forEach(function (n, k) { E.op(n, f(st, T.normals + 0.15 * k, 0.3) * npar); });
          E.op(chosen, f(st, T.dist, 0.3) * npar); E.op(chosenL, f(st, T.dist, 0.3) * npar); E.op(normL, f(st, T.inf, 0.3) * npar);
          tiltArc.set(0, Math.max(1e-3, th)); E.op(tiltArc.g, th > 0.01 ? f(st, T.deg01, 0.3) * (1 - f(st, T.rule2, 0.3)) : 0);
          E.op(tiltL, f(st, T.deg01, 0.3) * (1 - f(st, T.rule2, 0.3)));
          var mx = Math.abs(tn) > 0.05 ? PX + 80 / tn : null, inPanel = mx !== null && mx >= XL && mx <= XR;
          if (inPanel) { meetDot.setAttribute('cx', mx.toFixed(1)); meetDot.setAttribute('cy', YA); meetL.setAttribute('x', mx.toFixed(1)); meetL.setAttribute('y', YA - 18); }
          E.op(meetDot, inPanel ? f(st, T.meet, 0.3) : 0); E.op(meetL, inPanel ? f(st, T.meet, 0.3) * (1 - f(st, T.arc, 0.3)) : 0);
          // 弧和半径（句 10–11）：圆心在交点，从 S₂ 量到 S₃
          var R = Math.hypot(400 - PX, YB - YA), a0 = Math.PI, a1 = Math.PI + Math.atan((YB - YA) / (400 - PX));   // 从 S₂（朝左）量到挑的公垂线下端（S₃ 上的转点）
          var ap = []; for (var k = 0; k <= 24; k++) { var aa = a0 + (a1 - a0) * k / 24; ap.push([400 + R * Math.cos(aa), YA - R * Math.sin(aa)]); }
          radArc.setAttribute('d', E.poly(ap)); E.op(radArc, f(st, T.arc, 0.4) * (1 - f(st, T.rule2, 0.3)));
          E.op(arcL, f(st, T.arcEq, 0.3) * (1 - f(st, T.rule2, 0.3))); E.op(radL, f(st, T.r243, 0.3) * (1 - f(st, T.rule2, 0.3)));
          // D-H 坐标系原点（句 12）：从挑的那条公垂线上端，搬到交点；歪向另一边，搬到另一边
          var ox = PX;
          if (st >= T.moveO[0]) ox = PX + (400 - PX) * f(st, T.moveO[0], T.moveO[1] - T.moveO[0]);
          if (st >= T.other[0]) ox = 400 + (60 - 400) * f(st, T.other[0], T.other[1] - T.other[0]);
          oDot.setAttribute('cx', ox.toFixed(1)); oDot.setAttribute('cy', YA);
          oL.setAttribute('x', (ox < 150 ? ox - 12 : ox > 330 ? ox + 12 : ox).toFixed(1)); oL.setAttribute('y', YA - 18);
          oL.style.textAnchor = ox < 150 ? 'start' : ox > 330 ? 'end' : 'middle';   // 靠左边时从点往右写，靠右边时往左写，不出画
          oL.textContent = st >= T.far243 ? (st >= T.opp ? '原点：另一边 243 米外' : '原点：243 米外') : 'D-H 原点';
          E.op(oDot, f(st, T.origin - 0.6, 0.3) * (1 - f(st, T.poeE, 0.3))); E.op(oL, f(st, T.origin - 0.6, 0.3) * (1 - f(st, T.poeE, 0.3)));
          R6.lines.forEach(function (ln, i) {
            var keep = i >= 1 && i <= 3, on = keep ? 1 : 0.2, pulse = keep ? b(st, T.s234 + 0.3 * (i - 1), 0.8) : 0;
            E.op(ln, on); E.op(R6.labs[i], on);
            ln.style.strokeWidth = 3.5 + 3 * pulse; E.glow(ln, E.WARN, pulse > 0.05 || (keep && st > T.s234 + 1.2 && st < T.cost));
          });
          E.op(parL, fr(st, T.s234));

          /* ── 右栏 ── */
          E.op(gA.g, fr(st, T.ep37) * (1 - f(st, T.outA, 0.4)));
          E.op(gA.r[0].parts[1], fr(st, T.each)); E.op(gA.r[1].g, fr(st, T.xf)); E.op(gA.r[2].g, fr(st, T.common)); E.op(gA.r[2].parts[1], fr(st, T.put)); E.op(gA.r[3].g, fr(st, T.put));
          E.op(fG, fr(st, T.adj)); E.op(fNote.g, fr(st, T.m44)); E.op(fNote.parts[1], fr(st, T.ang)); E.op(fNote2.g, fr(st, T.base));
          E.op(gA.r[0].g, fr(st, T.ep37) * (1 - f(st, T.rule - 0.3, 0.3))); E.op(gA.r[1].g, fr(st, T.xf) * (1 - f(st, T.rule - 0.3, 0.3)));
          E.op(gA.r[2].g, fr(st, T.common) * (1 - f(st, T.rule - 0.3, 0.3))); E.op(gA.r[3].g, fr(st, T.put) * (1 - f(st, T.rule - 0.3, 0.3)));
          E.op(dG.g, fr(st, T.rule) * (1 - f(st, T.outA, 0.4)));
          E.op(dG.r[0].parts[1], fr(st, T.dh)); E.op(dG.r[1].g, fr(st, T.dh2)); E.op(dG.r[2].g, fr(st, T.four));
          E.op(gB.g, fr(st, T.poe) * (1 - f(st, T.flaw - 0.3, 0.3)));
          var gbt = [T.poe, T.only, T.S, T.free, T.axis, T.rp, T.jac, T.col, T.ep13];
          gB.r.forEach(function (r, k) { E.op(r.g, fr(st, gbt[k])); });
          E.op(gB.r[0].parts[1], fr(st, T.noLink)); E.op(gB.r[1].parts[1], fr(st, T.M)); E.op(gB.r[3].parts[1], fr(st, T.freeW[0]));
          E.op(gB.r[4].parts[1], fr(st, T.see)); E.op(gB.r[5].parts[1], fr(st, T.same)); E.op(gB.r[7].parts[1], fr(st, T.now));
          E.op(gC.g, fr(st, T.flaw) * (1 - f(st, T.ex - 0.3, 0.3)));
          E.op(gE1.g, fr(st, T.ex) * (1 - f(st, T.rule2 - 0.3, 0.3)));
          var e1t = [T.ex, T.dist, T.inf, T.deg01, T.meet, T.arc, T.div, T.radius];
          gE1.r.forEach(function (r, k) { E.op(r.g, fr(st, e1t[k])); });
          E.op(gE1.r[0].parts[1], fr(st, T.ax23)); E.op(gE1.r[6].parts[1], fr(st, T.r175)); E.op(gE1.r[7].parts[1], fr(st, T.r243));
          E.op(gE2.g, fr(st, T.rule2) * (1 - f(st, T.cost - 0.3, 0.3)));
          var e2t = [T.rule2, T.a0, T.origin, T.other[0], T.poeE, T.max, T.v175, T.jump];
          gE2.r.forEach(function (r, k) { E.op(r.g, fr(st, e2t[k])); });
          E.op(gE2.r[2].parts[1], fr(st, T.far243)); E.op(gE2.r[3].parts[1], fr(st, T.opp)); E.op(gE2.r[7].parts[1], fr(st, T.hard2));
          var gct = [T.flaw, T.par, T.tilt, T.hard, T.ur];
          gC.r.forEach(function (r, k) { E.op(r.g, fr(st, gct[k])); });
          E.op(gC.r[2].g, fr(st, T.tilt)); E.op(gC.r[1].parts[1], fr(st, T.par));
          E.op(gD.g, fr(st, T.cost));
          var gdt = [T.cost, T.six, T.dh3, T.least];
          gD.r.forEach(function (r, k) { E.op(r.g, fr(st, gdt[k])); });
          E.op(gD.r[1].parts[1], fr(st, T.six)); E.op(gD.r[2].parts[1], fr(st, T.four2));
        }
      };
    }
  };
})();
