// 011 第 7 幕：比 D-H 参数好在哪（§L1 自测题 3；《Modern Robotics》式 (4.4)、4.1.1 节、附录 C、C.6 节、4.5 节）。
// 左边先是前几幕的网格 + UR5（零位），句 7 起换成斜着看的 UR5（第 2、3、4 根轴平行，例 4.5）；右栏一组组换。
// 句 1「第三集、第七集」「每过一个关节，就是一次变换」；「给每一节连杆，都摆一个坐标系」{0}、{1}、{2}、{b} 依次出来（MR 图 4.1 的摆法：
//    连杆坐标系放在关节上、x 沿着连杆）。句 2「相邻两节之间，一个四乘四」右栏出 T = T₀₁(θ₁) T₁₂(θ₂) T₂b，「一节一节乘到手」网格上三段箭头依次画。
// 句 3「严格的规矩」「德纳维特、哈滕伯格参数」右栏写 D-H 参数，「四个数」写 a、α、d、φ（附录 C）。
// 句 4「不用给每一节摆坐标系」{1}、{2} 打橙 × 淡掉；「零位时、手的位姿」{b} 鼓一下写 M；「螺旋轴」肩、肘出 ⊙ 和 S₁、S₂；「随便选」{0}、{b} 各晃一下。
// 句 5–6 右栏写螺旋轴就是关节轴、转动 / 移动一样写；雅可比的每一列 = 此刻的螺旋轴（附录 C.6）。
// 句 7「几乎平行」右栏写；「这台机械臂的第二、三、四根轴」左边换成斜着看的 UR5，S₂、S₃、S₄ 亮、其余变淡。句 8 数字的代价。
// 句 9（2026-11-04 用户要求：去掉 11-03 加的 D-H 实例，和之前一样只提一下它是什么、缺点是什么）：右栏最下面写「D-H 参数：本系列后面用不到」「想了解：教材附录 C」。
// 2026-11-04（观众零基础）：加「刚体（第 9 集）→ 连杆」两排、说「小臂」「连杆」时连杆亮；坐标系标签 {0}{1}{2}{b} 改成 0、1、2、手（{ } 的写法没讲过），
//    「编号」「右下角的小字」时编号和 T 的下标亮、右栏写「右下角小字：两节坐标系的编号」；去掉没讲过的 a、α、d、φ；转动 / 移动关节注明第 9 集；
//    雅可比拆成两句：第 3 集提过（关节转多快、手就走多快，中间隔着的矩阵）、和指数积公式一套说法（每一列 = 此刻的螺旋轴）。
// 时刻取自配音的逐词时间，见 ../ep011.py。
(function () {
  var E = window.EP011, K = E.K, H = E.HEX;
  var D = Math.PI / 180, A = E.ARM, RX = A.RX, X = RX + 10;
  var T = {
    ep37: 0.7, each: 2.67, xf: 4.48, seq: 5.11,
    rigid: 7.55, arm: 10.32, between: 11.07, link: 14.43,
    common: 16.08, put: 18.88, frames: [18.88, 19.3, 19.7, 20.1], num: 21.03,
    adj: 22.67, m44: 24.42, ang: 25.95, sub: 26.71, subW: 28.21, base: 29.99, hops: [30.79, 31.15, 31.5],
    rule: 34.86, strict: 35.35, dh: 36.65, dh2: 38.27, four: 40.64, dist: 41.5, dAng: 43.02, off: 44.55, jAng: 45.78, outA: 47.16,
    poe: 47.56, noLink: 48.64, only: 50.61, M: 52.16, S: 53.68, free: 54.52, freeW: [56.28, 57.02],
    axis: 58.47, joint: 59.35, see: 60.08, rp: 62.02, same: 64.66,
    jac3: 67.12, jacDef: 68.78, jacMat: 71.28, ep13: 72.92,
    jac: 75.17, col: 77.9, now: 80.19,
    flaw: 82.48, par: 85.79, tilt: 87.13, big: 88.88, hard: 89.46, ur: 90.85, s234: 92,
    cost: 96.2, six: 99.33, dh3: 100.15, three: 102.09, plus: 102.77, four2: 104.33, least: 105.76, many: 106.63,
    dhNo: 109.39, notUse: 111.66, skip: 113.33
  };
  var END = 115.37;
  var LAB = [[1, 0, -12, 'middle'], [0, -18, 6, 'end'], [0, -18, 6, 'end'], [1, 6, -12, 'start'], [0, 14, 10, 'start'], [1, 14, 6, 'start']];
  var LEN = [[-0.05, 0.32], [-0.2, 0.2], [-0.18, 0.18], [-0.16, 0.16], [-0.15, 0.15], [-0.14, 0.14]];

  E.scenes[6] = {
    title: '比 D-H 参数好在哪',
    dur: END,
    cues: [
      { at: 0, s: '第 3、7 集：每过一个关节，就是一次变换，按顺序乘起来' },
      { at: 7.45, s: '**连杆**：关节和关节之间，一节一节的刚体（刚体：第 9 集，怎么动都不变形）' },
      { at: 15.98, s: '常见的做法：给每一节连杆摆一个坐标系，从底座开始编号' },
      { at: 22.57, s: '相邻两节之间一个 4 × 4，跟着关节的**角度**变：T = T₀₁ T₁₂ T₂手，右下角的小字是两节坐标系的编号' },
      { at: 33.24, s: '摆坐标系的严格规矩：D-H 参数（Denavit–Hartenberg）。相邻两节只用 4 个数：两根关节轴的距离、夹角，沿着关节轴的偏移，再加上关节角' },
      { at: 47.46, s: '**指数积公式**不用给每一节摆坐标系：只要**零位**时手的**位姿**和各关节的**螺旋轴**' },
      { at: 58.37, s: '**螺旋轴**就是关节轴，看着就能写；转动关节、移动关节（第 9 集）同一种写法' },
      { at: 67.02, s: '雅可比（第 3 集提过）：关节转多快、手就走多快，中间隔着的那个矩阵（第 13 集讲）' },
      { at: 75.07, s: '和**指数积公式**一套说法：雅可比的每一列，就是关节轴此刻的**螺旋轴**' },
      { at: 82.38, s: 'D-H 的毛病：相邻关节轴几乎平行时，轴歪一点点参数就大变、难测准；UR5 第 2、3、4 根轴就平行' },
      { at: 96.1, s: '代价是数字多：每根**螺旋轴** 6 个数；D-H 每个关节 3 + 1 = 4 个，最少，很多资料还在用' },
      { at: 109.29, s: 'D-H 参数本系列后面用不到，就不展开了' }
    ],
    build: function (K) {
      var s = K.sceneSvg('比 D-H 参数好在哪：第 3、7 集说过，每过一个关节就是一次变换，按顺序乘起来；关节和关节之间一节一节的刚体（第 9 集）叫连杆；常见的做法是给每一节连杆摆一个坐标系，从底座开始编号 0、1、2，最后是手；相邻两节之间一个四乘四，T = T₀₁(θ₁) T₁₂(θ₂) T₂手，右下角的小字是两节坐标系的编号，从底座一节一节乘到手；怎么摆有严格的规矩，叫 D-H 参数（Denavit–Hartenberg），相邻两节只用四个数：两根关节轴的距离、夹角，沿着关节轴的偏移，再加上关节角。指数积公式不用给每一节摆坐标系，只要零位时手的位姿和每个关节的螺旋轴，底座和手的坐标系可以随便选；螺旋轴就是关节轴，看着就能写，第 9 集的转动、移动关节同一种写法；雅可比（第 3 集提过：关节转多快、手就走多快，中间隔着的矩阵，第 13 集讲）的每一列就是关节轴此刻的螺旋轴。D-H 的毛病：相邻关节轴几乎平行时轴歪一点点参数就大变、难测准，UR5 第二、三、四根轴就平行。代价：每根螺旋轴六个数，D-H 每个关节三个加关节角共四个，最少，很多资料还在用；D-H 参数本系列后面用不到');

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
      var F0 = frame(0, 0, 0, 0.9, E.INK2, E.INK2, '0', 34, 30);   // 右下：左下是 T₀₁ 的弯箭头（线压字检查）
      var F1 = frame(0, 0, Math.atan2(2, 1), 0.75, E.X, E.Y, '1', -36, -10);
      var F2 = frame(1, 2, 0, 0.75, E.X, E.Y, '2', -14, -40);
      var Fb = frame(3, 2, 0, 0.75, E.X, E.Y, '手', 30, -30);
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
        E.halo(E.txt(top, P(2.05, 3.2)[0], P(2.05, 3.2)[1], 'T₂手', 18, E.HI, 'middle', 900))];
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

      /* ══ 右栏 ══ */
      var RK = E.rowKit();
      function rows(y0, lines) { var g = E.el('g', {}, s); return { g: g, r: lines.map(function (l) { return RK.line(g, X, y0 + l[0], l[1], l[2] || 20); }) }; }
      var gA = rows(36, [[0, [['第 3、7 集：', E.INK2], ['每过一个关节', E.INK, 900]]], [30, [['就是一次变换，按顺序乘起来', E.INK2]]],
        [68, [['刚体（第 9 集）：怎么动都不变形', E.INK2]], 19], [98, [['连杆', E.HI, 900], ['：关节之间一节节的刚体', E.INK2]]],
        [134, [['常见的做法：', E.INK2], ['每节连杆摆一个坐标系', E.INK, 900]], 19]]);
      var fG = E.el('g', {}, gA.g);
      E.tex(fG, X, 214, 'T=T_{01}(\\textcolor{' + H.acc + '}{\\theta_1})\\,T_{12}(\\textcolor{' + H.acc + '}{\\theta_2})\\,T_{2\\text{手}}', 26, E.INK, 'start', { w: 330, h: 54 });
      var fNote = RK.line(gA.g, X, 256, [['相邻两节一个 4 × 4，跟着', E.INK2], ['角度', E.OMEGA, 900], ['变', E.INK2]], 19);
      var fNote3 = RK.line(gA.g, X, 286, [['右下角小字：', E.INK2], ['两节坐标系的编号', E.INK, 900]], 19);
      var fNote2 = RK.line(gA.g, X, 316, [['从底座，一节一节乘到手', E.INK2]], 19);
      // 2026-11-05（用户反馈：说「只用四个数」，就把四个数是什么列出来；附录 C 的连杆长度、扭角、偏距、关节角，用大白话写）
      var dG = rows(40, [[0, [['怎么摆：', E.INK2], ['D-H 参数', E.HI, 900]]], [28, [['（Denavit–Hartenberg）', E.INK2]], 17],
        [60, [['相邻两节，只用 ', E.INK2], ['4 个数', E.INK, 900], ['：', E.INK2]], 19],
        [90, [['两根关节轴的距离', E.INK, 900], ['、夹角', E.INK, 900]], 19], [118, [['沿着关节轴的偏移', E.INK, 900], ['、关节角', E.INK, 900]], 19]]);
      var gB = rows(36, [[0, [['指数积公式', E.HI, 900], ['：不用连杆坐标系', E.INK, 900]], 21],
        [36, [['只要：零位时手的位姿 ', E.INK2], ['M', E.GOOD, 900]]], [66, [['　　　各关节的', E.INK2], ['螺旋轴', E.WARN, 900]]],
        [100, [['底座、手的坐标系：', E.INK2], ['随便选', E.GOOD, 900]]],
        [148, [['螺旋轴 = 关节轴', E.WARN, 900], ['：看着就能写', E.INK2]]], [180, [['第 9 集的转动、移动关节：', E.INK2], ['同一种写法', E.INK, 900]], 18],
        [222, [['雅可比', E.INK, 900], ['（第 3 集提过）：', E.INK2]]], [250, [['关节转多快、手就走多快', E.INK2]], 19], [278, [['中间隔着的矩阵（第 13 集讲）', E.INK2]], 19],
        [312, [['和指数积公式一套说法：', E.INK2]]], [340, [['每一列 = ', E.INK2], ['此刻的螺旋轴', E.WARN, 900]]]]);
      var gC = rows(40, [[0, [['D-H 参数的毛病：', E.ERR, 900]], 21], [36, [['相邻关节轴', E.INK2], ['几乎平行', E.INK, 900], ['时', E.INK2]]],
        [66, [['轴歪一点点 → 参数大变', E.INK2]]], [96, [['很难测准', E.ERR, 900]]], [136, [['UR5：第 2、3、4 根轴就平行', E.WARN, 900]], 19]]);
      // 句 9（2026-11-04 用户要求）：D-H 参数本系列后面用不到——核对过发布清单 012–062、主路线 L2 以后、后面几集依据的 wiki 页都没有 D-H
      var gF = rows(380, [[0, [['D-H 参数：', E.INK2], ['本系列后面用不到', E.HI, 900]], 19]]);   // 2026-11-04 用户要求：不让观众自己去看别的书，删掉「想了解：教材附录 C」
      var gD = rows(222, [[0, [['代价：', E.INK2], ['用的数字多一些', E.INK, 900]], 21], [36, [['螺旋轴：每根 ', E.INK2], ['6 个数', E.WARN, 900]]],
        [66, [['D-H：每关节 3 个 + 关节角 = ', E.INK2], ['4 个', E.HI, 900]]], [104, [['数最少，很多资料还在用', E.INK2]], 19]]);

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) { RK.layout(); laid = true; }
          var f = E.fade, b = E.bump;
          /* ── 左 ① ── */
          var onL = 1 - f(st, T.ur - 0.6, 0.6);
          E.op(gL, onL);
          arm.pose(0, 0); V.clearTint(); V.tint(RIGID, E.BODY, 0.45);
          // 「小臂」小臂亮，「连杆」大臂、小臂（一节一节的刚体）一起亮（2026-11-04：连杆是这一幕新出的词）
          var lk = f(st, T.arm, 0.3) * (1 - f(st, T.common, 0.3));
          if (lk > 0.01) {
            V.tint(RIGID, E.BODY, 0.45 + 0.35 * lk);
            var lu = f(st, T.link, 0.3) * (1 - f(st, T.common, 0.3));
            if (lu > 0.01) V.tint(['upper_arm_link'], E.BODY, 0.8 * lu);
          }
          [F0, F1, F2, Fb].forEach(function (F, k) { E.op(F.g, fr(st, T.frames[k])); });
          var gone = f(st, T.noLink + 0.3, 0.6);
          E.op(F1.g, fr(st, T.frames[1]) * (1 - gone)); E.op(F2.g, fr(st, T.frames[2]) * (1 - gone));
          var cx = f(st, T.noLink, 0.3) * (1 - f(st, T.noLink + 0.9, 0.3));
          E.drawOn(x1, cx > 0 ? 1 : 0); E.drawOn(x2, cx > 0 ? 1 : 0); E.op(x1, cx); E.op(x2, cx);
          var w = Math.sin(Math.PI * E.clamp((st - T.freeW[0]) / (T.freeW[1] - T.freeW[0]), 0, 1)) * 25 * D;
          F0.set(w); Fb.set(-w);
          var bm = b(st, T.M, 0.9);
          E.op(mLab, fr(st, T.M)); E.glow(Fb.g, E.HI, bm > 0.05);
          // 「从底座开始编号」「右下角的小字」：坐标系的编号、T 的下标亮一下
          var numOn = (st >= T.num && st < T.num + 1.4) || (st >= T.sub && st < T.base);
          [F0, F1, F2, Fb].forEach(function (F) { E.glow(F.lab, E.HI, numOn); });
          hopL.forEach(function (l) { E.glow(l, E.HI, st >= T.sub && st < T.base); });
          hops.forEach(function (h, k) { E.drawOn(h, f(st, T.hops[k], 0.4)); E.op(h, (f(st, T.hops[k], 0.1) > 0 ? 1 : 0) * (1 - f(st, T.noLink, 0.4))); E.op(hopL[k], fr(st, T.hops[k] + 0.2) * (1 - f(st, T.noLink, 0.4))); });
          E.op(jS, fr(st, T.S)); E.op(jE, fr(st, T.S + 0.2));
          var ja = Math.max(b(st, T.axis, 0.9), b(st, T.joint, 0.9));
          [jS, jE].forEach(function (g) { E.glow(g, E.WARN, ja > 0.05); });

          /* ── 左 ② ── */
          E.op(g3, f(st, T.ur - 0.6, 0.6));
          R6.lines.forEach(function (ln, i) {
            var keep = i >= 1 && i <= 3, on = keep ? 1 : 0.2, pulse = keep ? b(st, T.s234 + 0.3 * (i - 1), 0.8) : 0;
            E.op(ln, on); E.op(R6.labs[i], on);
            ln.style.strokeWidth = 3.5 + 3 * pulse; E.glow(ln, E.WARN, pulse > 0.05 || (keep && st > T.s234 + 1.2 && st < T.cost));
          });
          E.op(parL, fr(st, T.s234));

          /* ── 右栏 ── */
          E.op(gA.g, fr(st, T.ep37) * (1 - f(st, T.outA, 0.4)));
          E.op(gA.r[0].parts[1], fr(st, T.each));
          var outR = 1 - f(st, T.rule - 0.3, 0.3);
          [T.ep37, T.xf, T.rigid, T.between, T.common].forEach(function (t0, k) { E.op(gA.r[k].g, fr(st, t0) * outR); });
          E.op(gA.r[3].parts[0], fr(st, T.link)); E.op(gA.r[4].parts[1], fr(st, T.put));
          E.op(fG, fr(st, T.adj)); E.op(fNote.g, fr(st, T.m44)); E.op(fNote.parts[1], fr(st, T.ang)); E.op(fNote2.g, fr(st, T.base) * (1 - f(st, T.rule - 0.7, 0.3)));   // 下面要写 D-H 参数，先收起
          E.op(fNote3.g, fr(st, T.sub) * (1 - f(st, T.rule - 0.7, 0.3))); E.op(fNote3.parts[1], fr(st, T.subW));
          E.op(dG.g, fr(st, T.rule) * (1 - f(st, T.outA, 0.4)));
          E.op(dG.r[0].parts[1], fr(st, T.dh)); E.op(dG.r[1].g, fr(st, T.dh2)); E.op(dG.r[2].g, fr(st, T.four));
          E.op(dG.r[3].g, fr(st, T.dist)); E.op(dG.r[3].parts[1], fr(st, T.dAng)); E.op(dG.r[4].g, fr(st, T.off)); E.op(dG.r[4].parts[1], fr(st, T.jAng));
          E.op(gB.g, fr(st, T.poe) * (1 - f(st, T.flaw - 0.3, 0.3)));
          var gbt = [T.poe, T.only, T.S, T.free, T.axis, T.rp, T.jac3, T.jacDef, T.jacMat, T.jac, T.col];
          gB.r.forEach(function (r, k) { E.op(r.g, fr(st, gbt[k])); });
          E.op(gB.r[0].parts[1], fr(st, T.noLink)); E.op(gB.r[1].parts[1], fr(st, T.M)); E.op(gB.r[3].parts[1], fr(st, T.freeW[0]));
          E.op(gB.r[4].parts[1], fr(st, T.see)); E.op(gB.r[5].parts[1], fr(st, T.same)); E.op(gB.r[10].parts[1], fr(st, T.now));
          E.op(gC.g, fr(st, T.flaw) * (1 - f(st, T.cost - 0.3, 0.3)));
          var gct = [T.flaw, T.par, T.tilt, T.hard, T.ur];
          gC.r.forEach(function (r, k) { E.op(r.g, fr(st, gct[k])); });
          E.op(gC.r[2].g, fr(st, T.tilt)); E.op(gC.r[1].parts[1], fr(st, T.par));
          E.op(gD.g, fr(st, T.cost));
          E.op(gF.g, fr(st, T.dhNo)); E.op(gF.r[0].parts[1], fr(st, T.notUse));
          var gdt = [T.cost, T.six, T.dh3, T.least];
          gD.r.forEach(function (r, k) { E.op(r.g, fr(st, gdt[k])); });
          E.op(gD.r[1].parts[1], fr(st, T.six)); E.op(gD.r[2].parts[1], fr(st, T.four2));
        }
      };
    }
  };
})();
