// 011 第 7 幕：比 D-H 参数好在哪（§L1 自测题 3；《Modern Robotics》式 (4.4)、4.1.1 节、附录 C、C.6 节、4.5 节）。
// 左边先是前几幕的网格 + UR5（零位），句 7 起换成斜着看的 UR5（第 2、3、4 根轴平行，例 4.5）；右栏一组组换。
// 句 1「第三集、第七集」「每过一个关节，就是一次变换」；「给每一节连杆，都摆一个坐标系」{0}、{1}、{2}、{b} 依次出来（MR 图 4.1 的摆法：
//    连杆坐标系放在关节上、x 沿着连杆）。句 2「相邻两节之间，一个四乘四」右栏出 T = T₀₁(θ₁) T₁₂(θ₂) T₂b，「一节一节乘到手」网格上三段箭头依次画。
// 句 3「严格的规矩」「德纳维特、哈滕伯格参数」右栏写 D-H 参数，「四个数」写 a、α、d、φ（附录 C）。
// 句 4「不用给每一节摆坐标系」{1}、{2} 打橙 × 淡掉；「零位时、手的位姿」{b} 鼓一下写 M；「螺旋轴」肩、肘出 ⊙ 和 S₁、S₂；「随便选」{0}、{b} 各晃一下。
// 句 5–6 右栏写螺旋轴就是关节轴、转动 / 移动一样写；雅可比的每一列 = 此刻的螺旋轴（附录 C.6）。
// 句 7「几乎平行」右栏写；「这台机械臂的第二、三、四根轴」左边换成斜着看的 UR5，S₂、S₃、S₄ 亮、其余变淡。句 8 数字的代价。
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
    cost: 70.94, six: 74.07, dh3: 74.89, three: 76.83, plus: 77.51, four2: 79.07, least: 80.03, many: 81.37
  };
  var END = 84.03;
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
      { at: 70.84, s: '代价是数字多：每根**螺旋轴** 6 个数；D-H 每个关节 3 + 1 = 4 个，最少，很多资料还在用' }
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
      var F0 = frame(0, 0, 0, 0.9, E.INK2, E.INK2, '{0}', -22, 28);
      var F1 = frame(0, 0, Math.atan2(2, 1), 0.75, E.X, E.Y, '{1}', -36, -10);
      var F2 = frame(1, 2, 0, 0.75, E.X, E.Y, '{2}', -14, -40);
      var Fb = frame(3, 2, 0, 0.75, E.X, E.Y, '{b}', 30, -30);
      var x1 = E.cross(top, F1.c[0] - 30, F1.c[1] - 40, 8, E.ERR, 4), x2 = E.cross(top, F2.c[0] - 30, F2.c[1] - 34, 8, E.ERR, 4);
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
      function joint(x, y, lab, dx, dy) {
        var g = E.el('g', {}, top), c = P(x, y);
        K.paint(E.el('circle', { cx: c[0], cy: c[1], r: 12, 'stroke-width': 3.5 }, g), E.S1, E.WARN);
        K.paint(E.el('circle', { cx: c[0], cy: c[1], r: 4 }, g), E.WARN);
        E.halo(E.txt(g, c[0] + dx, c[1] + dy, lab, 20, E.WARN, 'start', 900));
        return g;
      }
      var jS = joint(0, 0, 'S₁', 16, 30), jE = joint(1, 2, 'S₂', 16, 30);

      /* ── 左 ②：斜着看的 UR5 ── */
      var g3 = E.el('g', {}, s);
      var R6 = E.ur6(g3, 100, 352, 2.25, 32, 16, [0, 0, 440, 420]);
      R6.update(LAB, LEN);
      E.op(R6.handG, 0);
      var parL = E.halo(E.txt(g3, 60, 120, '第 2、3、4 根轴：平行', 22, E.WARN, 'start', 900));

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
          E.op(g3, f(st, T.ur - 0.6, 0.6));
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
          E.op(gC.g, fr(st, T.flaw) * (1 - f(st, T.cost - 0.3, 0.3)));
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
