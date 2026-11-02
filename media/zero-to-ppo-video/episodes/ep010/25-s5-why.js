// 010 第 5 幕：比一比——矩阵指数好在哪。左边先是第 1 幕的网格 + UR5，后半换成三张对照卡片（占满整幅）。
// 2026-11-01 按 005–009 的新配色复查：小臂淡黄，转动和角度蓝，打 × 用橙，手上的坐标架 x 红、y 绿、z 蓝；「九十除以二」右栏写 90° ÷ 2 = 45°。
// 句 1「两个朝向」小臂 0° 和 90° 两个位置画成虚线（第 5 集的两个朝向），「比如小臂原样和转了九十度」两个位置依次亮一下（交代 90 从哪来），「不能直接平均」右栏打橙 ×，「矩阵指数」右栏写 exp([S] θ / 2)，
//    「同一根轴」肘亮一圈，「只转一半的角度」小臂从 0° 转到 45°、标 45°，「九十除以二」写出 90° ÷ 2 = 45°；
// 句 2（2026-10-29 补）「两个朝向都不是原样」「矩阵对数」右栏下方写「两个都不是原样：先求矩阵对数」，「再只转一半」出 R₀ exp(½ log(R₀ᵀR₁))
//    （《Modern Robotics》3.2.3.3 节、Solà 等式 (25)(26)；005 的例子里一个朝向是原样，才是「绕同一根轴转一半」）；
// 句 3 换成三张卡片：欧拉角、四元数、矩阵指数（亮黄框）；
// 句 4「三个数」「万向锁」「四个数」「没有万向锁」「两套数」依次写进前两张卡片（2026-11-01 去掉「最直观」：《Modern Robotics》第 1 章说最直观的是指数坐标）；
// 句 5「旋转矩阵」「没有万向锁」「直接连乘」「运动旋量、螺旋轴」「九个数，有冗余」依次写进第三张；
// （2026-11-02 原先的句 6、7——UR5 的六根关节轴和指数积公式——挪到新的第 6 幕「矩阵指数用在哪」开头。）
// 时刻取自配音的逐词时间，见 ../ep010.py。
(function () {
  var E = window.EP010, K = E.K, H = E.HEX;
  var D = Math.PI / 180, A = E.ARM, RX = A.RX;
  var T = {
    ep5: 0.66, two: 1.79, mid: 2.48, avg: 3.38, ex: 5.01, mexp: 8.2, axis: 10.19, half: [11.35, 12.24], halfCalc: 12.9, log: 18.17, logEq: 22.45, out1: 24.3,
    cards: 24.55,
    eu: [28.88, 29.91], qu: [31.25, 31.89, 33.04], mx: [36.32, 37.85, 38.95, 40.35, 43.03]
  };
  var END = 46.03;

  E.scenes[4] = {
    title: '比一比：矩阵指数好在哪',
    dur: END,
    cues: [
      { at: 0, s: '两个**朝向**取中间值：比如小臂原样（0°）和转了 90°，绕同一根轴只转一半的**角度**：90° ÷ 2 = 45°' },
      { at: 15.65, s: '两个朝向都不是原样：先求**矩阵对数**，再转一半' },
      { at: 24.46, s: '和第 6 集的另外 2 种写法比一比' },
      { at: 28.06, s: '欧拉角 3 个数，可会万向锁；四元数 4 个数，可一个朝向 2 套数' },
      { at: 35.43, s: '**矩阵指数**给出的旋转矩阵：没有万向锁、能直接相乘、和**螺旋轴**一套说法；代价是 9 个数' }
    ],
    build: function (K) {
      var s = K.sceneSvg('两个朝向取中间值不能直接平均，用矩阵指数绕同一根轴只转一半的角度；两个朝向都不是原样时先用矩阵对数求出两者之间绕什么轴、转了多少，再转一半；和第六集另外两种写法比：欧拉角三个数但会万向锁，四元数四个数没有万向锁但一个朝向两套数，矩阵指数给出的旋转矩阵没有万向锁、能直接相乘、和运动旋量螺旋轴是一套说法，代价是九个数有冗余');

      /* ══ ① 网格 + UR5：取中间值 ══ */
      var g1 = E.el('g', {}, s), under = E.el('g', {}, g1);
      var arm = E.armAt(g1, A.OX, A.OY, A.S, 330);
      var u = arm.aim(1, 2, 0);
      var pl = E.plane(under, { ox: A.OX, oy: A.OY, u: u, x0: -1, x1: 4, y0: -1, y1: 5 });
      var P = pl.P, V1 = arm.view, RIGID = ['forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link'], Q = P(1, 2);
      var top1 = E.el('g', {}, g1);
      var ghosts = [0, 90].map(function (a) {
        var g = E.el('g', {}, top1), e = P(1 + 2.06 * Math.cos(a * D), 2 + 2.06 * Math.sin(a * D));
        K.paint(E.el('path', { d: E.poly([Q, e]), fill: 'none', 'stroke-width': 4, 'stroke-dasharray': '8 6', 'stroke-linecap': 'round' }, g), null, E.BODY);
        K.paint(E.el('circle', { cx: e[0], cy: e[1], r: 6, 'stroke-width': 2.5 }, g), E.S1, E.BODY);
        E.halo(E.txt(g, e[0] + (a ? -14 : 0.75 * u), e[1] + 8, a + '°', 22, E.OMEGA, a ? 'end' : 'start', 900));   // 0° 写在手腕右边：手腕往右还伸出一截
        return g;
      });
      var ring = K.paint(E.el('circle', { cx: Q[0], cy: Q[1], r: 12, 'stroke-width': 3.5 }, top1), E.S1, E.HI);
      K.paint(E.el('circle', { cx: Q[0], cy: Q[1], r: 4 }, top1), E.HI);
      var halfArc = E.arcArrow(top1, Q[0], Q[1], 1.2 * u, E.OMEGA, 3.5, { head: 13 });
      var halfLab = E.halo(E.txt(top1, 0, 0, '45°', 24, E.OMEGA, 'start', 900));
      var RK = E.rowKit(), X = RX + 10;
      var r1 = E.el('g', {}, g1);
      var a1 = RK.line(r1, X, 64, [['第 5 集：两个朝向取中间值', E.INK, 900]], 22);
      var a2 = RK.line(r1, X, 104, [['直接平均', E.ERR, 900], ['：跳出旋转矩阵', E.INK2]], 22);
      var a2x = E.cross(a2.g, X + 260, 96, 9, E.ERR, 4);
      var a3 = RK.line(r1, X, 166, [['矩阵指数：', E.HI, 900], ['绕同一根轴', E.WARN, 900]], 22);
      var a4 = RK.line(r1, X, 206, [['　　　　　只转一半的角度', E.OMEGA, 900]], 22);
      var a5 = E.el('g', {}, r1);
      E.tex(a5, X, 268, '\\textcolor{' + H.hi + '}{\\exp\\!\\big([\\mathcal S]\\,\\tfrac{\\theta}{2}\\big)}', 30, E.INK, 'start', { w: 190, h: 60 });
      var a5c = E.txt(a5, X + 196, 268, '90° ÷ 2 = 45°', 21, E.OMEGA, 'start', 900);
      var a5ok = E.check(a5, X + 200, 296, 22, E.GOOD, 4);
      var a6 = RK.line(r1, X, 346, [['两个都不是原样：', E.INK, 900], ['先求矩阵对数', E.HI, 900]], 22);
      var a7 = E.el('g', {}, r1);
      E.tex(a7, X, 398, 'R_0\\exp\\!\\big(\\tfrac12\\log(R_0^{\\top}R_1)\\big)', 26, E.INK, 'start', { w: 300, h: 56 });

      /* ══ ② 三张卡片 ══ */
      var g2 = E.el('g', {}, s);
      E.txt(g2, 400, 40, '和第 6 集的另外 2 种写法比一比', 24, E.INK, 'middle', 900);
      var CW = 244, CX = [22, 278, 534], CY = 62, CH = 340;
      function card(i, name, color, hi) {
        var g = E.el('g', {}, g2);
        K.paint(E.el('rect', { x: CX[i], y: CY, width: CW, height: CH, rx: 16, 'stroke-width': hi ? 3.5 : 2.5 }, g), E.tint(color, hi ? 12 : 6, 'var(--demo-surface)'), color);
        E.txt(g, CX[i] + CW / 2, CY + 42, name, 26, color, 'middle', 900);
        return g;
      }
      var cE = card(0, '欧拉角', E.INK2), cQ = card(1, '四元数', E.INK2), cM = card(2, '矩阵指数', E.HI, true);
      var sub = E.txt(cM, CX[2] + CW / 2, CY + 74, '给出的旋转矩阵', 20, E.INK2, 'middle', 700);
      function item(gc, i, k, str, good, size) {   // 卡片里的一行：✓ / ✗ + 字
        var g = E.el('g', {}, gc), y = CY + 118 + 46 * k;
        if (good === true) E.check(g, CX[i] + 18, y - 16, 18, E.GOOD, 3.5);
        else if (good === false) E.cross(g, CX[i] + 27, y - 7, 7, E.ERR, 3.5);
        E.txt(g, CX[i] + (good === null ? CW / 2 : 46), y, str, size || 21, good === null ? E.HI : good ? E.INK : E.ERR, good === null ? 'middle' : 'start', 900);
        return g;
      }
      // 不写「最直观」：《Modern Robotics》第 1 章说最自然、最直观的是指数坐标（绕哪根轴转多少），欧拉角没有这个说法
      var iE = [item(cE, 0, 0, '3 个数', null, 24), item(cE, 0, 1, '会万向锁', false)];
      var iQ = [item(cQ, 1, 0, '4 个数', null, 24), item(cQ, 1, 1, '没有万向锁', true), item(cQ, 1, 2, '一个朝向两套数', false)];
      var iM = [item(cM, 2, 0.35, '没有万向锁', true), item(cM, 2, 1.25, '能直接相乘', true), item(cM, 2, 2.15, '和运动旋量、', true), item(cM, 2, 4.4, '9 个数，有冗余', false)];
      var iM2 = E.txt(cM, CX[2] + 46, CY + 118 + 46 * 2.15 + 30, '螺旋轴一套说法', 21, E.INK, 'start', 900);

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) {
            RK.layout(); laid = true;
          }
          var f = E.fade, b = E.bump;

          /* ── ① ── */
          E.op(g1, 1 - f(st, T.out1, 0.4));
          var phi = 45 * D * f(st, T.half[0], T.half[1] - T.half[0]);
          arm.bend(phi); V1.clearTint(); V1.tint(RIGID, E.BODY, 0.45);
          ghosts.forEach(function (g) { E.op(g, f(st, T.two, 0.4)); });
          // 「比如小臂原样和转了九十度」两个朝向（0°、90°）鼓一下：90 从哪来（用户反馈「跨幕的信息要交代一下」）
          ghosts.forEach(function (g, k) { var bb = b(st, T.ex + 0.6 * k, 0.8); E.glow(g, E.HI, bb > 0.05); });
          ring.setAttribute('r', (12 + 5 * b(st, T.axis, 0.9)).toFixed(2)); E.glow(ring, E.HI, b(st, T.axis, 0.9) > 0.05);
          // 画满 45°：标着「45°」，弧就从 0° 画到 45°（同 009 第 3 幕的修正）；开始转才出现
          halfArc.set(0, Math.max(1e-3, phi)); E.op(halfArc.g, f(st, T.half[0], 0.2));
          var hl = P(1 + 1.45 * Math.cos(22 * D), 2 + 1.45 * Math.sin(22 * D)); halfLab.setAttribute('x', hl[0].toFixed(1)); halfLab.setAttribute('y', (hl[1] + 8).toFixed(1));
          E.op(halfLab, f(st, T.half[1], 0.3));
          E.op(a1.g, fr(st, T.ep5)); E.op(a2.g, fr(st, T.avg)); E.drawOn(a2x, f(st, T.avg + 0.4, 0.3));
          E.op(a3.g, fr(st, T.mexp)); E.op(a3.parts[1], fr(st, T.axis)); E.op(a4.g, fr(st, T.half[0]));
          E.op(a5, fr(st, T.half[0] + 0.2)); E.op(a5c, fr(st, T.halfCalc)); E.drawOn(a5ok, f(st, T.halfCalc + 1.0, 0.3));
          E.op(a6.g, fr(st, T.log)); E.op(a7, fr(st, T.logEq));

          /* ── ② ── */
          E.op(g2, f(st, T.cards, 0.4));
          E.op(cE, 1); E.op(cQ, 1); E.op(cM, 1);
          iE.forEach(function (g, k) { E.op(g, fr(st, T.eu[k])); });
          iQ.forEach(function (g, k) { E.op(g, fr(st, T.qu[k])); });
          E.op(sub, fr(st, T.mx[0]));
          iM.forEach(function (g, k) { E.op(g, fr(st, [T.mx[1], T.mx[2], T.mx[3], T.mx[4]][k])); });
          E.op(iM2, fr(st, T.mx[3]));

        }
      };
    }
  };
})();
