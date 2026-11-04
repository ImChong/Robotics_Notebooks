// 011 第 5 幕：六个关节也一样。左边斜着看的 UR5（URDF，example-robot-data），零位是胳膊平伸（《Modern Robotics》例 4.5、图 4.6）；
// 右栏三排写出六个矩阵指数 + M 的公式、六个角度，再化简成只剩第 2、5 个。
// 句 1「零位」「胳膊平伸」右栏写。句 2「螺旋轴」六根关节轴（琥珀虚线 + S₁…S₆）依次出来；「零位时、手的位姿」手上出 {b} 三根小箭头、写 M；
//    「教材上」右栏出卡片「例 4.5」。句 3 公式三排：「第一个」「第六个」那一块亮；「从右往左」箭头；「第六个关节先转」「第一个最后转」写在下面。
// 句 4 六个角度逐个写：θ₂ = −90°（「反着转」）、θ₅ = 90°，「别的都是零度」其余写 0。句 5「转零度的矩阵指数」第 1、3、4、6 块打灰、写 = I
//    （「第五集的单位矩阵」），「只剩第二个和第五个」下面写化简后的式子，只留 S₂、S₅ 两根轴。
// 句 6「先转第五个」手腕绕 S₅ 转 90°，「再转第二个」整条胳膊绕 S₂ 转到竖直（同原书图 4.7）。句 7「交给电脑算」「下一幕」右栏写，手旁打问号。
// 时刻取自配音的逐词时间，见 ../ep011.py。
(function () {
  var E = window.EP011, K = E.K, H = E.HEX;
  var D = Math.PI / 180, RX = E.ARM.RX, X = RX + 10;
  var T = {
    six: 0.7, own: 2.5, zero: 3.89, flat: 5.03,
    each: 6.71, axes: 8.1, byZero: 9.25, M: 10.33, hand: 11.2, book: 13.44, listed: 17.67, thisOne: 19.1, outA: 21.52,
    formula: 21.89, f1: 22.83, f6: 23.34, prod: 25.73, mM: 28.14, rl: [30.19, 30.88], firstT: 32.11, lastT: 33.49,
    eg: 34.91, th2: 36.51, rev: 38.08, th5: 39.53, th5v: 40.64, others: 41.41,
    zeroI: 43.59, ep5: 45.57, I: 46.64, keep: 48.89,
    rl2: 51.71, j5: [53.24, 55.14], j2: [56, 57.51], up: 57.27,
    d3: 59.02, same: 60.98, more: 62.12, pc: 62.72, next: 64.27
  };
  var END = 67.36;
  // 标签摆法：[哪一端, dx, dy, 对齐]；每根轴从关节往两边画多长（米）
  // S₄、S₅ 换到另一头：原先挤在手腕那一团，压在别的轴线上（线压字检查）
  var LAB = [[1, 0, -12, 'middle'], [0, -18, 6, 'end'], [0, -18, 6, 'end'], [0, -12, 16, 'end'], [0, -10, 4, 'end'], [1, 14, 6, 'start']];
  var LEN = [[-0.05, 0.32], [-0.2, 0.2], [-0.18, 0.18], [-0.16, 0.16], [-0.15, 0.15], [-0.14, 0.14]];

  E.scenes[4] = {
    title: '六个关节也一样',
    dur: END,
    cues: [
      { at: 0, s: '六个关节也一样：这台机械臂自己的**零位**，是胳膊平伸' },
      { at: 6.61, s: '每个关节一根**螺旋轴**，都按**零位**写；零位时、手的**位姿** M 也写好：机器人学教材《Modern Robotics》例 4.5 都列了出来' },
      { at: 21.79, s: 'T = exp([S₁]θ₁) ⋯ exp([S₆]θ₆) M（⋯：中间几个省略）：从右往左算，第 6 个关节先转，第 1 个最后转' },
      { at: 34.81, s: '比如 θ₂ = −90°（反着转 90°），θ₅ = 90°，别的都是 0°' },
      { at: 43.49, s: '转 0° 的**矩阵指数**是单位矩阵 I（第 5 集），乘谁都不变：只剩第 2、5 个' },
      { at: 51.61, s: '从右往左：先转第 5 个，手腕转 90°；再转第 2 个，整条胳膊竖起来' },
      { at: 58.92, s: '三维的 4 × 4，算法和平面一样，只是数多：交给电脑算，下一幕写成代码' }
    ],
    build: function (K) {
      var s = K.sceneSvg('六个关节也一样：这台机械臂自己的零位是胳膊平伸；每个关节一根螺旋轴，都按零位写好，零位时手的位姿 M 也写好，教材例 4.5 都列了出来。公式 T = exp([S₁]θ₁) exp([S₂]θ₂) ⋯ exp([S₆]θ₆) M，从右往左算，第六个关节先转、第一个最后转。比如 θ₂ = −90°、θ₅ = 90°，别的都是零度；转零度的矩阵指数是单位矩阵，乘谁都不变，只剩第二个和第五个：先转第五个，手腕转九十度，再转第二个，整条胳膊竖起来。三维的四乘四算法和平面一样，只是数多，交给电脑算');

      var g3 = E.el('g', {}, s);
      var R6 = E.ur6(g3, 100, 352, 2.25, 32, 16, [0, 0, 440, 420]), V = R6.view;
      var mLab = E.halo(E.txt(g3, 0, 0, 'M', 24, E.GOOD, 'start', 900)); mLab.style.fontFamily = 'KaTeX_Math, serif'; mLab.style.fontStyle = 'italic';
      var qm = E.halo(E.txt(g3, 0, 0, '?', 34, E.HI, 'middle', 900));
      var j5Arc = E.halo(E.txt(g3, 0, 0, 'θ₅ = 90°', 20, E.OMEGA, 'end', 900)), j2Arc = E.halo(E.txt(g3, 0, 0, 'θ₂ = −90°', 20, E.OMEGA, 'end', 900));

      /* ══ 右栏 ══ */
      var RK = E.rowKit();
      var gA = E.el('g', {}, s);
      var a0 = RK.line(gA, X, 34, [['六个关节也一样', E.HI, 900]], 26);
      var a1 = RK.line(gA, X, 78, [['这台机械臂自己的', E.INK2], ['零位', E.GOOD, 900], ['：', E.INK2], ['胳膊平伸', E.INK, 900]], 21);
      var a2 = RK.line(gA, X, 120, [['每个关节一根', E.INK2], ['螺旋轴', E.WARN, 900], ['，按零位写', E.INK2]], 21);
      var a3 = RK.line(gA, X, 156, [['零位时、手的位姿 ', E.INK2], ['M', E.GOOD, 900]], 21);
      var card = E.el('g', {}, gA);
      K.paint(E.el('rect', { x: RX, y: 182, width: 344, height: 74, rx: 12, 'stroke-width': 2.5 }, card), E.tint(E.GOOD, 10, 'var(--demo-surface)'), E.GOOD);
      E.txt(card, RX + 16, 212, '教材：《Modern Robotics》例 4.5', 20, E.GOOD, 'start', 900);
      E.txt(card, RX + 16, 242, '六根螺旋轴、M：都列了出来', 18, E.INK, 'start', 700);
      // 2026-11-04（观众零基础）：前几集没提过「教材」，这里点明是哪一本，后面说的教材都是它
      var thisOne = RK.line(gA, X, 290, [['后面说的「教材」，都是这一本', E.INK2]], 19);
      // 公式三排
      var gB = E.el('g', {}, s), FY = [40, 104, 168];
      E.tex(gB, X, FY[0], 'T=', 24, E.INK, 'start', { w: 40, h: 40 });
      function factor(x, y, w, tex, col) {
        var g = E.el('g', {}, gB);
        var box = K.paint(E.el('rect', { x: x - 6, y: y - 26, width: w, height: 36, rx: 8, 'stroke-width': 2 }, g), E.tint(col, 14, 'var(--demo-surface)'), col);
        var tx = E.tex(g, x, y, tex, 18, E.INK, 'start', { w: w - 8, h: 40 });
        var eqI = E.txt(g, x + w / 2 - 6, y + 1, 'I', 24, E.INK, 'middle', 700); eqI.style.fontFamily = 'KaTeX_Main, serif';
        return { g: g, box: box, tx: tx, eqI: eqI, x: x, y: y, w: w };
      }
      var FX = [[X + 48, FY[0]], [X + 160, FY[0]], [X, FY[1]], [X + 112, FY[1]], [X + 224, FY[1]], [X, FY[2]]];
      var facs = FX.map(function (q, i) { return factor(q[0], q[1], 106, '\\exp([\\mathcal S_' + (i + 1) + ']\\,\\theta_' + (i + 1) + ')', E.WARN); });
      var fM = factor(X + 112, FY[2], 34, 'M', E.GOOD);
      fM.eqI.textContent = '';
      var rlV = E.vec(gB, E.INK2, 3, { head: 11 });
      var lastT = E.txt(gB, X + 95, FY[0] + 30, '最后转', 16, E.WARN, 'middle', 900);
      var firstT = E.txt(gB, X + 47, FY[2] + 30, '先转', 16, E.GOOD, 'middle', 900);
      // 六个角度
      var thG = E.el('g', {}, s), TY = 234;
      var thR = RK.line(thG, X, TY, [['θ = (', E.INK2], ['0', E.MUT, 900], [', ', E.INK2], ['−90°', E.OMEGA, 900], [', ', E.INK2], ['0', E.MUT, 900], [', ', E.INK2], ['0', E.MUT, 900],
        [', ', E.INK2], ['90°', E.OMEGA, 900], [', ', E.INK2], ['0', E.MUT, 900], [')', E.INK2]], 22);
      var revR = RK.line(thG, X, TY + 30, [['−90°：反着转 90°', E.INK2]], 17);
      // 化简
      var gC = E.el('g', {}, s);
      E.tex(gC, X, 306, 'T=\\exp([\\mathcal S_2]\\,(-90^\\circ))\\,\\exp([\\mathcal S_5]\\,90^\\circ)\\,M', 20, E.INK, 'start', { w: 344, h: 46 });
      var c1 = RK.line(gC, X, 348, [['先转第 5 个：', E.INK2], ['手腕转 90°', E.OMEGA, 900]], 19);
      var c2 = RK.line(gC, X, 378, [['再转第 2 个：', E.INK2], ['整条胳膊竖起来', E.OMEGA, 900]], 19);
      var gD = E.el('g', {}, s);
      var d1 = RK.line(gD, X, 318, [['三维的 4 × 4：', E.INK2], ['算法和平面一样', E.INK, 900]], 19);
      var d2 = RK.line(gD, X, 350, [['只是数多 → ', E.INK2], ['交给电脑算', E.HI, 900]], 19);
      var d3 = RK.line(gD, X, 384, [['下一幕：写成代码', E.GOOD, 900]], 19);

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) { RK.layout(); laid = true; }
          var f = E.fade, b = E.bump;
          /* ── UR5：句 6 先转第 5 个、再转第 2 个 ── */
          var ang5 = 90 * D * f(st, T.j5[0], T.j5[1] - T.j5[0]), ang2 = -90 * D * f(st, T.j2[0], T.j2[1] - T.j2[0]);
          V.joints({ shoulder_pan_joint: 0, shoulder_lift_joint: ang2, elbow_joint: 0, wrist_1_joint: 0, wrist_2_joint: ang5, wrist_3_joint: 0 });
          R6.update(LAB, LEN);
          R6.lines.forEach(function (ln, i) {
            var on = f(st, T.axes + 0.25 * i, 0.35), keep = i === 1 || i === 4;
            var dim = keep ? 1 : 1 - 0.85 * f(st, T.zeroI + 0.2 * i, 0.4);
            E.op(ln, on * dim); E.op(R6.labs[i], on * dim);
            var pulse = Math.max(b(st, T.axes + 0.25 * i, 0.6), i === 0 ? b(st, T.f1, 0.7) : 0, i === 5 ? b(st, T.f6, 0.7) : 0,
              i === 4 ? Math.max(b(st, T.th5, 0.8), f(st, T.j5[0], 0.3) * (1 - f(st, T.j5[1], 0.4))) : 0,
              i === 1 ? Math.max(b(st, T.th2, 0.8), f(st, T.j2[0], 0.3) * (1 - f(st, T.j2[1], 0.4))) : 0);
            ln.style.strokeWidth = 3.5 + 3 * pulse; E.glow(ln, E.WARN, pulse > 0.05);
          });
          var h = R6.handAt(), ho = V.project(h.o);
          E.op(R6.handG, f(st, T.M, 0.4)); E.glow(R6.handG, E.HI, b(st, T.hand, 0.9) > 0.05);
          mLab.setAttribute('x', (ho[0] - 18).toFixed(1)); mLab.setAttribute('y', (ho[1] + 22).toFixed(1)); mLab.style.textAnchor = 'end'; E.op(mLab, f(st, T.M, 0.3) * (1 - f(st, T.j5[0], 0.3)));
          qm.setAttribute('x', (ho[0] + 40).toFixed(1)); qm.setAttribute('y', (ho[1] + 22).toFixed(1)); E.op(qm, f(st, T.pc, 0.3));
          var p5 = V.at('wrist_2_link'), p2 = V.at('upper_arm_link');
          j5Arc.setAttribute('x', (p5[0] - 10).toFixed(1)); j5Arc.setAttribute('y', (p5[1] - 78).toFixed(1));   // 写在 S₅ 标签上面一行（S₅ 标签在这根轴的上端）
          j2Arc.setAttribute('x', (p2[0] - 30).toFixed(1)); j2Arc.setAttribute('y', (p2[1] - 34).toFixed(1));
          E.op(j5Arc, f(st, T.j5[0], 0.3) * (1 - f(st, T.j2[0], 0.3)));   // 第 2 个关节开始转就退场：手腕跟着升上去，标签会出画 E.op(j2Arc, f(st, T.j2[0], 0.3) * (1 - f(st, T.d3, 0.4)));

          /* ── 右栏 ── */
          E.op(gA, 1 - f(st, T.outA, 0.4));
          E.op(a0.g, fr(st, T.six)); E.op(a1.g, fr(st, T.own)); E.op(a1.parts[1], fr(st, T.zero)); E.op(a1.parts[3], fr(st, T.flat));
          E.op(a2.g, fr(st, T.each)); E.op(a2.parts[1], fr(st, T.axes)); E.op(a2.parts[2], fr(st, T.byZero)); E.op(a3.g, fr(st, T.M));
          E.pop(card, RX + 172, 219, f(st, T.book, 0.35), 0.92); E.op(thisOne.g, f(st, T.thisOne, 0.3));
          E.op(gB, fr(st, T.formula));
          facs.forEach(function (F, i) {
            E.op(F.g, fr(st, T.formula + 0.12 * i));
            var hl = i === 0 ? b(st, T.f1, 0.8) : i === 5 ? b(st, T.f6, 0.8) : 0;
            var keep = i === 1 || i === 4, gray = keep ? 0 : f(st, T.zeroI + 0.2 * i, 0.4);
            F.box.style.strokeWidth = 2 + 2 * hl; E.glow(F.box, E.HI, hl > 0.05);
            E.op(F.box, 1 - 0.6 * gray); E.op(F.tx, 1 - gray); E.op(F.eqI, keep ? 0 : gray);
            F.g.style.filter = gray > 0.5 ? 'grayscale(1)' : '';
            if (keep) E.glow(F.box, E.HI, Math.max(hl, b(st, T.keep, 0.9)) > 0.05);
          });
          E.op(fM.g, fr(st, T.mM));
          var rp = f(st, T.rl[0], T.rl[1] - T.rl[0]);
          rlV.set(X + 300, FY[2] - 8, X + 300 - 120 * rp, FY[2] - 8); E.op(rlV.g, (rp > 0 ? 1 : 0) * (1 - f(st, T.zeroI, 0.4)));
          E.op(firstT, fr(st, T.firstT) * (1 - f(st, T.zeroI, 0.4))); E.op(lastT, fr(st, T.lastT) * (1 - f(st, T.zeroI, 0.4)));
          E.op(thG, fr(st, T.eg));
          var parts = thR.parts;
          E.op(parts[3], fr(st, T.th2)); E.op(parts[9], fr(st, T.th5v));
          [1, 5, 7, 11].forEach(function (k) { E.op(parts[k], fr(st, T.others)); });
          E.op(revR.g, fr(st, T.rev) * (1 - f(st, T.zeroI, 0.4)));
          E.op(gC, fr(st, T.keep) * (1 - f(st, T.d3, 0.4)));
          E.op(c1.g, fr(st, T.j5[0])); E.op(c2.g, fr(st, T.j2[0]));
          E.op(gD, fr(st, T.d3)); E.op(d1.g, fr(st, T.d3)); E.op(d1.parts[1], fr(st, T.same)); E.op(d2.g, fr(st, T.more)); E.op(d2.parts[1], fr(st, T.pc)); E.op(d3.g, fr(st, T.next));
        }
      };
    }
  };
})();
