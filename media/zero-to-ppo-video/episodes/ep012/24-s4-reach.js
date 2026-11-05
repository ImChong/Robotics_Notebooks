// 012 第 4 幕：够不到——一组解也没有。左边网格往右多放几格、UR5 缩小一点（目标要放到 (5, 0)、还要画出半径 4.24 的圆环），右栏一行行算。
// 2026-11-05 参照 011 重做：手照第 11 集放在小臂上离肘两格（小臂 2 格），数跟着改；「平方不会是负的」当场讲（正乘正、负乘负都是正），圆环说成「大、小两个圆中间夹的一圈」。
// 句 1「太远」「手要到五、零」紫色目标挪到 (5, 0)；
// 句 2「照同样的算法」右栏写 ① ②，「两个式子相减，右边五减四，是一」写；
// 句 3「x 减五的平方，照刚才的拆法」写 x² − 10x + 25，「再整个减掉，左边只剩」「十倍的 x，减二十五，等于一」写；
// 句 4「两边都加二十五」「十倍的 x 是二十六」「两边都除以十」「x 是二点六」写，网格上画 x = 2.6 的竖虚线；
// 句 5「二点六的平方，是六点七六」「y 的平方，等于五减六点七六，是负一点七六」写（橙）；
// 句 6「正数乘以正数是正的」「负数乘以负数也是正的」「平方不会是负的」「一组解也没有」写，「两个圆根本碰不到」绿色的圆（离手 2 格）画出来，两圆之间的空隙描橙；
// 句 7「二点二四加二，是四点二四格」右栏写，「胳膊全伸直」UR5 伸直，「手最远也只到四点二四格」量出 4.24 格，「到不了五格」空隙打 ×；
// 句 8「目标正好在四点二四格远」目标挪到 (4.24, 0)、绿圆跟着挪，「两个圆只碰一下」切点鼓一下，「y 的平方是零」「一组解」「胳膊伸直」右栏写；
// 句 9「离肩太近也到不了」目标挪到 (0.1, 0)，绿圆缩进琥珀圆里，「二点二四减二，是零点二四格」右栏写，「小臂折回来」UR5 把小臂折回来，
//    「手离肩最近也有这么远」肩周围半径 0.24 的小圆描橙；
// 句 10「大、小两个圆」半径 4.24、0.24 两个圆描绿，「中间夹的一圈」涂绿，「叫圆环」写，「在环里面，两组解」(1, 0) 出点、写 2 组，
//    「正好在边上，一组」(4.24, 0) 写 1 组，「在环外面，一组也没有」(5, 0) 写 0 组。
// 时刻取自配音的逐词时间，见 ../ep012.py。
(function () {
  var E = window.EP012, K = E.K, H = E.HEX;
  var D = Math.PI / 180, RX = E.ARM.RX, A = { OX: 196, OY: 210, S: 1.2 };
  var T = {
    far: 1.5, to5: 3.38,
    algo: 5.2, eqs: 6.48, eq2: 8.9, sub: 12.4, rhs: 13.71, rhsR: 14.99,
    exp: 16.62, rule: 17.83, exp2: 19.19, exp3: 20.15, exp4: 21.36, minus: 22.56, only: 23.77, res: 24.82, res2: 27.05,
    add: 28.79, tenx: 30.11, div: 32.09, x: 33.27, xline: [33.7, 34.4],
    back: 35.32, sq: 36.67, sqv: 37.81, y2: 39.05, neg: 41.43,
    pp: 44.29, nn: 46.25, never: 48.41, none: 50.14, draw: 51.39, gap: 52.59,
    sum: 54.69, sumr: 56.94, str: [58.29, 59.0], max: 59.46, maxr: 60.56, cant: 61.55,
    move43: [63.64, 64.6], touch: 66.21, y0: 67.08, one: 68.33, straight: 69.65,
    near: 71.51, moveNear: [71.51, 72.4], diff: 73.02, diffr: 74.18, fold: [75.47, 76.4], minR: 76.45,
    ring: 79.62, twoC: 81.15, between: 82.37, name: 83.67, inside: 84.59, edge: 86.39, outside: 88.2
  };
  var END = 91.12, RMAX = E.L1 + E.L2, RMIN = E.L1 - E.L2;
  var UP15 = E.ik2(1, 0, true);   // 句 10 回到第 1–3 幕的肘上（目标 (1, 0)）
  function c(hex, s) { return '\\textcolor{' + hex + '}{' + s + '}'; }

  E.scenes[3] = {
    title: '够不到：一组解也没有',
    dur: END,
    cues: [
      { at: 0, s: '目标太远：手要到 (5, 0)' },
      { at: 5.11, s: '同样的算法：$x^2+y^2=5$，$(x-5)^2+y^2=4$，相减后右边 $5-4=1$' },
      { at: 16.53, s: '$(x-5)^2=x^2-10x+25$，左边只剩 $10x-25=1$' },
      { at: 28.7, s: '两边加 $25$：$10x=26$；两边除以 $10$：$x=2.6$' },
      { at: 35.23, s: '$2.6^2=6.76$，$y^2=5-6.76=-1.76$' },
      { at: 43.7, s: '正乘正、负乘负都是正的：平方不会是负的，**无解**——两个圆碰不到' },
      { at: 54.6, s: '大臂 + 小臂 = 2.24 + 2 = 4.24 格：全伸直也到不了 5 格' },
      { at: 63.55, s: '目标正好 4.24 格：两圆只碰一下，$y^2=0$，只有一组解（伸直）' },
      { at: 71.42, s: '太近也到不了：2.24 − 2 = 0.24 格' },
      { at: 79.05, s: '手能到的地方：大、小两个圆中间夹的一圈，叫**圆环**——环里 2 组，边上 1 组，环外 0 组' }
    ],
    build: function (K) {
      var s = K.sceneSvg('目标太远：手要到 (5, 0)；同样两式相减，右边 5 − 4 = 1；(x − 5)² 照刚才的拆法是 x² − 10x + 25，左边只剩 10x − 25 = 1，两边加 25 得 10x = 26，两边除以 10 得 x = 2.6；2.6² = 6.76，y² = 5 − 6.76 = −1.76，正数乘正数、负数乘负数都是正的，平方不会是负的，一组解也没有，两个圆碰不到；大臂加小臂 2.24 + 2 = 4.24 格，胳膊全伸直手最远 4.24 格，到不了 5 格；目标正好 4.24 格远，两圆只碰一下，y² = 0，只有一组解：胳膊伸直；太近也到不了：2.24 − 2 = 0.24 格；手能到的地方是大、小两个圆中间夹的一圈，叫圆环：环里两组解，边上一组，环外一组也没有');

      var under = E.el('g', {}, s);
      var armG = E.el('g', {}, s);
      var arm = E.arm2(armG, A.OX, A.OY, A.S, 0, [A.OX - 190, A.OY - 210, 440, 420]), u = arm.u, V = arm.view;
      // 网格放到 y ±5，圆环（半径 4.24）不碰 y 轴的标签；x 轴标签挪到轴下面：离手 2 格的圆会画到 x = 7，压住画在轴尾的 x
      var pl = E.plane(under, { ox: A.OX, oy: A.OY, u: u, x0: -5, x1: 6, y0: -5, y1: 5, labels: false });
      var P = pl.P, O = P(0, 0);
      var axX = E.txt(under, P(6, 0)[0] + 22, P(0, 0)[1] + 8, 'x', 24, E.INK2, 'start', 700), axY = E.txt(under, P(0, 5)[0] + 12, P(0, 5)[1] - 8, 'y', 24, E.INK2, 'start', 700);
      var top = E.el('g', {}, s);
      // 圆环（句 10）：外圆 4.24、内圆 0.24 先描边，再在中间涂绿
      var ringG = E.el('g', {}, under);
      var ringP = K.paint(E.el('path', { d: 'M ' + (O[0] - RMAX * u) + ' ' + O[1] + ' a ' + RMAX * u + ' ' + RMAX * u + ' 0 1 0 ' + 2 * RMAX * u + ' 0 a ' + RMAX * u + ' ' + RMAX * u + ' 0 1 0 ' + (-2 * RMAX * u) + ' 0 Z' +
        ' M ' + (O[0] - RMIN * u) + ' ' + O[1] + ' a ' + RMIN * u + ' ' + RMIN * u + ' 0 1 1 ' + 2 * RMIN * u + ' 0 a ' + RMIN * u + ' ' + RMIN * u + ' 0 1 1 ' + (-2 * RMIN * u) + ' 0 Z', 'fill-rule': 'evenodd', 'stroke-width': 2.5 }, ringG), E.tint(E.GOOD, 22, 'transparent'), E.GOOD);
      var c1 = E.circ(top, O[0], O[1], E.L1 * u, E.C1, 2.5, '9 6');
      var c2 = E.circ(top, 0, 0, E.L2 * u, E.C2, 2.5, '9 6');
      var tgt = E.target(top, 0, 0, 11, E.TGT);
      var tgLab = E.halo(E.txt(top, 0, 0, '(5, 0)', 20, E.TGT, 'middle', 900));
      // 句 4：x = 2.6
      var X25 = P(2.6, 0);
      var xl = K.paint(E.el('path', { d: 'M ' + X25[0] + ' ' + P(0, 3.2)[1] + ' V ' + P(0, -3.2)[1], fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '7 6' }, top), null, E.X);
      var xlLab = E.halo(E.txt(top, X25[0] + 8, P(0, 3.0)[1], 'x = 2.6', 19, E.X, 'start', 900));
      // 句 5：两圆之间的空隙；句 6：4.3 的量尺、到不了 5 的空隙
      function hseg(x0, x1, y, color, w) { var a = P(x0, y), b2 = P(x1, y); return K.paint(E.el('path', { d: 'M ' + a[0] + ' ' + a[1] + ' H ' + b2[0], fill: 'none', 'stroke-width': w || 6, 'stroke-linecap': 'round' }, top), null, color); }
      var gap1 = hseg(E.L1, 5 - E.L2, 0, E.ERR, 7);
      var dimG = E.el('g', {}, top), dy = 0.9 * u, M43 = P(RMAX, 0);
      K.paint(E.el('path', { d: 'M ' + O[0] + ' ' + (O[1] + dy - 7) + ' V ' + (O[1] + dy + 7) + ' M ' + O[0] + ' ' + (O[1] + dy) + ' H ' + M43[0] + ' M ' + M43[0] + ' ' + (O[1] + dy - 7) + ' V ' + (O[1] + dy + 7),
        fill: 'none', 'stroke-width': 2.5 }, dimG), null, E.INK);
      E.halo(E.txt(dimG, (O[0] + M43[0]) / 2, O[1] + dy + 24, '最远 4.24 格', 19, E.INK, 'middle', 900));
      var gap2 = hseg(RMAX, 5, 0.9 * -1, E.ERR, 5);
      var gap2X = E.cross(top, P(4.65, 0)[0], O[1] + dy, 7, E.ERR, 4);
      // 句 7：切点；句 8：最近 0.18 的小圆
      var tangent = K.paint(E.el('circle', { cx: P(E.L1, 0)[0], cy: O[1], r: 8, 'stroke-width': 3 }, top), E.S1, E.HI);
      var minC = E.circ(top, O[0], O[1], RMIN * u, E.ERR, 3);
      var minLab = E.halo(E.txt(top, O[0] - 14, O[1] - 18, '0.24', 19, E.ERR, 'end', 900));
      // 句 9：三个点
      function mark(x, n, color, below) {
        var g = E.el('g', {}, top), p = P(x, 0);
        K.paint(E.el('circle', { cx: p[0], cy: p[1], r: 8, 'stroke-width': 3 }, g), E.S1, color);
        E.halo(E.txt(g, p[0], below ? p[1] + 34 : p[1] - 16, n, 22, color, 'middle', 900));
        return g;
      }
      var mk2 = mark(1, '2 组', E.HI), mk1 = mark(RMAX, '1 组', E.GOOD), mk0 = mark(5, '0 组', E.ERR, true);   // 和 4.24 处的「1 组」只隔 0.76 格：写在点下面
      var ringLab = E.halo(E.txt(top, P(-2.9, 2.9)[0], P(-2.9, 2.9)[1], '圆环', 24, E.GOOD, 'middle', 900));

      /* ══ 右栏 ══ */
      var Rg = E.el('g', {}, s), X = RX + 10, SZ = 22;
      function tx(g, y, tex, size) { return E.tex(g, X, y, tex, size || SZ, E.INK, 'start', { w: 340, h: 44 }); }
      function lab(g, y, str, color, size) { return E.txt(g, X, y, str, size || 21, color || E.INK2, 'start', 900); }
      var g1 = E.el('g', {}, Rg);
      var a0 = lab(g1, 26, '目标：手到 (5, 0)', E.TGT, 21);
      var a1 = tx(g1, 62, c(H.warn, '\\text{①}') + '\\;x^2+y^2=5', 21);
      var a2 = tx(g1, 98, c(H.good, '\\text{②}') + '\\;(x-5)^2+y^2=4', 21);
      var a2b = tx(g1, 134, '\\text{①}-\\text{②：右边 }5-4=1', 21);
      var a3a = tx(g1, 172, '(x-5)^2=x^2', 21), a3b = tx(g1, 172, '(x-5)^2=x^2-10x', 21), a3 = tx(g1, 172, '(x-5)^2=x^2-10x+25', 21);
      var a4a = tx(g1, 210, '\\text{左边只剩：}10x-25', 21), a4 = tx(g1, 210, '\\text{左边只剩：}10x-25=1', 21);
      var a5a = tx(g1, 248, '+25\\text{：}\\;10x=26', 21), a5 = tx(g1, 248, '+25\\text{：}\\;10x=26\\qquad\\div 10\\text{：}\\;' + c(H.x, 'x=2.6'), 21);
      var a6a = tx(g1, 288, '2.6^2', 21), a6 = tx(g1, 288, '2.6^2=6.76', 21);
      var a7a = tx(g1, 326, 'y^2=5-6.76', 21), a7 = tx(g1, 326, 'y^2=5-6.76=' + c(H.err, '-1.76'), 21);
      var a8 = lab(g1, 370, '正 × 正、负 × 负，都是正的', E.INK, 19), a8b = lab(g1, 406, '平方不会是负的 → 无解', E.ERR, 21);
      var g2 = E.el('g', {}, Rg);
      var b1 = tx(g2, 36, '\\text{大臂}+\\text{小臂}=2.24+2=4.24', 21);
      var b2 = lab(g2, 76, '全伸直：手最远 4.24 格，到不了 5', E.INK2, 20);
      var b3 = lab(g2, 128, '目标正好 4.24 格：两圆只碰一下', E.INK, 20);
      var b4 = tx(g2, 166, 'y^2=0\\;\\Rightarrow\\;' + c(H.good, '\\text{一组解（伸直）}'), 21);
      var b5 = lab(g2, 218, '太近也到不了：', E.INK, 20);
      var b6 = tx(g2, 256, '2.24-2=' + c(H.err, '0.24') + '\\text{ 格}', 21);
      var b7 = lab(g2, 296, '小臂折回来，手离肩最近 0.24 格', E.INK2, 20);
      var b8 = lab(g2, 344, '大、小两个圆中间夹的一圈：圆环', E.GOOD, 20);
      var chips = [['环里 2 组', E.HI, T.inside], ['边上 1 组', E.GOOD, T.edge], ['环外 0 组', E.ERR, T.outside]].map(function (cc, k) {
        var tg = E.tag(Rg, X + k * 112, 372, 104, 38, cc[0], cc[1], { size: 19, weight: 900, ink: cc[1] });
        return { g: tg.g, at: cc[2], x: X + 52 + k * 112 };
      });

      function fr(st, a) { return E.fade(st, a, 0.3); }
      function only(list, st, times) { var k = -1; times.forEach(function (tt, i) { if (st >= tt) k = i; }); list.forEach(function (n, i) { E.op(n, i === k ? 1 : 0); }); }
      function lerp(a, b2, k) { return a + (b2 - a) * k; }

      return {
        el: s,
        draw: function (st, t) {
          var f = E.fade, b = E.bump;
          /* ── 目标的位置：5 → 4.3 → 0.1；句 9 换成三个点 ── */
          var tx5 = lerp(1, 5, f(st, T.to5 - 0.4, 0.6));
          var tx_ = lerp(tx5, RMAX, f(st, T.move43[0], T.move43[1] - T.move43[0]));
          tx_ = lerp(tx_, 0.1, f(st, T.moveNear[0], T.moveNear[1] - T.moveNear[0]));
          var tp = P(tx_, 0), ringOn = f(st, T.twoC - 0.6, 0.4);
          tgt.setAttribute('transform', 'translate(' + tp[0].toFixed(1) + ' ' + tp[1].toFixed(1) + ')');
          E.op(tgt, 1 - ringOn);
          tgLab.setAttribute('x', tp[0].toFixed(1)); tgLab.setAttribute('y', (tp[1] - 20).toFixed(1));
          tgLab.textContent = st < T.move43[0] ? '(5, 0)' : st < T.near ? '(4.24, 0)' : '(0.1, 0)';
          E.op(tgLab, f(st, T.to5, 0.3) * (1 - ringOn) * (st >= T.near ? 0 : 1));
          c2.setAttribute('cx', tp[0].toFixed(1)); c2.setAttribute('cy', tp[1].toFixed(1));
          // 离手 2 格的圆从 x = 7 挪到 6.24 的时候会扫过 x 轴的标签：扫过的那一下先藏起来
          var dAx = Math.abs(Math.hypot((P(6, 0)[0] + 28 - tp[0]) / u, 0) - E.L2);
          E.op(axX, E.clamp((dAx - 0.45) / 0.25, 0, 1));
          E.op(c2, f(st, T.draw, 0.6) * (1 - ringOn));   // <circle> 没有 d，drawOn 的遮罩画不出来：淡入
          E.op(c1, 1 - 0.7 * ringOn);
          /* ── UR5：伸直 → 折回来 → 回到 (1, 0) 的肘上 ── */
          var sp = f(st, T.str[0], T.str[1] - T.str[0]), fp = f(st, T.fold[0], T.fold[1] - T.fold[0]), rp = f(st, T.ring, 0.8);
          var q1 = lerp(lerp(lerp(UP15[0], 0, sp), Math.PI / 2, fp), UP15[0], rp), q2 = lerp(lerp(lerp(UP15[1], 0, sp), -Math.PI / 2, fp), UP15[1], rp);
          arm.pose(q1, q2);
          V.clearTint(); V.tint('all', null, 1, 0.35 + 0.65 * f(st, T.str[0] - 0.3, 0.3) * (1 - 0.6 * rp));
          /* ── 左：算式落在网格上 ── */
          E.drawOn(xl, f(st, T.xline[0], T.xline[1] - T.xline[0])); E.op(xl, (f(st, T.xline[0], 0.1) > 0 ? 0.9 : 0) * (1 - f(st, T.sum, 0.4)));
          E.op(xlLab, fr(st, T.xline[1]) * (1 - f(st, T.sum, 0.4)));
          E.op(gap1, fr(st, T.gap) * (1 - f(st, T.sum, 0.4)));
          E.op(dimG, fr(st, T.max) * (1 - f(st, T.move43[0], 0.3)));
          var g2p = f(st, T.cant, 0.3) * (1 - f(st, T.move43[0], 0.3));
          gap2.setAttribute('d', 'M ' + M43[0].toFixed(1) + ' ' + (O[1] + dy).toFixed(1) + ' H ' + P(5, 0)[0].toFixed(1));
          E.op(gap2, g2p); E.drawOn(gap2X, f(st, T.cant + 0.2, 0.3)); E.op(gap2X, g2p > 0 ? g2p : 0);
          tangent.setAttribute('r', (8 + 6 * b(st, T.touch, 0.9)).toFixed(1));
          E.op(tangent, fr(st, T.touch) * (1 - f(st, T.near, 0.3)));
          E.drawOn(minC, f(st, T.minR, 0.5)); E.op(minC, (f(st, T.minR, 0.1) > 0 ? 1 : 0)); E.op(minLab, fr(st, T.minR + 0.4) * (1 - ringOn));
          E.op(ringG, f(st, T.twoC, 0.5)); ringP.style.fillOpacity = f(st, T.between, 0.6); E.op(ringLab, fr(st, T.name));
          E.pop(mk2, P(1, 0)[0], O[1], f(st, T.inside, 0.35), 0.5); E.pop(mk1, M43[0], O[1], f(st, T.edge, 0.35), 0.5); E.pop(mk0, P(5, 0)[0], O[1], f(st, T.outside, 0.35), 0.5);

          /* ── 右栏 ── */
          E.op(g1, 1 - f(st, T.sum - 0.3, 0.4));
          E.op(a0, fr(st, T.to5)); E.op(a1, fr(st, T.eqs)); E.op(a2, fr(st, T.eq2));
          [a1, a2].forEach(function (n) { E.glow(n, E.HI, b(st, T.sub, 0.9) > 0.05); });
          E.op(a2b, fr(st, T.rhs));
          only([a3a, a3b, a3], st, [T.exp2, T.exp3, T.exp4]);
          only([a4a, a4], st, [T.only, T.res2]);
          only([a5a, a5], st, [T.tenx, T.x]);
          only([a6a, a6], st, [T.sq, T.sqv]);
          only([a7a, a7], st, [T.y2, T.neg]);
          E.op(a8, fr(st, T.pp)); E.op(a8b, fr(st, T.never));
          E.op(g2, f(st, T.sum - 0.2, 0.3));
          E.op(b1, fr(st, T.sum)); E.op(b2, fr(st, T.max)); E.op(b3, fr(st, T.move43[0])); E.op(b4, fr(st, T.y0));
          E.op(b5, fr(st, T.near)); E.op(b6, fr(st, T.diff)); E.op(b7, fr(st, T.fold[0])); E.op(b8, fr(st, T.twoC));
          chips.forEach(function (cc) { E.pop(cc.g, cc.x, 391, f(st, cc.at, 0.35), 0.7); });
        }
      };
    }
  };
})();
