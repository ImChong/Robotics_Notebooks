// 012 第 4 幕：够不到——一组解也没有。左边网格往右多放几格、UR5 缩小一点（目标要放到 (5, 0)、还要画出半径 4.3 的圆环），右栏一行行算。
// 句 1「太远」「手腕要到五、零」紫色目标挪到 (5, 0)；
// 句 2「照同样的算法」右栏写 ① ②，「x 减五的平方，展开」「相减以后」「十倍的 x 减二十五，等于零点七五」逐项写；
// 句 3「十倍的 x 是二十五点七五」「x 是二点五七五」右栏写，网格上画 x = 2.575 的竖虚线；
// 句 4「二点五七五的平方，约等于六点六三」「y 的平方，等于五减六点六三，是负一点六三」右栏写（橙）；
// 句 5「平方都不会是负的」「一组解也没有」右栏写「无解」，「两个圆根本碰不到」绿色的圆（离手腕 2.06）画出来，两圆之间的空隙描橙；
// 句 6「二点二四加二点零六，约四点三格」右栏写，「胳膊全伸直」UR5 伸直，「手腕最远也只到四点三格」量出 4.3 格，「到不了五格」空隙打 ×；
// 句 7「目标正好在四点三格远」目标挪到 (4.3, 0)、绿圆跟着挪，「两个圆只碰一下」切点鼓一下，「y 的平方是零」「一组解」「胳膊伸直」右栏写；
// 句 8「离肩太近」目标挪到 (0.1, 0)，绿圆缩进琥珀圆里，「小臂比大臂短」UR5 把小臂折回来，「二点二四减二点零六，约零点一八格」右栏写，
//    「手腕离肩，最近也有这么远」肩周围半径 0.18 的小圆描橙；
// 句 9「一个圆环」0.18 到 4.3 之间涂绿，「在环里面，两组解」(1.5, 0) 出点、写 2，「正好在边上，一组」(4.3, 0) 写 1，「在环外面，一组也没有」(5, 0) 写 0。
// 时刻取自配音的逐词时间，见 ../ep012.py。
(function () {
  var E = window.EP012, K = E.K, H = E.HEX;
  var D = Math.PI / 180, RX = E.ARM.RX, A = { OX: 196, OY: 210, S: 1.2 };
  var T = {
    far: 1.51, to5: 3.59,
    algo: 5.42, eqs: 6.08, sub: 6.75, exp: 7.92, exp2: 9.65, exp3: 11.61, after: 12.51, res: 13.28, res2: 15.02,
    tenx: 17.28, x: 19.75, xline: [19.9, 20.6],
    back: 22.3, sq: 23.08, sqv: 24.65, y2: 26.11, neg: 28.49,
    never: 30.89, none: 33.56, draw: 35.2, gap: 36.54,
    sum: 38.64, sumr: 41.54, str: [42.74, 43.6], max: 43.85, maxr: 45.06, cant: 46.0,
    move43: [48.1, 49.0], touch: 50.57, y0: 51.44, one: 52.7, straight: 54.04,
    near: 55.9, moveNear: [55.9, 56.8], short: 57.42, fold: [57.42, 58.4], diff: 58.64, diffr: 60.41, minR: 61.67,
    ring: 66.94, inside: 67.97, edge: 69.73, outside: 71.56
  };
  var END = 74.56, RMAX = E.L1 + E.L2, RMIN = E.L1 - E.L2;
  var UP15 = E.ik2(1.5, 0, true);
  function c(hex, s) { return '\\textcolor{' + hex + '}{' + s + '}'; }

  E.scenes[3] = {
    title: '够不到：一组解也没有',
    dur: END,
    cues: [
      { at: 0, s: '目标太远：**手腕**要到 (5, 0)' },
      { at: 5.33, s: '同样的算法：$(x-5)^2=x^2-10x+25$，相减得 $10x-25=0.75$' },
      { at: 17.19, s: '$10x=25.75$，$x=2.575$' },
      { at: 22.21, s: '$2.575^2\\approx 6.63$，$y^2=5-6.63=-1.63$' },
      { at: 30.8, s: '平方不会是负的：**无解**——两个圆碰不到' },
      { at: 38.55, s: '大臂 + 小臂 = 2.24 + 2.06 ≈ 4.3 格：全伸直也到不了 5 格' },
      { at: 48.01, s: '目标正好 4.3 格：两圆只碰一下，$y^2=0$，只有一组解（伸直）' },
      { at: 55.81, s: '太近也不行：2.24 − 2.06 ≈ 0.18 格' },
      { at: 64.79, s: '**手腕**能到的地方是一个**圆环**：环里 2 组，边上 1 组，环外 0 组' }
    ],
    build: function (K) {
      var s = K.sceneSvg('目标太远：手腕要到 (5, 0)；同样两式相减：(x − 5)² 展开是 x² − 10x + 25，相减得 10x − 25 = 0.75，10x = 25.75，x = 2.575；2.575² ≈ 6.63，y² = 5 − 6.63 = −1.63，平方不会是负的，一组解也没有，两个圆碰不到；大臂加小臂 2.24 + 2.06 ≈ 4.3 格，胳膊全伸直手腕最远 4.3 格，到不了 5 格；目标正好 4.3 格远，两圆只碰一下，y² = 0，只有一组解：胳膊伸直；太近也不行：2.24 − 2.06 ≈ 0.18 格；手腕能到的地方是一个圆环：环里两组解，边上一组，环外一组也没有');

      var under = E.el('g', {}, s);
      var armG = E.el('g', {}, s);
      var arm = E.arm2(armG, A.OX, A.OY, A.S, 0, [A.OX - 190, A.OY - 210, 440, 420]), u = arm.u, V = arm.view;
      var pl = E.plane(under, { ox: A.OX, oy: A.OY, u: u, x0: -4, x1: 6, y0: -4, y1: 4 });
      var P = pl.P, O = P(0, 0);
      var top = E.el('g', {}, s);
      // 圆环（句 9）：外圆 4.3、内圆 0.18 之间涂绿
      var ringG = E.el('g', {}, under);
      var ringP = K.paint(E.el('path', { d: 'M ' + (O[0] - RMAX * u) + ' ' + O[1] + ' a ' + RMAX * u + ' ' + RMAX * u + ' 0 1 0 ' + 2 * RMAX * u + ' 0 a ' + RMAX * u + ' ' + RMAX * u + ' 0 1 0 ' + (-2 * RMAX * u) + ' 0 Z' +
        ' M ' + (O[0] - RMIN * u) + ' ' + O[1] + ' a ' + RMIN * u + ' ' + RMIN * u + ' 0 1 1 ' + 2 * RMIN * u + ' 0 a ' + RMIN * u + ' ' + RMIN * u + ' 0 1 1 ' + (-2 * RMIN * u) + ' 0 Z', 'fill-rule': 'evenodd', 'stroke-width': 2.5 }, ringG), E.tint(E.GOOD, 22, 'transparent'), E.GOOD);
      var c1 = E.circ(top, O[0], O[1], E.L1 * u, E.C1, 2.5, '9 6');
      var c2 = E.circ(top, 0, 0, E.L2 * u, E.C2, 2.5, '9 6');
      var tgt = E.target(top, 0, 0, 11, E.TGT);
      var tgLab = E.halo(E.txt(top, 0, 0, '(5, 0)', 20, E.TGT, 'middle', 900));
      // 句 3：x = 2.575
      var X25 = P(2.575, 0);
      var xl = K.paint(E.el('path', { d: 'M ' + X25[0] + ' ' + P(0, 3.2)[1] + ' V ' + P(0, -3.2)[1], fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '7 6' }, top), null, E.X);
      var xlLab = E.halo(E.txt(top, X25[0] + 8, P(0, 3.0)[1], 'x = 2.575', 19, E.X, 'start', 900));
      // 句 5：两圆之间的空隙；句 6：4.3 的量尺、到不了 5 的空隙
      function hseg(x0, x1, y, color, w) { var a = P(x0, y), b2 = P(x1, y); return K.paint(E.el('path', { d: 'M ' + a[0] + ' ' + a[1] + ' H ' + b2[0], fill: 'none', 'stroke-width': w || 6, 'stroke-linecap': 'round' }, top), null, color); }
      var gap1 = hseg(E.L1, 5 - E.L2, 0, E.ERR, 7);
      var dimG = E.el('g', {}, top), dy = 0.9 * u, M43 = P(RMAX, 0);
      K.paint(E.el('path', { d: 'M ' + O[0] + ' ' + (O[1] + dy - 7) + ' V ' + (O[1] + dy + 7) + ' M ' + O[0] + ' ' + (O[1] + dy) + ' H ' + M43[0] + ' M ' + M43[0] + ' ' + (O[1] + dy - 7) + ' V ' + (O[1] + dy + 7),
        fill: 'none', 'stroke-width': 2.5 }, dimG), null, E.INK);
      E.halo(E.txt(dimG, (O[0] + M43[0]) / 2, O[1] + dy + 24, '最远 4.3 格', 19, E.INK, 'middle', 900));
      var gap2 = hseg(RMAX, 5, 0.9 * -1, E.ERR, 5);
      var gap2X = E.cross(top, P(4.65, 0)[0], O[1] + dy, 7, E.ERR, 4);
      // 句 7：切点；句 8：最近 0.18 的小圆
      var tangent = K.paint(E.el('circle', { cx: P(E.L1, 0)[0], cy: O[1], r: 8, 'stroke-width': 3 }, top), E.S1, E.HI);
      var minC = E.circ(top, O[0], O[1], RMIN * u, E.ERR, 3);
      var minLab = E.halo(E.txt(top, O[0] - 14, O[1] - 18, '0.18', 19, E.ERR, 'end', 900));
      // 句 9：三个点
      function mark(x, n, color, below) {
        var g = E.el('g', {}, top), p = P(x, 0);
        K.paint(E.el('circle', { cx: p[0], cy: p[1], r: 8, 'stroke-width': 3 }, g), E.S1, color);
        E.halo(E.txt(g, p[0], below ? p[1] + 34 : p[1] - 16, n, 22, color, 'middle', 900));
        return g;
      }
      var mk2 = mark(1.5, '2 组', E.HI), mk1 = mark(RMAX, '1 组', E.GOOD), mk0 = mark(5, '0 组', E.ERR, true);   // 和 4.3 处的「1 组」只隔 0.7 格：写在点下面

      /* ══ 右栏 ══ */
      var Rg = E.el('g', {}, s), X = RX + 10, SZ = 22;
      function tx(g, y, tex, size) { return E.tex(g, X, y, tex, size || SZ, E.INK, 'start', { w: 340, h: 44 }); }
      function lab(g, y, str, color, size) { return E.txt(g, X, y, str, size || 21, color || E.INK2, 'start', 900); }
      var g1 = E.el('g', {}, Rg);
      var a0 = lab(g1, 30, '目标：手腕到 (5, 0)', E.TGT, 21);
      var a1 = tx(g1, 70, c(H.warn, '\\text{①}') + '\\;x^2+y^2=5');
      var a2 = tx(g1, 108, c(H.good, '\\text{②}') + '\\;(x-5)^2+y^2=4.25');
      var a3a = tx(g1, 150, '(x-5)^2=x^2'), a3b = tx(g1, 150, '(x-5)^2=x^2-10x'), a3 = tx(g1, 150, '(x-5)^2=x^2-10x+25');
      var a4a = tx(g1, 192, '\\text{①}-\\text{②：}\\;10x-25'), a4 = tx(g1, 192, '\\text{①}-\\text{②：}\\;10x-25=0.75');
      var a5a = tx(g1, 236, '10x=25.75'), a5 = tx(g1, 236, '10x=25.75，\\;' + c(H.x, 'x=2.575'));
      var a6a = tx(g1, 280, '2.575^2'), a6 = tx(g1, 280, '2.575^2\\approx 6.63');
      var a7a = tx(g1, 324, 'y^2=5-6.63'), a7 = tx(g1, 324, 'y^2=5-6.63=' + c(H.err, '-1.63'));
      var a8 = lab(g1, 374, '平方不会是负的', E.INK, 21), a8b = lab(g1, 410, '→ 一组解也没有：无解', E.ERR, 22);
      var g2 = E.el('g', {}, Rg);
      var b1 = tx(g2, 36, '\\text{大臂}+\\text{小臂}=2.24+2.06\\approx 4.3', 21);
      var b2 = lab(g2, 76, '全伸直：手腕最远 4.3 格，到不了 5', E.INK2, 20);
      var b3 = lab(g2, 128, '目标正好 4.3 格：两圆只碰一下', E.INK, 20);
      var b4 = tx(g2, 166, 'y^2=0\\;\\Rightarrow\\;' + c(H.good, '\\text{一组解（伸直）}'), 21);
      var b5 = lab(g2, 218, '太近也不行：', E.INK, 20);
      var b6 = tx(g2, 256, '2.24-2.06\\approx' + c(H.err, '0.18') + '\\text{ 格}', 21);
      var b7 = lab(g2, 296, '手腕离肩，最近也有 0.18 格', E.INK2, 20);
      var b8 = lab(g2, 350, '手腕能到的地方：一个圆环', E.GOOD, 22);
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
          var tx5 = lerp(1.5, 5, f(st, T.to5 - 0.4, 0.6));
          var tx_ = lerp(tx5, RMAX, f(st, T.move43[0], T.move43[1] - T.move43[0]));
          tx_ = lerp(tx_, 0.1, f(st, T.moveNear[0], T.moveNear[1] - T.moveNear[0]));
          var tp = P(tx_, 0), ringOn = f(st, T.ring - 0.6, 0.4);
          tgt.setAttribute('transform', 'translate(' + tp[0].toFixed(1) + ' ' + tp[1].toFixed(1) + ')');
          E.op(tgt, 1 - ringOn);
          tgLab.setAttribute('x', tp[0].toFixed(1)); tgLab.setAttribute('y', (tp[1] - 20).toFixed(1));
          tgLab.textContent = st < T.move43[0] ? '(5, 0)' : st < T.near ? '(4.3, 0)' : '(0.1, 0)';
          E.op(tgLab, f(st, T.to5, 0.3) * (1 - ringOn) * (st >= T.near ? 0 : 1));
          c2.setAttribute('cx', tp[0].toFixed(1)); c2.setAttribute('cy', tp[1].toFixed(1));
          E.op(c2, f(st, T.draw, 0.6) * (1 - ringOn));   // <circle> 没有 d，drawOn 的遮罩画不出来：淡入
          E.op(c1, 1 - 0.7 * ringOn);
          /* ── UR5：伸直 → 折回来 → 回到 (1.5, 0) 的肘上 ── */
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
          E.op(ringG, f(st, T.ring, 0.5));
          E.pop(mk2, P(1.5, 0)[0], O[1], f(st, T.inside, 0.35), 0.5); E.pop(mk1, M43[0], O[1], f(st, T.edge, 0.35), 0.5); E.pop(mk0, P(5, 0)[0], O[1], f(st, T.outside, 0.35), 0.5);

          /* ── 右栏 ── */
          E.op(g1, 1 - f(st, T.sum - 0.3, 0.4));
          E.op(a0, fr(st, T.to5)); E.op(a1, fr(st, T.eqs)); E.op(a2, fr(st, T.eqs + 0.3));
          [a1, a2].forEach(function (n) { E.glow(n, E.HI, b(st, T.sub, 0.9) > 0.05); });
          only([a3a, a3b, a3], st, [T.exp2, T.res - 2.2, T.exp3]);
          only([a4a, a4], st, [T.res, T.res2]);
          only([a5a, a5], st, [T.tenx, T.x]);
          only([a6a, a6], st, [T.sq, T.sqv]);
          only([a7a, a7], st, [T.y2, T.neg]);
          E.op(a8, fr(st, T.never)); E.op(a8b, fr(st, T.none));
          E.op(g2, f(st, T.sum - 0.2, 0.3));
          E.op(b1, fr(st, T.sum)); E.op(b2, fr(st, T.max)); E.op(b3, fr(st, T.move43[0])); E.op(b4, fr(st, T.y0));
          E.op(b5, fr(st, T.near)); E.op(b6, fr(st, T.diff)); E.op(b7, fr(st, T.minR)); E.op(b8, fr(st, T.ring));
          chips.forEach(function (cc) { E.pop(cc.g, cc.x, 391, f(st, cc.at, 0.35), 0.7); });
        }
      };
    }
  };
})();
