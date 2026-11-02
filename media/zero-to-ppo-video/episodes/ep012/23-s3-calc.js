// 012 第 3 幕：手算——开平方，正负两个。左边同第 2 幕的网格（两个圆淡淡地留着，紫色目标在 (1.5, 0)），右栏用 KaTeX 一行行推。
// 句 1「肘的位置记作 x、y」(1, 2) 处出点、写 (x, y)，「往右 x 格」原点往右画红虚线，「往上 y 格」再往上画绿虚线；
// 句 2「离肩二点二四格」斜边描琥珀、写 2.24，「勾股定理」直角亮一下，「x 的平方」「加 y 的平方」「等于五」右栏逐项写 ①；
// 句 3「离手腕二点零六格」从目标到肘描绿、写 2.06，「往右是 x 减一点五」目标往左画到 x 的红虚线、写 x − 1.5，「往上是 y」y 亮一下，
//    「平方加起来」「等于小臂长度的平方」「取四点二五」右栏写 ②，「四点二五开平方，约二点零六」写验算；
// 句 4「两个式子相减」① ② 框亮，「y 的平方正好消掉」两个 y² 打橙色斜杠，「左边剩下」「右边五减四点二五」「零点七五」逐项写；
// 句 5「展开」(x − 1.5)² = x² − 3x + 2.25 逐项写，「前面的 x 平方一减，也消掉」两个 x² 打斜杠，「三倍的 x，减去二点二五」写出；
// 句 6「三倍的 x 减去二点二五，等于零点七五」「三倍的 x 是三」「x 等于一」逐行写，网格上画 x = 1 的竖虚线（红），穿过两个圆的交点；
// 句 7「代回第一个式子」「一的平方是一」「五减一是四」「y 的平方等于四」逐项写；
// 句 8「二乘以二是四」「负二乘以负二，也是四」「y 可以是二」(1, 2) 出亮黄圈，「也可以是负二」(1, −2) 出青圈；
// 句 9「肘上」「肘下」写名字，「开平方」「正的一个」「负的一个」右栏写；
// 句 10–13 两台 UR5 回到正常亮度：「大臂从肩指向一、二」量肩的转角（蓝弧，63.4°），「小臂从肘指向手腕」画 (0.5, −2)、在肘上量方向（−76°），
//    「肘这个关节，转的是…之间的角度」大臂的延长线（虚线）转到小臂（−139.4°），「肘下正好反过来」肘下这台亮起来、量 −63.4°、+139.4°；
// 句 14「先走大臂一、二，再走小臂零点五、负二」两段箭头头尾相接到目标，「一加零点五」「二减二」右栏写，「肘下那组」青色两段也到目标。
// 时刻取自配音的逐词时间，见 ../ep012.py。
(function () {
  var E = window.EP012, K = E.K, H = E.HEX;
  var D = Math.PI / 180, A = E.ARM, RX = A.RX;
  var T = {
    pts: 2.37, pos: 3.97, xy: 4.61, xr: 5.71, yu: 6.5,
    d1: 8.37, d1n: 8.9, py: 10.2, x2: 11.16, y2: 12.06, eq5: 13.17,
    d2: 14.85, d2n: 15.55, xm: 17.15, yy: 18.67, sq: 19.64, eqL: 20.65, take: 23.53, sqrt: 24.81, ok: 26.32, out1: 28.1,
    sub: 28.46, ycan: 29.66, left: 31.34, lx: 32.05, minus: 32.9, right: 34.84, r075: 36.46,
    exp: 38.61, expand: 40.15, m3x: 41.85, p225: 43.08, front: 44.51, xcan: 45.57, only: 46.83, res: 47.9,
    e1: 51.11, e1r: 53.38, e2: 54.74, e3: 56.15, xline: [56.2, 56.9], out2: 57.8,
    back: 58.02, one: 59.21, fm1: 60.43, four: 61.2, y2b: 61.58,
    sq4: 64.0, t1: 66.19, t2: 67.67, so: 70.06, yp: 71.06, yn: 72.08,
    up: 75.21, dn: 77.3, src: 80.15, pos2: 81.3, neg: 82.44, out3: 84.3,
    ang: 84.54, upper: 86.46, r1: 88.46, calc: 89.96, deg: 92.12,
    fore: 94.48, dx: 96.31, dxr: 97.52, dy: 98.31, dyr: 99.15, dir2: 99.91, deg2: 100.79,
    el: 103.38, rel: 104.9, m76: 107.16, m634: 108.15, res3: 109.63,
    dn2: 112.52, sh2: 113.98, el2: 116.01, sign: 118.43, sneg: 120.04, spos: 120.6, out4: 122.0,
    chk: 122.23, upA: 124.6, upB: 126.12, add1: 128.24, add1r: 129.48, add2: 130.39, add2r: 131.36, hit: 131.75, dnChk: 133.38
  };
  var END = 136.31;
  var UP = E.ik2(1.5, 0, true), DN = E.ik2(1.5, 0, false);
  function c(hex, s) { return '\\textcolor{' + hex + '}{' + s + '}'; }
  function cancel(s) { return c(H.err, '\\cancel{' + c(H.ink, s) + '}'); }

  E.scenes[2] = {
    title: '手算：开平方，正负两个',
    dur: END,
    cues: [
      { at: 0, s: '肘的位置记作 (x, y)：往右 x 格，往上 y 格' },
      { at: 8.28, s: '① 离肩 2.24 格：$x^2+y^2=5$' },
      { at: 14.76, s: '② 离**手腕** 2.06 格：$(x-1.5)^2+y^2=4.25$（$\\sqrt{4.25}\\approx 2.06$）' },
      { at: 28.37, s: '① − ②：$y^2$ 消掉，$x^2-(x-1.5)^2=5-4.25=0.75$' },
      { at: 38.52, s: '$(x-1.5)^2=x^2-3x+2.25$，左边只剩 $3x-2.25$' },
      { at: 51.02, s: '$3x-2.25=0.75$：$3x=3$，$x=1$' },
      { at: 57.93, s: '代回 ①：$1+y^2=5$，$y^2=4$' },
      { at: 63.91, s: '$2\\times 2=4$，$(-2)\\times(-2)=4$：$y=2$ 或 $y=-2$' },
      { at: 73.7, s: '**肘上** (1, 2)、**肘下** (1, −2)：两组解来自**开平方**的正负两个' },
      { at: 84.45, s: '大臂指向 (1, 2)：方向约 63.4°' },
      { at: 94.39, s: '小臂 (1.5 − 1, 0 − 2) = (0.5, −2)：方向约 −76°' },
      { at: 103.29, s: '肘转的**角度** = −76 − 63.4 ≈ −139.4°' },
      { at: 112.43, s: '**肘下**：肩 −63.4°、肘 +139.4°——肘的**角度**一负一正' },
      { at: 122.13, s: '验算：(1, 2) + (0.5, −2) = (1.5, 0)，正好到**手腕**' }
    ],
    build: function (K) {
      var s = K.sceneSvg('肘的位置记作 (x, y)；离肩 2.24 格，勾股定理 x² + y² = 5；离手腕 2.06 格，(x − 1.5)² + y² = 4.25（4.25 开平方约 2.06）；两式相减 y² 消掉：x² − (x − 1.5)² = 5 − 4.25 = 0.75；(x − 1.5)² 展开是 x² − 3x + 2.25，x² 也消掉，左边只剩 3x − 2.25；3x − 2.25 = 0.75，3x = 3，x = 1；代回第一个式子 1 + y² = 5，y² = 4；2 × 2 = 4，(−2) × (−2) = 4，y 是 2 或 −2：肘上 (1, 2)、肘下 (1, −2)，两组解来自开平方的正负两个；关节角：大臂指向 (1, 2) 约 63.4°，小臂 (0.5, −2) 方向约 −76°，肘转 −76 − 63.4 ≈ −139.4°；肘下反过来：肩 −63.4°，肘 +139.4°；验算 (1, 2) + (0.5, −2) = (1.5, 0)');

      var under = E.el('g', {}, s);
      var box = [A.OX - 110, A.OY - 214, 440, 420];
      var ghostG = E.el('g', {}, s), ghost = E.arm2(ghostG, A.OX, A.OY, A.S, 0, box);
      var armG = E.el('g', {}, s), arm = E.arm2(armG, A.OX, A.OY, A.S, 0, box), u = arm.u, V = arm.view;
      arm.pose(UP[0], UP[1]); ghost.pose(DN[0], DN[1]);
      var pl = E.plane(under, { ox: A.OX, oy: A.OY, u: u, x0: -1, x1: 4, y0: -3, y1: 3 });
      var P = pl.P, O = P(0, 0), TG = P(1.5, 0), EU = P(1, 2), ED = P(1, -2), F10 = P(1, 0);
      var top = E.el('g', {}, s);
      var c1 = E.circ(top, O[0], O[1], E.L1 * u, E.C1, 2.5, '9 6'), c2 = E.circ(top, TG[0], TG[1], E.L2 * u, E.C2, 2.5, '9 6');
      var tgt = E.target(top, TG[0], TG[1], 13, E.TGT);
      function seg(color, w, dash) { var p = K.paint(E.el('path', { fill: 'none', 'stroke-width': w || 4, 'stroke-linecap': 'round' }, top), null, color); if (dash) p.setAttribute('stroke-dasharray', dash); return p; }
      function line(p, a, b) { p.setAttribute('d', 'M ' + a[0].toFixed(1) + ' ' + a[1].toFixed(1) + ' L ' + b[0].toFixed(1) + ' ' + b[1].toFixed(1)); return p; }
      // 句 1–3：两个直角三角形
      var gx = line(seg(E.X, 4, '8 5'), O, F10), gy = line(seg(E.Y, 4, '8 5'), F10, EU), gxm = line(seg(E.X, 4, '8 5'), TG, F10);
      var hyp1 = line(seg(E.C1, 5), O, EU), hyp2 = line(seg(E.C2, 5), EU, TG);
      var corner = K.paint(E.el('path', { d: 'M ' + (F10[0] - 10) + ' ' + F10[1] + ' V ' + (F10[1] - 10) + ' H ' + F10[0], fill: 'none', 'stroke-width': 2 }, top), null, E.INK2);
      var lx = E.halo(E.txt(top, (O[0] + F10[0]) / 2 - 2, O[1] + 28, 'x', 24, E.X, 'middle', 900)), ly = E.halo(E.txt(top, F10[0] - 14, (F10[1] + EU[1]) / 2 + 8, 'y', 24, E.Y, 'end', 900));
      [lx, ly].forEach(function (n) { n.style.fontFamily = 'KaTeX_Math, serif'; n.style.fontStyle = 'italic'; });
      var lxm = E.halo(E.txt(top, TG[0] + 16, O[1] + 28, 'x − 1.5', 20, E.X, 'start', 900));
      var l224 = E.halo(E.txt(top, (O[0] + EU[0]) / 2 - 16, (O[1] + EU[1]) / 2 - 2, '2.24', 21, E.C1, 'end', 900));
      var l206 = E.halo(E.txt(top, (EU[0] + TG[0]) / 2 + 14, (EU[1] + TG[1]) / 2 - 4, '2.06', 21, E.C2, 'start', 900));
      var pxy = K.paint(E.el('circle', { cx: EU[0], cy: EU[1], r: 7, 'stroke-width': 3 }, top), E.S1, E.HI);
      var pxyLab = E.halo(E.txt(top, EU[0] - 16, EU[1] - 16, '(x, y)', 22, E.HI, 'end', 900));
      // 句 6：x = 1 的竖虚线
      var xl = K.paint(E.el('path', { d: 'M ' + F10[0] + ' ' + P(1, 3)[1] + ' V ' + P(1, -3)[1], fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '8 6' }, top), null, E.X);
      var xlLab = E.halo(E.txt(top, F10[0] + 10, P(1, 2.75)[1], 'x = 1', 21, E.X, 'start', 900));
      // 句 8–9：两个交点
      var ringU = E.el('g', {}, top), ringD = E.el('g', {}, top);
      K.paint(E.el('circle', { cx: EU[0], cy: EU[1], r: 13, 'stroke-width': 4, fill: 'none' }, ringU), null, E.UP);
      K.paint(E.el('circle', { cx: ED[0], cy: ED[1], r: 13, 'stroke-width': 4, fill: 'none' }, ringD), null, E.DOWN);
      var yU = E.halo(E.txt(top, EU[0] + 20, EU[1] + 8, 'y = 2', 21, E.UP, 'start', 900)), yD = E.halo(E.txt(top, ED[0] + 20, ED[1] + 8, 'y = −2', 21, E.DOWN, 'start', 900));
      var nU = E.halo(E.txt(top, EU[0] - 18, EU[1] - 16, '肘上 (1, 2)', 21, E.UP, 'end', 900)), nD = E.halo(E.txt(top, ED[0] - 18, ED[1] + 32, '肘下 (1, −2)', 21, E.DOWN, 'end', 900));
      // 句 10–13：转角
      var shArc = E.arcArrow(top, O[0], O[1], 0.62 * u, E.OMEGA, 3.5, { head: 11 });
      var shLab = E.halo(E.txt(top, 0, 0, '', 21, E.OMEGA, 'end', 900));
      var foreV = E.vec(top, E.TGT, 4, { head: 13 });
      var hRef = seg(E.INK2, 2.5, '5 5'), ext = seg(E.C1, 3, '7 6');
      var elArc = E.arcArrow(top, 0, 0, 0.62 * u, E.OMEGA, 3.5, { head: 11 });
      var elLab = E.halo(E.txt(top, 0, 0, '', 21, E.OMEGA, 'start', 900));
      // 句 14：头尾相接的两段
      var vU1 = E.vec(top, E.UP, 4.5, { head: 14 }), vU2 = E.vec(top, E.UP, 4.5, { head: 14 }), vD1 = E.vec(top, E.DOWN, 4.5, { head: 14 }), vD2 = E.vec(top, E.DOWN, 4.5, { head: 14 });

      /* ══ 右栏 ══ */
      var Rg = E.el('g', {}, s), X = RX + 10, SZ = 25;
      function tx(g, y, tex, size, w) { return E.tex(g, X, y, tex, size || SZ, E.INK, 'start', { w: w || 340, h: 46 }); }
      function lab(g, y, str, color, size) { return E.txt(g, X, y, str, size || 21, color || E.INK2, 'start', 900); }
      // 第一组：两个式子
      var g1 = E.el('g', {}, Rg);
      var p1 = lab(g1, 40, '肘的位置：(x, y)', E.HI, 22);
      var q1L = lab(g1, 92, '① 离肩 2.24 格（勾股定理）', E.C1);
      var q1 = tx(g1, 132, c(H.warn, '\\text{①}') + '\\;x^2+y^2=5');
      var q1a = tx(g1, 132, c(H.warn, '\\text{①}') + '\\;x^2'), q1b = tx(g1, 132, c(H.warn, '\\text{①}') + '\\;x^2+y^2');
      var q2L = lab(g1, 196, '② 离手腕 2.06 格', E.C2);
      var q2 = tx(g1, 236, c(H.good, '\\text{②}') + '\\;(x-1.5)^2+y^2=4.25');
      var q2a = tx(g1, 236, c(H.good, '\\text{②}') + '\\;(x-1.5)^2+y^2'), q2b = tx(g1, 236, c(H.good, '\\text{②}') + '\\;(x-1.5)^2+y^2=\\text{小臂}^2');
      var q2n = lab(g1, 284, '为了好算，取 4.25', E.INK2, 20);
      var q2c = E.txt(g1, X, 332, '√4.25 ≈ 2.06 ✓', 24, E.GOOD, 'start', 900);   // 不用 KaTeX 的 \\sqrt：它的根号是 400em 宽、靠 overflow 裁掉的 svg，版式检查会量成超出画面
      // 第二组：相减
      var g2 = E.el('g', {}, Rg), S2 = 22;
      var r1 = tx(g2, 36, c(H.warn, '\\text{①}') + '\\;x^2+y^2=5', S2), r1c = tx(g2, 36, c(H.warn, '\\text{①}') + '\\;x^2+' + cancel('y^2') + '=5', S2);
      var r2 = tx(g2, 76, c(H.good, '\\text{②}') + '\\;(x-1.5)^2+y^2=4.25', S2), r2c = tx(g2, 76, c(H.good, '\\text{②}') + '\\;(x-1.5)^2+' + cancel('y^2') + '=4.25', S2);
      var sLab = lab(g2, 122, '① − ②：y² 消掉', E.INK, 20);
      var s1a = tx(g2, 164, 'x^2-(x-1.5)^2', S2), s1b = tx(g2, 164, 'x^2-(x-1.5)^2=5-4.25', S2), s1c = tx(g2, 164, 'x^2-(x-1.5)^2=5-4.25=0.75', S2);
      var eLab = lab(g2, 214, '展开：', E.INK2, 20);
      var x1a = tx(g2, 254, '(x-1.5)^2=x^2', S2), x1b = tx(g2, 254, '(x-1.5)^2=x^2-3x', S2), x1c = tx(g2, 254, '(x-1.5)^2=x^2-3x+2.25', S2);
      var x2a = tx(g2, 302, 'x^2-(x^2-3x+2.25)', S2), x2b = tx(g2, 302, cancel('x^2') + '-(' + cancel('x^2') + '-3x+2.25)', S2), x2c = tx(g2, 302, cancel('x^2') + '-(' + cancel('x^2') + '-3x+2.25)=3x-2.25', S2);
      var z1 = tx(g2, 350, '3x-2.25=0.75', S2), z2 = tx(g2, 350, '3x-2.25=0.75\\;\\Rightarrow\\;3x=3', S2), z3 = tx(g2, 396, '\\Rightarrow\\;' + c(H.x, 'x=1'), 26);
      // 第三组：求 y
      var g3 = E.el('g', {}, Rg);
      var w1 = lab(g3, 40, 'x = 1，代回 ①：', E.INK, 21);
      var w2 = tx(g3, 82, '1^2+y^2=5'), w3 = tx(g3, 126, 'y^2=5-1=4'), w3a = tx(g3, 126, 'y^2=5-1');
      var w4 = lab(g3, 180, '平方等于 4 的数，有 2 个：', E.INK2, 20);
      var w5 = tx(g3, 222, '2\\times 2=4'), w6 = tx(g3, 262, '(-2)\\times(-2)=4');
      var w7 = tx(g3, 312, 'y=' + c(H.hi, '2') + '\\;\\text{或}\\;y=' + c(H.down, '-2'), 28), w7a = tx(g3, 312, 'y=' + c(H.hi, '2'), 28);
      var w8 = tx(g3, 362, '\\text{开平方：}\\;' + c(H.hi, '+') + '\\text{ 一个，}' + c(H.down, '-') + '\\text{ 一个}', 22), w8a = tx(g3, 362, '\\text{开平方：}\\;' + c(H.hi, '+') + '\\text{ 一个}', 22), w8b = tx(g3, 362, '\\text{开平方}', 22);
      // 第四组：关节角
      var g4 = E.el('g', {}, Rg), S4 = 23;
      var k1 = lab(g4, 36, '肘上：大臂指向 (1, 2)', E.UP, 21);
      var k2 = tx(g4, 74, '\\text{往右 }1\\text{、往上 }2', 21), k3 = tx(g4, 112, '\\text{计算器 }\\operatorname{atan2}(2,1)\\approx' + c(H.acc, '63.4^\\circ'), 21);
      var k4 = lab(g4, 160, '小臂：从肘指向手腕', E.INK, 21);
      var k5a = tx(g4, 198, '(1.5-1,\\;0-2)', S4), k5b = tx(g4, 198, '(1.5-1,\\;0-2)=(0.5,', S4), k5 = tx(g4, 198, '(1.5-1,\\;0-2)=(0.5,\\,-2)', S4);
      var k6 = tx(g4, 236, '\\text{方向}\\approx' + c(H.acc, '-76^\\circ'), 21);
      var k7 = lab(g4, 284, '肘转的角 = 小臂方向 − 大臂方向', E.INK, 20);
      var k8a = tx(g4, 322, '-76', S4), k8b = tx(g4, 322, '-76-63.4', S4), k8 = tx(g4, 322, '-76-63.4\\approx' + c(H.acc, '-139.4^\\circ'), S4);
      var k9 = tx(g4, 366, c(H.down, '\\text{肘下：}') + '\\text{肩 }' + c(H.acc, '-63.4^\\circ') + '\\text{，肘 }' + c(H.acc, '+139.4^\\circ'), 21);
      var k9a = tx(g4, 366, c(H.down, '\\text{肘下：}') + '\\text{肩 }' + c(H.acc, '-63.4^\\circ'), 21);
      var k10 = lab(g4, 404, '肘的角度：一负、一正', E.HI, 20);
      // 第五组：验算
      var g5 = E.el('g', {}, Rg);
      var v1 = lab(g5, 60, '验算：先走大臂，再走小臂', E.INK, 22);
      var v2a = tx(g5, 108, c(H.hi, '(1,\\,2)'), 27), v2 = tx(g5, 108, c(H.hi, '(1,\\,2)') + '+' + c(H.hi, '(0.5,\\,-2)'), 27);
      var v3 = tx(g5, 156, '1+0.5=1.5', 25), v4 = tx(g5, 200, '2-2=0', 25);
      var v5 = tx(g5, 252, '=' + c(H.violet, '(1.5,\\,0)') + '\\;' + c(H.good, '\\checkmark'), 28);
      var v6 = tx(g5, 330, c(H.down, '\\text{肘下：}(1,\\,-2)+(0.5,\\,2)') + '=' + c(H.violet, '(1.5,\\,0)') + '\\;' + c(H.good, '\\checkmark'), 21);

      function fr(st, a) { return E.fade(st, a, 0.3); }
      function only(list, st, times) {   // 同一行的几个版本：到了哪个时刻就只显示哪一版
        var k = -1; times.forEach(function (tt, i) { if (st >= tt) k = i; });
        list.forEach(function (n, i) { E.op(n, i === k ? 1 : 0); });
      }

      return {
        el: s,
        draw: function (st, t) {
          var f = E.fade, b = E.bump;
          /* ── 两台 UR5：句 1–9 淡，句 10 起亮；句 13 肘下这台亮、肘上这台淡 ── */
          var armOn = f(st, T.ang, 0.4) * (1 - f(st, T.chk, 0.4) * 0.5);
          V.clearTint(); V.tint('all', null, 1, 0.2 + 0.8 * armOn * (1 - 0.6 * f(st, T.dn2, 0.4) * (1 - f(st, T.out4, 0.4))));
          ghost.view.clearTint(); ghost.view.tint('all', E.DOWN, 0.8, 0.25 + 0.35 * f(st, T.dn2, 0.4) * (1 - f(st, T.out4, 0.4)));
          E.op(ghostG, f(st, T.yn, 0.4));   // 算出 y = −2 之前不出现肘下这台，不提前给答案
          E.op(c1, 0.6 * (1 - 0.5 * f(st, T.ang, 0.4))); E.op(c2, 0.6 * (1 - 0.5 * f(st, T.ang, 0.4)));

          /* ── 句 1–3 ── */
          var on1 = 1 - f(st, T.out1, 0.4);
          E.op(pxy, fr(st, T.pts) * (1 - f(st, T.out2, 0.3))); E.op(pxyLab, fr(st, T.xy) * on1);
          E.drawOn(gx, f(st, T.xr, 0.5)); E.op(gx, (f(st, T.xr, 0.1) > 0 ? 1 : 0) * on1); E.op(lx, fr(st, T.xr) * on1);
          E.drawOn(gy, f(st, T.yu, 0.5)); E.op(gy, (f(st, T.yu, 0.1) > 0 ? 1 : 0) * on1); E.op(ly, fr(st, T.yu) * on1);
          ly.style.fontSize = (24 * (1 + 0.3 * b(st, T.yy, 0.8))) + 'px';
          E.drawOn(hyp1, f(st, T.d1, 0.5)); E.op(hyp1, (f(st, T.d1, 0.1) > 0 ? 1 : 0) * on1); E.op(l224, fr(st, T.d1n) * on1);
          E.op(corner, fr(st, T.py) * on1); E.glow(corner, E.HI, b(st, T.py, 0.9) > 0.05);
          E.drawOn(hyp2, f(st, T.d2, 0.5)); E.op(hyp2, (f(st, T.d2, 0.1) > 0 ? 1 : 0) * on1); E.op(l206, fr(st, T.d2n) * on1);
          E.drawOn(gxm, f(st, T.xm, 0.4)); E.op(gxm, (f(st, T.xm, 0.1) > 0 ? 1 : 0) * on1); E.op(lxm, fr(st, T.xm) * on1);
          /* ── 句 6：x = 1 ── */
          E.drawOn(xl, f(st, T.xline[0], T.xline[1] - T.xline[0])); E.op(xl, (f(st, T.xline[0], 0.1) > 0 ? 0.9 : 0) * (1 - f(st, T.out3, 0.4)));
          E.op(xlLab, fr(st, T.xline[1]) * (1 - f(st, T.out3, 0.4)));
          /* ── 句 8–9：两个交点 ── */
          var onR = 1 - f(st, T.out3, 0.4);
          E.pop(ringU, EU[0], EU[1], f(st, T.yp, 0.35), 0.4); E.pop(ringD, ED[0], ED[1], f(st, T.yn, 0.35), 0.4);
          E.op(ringU, f(st, T.yp, 0.35) * (1 - 0.6 * f(st, T.ang, 0.4))); E.op(ringD, f(st, T.yn, 0.35) * (1 - 0.6 * f(st, T.ang, 0.4)));
          E.op(yU, fr(st, T.yp) * (1 - f(st, T.up, 0.3))); E.op(yD, fr(st, T.yn) * (1 - f(st, T.dn, 0.3)));
          E.op(nU, fr(st, T.up) * onR); E.op(nD, fr(st, T.dn) * onR);
          [ringU, ringD].forEach(function (g) { E.glow(g, E.HI, b(st, T.src, 1.0) > 0.05); });

          /* ── 句 10–13：转角 ── */
          var dnOn = f(st, T.dn2, 0.4);
          var useA1 = dnOn > 0.5 ? DN[0] : UP[0], useA2 = dnOn > 0.5 ? DN[1] : UP[1], eP = dnOn > 0.5 ? ED : EU;
          var shp = f(st, T.upper, 0.6) * (1 - f(st, T.dn2 - 0.3, 0.3)) + f(st, T.sh2, 0.6);
          shArc.set(0, useA1 * Math.max(1e-3, Math.min(1, shp)));
          E.op(shArc.g, (shp > 0 ? 1 : 0) * (1 - f(st, T.out4, 0.3)));
          var shL = P(-0.42, dnOn > 0.5 ? -0.62 : 0.42);   // y 轴左边：弧的右边是小臂和目标
          shLab.textContent = dnOn > 0.5 ? '−63.4°' : '63.4°';
          shLab.setAttribute('x', shL[0].toFixed(1)); shLab.setAttribute('y', (shL[1] + 8).toFixed(1));
          E.op(shLab, (dnOn > 0.5 ? fr(st, T.sh2 + 0.4) : fr(st, T.deg) * (1 - f(st, T.dn2 - 0.3, 0.3))) * (1 - f(st, T.out4, 0.3)));
          // 小臂：从肘指向手腕（紫），方向在肘上量（水平参考线 → 小臂）
          foreV.set(EU[0], EU[1], EU[0] + (TG[0] - EU[0]) * f(st, T.fore, 0.6), EU[1] + (TG[1] - EU[1]) * f(st, T.fore, 0.6));
          E.op(foreV.g, (f(st, T.fore, 0.1) > 0 ? 1 : 0) * (1 - f(st, T.el, 0.3)));
          line(hRef, [eP[0] - 0.2 * u, eP[1]], [eP[0] + 1.1 * u, eP[1]]);
          E.op(hRef, fr(st, T.dir2) * (1 - f(st, T.el, 0.3)));
          var extE = [eP[0] + 1.2 * u * Math.cos(useA1), eP[1] - 1.2 * u * Math.sin(useA1)];
          line(ext, eP, extE); E.drawOn(ext, f(st, T.el, 0.5)); E.op(ext, (f(st, T.el, 0.1) > 0 ? 1 : 0) * (1 - f(st, T.out4, 0.3)));
          elArc.g.setAttribute('transform', 'translate(' + eP[0].toFixed(1) + ' ' + eP[1].toFixed(1) + ')');
          var dirPhase = f(st, T.dir2, 0.6) * (1 - f(st, T.el, 0.3)), relPhase = f(st, T.rel, 0.8) * (1 - f(st, T.dn2 - 0.3, 0.3)) + f(st, T.el2, 0.8);
          if (relPhase > 0) elArc.set(useA1, useA1 + (useA2 - useA1) * Math.min(1, relPhase));
          else elArc.set(0, Math.min(-1e-3, useA2 * dirPhase));
          E.op(elArc.g, Math.max(dirPhase > 0 ? 1 : 0, relPhase > 0 ? 1 : 0) * (1 - f(st, T.out4, 0.3)));
          var mid = relPhase > 0 ? useA1 + (useA2 - useA1) / 2 : useA2 / 2, rr = 1.05 * u;
          elLab.setAttribute('x', (eP[0] + rr * Math.cos(mid) + 6).toFixed(1)); elLab.setAttribute('y', (eP[1] - rr * Math.sin(mid) + 8).toFixed(1));
          elLab.textContent = relPhase > 0 ? (dnOn > 0.5 ? '+139.4°' : '−139.4°') : '−76°';
          E.op(elLab, (relPhase > 0 ? (dnOn > 0.5 ? fr(st, T.el2 + 0.5) : fr(st, T.res3)) : fr(st, T.deg2) * (1 - f(st, T.el, 0.3))) * (1 - f(st, T.out4, 0.3)));

          /* ── 句 14：头尾相接 ── */
          var a1p = f(st, T.upA, 0.6), a2p = f(st, T.upB, 0.6);
          vU1.set(O[0], O[1], O[0] + (EU[0] - O[0]) * a1p, O[1] + (EU[1] - O[1]) * a1p); E.op(vU1.g, a1p > 0 ? 1 : 0);
          vU2.set(EU[0], EU[1], EU[0] + (TG[0] - EU[0]) * a2p, EU[1] + (TG[1] - EU[1]) * a2p); E.op(vU2.g, a2p > 0 ? 1 : 0);
          var d1p = f(st, T.dnChk, 0.5), d2p = f(st, T.dnChk + 0.5, 0.5);
          vD1.set(O[0], O[1], O[0] + (ED[0] - O[0]) * d1p, O[1] + (ED[1] - O[1]) * d1p); E.op(vD1.g, d1p > 0 ? 1 : 0);
          vD2.set(ED[0], ED[1], ED[0] + (TG[0] - ED[0]) * d2p, ED[1] + (TG[1] - ED[1]) * d2p); E.op(vD2.g, d2p > 0 ? 1 : 0);
          tgt.setAttribute('transform', 'translate(' + TG[0] + ' ' + TG[1] + ') scale(' + (1 + 0.5 * Math.max(b(st, T.hit, 1.0), b(st, T.dnChk + 1.0, 1.0))) + ') translate(' + (-TG[0]) + ' ' + (-TG[1]) + ')');

          /* ── 右栏 ── */
          E.op(g1, 1 - f(st, T.out1, 0.4));
          E.op(p1, fr(st, T.pos));
          E.op(q1L, fr(st, T.d1)); only([q1a, q1b, q1], st, [T.x2, T.y2, T.eq5]);
          E.op(q2L, fr(st, T.d2)); only([q2a, q2b, q2], st, [T.sq, T.eqL, T.take]);
          E.op(q2n, fr(st, T.take)); E.op(q2c, fr(st, T.sqrt));
          E.op(g2, f(st, T.sub - 0.2, 0.3) * (1 - f(st, T.out2, 0.4)));
          only([r1, r1c], st, [0, T.ycan]); only([r2, r2c], st, [0, T.ycan]);
          [r1, r1c, r2, r2c].forEach(function (n) { E.glow(n, E.HI, b(st, T.sub, 0.9) > 0.05); });
          E.op(sLab, fr(st, T.sub)); only([s1a, s1b, s1c], st, [T.left, T.right, T.r075]);
          E.op(eLab, fr(st, T.exp)); only([x1a, x1b, x1c], st, [T.expand, T.m3x, T.p225]);
          only([x2a, x2b, x2c], st, [T.front, T.xcan, T.res]);
          only([z1, z2], st, [T.e1, T.e2]); E.op(z3, fr(st, T.e3));
          E.op(g3, f(st, T.back - 0.2, 0.3) * (1 - f(st, T.out3, 0.4)));
          E.op(w1, fr(st, T.back)); E.op(w2, fr(st, T.one)); only([w3a, w3], st, [T.fm1, T.four]);
          E.op(w4, fr(st, T.sq4)); E.op(w5, fr(st, T.t1)); E.op(w6, fr(st, T.t2));
          only([w7a, w7], st, [T.yp, T.yn]); only([w8b, w8a, w8], st, [T.src, T.pos2, T.neg]);
          E.op(g4, f(st, T.ang - 0.2, 0.3) * (1 - f(st, T.out4, 0.4)));
          E.op(k1, fr(st, T.upper)); E.op(k2, fr(st, T.r1)); E.op(k3, fr(st, T.calc));
          E.op(k4, fr(st, T.fore)); only([k5a, k5b, k5], st, [T.dx, T.dxr, T.dyr]); E.op(k6, fr(st, T.deg2));
          E.op(k7, fr(st, T.el)); only([k8a, k8b, k8], st, [T.m76, T.m634, T.res3]);
          only([k9a, k9], st, [T.sh2, T.el2]); E.op(k10, fr(st, T.sign));
          E.op(g5, f(st, T.chk - 0.1, 0.3));
          E.op(v1, fr(st, T.chk)); only([v2a, v2], st, [T.upA, T.upB]); E.op(v3, fr(st, T.add1)); E.op(v4, fr(st, T.add2));
          E.op(v5, fr(st, T.hit)); E.op(v6, fr(st, T.dnChk));
        }
      };
    }
  };
})();
