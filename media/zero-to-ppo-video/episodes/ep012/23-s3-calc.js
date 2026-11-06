// 012 第 3 幕：手算——开平方，正负两个。左边同第 2 幕的网格（两个圆淡淡地留着，紫色目标是手 (1, 0)），右栏用 KaTeX 一行行推。
// 2026-11-05 参照 011 重做：手照第 11 集放在小臂上离肘两格，目标 (1, 0)；按「观众完全零基础」把每一步的写法当场讲清。
// 句 1「肘的位置先不知道」网格上出一个还不知道在哪的肘（示意点 (1.6, 1.5)），「两个字母代替」写 (x, y)，「往右 x 格」原点往右画红虚线，「往上 y 格」再往上画绿虚线；
// 句 2「离肩二点二四格」斜边描琥珀、写「要 = 2.24」，「勾股定理」直角亮一下，「往右的平方」「加上往上的平方」「等于斜着的那条边的平方」右栏写一般说法，
//    「x 的平方」「加 y 的平方」「等于五」逐项写 ①；
// 句 3「离手两格」从手到肘描绿、写「要 = 2」，「从手到肘，往右是 x 减一」手往右画到 x 的红虚线、写 x − 1，「往上是 y」y 亮一下（两个三角形共用这条竖边），
//    「平方加起来」「等于二的平方，也就是四」逐项写 ②；
// 句 4「等号两边都一样多」「左边减左边、右边减右边」右栏写，「y 的平方正好减掉了」两个 y² 打橙色斜杠，「右边五减四，是一」写；
// 句 5「左边剩下」写 x² − (x − 1)²，「x 减一的平方，就是 x 减一，乘以 x 减一」写；
// 句 6「拆开来，每一项都乘一遍」左边网格淡掉、换成 2 × 2 的表（上面 x、−1，左边 x、−1），「x 乘以 x」「x 乘以负一」「负一乘以 x」「负一乘以负一」逐格填，
//    「负负得正」写在表下；句 7「合起来」表下写 x² − 2x + 1，右栏 (x − 1)² = x² − 2x + 1；
// 句 8「再整个减掉」写 x² − (x² − 2x + 1)，「括号前面是减号」减号亮，「括号里每一项都要变号」写 x² − x² + 2x − 1、变号的项亮，「就没有了」两个 x² 打斜杠，「只剩」写 2x − 1；
// 句 9 表收起、网格回来：「两倍的 x 减一，等于一」「两边都加一」「两倍的 x 是二」「两边都除以二」「x 等于一」逐行写，网格上画 x = 1 的竖虚线（红）；
// 句 10「放回第一个式子」「一的平方是一」「五减一，等于四」逐项写；句 11「二乘以二」「负二乘以负二，负负得正」「y 可以是二」(1, 2) 出亮黄圈，「也可以是负二」(1, −2) 出青圈；
// 句 12「叫做开平方」「总有两个答案：一正、一负」右栏写；句 13「肘上」「肘下」写名字，「开平方」两个圈亮一下；
// 句 14–17 两台 UR5 亮起来：「大臂从肩指向一、二」量肩的转角（蓝弧，63.4°），「电脑里有现成的函数」右栏写 atan2（第 11 集讲过函数），
//    「小臂从肘指向手」画 (0, −2)、在肘上量方向（−90°），「肘这个关节，转的是…之间的角度」大臂的延长线（虚线）转到小臂（−153.4°），「肘下正好反过来」量 −63.4°、+153.4°；
// 句 18「先走大臂一、二，再走小臂零、负二」两段箭头头尾相接到手，「一加零」「二减二」右栏写，「肘下那组」青色两段也到手。
// 时刻取自配音的逐词时间，见 ../ep012.py。
(function () {
  var E = window.EP012, K = E.K, H = E.HEX;
  var D = Math.PI / 180, A = E.ARM, RX = A.RX;
  var T = {
    pts: 3.73, xy: 5.3, xr: 6.76, yu: 7.66,
    d1: 9.57, d1n: 10.16, py: 12.37, legs: 13.37, legs2: 14.4, hyp: 15.77, x2: 17.72, y2: 18.68, eq5: 19.86,
    d2: 21.57, d2n: 22.05, xm: 24.11, yy: 24.99, sq: 25.95, eq4: 26.93, eq4r: 28, out1: 29.5,
    sub: 29.8, sides: 30.71, lml: 32.76, rmr: 33.96, ycan: 36.14, rhs: 37.97, rhsR: 39.05,
    left: 40.67, lhs: 41.64, lhs2: 42.52, sqdef: 44.39, sqdef2: 45.52,
    split: 48.76, t1: 50.85, t2: 52.92, t3: 54.05, t23: 55.14, t4: 56.34, negneg: 57.6, t4r: 58.4,
    comb: 59.92, comb2: 61.62, comb3: 62.84,
    minus: 64.36, paren: 65.63, flip: 67.11, flip2: 68.42, xcan: 69.34, gone: 71.34, only: 72.58, out2: 75.55,
    e1: 75.86, add1: 77.85, e2: 78.88, div: 80.26, e3: 81.45, xline: [81.55, 82.25],
    back: 83.37, one: 85.69, y2b: 86.89, four: 88.8,
    sq4: 90.52, t1b: 92.7, t2b: 94.22, nn2: 95.57, t2r: 96.53, so: 97.59, yp: 98.73, yn: 99.33,
    def: 101.42, sqrt: 103.44, alsoK: 104.57, pos: 105.6, pm: 107.93,
    root: 110.16, r4: 112.39, r4v: 113.57, neg: 114.1, nr4: 116.2, nr4v: 117.61,
    up: 120.61, dn: 122.7, two: 123.67, src: 125.47, out3: 127.26,
    ang: 127.52, upper: 129.45, r1: 131.64, fn: 133.52, tell: 135.38, calc: 137.67, deg: 139.59,
    fore: 142.2, from: 143.79, dx: 145.68, dy: 146.65, down: 147.61, deg2: 149.01,
    el: 151.1, rel: 152.19, m90: 155.06, m634: 155.9, res3: 157.37,
    dn2: 160.27, sh2: 161.74, el2: 163.74, sign: 166.11, out4: 169.56,
    chk: 169.9, upA: 172.34, upB: 173.82, add1b: 175.57, add2b: 176.65, hit: 177.8, dnChk: 179.76
  };
  var END = 182.71;
  var UP = E.ik2(1, 0, true), DN = E.ik2(1, 0, false);
  var EG = [1.6, 1.5];   // 句 1–3：还不知道在哪的肘（示意）
  function c(hex, s) { return '\\textcolor{' + hex + '}{' + s + '}'; }
  function cancel(s) { return c(H.err, '\\cancel{' + c(H.ink, s) + '}'); }

  E.scenes[2] = {
    title: '手算：开根号，正负两个',
    dur: END,
    cues: [
      { at: 0, s: '肘的位置先不知道，用 2 个字母代替：往右 x 格，往上 y 格' },
      { at: 9.48, s: '① 离肩 2.24 格（勾股定理，第 5 集）：$x^2+y^2=5$' },
      { at: 21.48, s: '② 离手 2 格：$(x-1)^2+y^2=2^2=4$' },
      { at: 29.71, s: '等号两边一样多：左减左、右减右，$y^2$ 减掉了，右边 $5-4=1$' },
      { at: 40.58, s: '左边剩下 $x^2-(x-1)^2$；$(x-1)^2=(x-1)\\times(x-1)$' },
      { at: 48.67, s: '每一项都乘一遍：$x^2$、$-x$、$-x$、$+1$（负负得正）' },
      { at: 59.83, s: '合起来：$(x-1)^2=x^2-2x+1$' },
      { at: 64.27, s: '括号前面是减号，括号里每一项都变号：左边只剩 $2x-1$' },
      { at: 75.77, s: '$2x-1=1$：两边加 $1$，$2x=2$；两边除以 $2$，$x=1$' },
      { at: 83.28, s: '放回 ①：$1+y^2=5$，$y^2=4$' },
      { at: 90.43, s: '$2\\times 2=4$，$(-2)\\times(-2)=4$：$y=2$ 或 $y=-2$' },
      { at: 101.33, s: '从平方倒推回原来的数叫**开平方**，也叫**开根号**：正数开根号，一正、一负两个答案' },
      { at: 110.07, s: '正的那个写成根号：$\\sqrt4=2$；负的那个前面加负号：$-\\sqrt4=-2$' },
      { at: 119.12, s: '**肘上** (1, 2)、**肘下** (1, −2)：两组解来自**开根号**' },
      { at: 127.42, s: '大臂指向 (1, 2)：电脑里现成的函数 atan2，算出约 63.4°' },
      { at: 142.11, s: '小臂从 (1, 2) 到 (1, 0)：朝正下方，−90°' },
      { at: 151.01, s: '肘转的**角度** = −90 − 63.4 = −153.4°' },
      { at: 160.18, s: '**肘下**：肩 −63.4°、肘 +153.4°——肘的**角度**一负一正' },
      { at: 169.8, s: '验算：(1, 2) + (0, −2) = (1, 0)，正好是手的位置' }
    ],
    build: function (K) {
      var s = K.sceneSvg('肘的位置先不知道，用 x、y 代替；离肩 2.24 格，勾股定理 x² + y² = 5；离手 2 格，(x − 1)² + y² = 4；等号两边一样多，左减左、右减右，y² 减掉了，右边 5 − 4 = 1，左边剩下 x² − (x − 1)²；(x − 1)² = (x − 1) × (x − 1)，每一项都乘一遍：x²、−x、−x、+1（负负得正），合起来 x² − 2x + 1；括号前面是减号，括号里每一项都变号，左边只剩 2x − 1；2x − 1 = 1，两边加 1 得 2x = 2，两边除以 2 得 x = 1；放回第一个式子 1 + y² = 5，y² = 4；2 × 2 = 4，(−2) × (−2) = 4，y 是 2 或 −2，从平方倒推回原来的数叫开平方，正数开平方一正一负两个答案：肘上 (1, 2)、肘下 (1, −2)；关节角：大臂指向 (1, 2)，电脑里现成的函数 atan2 算出约 63.4°，小臂从 (1, 2) 到 (1, 0) 朝正下方 −90°，肘转 −90 − 63.4 = −153.4°；肘下反过来：肩 −63.4°，肘 +153.4°；验算 (1, 2) + (0, −2) = (1, 0)');

      var gridG = E.el('g', {}, s);
      var under = E.el('g', {}, gridG);
      var box = [A.OX - 110, A.OY - 214, 440, 420];
      var ghostG = E.el('g', {}, gridG), ghost = E.arm2(ghostG, A.OX, A.OY, A.S, 0, box);
      var armG = E.el('g', {}, gridG), arm = E.arm2(armG, A.OX, A.OY, A.S, 0, box), u = arm.u, V = arm.view;
      arm.pose(UP[0], UP[1]); ghost.pose(DN[0], DN[1]);
      var pl = E.plane(under, { ox: A.OX, oy: A.OY, u: u, x0: -1, x1: 4, y0: -3, y1: 3 });
      var P = pl.P, O = P(0, 0), TG = P(1, 0), EU = P(1, 2), ED = P(1, -2), G = P(EG[0], EG[1]), GX = P(EG[0], 0);
      var top = E.el('g', {}, gridG);
      var c1 = E.circ(top, O[0], O[1], E.L1 * u, E.C1, 2.5, '9 6'), c2 = E.circ(top, TG[0], TG[1], E.L2 * u, E.C2, 2.5, '9 6');
      var tgt = E.target(top, TG[0], TG[1], 13, E.TGT);
      function seg(color, w, dash) { var p = K.paint(E.el('path', { fill: 'none', 'stroke-width': w || 4, 'stroke-linecap': 'round' }, top), null, color); if (dash) p.setAttribute('stroke-dasharray', dash); return p; }
      function line(p, a, b) { p.setAttribute('d', 'M ' + a[0].toFixed(1) + ' ' + a[1].toFixed(1) + ' L ' + b[0].toFixed(1) + ' ' + b[1].toFixed(1)); return p; }
      // 句 1–3：两个直角三角形，共用竖边 y
      var gx = line(seg(E.X, 4, '8 5'), O, GX), gy = line(seg(E.Y, 4, '8 5'), GX, G), gxm = line(seg(E.X, 4, '8 5'), TG, GX);
      var hyp1 = line(seg(E.C1, 5), O, G), hyp2 = line(seg(E.C2, 5), G, TG);
      var corner = K.paint(E.el('path', { d: 'M ' + (GX[0] - 10) + ' ' + GX[1] + ' V ' + (GX[1] - 10) + ' H ' + GX[0], fill: 'none', 'stroke-width': 2 }, top), null, E.INK2);
      var lx = E.halo(E.txt(top, (O[0] + GX[0]) / 2 - 18, O[1] + 28, 'x', 24, E.X, 'middle', 900)), ly = E.halo(E.txt(top, GX[0] + 12, GX[1] + (G[1] - GX[1]) * 0.7 + 8, 'y', 24, E.Y, 'start', 900));
      [lx, ly].forEach(function (n) { n.style.fontFamily = 'KaTeX_Math, serif'; n.style.fontStyle = 'italic'; });
      var lxm = E.halo(E.txt(top, (TG[0] + GX[0]) / 2 + 2, O[1] + 50, 'x − 1', 19, E.X, 'middle', 900));
      var l224 = E.halo(E.txt(top, (O[0] + G[0]) / 2 - 12, (O[1] + G[1]) / 2 - 6, '要 = 2.24', 19, E.C1, 'end', 900));
      var l2 = E.halo(E.txt(top, GX[0] + 10, P(0, 0.28)[1], '要 = 2', 19, E.C2, 'start', 900));   // 竖边右下：斜边左边夹在两个三角形中间，放不下
      var pxy = K.paint(E.el('circle', { cx: G[0], cy: G[1], r: 7, 'stroke-width': 3 }, top), E.S1, E.HI);
      var pxyLab = E.halo(E.txt(top, G[0] + 14, G[1] - 14, '肘 (x, y)', 21, E.HI, 'start', 900));
      // 句 9：x = 1 的竖虚线
      var xl = K.paint(E.el('path', { d: 'M ' + TG[0] + ' ' + P(1, 3)[1] + ' V ' + P(1, -3)[1], fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '8 6' }, top), null, E.X);
      var xlLab = E.halo(E.txt(top, TG[0] + 10, P(1, 2.75)[1], 'x = 1', 21, E.X, 'start', 900));
      // 句 11–13：两个交点
      var ringU = E.el('g', {}, top), ringD = E.el('g', {}, top);
      K.paint(E.el('circle', { cx: EU[0], cy: EU[1], r: 13, 'stroke-width': 4, fill: 'none' }, ringU), null, E.UP);
      K.paint(E.el('circle', { cx: ED[0], cy: ED[1], r: 13, 'stroke-width': 4, fill: 'none' }, ringD), null, E.DOWN);
      var yU = E.halo(E.txt(top, EU[0] + 20, EU[1] + 8, 'y = 2', 21, E.UP, 'start', 900)), yD = E.halo(E.txt(top, ED[0] + 20, ED[1] + 8, 'y = −2', 21, E.DOWN, 'start', 900));
      var nU = E.halo(E.txt(top, EU[0] - 18, EU[1] - 16, '肘上 (1, 2)', 21, E.UP, 'end', 900)), nD = E.halo(E.txt(top, ED[0] - 18, ED[1] + 32, '肘下 (1, −2)', 21, E.DOWN, 'end', 900));
      // 句 14–17：转角
      var shArc = E.arcArrow(top, O[0], O[1], 0.62 * u, E.OMEGA, 3.5, { head: 11 });
      var shLab = E.halo(E.txt(top, 0, 0, '', 21, E.OMEGA, 'end', 900));
      var foreV = E.vec(top, E.TGT, 4, { head: 13 });
      var hRef = seg(E.INK2, 2.5, '5 5'), ext = seg(E.C1, 3, '7 6');
      var elArc = E.arcArrow(top, 0, 0, 0.62 * u, E.OMEGA, 3.5, { head: 11 });
      var elLab = E.halo(E.txt(top, 0, 0, '', 21, E.OMEGA, 'start', 900));
      // 句 18：头尾相接的两段
      var vU1 = E.vec(top, E.UP, 4.5, { head: 14 }), vU2 = E.vec(top, E.UP, 4.5, { head: 14 }), vD1 = E.vec(top, E.DOWN, 4.5, { head: 14 }), vD2 = E.vec(top, E.DOWN, 4.5, { head: 14 });

      /* ── 句 6–8：左边换成 2 × 2 的表（每一项都乘一遍） ── */
      var tabG = E.el('g', {}, s), TX = 70, TY = 70, CW = 96, CH = 66;
      K.paint(E.el('rect', { x: 12, y: 12, width: 396, height: 396, rx: 16, 'stroke-width': 2 }, tabG), E.tint(E.HI, 5, 'var(--demo-surface)'), E.BORDER);
      E.tex(tabG, 210, 48, '(x-1)\\times(x-1)', 28, E.INK, 'middle', { w: 380, h: 48 });
      E.txt(tabG, 210, 92, '每一项都乘一遍', 21, E.INK2, 'middle', 900);
      // 表头：上面一排是第二个 (x − 1) 的两项，左边一列是第一个的两项；格子中心 (TX + (1.5 + 列) × CW + 40, 160 + 行 × CH)
      var hdrTop = [E.tex(tabG, TX + 1.5 * CW + 40, 120, 'x', 28, E.INK, 'middle', { w: 80, h: 44 }), E.tex(tabG, TX + 2.5 * CW + 40, 120, '-1', 28, E.INK, 'middle', { w: 80, h: 44 })];
      var hdrLeft = [E.tex(tabG, TX + 0.5 * CW + 40, 170, 'x', 28, E.INK, 'middle', { w: 80, h: 44 }), E.tex(tabG, TX + 0.5 * CW + 40, 170 + CH, '-1', 28, E.INK, 'middle', { w: 80, h: 44 })];
      E.txt(tabG, TX + 0.5 * CW + 40, 122, '×', 26, E.MUT, 'middle', 900);
      var cellsT = [['x^2', 0, 0, T.t1], ['-x', 0, 1, T.t2], ['-x', 1, 0, T.t3], ['+1', 1, 1, T.t4r]].map(function (cc) {
        var cx = TX + (1.5 + cc[2]) * CW + 40, cy = 160 + cc[1] * CH;
        var g = E.el('g', {}, tabG);
        K.paint(E.el('rect', { x: cx - CW / 2 + 4, y: cy - CH / 2 + 4, width: CW - 8, height: CH - 8, rx: 10, 'stroke-width': 2 }, g), E.tint(cc[0] === '+1' ? E.GOOD : E.OMEGA, 14, 'var(--demo-surface)'), cc[0] === '+1' ? E.GOOD : E.OMEGA);
        E.tex(g, cx, cy + 10, cc[0], 28, E.INK, 'middle', { w: 90, h: 44 });
        return { g: g, at: cc[3], cx: cx, cy: cy };
      });
      var nnLab = E.tex(tabG, 210, 318, '(-1)\\times(-1)=+1\\;\\text{负负得正}', 22, E.GOOD, 'middle', { w: 380, h: 44 });
      var combLab = E.tex(tabG, 210, 374, '\\text{合起来：}x^2-x-x+1=x^2-2x+1', 22, E.INK, 'middle', { w: 390, h: 44 });

      /* ══ 右栏 ══ */
      var Rg = E.el('g', {}, s), X = RX + 10, SZ = 24;
      function tx(g, y, tex, size, w) { return E.tex(g, X, y, tex, size || SZ, E.INK, 'start', { w: w || 340, h: 46 }); }
      function lab(g, y, str, color, size) { return E.txt(g, X, y, str, size || 21, color || E.INK2, 'start', 900); }
      // 第一组：两个式子
      var g1 = E.el('g', {}, Rg);
      var p1 = lab(g1, 36, '肘的位置先不知道：(x, y)', E.HI, 21);
      var q1L = lab(g1, 84, '① 离肩 2.24 格（勾股定理，第 5 集）', E.C1, 19);
      var q1G = lab(g1, 118, '往右² + 往上² = 斜边²', E.INK2, 20);
      var q1a = tx(g1, 158, c(H.warn, '\\text{①}') + '\\;x^2'), q1b = tx(g1, 158, c(H.warn, '\\text{①}') + '\\;x^2+y^2'), q1 = tx(g1, 158, c(H.warn, '\\text{①}') + '\\;x^2+y^2=5');
      var q1n = lab(g1, 194, '（2.24 × 2.24 ≈ 5，第 1 幕）', E.MUT, 18);
      var q2L = lab(g1, 246, '② 离手 2 格：往右 x − 1，往上 y', E.C2, 19);
      var q2a = tx(g1, 288, c(H.good, '\\text{②}') + '\\;(x-1)^2+y^2'), q2b = tx(g1, 288, c(H.good, '\\text{②}') + '\\;(x-1)^2+y^2=2^2'), q2 = tx(g1, 288, c(H.good, '\\text{②}') + '\\;(x-1)^2+y^2=2^2=4');
      // 第二组：相减、拆括号
      var g2 = E.el('g', {}, Rg), S2 = 22;
      var r1 = tx(g2, 32, c(H.warn, '\\text{①}') + '\\;x^2+y^2=5', S2), r1c = tx(g2, 32, c(H.warn, '\\text{①}') + '\\;x^2+' + cancel('y^2') + '=5', S2);
      var r2 = tx(g2, 70, c(H.good, '\\text{②}') + '\\;(x-1)^2+y^2=4', S2), r2c = tx(g2, 70, c(H.good, '\\text{②}') + '\\;(x-1)^2+' + cancel('y^2') + '=4', S2);
      var sLab = lab(g2, 110, '等号两边一样多：', E.INK, 19), sLab2 = lab(g2, 110, '等号两边一样多：左减左、右减右', E.INK, 19);
      var rhs = tx(g2, 148, '\\text{右边：}5-4=1', S2);
      var lhs = tx(g2, 186, '\\text{左边：}x^2-(x-1)^2', S2);
      var sq1 = tx(g2, 232, '(x-1)^2=(x-1)\\times(x-1)', S2), sq2 = tx(g2, 232, '(x-1)^2=x^2-2x+1', S2);
      var m1 = tx(g2, 278, 'x^2-(x^2-2x+1)', S2), m1h = tx(g2, 278, 'x^2' + c(H.err, '\\,-\\,') + '(x^2-2x+1)', S2);
      var m2 = tx(g2, 320, '=x^2' + c(H.err, '-x^2+2x-1'), S2), m2c = tx(g2, 320, '=' + cancel('x^2') + c(H.err, '-') + cancel('x^2') + c(H.err, '+2x-1'), S2);
      var m3 = tx(g2, 366, '=' + c(H.hi, '2x-1'), 26);
      var mLab = lab(g2, 404, '括号前是减号：括号里每一项都变号', E.ERR, 18);
      // 第二组（续）：解 x
      var g2b = E.el('g', {}, Rg);
      var z0 = lab(g2b, 46, '左边 = 右边：', E.INK, 21);
      var z1 = tx(g2b, 90, '2x-1=1', 28);
      var z2L = lab(g2b, 140, '两边都加 1：', E.INK2, 20), z2 = tx(g2b, 178, '2x=2', 28);
      var z3L = lab(g2b, 228, '两边都除以 2：', E.INK2, 20), z3 = tx(g2b, 266, c(H.x, 'x=1'), 30);
      // 第三组：求 y
      var g3 = E.el('g', {}, Rg);
      var w1 = lab(g3, 36, 'x = 1，放回 ①：', E.INK, 21);
      var w2 = tx(g3, 76, '1^2+y^2=5'), w3a = tx(g3, 116, 'y^2=5-1'), w3 = tx(g3, 116, 'y^2=5-1=4');
      var w4 = lab(g3, 164, '平方等于 4 的数，有 2 个：', E.INK2, 20);
      var w5 = tx(g3, 202, '2\\times 2=4'), w6 = tx(g3, 240, '(-2)\\times(-2)=4\\;' + c(H.good, '\\text{负负得正}'), 22);
      var w7a = tx(g3, 290, 'y=' + c(H.hi, '2'), 28), w7 = tx(g3, 290, 'y=' + c(H.hi, '2') + '\\;\\text{或}\\;y=' + c(H.down, '-2'), 28);
      // 句 12、12b（2026-11-06 用户要求「开平方也叫根号，后续都用根号的方式说，也用根号的 LaTeX 表示」）：上面推 y² = 4 的几行先收起，
      // 腾出地方写开平方 / 开根号、√4 = 2、−√4 = −2（根号指正的那个：C 标准 7.12.7.5「nonnegative square root … Returns √x」）
      var w8 = lab(g3, 36, '从平方倒推回原来的数：开平方', E.WARN, 19), w8b = lab(g3, 64, '也叫开根号', E.WARN, 19);
      var w9 = lab(g3, 100, '正数开根号：一正、一负两个答案', E.WARN, 19);
      var w10 = tx(g3, 150, c(H.hi, '\\sqrt{4}=2') + '\\;\\text{正的那个}', 24), w11 = tx(g3, 200, c(H.down, '-\\sqrt{4}=-2') + '\\;\\text{负的那个}', 24);
      // 第四组：关节角
      var g4 = E.el('g', {}, Rg), S4 = 22;
      var k1 = lab(g4, 34, '肘上：大臂指向 (1, 2)', E.UP, 21);
      var k2 = lab(g4, 70, '往右 1、往上 2', E.INK2, 20);
      var k3 = lab(g4, 106, '电脑里现成的函数（第 11 集讲过函数）：', E.INK2, 17);
      var k3b = tx(g4, 142, '\\operatorname{atan2}(\\text{往上 }2,\\ \\text{往右 }1)\\approx' + c(H.acc, '63.4^\\circ'), 20);
      var k4 = lab(g4, 188, '小臂：从肘 (1, 2) 到手 (1, 0)', E.INK, 20);
      var k5 = lab(g4, 224, '往右 0、往下 2：朝正下方', E.INK2, 20), k6 = E.txt(g4, X + 268, 224, '−90°', 21, E.OMEGA, 'start', 900);
      var k7 = lab(g4, 272, '肘转的角 = 小臂方向 − 大臂方向', E.INK, 19);
      var k8a = tx(g4, 310, '-90', S4), k8b = tx(g4, 310, '-90-63.4', S4), k8 = tx(g4, 310, '-90-63.4=' + c(H.acc, '-153.4^\\circ'), S4);
      var k9a = tx(g4, 356, c(H.down, '\\text{肘下：}') + '\\text{肩 }' + c(H.acc, '-63.4^\\circ'), 20), k9 = tx(g4, 356, c(H.down, '\\text{肘下：}') + '\\text{肩 }' + c(H.acc, '-63.4^\\circ') + '\\text{，肘 }' + c(H.acc, '+153.4^\\circ'), 20);
      var k10 = lab(g4, 398, '肘的角度：一负、一正', E.HI, 20);
      // 第五组：验算
      var g5 = E.el('g', {}, Rg);
      var v1 = lab(g5, 60, '验算：先走大臂，再走小臂', E.INK, 22);
      var v2a = tx(g5, 108, c(H.hi, '(1,\\,2)'), 27), v2 = tx(g5, 108, c(H.hi, '(1,\\,2)') + '+' + c(H.hi, '(0,\\,-2)'), 27);
      var v3 = tx(g5, 156, '1+0=1', 25), v4 = tx(g5, 200, '2-2=0', 25);
      var v5 = tx(g5, 252, '=' + c(H.violet, '(1,\\,0)') + '\\;' + c(H.good, '\\checkmark') + '\\;\\text{手的位置}', 26);
      var v6 = tx(g5, 330, c(H.down, '\\text{肘下：}(1,\\,-2)+(0,\\,2)') + '=' + c(H.violet, '(1,\\,0)') + '\\;' + c(H.good, '\\checkmark'), 21);

      function fr(st, a) { return E.fade(st, a, 0.3); }
      function only(list, st, times) {   // 同一行的几个版本：到了哪个时刻就只显示哪一版
        var k = -1; times.forEach(function (tt, i) { if (st >= tt) k = i; });
        list.forEach(function (n, i) { E.op(n, i === k ? 1 : 0); });
      }

      return {
        el: s,
        draw: function (st, t) {
          var f = E.fade, b = E.bump;
          /* ── 左：句 6–8 网格让给 2 × 2 的表 ── */
          var tabOn = f(st, T.split - 0.2, 0.4) * (1 - f(st, T.out2, 0.4));
          E.op(gridG, 1 - 0.92 * tabOn); E.op(tabG, tabOn);
          cellsT.forEach(function (cc) { E.pop(cc.g, cc.cx, cc.cy, f(st, cc.at, 0.35), 0.6); });
          E.op(nnLab, fr(st, T.negneg)); E.op(combLab, fr(st, T.comb));
          hdrTop.concat(hdrLeft).forEach(function (n) { E.op(n, fr(st, T.split + 0.3)); });

          /* ── 两台 UR5：句 1–13 淡，句 14 起亮；句 17 肘下这台亮、肘上这台淡 ── */
          var armOn = f(st, T.ang, 0.4) * (1 - f(st, T.chk, 0.4) * 0.5);
          V.clearTint(); V.tint('all', null, 1, 0.2 + 0.8 * armOn * (1 - 0.6 * f(st, T.dn2, 0.4) * (1 - f(st, T.out4, 0.4))));
          ghost.view.clearTint(); ghost.view.tint('all', E.DOWN, 0.8, 0.25 + 0.35 * f(st, T.dn2, 0.4) * (1 - f(st, T.out4, 0.4)));
          E.op(ghostG, f(st, T.yn, 0.4));   // 算出 y = −2 之前不出现肘下这台，不提前给答案
          E.op(c1, 0.6 * (1 - 0.5 * f(st, T.ang, 0.4))); E.op(c2, 0.6 * (1 - 0.5 * f(st, T.ang, 0.4)));

          /* ── 句 1–3：示意的肘、两个直角三角形 ── */
          var on1 = 1 - f(st, T.out1, 0.4);
          E.op(pxy, fr(st, T.pts) * on1); E.op(pxyLab, fr(st, T.xy) * on1);
          E.drawOn(gx, f(st, T.xr, 0.5)); E.op(gx, (f(st, T.xr, 0.1) > 0 ? 1 : 0) * on1); E.op(lx, fr(st, T.xr) * on1);
          E.drawOn(gy, f(st, T.yu, 0.5)); E.op(gy, (f(st, T.yu, 0.1) > 0 ? 1 : 0) * on1); E.op(ly, fr(st, T.yu) * on1);
          ly.style.fontSize = (24 * (1 + 0.3 * b(st, T.yy, 0.8))) + 'px';
          E.drawOn(hyp1, f(st, T.d1, 0.5)); E.op(hyp1, (f(st, T.d1, 0.1) > 0 ? 1 : 0) * on1); E.op(l224, fr(st, T.d1n) * on1);
          E.op(corner, fr(st, T.py) * on1); E.glow(corner, E.HI, b(st, T.py, 0.9) > 0.05);
          E.drawOn(hyp2, f(st, T.d2, 0.5)); E.op(hyp2, (f(st, T.d2, 0.1) > 0 ? 1 : 0) * on1); E.op(l2, fr(st, T.d2n) * on1);
          E.drawOn(gxm, f(st, T.xm, 0.4)); E.op(gxm, (f(st, T.xm, 0.1) > 0 ? 1 : 0) * on1); E.op(lxm, fr(st, T.xm) * on1);
          /* ── 句 9：x = 1 ── */
          E.drawOn(xl, f(st, T.xline[0], T.xline[1] - T.xline[0])); E.op(xl, (f(st, T.xline[0], 0.1) > 0 ? 0.9 : 0) * (1 - f(st, T.out3, 0.4)));
          E.op(xlLab, fr(st, T.xline[1]) * (1 - f(st, T.out3, 0.4)));
          /* ── 句 11–13：两个交点 ── */
          var onR = 1 - f(st, T.out3, 0.4);
          E.pop(ringU, EU[0], EU[1], f(st, T.yp, 0.35), 0.4); E.pop(ringD, ED[0], ED[1], f(st, T.yn, 0.35), 0.4);
          E.op(ringU, f(st, T.yp, 0.35) * (1 - 0.6 * f(st, T.ang, 0.4))); E.op(ringD, f(st, T.yn, 0.35) * (1 - 0.6 * f(st, T.ang, 0.4)));
          E.op(yU, fr(st, T.yp) * (1 - f(st, T.up, 0.3))); E.op(yD, fr(st, T.yn) * (1 - f(st, T.dn, 0.3)));
          E.op(nU, fr(st, T.up) * onR); E.op(nD, fr(st, T.dn) * onR);
          [ringU, ringD].forEach(function (g) { E.glow(g, E.HI, b(st, T.src, 1.0) > 0.05); });

          /* ── 句 14–17：转角 ── */
          var dnOn = f(st, T.dn2, 0.4);
          var useA1 = dnOn > 0.5 ? DN[0] : UP[0], useA2 = dnOn > 0.5 ? DN[1] : UP[1], eP = dnOn > 0.5 ? ED : EU;
          var shp = f(st, T.upper, 0.6) * (1 - f(st, T.dn2 - 0.3, 0.3)) + f(st, T.sh2, 0.6);
          shArc.set(0, useA1 * Math.max(1e-3, Math.min(1, shp)));
          E.op(shArc.g, (shp > 0 ? 1 : 0) * (1 - f(st, T.out4, 0.3)));
          var shL = P(-0.42, dnOn > 0.5 ? -0.62 : 0.42);   // y 轴左边：弧的右边是小臂和目标
          shLab.textContent = dnOn > 0.5 ? '−63.4°' : '63.4°';
          shLab.setAttribute('x', shL[0].toFixed(1)); shLab.setAttribute('y', (shL[1] + 8).toFixed(1));
          E.op(shLab, (dnOn > 0.5 ? fr(st, T.sh2 + 0.4) : fr(st, T.deg) * (1 - f(st, T.dn2 - 0.3, 0.3))) * (1 - f(st, T.out4, 0.3)));
          // 小臂：从肘指向手（紫），方向在肘上量（水平参考线 → 小臂）
          var fp = f(st, T.fore, 0.6);
          foreV.set(EU[0] + 16, EU[1], EU[0] + 16 + (TG[0] - EU[0]) * fp, EU[1] + (TG[1] - EU[1]) * fp);   // 挪到小臂右边一点，不压在 UR5 上
          E.op(foreV.g, (fp > 0 ? 1 : 0) * (1 - f(st, T.el, 0.3)));
          line(hRef, [eP[0] - 0.2 * u, eP[1]], [eP[0] + 1.1 * u, eP[1]]);
          E.op(hRef, fr(st, T.down) * (1 - f(st, T.el, 0.3)));
          var extE = [eP[0] + 1.2 * u * Math.cos(useA1), eP[1] - 1.2 * u * Math.sin(useA1)];
          line(ext, eP, extE); E.drawOn(ext, f(st, T.el, 0.5)); E.op(ext, (f(st, T.el, 0.1) > 0 ? 1 : 0) * (1 - f(st, T.out4, 0.3)));
          elArc.g.setAttribute('transform', 'translate(' + eP[0].toFixed(1) + ' ' + eP[1].toFixed(1) + ')');
          var dirPhase = f(st, T.down, 0.6) * (1 - f(st, T.el, 0.3)), relPhase = f(st, T.rel, 0.8) * (1 - f(st, T.dn2 - 0.3, 0.3)) + f(st, T.el2, 0.8);
          if (relPhase > 0) elArc.set(useA1, useA1 + (useA2 - useA1) * Math.min(1, relPhase));
          else elArc.set(0, Math.min(-1e-3, useA2 * dirPhase));
          E.op(elArc.g, Math.max(dirPhase > 0 ? 1 : 0, relPhase > 0 ? 1 : 0) * (1 - f(st, T.out4, 0.3)));
          var mid = relPhase > 0 ? useA1 + (useA2 - useA1) / 2 : useA2 / 2, rr = 1.05 * u;
          elLab.setAttribute('x', (eP[0] + rr * Math.cos(mid) + 6).toFixed(1)); elLab.setAttribute('y', (eP[1] - rr * Math.sin(mid) + 8).toFixed(1));
          elLab.textContent = relPhase > 0 ? (dnOn > 0.5 ? '+153.4°' : '−153.4°') : '−90°';
          E.op(elLab, (relPhase > 0 ? (dnOn > 0.5 ? fr(st, T.el2 + 0.5) : fr(st, T.res3)) : fr(st, T.deg2) * (1 - f(st, T.el, 0.3))) * (1 - f(st, T.out4, 0.3)));

          /* ── 句 18：头尾相接 ── */
          var a1p = f(st, T.upA, 0.6), a2p = f(st, T.upB, 0.6);
          vU1.set(O[0], O[1], O[0] + (EU[0] - O[0]) * a1p, O[1] + (EU[1] - O[1]) * a1p); E.op(vU1.g, a1p > 0 ? 1 : 0);
          vU2.set(EU[0], EU[1], EU[0] + (TG[0] - EU[0]) * a2p, EU[1] + (TG[1] - EU[1]) * a2p); E.op(vU2.g, a2p > 0 ? 1 : 0);
          var d1p = f(st, T.dnChk, 0.5), d2p = f(st, T.dnChk + 0.5, 0.5);
          vD1.set(O[0], O[1], O[0] + (ED[0] - O[0]) * d1p, O[1] + (ED[1] - O[1]) * d1p); E.op(vD1.g, d1p > 0 ? 1 : 0);
          vD2.set(ED[0], ED[1], ED[0] + (TG[0] - ED[0]) * d2p, ED[1] + (TG[1] - ED[1]) * d2p); E.op(vD2.g, d2p > 0 ? 1 : 0);
          tgt.setAttribute('transform', 'translate(' + TG[0] + ' ' + TG[1] + ') scale(' + (1 + 0.5 * Math.max(b(st, T.hit, 1.0), b(st, T.dnChk + 1.0, 1.0))) + ') translate(' + (-TG[0]) + ' ' + (-TG[1]) + ')');

          /* ── 右栏 ── */
          E.op(g1, 1 - f(st, T.out1, 0.4));
          E.op(p1, fr(st, T.pts));
          E.op(q1L, fr(st, T.d1)); E.op(q1G, fr(st, T.legs)); only([q1a, q1b, q1], st, [T.x2, T.y2, T.eq5]); E.op(q1n, fr(st, T.eq5 + 0.4));
          E.op(q2L, fr(st, T.d2)); only([q2a, q2b, q2], st, [T.sq, T.eq4, T.eq4r]);
          E.op(g2, f(st, T.sub - 0.2, 0.3) * (1 - f(st, T.out2, 0.4)));
          only([r1, r1c], st, [0, T.ycan]); only([r2, r2c], st, [0, T.ycan]);
          [r1, r1c, r2, r2c].forEach(function (n) { E.glow(n, E.HI, b(st, T.sub, 0.9) > 0.05); });
          only([sLab, sLab2], st, [T.sides, T.lml]);
          E.op(rhs, fr(st, T.rhs)); E.op(lhs, fr(st, T.lhs));
          only([sq1, sq2], st, [T.sqdef2, T.comb]);
          only([m1, m1h], st, [T.minus, T.paren]); only([m2, m2c], st, [T.flip2, T.xcan]); E.op(m3, fr(st, T.only)); E.op(mLab, fr(st, T.flip));
          E.op(g2b, f(st, T.e1 - 0.1, 0.3) * (1 - f(st, T.back - 0.3, 0.3)));
          E.op(z0, fr(st, T.e1)); E.op(z1, fr(st, T.e1)); E.op(z2L, fr(st, T.add1)); E.op(z2, fr(st, T.e2)); E.op(z3L, fr(st, T.div)); E.op(z3, fr(st, T.e3));
          E.op(g3, f(st, T.back - 0.2, 0.3) * (1 - f(st, T.out3, 0.4)));
          var upOut = 1 - f(st, T.def - 0.1, 0.3);   // 讲开平方时，上面推 y² = 4 的几行收起
          E.op(w1, fr(st, T.back) * upOut); E.op(w2, fr(st, T.one) * upOut); only([w3a, w3], st, [T.y2b, T.four]); E.op(w3a, +w3a.style.opacity * upOut); E.op(w3, +w3.style.opacity * upOut);
          E.op(w4, fr(st, T.sq4) * upOut); E.op(w5, fr(st, T.t1b) * upOut); E.op(w6, fr(st, T.t2b) * upOut);
          only([w7a, w7], st, [T.yp, T.yn]); E.op(w8, fr(st, T.sqrt)); E.op(w8b, fr(st, T.alsoK)); E.op(w9, fr(st, T.pos));
          E.op(w10, fr(st, T.r4)); E.op(w11, fr(st, T.nr4));
          E.op(g4, f(st, T.ang - 0.2, 0.3) * (1 - f(st, T.out4, 0.4)));
          E.op(k1, fr(st, T.upper)); E.op(k2, fr(st, T.r1)); E.op(k3, fr(st, T.fn)); E.op(k3b, fr(st, T.calc));
          E.op(k4, fr(st, T.fore)); E.op(k5, fr(st, T.dx)); E.op(k6, fr(st, T.deg2));
          E.op(k7, fr(st, T.el)); only([k8a, k8b, k8], st, [T.m90, T.m634, T.res3]);
          only([k9a, k9], st, [T.sh2, T.el2]); E.op(k10, fr(st, T.sign));
          E.op(g5, f(st, T.chk - 0.1, 0.3));
          E.op(v1, fr(st, T.chk)); only([v2a, v2], st, [T.upA, T.upB]); E.op(v3, fr(st, T.add1b)); E.op(v4, fr(st, T.add2b));
          E.op(v5, fr(st, T.hit)); E.op(v6, fr(st, T.dnChk));
        }
      };
    }
  };
})();
