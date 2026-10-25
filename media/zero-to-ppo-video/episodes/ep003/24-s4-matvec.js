// 003 第 4 幕：矩阵乘向量，手算一遍。左边网格 + UR5 侧视（肩关节 = 原点，手在格点 (3, 2)），右边一行行写算式。
// 句 1「手」手上一圈光、蓝箭头画到 (3, 2) + [3;2]；「肩关节」肩关节鼓一圈光；「转九十度」肩关节一支逆时针弯箭头「90°」，
//   「转到哪」手要走的虚线圆弧画出来，终点 (−2, 3) 一个「?」（先不转）。
// 句 2「整条胳膊跟着转」大臂到手腕上紫色闪一下；「乘上」右上出算式 M·[3;2] = ?（两列绿 / 琥珀，3 绿、2 琥珀）。
// 句 3「第一列取三份」第一列和 3 亮、别的变暗，「第二列取两份」第二列和 2 亮，「加起来」第二行 3[0;1] + 2[−1;0]。
// 句 4 网格上头尾相接：绿 (0,0)→(0,3)、琥珀 (0,3)→(−2,3)、蓝 (0,0)→(−2,3)；右边第三行依次写 [0;3]、+[−2;0]，第四行 = [−2;3]，第一行的「?」换成 [−2;3]。
// 句 5「看图」机械臂真的转 90°，手的位置箭头跟着转，落在蓝箭头尖上；「往上三格」打 ✓。
// 句 6「按横排算」第二到四行淡掉，第一行的矩阵按横排高亮：0×3 + (−1)×2 = −2、1×3 + 0×2 = 3，「答案一样」✓。
// 句 7「三维」右边换成 3×3 矩阵 × [x;y;z] = [x′;y′;z′]，「位置进去」「位置出来」各一支箭头。
// 时刻取自配音的逐词时间（分镜时间 ≈ 念到的时刻），见 ../ep003.py。
(function () {
  var E = window.EP003, K = E.K;
  var OX = 185, OY = 262, ARM_S = 1.5, A1 = 30;           // 肩关节在画面里的位置、UR5 缩放、大臂离竖直的角
  var RX = 445;                                            // 右边算式的左缘
  var ROW_Y = [84, 178, 268, 356];
  var HAND = 1.97, VEC = 2.4, SHOULDER = 4.93, ARC = 5.48, PATH = 6.63, ASK = 7.15;
  var TINT = [9.6, 11.3], ROW_A = 12.55, COL1 = [17.1, 18.3], COL2 = [18.35, 19.29], ROW_B = 19.29;
  var GREEN = 22.26, AMBER = 24.91, BLUE = 26.4, ROW_D = 26.89, TURN = 29.07, OK = 32.08;
  var BYROW = 33.9, ROW1 = 36.0, LINE1 = 37.99, ROW2 = 38.5, LINE2 = 38.81, SAME = 39.42;
  var THREE = 41.22, BOX3 = 42.56, LBL3 = 43.0, IN = 44.4, OUT = 46.2;
  var H = E.HEX, DIM = '#5c5c64';

  function bell(st, a, b) { var u = K.seg(st, a, b); return u > 0 && u < 1 ? Math.sin(Math.PI * u) : 0; }
  function n(x) { return String(x).replace(/^-/, '-'); }
  function c(x, hex) { return '\\textcolor{' + hex + '}{' + n(x) + '}'; }
  // 第一行算式：矩阵四个数、向量两个数各自的颜色，等号后面是「?」或结果向量
  function rowA(mc, vc, res) {
    return '\\begin{bmatrix}' + c(0, mc[0]) + '&' + c(-1, mc[1]) + '\\\\' + c(1, mc[2]) + '&' + c(0, mc[3]) + '\\end{bmatrix}' +
      '\\begin{bmatrix}' + c(3, vc[0]) + '\\\\' + c(2, vc[1]) + '\\end{bmatrix}=' + res;
  }

  E.scenes[3] = {
    title: '矩阵乘向量：手算一遍',
    dur: 48.03,
    cues: [
      { at: 0, s: '肩关节转 90°，手转到哪？' },
      { at: 8.42, s: '整条胳膊跟着转：手的位置乘上这个**矩阵**' },
      { at: 15.09, s: '第一列取 3 份，第二列取 2 份，加起来' },
      { at: 20.87, s: '$3\\begin{bmatrix}0\\\\1\\end{bmatrix}+2\\begin{bmatrix}-1\\\\0\\end{bmatrix}=\\begin{bmatrix}-2\\\\3\\end{bmatrix}$' },
      { at: 28.98, s: '手果然转到了 (−2, 3)' },
      { at: 33.78, s: '按横排算：对应相乘再相加，答案一样' },
      { at: 41.12, s: '三维：3×3 矩阵，位置进、新位置出' }
    ],
    build: function (K) {
      var s = K.sceneSvg('矩阵乘向量手算一遍：机械臂的手在往右 3、往上 2，肩关节转 90 度，手的位置乘上转 90 度的矩阵：第一列取 3 份、第二列取 2 份加起来，得到往左 2、往上 3，机械臂真的转过去手正好落在那里；按横排对应相乘再相加答案一样；三维就是 3×3 的矩阵，一个位置进去、变换后的位置出来');

      /* ── 左：网格（底层先占位）+ UR5，肩关节 = 原点 ── */
      var left = E.el('g', {}, s);
      var under = E.el('g', {}, left);
      var arm = E.armAt(left, OX, OY, ARM_S, 236);
      var u = arm.aim(3, 2, A1);
      var pl = E.plane(under, { ox: OX, oy: OY, u: u, x0: -3, x1: 4, y0: -1, y1: 4 });
      var P = pl.P;
      // 刻度自己写：y 轴上的「3」会被绿箭头尖、琥珀箭头压住，画箭头时把它淡掉
      var tickY3 = null;
      [-3, -2, -1, 1, 2, 3, 4].forEach(function (x) { E.txt(under, P(x, 0)[0], OY + 24, String(x).replace('-', '−'), 18, E.MUT); });
      [-1, 1, 2, 3, 4].forEach(function (y) {
        var n = E.txt(under, OX - 12, P(0, y)[1] + 6, String(y).replace('-', '−'), 18, E.MUT, 'end');
        if (y === 3) tickY3 = n;
      });
      var R = Math.hypot(3, 2) * u, a0 = Math.atan2(2, 3);
      var LINKS = ['upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link'];

      // 手要走的虚线圆弧（逆时针 90°）+ 终点「?」；肩关节的小弯箭头「90°」
      function arcPath(r, from, to) {
        var p0 = [OX + r * Math.cos(from), OY - r * Math.sin(from)], p1 = [OX + r * Math.cos(to), OY - r * Math.sin(to)];
        return 'M ' + p0[0].toFixed(1) + ' ' + p0[1].toFixed(1) + ' A ' + r.toFixed(1) + ' ' + r.toFixed(1) + ' 0 0 0 ' + p1[0].toFixed(1) + ' ' + p1[1].toFixed(1);
      }
      function endDeg(th) { return Math.atan2(-Math.cos(th), -Math.sin(th)) * 180 / Math.PI; }   // 逆时针走到 th 时的切线方向（画面角度）
      var path = E.curveArrow(left, arcPath(R, a0 + 0.06, a0 + Math.PI / 2 - 0.17), E.HI, 2.5, endDeg(a0 + Math.PI / 2 - 0.17), { dash: '7 6', head: 12 });
      var small = E.curveArrow(left, arcPath(32, a0, a0 + Math.PI / 2), E.HI, 3, endDeg(a0 + Math.PI / 2), { head: 11 });
      var deg = E.txt(left, OX - 52, OY - 30, '90°', 22, E.HI, 'middle', 900);
      var tgt = P(-2, 3);
      var ask = E.txt(left, tgt[0], tgt[1] + 14, '?', 44, E.HI, 'middle', 900);

      // 肩关节、手的光圈
      var pulse = K.paint(E.el('circle', { cx: OX, cy: OY, r: 10, fill: 'none', 'stroke-width': 3 }, left), null, E.HI);
      var handRing = K.paint(E.el('circle', { r: 14, fill: 'none', 'stroke-width': 3 }, left), null, E.HI);

      // 箭头：手的位置（跟着手转）+ 原位虚影；网格上头尾相接的绿、琥珀、蓝
      var ghost = E.vec(left, E.ACC, 3, { head: 13, dash: '6 6' });
      var green = E.vec(left, E.GOOD, 5, { head: 16 }), amber = E.vec(left, E.WARN, 5, { head: 16 }), blue = E.vec(left, E.ACC, 5, { head: 16 });
      var pos = E.vec(left, E.ACC, 5, { head: 16 });
      var hp = P(3, 2);
      var posLbl = E.tex(left, hp[0] + 44, hp[1] + 6, E.col([3, 2], [H.good, H.warn]), 26, E.ACC);
      var okG = E.el('g', {}, left);
      E.check(okG, tgt[0] + 22, tgt[1] - 44, 26, E.GOOD, 4);

      /* ── 右：四行算式（左缘对齐 RX；同一行前缀不变，后面换内容不挪位置） ── */
      var right = E.el('g', {}, s);
      var GG = [H.good, H.warn, H.good, H.warn];                 // 矩阵四个数按列上色
      var rA = E.tex(right, RX, ROW_Y[0], rowA(GG, [H.good, H.warn], '\\,?'), 30, E.INK, 'start', { w: 360, h: 90 });
      var rB = E.tex(right, RX, ROW_Y[1], '\\phantom{=}' + c(3, H.good) + E.col([0, 1], H.good) + '+' + c(2, H.warn) + E.col([-1, 0], H.warn), 30, E.INK, 'start', { w: 360, h: 90 });
      var C1 = '=\\phantom{3}' + E.col([0, 3], H.good), C2 = '+\\phantom{2}' + E.col([-2, 0], H.warn);
      var rC = E.tex(right, RX, ROW_Y[2], C1 + '\\phantom{' + C2 + '}', 30, E.INK, 'start', { w: 360, h: 90 });
      var rD = E.tex(right, RX, ROW_Y[3], '=\\phantom{3}' + E.col([-2, 3], H.acc), 30, E.INK, 'start', { w: 360, h: 90 });
      // 按横排算的两行
      var L1 = E.tex(right, RX, 186, c(0, H.violet) + '\\times' + c(3, H.good) + '+(' + c(-1, H.violet) + ')\\times' + c(2, H.warn) + '=' + c(-2, H.acc), 30, E.INK, 'start', { w: 360 });
      var L2 = E.tex(right, RX, 262, c(1, H.violet) + '\\times' + c(3, H.good) + '+' + c(0, H.violet) + '\\times' + c(2, H.warn) + '=' + c(3, H.acc), 30, E.INK, 'start', { w: 360 });
      var sameG = E.el('g', {}, right);
      E.check(sameG, RX + 2, 318, 24, E.GOOD, 4);
      E.txt(sameG, RX + 36, 338, '答案一样', 22, E.GOOD, 'start', 900);

      /* ── 三维示意：3×3 矩阵 × [x;y;z] = [x′;y′;z′]，位置进去、新位置出来 ── */
      var three = E.el('g', {}, s);
      var X3 = [505, 615, 678, 738], Y3 = 196;
      var m3 = E.tex(three, X3[0], Y3, '\\begin{bmatrix}\\ast&\\ast&\\ast\\\\\\ast&\\ast&\\ast\\\\\\ast&\\ast&\\ast\\end{bmatrix}', 30, E.VIOLET, 'middle', { w: 200, h: 140 });
      var v3 = E.tex(three, X3[1], Y3, E.col(['x', 'y', 'z'], H.acc), 30, E.ACC, 'middle', { w: 80, h: 140 });
      var eq3 = E.tex(three, X3[2], Y3, '=', 30, E.INK, 'middle', { w: 40 });
      var o3 = E.tex(three, X3[3], Y3, E.col(["x'", "y'", "z'"], H.acc), 30, E.ACC, 'middle', { w: 80, h: 140 });
      var lbl3 = E.txt(three, X3[0], 112, '3 × 3', 24, E.VIOLET, 'middle', 900);
      var inA = E.arrow(three, [[X3[1], 330], [X3[1], 262]], E.ACC, 3, { head: 12 });
      var inT = E.txt(three, X3[1], 360, '位置进去', 20, E.ACC, 'middle', 700);
      var outA = E.arrow(three, [[X3[3], 262], [X3[3], 330]], E.ACC, 3, { head: 12 });
      var outT = E.txt(three, X3[3], 360, '新位置出来', 20, E.ACC, 'middle', 700);

      return {
        el: s,
        draw: function (st) {
          var fade = E.fade;
          E.op(left, (fade(st, 0.1, 0.5) * (1 - 0.55 * fade(st, THREE, 0.5))).toFixed(3));

          /* 机械臂：句 2 整条胳膊上紫色闪一下；句 5 真的转 90° */
          var tw = E.win(st, TINT[0], TINT[1], 0.35);
          arm.view.clearTint();
          if (tw > 0) arm.view.tint(LINKS, E.VIOLET, tw);
          var turn = Math.PI / 2 * fade(st, TURN, 1.3);
          var tp = arm.turn(turn);

          /* 句 1：手上一圈光、位置箭头画到手上、[3;2]；肩关节鼓一圈光 */
          var hr = Math.max(bell(st, HAND, HAND + 1.2), bell(st, OK - 0.3, OK + 1.0));
          handRing.setAttribute('cx', tp[0].toFixed(1)); handRing.setAttribute('cy', tp[1].toFixed(1));
          handRing.setAttribute('r', (12 + 6 * hr).toFixed(2));
          E.op(handRing, hr.toFixed(3));
          var pv = fade(st, VEC, 0.7);
          pos.set(OX, OY, OX + (tp[0] - OX) * pv, OY + (tp[1] - OY) * pv);
          E.op(pos.g, pv > 0 ? 1 : 0);
          ghost.set(OX, OY, hp[0], hp[1]);
          E.op(ghost.g, (0.5 * fade(st, TURN, 0.4)).toFixed(3));
          E.op(posLbl, (fade(st, 3.5, 0.3) * (1 - 0.5 * fade(st, TURN, 0.4))).toFixed(3));
          var pu = K.seg(st, SHOULDER, SHOULDER + 0.8);
          pulse.setAttribute('r', (10 + 26 * pu).toFixed(2));
          E.op(pulse, pu > 0 && pu < 1 ? (1 - pu).toFixed(3) : 0);

          /* 「转九十度」小弯箭头 + 90°；「转到哪」虚线圆弧 + 「?」；句 5 转完后退场 */
          var ps = fade(st, ARC, 0.5), gone = 1 - fade(st, TURN + 1.6, 0.5);
          E.drawOn(small.line, ps); E.op(small.head, K.seg(ps, 0.85, 1)); E.op(small.g, gone);
          E.op(deg, fade(st, 5.66, 0.3) * gone);
          var pp = fade(st, PATH, 0.6);
          E.drawOn(path.line, pp); E.op(path.head, K.seg(pp, 0.85, 1)); E.op(path.g, gone);
          E.pop(ask, tgt[0], tgt[1], fade(st, ASK, 0.3) * (1 - fade(st, AMBER, 0.3)), 0.6);
          E.op(tickY3, 1 - fade(st, GREEN + 0.3, 0.3));

          /* 句 4：头尾相接的绿、琥珀、蓝 */
          var g0 = P(0, 0), g3 = P(0, 3), gT = P(-2, 3);
          var pg = fade(st, GREEN, 0.6), pa = fade(st, AMBER, 0.6), pb = fade(st, BLUE, 0.8);
          green.set(g0[0], g0[1], g0[0], g0[1] + (g3[1] - g0[1]) * pg); E.op(green.g, pg > 0 ? 1 : 0);
          amber.set(g3[0], g3[1], g3[0] + (gT[0] - g3[0]) * pa, g3[1]); E.op(amber.g, pa > 0 ? 1 : 0);
          blue.set(g0[0], g0[1], g0[0] + (gT[0] - g0[0]) * pb, g0[1] + (gT[1] - g0[1]) * pb); E.op(blue.g, pb > 0 ? 1 : 0);
          E.pop(okG, tgt[0] + 35, tgt[1] - 31, fade(st, OK, 0.3), 0.5);

          /* 右边第一行：12.55「乘上」出现；句 3 先亮第一列和 3、再亮第二列和 2；27.59 「?」换成结果；句 6 按横排亮 */
          var mc = GG.slice(), vc = [H.good, H.warn], res = '\\,?';
          if (st >= COL1[0] && st < COL1[1]) { mc = [H.good, DIM, H.good, DIM]; vc = [H.good, DIM]; }
          else if (st >= COL2[0] && st < COL2[1]) { mc = [DIM, H.warn, DIM, H.warn]; vc = [DIM, H.warn]; }
          if (st >= 27.4) res = E.col([-2, 3], H.acc);
          if (st >= ROW1 && st < ROW2) { mc = [H.violet, H.violet, DIM, DIM]; res = E.col([-2, 3], [H.acc, DIM]); }
          else if (st >= ROW2 && st < SAME + 0.8) { mc = [DIM, DIM, H.violet, H.violet]; res = E.col([-2, 3], [DIM, H.acc]); }
          rA.setTex(rowA(mc, vc, res));
          E.op(rA, fade(st, ROW_A, 0.4));
          var byRow = 1 - fade(st, BYROW, 0.5);
          E.op(rB, fade(st, ROW_B, 0.4) * byRow);
          rC.setTex(st >= 25.1 ? C1 + C2 : C1 + '\\phantom{' + C2 + '}');
          E.op(rC, fade(st, 22.54, 0.3) * byRow);
          E.op(rD, fade(st, ROW_D, 0.3) * byRow);
          E.op(L1, fade(st, LINE1, 0.4));
          E.op(L2, fade(st, LINE2, 0.4));
          E.pop(sameG, RX + 70, 330, fade(st, SAME, 0.3), 0.6);
          E.op(right, (1 - fade(st, THREE, 0.4)).toFixed(3));

          /* 句 7：三维示意 */
          E.op(three, fade(st, THREE + 0.2, 0.4));
          E.op(m3, fade(st, BOX3, 0.4)); E.op(v3, fade(st, BOX3 + 0.2, 0.4)); E.op(eq3, fade(st, BOX3 + 0.3, 0.4)); E.op(o3, fade(st, BOX3 + 0.4, 0.4));
          E.pop(lbl3, X3[0], 104, fade(st, LBL3, 0.3), 0.7);
          var pi = fade(st, IN, 0.6), po = fade(st, OUT, 0.6);
          E.drawOn(inA.line, pi); E.op(inA.head, K.seg(pi, 0.85, 1)); E.op(inT, fade(st, 44.72, 0.3));
          E.drawOn(outA.line, po); E.op(outA.head, K.seg(po, 0.85, 1)); E.op(outT, fade(st, 46.55, 0.3));
        }
      };
    }
  };
})();
