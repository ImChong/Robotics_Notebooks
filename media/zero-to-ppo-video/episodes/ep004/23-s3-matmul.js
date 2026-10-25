// 004 第 3 幕：矩阵乘向量用 @，不用 *。左边交互窗口（上一幕的 p、R 还在，淡一点），右边说明区。
// ① 「R 和 p」先打出 R、p，中间空一格；「中间」空格上框一圈亮黄，「艾特」@ 落进去。
// ② 「回车」出 array([-2,  3])（蓝）；「和上一集」右上出「上一集手算」卡（003 第 4 幕的算式），「完全一样」输出旁打 ✓。
// ③ 右下一直是网格 + UR5 侧视（肩关节 = 原点，手在 (3, 2)，蓝箭头指到手）；「画到格子上」手要走的虚线圆弧画出来，
//    「转九十度」机械臂绕肩关节转 90°，手的位置箭头跟着转，(3, 2) 留虚影；「负二、三」落点标 (−2, 3)，「三」打 ✓。
// ④ 「用星号来乘」打 R * p；「不是矩阵乘法」那一句旁打红 ×，右边换成说明：上方「* 不是矩阵乘法」，
//    「对应位置」R 的两列头上出 ×3、×2（p 的两个数，蓝）；「结果」出二乘二的输出，右边 R = 结果矩阵 [[0, −2], [3, 0]]；
//    「第一列乘了三」「第二列乘了二」结果的两列依次框（代码里同一列也框）；「三份第一列、两份第二列」下面写 3[0;1]、2[−1;0]；
//    「没有加起来」两者中间一个红色虚线圈「+ ?」、下面「没加起来」。
// ⑤ 「矩阵乘法用艾特」R @ p 那一句垫蓝底、旁边 ✓；「星号只做对应位置相乘」R * p 那一句垫红底；右下两个小牌总结。
// 时刻取自配音的逐词时间（分镜时间 ≈ 念到的时刻），见 ../ep004.py。
(function () {
  var E = window.EP004, K = E.K, H = E.HEX;
  var RX = 466, RW = 318, RC = RX + RW / 2;
  var KEYS = [3.01, 3.2, 4.42, 3.3, 3.36], GAP_AT = 3.51, OUT = 6.14, CARD = 8.32, SAME = 9.81;
  var PATH = 11.61, TURN = [13.53, 14.6], GHOST = 15.5, LAND = 16.73, OK = 17.36, PART_A_OUT = 21.6;
  var STAR = 20.59, NOT = 22.12, POS = 24.06, RES = 26.59, COL1 = [28.72, 29.8], COL2 = [29.87, 30.8], SPLIT = 30.85, NOSUM = 32.99;
  var SUM_UP = 35.57, AT_OK = 37.03, STAR_SUM = 37.87;
  var OX = 592, OY = 370, ARM_S = 1.26, A1 = 30;            // 网格原点（= UR5 肩关节）、UR5 缩放、大臂离竖直的角
  var LINKS = ['upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link'];

  function bell(st, a, b) { var u = K.seg(st, a, b); return u > 0 && u < 1 ? Math.sin(Math.PI * u) : 0; }

  E.scenes[2] = {
    title: '矩阵乘向量：用 @，不用 *',
    dur: 40.73,
    cues: [
      { at: 0, s: '矩阵乘向量：`R @ p`' },
      { at: 6.05, s: '结果 (−2, 3)，和手算的完全一样' },
      { at: 11.52, s: '画到格子上：手从 (3, 2) 转到 (−2, 3)' },
      { at: 18.72, s: '`*` 不是矩阵乘法，是**对应位置**的数相乘' },
      { at: 26.5, s: '`R * p`：三份第一列、两份第二列，没加起来' },
      { at: 35.14, s: '矩阵乘法用 `@`，`*` 只做对应位置相乘' }
    ],
    build: function (K) {
      var s = K.sceneSvg('在 R 和 p 中间写 @ 就是矩阵乘向量，结果 (−2, 3) 和上一集手算的一样，画到格子上机械臂转 90 度、手正好从 (3, 2) 到 (−2, 3)；用星号 * 是对应位置的数相乘，结果是一个 2×2 数组：第一列乘了 3、第二列乘了 2，没有加起来');

      /* ── 左：交互窗口 ── */
      var py = E.code(s, 10, 14, 440, 392, { title: 'Python', size: 25 });
      // 上一幕已经跑过的三句：当成已有的内容（out），光标不停在它们后面
      py.line('>>> p = np.array([{a:3}, {a:2}])', 0, { out: true, alpha: 0.55 });
      py.line('>>> R = np.array([[{g:0}, {w:-1}],', 0, { out: true, alpha: 0.55 });
      py.line('...               [{g:1},  {w:0}]])', 0, { out: true, alpha: 0.55 });
      var L_AT = py.line('>>> R {h:@} p', KEYS[0], { keys: KEYS });
      var L_OUT = py.line('array([{a:-2},  {a:3}])', OUT, { out: true });
      var L_STAR = py.line('>>> R {b:*} p', STAR, { dur: 0.6 });
      var L_S1 = py.line('array([[ {g:0}, {w:-2}],', RES, { out: true });
      var L_S2 = py.line('       [ {g:3},  {w:0}]])', RES, { out: true });
      var okAt = E.el('g', {}, s), okSame = E.el('g', {}, s), badStar = E.el('g', {}, s);
      E.check(okSame, 0, 0, 22, E.GOOD, 4);
      E.txt(okSame, 32, 16, '和手算的一样', 20, E.GOOD, 'start', 900);
      E.check(okAt, 0, 0, 22, E.GOOD, 4);
      E.cross(badStar, 0, 0, 10, E.BAD, 4);

      /* ── 右 ②：上一集手算 ── */
      var card = E.el('g', {}, s);
      K.paint(E.el('rect', { x: RX + 4, y: 18, width: RW - 8, height: 128, rx: 14, 'stroke-width': 2 }, card), E.S2, E.BORDER);
      E.txt(card, RX + 20, 46, '上一集手算', 20, E.INK2, 'start', 700);
      E.tex(card, RC, 106, E.mat2([0, -1, 1, 0], [H.good, H.warn]) + E.col([3, 2], H.acc) + '=' + E.col([-2, 3], H.acc), 26, E.INK, 'middle', { w: 300, h: 90 });

      /* ── 右 ③：网格 + UR5（肩关节 = 原点） ── */
      var partA = E.el('g', {}, s);
      var under = E.el('g', {}, partA);
      var arm = E.armAt(partA, OX, OY, ARM_S, 200);
      var u = arm.aim(3, 2, A1);
      var pl = E.plane(under, { ox: OX, oy: OY, u: u, x0: -3, x1: 4, y0: -1, y1: 4 });
      var P = pl.P, hp = P(3, 2), tgt = P(-2, 3);
      var R0 = Math.hypot(3, 2) * u, a0 = Math.atan2(2, 3);
      function arcPath(r, from, to) {
        var p0 = [OX + r * Math.cos(from), OY - r * Math.sin(from)], p1 = [OX + r * Math.cos(to), OY - r * Math.sin(to)];
        return 'M ' + p0[0].toFixed(1) + ' ' + p0[1].toFixed(1) + ' A ' + r.toFixed(1) + ' ' + r.toFixed(1) + ' 0 0 0 ' + p1[0].toFixed(1) + ' ' + p1[1].toFixed(1);
      }
      var th1 = a0 + Math.PI / 2 - 0.16;
      var path = E.curveArrow(partA, arcPath(R0, a0 + 0.06, th1), E.HI, 2.5, Math.atan2(-Math.cos(th1), -Math.sin(th1)) * 180 / Math.PI, { dash: '7 6', head: 11 });
      var ghost = E.vec(partA, E.ACC, 3, { head: 12, dash: '6 6' });
      var pos = E.vec(partA, E.ACC, 5, { head: 15 });
      var landLbl = E.txt(partA, tgt[0] - 8, tgt[1] - 22, '(−2, 3)', 22, E.ACC, 'middle', 900);
      var okLand = E.el('g', {}, partA);
      E.check(okLand, tgt[0] + 36, tgt[1] - 40, 22, E.GOOD, 4);

      /* ── 右 ④：星号的说明 ── */
      var partB = E.el('g', {}, s);
      var notT = E.el('g', {}, partB);
      var notCode = E.txt(notT, RX + 26, 62, '*', 40, E.BAD, 'middle', 700); notCode.style.fontFamily = E.MONO;
      E.txt(notT, RX + 52, 60, '不是矩阵乘法', 26, E.BAD, 'start', 900);
      var MA = 540, MB = 720, MY = 204;
      var mR = E.mat(partB, MA, MY, [0, -1, 1, 0], { size: 34, colors: [H.good, H.warn, H.good, H.warn] });
      var mul = E.el('g', {}, partB);
      ['×3', '×2'].forEach(function (s2, c) { E.txt(mul, MA + (c ? 1 : -1) * mR.dx / 2, mR.box[1] - 16, s2, 26, E.ACC, 'middle', 900); });
      var eq = E.txt(partB, (MA + MB) / 2, MY + 10, '=', 32, E.INK, 'middle');
      eq.style.fontFamily = 'KaTeX_Main, serif';
      var mS = E.mat(partB, MB, MY, [0, -2, 3, 0], { size: 34, colors: [H.good, H.warn, H.good, H.warn] });
      var split = E.el('g', {}, partB);
      E.tex(split, MA - 4, 334, '3' + E.col([0, 1], H.good), 28, E.GOOD, 'middle', { w: 120, h: 90 });
      E.tex(split, MB - 4, 334, '2' + E.col([-1, 0], H.warn), 28, E.WARN, 'middle', { w: 130, h: 90 });
      var noSum = E.el('g', {}, partB);
      K.paint(E.el('circle', { cx: (MA + MB) / 2, cy: 324, r: 24, fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '5 4' }, noSum), null, E.BAD);
      E.txt(noSum, (MA + MB) / 2, 333, '+', 30, E.BAD, 'middle', 900);
      E.txt(noSum, (MA + MB) / 2, 396, '没加起来', 22, E.BAD, 'middle', 900);

      /* ── 右 ⑤：两个小牌 ── */
      var sum = E.el('g', {}, s);
      function rule(y, code, text, color) {
        var g = E.el('g', {}, sum);
        K.paint(E.el('rect', { x: RX + 4, y: y, width: RW - 8, height: 56, rx: 12, 'stroke-width': 2.5 }, g), E.tint(color, 12, 'var(--demo-surface)'), color);
        var c = E.txt(g, RX + 46, y + 38, code, 32, color, 'middle', 700); c.style.fontFamily = E.MONO;
        E.txt(g, RX + 82, y + 37, text, 24, E.INK, 'start', 900);
        return g;
      }
      var ruleAt = rule(150, '@', '矩阵乘法', E.ACC), ruleStar = rule(230, '*', '对应位置相乘', E.BAD);

      return {
        el: s,
        draw: function (st, t) {
          var fade = E.fade, win = E.win;
          py.draw(st, t);
          // 「中间」：R 和 p 之间的空格框一圈，@ 落进去以后收掉
          py.boxCol(L_AT, 6, 1, E.HI, win(st, GAP_AT, KEYS[2] + 0.4, 0.2));
          var yOut = py.y(L_OUT), yAt = py.y(L_AT), yStar = py.y(L_STAR);
          okSame.setAttribute('transform', 'translate(' + (py.x(16) + 6) + ' ' + (yOut - 18) + ')');
          E.op(okSame, fade(st, SAME, 0.3) * (1 - fade(st, PART_A_OUT, 0.4)));
          okAt.setAttribute('transform', 'translate(' + (py.x(10) + 6) + ' ' + (yAt - 18) + ')');
          E.op(okAt, fade(st, AT_OK, 0.3));
          badStar.setAttribute('transform', 'translate(' + (py.x(11) + 22) + ' ' + (yStar - 8) + ')');
          E.op(badStar, fade(st, NOT, 0.3));
          py.hl(L_AT, E.ACC, fade(st, AT_OK - 0.4, 0.3));
          py.hl(L_STAR, E.BAD, fade(st, STAR_SUM - 0.3, 0.3));
          // 「第一列乘了三」「第二列乘了二」：输出里同一列也框
          var c1 = win(st, COL1[0], SUM_UP, 0.25), c2 = win(st, COL2[0], SUM_UP, 0.25);
          py.boxCol(L_S1, 9, 1, E.GOOD, c1); py.boxCol(L_S2, 9, 1, E.GOOD, c1);
          py.boxCol(L_S1, 12, 2, E.WARN, c2); py.boxCol(L_S2, 13, 1, E.WARN, c2);

          /* ② 上一集手算卡 */
          E.pop(card, RC, 82, fade(st, CARD, 0.35) * (1 - fade(st, PART_A_OUT, 0.4)), 0.85);

          /* ③ 网格 + UR5 */
          E.op(partA, fade(st, 0.1, 0.5) * (1 - fade(st, PART_A_OUT, 0.4)));
          arm.view.clearTint();
          var tw = bell(st, TURN[0] - 0.1, TURN[1] + 0.3);
          if (tw > 0) arm.view.tint(LINKS, E.VIOLET, tw);
          var tp = arm.turn(Math.PI / 2 * fade(st, TURN[0], TURN[1] - TURN[0]));
          pos.set(OX, OY, tp[0], tp[1]);
          ghost.set(OX, OY, hp[0], hp[1]);
          E.op(ghost.g, 0.6 * fade(st, GHOST, 0.3));
          var pp = fade(st, PATH, 0.6), gone = 1 - fade(st, TURN[1] + 0.6, 0.4);
          E.drawOn(path.line, pp); E.op(path.head, K.seg(pp, 0.85, 1) * gone); E.op(path.line, pp > 0 ? gone : 0);
          E.op(landLbl, fade(st, LAND, 0.3));
          E.pop(okLand, tgt[0] + 47, tgt[1] - 29, fade(st, OK, 0.3), 0.5);

          /* ④ 星号的说明 */
          E.op(partB, fade(st, NOT, 0.4) * (1 - fade(st, SUM_UP, 0.4)));
          E.pop(notT, RX + 130, 52, fade(st, NOT, 0.3), 0.85);
          E.op(mR.g, fade(st, POS - 0.3, 0.4));
          E.op(mul, fade(st, POS, 0.4));
          E.op(eq, fade(st, RES, 0.3)); E.op(mS.g, fade(st, RES + 0.15, 0.4));
          mS.colBox(0, E.GOOD, c1); mS.colBox(1, E.WARN, c2);
          mR.colBox(0, E.GOOD, win(st, COL1[0], COL1[1], 0.2)); mR.colBox(1, E.WARN, win(st, COL2[0], COL2[1], 0.2));
          E.op(split, fade(st, SPLIT, 0.4));
          E.op(noSum, fade(st, NOSUM, 0.3));

          /* ⑤ 总结小牌 */
          E.op(sum, fade(st, SUM_UP + 0.2, 0.4));
          E.pop(ruleAt, RC, 178, fade(st, AT_OK, 0.3), 0.85);
          E.pop(ruleStar, RC, 258, fade(st, STAR_SUM, 0.3), 0.85);
        }
      };
    }
  };
})();
