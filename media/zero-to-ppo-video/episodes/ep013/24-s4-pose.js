// 013 第 4 幕：姿势一变，雅可比就变（《Modern Robotics》第 5 章开头：J(θ) 是关节角的函数；θ2 = 0° 时两列共线，叫 singularity）。
// 左边网格 + UR5，手上两支箭头（肩那一列亮黄、肘那一列青）跟着此刻的姿势算；右栏写此刻的雅可比（数跟着变，保留两位小数）。
// 句 1「雅可比」「不是一个固定的矩阵」右栏出 J（零位的数），「此刻的位置」手和两个关节亮一下，「胳膊一动」胳膊慢慢动、两支箭头和 J 的数一起变；
// 句 2「胳膊伸直」「平放在 x 轴上」胳膊转到伸直，「第十二集里」「最远的那一点」画出半径 4.24 的圆弧（绿虚线，第 12 集手能到的最远）；
// 句 3「根号五、零」「四点二四、零」标出肘、手的位置；
// 句 4「只转肩」白箭头肩 → 手（贴着 x 轴），「转九十度」绕原点转到朝正上方，「第一列取四点二四份」右栏写，「潮正上方」平移到手上、变亮黄；
// 句 5「只转肘」白箭头肘 → 手，绕肘转 90°，平移到手上、变青；
// 句 6「两列都潮正上方」右栏 J 的第一排（往右的那一排）两个 0 框亮，「不管两个关节怎么转」转速来回变、紫色的和只在竖线上伸缩，
//    「往左、往右都走不了」手上左右各一支橙色虚线箭头打 ×；
// 句 7「伸到最远」圆弧亮，「奇异位形」手旁写（橙），「第十五集细讲」右栏写。
// 时刻取自配音的逐词时间，见 ../ep013.py。
(function () {
  var E = window.EP013, K = E.K, H = E.HEX;
  var RX = E.ARM.RX, X = RX + 10, D90 = Math.PI / 2, R5 = Math.sqrt(5), FAR = R5 + 2;
  var T = {
    note: 0.7, jac: 1.66, notFix: 2.36, each: 4.4, pos: 5.66, now: 6.42, calc: 7.56, move: 8.05, change: 8.96, out1: 10.9,
    pose: 11.22, straight: [12.33, 13.6], flat: 13.24, ep12: 14.91, far: 15.97, farPt: 17.24,
    elW: 18.98, e5: 19.41, e224: 20.79, handW: 22.66, h424: 24.05,
    sh: 26.77, from: 27.59, d424: 29.3, turn: [30.71, 31.5], take: 31.72, take0: 33.45, res: 34.54, up: 36.05, moveC1: [35.4, 36.2],
    el: 38.16, from2: 39.07, d20: 40.87, turn2: [41.81, 42.6], res2: 42.87, up2: 43.99, moveC2: [43.1, 43.9],
    both: 46.26, any: 47.74, wig: [48.2, 54.6], updown: 49.35, now6: 51.54, lr: 52.49, cant: 53.82,
    farA: 55.56, noMore: 57.01, kind: 58.69, sing: 59.65, ep15: 60.93
  };
  var END = 63.27;

  E.scenes[3] = {
    title: '姿势一变，雅可比就变',
    dur: END,
    cues: [
      { at: 0, s: '**雅可比**不是一个固定的矩阵：每一列都用手和关节此刻的位置来算，胳膊一动，它就变' },
      { at: 11.12, s: '换个姿势：胳膊伸直，平放在 $x$ 轴上——第 12 集手能到的最远的那一点' },
      { at: 18.88, s: '肘在 $(\\sqrt5,\\,0)\\approx(2.24,\\,0)$，手在 $(4.24,\\,0)$' },
      { at: 26.67, s: '只转肩：$(4.24,\\,0)$ 转 $90^\\circ$：$4.24\\times(0,\\,1)+0\\times(-1,\\,0)=(0,\\,4.24)$，朝正上方' },
      { at: 38.06, s: '只转肘：$(2,\\,0)$ 转 $90^\\circ$ $=(0,\\,2)$，也朝正上方' },
      { at: 46.16, s: '$2$ 列都朝正上方：不管两个关节怎么转，手这一刻只能上、下，往左、往右都**走不了**' },
      { at: 55.46, s: '胳膊伸到最远，再也伸不出去：这种姿势叫**奇异位形**（第 15 集细讲）' }
    ],
    build: function (K) {
      var s = K.sceneSvg('雅可比不是一个固定的矩阵：每一列都要用手和关节此刻的位置来算，胳膊一动它就变。换一个姿势：胳膊伸直，平放在 x 轴上，就是第十二集手能到的最远的那一点；肘在 (√5, 0) ≈ (2.24, 0)，手在 (4.24, 0)。只转肩：从肩关节指向手 (4.24, 0)，转九十度是 (0, 4.24)，朝正上方；只转肘：从肘关节指向手 (2, 0)，转九十度是 (0, 2)，也朝正上方。两列都朝正上方：不管两个关节怎么转，加起来也只能朝上、朝下，手这一刻往左、往右都走不了。胳膊伸到最远，再也伸不出去，这种姿势叫奇异位形，第十五集细讲');

      var pn = E.panel(s), P = pn.P, u = pn.u, top = pn.top, O = P(0, 0);
      function arcPath(cx, cy, r, a0, a1) {
        var pts = [];
        for (var k = 0; k <= 48; k++) { var a = a0 + (a1 - a0) * k / 48; pts.push(P(cx + r * Math.cos(a), cy + r * Math.sin(a))); }
        return E.poly(pts);
      }
      var farArc = K.paint(E.el('path', { d: arcPath(0, 0, FAR, -0.235, 1.62), fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '8 6' }, top), null, E.GOOD);
      var farL = E.halo(E.txt(top, P(-1.9, 4.6)[0], P(-1.9, 4.6)[1], '最远 4.24 格', 18, E.GOOD, 'start', 900));
      var jS = E.joint(top, E.HI), jE = E.joint(top, E.HI), hand = E.dot(top, E.V, 7);
      var c1 = E.gv(top, P, E.J1, 5.5, { head: 17 }), c2 = E.gv(top, P, E.J2, 5.5, { head: 17 }), vv = E.gv(top, P, E.V, 6, { head: 18 });
      var dA = E.gv(top, P, E.INK, 4.5, { head: 15 }), dSh = E.gv(top, P, E.INK2, 3, { head: 12, dash: '6 6' });
      var turnArc = E.arcArrow(top, O[0], O[1], 0.95 * u, E.OMEGA, 3, { head: 11 });
      var dA2 = E.gv(top, P, E.INK, 4.5, { head: 15 }), dSh2 = E.gv(top, P, E.INK2, 3, { head: 12, dash: '6 6' });
      var turnArc2 = E.arcArrow(top, P(R5, 0)[0], O[1], 0.8 * u, E.OMEGA, 3, { head: 11 });
      var eLab = E.halo(E.txt(top, P(R5, 0)[0], O[1] + 30, '(2.24, 0)', 18, E.HI, 'middle', 900));
      var hLab = E.halo(E.txt(top, P(FAR, 0)[0] + 2, O[1] + 30, '(4.24, 0)', 18, E.V, 'middle', 900));
      var c1L = E.halo(E.txt(top, 0, 0, '(0, 4.24)', 18, E.J1, 'end', 900)), c2L = E.halo(E.txt(top, 0, 0, '(0, 2)', 18, E.J2, 'start', 900));
      // 句 6：左右两支走不了的箭头
      var noL = E.vec(top, E.ERR, 4, { head: 14, dash: '7 5' }), noR = E.vec(top, E.ERR, 4, { head: 14, dash: '7 5' });
      var HF = P(FAR, 0);
      noL.set(HF[0] - 8, HF[1] - 18, HF[0] - 1.3 * u, HF[1] - 18); noR.set(HF[0] + 8, HF[1] - 18, HF[0] + 1.2 * u, HF[1] - 18);
      var xL = E.cross(top, HF[0] - 0.75 * u, HF[1] - 34, 7, E.ERR, 4), xR = E.cross(top, HF[0] + 0.7 * u, HF[1] - 34, 7, E.ERR, 4);
      var singL = E.halo(E.txt(top, HF[0] - 6, HF[1] + 58, '奇异位形', 22, E.ERR, 'end', 900));

      /* ══ 右栏 ══ */
      var RK = E.rowKit(), Rg = E.el('g', {}, s);
      function grp() { return E.el('g', {}, Rg); }
      var gA = grp();
      var a0 = RK.line(gA, X, 44, [['雅可比', E.HI, 900], ['：不是固定的', E.INK, 900]], 24);
      var JA = E.matN(gA, X + 170, 136, [['−2', '0'], ['3', '2']], { size: 28, dx: 86, dy: 40, wide: [['−2.00', '−0.00'], ['−0.00', '−0.00']],
        color: function (i, j) { return j === 0 ? E.J1 : E.J2; } });
      var aJ = E.txt(gA, X + 14, 146, 'J =', 26, E.INK2, 'start', 700);
      var a2 = RK.line(gA, X, 226, [['每一列：用', E.INK2], ['手', E.V, 900], ['和', E.INK2], ['关节', E.HI, 900], ['此刻的位置算', E.INK2]], 21);
      var a3 = RK.line(gA, X, 262, [['胳膊一动 → 雅可比就变', E.INK, 900]], 21);
      var gB = grp();
      var b0 = RK.line(gB, X, 34, [['换个姿势：', E.INK, 900], ['胳膊伸直，平放在 x 轴上', E.INK2]], 19);
      var b1 = RK.line(gB, X, 62, [['（第 12 集：手能到的最远的那一点）', E.MUT, 700]], 16);
      var b2 = RK.line(gB, X, 96, [['肘 (√5, 0) ≈ (2.24, 0)；手 (4.24, 0)', E.INK2, 700]], 19);   // 一段写完：带 √ 的段量出来的宽度偏小，下一段会压上去
      var b3 = RK.line(gB, X, 136, [['只转肩：', E.J1, 900], ['(4.24, 0) 转 90°', E.INK2]], 19);
      var b4b = RK.line(gB, X + 16, 192, [['= ', E.INK2], ['(0, 4.24)', E.J1, 900], ['，朝正上方', E.INK2]], 18);
      var b4 = RK.line(gB, X + 16, 164, [['4.24 × ', E.INK2], ['(0, 1)', E.X, 900], [' + 0 × ', E.INK2], ['(−1, 0)', E.Y, 900]], 18);   // 一排写不下：结果挪到下一排
      var b5 = RK.line(gB, X, 226, [['只转肘：', E.J2, 900], ['(2, 0) 转 90° = ', E.INK2], ['(0, 2)', E.J2, 900]], 19);
      var JB = E.matN(gB, X + 170, 286, [[0, 0], [4.24, 2]], { size: 26, dx: 78, dy: 36, color: function (i, j) { return j === 0 ? E.J1 : E.J2; } });
      var bJ = E.txt(gB, X + 40, 295, 'J =', 24, E.INK2, 'start', 700);
      var bRow = E.txt(gB, JB.box[2] + 10, JB.pos(0, 0)[1] + 7, '← 往右', 17, E.ERR, 'start', 900);
      var b6 = RK.line(gB, X, 352, [['2 列都朝正上方：只能上、下', E.INK, 900]], 19);
      var b7 = RK.line(gB, X, 380, [['往左、往右：', E.INK2], ['走不了', E.ERR, 900]], 19);
      var b8 = RK.line(gB, X, 408, [['奇异位形', E.ERR, 900], ['（第 15 集细讲）', E.MUT, 700]], 20);

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }
      function rot(v, a) { return [v[0] * Math.cos(a) - v[1] * Math.sin(a), v[0] * Math.sin(a) + v[1] * Math.cos(a)]; }
      function n2(x) { var v = Math.abs(x) < 0.005 ? 0 : x; return (v < 0 ? '−' : '') + Math.abs(v).toFixed(2); }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) { RK.layout(); laid = true; }
          var f = E.fade, b = E.bump;
          /* ── UR5：句 1「胳膊一动」慢慢动（两个关节各转一点），句 2 转到伸直 ── */
          var mv = E.win(st, T.move, T.straight[0] + 0.2, 0.5), sk = f(st, T.straight[0], T.straight[1] - T.straight[0]);
          var ph = st - T.move, d1 = 0.35 * Math.sin(ph * 1.4) * mv, d2 = -0.5 * Math.sin(ph * 1.1) * mv;
          var ang1 = (E.ZERO[0] + d1) * (1 - sk), ang2 = (E.ZERO[1] + d1 + d2) * (1 - sk);
          pn.pose(ang1, ang2);
          pn.view.clearTint();
          var J = pn.joints(), pE = P(J[1][0], J[1][1]), pH = P(J[2][0], J[2][1]);
          jS.at(O); jE.at(pE); hand.at(pH, 7 + 4 * b(st, T.pos, 0.8));
          E.glow(jS, E.HI, b(st, T.now, 0.8) > 0.05); E.glow(jE, E.HI, b(st, T.now, 0.8) > 0.05);
          /* 第一部分：手上两支箭头跟着此刻的姿势（句 1–2）；转到伸直以后先收起，句 4、5 再一支一支算出来 */
          var col1 = rot(J[2], D90), col2 = rot([J[2][0] - J[1][0], J[2][1] - J[1][1]], D90);
          var live = 1 - fr(st, T.straight[0]);   // 转到伸直的途中就收起：箭头会扫过 y 轴的标签
          if (st < T.sh) {
            c1.at(J[2][0], J[2][1], J[2][0] + col1[0], J[2][1] + col1[1]); c2.at(J[2][0], J[2][1], J[2][0] + col2[0], J[2][1] + col2[1]);
            E.op(c1.g, live); E.op(c2.g, live);
          }
          JA.set([[n2(col1[0]), n2(col2[0])], [n2(col1[1]), n2(col2[1])]]);
          if (st < T.move) JA.set([['−2', '0'], ['3', '2']]);
          JA.hl('c1', 0, 0, 1, 0, E.J1, b(st, T.each, 1.2)); JA.hl('c2', 0, 1, 1, 1, E.J2, b(st, T.each + 0.4, 1.2));
          E.drawOn(farArc, f(st, T.far, 1.0)); E.op(farArc, st > T.far ? 0.85 : 0); E.glow(farArc, E.GOOD, b(st, T.farA, 1.4) > 0.05);
          E.op(farL, fr(st, T.farPt));
          E.op(eLab, fr(st, T.e5)); E.op(hLab, fr(st, T.h424));
          /* 句 4：只转肩（白箭头肩 → 手，绕原点转 90°，平移到手上） */
          var g4 = f(st, T.from, 0.8), tr = f(st, T.turn[0], T.turn[1] - T.turn[0]), m1 = f(st, T.moveC1[0], T.moveC1[1] - T.moveC1[0]);
          var dv = rot([FAR, 0], D90 * tr);
          dA.at(0, 0, dv[0] * g4, dv[1] * g4); E.op(dA.g, (st > T.from ? 1 : 0) * (1 - fr(st, T.moveC1[0])));
          dSh.at(0, 0, FAR, 0); E.op(dSh.g, (tr > 0.02 ? 0.6 : 0) * (1 - fr(st, T.moveC1[0])));
          turnArc.set(0, D90 * Math.max(tr, 0.01)); E.op(turnArc.g, (tr > 0.01 ? 1 : 0) * (1 - fr(st, T.moveC1[0])));
          /* 句 5：只转肘 */
          var g5 = f(st, T.from2, 0.8), tr2 = f(st, T.turn2[0], T.turn2[1] - T.turn2[0]), m2 = f(st, T.moveC2[0], T.moveC2[1] - T.moveC2[0]);
          var dv2 = rot([2, 0], D90 * tr2);
          dA2.at(R5, 0, R5 + dv2[0] * g5, dv2[1] * g5); E.op(dA2.g, (st > T.from2 ? 1 : 0) * (1 - fr(st, T.moveC2[0])));
          dSh2.at(R5, 0, FAR, 0); E.op(dSh2.g, (tr2 > 0.02 ? 0.6 : 0) * (1 - fr(st, T.moveC2[0])));
          turnArc2.set(0, D90 * Math.max(tr2, 0.01)); E.op(turnArc2.g, (tr2 > 0.01 ? 1 : 0) * (1 - fr(st, T.moveC2[0])));
          if (st >= T.sh) {
            var x1 = FAR * m1; c1.at(x1, 0, x1, FAR); E.op(c1.g, st > T.moveC1[0] ? 1 : 0);
            var x2 = R5 + 2 * m2; c2.at(x2, 0, x2, 2); E.op(c2.g, st > T.moveC2[0] ? 1 : 0);
            var c1t = P(x1, FAR), c2t = P(x2, 2);
            c1L.setAttribute('x', (c1t[0] - 12).toFixed(1)); c1L.setAttribute('y', (c1t[1] + 6).toFixed(1)); E.op(c1L, fr(st, T.moveC1[1]) * (1 - fr(st, T.any)));
            c2L.setAttribute('x', (c2t[0] + 12).toFixed(1)); c2L.setAttribute('y', (c2t[1] + 6).toFixed(1)); E.op(c2L, fr(st, T.moveC2[1]) * (1 - fr(st, T.any)));
          } else { E.op(c1L, 0); E.op(c2L, 0); }
          /* 句 6：转速来回变，和只在竖线上伸缩 */
          var w6 = E.win(st, T.wig[0], T.wig[1], 0.4);
          var vy = 0.9 + 2.1 * Math.sin((st - T.wig[0]) * 1.5);   // 两列都朝上：和只在竖线上伸缩（往下最多 1.2 格，原先伸出画面）
          vv.at(FAR, 0, FAR, vy); E.op(vv.g, w6 * (Math.abs(vy) > 0.15 ? 1 : 0));
          if (st > T.wig[0]) { E.op(c1.g, 0.45 + 0.55 * (1 - w6)); E.op(c2.g, 0.45 + 0.55 * (1 - w6)); }
          var no = fr(st, T.lr);
          E.op(noL.g, no); E.op(noR.g, no); E.op(xL, fr(st, T.cant)); E.op(xR, fr(st, T.cant));
          E.op(singL, fr(st, T.sing));

          /* ── 右栏 ── */
          E.op(gA, 1 - fr(st, T.out1));
          E.op(a0.g, fr(st, T.jac)); E.op(a0.parts[1], fr(st, T.notFix)); E.op(JA.g, fr(st, T.notFix)); E.op(aJ, fr(st, T.notFix));
          E.op(a2.g, fr(st, T.each)); E.op(a3.g, fr(st, T.move));
          E.op(gB, fr(st, T.pose - 0.1));
          E.op(b0.g, fr(st, T.pose)); E.op(b0.parts[1], fr(st, T.straight[0])); E.op(b1.g, fr(st, T.ep12));
          E.op(b2.g, fr(st, T.elW));
          E.op(b3.g, fr(st, T.sh)); E.op(b3.parts[1], fr(st, T.d424));
          E.op(b4.g, fr(st, T.take)); E.op(b4.parts[2], fr(st, T.take0)); E.op(b4.parts[3], fr(st, T.take0)); E.op(b4b.g, fr(st, T.res)); E.op(b4b.parts[2], fr(st, T.up));
          E.op(b5.g, fr(st, T.el)); E.op(b5.parts[1], fr(st, T.d20)); E.op(b5.parts[2], fr(st, T.res2));
          E.op(JB.g, fr(st, T.both)); E.op(bJ, fr(st, T.both));
          JB.hl('r0', 0, 0, 0, 1, E.ERR, fr(st, T.lr));
          E.op(bRow, fr(st, T.lr));
          E.op(b6.g, fr(st, T.both)); E.op(b7.g, fr(st, T.lr)); E.op(b7.parts[1], fr(st, T.cant));
          E.op(b8.g, fr(st, T.sing)); E.op(b8.parts[1], fr(st, T.ep15));
        }
      };
    }
  };
})();
