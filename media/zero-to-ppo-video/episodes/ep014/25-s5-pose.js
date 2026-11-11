// 014 第 5 幕：姿势一变，两种雅可比各自怎么变（《Modern Robotics》5.1.3 节图 5.9：前面的关节不挪动后面的轴相对 {b} 的位置……；
// 空间雅可比第 i 列只跟前面的关节有关（第 13 集），物体雅可比第 i 列只跟后面的关节有关，最后一列就是零位时的 Bₙ，式 (5.18)）。
// 左边网格 + UR5 + 底座、手两个坐标系（肩一路逆时针多转 20°，再肘一路逆时针转 40°）；右栏两个实时的 6 × 2：站在底座上看（空间雅可比）、站在手上看（物体雅可比），每帧按此刻的姿势用 E.jac2 算，保留 2 位小数
// （官方库 JacobianSpace / JacobianBody 在 θ = (90°, 0)、(118.6°, 0)、(90°, 28.6°)、(90°, 57.3°) 对过）。
// 句 2「让肩继续转」肩逆时针一路转 20°（蓝色小弯箭头），肘的 ⊙ 亮，J_s 第 2 列框亮、数在变；句 3 J_b 两列打勾「不变」；
// 句 4「再让肘转」肘逆时针转 40°，J_s 两列打勾，J_b 第 1 列框亮、数在变，肩的 ⊙ 亮；句 5 右栏写两句对照；句 6 写 J_b = [Ad_Tbs] J_s。
// 时刻取自配音的逐词时间，见 ../ep014.py。
(function () {
  var E = window.EP014, K = E.K, H = E.HEX;
  var X = 455, D90 = Math.PI / 2;
  var T = {
    pose: 0.7, change: 1.19, two: 1.6, recalc: 3.07, diff: 3.75,
    sh: 6.57, rot1: [7.14, 27.8], base: 7.59, elAx: 9.03, moved: 10.13, col2: 17.49, changing: 18.15, ep13: 18.82,
    standH: 20.1, twoCols: 21.25, noChange: 22.88, you: 23.54, whole: 24.59, eyes: 26.49, shEl: 27.41,
    el: 30.84, rot2: [31.14, 38.44], base2: 32.24, noChange2: 33.8, front: 34.61, standH2: 36.32, col1: 37.4, changing2: 37.78, shMoved: 39.69,
    s5: 42.03, front5: 42.96, follow5: 45.64, hand5: 46.33, rev5: 47.86, follow6: 50.89,
    always: 52.72, adj: 56.53,
    eo0: 13.64
  };
  var END = 58.46, A1 = 0.35, A2 = 0.7;   // 肩多转 20°、肘转 40°：胳膊一直在 x 轴上面

  E.scenes[4] = {
    title: '姿势变了：两种雅可比各自怎么变',
    dur: END,
    cues: [
      { at: 0, s: '姿势一变，**空间雅可比**、**物体雅可比**都要重算，但变的方式不一样' },
      { at: 6.22, s: '肩继续转：**站在底座上看**，第 $2$ 列要用肘此刻的位置算（从肘关节指向原点、转 $90^\\circ$），肘一挪就变' },
      { at: 19.93, s: '**站在手上看**，$2$ 列一个数字都不变：整条胳膊跟着你一起转，肩和肘的位置都没变' },
      { at: 30.31, s: '肘转：**站在底座上看** $2$ 列都不变；**站在手上看**第 $1$ 列在变——从手上看，肩挪了地方' },
      { at: 41.38, s: '底座上看：前面的关节一转，后面的轴跟着挪；手上看：后面的关节一转，前面的轴跟着挪' },
      { at: 52.07, s: '不管姿势怎么变，两种雅可比之间，永远隔着一个**伴随变换**' }
    ],
    build: function (K) {
      var s = K.sceneSvg('姿势一变，两种雅可比都要重算，但变的方式不一样。让肩继续转：站在底座上看，肘的轴被带着挪了地方，空间雅可比的第二列一直在变；站在手上看，物体雅可比两列一个数都不变，整条胳膊跟着你一起转。再让肘转：站在底座上看两列都不变；站在手上看第一列在变，从手上看肩挪了地方。站在底座上看，前面的关节一转，后面的轴跟着挪；站在手上看，后面的关节一转，前面的轴跟着挪。不管姿势怎么变，两种雅可比之间永远隔着一个伴随变换');

      var pn = E.panel14(s), P = pn.P, u = pn.u, top = pn.top, O = P(0, 0);
      var bF = E.frame2(top, P, 1, '底座', { labAt: [0.75, -0.62], tipOff: E.BASE_TIP }), hF = E.frame2(top, P, 1, '手', { labAt: [0.5, -0.45], labColor: E.V, tipOff: [[-0.15, 0.3], [-0.35, 0.1]] });   // 肘转完手的 y 轴朝左下，字母写在尖外面会压到白色 x 轴：改写在尖的左上
      var jS = E.joint(top, E.HI), jE = E.joint(top, E.HI), hand = E.dot(top, E.V, 7);
      var eoA = E.gv(top, P, E.INK, 3.5, { head: 13, dash: '8 6' });   // 第 2 列怎么算：从肘关节指向原点（跟着肘一起挪）
      var sArc = E.arcArrow(top, O[0], O[1], 0.6 * u, E.OMEGA, 3.5, { head: 12 });
      var sLab = E.tex(top, O[0] - 132, O[1] + 46, '\\dot\\theta_1', 22, E.OMEGA, 'start', { w: 50, h: 36 });
      var eArc = E.arcArrow(top, 0, 0, 0.55 * u, E.OMEGA, 3.5, { head: 12 });
      var eLab = E.tex(top, 0, 0, '\\dot\\theta_2', 22, E.OMEGA, 'start', { w: 50, h: 36 });

      /* ══ 右栏：两个实时的 6 × 2 ══ */
      var RK = E.rowKit(), Rg = E.el('g', {}, s), MY = 146;
      var W = [0, 1, 2, 3, 4, 5].map(function () { return ['−2.22', '−2.22']; });
      function col(i, j) { return j === 0 ? E.J1 : E.J2; }
      var gM = E.el('g', {}, Rg);
      var Ms = E.matN(gM, X + 82, MY, W, { size: 17, dx: 54, dy: 24, wide: W, color: col });
      var Mb = E.matN(gM, X + 258, MY, W, { size: 17, dx: 54, dy: 24, wide: W, color: col });
      var ls = E.txt(gM, X + 82, 46, '站在底座上看', 18, E.INK, 'middle', 900), lb = E.txt(gM, X + 258, 46, '站在手上看', 18, E.V, 'middle', 900);
      var ns = E.txt(gM, X + 82, 70, '空间雅可比', 16, E.INK2, 'middle', 900), nb = E.txt(gM, X + 258, 70, '物体雅可比', 16, E.INK2, 'middle', 900);
      // 「不变」的勾：每个矩阵每列下面一个
      var oks = [Ms, Mb].map(function (M) { return [0, 1].map(function (j) { return E.check(gM, M.pos(5, j)[0] - 10, M.box[3] + 8, 20, E.GOOD, 3.5); }); });
      var thL = E.txt(gM, X, 270, '', 18, E.INK2, 'start', 700);
      var gS = E.el('g', {}, Rg);
      var s1 = RK.line(gS, X, 314, [['底座上看：', E.INK, 900], ['前面的关节一转，后面的轴跟着挪', E.INK2]], 16);
      var s2 = RK.line(gS, X, 344, [['手上看：', E.V, 900], ['后面的关节一转，前面的轴跟着挪', E.INK2]], 16);
      var s3 = E.tex(gS, X, 386, '\\textcolor{' + H.violet + '}{J_b}=\\textcolor{' + H.hi + '}{[\\mathrm{Ad}_{T_{bs}}]}\\,J_s\\quad\\text{（永远成立）}', 21, E.INK, 'start', { w: 330, h: 40 });

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) { RK.layout(); laid = true; }
          var f = E.fade, b = E.bump;
          // 只朝蓝色小弯箭头的方向（逆时针）转：句 2 肩一路转 A1，句 4 肘一路转 A2，都不往回转
          var th1 = D90 + A1 * f(st, T.rot1[0], T.rot1[1] - T.rot1[0]), th2 = A2 * f(st, T.rot2[0], T.rot2[1] - T.rot2[0]);
          pn.pose(E.ZERO[0] + th1, th1 + th2); pn.view.clearTint();
          var J = pn.joints(), pE = P(J[1][0], J[1][1]), pH = P(J[2][0], J[2][1]);
          jS.at(O); jE.at(pE); hand.at(pH);
          eoA.at(J[1][0], J[1][1], -0.06 * J[1][0], -0.06 * J[1][1]); E.op(eoA.g, E.win(st, T.eo0, T.standH - 0.2, 0.3));
          bF.set(0, 0, 0); hF.set(J[2][0], J[2][1], th1 + th2);
          E.glow(jE, E.HI, E.win(st, T.elAx, T.standH, 0.2) > 0.5);
          E.glow(jS, E.HI, E.win(st, T.shMoved - 0.3, T.s5, 0.2) > 0.5);
          E.glow(hF.g, E.V, E.win(st, T.standH, T.el, 0.2) > 0.5 || E.win(st, T.standH2, T.s5, 0.2) > 0.5);
          E.glow(bF.g, E.INK, b(st, T.base, 1.2) > 0.05 || b(st, T.base2, 1.2) > 0.05);
          sArc.set(3.45, 4.55); var on1 = E.win(st, T.sh, T.rot1[1] + 0.3, 0.3);
          E.op(sArc.g, on1); E.op(sLab, on1);
          eArc.g.setAttribute('transform', 'translate(' + pE[0].toFixed(1) + ' ' + pE[1].toFixed(1) + ')');
          var ea = Math.atan2(J[1][1], J[1][0]) + Math.PI;   // 从肘指回肩的方向；小弯箭头画在它的两边各偏一点（不压大臂）
          eArc.set(ea + 0.5, ea + 1.6);
          eLab.setAttribute('transform', 'translate(' + (pE[0] - 66).toFixed(1) + ' ' + (pE[1] + 40).toFixed(1) + ')');
          var on2 = E.win(st, T.el, T.rot2[1] + 0.3, 0.3);
          E.op(eArc.g, on2); E.op(eLab, on2);

          /* ── 右栏 ── */
          var jj = E.jac2(J, th1 + th2);
          Ms.set(jj.Js.map(function (r) { return r.map(function (x) { return E.num(x, 2); }); }));
          Mb.set(jj.Jb.map(function (r) { return r.map(function (x) { return E.num(x, 2); }); }));
          E.op(gM, fr(st, T.pose - 0.1));
          E.op(Ms.g, fr(st, T.two)); E.op(ls, fr(st, T.two)); E.op(ns, fr(st, T.two)); E.op(Mb.g, fr(st, T.two + 0.4)); E.op(lb, fr(st, T.two + 0.4)); E.op(nb, fr(st, T.two + 0.4));
          Ms.hl('c2', 0, 1, 5, 1, E.J2, E.win(st, T.col2, T.standH, 0.3));
          Mb.hl('c1', 0, 0, 5, 0, E.J1, E.win(st, T.col1, T.s5, 0.3));
          // 勾：肩转时 J_b 两列不变；肘转时 J_s 两列不变（J_b 第 2 列也一直不变）
          E.op(oks[1][0], E.win(st, T.noChange, T.el, 0.3)); E.op(oks[1][1], E.win(st, T.noChange, T.s5, 0.3));
          E.op(oks[0][0], E.win(st, T.noChange2, T.s5, 0.3)); E.op(oks[0][1], E.win(st, T.noChange2, T.s5, 0.3));
          thL.textContent = '肩 θ₁ = ' + Math.round(th1 * 180 / Math.PI) + '°　肘 θ₂ = ' + Math.round(th2 * 180 / Math.PI) + '°';
          E.op(thL, fr(st, T.sh));
          E.op(gS, fr(st, T.s5 - 0.1));
          E.op(s1.g, fr(st, T.s5)); E.op(s1.parts[1], fr(st, T.front5)); E.op(s2.g, fr(st, T.hand5)); E.op(s2.parts[1], fr(st, T.rev5)); E.op(s3, fr(st, T.always));
          E.glow(s3, E.HI, b(st, T.adj, 1.2) > 0.05);
        }
      };
    }
  };
})();
