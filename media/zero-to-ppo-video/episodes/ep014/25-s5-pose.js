// 014 第 5 幕：姿势一变，两种雅可比各自怎么变（《Modern Robotics》5.1.3 节图 5.9：前面的关节不挪动后面的轴相对 {b} 的位置……；
// 空间雅可比第 i 列只跟前面的关节有关（第 13 集），物体雅可比第 i 列只跟后面的关节有关，最后一列就是零位时的 Bₙ，式 (5.18)）。
// 左边网格 + UR5 + 底座、手两个坐标系；右栏两个实时的 6 × 2：底座上看（空间雅可比）、手上看（物体雅可比），每帧按此刻的姿势用 E.jac2 算，保留 2 位小数
// （官方库 JacobianSpace / JacobianBody 在 θ = (90°, 0)、(110°, 0)、(90°, 40°) 对过）。
// 2026-11-11 用户要求「后半部分对比两种雅可比的数怎么变时，引入透明板」：第 9 集把小臂想成一张很大的透明板；底座上看的数字照固定在底座上的网格算，
// 手上看的数字照贴在手上的一张透明板算（淡紫的板，画着手的两根轴方向的格子，跟着手一起动，E.handBoard）。
// 句 2「让肩继续转」肩逆时针一路转 20°（蓝色小弯箭头），「肘的轴被带着挪了地方」肘原来的地方留一个白虚线圈、肘走过的路白点线，空间雅可比第 2 列框亮、数在变；
// 句 3「第九集的透明板」：「固定在底座上的这张网格」网格四周白虚线框亮，「在手上贴了一张透明板」手上的板淡入；
// 句 4「让肩转回去」肩顺时针转回 20°（顺时针的小弯箭头），板连同胳膊一起转，「肩和肘在板上的位置」板上肩、肘各一个紫虚线圈，一直套在肩、肘上：物体雅可比两列打勾；
// 句 5「再让肘转」肘逆时针转 40°，「肩和肘在网格上都没有挪」网格上肩、肘各一个白虚线圈，一直套着：空间雅可比两列打勾；
//    「手上的板跟着小臂绕肘转」板绕肘转，「肩在板上挪了地方」板上肩的紫圈跟着板走开，肩在板上走过的路紫点线：物体雅可比第 1 列框亮、数在变；
// 句 6 右栏写两句对照；句 7 写 J_b = [Ad_Tbs] J_s。胳膊只朝小弯箭头所指的方向转（发布清单 013 的规矩）。
// 时刻取自配音的逐词时间，见 ../ep014.py。
(function () {
  var E = window.EP014, K = E.K, H = E.HEX;
  var X = 455, D90 = Math.PI / 2;
  var T = {
    pose: 0.7, change: 1.19, two: 1.6, recalc: 3.07, diff: 3.75,
    sh: 6.32, rot1: [6.8, 18.6], base: 7.86, elAx: 8.98, moved: 10.07, col2: 11.29, eo0: 13.64, changing: 17.43,
    ep9: 20.03, standB: 22.15, grid: 24.44, standH: 27.22, stick: 29.25, board: 29.78,
    back: 34.04, rot1b: [34.4, 43.0], together: 36.14, rings: 37.89, noChange: 42.09,
    el: 44.89, rot2: [45.2, 57.4], gridStill: 45.83, base2: 47.19, noChange2: 50.64, boardTurn: 51.84, shMoved: 54.06, col1: 56.78,
    s5: 58.77, front5: 60.18, hand5: 63.07, rev5: 64.6,
    always: 69.46, adj: 73.27
  };
  var END = 75.75, A1 = 0.35, A2 = 0.7;   // 肩转 20°、肘转 40°：胳膊一直在 x 轴上面
  var BR = [-4.4, 0.6, -2.5, 0.6];        // 手上的板在手的坐标里的范围（盖住肩、肘和肩在板上走过的路）

  E.scenes[4] = {
    title: '姿势变了：两种雅可比各自怎么变',
    dur: END,
    cues: [
      { at: 0, s: '姿势一变，**空间雅可比**、**物体雅可比**都要重算，但变的方式不一样' },
      { at: 6.22, s: '肩继续转：**底座上看**第 $2$ 列用肘此刻的位置算（肘关节指向原点、转 $90^\\circ$），肘一挪就变' },
      { at: 19.93, s: '第 9 集的透明板：**底座上看**照固定在底座上的网格算；**手上看**照贴在手上的透明板算' },
      { at: 33.31, s: '肩转回去：胳膊连同手上的板一起转，肩、肘在板上的位置不变——**手上看** $2$ 列一个数字都不变' },
      { at: 44.6, s: '肘转：肩、肘在网格上没挪——**底座上看**不变；肩在板上挪了——**手上看**第 $1$ 列在变' },
      { at: 58.67, s: '底座上看：前面的关节一转，后面的轴跟着挪；手上看：后面的关节一转，前面的轴跟着挪' },
      { at: 69.36, s: '不管姿势怎么变，两种雅可比之间，永远隔着一个**伴随变换**' }
    ],
    build: function (K) {
      var s = K.sceneSvg('姿势一变，两种雅可比都要重算，但变的方式不一样。让肩继续转：站在底座上看，肘的轴被带着挪了地方，空间雅可比的第二列一直在变。用第九集的透明板来想：底座上看的数字照固定在底座上的网格算，手上看的数字照贴在手上的一张透明板算。让肩转回去：胳膊连同板一起转，肩和肘在板上的位置不变，物体雅可比两列都不变。再让肘转：网格上肩和肘都没挪，空间雅可比两列不变；板跟着小臂绕肘转，肩在板上挪了地方，物体雅可比第一列在变。站在底座上看，前面的关节一转，后面的轴跟着挪；站在手上看，后面的关节一转，前面的轴跟着挪。不管姿势怎么变，两种雅可比之间永远隔着一个伴随变换');

      var pn = E.panel14(s), P = pn.P, u = pn.u, top = pn.top, O = P(0, 0), pl = pn.plane;
      // 底座上的网格（不动）：四周白虚线框 + 字
      var gb = pl.box, gridF = K.paint(E.el('rect', { x: gb[0] - 4, y: gb[1] - 4, width: gb[2] - gb[0] + 8, height: gb[3] - gb[1] + 8, rx: 6, fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '9 6' }, top), null, E.INK);
      var gridL = E.halo(E.txt(top, gb[0] + 4, gb[1] - 12, '固定在底座上的网格（不动）', 16, E.INK, 'start', 900));
      // 手上的透明板（画在网格上面、机械臂下面）
      var hb = E.handBoard(pn.under, P, BR);
      var hbL = E.halo(E.txt(top, 0, 0, '贴在手上的透明板', 16, E.V, 'end', 900));   // 写在板靠手 y 轴那一边的左边（不压坐标轴）
      // 圈：网格上（白）、板上（紫）关节原来的地方；点线：关节在网格上 / 板上走过的路（先画：两个坐标系的字母描了底色，压在圈上面也看得清）
      function ring(c) { return K.paint(E.el('circle', { r: 20, fill: 'none', 'stroke-width': 3.5, 'stroke-dasharray': '7 5' }, top), null, c); }
      function at(n, p) { n.setAttribute('cx', p[0].toFixed(1)); n.setAttribute('cy', p[1].toFixed(1)); }
      function trail(c) { return K.paint(E.el('path', { fill: 'none', 'stroke-width': 4, 'stroke-dasharray': '1 7', 'stroke-linecap': 'round' }, top), null, c); }
      var wE1 = ring(E.INK), wTr = trail(E.INK), wS2 = ring(E.INK), wE2 = ring(E.INK);
      var vS = ring(E.V), vE = ring(E.V), vTr = trail(E.V);
      var bF = E.frame2(top, P, 1, '底座', { labAt: [0.75, -0.62], tipOff: E.BASE_TIP }), hF = E.frame2(top, P, 1, '手', { labAt: [0.5, -0.45], labColor: E.V, tipOff: [[-0.15, 0.3], [-0.35, 0.1]] });   // 肘转完手的 y 轴朝左下，字母写在尖的左上（不压白色 x 轴）
      var jS = E.joint(top, E.HI), jE = E.joint(top, E.HI), hand = E.dot(top, E.V, 7);
      var eoA = E.gv(top, P, E.INK, 3.5, { head: 13, dash: '8 6' });   // 第 2 列怎么算：从肘关节指向原点（跟着肘一起挪）
      var sArc = E.arcArrow(top, O[0], O[1], 0.6 * u, E.OMEGA, 3.5, { head: 12 });
      var sArcB = E.arcArrow(top, O[0], O[1], 0.6 * u, E.OMEGA, 3.5, { head: 12 });
      var sLab = E.tex(top, O[0] - 132, O[1] + 46, '\\dot\\theta_1', 22, E.OMEGA, 'start', { w: 50, h: 36 });
      var eArc = E.arcArrow(top, 0, 0, 0.55 * u, E.OMEGA, 3.5, { head: 12 });
      var eLab = E.tex(top, 0, 0, '\\dot\\theta_2', 22, E.OMEGA, 'start', { w: 50, h: 36 });

      /* ══ 右栏：两个实时的 6 × 2 ══ */
      var RK = E.rowKit(), Rg = E.el('g', {}, s), MY = 150;
      var W = [0, 1, 2, 3, 4, 5].map(function () { return ['−2.22', '−2.22']; });
      function col(i, j) { return j === 0 ? E.J1 : E.J2; }
      var gM = E.el('g', {}, Rg);
      var Ms = E.matN(gM, X + 82, MY, W, { size: 17, dx: 54, dy: 24, wide: W, color: col });
      var Mb = E.matN(gM, X + 258, MY, W, { size: 17, dx: 54, dy: 24, wide: W, color: col });
      // 两个雅可比正上方：视角标签（全片同一个样式）；下面一行写名字
      var ls = E.viewTag(gM, X + 82, 22, 'base').g, lb = E.viewTag(gM, X + 258, 22, 'hand').g;
      var ns = E.txt(gM, X + 82, 70, '空间雅可比', 16, E.INK2, 'middle', 900), nb = E.txt(gM, X + 258, 70, '物体雅可比', 16, E.INK2, 'middle', 900);
      // 「不变」的勾：每个矩阵每列下面一个
      var oks = [Ms, Mb].map(function (M) { return [0, 1].map(function (j) { return E.check(gM, M.pos(5, j)[0] - 10, M.box[3] + 8, 20, E.GOOD, 3.5); }); });
      var thL = E.txt(gM, X, 274, '', 18, E.INK2, 'start', 700);
      var gS = E.el('g', {}, Rg);
      var s1 = E.tagRow(gS, RK, X, 318, 'base', [['前面的关节一转，后面的轴跟着挪', E.INK2]], 16);
      var s2 = E.tagRow(gS, RK, X, 350, 'hand', [['后面的关节一转，前面的轴跟着挪', E.INK2]], 16);
      var s3 = E.tex(gS, X, 392, '\\textcolor{' + H.violet + '}{J_b}=\\textcolor{' + H.hi + '}{[\\mathrm{Ad}_{T_{bs}}]}\\,J_s\\quad\\text{（永远成立）}', 21, E.INK, 'start', { w: 330, h: 40 });

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }
      // 姿势：肩 = 90° + A1 ×（句 2 转过去 − 句 4 转回来），肘 = A2 ×（句 5 转过的）
      function th(st) {
        var f = E.fade;
        return [D90 + A1 * (f(st, T.rot1[0], T.rot1[1] - T.rot1[0]) - f(st, T.rot1b[0], T.rot1b[1] - T.rot1b[0])), A2 * f(st, T.rot2[0], T.rot2[1] - T.rot2[0])];
      }
      // 第 13 集的两关节：肩在原点，大臂 √5 格，小臂 2 格；手的 x 轴沿小臂
      function joints(t1, t2) {
        var a1 = E.ZERO[0] + t1, a2 = t1 + t2, e = [E.L1 * Math.cos(a1), E.L1 * Math.sin(a1)];
        return [[0, 0], e, [e[0] + E.L2 * Math.cos(a2), e[1] + E.L2 * Math.sin(a2)], a2];
      }
      // 网格上一点在手的板上的坐标（手在 h、手的 x 轴方向角 a）
      function toBoard(p, h, a) { var c = Math.cos(a), s0 = Math.sin(a), dx = p[0] - h[0], dy = p[1] - h[1]; return [dx * c + dy * s0, -dx * s0 + dy * c]; }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) { RK.layout(); laid = true; }
          var f = E.fade, b = E.bump;
          var tt = th(st), th1 = tt[0], th2 = tt[1];
          pn.pose(E.ZERO[0] + th1, th1 + th2); pn.view.clearTint();
          var J = pn.joints(), pE = P(J[1][0], J[1][1]), pH = P(J[2][0], J[2][1]), a2 = th1 + th2;
          jS.at(O); jE.at(pE); hand.at(pH);
          eoA.at(J[1][0], J[1][1], -0.06 * J[1][0], -0.06 * J[1][1]); E.op(eoA.g, E.win(st, T.eo0, T.ep9 - 0.3, 0.3));
          bF.set(0, 0, 0); hF.set(J[2][0], J[2][1], a2);
          E.glow(jE, E.HI, E.win(st, T.elAx, T.ep9, 0.2) > 0.5);
          E.glow(bF.g, E.INK, b(st, T.base, 1.2) > 0.05 || b(st, T.standB, 1.2) > 0.05);
          E.glow(hF.g, E.V, b(st, T.standH, 1.2) > 0.05);

          // 句 2：肘在网格上走开（白圈留在原来的地方，白点线是走过的路）
          var j0 = joints(D90, 0), aE0 = Math.atan2(j0[1][1], j0[1][0]), aE = Math.atan2(J[1][1], J[1][0]);
          at(wE1, P(j0[1][0], j0[1][1])); E.op(wE1, E.win(st, T.moved - 0.2, T.back - 0.4, 0.3));
          var tr = []; for (var k = 0; k <= 24; k++) { var aa = aE0 + (aE - aE0) * k / 24; tr.push(P(E.L1 * Math.cos(aa), E.L1 * Math.sin(aa))); }
          wTr.setAttribute('d', 'M ' + tr.map(function (p) { return p[0].toFixed(1) + ' ' + p[1].toFixed(1); }).join(' L ')); E.op(wTr, E.win(st, T.moved - 0.2, T.back - 0.4, 0.3));
          // 句 3：网格（底座上的板）框亮；手上的板淡入，跟着手动
          E.op(gridF, E.win(st, T.grid, T.back - 0.4, 0.3) + E.win(st, T.gridStill, T.boardTurn - 0.2, 0.3));
          E.op(gridL, E.win(st, T.grid, T.back - 0.4, 0.3) + E.win(st, T.gridStill, T.boardTurn - 0.2, 0.3));
          hb.set(J[2][0], J[2][1], a2); E.op(hb.g, E.win(st, T.stick, T.s5 - 0.3, 0.4));
          var lp = hb.at(-1.2, 0.6); hbL.setAttribute('x', (lp[0] - 8).toFixed(1)); hbL.setAttribute('y', (lp[1] + 6).toFixed(1));
          E.op(hbL, E.win(st, T.board, T.s5 - 0.3, 0.3));
          // 句 4 起：板上肩、肘原来的地方（紫圈钉在板上）：肩转时一直套着；句 5 肘转时肩走开、紫点线是肩在板上走过的路
          var jP = joints(D90, 0), sB = toBoard([0, 0], jP[2], jP[3]), eB = toBoard(jP[1], jP[2], jP[3]);
          at(vS, hb.at(sB[0], sB[1])); at(vE, hb.at(eB[0], eB[1]));
          E.op(vS, E.win(st, T.rings, T.s5 - 0.3, 0.3)); E.op(vE, E.win(st, T.rings, T.s5 - 0.3, 0.3));
          var vt = []; for (k = 0; k <= 24; k++) { var jk = joints(D90, th2 * k / 24), q = toBoard([0, 0], jk[2], jk[3]); vt.push(hb.at(q[0], q[1])); }
          vTr.setAttribute('d', 'M ' + vt.map(function (p) { return p[0].toFixed(1) + ' ' + p[1].toFixed(1); }).join(' L ')); E.op(vTr, E.win(st, T.rot2[0], T.s5 - 0.3, 0.3));
          // 句 5：网格上肩、肘的白圈（不动）
          at(wS2, O); at(wE2, P(jP[1][0], jP[1][1]));
          E.op(wS2, E.win(st, T.base2, T.s5 - 0.3, 0.3)); E.op(wE2, E.win(st, T.base2, T.s5 - 0.3, 0.3));

          // 小弯箭头：句 2 肩逆时针，句 4 肩顺时针，句 5 肘逆时针
          sArc.set(3.45, 4.55); var on1 = E.win(st, T.sh, T.rot1[1] + 0.3, 0.3);
          sArcB.set(4.55, 3.45); var on1b = E.win(st, T.back, T.rot1b[1] + 0.3, 0.3);
          E.op(sArc.g, on1); E.op(sArcB.g, on1b); E.op(sLab, Math.max(on1, on1b));
          eArc.g.setAttribute('transform', 'translate(' + pE[0].toFixed(1) + ' ' + pE[1].toFixed(1) + ')');
          var ea = Math.atan2(J[1][1], J[1][0]) + Math.PI;   // 从肘指回肩的方向；小弯箭头画在它的两边各偏一点（不压大臂）
          eArc.set(ea + 0.5, ea + 1.6);
          eLab.setAttribute('transform', 'translate(' + (pE[0] - 66).toFixed(1) + ' ' + (pE[1] + 68).toFixed(1) + ')');   // 肘在 (−2, 1)：写在 x 轴下面（不压白色坐标轴）
          var on2 = E.win(st, T.el, T.rot2[1] + 0.3, 0.3);
          E.op(eArc.g, on2); E.op(eLab, on2);

          /* ── 右栏 ── */
          var jj = E.jac2(J, a2);
          Ms.set(jj.Js.map(function (r) { return r.map(function (x) { return E.num(x, 2); }); }));
          Mb.set(jj.Jb.map(function (r) { return r.map(function (x) { return E.num(x, 2); }); }));
          E.op(gM, fr(st, T.pose - 0.1));
          E.op(Ms.g, fr(st, T.two)); E.op(ls, fr(st, T.two)); E.op(ns, fr(st, T.two)); E.op(Mb.g, fr(st, T.two + 0.4)); E.op(lb, fr(st, T.two + 0.4)); E.op(nb, fr(st, T.two + 0.4));
          Ms.hl('c2', 0, 1, 5, 1, E.J2, E.win(st, T.col2, T.ep9 - 0.3, 0.3));
          Mb.hl('c1', 0, 0, 5, 0, E.J1, E.win(st, T.col1, T.s5, 0.3));
          // 勾：句 4 肩转回去时物体雅可比两列不变；句 5 肘转时空间雅可比两列不变（物体雅可比第 2 列也一直不变）
          E.op(oks[1][0], E.win(st, T.noChange, T.el, 0.3)); E.op(oks[1][1], E.win(st, T.noChange, T.s5, 0.3));
          E.op(oks[0][0], E.win(st, T.noChange2, T.s5, 0.3)); E.op(oks[0][1], E.win(st, T.noChange2, T.s5, 0.3));
          thL.textContent = '肩 θ₁ = ' + Math.round(th1 * 180 / Math.PI) + '°　肘 θ₂ = ' + Math.round(th2 * 180 / Math.PI) + '°';
          E.op(thL, fr(st, T.sh));
          E.op(gS, fr(st, T.s5 - 0.1));
          E.op(s1.row.g, fr(st, T.s5)); E.op(s1.tag, fr(st, T.s5)); E.op(s1.row.parts[0], fr(st, T.front5));
          E.op(s2.row.g, fr(st, T.hand5)); E.op(s2.tag, fr(st, T.hand5)); E.op(s2.row.parts[0], fr(st, T.rev5)); E.op(s3, fr(st, T.always));
          E.glow(s3, E.HI, b(st, T.adj, 1.2) > 0.05);
        }
      };
    }
  };
})();
