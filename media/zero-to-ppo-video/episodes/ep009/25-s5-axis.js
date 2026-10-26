// 009 第 5 幕：螺旋轴——关节就是螺旋轴。左边先是公式，再是肘关节的小图、两种关节的图标，最后是斜着看的 UR5 和它的六根关节轴；右栏放数和要点。
// 句 1「把转速单独拿出来」左边写 𝒱 = [ŝ θ̇; (−ŝ × q + h ŝ) θ̇]，「拿出来」下一行把 θ̇ 提到外面，「螺旋轴 S」矩阵下标「螺旋轴 S」；右栏「θ̇：转多快」「S：轴在什么位置、怎么转」；
// 句 2 左边下方出框：𝒱 = S θ̇（运动旋量 = 转速 × 螺旋轴）；
// 句 3 左边换成肘关节的小图（网格、原点、肘上 ⊙）：「朝向你」⊙ 鼓一下，「一、二」写 (1, 2)，「螺距是零」写 h = 0；
//    右栏 S 的六个数按念到的顺序弹出，「乘以转速二」出「× 2 =」，「运动旋量」右边一列 (0, 0, 2, 4, −2, 0) 出来、打勾；
// 句 4 左边换成两个关节图标：「转动关节」铰链绕销转（标 h = 0），「移动关节」滑块沿导轨来回走；右栏两行 S 的写法；
// 句 5 左边换成斜着看的 UR5，「六个」「转动关节」「六根螺旋轴」六根关节轴（琥珀虚线）依次画出、标 S₁…S₆；
// 句 6「第十集」右栏第 10 集卡片，「螺旋轴」「转一个角度」S₂ 亮起、整条胳膊绕它转一个角度，「停在什么位置」手上出坐标架；
//    「第十一集」第 11 集卡片，「串起来」六根轴依次鼓一下，「末端的位姿」手上的坐标架鼓一下。
// 时刻取自配音的逐词时间，见 ../ep009.py。
(function () {
  var E = window.EP009, K = E.K, H = E.HEX;
  var D = Math.PI / 180, RX = E.ARM.RX;
  var T = {
    f1: 0.62, out: 1.69, rest: 2.14, where: 3.55, how: 4.7, nameS: 5.8,
    big: 7.6, bigS: 9.6, out2: 11.1,
    elbow: 11.27, you: 12.96, q: 14.06, h0: 15.22, sNums: [16.53, 16.93, 17.29, 17.73, 18.16, 18.85], times: 19.35, two: 20.1, twist: 21.68, out3: 23.4,
    joints: 24.17, rev: 26.2, rev0: 27.58, pri: 28.26, priS: 29.85, pri0: 31.01, out4: 32.2,
    arm: 32.39, axes: [33.37, 33.69, 34.04, 34.62, 35.0, 35.37],
    ep10: 37.06, s2: 38.41, turn: [39.13, 40.6], where2: 41.31, ep11: 42.65, chain: 44.11, pose: 45.46
  };
  var END = 47.23;
  var JN = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'];

  E.scenes[4] = {
    title: '螺旋轴：关节就是螺旋轴',
    dur: END,
    cues: [
      { at: 0, s: '把转速 θ̇ 拿出来，剩下的六个数就是**螺旋轴** S' },
      { at: 7.51, s: '**运动旋量** = 转速 × **螺旋轴**：𝒱 = S θ̇' },
      { at: 11.18, s: '肘关节：S = (0, 0, 1, 2, −1, 0)，× 2 就是刚才的**运动旋量**' },
      { at: 23.49, s: '转动关节：**螺距**是 0；移动关节只移不转，前三个数是 0' },
      { at: 32.3, s: '六个转动关节 = 六根**螺旋轴**' },
      { at: 36.96, s: '第 10 集：绕**螺旋轴**转一个角度停在哪；第 11 集：串起来得到末端位姿' }
    ],
    build: function (K) {
      var s = K.sceneSvg('把转速单独拿出来，剩下的六个数只说轴在什么位置、怎么转，就是螺旋轴 S，运动旋量等于转速乘以螺旋轴；肘关节的螺旋轴是 (0, 0, 1, 2, −1, 0)，乘转速 2 就是刚才的运动旋量；转动关节螺距为零，移动关节只移不转、前三个数为零；UR5 有六个转动关节，就是六根螺旋轴；第十集讲绕螺旋轴转一个角度停在哪，第十一集把六个关节串起来得到末端位姿');

      /* ══ ① 公式 ══ */
      var fm = E.el('g', {}, s);
      var f1 = E.tex(fm, 30, 92, '\\mathcal{V}=\\begin{bmatrix}\\textcolor{' + H.violet + '}{\\omega}\\\\ \\textcolor{' + H.acc + '}{v}\\end{bmatrix}=\\begin{bmatrix}\\hat s\\,\\textcolor{' + H.good + '}{\\dot\\theta}\\\\ (-\\hat s\\times q+h\\,\\hat s)\\,\\textcolor{' + H.good + '}{\\dot\\theta}\\end{bmatrix}', 28, E.INK, 'start', { w: 400, h: 100 });
      var f2 = E.tex(fm, 104, 212, '=\\begin{bmatrix}\\hat s\\\\ -\\hat s\\times q+h\\,\\hat s\\end{bmatrix}\\textcolor{' + H.good + '}{\\dot\\theta}', 28, E.INK, 'start', { w: 320, h: 100 });
      var sBr = K.paint(E.el('path', { d: 'M 132 250 Q 132 262 144 262 H 214 Q 222 262 225 272 Q 228 262 236 262 H 306 Q 318 262 318 250', fill: 'none', 'stroke-width': 2.5 }, fm), null, E.WARN);
      var sName = E.txt(fm, 225, 298, '螺旋轴 S', 24, E.WARN, 'middle', 900);
      var tName = E.txt(fm, 352, 298, '转速', 22, E.GOOD, 'middle', 900);
      var bigG = E.el('g', {}, s);
      K.paint(E.el('rect', { x: 40, y: 318, width: 380, height: 92, rx: 16, 'stroke-width': 3 }, bigG), E.tint(E.HI, 10, 'var(--demo-surface)'), E.HI);
      E.tex(bigG, 130, 372, '\\mathcal{V}=\\textcolor{' + H.warn + '}{\\mathcal{S}}\\,\\textcolor{' + H.good + '}{\\dot\\theta}', 40, E.INK, 'middle', { w: 180, h: 70 });
      E.txt(bigG, 230, 360, '运动旋量 =', 22, E.HI, 'start', 900);
      E.txt(bigG, 230, 392, '螺旋轴 × 转速', 22, E.INK, 'start', 900);

      /* ══ ② 肘关节的小图 ══ */
      var mini = E.el('g', {}, s), MU = 64, MO = [120, 330];
      var mpl = E.plane(mini, { ox: MO[0], oy: MO[1], u: MU, x0: -1, x1: 3, y0: -1, y1: 3 });
      var mq = mpl.P(1, 2);
      var mOrig = K.paint(E.el('circle', { cx: MO[0], cy: MO[1], r: 6 }, mini), E.HI);
      E.halo(E.txt(mini, MO[0] - 12, MO[1] + 26, '原点', 20, E.HI, 'end', 900));
      var mOut = E.el('g', {}, mini);
      var mRing = K.paint(E.el('circle', { cx: mq[0], cy: mq[1], r: 15, 'stroke-width': 3.5 }, mOut), E.S1, E.VIOLET);
      K.paint(E.el('circle', { cx: mq[0], cy: mq[1], r: 5 }, mOut), E.VIOLET);
      var mYou = E.halo(E.txt(mini, mq[0] + 24, mq[1] - 18, '转轴朝向你', 20, E.VIOLET, 'start', 900));
      var mQ = E.halo(E.txt(mini, mq[0] + 24, mq[1] + 10, '穿过 (1, 2)', 20, E.HI, 'start', 900));
      var mH = E.halo(E.txt(mini, mq[0] + 24, mq[1] + 38, '螺距 h = 0', 20, E.GOOD, 'start', 900));
      var mSpin = E.arcArrow(mini, mq[0], mq[1], 30, E.VIOLET, 3.5, { head: 12 }); mSpin.set(115 * D, 205 * D);

      /* ══ ③ 两种关节 ══ */
      var jg = E.el('g', {}, s);
      // 转动关节：两根连杆 + 销（⊙）+ 弯箭头
      var revG = E.el('g', {}, jg), RC = [110, 200];
      var link1 = K.paint(E.el('rect', { x: RC[0] - 90, y: RC[1] - 14, width: 90, height: 28, rx: 14, 'stroke-width': 2.5 }, revG), E.tint(E.INK2, 30, 'var(--demo-surface)'), E.INK2);
      var link2 = E.el('g', {}, revG);
      K.paint(E.el('rect', { x: RC[0], y: RC[1] - 13, width: 92, height: 26, rx: 13, 'stroke-width': 2.5 }, link2), E.tint(E.VIOLET, 30, 'var(--demo-surface)'), E.VIOLET);
      K.paint(E.el('circle', { cx: RC[0], cy: RC[1], r: 13, 'stroke-width': 3 }, revG), E.S1, E.WARN);
      K.paint(E.el('circle', { cx: RC[0], cy: RC[1], r: 4 }, revG), E.WARN);
      var revSpin = E.arcArrow(revG, RC[0], RC[1], 44, E.VIOLET, 3.5, { head: 12 });
      E.txt(revG, RC[0], RC[1] + 92, '转动关节', 24, E.INK, 'middle', 900);
      var revH = E.txt(revG, RC[0], RC[1] + 124, '螺距 h = 0', 22, E.GOOD, 'middle', 900);
      // 移动关节：导轨 + 滑块 + 双向箭头
      var priG = E.el('g', {}, jg), PC = [330, 200];
      K.paint(E.el('rect', { x: PC[0] - 90, y: PC[1] + 12, width: 180, height: 14, rx: 4, 'stroke-width': 2 }, priG), E.tint(E.INK2, 30, 'var(--demo-surface)'), E.INK2);
      var slider = K.paint(E.el('rect', { width: 56, height: 36, rx: 8, 'stroke-width': 2.5 }, priG), E.tint(E.VIOLET, 30, 'var(--demo-surface)'), E.VIOLET);
      var priArr = E.vec(priG, E.GOOD, 4, { head: 13 });
      E.txt(priG, PC[0], PC[1] + 92, '移动关节', 24, E.INK, 'middle', 900);
      var priH = E.txt(priG, PC[0], PC[1] + 124, '只移不转', 22, E.GOOD, 'middle', 900);

      /* ══ ④ UR5 斜视 + 六根关节轴 ══ */
      var armG = E.el('g', {}, s);
      var ur = window.Robot3D.ur5(armG, 105, 404, 2.5, { az: 32, el: 16 }), V = ur.view, THREE = window.Robot3D.THREE;
      var POSE0 = { shoulder_pan_joint: 20 * D, shoulder_lift_joint: -95 * D, elbow_joint: 75 * D, wrist_1_joint: -70 * D, wrist_2_joint: -90 * D, wrist_3_joint: 0 };
      V.camera({ box: [0, 0, 440, 420] });
      V.joints(POSE0);
      var axG = E.el('g', {}, armG);
      var axLines = JN.map(function () { return K.paint(E.el('path', { fill: 'none', 'stroke-width': 3.5, 'stroke-dasharray': '8 5' }, axG), null, E.WARN); });
      var axLabs = JN.map(function (n, i) { return E.halo(E.txt(axG, 0, 0, 'S' + '₁₂₃₄₅₆'[i], 22, E.WARN, 'middle', 900)); });
      var toolG = E.el('g', {}, armG);
      var toolAx = [E.GOOD, E.WARN, E.ACC].map(function (c) { return E.vec(toolG, c, 4, { head: 12 }); });
      function axisOf(name) {
        var j = V.robot.joints[name];
        j.updateMatrixWorld(true);
        var o = new THREE.Vector3().setFromMatrixPosition(j.matrixWorld), d = j.axis.clone().transformDirection(j.matrixWorld);
        return { o: o, d: d };
      }
      // 标签放在机器人旁边的空处：[哪一端（0 / 1 / 'mid'）, dx, dy, 对齐]——S₁ 在大臂左边，S₃ 在肘左边，S₆ 在手的上方（手上要出坐标架）
      var LAB = [['mid', -26, 7, 'end'], [1, 16, 4, 'start'], [0, -24, 2, 'end'], [1, -8, -18, 'end'], [1, 12, 7, 'start'], [0, 0, -12, 'middle']];
      var LEN = [[-0.05, 0.3], [-0.22, 0.22], [-0.2, 0.2], [-0.16, 0.16], [-0.15, 0.15], [-0.14, 0.14]];   // 每根轴从关节往两边画多长（米）；S₁ 只往上画，往下就出了底座

      /* ══ 右栏 ══ */
      var Rg = E.el('g', {}, s);
      function rt(g, x, y, str, size, color, w, anchor) { var n = E.txt(g, x, y, str, size || 23, color || E.INK, anchor || 'start', w || 700); n.style.whiteSpace = 'pre'; return n; }
      var r1 = E.el('g', {}, Rg);
      var r1a = E.el('g', {}, r1);
      E.tex(r1a, RX + 10, 92, '\\textcolor{' + H.good + '}{\\dot\\theta}', 30, E.INK, 'start', { w: 40, h: 46 });
      rt(r1a, RX + 50, 92, '：转速（转多快）', 23, E.GOOD, 900);
      var r1b = E.el('g', {}, r1);
      E.tex(r1b, RX + 10, 150, '\\textcolor{' + H.warn + '}{\\mathcal{S}}', 30, E.INK, 'start', { w: 40, h: 46 });
      rt(r1b, RX + 50, 150, '：螺旋轴，六个数', 23, E.WARN, 900);
      var r1c = rt(r1, RX + 50, 190, '只说轴在什么位置、', 22, E.INK);
      var r1d = rt(r1, RX + 50, 226, '怎么转', 22, E.INK);
      var r2 = E.el('g', {}, Rg), SC = RX + 60, VC = RX + 250, CY = 220;
      var sCol = E.colvec(r2, SC, CY, [0, 0, 1, 2, -1, 0], { size: 28, dy: 36, colors: [E.WARN, E.WARN, E.WARN, E.WARN, E.WARN, E.WARN], w: 66 });
      var vCol = E.colvec(r2, VC, CY, [0, 0, 2, 4, -2, 0], { size: 28, dy: 36, colors: [E.VIOLET, E.VIOLET, E.VIOLET, E.ACC, E.ACC, E.ACC], w: 66 });
      var sTop = rt(r2, SC, 92, 'S', 26, E.WARN, 900, 'middle');
      var vTop = rt(r2, VC, 92, '运动旋量', 22, E.HI, 900, 'middle');
      var times = rt(r2, (SC + VC) / 2, CY + 10, '× 2 =', 28, E.GOOD, 900, 'middle');
      var ok = E.check(r2, VC + 46, 360, 24, E.GOOD, 4.5);
      var r3 = E.el('g', {}, Rg);
      var j1 = E.el('g', {}, r3);
      rt(j1, RX + 10, 80, '转动关节：', 23, E.INK, 900);
      E.tex(j1, RX + 10, 132, '\\mathcal{S}=\\begin{bmatrix}\\hat s\\\\ -\\hat s\\times q\\end{bmatrix}', 26, E.WARN, 'start', { w: 220, h: 80 });
      rt(j1, RX + 220, 132, '螺距 0', 22, E.GOOD, 900);
      var j2 = E.el('g', {}, r3);
      rt(j2, RX + 10, 236, '移动关节：', 23, E.INK, 900);
      E.tex(j2, RX + 10, 288, '\\mathcal{S}=\\begin{bmatrix}0\\\\ \\hat v\\end{bmatrix}', 26, E.WARN, 'start', { w: 160, h: 80 });
      var j2b = rt(j2, RX + 150, 280, '前三个数都是 0', 22, E.GOOD, 900);
      var j2c = rt(j2, RX + 150, 312, 'v̂：滑动的方向', 20, E.INK2, 700);
      var r4 = E.el('g', {}, Rg);
      var six = E.el('g', {}, r4);
      rt(six, RX + 10, 60, '这台机械臂：', 23, E.INK2, 700);
      rt(six, RX + 10, 98, '6 个转动关节 = 6 根螺旋轴', 23, E.WARN, 900);
      function card(g, y, ep, l1, l2, color) {
        var c = E.el('g', {}, g);
        K.paint(E.el('rect', { x: RX, y: y, width: 340, height: 100, rx: 14, 'stroke-width': 2.5 }, c), E.tint(color, 10, 'var(--demo-surface)'), color);
        rt(c, RX + 18, y + 38, ep, 22, color, 900);
        rt(c, RX + 18, y + 76, l1, 22, E.INK, 900);
        if (l2) rt(c, RX + 18, y + 108, l2, 22, E.INK, 900);
        return c;
      }
      var c10 = card(r4, 134, '第 10 集 · 矩阵指数', '绕螺旋轴转一个角度 → 停在哪', null, E.ACC);
      var c11 = card(r4, 254, '第 11 集 · 正运动学', '六个关节串起来 → 末端位姿', null, E.GOOD);

      return {
        el: s,
        draw: function (st, t) {
          var f = E.fade, b = E.bump;

          /* ── ① 公式 ── */
          E.op(fm, f(st, T.f1 - 0.4, 0.4) * (1 - f(st, T.out2, 0.4)));
          E.op(f1, f(st, T.f1, 0.3)); E.op(f2, f(st, T.out, 0.4));
          E.op(sBr, f(st, T.nameS, 0.3)); E.op(sName, f(st, T.nameS, 0.3)); E.op(tName, f(st, T.out + 0.3, 0.3));
          E.pop(bigG, 230, 364, f(st, T.big, 0.4) * (1 - f(st, T.out2, 0.4)), 0.9);

          /* ── ② 肘关节小图 ── */
          E.op(mini, f(st, T.elbow, 0.4) * (1 - f(st, T.out3, 0.4)));
          mRing.setAttribute('r', (15 + 5 * b(st, T.you, 0.8)).toFixed(2));
          E.op(mYou, f(st, T.you, 0.3)); E.op(mQ, f(st, T.q, 0.3)); E.op(mH, f(st, T.h0, 0.3));

          /* ── ③ 两种关节 ── */
          E.op(jg, f(st, T.joints, 0.4) * (1 - f(st, T.out4, 0.4)));
          var ra = 0.5 * Math.sin(2 * Math.PI * t / 3.2);
          link2.setAttribute('transform', 'rotate(' + (-ra / D).toFixed(2) + ' ' + RC[0] + ' ' + RC[1] + ')');
          revSpin.set(-20 * D, 60 * D);
          E.op(revSpin.g, f(st, T.rev, 0.3)); E.op(revH, f(st, T.rev0, 0.3));
          var sx = PC[0] - 28 + 52 * Math.sin(2 * Math.PI * t / 3.2);
          slider.setAttribute('x', sx.toFixed(1)); slider.setAttribute('y', PC[1] - 26);
          var pa = f(st, T.pri, 0.4);
          priArr.set(PC[0] - 60, PC[1] - 48, PC[0] - 60 + 120 * pa, PC[1] - 48); E.op(priArr.g, pa > 0.01 ? 1 : 0);
          E.op(priH, f(st, T.pri + 0.6, 0.3));

          /* ── ④ UR5 ── */
          E.op(armG, f(st, T.arm, 0.4));
          var turn = 28 * D * f(st, T.turn[0], T.turn[1] - T.turn[0]);
          var pose = Object.assign({}, POSE0);
          pose.shoulder_lift_joint = POSE0.shoulder_lift_joint + turn;
          V.joints(pose);
          var hiS2 = f(st, T.s2, 0.3);
          JN.forEach(function (n, i) {
            var a = axisOf(n), L = LEN[i];
            var p0 = V.project(a.o.clone().addScaledVector(a.d, L[0])), p1 = V.project(a.o.clone().addScaledVector(a.d, L[1]));
            axLines[i].setAttribute('d', 'M ' + p0[0].toFixed(1) + ' ' + p0[1].toFixed(1) + ' L ' + p1[0].toFixed(1) + ' ' + p1[1].toFixed(1));
            var on = f(st, T.axes[i], 0.35);
            E.op(axLines[i], on); E.op(axLabs[i], on);
            var lp = LAB[i], at = lp[0] === 'mid' ? [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2] : lp[0] ? p1 : p0;
            axLabs[i].setAttribute('x', (at[0] + lp[1]).toFixed(1)); axLabs[i].setAttribute('y', (at[1] + lp[2]).toFixed(1)); axLabs[i].style.textAnchor = lp[3];
            var pulse = Math.max(b(st, T.axes[i], 0.6), b(st, T.chain + 0.18 * i, 0.5), i === 1 ? hiS2 * (1 - f(st, T.where2 + 0.6, 0.4)) : 0);
            axLines[i].style.strokeWidth = 3.5 + 3 * pulse; E.glow(axLines[i], E.WARN, pulse > 0.05);
          });
          // 手上的坐标架（tool0 的三根轴）
          var tl = V.robot.links.tool0 || V.robot.frames && V.robot.frames.tool0;
          V.robot.updateMatrixWorld(true);
          var to = new THREE.Vector3().setFromMatrixPosition(tl.matrixWorld), op0 = V.project(to);
          [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)].forEach(function (ax, i) {
            var tp = V.project(to.clone().addScaledVector(ax.transformDirection(tl.matrixWorld), 0.13));
            toolAx[i].set(op0[0], op0[1], tp[0], tp[1]);
          });
          var tb = b(st, T.pose, 0.9);
          E.op(toolG, f(st, T.where2, 0.4)); E.glow(toolG, E.HI, tb > 0.05);
          toolAx.forEach(function (a) { a.line.style.strokeWidth = 4 + 2 * tb; });

          /* ── 右栏 ── */
          E.op(r1, 1 - f(st, T.out2, 0.4));
          E.op(r1a, f(st, T.out + 0.2, 0.3)); E.op(r1b, f(st, T.nameS, 0.3)); E.op(r1c, f(st, T.where, 0.3)); E.op(r1d, f(st, T.how, 0.3));
          E.op(r2, f(st, T.elbow + 0.3, 0.4) * (1 - f(st, T.out3, 0.4)));
          E.op(sTop, 1);
          sCol.cells.forEach(function (c, i) { var q = sCol.pos(i), pc = f(st, T.sNums[i], 0.3); E.op(c, pc); E.scaleAt(c, q[0], q[1], 1 + 0.5 * (1 - pc)); });
          E.op(times, f(st, T.times, 0.3));
          E.op(vCol.g, f(st, T.two, 0.4)); E.op(vTop, f(st, T.twist, 0.3)); E.drawOn(ok, f(st, T.twist + 0.3, 0.4));
          E.op(r3, f(st, T.joints, 0.4) * (1 - f(st, T.out4, 0.4)));
          E.op(j1, f(st, T.rev, 0.3)); E.op(j2, f(st, T.pri, 0.3)); E.op(j2b, f(st, T.priS, 0.3)); E.op(j2c, f(st, T.pri0, 0.3));
          E.op(r4, f(st, T.arm, 0.4));
          E.op(six, f(st, T.axes[0], 0.3));
          E.pop(c10, RX + 170, 184, f(st, T.ep10, 0.35), 0.92); E.pop(c11, RX + 170, 304, f(st, T.ep11, 0.35), 0.92);
        }
      };
    }
  };
})();
