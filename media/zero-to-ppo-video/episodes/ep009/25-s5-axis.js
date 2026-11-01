// 009 第 5 幕：螺旋轴——关节就是螺旋轴。左边先是公式，再是肘关节的小图、两种关节的图标，最后是斜着看的 UR5 和它的六根关节轴；右栏放数和要点。
// 句 1（2026-10-30 补，用户反馈「第五幕开头解释一下矩阵中各参数是什么」）左边写 𝒱 = [ω; v]，右栏按旁白一行行写出符号表：
//    「轴的方向」ŝ（长度为 1 的箭头），「轴上的一个点」q，「螺距」h，「转速」θ̇（每秒转多少弧度）；
// 句 2（同上，展开代入的步骤）「角速度就是轴的方向乘以转速」左边写 ω = ŝ θ̇，「代进 v」v = −ω × q + h ω → = −ŝ θ̇ × q + h ŝ θ̇ → = (−ŝ × q + h ŝ) θ̇，
//    「六个数」𝒱 右边接出 = [ŝ θ̇; (−ŝ × q + h ŝ) θ̇]，「都带着转速」它鼓一下；
// 句 3「拿出来」推导的几行淡出，下一行把 θ̇ 提到外面，「螺旋轴 S」矩阵下标「螺旋轴 S」；右栏「S：螺旋轴，六个数，只说轴在什么位置、怎么转」；
// 句 4 左边下方出框：𝒱 = S θ̇（运动旋量 = 转速 × 螺旋轴）；
// 句 5–8 肘关节，逐步算出 S，一步不跳（用户反馈「全篇的计算都展开讲不要跳过」）：左边换成肘关节的小网格（原点、肘上 ⊙），下面两行写算式：
//    「朝向你」⊙ 鼓一下，「零、零、一」右栏 S 的前三个数弹出、旁注「方向 ŝ」；
//    「后三个数」「穿过一、二」写 (1, 2)，「从这一点指到原点」肘 → 原点画黄箭头，「零减一、零减二」左下写 [0; 0] − [1; 2] = [0 − 1; 0 − 2]，
//    「是负一、负二」= [−1; −2]，箭头平移到原点（尾巴在原点）、写 (−1, −2)；
//    「转九十度，和前面一样」左下第二行 R [−1; −2]，「第一列取负一份」= (−1)[0; 1]、网格上从原点往下画红的一段，「第二列取负二份」+ (−2)[−1; 0]、接着往右画绿的一段，
//    「加起来」黄箭头绕原点逆时针转 90°（蓝弧）落在绿的尖上，「是二、负一」= [2; −1]、写 (2, −1)，S 的第 4、5 个数弹出、旁注「−ŝ × q」；
//    「螺距是零」写 h = 0，「零乘方向还是零」旁注「+ h ŝ = 0 × ŝ = 0」，「所以后三个数是二、负一、零」S 的第 6 个数弹出；
//    「螺旋轴就是……」S 的六个数逐个鼓一下，「乘以转速二」出「× 2 =」，「零还是零」右边一列的三个 0 弹出，
//    「一乘二是二」「二乘二是四」「负一乘二是负二」S 的那一个数鼓一下、右边对应的数弹出，「运动旋量」打勾；
// 句 8 左边换成两个关节图标：「转动关节」铰链绕销转（标 h = 0），「移动关节」滑块沿导轨来回走；右栏两行 S 的写法；
// 句 9 左边换成斜着看的 UR5，「六个」「转动关节」「六根螺旋轴」六根关节轴（琥珀虚线）依次画出、标 S₁…S₆；
// 句 10「第十集」右栏第 10 集卡片，「螺旋轴」「转一个角度」S₂ 亮起、整条胳膊绕它转一个角度，「停在什么位置」手上出坐标架；
//    「第十一集」第 11 集卡片，「串起来」六根轴依次鼓一下，「末端的位姿」手上的坐标架鼓一下。
// 时刻取自配音的逐词时间，见 ../ep009.py。
(function () {
  var E = window.EP009, K = E.K, H = E.HEX;
  var D = Math.PI / 180, RX = E.ARM.RX;
  var T = {
    f1: 0.62, sym: [2.61, 5.43, 6.58, 7.24], symLen: 3.88, symRad: 8.47,
    w: 10.87, wEq: 11.58, wPulse: 12.97, sub: 13.89, subV: 14.47, sub2: 15.1, six: 15.47, carry: 16.13,
    dOut: 18.2, out: 19.22, where: 21.02, nameS: 23.82,
    big: 25.75, bigS: 27.75, out2: 29.3,
    elbow: 29.42, you: 30.65, sNums: [32.49, 32.93, 33.39, 49.95, 50.46, 57.87], front3: 34.9,
    back3: 36.84, q: 37.73, arr: [39.82, 40.3], sub: [40.81, 41.61], neg: 42.52, slide: [42.6, 43.2],
    same: 44.66, take1: 46.51, take2: 47.79, rot: [49.03, 49.8], pos: 49.76,
    h0: 52.3, hs0: 53.05, back3b: 56.74,
    again: [60.31, 60.83, 61.34, 61.86, 62.32, 62.93], times: 63.94, zeros: 65.42, mul: [66.41, 67.37, 68.58], got: [66.98, 68.1, 69.35], twist: 71.07, out3: 72.82,
    joints: 73.5, rev: 75.53, rev0: 76.91, pri: 77.59, priS: 79.18, pri0: 80.34, out4: 81.53,
    arm: 81.72, axes: [82.7, 83.02, 83.37, 83.95, 84.33, 84.7],
    ep10: 86.39, s2: 87.74, turn: [88.46, 89.93], where2: 90.64, ep11: 91.98, chain: 93.44, pose: 94.79
  };
  var END = 96.56;
  var JN = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'];

  E.scenes[4] = {
    title: '螺旋轴：关节就是螺旋轴',
    dur: END,
    cues: [
      { at: 0, s: '符号：ŝ 轴的方向（长度为 1）、q 轴上一点、h **螺距**、θ̇ 转速' },
      { at: 10.78, s: '**角速度** ω = ŝ θ̇，代进 **v** = −ω × q + h ω：六个数都带着转速 θ̇' },
      { at: 18.15, s: '把转速 θ̇ 拿出来，剩下的六个数就是**螺旋轴** S' },
      { at: 25.66, s: '**运动旋量** = 转速 × **螺旋轴**：𝒱 = S θ̇' },
      { at: 29.33, s: '肘关节：方向 ŝ = (0, 0, 1)，是 S 的前三个数' },
      { at: 36.75, s: '后三个数：(1, 2) 指到原点 = (0 − 1, 0 − 2) = (−1, −2)' },
      { at: 44.57, s: '转 90°：第一列取 −1 份 + 第二列取 −2 份 = (0, −1) + (2, 0) = (2, −1)' },
      { at: 51.99, s: '**螺距**是 0：h ŝ = 0 × ŝ = 0，后三个数是 (2, −1, 0)' },
      { at: 59.19, s: 'S = (0, 0, 1, 2, −1, 0)，每个数 × 2：(0, 0, 2, 4, −2, 0) 就是前面的**运动旋量**' },
      { at: 72.82, s: '转动关节：**螺距**是 0；移动关节只移不转，前三个数是 0' },
      { at: 81.63, s: '六个转动关节 = 六根**螺旋轴**' },
      { at: 86.29, s: '第 10 集：绕**螺旋轴**转一个角度停在哪；第 11 集：串起来得到末端位姿' }
    ],
    build: function (K) {
      var s = K.sceneSvg('式子里的符号：ŝ 是轴的方向（长度为一的箭头），q 是轴上一点，h 是螺距，θ̇ 是转速；角速度 ω = ŝ θ̇，代进 v = −ω × q + h ω 得 (−ŝ × q + h ŝ) θ̇，六个数都带着转速；把转速单独拿出来，剩下的六个数只说轴在什么位置、怎么转，就是螺旋轴 S，运动旋量等于转速乘以螺旋轴；肘关节：方向 (0, 0, 1) 是 S 的前三个数，从 (1, 2) 指到原点是 (0 − 1, 0 − 2) = (−1, −2)，转 90° 按列算：(−1) × (0, 1) + (−2) × (−1, 0) = (2, −1)，螺距为零、h ŝ = 0，所以 S = (0, 0, 1, 2, −1, 0)，每个数乘转速 2 得 (0, 0, 2, 4, −2, 0)，就是前面的运动旋量；转动关节螺距为零，移动关节只移不转、前三个数为零；UR5 有六个转动关节，就是六根螺旋轴；第十集讲绕螺旋轴转一个角度停在哪，第十一集把六个关节串起来得到末端位姿');

      /* ══ ① 公式：先 𝒱 = [ω; v]，句 2 推出右半边，句 3 把 θ̇ 提出来（θ̇ 用蓝：它是角速度的大小；绿留给螺距 h） ══ */
      var fm = E.el('g', {}, s), TH = '\\textcolor{' + H.acc + '}{\\dot\\theta}', OM = '\\textcolor{' + H.acc + '}{\\omega}';
      var f1 = E.tex(fm, 30, 92, '\\mathcal{V}=\\begin{bmatrix}' + OM + '\\\\ \\textcolor{' + H.violet + '}{v}\\end{bmatrix}', 28, E.INK, 'start', { w: 120, h: 100 });
      var f1b = E.tex(fm, 152, 92, '=\\begin{bmatrix}\\hat s\\,' + TH + '\\\\ (-\\hat s\\times q+h\\,\\hat s)\\,' + TH + '\\end{bmatrix}', 28, E.INK, 'start', { w: 290, h: 100 });
      var der = E.el('g', {}, fm);                              // 句 2 的推导：ω = ŝ θ̇ 代进 v
      var d1 = E.tex(der, 30, 186, OM + '=\\hat s\\,' + TH, 28, E.INK, 'start', { w: 160, h: 50 });
      var d2 = E.tex(der, 30, 244, '\\textcolor{' + H.violet + '}{v}=-' + OM + '\\times q+h\\,' + OM, 28, E.INK, 'start', { w: 260, h: 50 });
      var d3 = E.tex(der, 54, 298, '=-\\hat s\\,' + TH + '\\times q+h\\,\\hat s\\,' + TH, 28, E.INK, 'start', { w: 300, h: 50 });
      var d4 = E.tex(der, 54, 352, '=(-\\hat s\\times q+h\\,\\hat s)\\,' + TH, 28, E.INK, 'start', { w: 280, h: 50 });
      var f2 = E.tex(fm, 104, 212, '=\\begin{bmatrix}\\hat s\\\\ -\\hat s\\times q+h\\,\\hat s\\end{bmatrix}' + TH, 28, E.INK, 'start', { w: 320, h: 100 });
      var sBr = K.paint(E.el('path', { d: 'M 132 250 Q 132 262 144 262 H 214 Q 222 262 225 272 Q 228 262 236 262 H 306 Q 318 262 318 250', fill: 'none', 'stroke-width': 2.5 }, fm), null, E.WARN);
      var sName = E.txt(fm, 225, 298, '螺旋轴 S', 24, E.WARN, 'middle', 900);
      var tName = E.txt(fm, 352, 298, '转速', 22, E.OMEGA, 'middle', 900);
      var bigG = E.el('g', {}, s);
      K.paint(E.el('rect', { x: 40, y: 318, width: 380, height: 92, rx: 16, 'stroke-width': 3 }, bigG), E.tint(E.HI, 10, 'var(--demo-surface)'), E.HI);
      E.tex(bigG, 130, 372, '\\mathcal{V}=\\textcolor{' + H.warn + '}{\\mathcal{S}}\\,' + TH, 40, E.INK, 'middle', { w: 180, h: 70 });
      E.txt(bigG, 230, 360, '运动旋量 =', 22, E.HI, 'start', 900);
      E.txt(bigG, 230, 392, '螺旋轴 × 转速', 22, E.INK, 'start', 900);

      /* ══ ② 肘关节的小网格：x −2…3、y −2…3（装得下平移到原点的 (−1, −2)）；下面两行写算式 ══ */
      var mini = E.el('g', {}, s), MU = 44, MO = [128, 172];
      var mpl = E.plane(mini, { ox: MO[0], oy: MO[1], u: MU, x0: -2, x1: 3, y0: -2, y1: 3 });
      var MP = mpl.P, mq = MP(1, 2);
      var mOrig = K.paint(E.el('circle', { cx: MO[0], cy: MO[1], r: 6 }, mini), E.HI);
      E.halo(E.txt(mini, MO[0] - 12, MO[1] - 10, '原点', 20, E.HI, 'end', 900));   // 左上：左下、右下要画箭头
      var mOut = E.el('g', {}, mini);
      var mRing = K.paint(E.el('circle', { cx: mq[0], cy: mq[1], r: 15, 'stroke-width': 3.5 }, mOut), E.S1, E.OMEGA);
      K.paint(E.el('circle', { cx: mq[0], cy: mq[1], r: 5 }, mOut), E.OMEGA);
      var mYou = E.halo(E.txt(mini, mq[0] + 24, mq[1] - 18, '转轴朝向你', 20, E.OMEGA, 'start', 900));
      var mQ = E.halo(E.txt(mini, mq[0] + 24, mq[1] + 10, '穿过 (1, 2)', 20, E.HI, 'start', 900));
      var mH = E.halo(E.txt(mini, mq[0] + 24, mq[1] + 38, '螺距 h = 0', 20, E.GOOD, 'start', 900));
      var mSpin = E.arcArrow(mini, mq[0], mq[1], 30, E.OMEGA, 3.5, { head: 12 }); mSpin.set(115 * D, 205 * D);
      // 句 6：肘 → 原点的黄箭头，平移到原点后绕原点逆时针转 90°（同第 3 幕，只是不再乘角速度：ŝ 长度是 1）
      var mTurn = E.arcArrow(mini, MO[0], MO[1], 0.62 * MU, E.OMEGA, 3, { head: 11 });
      var mA = E.vec(mini, E.X, 4, { head: 12 }), mB = E.vec(mini, E.Y, 4, { head: 12 });   // 第一列取 −1 份（红）、第二列取 −2 份（绿），头尾相接
      var mArr = E.vec(mini, E.HI, 4.5, { head: 15 });
      var mArrLab = E.halo(E.txt(mini, 0, 0, '', 20, E.HI, 'start', 900));
      // 两行算式（KaTeX）：原点 − 肘；R 乘它，按列取份
      var C = E.col, cn = E.cnum, HX = H.x, HY = H.y;
      function mx(y, tex) { return E.tex(mini, 30, y, tex, 22, E.INK, 'start', { w: 160, h: 70 }); }
      var m1a = mx(318, C([0, 0]) + '-' + C([1, 2], H.hi)), m1b = mx(318, '=\\begin{bmatrix}0-1\\\\ 0-2\\end{bmatrix}'), m1c = mx(318, '=' + C([-1, -2]));
      var m2a = mx(388, 'R' + C([-1, -2], [HX, HY])), m2b = mx(388, '=' + cn('(-1)', HX) + C([0, 1], HX)), m2c = mx(388, '+' + cn('(-2)', HY) + C([-1, 0], HY)), m2d = mx(388, '=' + C([2, -1], H.hi));
      var flows = [[m1a, m1b, m1c], [m2a, m2b, m2c, m2d]];
      function layoutFlows() {                                // 量出每段公式的实际宽度，挨着排（每帧量：KaTeX 的字体晚到也排得对）
        flows.forEach(function (fl) {
          var x = 30;
          fl.forEach(function (fo) {
            var r = fo.getBoundingClientRect(), k = r.width / +fo.getAttribute('width') || 1, c = fo.texBox.firstElementChild;
            fo.setAttribute('x', x.toFixed(1)); x += (c ? c.getBoundingClientRect().width / k : 0) + 5;
          });
        });
      }

      /* ══ ③ 两种关节 ══ */
      var jg = E.el('g', {}, s);
      // 转动关节：两根连杆 + 销（⊙）+ 弯箭头
      var revG = E.el('g', {}, jg), RC = [110, 200];
      var link1 = K.paint(E.el('rect', { x: RC[0] - 90, y: RC[1] - 14, width: 90, height: 28, rx: 14, 'stroke-width': 2.5 }, revG), E.tint(E.INK2, 30, 'var(--demo-surface)'), E.INK2);
      var link2 = E.el('g', {}, revG);
      K.paint(E.el('rect', { x: RC[0], y: RC[1] - 13, width: 92, height: 26, rx: 13, 'stroke-width': 2.5 }, link2), E.tint(E.BODY, 30, 'var(--demo-surface)'), E.BODY);
      K.paint(E.el('circle', { cx: RC[0], cy: RC[1], r: 13, 'stroke-width': 3 }, revG), E.S1, E.WARN);
      K.paint(E.el('circle', { cx: RC[0], cy: RC[1], r: 4 }, revG), E.WARN);
      var revSpin = E.arcArrow(revG, RC[0], RC[1], 44, E.OMEGA, 3.5, { head: 12 });
      E.txt(revG, RC[0], RC[1] + 92, '转动关节', 24, E.INK, 'middle', 900);
      var revH = E.txt(revG, RC[0], RC[1] + 124, '螺距 h = 0', 22, E.GOOD, 'middle', 900);
      // 移动关节：导轨 + 滑块 + 双向箭头
      var priG = E.el('g', {}, jg), PC = [330, 200];
      K.paint(E.el('rect', { x: PC[0] - 90, y: PC[1] + 12, width: 180, height: 14, rx: 4, 'stroke-width': 2 }, priG), E.tint(E.INK2, 30, 'var(--demo-surface)'), E.INK2);
      var slider = K.paint(E.el('rect', { width: 56, height: 36, rx: 8, 'stroke-width': 2.5 }, priG), E.tint(E.BODY, 30, 'var(--demo-surface)'), E.BODY);
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
      var toolAx = [E.X, E.Y, E.Z].map(function (c) { return E.vec(toolG, c, 4, { head: 12 }); });
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
      // 句 1 符号表：符号上色（ŝ 琥珀、q 黄、h 绿、θ̇ 蓝），解释照常
      function symRow(y, tex, color, str) {
        var g = E.el('g', {}, r1);
        E.tex(g, RX + 10, y, tex, 30, color, 'start', { w: 40, h: 46 });
        rt(g, RX + 50, y, str, 23, E.INK, 900);
        return g;
      }
      var y1 = symRow(70, '\\hat s', E.WARN, '：轴的方向'), y1b = rt(r1, RX + 50, 102, '长度为 1 的箭头', 19, E.INK2);
      var y2 = symRow(150, 'q', E.HI, '：轴上一点');
      var y3 = symRow(198, 'h', E.GOOD, '：螺距');
      var y4 = symRow(246, '\\dot\\theta', E.OMEGA, '：转速'), y4b = rt(r1, RX + 50, 278, '每秒转多少弧度', 19, E.INK2);
      var r1b = E.el('g', {}, r1);
      E.tex(r1b, RX + 10, 336, '\\textcolor{' + H.warn + '}{\\mathcal{S}}', 30, E.INK, 'start', { w: 40, h: 46 });
      rt(r1b, RX + 50, 336, '：螺旋轴，六个数', 23, E.WARN, 900);
      var r1c = rt(r1, RX + 50, 372, '只说轴在什么位置、怎么转', 20, E.INK);
      var r2 = E.el('g', {}, Rg), SC = RX + 60, VC = RX + 250, CY = 220;
      var sCol = E.colvec(r2, SC, CY, [0, 0, 1, 2, -1, 0], { size: 28, dy: 36, colors: [E.WARN, E.WARN, E.WARN, E.WARN, E.WARN, E.WARN], w: 66 });
      var vCol = E.colvec(r2, VC, CY, [0, 0, 2, 4, -2, 0], { size: 28, dy: 36, colors: [E.OMEGA, E.OMEGA, E.OMEGA, E.VEL, E.VEL, E.VEL], w: 66 });
      var sTop = rt(r2, SC, 92, 'S', 26, E.WARN, 900, 'middle');
      var vTop = rt(r2, VC, 92, '运动旋量', 22, E.HI, 900, 'middle');
      var times = rt(r2, (SC + VC) / 2, CY + 10, '× 2 =', 28, E.OMEGA, 900, 'middle');
      var ok = E.check(r2, VC + 46, 360, 24, E.GOOD, 4.5);
      // S 旁边的注：前三个数是方向 ŝ，后三个数是 −ŝ × q + h ŝ（「× 2 =」出来前收掉，腾地方）
      var nTop = rt(r2, SC + 44, sCol.pos(1)[1] + 7, '← 方向 ŝ', 20, E.WARN, 900);
      var nBot = rt(r2, SC + 44, sCol.pos(4)[1] - 6, '← −ŝ × q', 20, E.HI, 900);
      var nBot2 = rt(r2, SC + 70, sCol.pos(4)[1] + 24, '+ h ŝ = 0 × ŝ = 0', 20, E.GOOD, 900);
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
          E.op(f1, f(st, T.f1, 0.3));
          var cb = b(st, T.carry, 0.9);
          E.op(f1b, f(st, T.six, 0.4)); E.scaleAt(f1b, 297, 92, 1 + 0.06 * cb);
          E.op(der, 1 - f(st, T.dOut, 0.4));
          E.op(d1, f(st, T.wEq, 0.3)); E.scaleAt(d1, 100, 180, 1 + 0.08 * b(st, T.wPulse, 0.8));
          E.op(d2, f(st, T.sub, 0.3)); E.op(d3, f(st, T.subV, 0.3)); E.op(d4, f(st, T.sub2, 0.3));
          E.op(f2, f(st, T.out, 0.4));
          E.op(sBr, f(st, T.nameS, 0.3)); E.op(sName, f(st, T.nameS, 0.3)); E.op(tName, f(st, T.out + 0.3, 0.3));
          E.pop(bigG, 230, 364, f(st, T.big, 0.4) * (1 - f(st, T.out2, 0.4)), 0.9);

          /* ── ② 肘关节小网格 ── */
          E.op(mini, f(st, T.elbow, 0.4) * (1 - f(st, T.out3, 0.4)));
          mRing.setAttribute('r', (15 + 5 * b(st, T.you, 0.8)).toFixed(2));
          E.op(mYou, f(st, T.you, 0.3)); E.op(mQ, f(st, T.q, 0.3)); E.op(mH, f(st, T.h0, 0.3));
          // 肘 → 原点（画出来）→ 平移到原点（尾巴在原点，指到 (−1, −2)）→ 绕原点逆时针转 90° 到 (2, −1)
          if (s.isConnected) layoutFlows();
          E.op(m1a, f(st, T.sub[0], 0.3)); E.op(m1b, f(st, T.sub[0] + 0.3, 0.3)); E.op(m1c, f(st, T.neg, 0.3));
          E.op(m2a, f(st, T.same, 0.3)); E.op(m2b, f(st, T.take1, 0.3)); E.op(m2c, f(st, T.take2, 0.3)); E.op(m2d, f(st, T.pos, 0.3));
          var ma = f(st, T.take1, 0.5), mb = f(st, T.take2, 0.5), ma1 = MP(0, -ma), mb1 = MP(2 * mb, -1);
          mA.set(MO[0], MO[1], ma1[0], ma1[1]); E.op(mA.g, ma > 0.01 ? 1 : 0);
          mB.set(ma1[0], ma1[1], mb1[0], mb1[1]); E.op(mB.g, mb > 0.01 ? 1 : 0);
          var ga = f(st, T.arr[0], T.arr[1] - T.arr[0]), gs = f(st, T.slide[0], T.slide[1] - T.slide[0]), gr = f(st, T.rot[0], T.rot[1] - T.rot[0]);
          var ra2 = Math.atan2(-2, -1) + Math.PI / 2 * gr, rv = [Math.sqrt(5) * Math.cos(ra2), Math.sqrt(5) * Math.sin(ra2)];
          var tw = gs > 0 ? [1 - gs, 2 - 2 * gs] : [1, 2], hw = gs > 0 ? [tw[0] + rv[0], tw[1] + rv[1]] : [1 - ga, 2 - 2 * ga];
          var mt0 = MP(tw[0], tw[1]), mt1 = MP(hw[0], hw[1]);
          mArr.set(mt0[0], mt0[1], mt1[0], mt1[1]); E.op(mArr.g, ga > 0.01 ? 1 : 0);
          mTurn.set(Math.atan2(-2, -1), ra2); E.op(mTurn.g, gr > 0.02 ? 1 : 0);
          mArrLab.textContent = gr > 0.98 ? '(2, −1)' : '(−1, −2)';
          var mlp = gr > 0.98 ? [mt1[0] - 8, mt1[1] + 30] : [mt1[0] + 14, mt1[1] + 8];
          mArrLab.setAttribute('x', mlp[0].toFixed(1)); mArrLab.setAttribute('y', mlp[1].toFixed(1));
          E.op(mArrLab, gs > 0.98 && (gr < 0.02 || gr > 0.98) ? (gr > 0.98 ? f(st, T.pos, 0.25) : 1) : 0);

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
          E.op(y1, f(st, T.sym[0], 0.3)); E.op(y1b, f(st, T.symLen, 0.3)); E.op(y2, f(st, T.sym[1], 0.3)); E.op(y3, f(st, T.sym[2], 0.3));
          E.op(y4, f(st, T.sym[3], 0.3)); E.op(y4b, f(st, T.symRad, 0.3));
          E.op(r1b, f(st, T.nameS - 0.2, 0.3)); E.op(r1c, f(st, T.where, 0.3));
          E.op(r2, f(st, T.elbow + 0.3, 0.4) * (1 - f(st, T.out3, 0.4)));
          E.op(sTop, 1);
          // 「乘二」时：S 的那个数鼓一下，右边对应的数弹出（三个 0 在「零还是零」一起出来）
          var MUL = [null, null, 0, 1, 2, null];
          sCol.cells.forEach(function (c, i) {
            var q = sCol.pos(i), pc = f(st, T.sNums[i], 0.3), mi = MUL[i];
            var pulse = Math.max(b(st, T.again[i], 0.45), i >= 3 ? b(st, T.back3b + 0.5 * (i - 3), 0.45) : 0, mi !== null ? b(st, T.mul[mi], 0.6) : b(st, T.zeros, 0.6));
            E.op(c, pc); E.scaleAt(c, q[0], q[1], 1 + 0.5 * (1 - pc) + 0.25 * pulse);
          });
          var nOff = 1 - f(st, T.times - 0.3, 0.3);
          E.op(nTop, f(st, T.front3, 0.3) * nOff); E.op(nBot, f(st, T.sNums[3], 0.3) * nOff); E.op(nBot2, f(st, T.hs0, 0.3) * nOff);
          E.op(times, f(st, T.times, 0.3));
          E.op(vCol.g, f(st, T.zeros - 0.1, 0.2));
          vCol.cells.forEach(function (c, i) {
            var q = vCol.pos(i), mi = MUL[i], pc = f(st, mi !== null ? T.got[mi] : T.zeros, 0.25);
            E.op(c, pc); E.scaleAt(c, q[0], q[1], 1 + 0.5 * (1 - pc));
          });
          E.op(vTop, f(st, T.twist, 0.3)); E.drawOn(ok, f(st, T.twist + 0.3, 0.4));
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
