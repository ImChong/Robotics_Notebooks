// 010 第 6 幕：矩阵指数用在哪（2026-11-02 新增；用户反馈「看完没理解矩阵指数的作用和应用场景，以及为什么要学」）。
// ① 正向运动学：斜着看的 UR5 和六根关节轴，指数积公式（从第 5 幕挪过来；《Modern Robotics》式 (4.14)，例 4.5 正是 UR5）；
// ② 更新朝向：俯视一块「身体」，陀螺仪测到角速度 2 弧度/秒，每 0.01 秒转 2 × 0.01 = 0.02 弧度，交给矩阵指数、乘到原来的朝向上（右乘：
//    绕身体自己的轴转，原书式 (3.51) 下面那段），「一步一步乘起来」连走 50 步、转 1 弧度；MuJoCo 源码 engine_util_spatial.c 的
//    mju_quatIntegrate 就是「角速度 × 时间 → 绕轴转多少 → 乘到朝向上」（用四元数写）；
// ③ 反过来：手要去抓杯子（杯子按同一个投影画成三维圆柱杯），矩阵对数算出还差多少（原书 6.2.2 节 [V_b] = log T_bd），一点一点改关节角（第 15 集）；
// ①b 和齐次变换连乘有什么区别（2026-11-02 用户要求加进视频）：连乘把已有的几个四乘四按顺序接起来（原书 3.3.1 节 T_ab T_bc = T_ac、式 (4.4)，第 7 集）；
//    矩阵指数从螺旋轴和转了多少算出这次转动的四乘四（命题 3.25）；指数积公式两样都用（式 (4.14)）；矩阵指数自己也能看成同一小步连乘无穷多次（Hall 命题 3.8，第 2 幕）；
// ④ 后面几集：第 11 集正向运动学（指数积公式）、第 13 集雅可比（对指数积公式求导，原书 5.1.1 节）、第 15 集数值逆运动学（矩阵对数，原书 6.2.2 节）。
// 时刻取自配音的逐词时间，见 ../ep010.py。
(function () {
  var E = window.EP010, K = E.K, H = E.HEX;
  var D = Math.PI / 180, RX = E.ARM.RX;
  var T = {
    title: 0.65, first: 2.59, fk: 3.12, arm: 0.3, joint: 4.22, axes: [7.0, 7.25, 7.5, 7.75, 8.0, 8.25], turn: [8.97, 9.8], ang: 10.4,
    each: 11.95, facs: [12.77, 12.98, 13.28, 13.5, 13.71, 13.92], mult: 14.28, mM: 15.54, zero: 17.11, hand: 19.36, name: 21.37, ep11: 22.49, outA: 24.4,
    x0: 24.7, q7: 26.09, chA: 29.66, mats: 31.21, join: 32.87, expB: 33.56, axis: 35.27, turnQ: 36.02, calc: 36.87, out44: 38.07,
    poe: 39.84, useExp: 43.16, thenMul: 46.13, self: 47.07, sc2: 49.87, small: 51.08, inf: 52.01, outX: 53.6,
    two: 54.07, upd: 54.61, gyro: 55.67, omega: 57.75, w2: 59.87, dt: 60.82, mul: 62.16, res: 65.85,
    give: 69.09, onto: 70.56, newR: 72.45, same: 73.4, run: [74.46, 76.78], sim: 76.44, outB: 80.28,
    three: 80.53, rev: 81.21, cup: 82.67, now: 83.25, diff: 84.55, log: 86.09, axisq: 87.71, iter: [89.23, 91.28], ep15: 91.74, outC: 94.88,
    d0: 95.12, e11: 97.45, e13: 99.47, e15: 101.1, from: 103.48
  };
  var END = 106.1;
  var JN = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'];
  var HOP = [0.6, 0.84, 0.936, 1];
  function iterFrac(st) {   // ③ 一共 4 跳，每跳 0.35 s
    var dt = (T.iter[1] - T.iter[0]) / HOP.length, it = 0;
    HOP.forEach(function (c, k) { var prev = k ? HOP[k - 1] : 0; it += (c - prev) * E.fade(st, T.iter[0] + k * dt, 0.35); });
    return it;
  }

  E.scenes[5] = {
    title: '矩阵指数用在哪',
    dur: END,
    cues: [
      { at: 0, s: '矩阵指数用在哪？① 正向运动学：每个关节是一根**螺旋轴**，转了多少就是**角度**' },
      { at: 11.86, s: '各算一个**矩阵指数**按顺序相乘，再乘 M（关节都在 0° 时手的位姿）：手的**位姿**（第 11 集）' },
      { at: 24.6, s: '连乘：已有的几个 4 × 4 **按顺序接起来**；**矩阵指数**：从螺旋轴、角度算出这次转动的 4 × 4' },
      { at: 39.74, s: '指数积公式两样都用：每个关节先算**矩阵指数**，再**连乘**；矩阵指数自己 = 同一小步乘无穷多次' },
      { at: 53.98, s: '② 更新**朝向**：陀螺仪测到角速度 2 弧度/秒，隔 0.01 秒：2 × 0.01 = 0.02 弧度' },
      { at: 68.06, s: '交给**矩阵指数**、乘到原来的**朝向**上，一步一步乘（MuJoCo 源码就这样更新朝向）' },
      { at: 80.42, s: '③ 反过来：用**矩阵对数**算出还差多少，再一点一点改关节角（第 15 集）' },
      { at: 95.02, s: '第 11 集正向运动学、第 13 集雅可比、第 15 集数值逆运动学：都从**矩阵指数**出发' }
    ],
    build: function (K) {
      var s = K.sceneSvg('矩阵指数用在哪：一、正向运动学，每个关节是一根螺旋轴，各算一个矩阵指数按顺序相乘再乘上零位时手的位姿，就是手现在的位姿（指数积公式，第十一集）；和第七集齐次变换的连乘有什么区别：连乘把已有的几个四乘四按顺序接起来，矩阵指数从螺旋轴和转了多少算出这次转动的四乘四，指数积公式两样都用，矩阵指数自己也能看成同一小步乘无穷多次；二、更新朝向，陀螺仪测到角速度每秒两弧度，隔 0.01 秒转 2 × 0.01 = 0.02 弧度，交给矩阵指数乘到原来的朝向上，一步一步乘起来，物理仿真器 MuJoCo 就这样更新朝向；三、反过来用矩阵对数算出手离目标还差多少，一点一点改关节角（数值逆运动学，第十五集）；第十一集正向运动学、第十三集雅可比、第十五集数值逆运动学都从矩阵指数出发');
      var X = RX + 10;
      /* ══ ① 正向运动学：UR5 斜视 + 六根关节轴 + 指数积公式（2026-11-02 从第 5 幕挪过来）══ */
      var g3 = E.el('g', {}, s);
      var ur = window.Robot3D.ur5(g3, 92, 404, 2.9, { az: 32, el: 16 }), V = ur.view, THREE = window.Robot3D.THREE;
      var POSE0 = { shoulder_pan_joint: 20 * D, shoulder_lift_joint: -95 * D, elbow_joint: 75 * D, wrist_1_joint: -70 * D, wrist_2_joint: -90 * D, wrist_3_joint: 0 };
      V.camera({ box: [0, 0, 440, 420] });
      V.joints(POSE0);
      var axG = E.el('g', {}, g3);
      var axLines = JN.map(function () { return K.paint(E.el('path', { fill: 'none', 'stroke-width': 3.5, 'stroke-dasharray': '8 5' }, axG), null, E.WARN); });
      var axLabs = JN.map(function (n, i) { return E.halo(E.txt(axG, 0, 0, 'S' + '₁₂₃₄₅₆'[i], 22, E.WARN, 'middle', 900)); });
      var thLab = E.halo(E.txt(axG, 0, 0, 'θ₂', 24, E.OMEGA, 'start', 900));
      var toolG = E.el('g', {}, g3);
      var toolAx = [E.X, E.Y, E.Z].map(function (c) { return E.vec(toolG, c, 4, { head: 12 }); });
      function axisOf(name) {
        var j = V.robot.joints[name];
        j.updateMatrixWorld(true);
        return { o: new THREE.Vector3().setFromMatrixPosition(j.matrixWorld), d: j.axis.clone().transformDirection(j.matrixWorld) };
      }
      // 标签摆法同 009 第 5 幕：[哪一端（0 / 1 / 'mid'）, dx, dy, 对齐]；每根轴从关节往两边画多长（米）
      var LAB = [['mid', -26, 7, 'end'], [1, 22, 2, 'start'], [0, -24, 2, 'end'], [1, -8, -18, 'end'], [1, 12, 7, 'start'], [0, 0, -12, 'middle']];
      var LEN = [[-0.05, 0.3], [-0.22, 0.22], [-0.2, 0.2], [-0.16, 0.16], [-0.15, 0.15], [-0.14, 0.14]];
      var RK3 = E.rowKit(), r3 = E.el('g', {}, g3);
      var b0 = RK3.line(r3, X, 26, [['① 正向运动学', E.HI, 900]], 24);
      var b1 = RK3.line(r3, X, 64, [['关节的运动可以用它来写：', E.INK, 900]], 22);
      var b2 = RK3.line(r3, X, 98, [['每个关节 = 一根', E.INK2], ['螺旋轴', E.WARN, 900]], 22);
      var b3 = RK3.line(r3, X, 132, [['转了多少 = ', E.INK2], ['角度 θ', E.OMEGA, 900]], 22);
      var fm = E.el('g', {}, r3), FY = 190;
      // 两行：T手 = exp([S₁]θ₁) exp([S₂]θ₂) / ⋯ exp([S₆]θ₆) M（中间三个因子写成 ⋯；上标太小，不用 e 的上标写法）
      var tEq = E.tex(fm, X, FY, 'T_{\\text{手}}=', 24, E.HI, 'start', { w: 72, h: 44 });
      var facs = [[1, X + 78, FY], [2, X + 210, FY], [6, X + 104, FY + 46]].map(function (q) {   // 框比公式宽一圈，两个框隔开（压线检查：右括号原先压在框线上）
        var g = E.el('g', {}, fm);
        var box = K.paint(E.el('rect', { x: q[1] - 7, y: q[2] - 29, width: 126, height: 42, rx: 8, 'stroke-width': 2 }, g), E.tint(E.HI, 16, 'var(--demo-surface)'), E.HI);
        E.tex(g, q[1], q[2], '\\exp([\\mathcal S_' + q[0] + ']\\,\\theta_' + q[0] + ')', 22, E.INK, 'start', { w: 104, h: 44 });
        return { g: g, box: box };
      });
      var dots = E.tex(fm, X + 70, FY + 46, '\\cdots', 22, E.INK, 'start', { w: 30, h: 44 });
      var mTex = E.tex(fm, X + 234, FY + 46, 'M', 24, E.INK, 'start', { w: 30, h: 44 });
      var mNote = RK3.line(r3, X, FY + 82, [['M：所有关节都在 0° 时，手的位姿', E.MUT, 700]], 17);
      var nm = RK3.line(r3, X, FY + 122, [['指数积公式', E.HI, 900], ['（PoE）', E.INK2, 700]], 24);
      var c11 = E.el('g', {}, r3);
      K.paint(E.el('rect', { x: RX, y: 334, width: 340, height: 64, rx: 14, 'stroke-width': 2.5 }, c11), E.tint(E.GOOD, 10, 'var(--demo-surface)'), E.GOOD);
      E.txt(c11, RX + 18, 362, '第 11 集 · 正运动学', 22, E.GOOD, 'start', 900);
      E.txt(c11, RX + 18, 388, '六个关节连乘 → 手的位姿', 19, E.INK, 'start', 700);


      /* ══ ①b 矩阵指数和齐次变换连乘有什么区别（2026-11-02 用户要求加进视频）══ */
      var gX = E.el('g', {}, s), RX2 = E.rowKit();
      RX2.line(gX, 190, 32, [['矩阵指数', E.HI, 900], ['和', E.INK2, 700], ['齐次变换连乘', E.INK, 900], ['：有什么区别？', E.INK2, 700]], 24);
      // 左卡：连乘，三个已有的四乘四按顺序接起来（第 7 集）
      var cardA = E.el('g', {}, gX);
      var rectA = K.paint(E.el('rect', { x: 24, y: 50, width: 364, height: 150, rx: 14, 'stroke-width': 2.5 }, cardA), E.tint(E.INK2, 8, 'var(--demo-surface)'), E.INK2);
      RX2.line(cardA, 44, 84, [['齐次变换连乘', E.INK, 900], ['（第 7 集）', E.INK2, 700]], 22);
      var sq = [0, 1, 2].map(function (i) {
        var g = E.el('g', {}, cardA), x0 = 56 + 92 * i;
        K.paint(E.el('rect', { x: x0, y: 102, width: 60, height: 56, rx: 8, 'stroke-width': 2.5 }, g), E.tint(E.INK2, 22, 'var(--demo-surface)'), E.INK);
        E.txt(g, x0 + 30, 139, 'T' + '₁₂₃'[i], 24, E.INK, 'middle', 900);
        return g;
      });
      var links = [0, 1].map(function (i) { var v = E.vec(cardA, E.INK2, 3, { head: 10 }); v.set(56 + 92 * i + 64, 130, 56 + 92 * (i + 1) - 4, 130); return v; });
      var dotsA = E.txt(cardA, 330, 138, '⋯', 24, E.INK2, 'middle', 900);
      var capA = E.txt(cardA, 44, 186, '已有的几个 4 × 4 → 按顺序接起来', 18, E.INK2, 'start', 700);
      // 右卡：矩阵指数，螺旋轴 + 转了多少 → 这次转动的四乘四（这一集）
      var cardB = E.el('g', {}, gX);
      var rectB = K.paint(E.el('rect', { x: 412, y: 50, width: 364, height: 150, rx: 14, 'stroke-width': 2.5 }, cardB), E.tint(E.HI, 10, 'var(--demo-surface)'), E.HI);
      RX2.line(cardB, 432, 84, [['矩阵指数', E.HI, 900], ['（这一集）', E.INK2, 700]], 22);
      var inS = E.txt(cardB, 436, 122, '螺旋轴 S', 20, E.WARN, 'start', 900);
      var inTh = E.txt(cardB, 436, 154, '转了 θ', 20, E.OMEGA, 'start', 900);
      var arrB = E.vec(cardB, E.INK2, 3, { head: 11 }); arrB.set(530, 130, 572, 130);
      var boxB = E.el('g', {}, cardB);
      K.paint(E.el('rect', { x: 580, y: 102, width: 176, height: 56, rx: 8, 'stroke-width': 2.5 }, boxB), E.tint(E.HI, 18, 'var(--demo-surface)'), E.HI);
      E.tex(boxB, 668, 138, '\\exp([\\mathcal S]\\,\\theta)', 24, E.INK, 'middle', { w: 150, h: 44 });
      var capB = E.txt(cardB, 432, 186, '一根轴、转了多少 → 这次转动的 4 × 4', 18, E.INK2, 'start', 700);
      // 中间：指数积公式，每个 exp 是矩阵指数，整串是连乘
      var FY2 = 274, X0 = 128, poe = E.el('g', {}, gX);
      E.tex(poe, X0, FY2, 'T_{\\text{手}}=', 24, E.INK, 'start', { w: 72, h: 44 });
      var expX = [X0 + 78, X0 + 212, X0 + 384];
      var expBx = [1, 2, 6].map(function (j, k) {
        var g = E.el('g', {}, poe), x = expX[k];
        var box = K.paint(E.el('rect', { x: x - 7, y: FY2 - 29, width: 126, height: 42, rx: 8, 'stroke-width': 2 }, g), E.tint(E.HI, 16, 'var(--demo-surface)'), E.HI);
        E.tex(g, x, FY2, '\\exp([\\mathcal S_' + j + ']\\,\\theta_' + j + ')', 22, E.INK, 'start', { w: 104, h: 44 });
        return { g: g, box: box, lab: E.txt(poe, x + 56, FY2 + 38, '矩阵指数', 16, E.HI, 'middle', 900) };
      });
      E.tex(poe, X0 + 344, FY2, '\\cdots', 22, E.INK, 'start', { w: 30, h: 44 });
      E.tex(poe, X0 + 518, FY2, 'M', 24, E.INK, 'start', { w: 30, h: 44 });
      var BK = [X0 + 71, X0 + 540];
      var brk = K.paint(E.el('path', { d: 'M ' + BK[0] + ' ' + (FY2 - 35) + ' V ' + (FY2 - 45) + ' H ' + BK[1] + ' V ' + (FY2 - 35), fill: 'none', 'stroke-width': 2.5 }, gX), null, E.INK);
      var brkLab = E.txt(gX, (BK[0] + BK[1]) / 2, FY2 - 54, '连乘：按顺序接起来', 18, E.INK, 'middle', 900);
      // 下面：矩阵指数自己，也能看成同一小步连乘无穷多次（第 2 幕）
      var xS = RX2.line(gX, 100, 354, [['矩阵指数自己，也能看成连乘：', E.INK, 900], ['同一小步，乘无穷多次', E.HI, 900], ['（第 2 幕）', E.INK2, 700]], 20);
      var limTex = E.tex(gX, 400, 400, '\\left(I+\\tfrac{[\\mathcal S]\\theta}{n}\\right)^{n}\\to \\textcolor{' + H.hi + '}{\\exp([\\mathcal S]\\theta)}\\quad (n\\to\\infty)', 24, E.INK, 'middle', { w: 400, h: 50 });

      // ③ 目标姿势（手伸到杯子那里）和 ① 末的姿势
      var POSEA = Object.assign({}, POSE0); POSEA.shoulder_lift_joint = POSE0.shoulder_lift_joint + 28 * D;
      var POSEB = { shoulder_pan_joint: 48 * D, shoulder_lift_joint: -78 * D, elbow_joint: 58 * D, wrist_1_joint: -60 * D, wrist_2_joint: -90 * D, wrist_3_joint: 30 * D };
      function toolFrame() {
        var tl = V.robot.links.tool0 || V.robot.frames && V.robot.frames.tool0;
        V.robot.updateMatrixWorld(true);
        var o = new THREE.Vector3().setFromMatrixPosition(tl.matrixWorld);
        return { o: V.project(o), ax: [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)].map(function (a) { return V.project(o.clone().addScaledVector(a.transformDirection(tl.matrixWorld), 0.13)); }) };
      }
      V.joints(POSEB); var GT = toolFrame(); V.joints(POSE0);
      var ghostG = E.el('g', {}, g3);
      [E.X, E.Y, E.Z].forEach(function (c, i) { var v = E.vec(ghostG, c, 3, { head: 11, dash: '5 4' }); v.set(GT.o[0], GT.o[1], GT.ax[i][0], GT.ax[i][1]); });
      // 杯子按 UR5 同一个投影画成三维的圆柱杯（用户反馈：机器人是三维的，杯子也要三维）：杯口在目标手的正下方，竖直向上（URDF 的 z）
      var cupG = E.el('g', {}, g3);
      (function () {
        V.joints(POSEB); V.robot.updateMatrixWorld(true);
        var tl = V.robot.links.tool0 || V.robot.frames && V.robot.frames.tool0;
        var o3 = new THREE.Vector3().setFromMatrixPosition(tl.matrixWorld);
        var up = new THREE.Vector3(0, 0, 1).transformDirection(V.robot.matrixWorld);
        var e1 = new THREE.Vector3(1, 0, 0).transformDirection(V.robot.matrixWorld), e2 = new THREE.Vector3().crossVectors(up, e1).normalize();
        V.joints(POSE0); V.robot.updateMatrixWorld(true);
        var R = 0.04, H0 = -0.15, H1 = -0.05;   // 半径 4 cm；杯底、杯口在手下面 15 cm、5 cm
        function ring(h) { var pts = []; for (var k = 0; k < 48; k++) { var a = 2 * Math.PI * k / 48; pts.push(V.project(o3.clone().addScaledVector(up, h).addScaledVector(e1, R * Math.cos(a)).addScaledVector(e2, R * Math.sin(a)))); } return pts; }
        var top = ring(H1), bot = ring(H0), all = top.concat(bot).slice().sort(function (p, q) { return p[0] - q[0] || p[1] - q[1]; });
        function cross(o, a, b) { return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]); }
        var lo = [], hi = [];
        all.forEach(function (p) { while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], p) <= 0) lo.pop(); lo.push(p); });
        all.slice().reverse().forEach(function (p) { while (hi.length >= 2 && cross(hi[hi.length - 2], hi[hi.length - 1], p) <= 0) hi.pop(); hi.push(p); });
        var hull = lo.slice(0, -1).concat(hi.slice(0, -1));
        K.paint(E.el('path', { d: E.poly(hull, true), 'stroke-width': 3, 'stroke-linejoin': 'round' }, cupG), E.tint(E.INK2, 25, 'var(--demo-surface)'), E.INK);
        K.paint(E.el('path', { d: E.poly(top, true), fill: 'none', 'stroke-width': 2.5 }, cupG), null, E.INK);   // 杯口一圈
        var hd = []; for (var k = 0; k <= 16; k++) { var f = Math.PI * k / 16; hd.push(V.project(o3.clone().addScaledVector(up, (H0 + H1) / 2 + 0.028 * Math.cos(f)).addScaledVector(e1, R + 0.026 * Math.sin(f)))); }
        K.paint(E.el('path', { d: E.poly(hd), fill: 'none', 'stroke-width': 3.5, 'stroke-linecap': 'round' }, cupG), null, E.INK);   // 把手
      })();
      var gapPath = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '7 6' }, g3), null, E.HI);
      var gapLab = E.halo(E.txt(g3, 0, 0, '还差多少？', 20, E.HI, 'middle', 900));
      var RC = E.rowKit(), rC = E.el('g', {}, s);
      var c0 = RC.line(rC, X, 26, [['③ 反过来：逆向运动学', E.HI, 900]], 24);
      var c1 = RC.line(rC, X, 68, [['手 → 杯子：', E.INK2], ['还差多少？', E.HI, 900]], 22);
      var c2 = E.el('g', {}, rC);
      E.tex(c2, X, 124, '\\textcolor{' + H.hi + '}{\\log}\\big(T_{\\text{手}}^{-1}\\,T_{\\text{目标}}\\big)', 26, E.INK, 'start', { w: 230, h: 56 });
      var c3 = RC.line(rC, X, 176, [['矩阵对数：', E.HI, 900], ['还要绕什么轴、转多少', E.INK, 900]], 20);
      var c4 = RC.line(rC, X, 214, [['一点一点改关节角 → 手到杯子', E.INK, 900]], 20);
      var c15 = E.el('g', {}, rC);
      K.paint(E.el('rect', { x: RX, y: 334, width: 340, height: 64, rx: 14, 'stroke-width': 2.5 }, c15), E.tint(E.GOOD, 10, 'var(--demo-surface)'), E.GOOD);
      E.txt(c15, RX + 18, 362, '第 15 集 · 数值逆运动学', 22, E.GOOD, 'start', 900);
      E.txt(c15, RX + 18, 388, '用矩阵对数求还差多少（原书 6.2 节）', 18, E.INK, 'start', 700);

      /* ══ ② 更新朝向：俯视一块身体，陀螺仪测角速度 ══ */
      var gB = E.el('g', {}, s), BC = [215, 222], BS = 72;
      var ref0 = K.paint(E.el('path', { d: 'M ' + BC[0] + ' ' + BC[1] + ' H ' + (BC[0] + 150), fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '6 6' }, gB), null, E.MUT);
      var bodyG = E.el('g', {}, gB);
      K.paint(E.el('rect', { x: BC[0] - BS, y: BC[1] - BS, width: 2 * BS, height: 2 * BS, rx: 18, 'stroke-width': 3 }, bodyG), E.tint(E.HI, 22, 'var(--demo-surface)'), E.BODY);
      // 陀螺仪芯片放在身体左下角：中心往右、往上是两根小箭头
      K.paint(E.el('circle', { cx: BC[0] - 30, cy: BC[1] + 30, r: 29, 'stroke-width': 2 }, bodyG), E.tint(E.INK2, 30, 'var(--demo-surface)'), E.INK2);   // 圆形芯片：转多少都包得住正着写的字
      var gyroLab = E.txt(gB, 0, 0, '陀螺仪', 17, E.INK, 'middle', 900);   // 字不跟着身体转（转着的字外框大，会压到芯片框）：每帧放到芯片中心
      var bx = E.vec(gB, E.X, 4.5, { head: 14 }), by = E.vec(gB, E.Y, 4.5, { head: 14 });
      var wArc = E.arcArrow(gB, BC[0], BC[1], 118, E.OMEGA, 3.5, { head: 13 });
      var wLab = E.halo(E.txt(gB, BC[0] + 110, BC[1] + 66, 'ω = 2 弧度/秒', 20, E.OMEGA, 'start', 900));   // 写在右下：左上会压到身体的 y 箭头
      var turnArc = E.arcArrow(gB, BC[0], BC[1], 150, E.OMEGA, 3, { head: 11 });
      var cnt = E.halo(E.txt(gB, BC[0], BC[1] + 168, '', 20, E.INK, 'middle', 900));
      var RB = E.rowKit(), rB = E.el('g', {}, s);
      var e0 = RB.line(rB, X, 26, [['② 更新朝向', E.HI, 900]], 24);
      var e1 = RB.line(rB, X, 68, [['陀螺仪测角速度：', E.INK2], ['2 弧度/秒', E.OMEGA, 900]], 21);
      var e2 = RB.line(rB, X, 104, [['每隔 0.01 秒：', E.INK2], ['2 × 0.01 = 0.02', E.OMEGA, 900], [' 弧度', E.INK2]], 21);
      var e3 = E.el('g', {}, rB);
      E.tex(e3, X, 158, 'R_{\\text{新}}=R\\,\\textcolor{' + H.hi + '}{\\exp}\\big(0.02\\,[\\hat\\omega]\\big)', 26, E.INK, 'start', { w: 280, h: 56 });
      var e4 = RB.line(rB, X, 210, [['交给', E.INK2], ['矩阵指数', E.HI, 900], ['，乘到原来的朝向上', E.INK2]], 20);
      var e5 = RB.line(rB, X, 246, [['和第 2 幕一样：一步一步乘', E.INK, 900]], 20);
      var e6 = RB.line(rB, X, 300, [['MuJoCo 源码 mju_quatIntegrate：', E.MUT, 700]], 17);
      var e7 = RB.line(rB, X, 326, [['角速度 × 时间 → 绕轴转多少 → 乘上去', E.MUT, 700]], 17);

      /* ══ ④ 后面几集：都从矩阵指数出发 ══ */
      var gD = E.el('g', {}, s), NX = 400, NY = 96;
      var node = E.el('g', {}, gD);
      K.paint(E.el('rect', { x: NX - 130, y: NY - 40, width: 260, height: 80, rx: 16, 'stroke-width': 3.5 }, node), E.tint(E.HI, 14, 'var(--demo-surface)'), E.HI);
      E.tex(node, NX - 110, NY + 12, '\\exp', 36, E.HI, 'start', { w: 70, h: 54 });
      E.txt(node, NX - 40, NY + 12, '= 矩阵指数', 28, E.HI, 'start', 900);
      var BOX = [[140, '第 11 集 · 正向运动学', '各关节的矩阵指数连乘'], [400, '第 13 集 · 雅可比', '对指数积公式求导（原书 5.1）'], [660, '第 15 集 · 数值逆运动学', '矩阵对数求还差多少（原书 6.2）']];
      var boxes = BOX.map(function (bx_) {
        var g = E.el('g', {}, gD), cx = bx_[0];
        var arr = E.vec(g, E.INK2, 3.5, { head: 13 }); arr.set(NX + (cx - NX) * 0.3, NY + 44, cx, 232);
        K.paint(E.el('rect', { x: cx - 120, y: 240, width: 240, height: 100, rx: 14, 'stroke-width': 2.5 }, g), E.tint(E.GOOD, 10, 'var(--demo-surface)'), E.GOOD);
        E.txt(g, cx, 280, bx_[1], 21, E.GOOD, 'middle', 900);
        E.txt(g, cx, 314, bx_[2], 15, E.INK, 'middle', 700);   // 15 号：17 号时第 3 个框的字超出框
        return { g: g, arr: arr };
      });
      var footD = E.txt(gD, NX, 392, '后面几集都从这里出发', 24, E.HI, 'middle', 900);

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) { RX2.layout(); RK3.layout(); RC.layout(); RB.layout(); laid = true; }
          var f = E.fade, b = E.bump;
          /* ── ① 正向运动学（UR5 到 ③ 再用一次） ── */
          var inA = f(st, T.arm, 0.4) * (1 - f(st, T.outA, 0.4)), inC = f(st, T.three, 0.4) * (1 - f(st, T.outC, 0.4));
          E.op(g3, Math.max(inA, inC)); E.op(r3, 1 - f(st, T.outA, 0.4)); E.op(axG, 1 - f(st, T.outA, 0.4));
          var turn = 28 * D * f(st, T.turn[0], T.turn[1] - T.turn[0]);
          var pose = Object.assign({}, POSE0); pose.shoulder_lift_joint = POSE0.shoulder_lift_joint + turn;
          // ③ 一点一点改关节角：从 ① 末的姿势往目标走，每一步走剩下的六成（0.6、0.84、0.94，最后到位）
          if (st > T.outA) { var it = iterFrac(st); JN.forEach(function (n) { pose[n] = POSEA[n] + (POSEB[n] - POSEA[n]) * it; }); }
          V.joints(pose);
          JN.forEach(function (n, i) {
            var a = axisOf(n), L = LEN[i];
            var p0 = V.project(a.o.clone().addScaledVector(a.d, L[0])), p1 = V.project(a.o.clone().addScaledVector(a.d, L[1]));
            axLines[i].setAttribute('d', 'M ' + p0[0].toFixed(1) + ' ' + p0[1].toFixed(1) + ' L ' + p1[0].toFixed(1) + ' ' + p1[1].toFixed(1));
            var on = f(st, T.axes[i], 0.35);
            E.op(axLines[i], on); E.op(axLabs[i], on);
            var lp = LAB[i], at = lp[0] === 'mid' ? [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2] : lp[0] ? p1 : p0;
            axLabs[i].setAttribute('x', (at[0] + lp[1]).toFixed(1)); axLabs[i].setAttribute('y', (at[1] + lp[2]).toFixed(1)); axLabs[i].style.textAnchor = lp[3];
            var pulse = Math.max(b(st, T.axes[i], 0.6), b(st, T.facs[i], 0.6), i === 1 ? f(st, T.turn[0], 0.3) * (1 - f(st, T.each, 0.4)) : 0);
            axLines[i].style.strokeWidth = 3.5 + 3 * pulse; E.glow(axLines[i], E.WARN, pulse > 0.05);
            if (i === 1) { thLab.setAttribute('x', (p1[0] + 16).toFixed(1)); thLab.setAttribute('y', (p1[1] + 32).toFixed(1)); }   // S₂ 标签下面
          });
          E.op(thLab, f(st, T.ang, 0.3) * (1 - f(st, T.each, 0.4)));
          var tl = V.robot.links.tool0 || V.robot.frames && V.robot.frames.tool0;
          V.robot.updateMatrixWorld(true);
          var to = new THREE.Vector3().setFromMatrixPosition(tl.matrixWorld), op0 = V.project(to);
          [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)].forEach(function (ax, i) {
            var tp = V.project(to.clone().addScaledVector(ax.transformDirection(tl.matrixWorld), 0.13));
            toolAx[i].set(op0[0], op0[1], tp[0], tp[1]);
          });
          var tb = b(st, T.hand, 0.9);
          E.op(toolG, f(st, T.hand, 0.4)); E.glow(toolG, E.HI, tb > 0.05);
          toolAx.forEach(function (a) { a.line.style.strokeWidth = 4 + 2 * tb; });
          E.op(b0.g, fr(st, T.first)); E.op(b1.g, fr(st, T.joint)); E.op(b2.g, fr(st, T.axes[0])); E.op(b3.g, fr(st, T.turn[0]));
          E.op(fm, fr(st, T.each));
          [0, 1, 5].forEach(function (i, k) { E.op(facs[k].box, b(st, T.facs[i], 0.7)); });
          E.op(dots, 1); E.op(mTex, fr(st, T.mM)); E.op(tEq, fr(st, T.hand));
          E.op(mNote.g, fr(st, T.zero));
          E.op(nm.g, fr(st, T.name));
          E.pop(c11, RX + 170, 366, f(st, T.ep11, 0.35), 0.92);

          /* ── ①b 和连乘的区别 ── */
          E.op(gX, f(st, T.x0 - 0.1, 0.4) * (1 - f(st, T.outX, 0.4)));
          E.op(cardB, fr(st, T.x0 + 0.17)); E.op(cardA, fr(st, T.q7));   // 问「有什么区别」时两张卡先摆出来，讲到哪张、哪张亮一下
          E.glow(rectA, E.INK, b(st, T.chA, 0.9) > 0.05); E.glow(rectB, E.HI, b(st, T.expB, 0.9) > 0.05);
          sq.forEach(function (g, i) { E.op(g, fr(st, T.mats + 0.2 * i)); E.glow(g, E.INK, b(st, T.join + 0.3 * i, 0.6) > 0.05); });
          links.forEach(function (v, i) { E.op(v.g, fr(st, T.join + 0.3 * i + 0.15)); });
          E.op(dotsA, fr(st, T.join + 0.75)); E.op(capA, fr(st, T.join));
          E.op(inS, fr(st, T.axis)); E.op(inTh, fr(st, T.turnQ));
          E.op(arrB.g, fr(st, T.calc)); E.op(boxB, fr(st, T.calc)); E.glow(boxB, E.HI, b(st, T.out44, 0.8) > 0.05); E.op(capB, fr(st, T.out44));
          E.op(poe, fr(st, T.poe));
          expBx.forEach(function (q, k) { var pb = b(st, T.useExp + 0.25 * k, 0.7); q.box.style.strokeWidth = 2 + 2 * pb; E.glow(q.g, E.HI, pb > 0.05); E.op(q.lab, fr(st, T.useExp + 0.25 * k)); });
          E.drawOn(brk, f(st, T.thenMul - 0.3, 0.6)); E.op(brkLab, fr(st, T.thenMul));
          E.op(xS.g, fr(st, T.self)); E.op(xS.parts[2], fr(st, T.sc2)); E.op(xS.parts[1], fr(st, T.small));
          E.op(limTex, fr(st, T.sc2)); E.glow(limTex, E.HI, b(st, T.inf, 0.9) > 0.05);

          /* ── ③ 杯子、目标、还差多少 ── */
          E.op(ghostG, f(st, T.cup, 0.3)); E.op(cupG, f(st, T.cup, 0.3));
          var cur = toolFrame();
          var mx = (cur.o[0] + GT.o[0]) / 2, my = Math.min(cur.o[1], GT.o[1]) - 50;
          gapPath.setAttribute('d', 'M ' + cur.o[0].toFixed(1) + ' ' + cur.o[1].toFixed(1) + ' Q ' + mx.toFixed(1) + ' ' + my.toFixed(1) + ' ' + GT.o[0].toFixed(1) + ' ' + GT.o[1].toFixed(1));
          var gapOn = f(st, T.diff, 0.3) * (1 - f(st, T.iter[0], 0.3));
          E.op(gapPath, gapOn); gapLab.setAttribute('x', mx.toFixed(1)); gapLab.setAttribute('y', (my + 8).toFixed(1)); E.op(gapLab, gapOn);
          E.op(rC, f(st, T.three, 0.4) * (1 - f(st, T.outC, 0.4)));
          E.op(c0.g, fr(st, T.three)); E.op(c1.g, fr(st, T.cup)); E.op(c1.parts[1], fr(st, T.diff)); E.op(c2, fr(st, T.log));
          E.op(c3.g, fr(st, T.axisq)); E.op(c4.g, fr(st, T.iter[0])); E.pop(c15, RX + 170, 366, f(st, T.ep15, 0.35), 0.92);

          /* ── ② 更新朝向 ── */
          E.op(gB, f(st, T.two, 0.4) * (1 - f(st, T.outB, 0.4)));
          var k50 = Math.floor(50 * E.clamp((st - T.run[0]) / (T.run[1] - T.run[0]), 0, 1) + 1e-9), nStep = (st >= T.res ? 1 : 0);
          var steps = Math.max(nStep, k50), ang = 0.02 * steps;
          bodyG.setAttribute('transform', 'rotate(' + (-ang / D).toFixed(3) + ' ' + BC[0] + ' ' + BC[1] + ')');
          var gx = -30, gy = 30;   // 芯片中心相对身体中心（屏幕坐标，y 朝下）
          gyroLab.setAttribute('x', (BC[0] + gx * Math.cos(ang) + gy * Math.sin(ang)).toFixed(1)); gyroLab.setAttribute('y', (BC[1] - gx * Math.sin(ang) + gy * Math.cos(ang) + 6).toFixed(1));
          bx.set(BC[0], BC[1], BC[0] + 112 * Math.cos(ang), BC[1] - 112 * Math.sin(ang));
          by.set(BC[0], BC[1], BC[0] - 112 * Math.sin(ang), BC[1] - 112 * Math.cos(ang));
          wArc.set(-25 * D, 55 * D); E.op(wArc.g, f(st, T.omega, 0.3)); E.op(wLab, f(st, T.omega, 0.3));
          turnArc.set(0, Math.max(1e-3, ang)); E.op(turnArc.g, ang > 0.03 ? 1 : 0); E.op(ref0, f(st, T.res, 0.3));
          cnt.textContent = steps <= 1 ? (st >= T.res ? '1 步：转了 0.02 弧度' : '') : steps + ' 步：' + steps + ' × 0.02 = ' + E.num(0.02 * steps) + ' 弧度';
          E.op(cnt, f(st, T.res, 0.3));
          E.op(rB, f(st, T.two, 0.4) * (1 - f(st, T.outB, 0.4)));
          E.op(e0.g, fr(st, T.two)); E.op(e1.g, fr(st, T.gyro)); E.op(e1.parts[1], fr(st, T.omega)); E.op(e2.g, fr(st, T.dt)); E.op(e2.parts[1], fr(st, T.mul));
          E.op(e2.parts[2], fr(st, T.mul)); E.op(e3, fr(st, T.give)); E.op(e4.g, fr(st, T.onto)); E.op(e5.g, fr(st, T.same)); E.op(e6.g, fr(st, T.sim)); E.op(e7.g, fr(st, T.sim));

          /* ── ④ 后面几集 ── */
          E.op(gD, f(st, T.d0, 0.4));
          [T.e11, T.e13, T.e15].forEach(function (a, i) { E.op(boxes[i].g, fr(st, a)); var gb = b(st, T.from + 0.3 * i, 0.9); boxes[i].arr.line.style.strokeWidth = 3.5 + 2.5 * gb; E.glow(boxes[i].g, E.GOOD, gb > 0.05); });
          E.pop(node, NX, NY, f(st, T.d0, 0.35), 0.9); E.op(footD, fr(st, T.from));
        }
      };
    }
  };
})();
