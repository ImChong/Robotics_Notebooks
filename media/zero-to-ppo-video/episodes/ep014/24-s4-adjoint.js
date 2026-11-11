// 014 第 4 幕：伴随变换——把两步写成一个 6 × 6 的矩阵（《Modern Robotics》定义 3.20、式 (3.84) [Ad_Tbs] = [[Rᵀ, 0], [−Rᵀ[p], Rᵀ]]；
// 5.1.4 节式 (5.22) J_b = [Ad_Tbs] J_s、式 (5.23) J_s = [Ad_Tsb] J_b、式 (3.78)）。数用这一集的姿势（手在 (−2, 3)，x 轴朝上、y 轴朝左），官方库 Adjoint(TransInv(T)) 核对过。
// 2026-11-08 用户反馈「算矩阵时要显示机械臂的画面，要不然不好理解」：左边这一集的网格 + UR5 + 底座、手两个坐标系整幕都在，6 × 6 挪到右栏，
// 每算一列、每乘一份，左边机械臂上同时画出来：
// 句 1「手此刻的位置和潮向」手亮、写 (−2, 3)，右栏写手的位姿（第 7 集）；句 2「底座看」底座的坐标系亮、「站在手上看」手的坐标系亮；
// 句 3「第三列」= 只转肩：肩旁蓝色小弯箭头、手上紫箭头 (−3, −2)，「换到手上」按手的轴拿份：红 −2（往下 2 格）、绿 3（往左 3 格），和第 3 列的 −2、3 一起出；
// 句 4「第四列」整只手往右平移 1 格：手上、原点各一支紫箭头 (1, 0)（不转：每一点都一样，第一步什么也不加），「换轴」沿 x 轴 0、沿 y 轴 −1（绿箭头朝右）；
// 句 5「第五列」往上平移 1 格：紫箭头 (0, 1)，沿 x 轴 1；
// 句 7「用它乘只转肘那一列」肘旁蓝色小弯箭头，「第三列取一份」蓝 (−3, −2)、「第四列取一份」红 (1, 0)、「第五列取两份」绿 (0, 2) 在手上头尾相接，「加起来」紫 (−2, 0)
//    = 只转肘时手的速度，「得到」按手的轴拿份：沿 y 轴 2；句 9「整个雅可比」肩、肘两个关节亮（两列）；句 10「左下一块和手的位置有关」白虚线从原点指到手，
// 「对角上两块，管换轴」两个坐标系亮；句 11 一条从底座弯到手的虚线箭头（伴随变换），「反过来」再一条从手弯回底座。时刻取自配音的逐词时间，见 ../ep014.py。
(function () {
  var E = window.EP014, K = E.K, H = E.HEX;
  var X = 455, D90 = Math.PI / 2;
  var T = {
    two: 0.7, only: 1.94, hand: 2.56, pos: 3.31, ori: 3.77, ep7: 4.87, pose: 5.63, write: 17.3, m66: 17.99, times: 19.37, over: 20.43,
    nth: 22.76, base6: 24.21, isOne: 26.53, hand6: 27.45, ep13: 29.41,
    c3: 38.21, wz: 39.32, one3: 40.71, sh: 41.9, toH: 47.63, n3: [48.68, 49.15, 49.62, 50.16, 50.92, 51.5],
    c4: 52.93, right: 54.99, noRot: 56.28, w0: 57.11, s1: 58.15, s2: 59.52, axes: 59.91, goR: 60.59, x0: 61.41, y1: 62.72, ym1: 63.55,
    c5: 65.18, up: 65.95, x1: 67.18, y0: 68.4,
    rest: 70.64, outPl: 72.18, perp: 73.76, unused: 75.19, shown: 76.39,
    mul: 79.72, onlyE: 84.86, vec: 93.51, t3: 97.3, t4: 98.56, t5: 99.89,
    add: 102.2, r4: 102.94, r4s: 104.07, r4r: 105.31, r5: 106.02, r5s: 107.08, r5r: 108.4, got: 109.33, same: 112.77,
    each: 115.41, twist: 117.52, whole: 118.86, together: 120.34, jb: 121.58, eq: 122.73, mat: 123.03, js: 124.19,
    name66: 126.39, rep: 128.58, tw: 129.67, adT: 131.93, ll: 133.38, pos2: 134.51, step1: 135.83, diag: 138.79, axes2: 139.96,
    rev: 141.9, back: 143.37, also: 144.8, adj2: 145.29, undo: 147.28,
    tX: 7.32, tY: 8.81, tP: 10.41, tB: 12.91, tEnd: 16.2, ep13j: 29.41, jcol: 32.05, hv3: 43.23, rG: 45.07, rM2: 46.43, rG3: 47.04, mulRec: 82.09, shNo: 84.12, fromEO: 88.16, n21: 90.0, turnQ: 91.0, n12: 91.98, soQ: 92.98
  };
  var END = 149.96;
  var AD = [[0, 1, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, -2, 0, 1, 0], [0, 0, 3, -1, 0, 0], [3, 2, 0, 0, 0, 1]];

  E.scenes[3] = {
    title: '伴随变换：一个 6 × 6 的矩阵',
    dur: END,
    cues: [
      { at: 0, s: '手此刻的位姿（第 7 集）：左边 $2$ 列是手的 $x$、$y$ 轴，右边一列是位置 $(-2,\\,3)$，最下面补 $0,\\,0,\\,1$' },
      { at: 16.64, s: '这 $2$ 步只用到手的位姿：写成 $6\\times 6$ 的矩阵，乘上它就换过去' },
      { at: 22.38, s: '第几列：底座看的 $6$ 个数字里第几个是 $1$ 时，站在手上看是多少——第 13 集雅可比的列也是这样' },
      { at: 38.11, s: '第 $3$ 列 = 第 2 幕只转肩那个：$(-3,\\,-2)$ 按手的轴拿份 → $(0,\\,0,\\,1,\\,-2,\\,3,\\,0)$' },
      { at: 52.83, s: '第 $4$ 列：往右平移 $1$ 格、不转：第 $1$ 步不加；换轴：沿 $x$ 轴 $0$、沿 $y$ 轴 $-1$' },
      { at: 65.08, s: '第 $5$ 列：往上平移 $1$ 格：沿 $x$ 轴 $1$、沿 $y$ 轴 $0$' },
      { at: 70.54, s: '剩下 $3$ 列：转出这个平面、垂直屏幕的移动，这一集用不到' },
      { at: 79.62, s: '只转肘那一列（第 3 幕）：肩不动、肘每秒转 $1$ 弧度，$(2,\\,-1)$ 转 $90^\\circ$ → $(1,\\,2)$' },
      { at: 97.2, s: '$\\times(0,\\,0,\\,1,\\,1,\\,2,\\,0)$：第 $3$ 列取 $1$ 份，第 $4$ 列取 $1$ 份，第 $5$ 列取 $2$ 份' },
      { at: 102.1, s: '第 $4$ 个数字 $-2+0+2=0$，第 $5$ 个数字 $3+(-1)+0=2$：和分 $2$ 步算的一样 ✓' },
      { at: 115.31, s: '每一列都是**运动旋量**：**物体雅可比** $=$ 这个矩阵 $\\times$ **空间雅可比**' },
      { at: 126.29, s: '书上叫**伴随表示**，乘上它就是**伴随变换**：左下一块管第 $1$ 步要加的，对角 $2$ 块管换轴' },
      { at: 141.8, s: '反过来，从手上看换回底座上看，也有一个**伴随变换**：乘上去，正好换回来' }
    ],
    build: function (K) {
      var s = K.sceneSvg('这两步只用到手此刻的位置 (−2, 3) 和朝向（x 轴朝上、y 轴朝左），也就是第七集的位姿，可以写成一个六乘六的矩阵，乘上它就换过去：第 i 列就是底座看的六个数字里第 i 个是一的时候，站在手上看是多少。第三列 (0, 0, 1, −2, 3, 0)：只转肩，手的速度 (−3, −2) 按手的轴拿份；第四列：往右平移一格，(0, 0, 0, 0, −1, 0)；第五列：往上平移一格，(0, 0, 0, 1, 0, 0)；剩下三列管转出平面、垂直屏幕的移动。乘只转肘那一列 (0, 0, 1, 1, 2, 0)：第三列取一份、第四列取一份、第五列取两份，左边机械臂上头尾相接是手的速度 (−2, 0)；第四个数字 −2 + 0 + 2 = 0，第五个数字 3 + (−1) + 0 = 2，得到 (0, 0, 1, 0, 2, 0)。整个雅可比一起乘：物体雅可比等于这个矩阵乘以空间雅可比。书上叫伴随表示，乘上它换一种写法就是伴随变换：左下一块和手的位置有关，对角上两块管换轴；反过来从手上看换回底座上看也有一个伴随变换');

      /* ══ 左：网格 + UR5 + 两个坐标系（整幕都在） ══ */
      var pn = E.panel14(s), P = pn.P, u = pn.u, top = pn.top, O = P(0, 0);
      pn.pose(E.P90[0], E.P90[1]);
      var J = pn.joints(), pE = P(J[1][0], J[1][1]), pH = P(J[2][0], J[2][1]);
      var bF = E.frame2(top, P, 1, '底座', { labAt: [0.8, -0.95], tipOff: E.BASE_TIP });   // 往下挪一点：x 轴下面要画原点那一点的紫箭头
      var hF = E.frame2(top, P, 1, '手', { labAt: [0.62, 0.5], labColor: E.V, tipOff: [[-0.3, -0.05], [0.1, 0.38]] });   // 手的 y 轴字母写在尖的上面（尖的外面要画箭头）
      bF.set(0, 0, 0); hF.set(-2, 3, D90);
      var jS = E.joint(top, E.HI), jE = E.joint(top, E.HI), hand = E.dot(top, E.V, 7);
      jS.at(O); jE.at(pE); hand.at(pH);
      var pLab = E.halo(E.txt(top, P(-2.6, 2.45)[0], P(-2.6, 2.45)[1] + 6, '(−2, 3)', 19, E.V, 'end', 900));   // 小臂左边、手的 y 轴下面（上面是 x、y 两个字母）
      var sArc = E.arcArrow(top, O[0], O[1], 0.6 * u, E.OMEGA, 3.5, { head: 12 }); sArc.set(3.45, 4.55);
      var eArc = E.arcArrow(top, pE[0], pE[1], 0.55 * u, E.OMEGA, 3.5, { head: 12 }); eArc.set(3.3, 4.5);
      function arrow(c, w) { var a = E.gv(top, P, c, w || 5, { head: 15 }); a.line.style.filter = 'drop-shadow(0 0 2px #000)'; return a; }
      function lab(x, y, str, c, anchor) { return E.halo(E.txt(top, P(x, y)[0], P(x, y)[1] + 6, str, 17, c, anchor || 'middle', 900)); }
      // 第 3 列：只转肩 → 手的速度 (−3, −2)，按手的轴拿份：红 −2、绿 3（同第 2 幕）
      var v3 = arrow(E.V, 5.5), r3 = arrow(E.X), g3 = arrow(E.Y);
      var r3L = lab(-5.35, 2, '−2', E.X, 'end'), g3L = lab(-3.5, 3.38, '3', E.Y);
      // 第 4 列：往右平移 1 格（手、原点各一支），绿 −1（朝右）；第 5 列：往上平移 1 格，红 1
      var v4 = arrow(E.V, 5.5), v4o = E.gv(top, P, E.V, 3.5, { head: 12, dash: '6 5' }), g4 = arrow(E.Y, 4);
      var x0L = lab(-2, 4.55, '沿 x：0', E.X), g4L = lab(-0.15, 2.3, '沿 y：−1', E.Y, 'end');   // 都在 y 轴左边，不压白色坐标轴
      var v5 = arrow(E.V, 7), x5L = lab(-2, 4.55, '沿 x：1', E.X);
      // 乘只转肘那一列：蓝 (−3, −2) + 红 (1, 0) + 绿 (0, 2) 头尾相接 = 紫 (−2, 0)；按手的轴：沿 y 轴 2
      var m3 = arrow(E.OMEGA), m4 = arrow(E.X), m5 = arrow(E.Y), mS = arrow(E.V, 6);
      var m3L = lab(-3.6, 1.35, '× 1', E.OMEGA, 'start'), m4L = lab(-4.5, 0.62, '× 1', E.X), m5L = lab(-4.3, 2, '× 2', E.Y, 'end');
      var mSL = lab(-3, 3.42, '(−2, 0)', E.V), mY = lab(-3, 3.42, '沿 y：2', E.Y);
      var qA = E.gv(top, P, E.INK, 3.5, { head: 13, dash: '8 6' }), qAL = lab(-1.4, -0.4, '(2, −1)', E.INK, 'end');   // x 轴下面，不压坐标轴
      var qV = arrow(E.V, 5), qVL = lab(0.75, 1.0, '(1, 2)', E.V, 'start');
      // 句 10：位置（白虚线）
      var pA = E.gv(top, P, E.INK, 3.5, { head: 13, dash: '8 6' });
      var pAL = lab(-0.2, 1.95, '位置', E.INK, 'end');   // 白虚线和 y 轴之间
      // 句 11：伴随变换的两条弯箭头
      function curve(a, c, b) {
        var g = E.el('g', {}, top), pth = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3.5, 'stroke-dasharray': '8 6' }, g), null, E.HI);
        var A = P(a[0], a[1]), C = P(c[0], c[1]), B = P(b[0], b[1]);
        pth.setAttribute('d', 'M ' + A[0] + ' ' + A[1] + ' Q ' + C[0] + ' ' + C[1] + ' ' + B[0] + ' ' + B[1]);
        var hd = E.vec(g, E.HI, 3.5, { head: 14 }), t0 = 0.88, q = function (t) { return [(1 - t) * (1 - t) * A[0] + 2 * t * (1 - t) * C[0] + t * t * B[0], (1 - t) * (1 - t) * A[1] + 2 * t * (1 - t) * C[1] + t * t * B[1]]; };
        var h0 = q(t0); hd.set(h0[0], h0[1], B[0], B[1]);
        return { g: g, path: pth, head: hd };
      }
      var cF = curve([0.9, 0.6], [1.6, 3.9], [-0.9, 3.6]), cB = curve([-2.75, 2.2], [-4.4, -1.6], [-0.6, -0.75]);
      var cFL = lab(1.45, 2.55, '伴随变换', E.HI, 'start'), cBL = lab(-4.5, -1.3, '换回来', E.HI, 'middle');

      /* ══ 右栏 ══ */
      var RK = E.rowKit(), Rg = E.el('g', {}, s);
      function grp() { return E.el('g', {}, Rg); }
      // 句 1：手的位姿
      var gA = grp();
      var a1 = RK.line(gA, X, 52, [['这 2 步只用到：', E.INK, 900]], 21);
      var a2 = RK.line(gA, X, 92, [['手的位置 ', E.INK2], ['(−2, 3)', E.V, 900]], 20);
      var a3 = RK.line(gA, X, 126, [['手的朝向：', E.INK2], ['x', E.X, 900], [' 朝上、', E.INK2], ['y', E.Y, 900], [' 朝左', E.INK2]], 20);
      var a4 = RK.line(gA, X, 166, [['= 手的位姿（第 7 集）', E.INK, 900]], 20);
      var Tm = E.matN(gA, X + 110, 252, [[0, -1, -2], [1, 0, 3], [0, 0, 1]], { size: 20, dx: 42, dy: 30, color: function (i, j) { return j === 0 ? E.X : j === 1 ? E.Y : E.V; } });
      var a5 = RK.line(gA, X, 350, [['→ 写成 ', E.INK2], ['6 × 6', E.HI, 900], [' 的矩阵，乘上它', E.INK2]], 19);
      // 6 × 6（句 2 起）
      var gM = grp(), MX = X + 92, MY = 162;
      var Ad = E.matN(gM, MX, MY, AD, { size: 16, dx: 30, dy: 26 });
      var heads = ['ω₁', 'ω₂', 'ω₃', 'v₁', 'v₂', 'v₃'].map(function (h0, k) { return E.txt(gM, Ad.pos(0, k)[0], Ad.box[1] - 9, h0, 15, k < 3 ? E.OMEGA : E.V, 'middle', 900); });
      var capT = E.txt(gM, MX, Ad.box[1] - 32, '底座看的 6 个数', 15, E.INK, 'middle', 900);
      var colN = [0, 1, 2, 3, 4, 5].map(function (k) { return E.txt(gM, Ad.pos(5, k)[0], Ad.box[3] + 19, String(k + 1), 15, E.MUT, 'middle', 900); });
      var fl = E.flow(Ad.box[2] + 1, MY, 7);
      var xS = fl.sym(gM, '×', 18, E.INK);
      var Vs = fl.mat(gM, [[0], [0], [1], [1], [2], [0]], { size: 16, dy: 26, wide: [['−2']], color: function (i) { return i < 3 ? E.OMEGA : E.V; } });
      var eqS = fl.sym(gM, '=', 18, E.INK);
      var Vb = fl.mat(gM, [[0], [0], [1], [0], [2], [0]], { size: 16, dy: 26, wide: [['−2']], color: function (i) { return i < 3 ? E.OMEGA : E.V; } });
      var vsL = E.txt(gM, Vs.pos(0, 0)[0], Vs.box[1] - 9, '只转肘', 14, E.J2, 'middle', 900), vbL = E.txt(gM, Vb.pos(0, 0)[0], Vb.box[1] - 9, '手上看', 14, E.V, 'middle', 900);
      // 下面的说明（每一步一组）
      var Y0 = 296, DY = 27;
      function rows(list) { var g = grp(); var r = list.map(function (pt, k) { return RK.line(g, X, Y0 + DY * k, pt[0], pt[1] || 17); }); return { g: g, r: r }; }
      var B = rows([[[['第几列：', E.HI, 900], ['底座看的第几个数字是 1 时，', E.INK2]]], [[['站在手上看', E.V, 900], ['是多少', E.INK2]]], [[['第 13 集的雅可比也这样：', E.MUT, 700]], 16], [[['第几列 = 只让第几个关节转 1 时，手的速度', E.INK2]], 16]]);
      var C = rows([[[['第 3 列：', E.HI, 900], ['只绕 ', E.INK2], ['z', E.Z, 900], [' 轴转 1 = 只转肩', E.INK2]]], [[['左边：手的速度 ', E.INK2], ['(−3, −2)', E.V, 900]]], [[['按手的轴：', E.INK2], ['−2', E.X, 900], ['、', E.INK2], ['3', E.Y, 900]]]]);
      var D = rows([[[['第 4 列：', E.HI, 900], ['整只手往右平移 1 格、不转', E.INK2]]], [[['① 角速度 0：每一点都一样，不加', E.STEP1, 900]], 16], [[['② 换轴：', E.STEP2, 900], ['沿 ', E.INK2], ['x', E.X, 900], [' 轴 0，沿 ', E.INK2], ['y', E.Y, 900], [' 轴 −1', E.INK2]]]]);
      var Ee = rows([[[['第 5 列：', E.HI, 900], ['往上平移 1 格', E.INK2]]], [[['沿 ', E.INK2], ['x', E.X, 900], [' 轴 1，沿 ', E.INK2], ['y', E.Y, 900], [' 轴 0', E.INK2]]], [[['剩下 3 列：转出平面、垂直屏幕移动——用不到', E.MUT, 700]], 15]]);
      var G = rows([[[['第 3 列 × 1 + 第 4 列 × 1 + 第 5 列 × 2', E.INK2]], 16], [[['第 4 个数字：', E.INK, 900], ['−2 + 0 + 2 = ', E.INK2], ['0', E.V, 900]]], [[['第 5 个数字：', E.INK, 900], ['3 + (−1) + 0 = ', E.INK2], ['2', E.V, 900]]], [[['= 分 2 步算的 ', E.INK2], ['(0, 0, 1, 0, 2, 0) ', E.V, 900], ['✓', E.GOOD, 900]], 16]]);
      var G0 = rows([[[['只转肘那一列（第 3 幕算的）：', E.J2, 900]], 16], [[['肩不动，肘每秒转 1 弧度，站在底座上看', E.INK2]], 16], [[['从肘关节指向原点 ', E.INK2], ['(2, −1)', E.INK, 900], ['，转 90° → ', E.OMEGA, 900], ['(1, 2)', E.V, 900]], 16], [[['→ ', E.INK2], ['(0, 0, 1, 1, 2, 0)', E.INK, 900]], 16]]);
      var Hh = rows([[[['每一列都是', E.INK2], ['运动旋量', E.WARN, 900], [' → 整个雅可比一起乘', E.INK2]], 16]]);
      var hF1 = E.tex(Hh.g, X, Y0 + 40, '\\textcolor{' + H.violet + '}{J_b}=\\textcolor{' + H.hi + '}{[\\mathrm{Ad}_{T_{bs}}]}\\,J_s', 23, E.INK, 'start', { w: 220, h: 42 });
      var hF2 = RK.line(Hh.g, X, Y0 + 88, [['物体雅可比', E.V, 900], [' = ', E.INK2], ['伴随变换', E.HI, 900], [' × ', E.INK2], ['空间雅可比', E.INK, 900]], 16);
      var I = rows([[[['书上：', E.INK2], ['伴随表示', E.HI, 900]]], [[['乘上它换写法：', E.INK2], ['伴随变换', E.HI, 900]]], [[['左下一块', E.STEP1, 900], ['：和手的位置有关，第 1 步要加的', E.INK2]], 16], [[['对角 2 块', E.STEP2, 900], ['：换轴（右上全是 0）', E.INK2]], 16]]);
      var iT = E.tex(I.g, X + 152, Y0, '[\\mathrm{Ad}_T]', 20, E.HI, 'start', { w: 80, h: 34 });
      var Kk = rows([[[['反过来：', E.INK, 900], ['手上看 → 底座上看', E.INK2]]], [[['乘上去，正好换回来', E.INK2]]]]);
      var kF = E.tex(Kk.g, X + 190, Y0 + 26, 'J_s=\\textcolor{' + H.hi + '}{[\\mathrm{Ad}_{T_{sb}}]}\\,\\textcolor{' + H.violet + '}{J_b}', 19, E.INK, 'start', { w: 160, h: 36 });

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }
      function grow(a, x0, y0, dx, dy, k) { a.at(x0, y0, x0 + dx * k, y0 + dy * k); }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) { RK.layout(); laid = true; }
          var f = E.fade, b = E.bump;
          pn.pose(E.P90[0], E.P90[1]); pn.view.clearTint();
          /* ── 左：机械臂上同步画 ── */
          E.glow(hand, E.V, b(st, T.pos, 1.0) > 0.05); E.op(pLab, fr(st, T.pos) * (1 - fr(st, T.c3)));
          E.glow(hF.g, E.V, b(st, T.ori, 1.2) > 0.05 || b(st, T.hand6, 1.2) > 0.05 || b(st, T.axes, 1.2) > 0.05 || b(st, T.jb, 1.2) > 0.05 || E.win(st, T.diag, T.rev, 0.2) > 0.5);
          E.glow(bF.g, E.INK, b(st, T.base6, 1.2) > 0.05 || b(st, T.js, 1.2) > 0.05 || E.win(st, T.diag, T.rev, 0.2) > 0.5);
          E.glow(jS, E.HI, b(st, T.whole, 1.0) > 0.05); E.glow(jE, E.HI, b(st, T.whole + 0.4, 1.0) > 0.05);
          // 第 3 列：只转肩
          var on3 = E.win(st, T.wz, T.c4, 0.3);
          E.op(sArc.g, on3);
          v3.at(-2, 3, -5, 1); E.op(v3.g, E.win(st, T.hv3, T.c4, 0.3));
          grow(r3, -5, 3, 0, -2, f(st, T.rM2, 0.5)); E.op(r3.g, (st > T.rM2 ? 1 : 0) * (1 - fr(st, T.c4)));
          grow(g3, -2, 3, -3, 0, f(st, T.rG3, 0.5)); E.op(g3.g, (st > T.rG3 ? 1 : 0) * (1 - fr(st, T.c4)));
          E.op(r3L, E.win(st, T.rM2 + 0.3, T.c4, 0.3)); E.op(g3L, E.win(st, T.rG3 + 0.3, T.c4, 0.3));
          // 第 4 列：往右平移 1 格
          grow(v4, -2, 3, 1, 0, f(st, T.right, 0.6)); E.op(v4.g, (st > T.right ? 1 : 0) * (1 - fr(st, T.c5)));
          grow(v4o, 0.05, -0.32, 1, 0, f(st, T.w0, 0.6)); E.op(v4o.g, (st > T.w0 ? 1 : 0) * (1 - fr(st, T.c5)));
          grow(g4, -2, 2.82, 1, 0, f(st, T.ym1, 0.5)); E.op(g4.g, (st > T.ym1 ? 1 : 0) * (1 - fr(st, T.c5)));
          E.op(x0L, E.win(st, T.x0, T.c5, 0.3)); E.op(g4L, E.win(st, T.ym1, T.c5, 0.3));
          // 第 5 列：往上平移 1 格（画在手的 x 轴下面一层，粗一点）
          grow(v5, -1.85, 3, 0, 1, f(st, T.up, 0.6)); E.op(v5.g, (st > T.up ? 1 : 0) * (1 - fr(st, T.rest)));
          E.op(x5L, E.win(st, T.x1, T.rest, 0.3));
          E.glow(hF.ax.g, E.X, E.win(st, T.x0, T.c5, 0.2) > 0.5 || E.win(st, T.x1, T.rest, 0.2) > 0.5);
          E.glow(hF.ay.g, E.Y, E.win(st, T.y1, T.c5, 0.2) > 0.5);
          // 乘只转肘那一列：肘旁小弯箭头，三份头尾相接
          E.op(eArc.g, E.win(st, T.onlyE, T.each, 0.3));
          // 只转肘那一列怎么来的（第 3 幕）：从肘关节指向原点 (2, −1)，转 90° 是 (1, 2)，画在原点
          qA.at(-2.08, 0.84, -0.08, -0.16); E.op(qA.g, E.win(st, T.fromEO, T.n12 + 0.6, 0.3)); E.op(qAL, E.win(st, T.n21, T.turnQ + 0.6, 0.3));
          qV.at(0, 0, 1, 2); E.op(qV.g, E.win(st, T.n12, T.t3, 0.3)); E.op(qVL, E.win(st, T.n12, T.t3, 0.3));
          [bF.lx, bF.ly].forEach(function (n) { E.op(n, 1 - E.win(st, T.n12 - 0.3, T.t3, 0.3)); });   // (1, 2) 从底座 y 轴的字母旁边过：先收起
          show(G0, T.mulRec, T.t3, [T.mulRec, T.shNo, T.fromEO, T.soQ]); E.op(G0.r[2].parts[1], fr(st, T.n21)); E.op(G0.r[2].parts[3], fr(st, T.n12));
          var onM = 1 - fr(st, T.each);
          grow(m3, -2, 3, -3, -2, f(st, T.t3, 0.6)); E.op(m3.g, (st > T.t3 ? 1 : 0) * onM);
          grow(m4, -5, 1, 1, 0, f(st, T.t4, 0.5)); E.op(m4.g, (st > T.t4 ? 1 : 0) * onM);
          grow(m5, -4, 1, 0, 2, f(st, T.t5, 0.6)); E.op(m5.g, (st > T.t5 ? 1 : 0) * onM);
          E.op(m3L, E.win(st, T.t3 + 0.3, T.each, 0.3)); E.op(m4L, E.win(st, T.t4 + 0.3, T.each, 0.3)); E.op(m5L, E.win(st, T.t5 + 0.3, T.each, 0.3));
          grow(mS, -2, 3, -2, 0, f(st, T.add, 0.6)); E.op(mS.g, (st > T.add ? 1 : 0) * onM);
          E.op(mSL, E.win(st, T.add + 0.4, T.got, 0.3)); E.op(mY, E.win(st, T.got, T.each, 0.3));
          E.glow(hF.ay.g, E.Y, E.win(st, T.got, T.each, 0.2) > 0.5 || E.win(st, T.y1, T.c5, 0.2) > 0.5);
          // 手的 x、y 字母：箭头画在旁边时先收起（不压线）
          [hF.lx, hF.ly].forEach(function (n) { E.op(n, 1 - E.win(st, T.n3[3] - 0.3, T.c4, 0.3) - E.win(st, T.t3 - 0.3, T.each, 0.3)); });
          // 句 10：位置
          pA.at(0.12, 0.18, -1.8, 2.75); E.op(pA.g, E.win(st, T.pos2, T.rev, 0.3)); E.op(pAL, E.win(st, T.pos2, T.rev, 0.3));
          // 句 11：伴随变换的弯箭头
          E.drawOn(cF.path, f(st, T.adT, 0.8)); E.op(cF.head.g, fr(st, T.adT + 0.6)); E.op(cFL, fr(st, T.adT));
          E.drawOn(cB.path, f(st, T.adj2, 0.8)); E.op(cB.head.g, fr(st, T.adj2 + 0.6)); E.op(cBL, fr(st, T.undo));

          /* ── 右栏：6 × 6 ── */
          E.op(gA, 1 - fr(st, T.nth - 0.4));
          E.op(a1.g, fr(st, T.two)); E.op(a2.g, fr(st, T.pos)); E.op(a3.g, fr(st, T.ori)); E.op(a4.g, fr(st, T.ep7));
          E.op(Tm.g, fr(st, T.pose));
          Tm.hl('x', 0, 0, 2, 0, E.X, E.win(st, T.tX, T.tY, 0.25)); Tm.hl('y', 0, 1, 2, 1, E.Y, E.win(st, T.tY, T.tP, 0.25)); Tm.hl('p', 0, 2, 2, 2, E.V, E.win(st, T.tP, T.tB, 0.25)); Tm.hl('b', 2, 0, 2, 2, E.INK2, E.win(st, T.tB, T.tEnd, 0.25));   // 位姿的三列、最下面一排，念到哪框哪 E.op(a5.g, fr(st, T.write)); E.op(a5.parts[1], fr(st, T.m66)); E.op(a5.parts[2], fr(st, T.times));
          E.op(gM, fr(st, T.nth - 0.1));
          var hlC = function (k, a, z) { Ad.hl('c' + k, 0, k, 5, k, E.HI, E.win(st, a, z, 0.3)); };
          hlC(2, T.c3, T.c4); hlC(3, T.c4, T.c5); hlC(4, T.c5, T.rest);
          Ad.hl('m3', 0, 2, 5, 2, E.OMEGA, E.win(st, T.t3, T.t4, 0.3) * 0.9); Ad.hl('m4', 0, 3, 5, 3, E.X, E.win(st, T.t4, T.t5, 0.3) * 0.9); Ad.hl('m5', 0, 4, 5, 4, E.Y, E.win(st, T.t5, T.add, 0.3) * 0.9);
          Ad.hl('r4', 3, 0, 3, 5, E.V, E.win(st, T.r4, T.r5, 0.3)); Ad.hl('r5', 4, 0, 4, 5, E.V, E.win(st, T.r5, T.got, 0.3));
          Ad.hl('ll', 3, 0, 5, 2, E.STEP1, E.win(st, T.ll, T.rev, 0.3));
          Ad.hl('d1', 0, 0, 2, 2, E.STEP2, E.win(st, T.diag, T.rev, 0.3)); Ad.hl('d2', 3, 3, 5, 5, E.STEP2, E.win(st, T.diag, T.rev, 0.3));
          var dimRest = fr(st, T.rest);
          for (var i = 0; i < 6; i++) for (var j = 0; j < 6; j++) {
            var a;
            if (j === 2) a = 0.35 + 0.65 * fr(st, T.n3[i]);
            else if (j === 3) a = 0.35 + 0.65 * fr(st, i === 4 ? T.ym1 : T.x0 - 0.6);
            else if (j === 4) a = 0.35 + 0.65 * fr(st, T.x1 - 0.5);
            else a = 0.35 + 0.65 * fr(st, T.nth) * (1 - 0.55 * dimRest);
            if (st < T.c3 && (j === 2 || j === 3 || j === 4)) a = 0.35;
            E.op(Ad.cells[i][j], a);
          }
          heads.forEach(function (n, k) { E.op(n, fr(st, T.base6)); E.glow(n, E.HI, (k === 2 && b(st, T.wz, 1.0) > 0.05) || (k === 3 && b(st, T.right, 1.0) > 0.05) || (k === 4 && b(st, T.up, 1.0) > 0.05)); });
          E.op(capT, fr(st, T.base6));
          colN.forEach(function (n, k) { E.op(n, fr(st, T.nth)); var on = (k === 2 && E.win(st, T.c3, T.c4, 0.2) > 0.5) || (k === 3 && E.win(st, T.c4, T.c5, 0.2) > 0.5) || (k === 4 && E.win(st, T.c5, T.rest, 0.2) > 0.5); n.style.fill = on ? H.hi : H.mut; });
          E.op(xS, fr(st, T.vec)); E.op(Vs.g, fr(st, T.vec)); E.op(vsL, fr(st, T.vec));
          Vs.hl('a', 2, 0, 2, 0, E.OMEGA, E.win(st, T.t3, T.t4, 0.3)); Vs.hl('b', 3, 0, 3, 0, E.X, E.win(st, T.t4, T.t5, 0.3)); Vs.hl('c', 4, 0, 4, 0, E.Y, E.win(st, T.t5, T.add, 0.3));
          E.op(eqS, fr(st, T.got)); E.op(Vb.g, fr(st, T.got)); E.op(vbL, fr(st, T.got));
          Vb.hl('a', 3, 0, 4, 0, E.V, E.win(st, T.got, T.each, 0.3));
          // 下面的说明
          function show(R, a, z, keys) { E.op(R.g, E.win(st, a - 0.1, z - 0.2, 0.3)); R.r.forEach(function (row, k) { E.op(row.g, fr(st, keys[k])); }); }
          show(B, T.nth, T.c3, [T.nth, T.hand6, T.ep13j, T.jcol]);
          show(C, T.c3, T.c4, [T.c3, T.hv3, T.rG]); E.op(C.r[2].parts[1], fr(st, T.rM2)); E.op(C.r[2].parts[3], fr(st, T.rG3));
          show(D, T.c4, T.c5, [T.c4, T.w0, T.axes]); E.op(D.r[0].parts[1], fr(st, T.right)); E.op(D.r[2].parts[3], fr(st, T.x0)); E.op(D.r[2].parts[4], fr(st, T.y1)); E.op(D.r[2].parts[5], fr(st, T.ym1));
          show(Ee, T.c5, T.mul, [T.c5, T.x1, T.rest]); E.op(Ee.r[1].parts[3], fr(st, T.y0)); E.op(Ee.r[1].parts[4], fr(st, T.y0));
          show(G, T.mul, T.each, [T.t3, T.r4, T.r5, T.got]);
          E.op(G.r[1].parts[1], fr(st, T.r4s)); E.op(G.r[1].parts[2], fr(st, T.r4r)); E.op(G.r[2].parts[1], fr(st, T.r5s)); E.op(G.r[2].parts[2], fr(st, T.r5r)); E.op(G.r[3].parts[2], fr(st, T.same));
          show(Hh, T.each, T.name66, [T.each]); E.op(Hh.r[0].parts[1], fr(st, T.twist)); E.op(Hh.r[0].parts[2], fr(st, T.whole)); E.op(hF1, fr(st, T.jb)); E.op(hF2.g, fr(st, T.eq)); E.op(hF2.parts[3], fr(st, T.mat)); E.op(hF2.parts[4], fr(st, T.js));
          show(I, T.name66, T.rev, [T.rep, T.tw, T.ll, T.diag]); E.op(iT, fr(st, T.rep)); E.op(I.r[1].parts[1], fr(st, T.adT)); E.op(I.r[2].parts[1], fr(st, T.pos2)); E.op(I.r[3].parts[1], fr(st, T.axes2));
          E.op(Kk.g, fr(st, T.rev - 0.1)); E.op(Kk.r[0].g, fr(st, T.rev)); E.op(Kk.r[0].parts[1], fr(st, T.back)); E.op(kF, fr(st, T.adj2)); E.op(Kk.r[1].g, fr(st, T.undo));
        }
      };
    }
  };
})();
