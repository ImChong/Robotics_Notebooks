// 014 第 4 幕：伴随变换——把两步写成一个 6 × 6 的矩阵（《Modern Robotics》定义 3.20、式 (3.84) [Ad_Tbs] = [[Rᵀ, 0], [−Rᵀ[p], Rᵀ]]；
// 5.1.4 节式 (5.22) J_b = [Ad_Tbs] J_s、式 (5.23) J_s = [Ad_Tsb] J_b、式 (3.78)）。数用这一集的姿势（手在 (−2, 3)，x 轴朝上、y 轴朝左），官方库 Adjoint(TransInv(T)) 核对过。
// 句 1「手此刻的位置和潮向」左边网格上手的坐标系亮，右栏写手的位姿（第 7 集：左边两列是手的 x、y 轴，右边一列是位置），「六乘六的矩阵」左边换成 6 × 6；
// 句 2「第几列」右栏写列的意思（同第 13 集）；句 3「第三列」第 3 列框亮、数逐个出（= 第 2 幕只转肩的那个）；
// 句 4「第四列」右栏小图：手的两根轴 + 往右平移的紫箭头，「沿 x 轴是零」「沿 y 轴是负一」，第 4 列框亮；句 5 第 5 列同样；句 6 剩下 3 列变淡；
// 句 7「用它乘只转肘那一列」矩阵右边乘 (0, 0, 1, 1, 2, 0)，「第三列取一份」「第四列取一份」「第五列取两份」各框亮；
// 句 8「第四个数」「第五个数」第 4、5 排框亮、右栏写加法，「得到」= (0, 0, 1, 0, 2, 0)；
// 句 9「整个雅可比」右边那一列换成 J_s（6 × 2）、结果换成 J_b，右栏写 J_b = [Ad] J_s；句 10「伴随表示」「伴随变换」，「左下一块」紫框、「对角上两块」蓝框；
// 句 11「反过来」右栏写 J_s = [Ad_Tsb] J_b。时刻取自配音的逐词时间，见 ../ep014.py。
(function () {
  var E = window.EP014, K = E.K, H = E.HEX;
  var X = 455, D90 = Math.PI / 2;
  var T = {
    two: 0.7, only: 2.04, hand: 2.73, pos: 3.5, ori: 4.01, ep7: 5.24, pose: 6.06, write: 7.56, m66: 8.24, times: 9.62, over: 10.68, swap: [11.2, 12.4],
    nth: 13.0, base6: 14.41, isOne: 16.56, hand6: 17.48, ep13: 19.2,
    c3: 22.51, wz: 23.68, one3: 25.04, sh: 26.12, toH: 27.38, n3: [28.53, 29.04, 29.56, 30.12, 30.89, 31.49],
    c4: 32.93, right: 34.25, noRot: 36.28, w0: 37.11, s1: 38.15, s2: 39.52, goR: 40.59, x0: 41.64, y1: 42.72, ym1: 43.55,
    c5: 45.18, up: 45.95, x1: 47.18, y0: 48.4,
    rest: 50.64, outPl: 52.16, perp: 53.68, unused: 55.13, shown: 56.3,
    mul: 58.69, vec: 60.62, t3: 63.36, t4: 64.59, t5: 65.82,
    add: 68.11, r4: 68.85, r4s: 69.79, r4r: 71.06, r5: 71.73, r5s: 72.68, r5r: 74.06, got: 74.97, same: 78.41,
    each: 81.05, twist: 83.16, whole: 84.5, together: 85.98, jb: 87.22, eq: 88.37, mat: 88.91, js: 89.83,
    name66: 92.03, rep: 94.22, tw: 95.31, adT: 97.57, ll: 99.02, pos2: 100.49, step1: 101.47, diag: 104.43, axes: 105.86,
    rev: 107.54, back: 109.01, also: 110.44, adj2: 110.93, undo: 112.91
  };
  var END = 115.6;
  var AD = [[0, 1, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, -2, 0, 1, 0], [0, 0, 3, -1, 0, 0], [3, 2, 0, 0, 0, 1]];

  E.scenes[3] = {
    title: '伴随变换：一个 6 × 6 的矩阵',
    dur: END,
    cues: [
      { at: 0, s: '这 $2$ 步只用到手此刻的位置和朝向——第 7 集的位姿：写成 $6\\times 6$ 的矩阵，乘上它就换过去' },
      { at: 12.63, s: '第几列：底座看的 $6$ 个数里第几个是 $1$ 时，站在手上看是多少（同第 13 集雅可比的列）' },
      { at: 22.41, s: '第 $3$ 列：只绕 $z$ 轴转 $1$——就是只转肩那个：$(0,\\,0,\\,1,\\,-2,\\,3,\\,0)$' },
      { at: 32.83, s: '第 $4$ 列：往右平移 $1$ 格、不转：第 $1$ 步不加；换轴：沿 $x$ 轴 $0$、沿 $y$ 轴 $-1$' },
      { at: 45.08, s: '第 $5$ 列：往上平移 $1$ 格：沿 $x$ 轴 $1$、沿 $y$ 轴 $0$' },
      { at: 50.54, s: '剩下 $3$ 列：转出这个平面、垂直屏幕的移动，这一集用不到' },
      { at: 58.59, s: '$\\times(0,\\,0,\\,1,\\,1,\\,2,\\,0)$：第 $3$ 列取 $1$ 份，第 $4$ 列取 $1$ 份，第 $5$ 列取 $2$ 份' },
      { at: 68.01, s: '第 $4$ 个数 $-2+0+2=0$，第 $5$ 个数 $3+(-1)+0=2$：$(0,\\,0,\\,1,\\,0,\\,2,\\,0)$，和分 $2$ 步算的一样 ✓' },
      { at: 80.95, s: '每一列都是**运动旋量**：**物体雅可比** $=$ 这个矩阵 $\\times$ **空间雅可比**' },
      { at: 91.93, s: '书上叫**伴随表示**，乘上它就是**伴随变换**：左下一块管第 $1$ 步要加的，对角 $2$ 块管换轴' },
      { at: 107.44, s: '反过来，从手上看换回底座上看，也有一个**伴随变换**：乘上去，正好换回来' }
    ],
    build: function (K) {
      var s = K.sceneSvg('这两步只用到手此刻的位置 (−2, 3) 和朝向（x 轴朝上、y 轴朝左），也就是第七集的位姿，可以写成一个六乘六的矩阵，乘上它就换过去：第 i 列就是底座看的六个数里第 i 个是一的时候，站在手上看是多少。第三列 (0, 0, 1, −2, 3, 0)：只转肩的那一个；第四列：往右平移一格，(0, 0, 0, 0, −1, 0)；第五列：往上平移一格，(0, 0, 0, 1, 0, 0)；剩下三列管转出平面、垂直屏幕的移动。乘只转肘那一列 (0, 0, 1, 1, 2, 0)：第三列取一份、第四列取一份、第五列取两份，第四个数 −2 + 0 + 2 = 0，第五个数 3 + (−1) + 0 = 2，得到 (0, 0, 1, 0, 2, 0)。整个雅可比一起乘：物体雅可比等于这个矩阵乘以空间雅可比。书上叫伴随表示，乘上它换一种写法就是伴随变换：左下一块和手的位置有关，对角上两块管换轴；反过来从手上看换回底座上看也有一个伴随变换');

      /* ── 句 1：网格 + 手的坐标系（之后换成 6 × 6） ── */
      var gL = E.el('g', {}, s);
      var pn = E.panel14(gL), P = pn.P, u = pn.u, top = pn.top, O = P(0, 0);
      pn.pose(E.P90[0], E.P90[1]);
      var J = pn.joints();
      var bF = E.frame2(top, P, 1, '底座', { labAt: [0.75, -0.62] }), hF = E.frame2(top, P, 1, '手', { labAt: [0.45, 0.45], labColor: E.V });
      bF.set(0, 0, 0); hF.set(-2, 3, D90);
      var jS = E.joint(top, E.HI), jE = E.joint(top, E.HI), hand = E.dot(top, E.V, 7);
      jS.at(O); jE.at(P(J[1][0], J[1][1])); hand.at(P(J[2][0], J[2][1]));
      var pLab = E.halo(E.txt(top, P(-2.5, 3.62)[0], P(-2.5, 3.62)[1] + 6, '(−2, 3)', 19, E.V, 'end', 900));   // 手的左上：左边是手的 y 轴字母

      /* ── 6 × 6（句 2 起，左边） ── */
      var gM = E.el('g', {}, s), MX = 128, MY = 222;
      var Ad = E.matN(gM, MX, MY, AD, { size: 19, dx: 38, dy: 31 });
      var heads = ['ω₁', 'ω₂', 'ω₃', 'v₁', 'v₂', 'v₃'].map(function (h0, k) { return E.txt(gM, Ad.pos(0, k)[0], Ad.box[1] - 12, h0, 17, k < 3 ? E.OMEGA : E.V, 'middle', 900); });
      var capT = E.txt(gM, MX, Ad.box[1] - 40, '底座看的 6 个数', 17, E.INK, 'middle', 900);
      var colN = [0, 1, 2, 3, 4, 5].map(function (k) { return E.txt(gM, Ad.pos(5, k)[0], Ad.box[3] + 22, String(k + 1), 16, E.MUT, 'middle', 900); });   // 列号；讲到的那一列亮黄
      var capB = E.txt(gM, MX, Ad.box[3] + 48, '手的位姿 → 伴随变换', 17, E.HI, 'middle', 900);
      // 右边乘的那一列，和结果
      var fl = E.flow(Ad.box[2] + 2, MY, 9);
      var xS = fl.sym(gM, '×', 20, E.INK);
      var Vs = fl.mat(gM, [[0], [0], [1], [1], [2], [0]], { size: 19, dy: 31, wide: [['−2']], color: function (i) { return i < 3 ? E.OMEGA : E.V; } });
      var eqS = fl.sym(gM, '=', 20, E.INK);
      var Vb = fl.mat(gM, [[0], [0], [1], [0], [2], [0]], { size: 19, dy: 31, wide: [['−2']], color: function (i) { return i < 3 ? E.OMEGA : E.V; } });
      var vsL = E.txt(gM, Vs.pos(0, 0)[0], Vs.box[1] - 12, '只转肘', 15, E.J2, 'middle', 900), vbL = E.txt(gM, Vb.pos(0, 0)[0], Vb.box[1] - 12, '手上看', 15, E.V, 'middle', 900);
      // 句 9：换成两列的雅可比
      var gJ = E.el('g', {}, gM);
      var fl3 = E.flow(Ad.box[2] + 2, MY, 6);
      var xS2 = fl3.sym(gJ, '×', 20, E.INK);
      var Js = fl3.mat(gJ, [[0, 0], [0, 0], [1, 1], [0, 1], [0, 2], [0, 0]], { size: 17, dx: 26, dy: 31, color: function (i, j) { return j === 0 ? E.J1 : E.J2; } });
      var eqS2 = fl3.sym(gJ, '=', 20, E.INK);
      var Jb = fl3.mat(gJ, [[0, 0], [0, 0], [1, 1], [-2, 0], [3, 2], [0, 0]], { size: 17, dx: 30, dy: 31, color: function (i, j) { return j === 0 ? E.J1 : E.J2; } });
      var jsL = E.tex(gJ, Js.pos(0, 0)[0] + 13, Js.box[1] - 8, 'J_s', 19, E.INK, 'middle', { w: 40, h: 30 }), jbL = E.tex(gJ, Jb.pos(0, 0)[0] + 15, Jb.box[1] - 8, 'J_b', 19, E.V, 'middle', { w: 40, h: 30 });

      /* ══ 右栏 ══ */
      var RK = E.rowKit(), Rg = E.el('g', {}, s);
      function grp() { return E.el('g', {}, Rg); }
      // 小图：手的两根轴（x 红朝上、y 绿朝左）+ 紫色的移动箭头
      function mini(g, cx, cy, mv) {
        var a = E.vec(g, E.X, 3.5, { head: 11 }), b2 = E.vec(g, E.Y, 3.5, { head: 11 }), m = E.vec(g, E.V, 4, { head: 12 });
        a.set(cx, cy, cx, cy - 34); b2.set(cx, cy, cx - 34, cy); m.set(cx, cy, cx + mv[0] * 38, cy - mv[1] * 38);
        E.txt(g, cx + 8, cy - 34, 'x', 16, E.X, 'start', 900); E.txt(g, cx - 34, cy - 8, 'y', 16, E.Y, 'middle', 900);
        return { a: a, b: b2, m: m };
      }
      var gA = grp();
      var a1 = RK.line(gA, X, 52, [['这 2 步只用到：', E.INK, 900]], 21);
      var a2 = RK.line(gA, X, 92, [['手的位置 ', E.INK2], ['(−2, 3)', E.V, 900]], 20);
      var a3 = RK.line(gA, X, 126, [['手的朝向：', E.INK2], ['x', E.X, 900], [' 朝上、', E.INK2], ['y', E.Y, 900], [' 朝左', E.INK2]], 20);
      var a4 = RK.line(gA, X, 166, [['= 手的位姿（第 7 集）', E.INK, 900]], 20);
      var Tm = E.matN(gA, X + 110, 252, [[0, -1, -2], [1, 0, 3], [0, 0, 1]], { size: 20, dx: 42, dy: 30, color: function (i, j) { return j === 0 ? E.X : j === 1 ? E.Y : E.V; } });
      var a5 = RK.line(gA, X, 350, [['→ 写成 ', E.INK2], ['6 × 6', E.HI, 900], [' 的矩阵，乘上它', E.INK2]], 19);
      var gB = grp();
      var b1 = RK.line(gB, X, 48, [['第几列：', E.HI, 900]], 21);
      var b2 = RK.line(gB, X, 82, [['底座看的 6 个数里', E.INK2]], 19);
      var b3 = RK.line(gB, X, 112, [['第几个是 1 时 → ', E.INK2], ['手上看', E.V, 900], ['是多少', E.INK2]], 19);
      var b4 = RK.line(gB, X, 142, [['（同第 13 集雅可比的列）', E.MUT, 700]], 16);
      var gC = grp();
      var c1 = RK.line(gC, X, 190, [['第 3 列：', E.HI, 900], ['只绕 ', E.INK2], ['z', E.Z, 900], [' 轴转 1', E.INK2]], 19);
      var c2 = RK.line(gC, X, 220, [['= 只转肩那个（第 2 幕）', E.INK2]], 18);
      var c3 = RK.line(gC, X, 250, [['(0, 0, 1, ', E.OMEGA, 900], ['−2, 3, 0)', E.V, 900]], 20);
      var gD = grp();
      var d1 = RK.line(gD, X, 190, [['第 4 列：', E.HI, 900], ['往右平移 1 格、不转', E.INK2]], 19);
      var d2 = RK.line(gD, X, 222, [['① 角速度 0 → 什么也不加', E.STEP1, 900]], 17);
      var d3 = RK.line(gD, X, 252, [['② 换轴：', E.STEP2, 900], ['沿 ', E.INK2], ['x', E.X, 900], [' 轴 0，沿 ', E.INK2], ['y', E.Y, 900], [' 轴 −1', E.INK2]], 17);
      var mD = E.el('g', {}, gD); mini(mD, X + 290, 230, [1, 0]);
      var gE = grp();
      var e1 = RK.line(gE, X, 300, [['第 5 列：', E.HI, 900], ['往上平移 1 格', E.INK2]], 19);
      var e2 = RK.line(gE, X, 330, [['沿 ', E.INK2], ['x', E.X, 900], [' 轴 1，沿 ', E.INK2], ['y', E.Y, 900], [' 轴 0', E.INK2]], 17);
      var mE = E.el('g', {}, gE); mini(mE, X + 290, 340, [0, 1]);
      var gF = grp();
      var f1 = RK.line(gF, X, 386, [['剩下 3 列：转出平面、垂直屏幕移动——用不到', E.MUT, 700]], 15);
      var gG = grp();
      var g1 = RK.line(gG, X, 52, [['× 只转肘那一列', E.J2, 900]], 20);
      var g2 = RK.line(gG, X, 90, [['第 3 列 × 1 + 第 4 列 × 1 + 第 5 列 × 2', E.INK2]], 16);
      var g3 = RK.line(gG, X, 140, [['第 4 个数：', E.INK, 900], ['−2 + 0 + 2 = ', E.INK2], ['0', E.V, 900]], 19);
      var g4 = RK.line(gG, X, 174, [['第 5 个数：', E.INK, 900], ['3 + (−1) + 0 = ', E.INK2], ['2', E.V, 900]], 19);
      var g5 = RK.line(gG, X, 214, [['= ', E.INK2], ['(0, 0, 1, 0, 2, 0)', E.V, 900]], 20);
      var g6 = RK.line(gG, X, 246, [['和分 2 步算的一样 ', E.INK2], ['✓', E.GOOD, 900]], 18);
      var gH = grp();
      var h1 = RK.line(gH, X, 300, [['每一列都是', E.INK2], ['运动旋量', E.WARN, 900], [' → 整个一起乘', E.INK2]], 17);
      var h2 = E.tex(gH, X, 344, '\\textcolor{' + H.violet + '}{J_b}=\\textcolor{' + H.hi + '}{[\\mathrm{Ad}_{T_{bs}}]}\\,J_s', 24, E.INK, 'start', { w: 300, h: 44 });
      var h3 = RK.line(gH, X, 388, [['物体雅可比', E.V, 900], [' = ', E.INK2], ['伴随变换', E.HI, 900], [' × ', E.INK2], ['空间雅可比', E.INK, 900]], 16);
      var gI = grp();
      var i1 = RK.line(gI, X, 52, [['书上：', E.INK2], ['伴随表示', E.HI, 900]], 20);
      var i1b = E.tex(gI, X + 152, 52, '[\\mathrm{Ad}_T]', 22, E.HI, 'start', { w: 90, h: 36 });
      var i2 = RK.line(gI, X, 88, [['乘上它换写法：', E.INK2], ['伴随变换', E.HI, 900]], 19);
      var i3 = RK.line(gI, X, 132, [['左下一块', E.STEP1, 900], ['：和手的位置有关', E.INK2]], 18);
      var i4 = RK.line(gI, X, 160, [['第 1 步要加的那一份', E.INK2]], 17);
      var i5 = RK.line(gI, X, 196, [['对角上 2 块', E.STEP2, 900], ['：换轴', E.INK2]], 18);
      var i6 = RK.line(gI, X, 224, [['（右上全是 0）', E.MUT, 700]], 16);
      var gK = grp();
      var k1 = RK.line(gK, X, 270, [['反过来：', E.INK, 900], ['手上看 → 底座上看', E.INK2]], 18);
      var k2 = E.tex(gK, X, 310, 'J_s=\\textcolor{' + H.hi + '}{[\\mathrm{Ad}_{T_{sb}}]}\\,\\textcolor{' + H.violet + '}{J_b}', 24, E.INK, 'start', { w: 300, h: 44 });
      var k3 = RK.line(gK, X, 352, [['乘上去，正好换回来', E.INK2]], 18);

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) { RK.layout(); laid = true; }
          var f = E.fade, b = E.bump;
          pn.pose(E.P90[0], E.P90[1]); pn.view.clearTint();
          var sw = f(st, T.swap[0], T.swap[1] - T.swap[0]);
          E.op(gL, 1 - sw); E.op(gM, sw);
          E.glow(hF.g, E.V, b(st, T.ori, 1.2) > 0.05); E.glow(hand, E.V, b(st, T.pos, 1.0) > 0.05);
          E.op(pLab, fr(st, T.pos));
          // 6 × 6：各列框亮、淡下去
          var hlC = function (k, a, z) { Ad.hl('c' + k, 0, k, 5, k, E.HI, E.win(st, a, z, 0.3)); };
          hlC(2, T.c3, T.c4); hlC(3, T.c4, T.c5); hlC(4, T.c5, T.rest);
          Ad.hl('m3', 0, 2, 5, 2, E.J2, E.win(st, T.t3, T.t4, 0.3) * 0.9); Ad.hl('m4', 0, 3, 5, 3, E.J2, E.win(st, T.t4, T.t5, 0.3) * 0.9); Ad.hl('m5', 0, 4, 5, 4, E.J2, E.win(st, T.t5, T.add, 0.3) * 0.9);
          Ad.hl('r4', 3, 0, 3, 5, E.V, E.win(st, T.r4, T.r5, 0.3)); Ad.hl('r5', 4, 0, 4, 5, E.V, E.win(st, T.r5, T.got, 0.3));
          Ad.hl('ll', 3, 0, 5, 2, E.STEP1, E.win(st, T.ll, T.rev, 0.3));
          Ad.hl('d1', 0, 0, 2, 2, E.STEP2, E.win(st, T.diag, T.rev, 0.3)); Ad.hl('d2', 3, 3, 5, 5, E.STEP2, E.win(st, T.diag, T.rev, 0.3));
          // 数：第 3、4、5 列各在讲到时逐个出（之前先灰着），剩下 3 列「剩下三列」时变淡
          var dimRest = fr(st, T.rest);
          for (var i = 0; i < 6; i++) for (var j = 0; j < 6; j++) {
            var c = Ad.cells[i][j], a = 1;
            if (j === 2) a = 0.35 + 0.65 * fr(st, T.n3[i]);
            else if (j === 3) a = 0.35 + 0.65 * fr(st, T.x0 - 0.6 + (i === 4 ? 1.6 : 0));
            else if (j === 4) a = 0.35 + 0.65 * fr(st, T.x1 - 0.5);
            else a = 0.35 + 0.65 * fr(st, T.m66) * (1 - 0.55 * dimRest);
            if (st < T.c3 && (j === 2 || j === 3 || j === 4)) a = 0.35;
            E.op(c, a);
          }
          heads.forEach(function (n, k) { E.op(n, fr(st, T.base6)); E.glow(n, E.HI, (k === 2 && b(st, T.wz, 1.0) > 0.05) || (k === 3 && b(st, T.right, 1.0) > 0.05) || (k === 4 && b(st, T.up, 1.0) > 0.05)); });
          E.op(capT, fr(st, T.base6)); E.op(capB, fr(st, T.m66));
          colN.forEach(function (n, k) { E.op(n, fr(st, T.nth)); var on = (k === 2 && E.win(st, T.c3, T.c4, 0.2) > 0.5) || (k === 3 && E.win(st, T.c4, T.c5, 0.2) > 0.5) || (k === 4 && E.win(st, T.c5, T.rest, 0.2) > 0.5); n.style.fill = on ? H.hi : H.mut; });
          // 乘那一列 → 换成 J_s
          var toJ = fr(st, T.whole);
          E.op(xS, fr(st, T.mul) * (1 - toJ)); E.op(Vs.g, fr(st, T.vec) * (1 - toJ)); E.op(vsL, fr(st, T.vec) * (1 - toJ));
          Vs.hl('a', 2, 0, 2, 0, E.J2, E.win(st, T.t3, T.t4, 0.3)); Vs.hl('b', 3, 0, 3, 0, E.J2, E.win(st, T.t4, T.t5, 0.3)); Vs.hl('c', 4, 0, 4, 0, E.J2, E.win(st, T.t5, T.add, 0.3));
          E.op(eqS, fr(st, T.got) * (1 - toJ)); E.op(Vb.g, fr(st, T.got) * (1 - toJ)); E.op(vbL, fr(st, T.got) * (1 - toJ));
          Vb.hl('a', 3, 0, 4, 0, E.V, E.win(st, T.got, T.whole, 0.3));
          E.op(gJ, toJ); E.op(Jb.g, fr(st, T.jb)); E.op(eqS2, fr(st, T.jb)); E.op(jbL, fr(st, T.jb)); E.op(Js.g, fr(st, T.together)); E.op(xS2, fr(st, T.together)); E.op(jsL, fr(st, T.js));

          /* ── 右栏 ── */
          E.op(gA, 1 - fr(st, T.nth - 0.3));
          E.op(a1.g, fr(st, T.two)); E.op(a2.g, fr(st, T.pos)); E.op(a3.g, fr(st, T.ori)); E.op(a4.g, fr(st, T.ep7)); E.op(a4.parts[0], fr(st, T.ep7));
          E.op(Tm.g, fr(st, T.pose)); E.op(a5.g, fr(st, T.write)); E.op(a5.parts[1], fr(st, T.m66)); E.op(a5.parts[2], fr(st, T.times));
          E.op(gB, E.win(st, T.nth - 0.1, T.mul - 0.2, 0.3));
          E.op(b1.g, fr(st, T.nth)); E.op(b2.g, fr(st, T.base6)); E.op(b3.g, fr(st, T.isOne)); E.op(b3.parts[1], fr(st, T.hand6)); E.op(b3.parts[2], fr(st, T.hand6)); E.op(b4.g, fr(st, T.ep13));
          E.op(gC, E.win(st, T.c3 - 0.1, T.c4 - 0.2, 0.3));
          E.op(c1.g, fr(st, T.c3)); E.op(c1.parts[1], fr(st, T.wz)); E.op(c1.parts[2], fr(st, T.wz)); E.op(c1.parts[3], fr(st, T.one3)); E.op(c2.g, fr(st, T.sh)); E.op(c3.g, fr(st, T.n3[0]));
          E.op(gD, E.win(st, T.c4 - 0.1, T.mul - 0.2, 0.3));
          E.op(d1.g, fr(st, T.c4)); E.op(d1.parts[1], fr(st, T.right)); E.op(mD, fr(st, T.right)); E.op(d2.g, fr(st, T.w0)); E.op(d3.g, fr(st, T.s2)); E.op(d3.parts[2], fr(st, T.x0)); E.op(d3.parts[3], fr(st, T.x0)); E.op(d3.parts[4], fr(st, T.y1)); E.op(d3.parts[5], fr(st, T.ym1));
          E.op(gE, E.win(st, T.c5 - 0.1, T.mul - 0.2, 0.3));
          E.op(e1.g, fr(st, T.c5)); E.op(e1.parts[1], fr(st, T.up)); E.op(mE, fr(st, T.up)); E.op(e2.g, fr(st, T.x1)); E.op(e2.parts[3], fr(st, T.y0)); E.op(e2.parts[4], fr(st, T.y0));
          E.op(gF, E.win(st, T.rest, T.mul - 0.2, 0.3));
          E.op(gG, E.win(st, T.mul - 0.1, T.name66 - 0.2, 0.3));
          E.op(g1.g, fr(st, T.mul)); E.op(g2.g, fr(st, T.t3)); E.op(g3.g, fr(st, T.r4)); E.op(g3.parts[1], fr(st, T.r4s)); E.op(g3.parts[2], fr(st, T.r4r));
          E.op(g4.g, fr(st, T.r5)); E.op(g4.parts[1], fr(st, T.r5s)); E.op(g4.parts[2], fr(st, T.r5r)); E.op(g5.g, fr(st, T.got)); E.op(g6.g, fr(st, T.same));
          E.op(gH, E.win(st, T.each - 0.1, T.name66 - 0.2, 0.3));
          E.op(h1.g, fr(st, T.each)); E.op(h1.parts[1], fr(st, T.twist)); E.op(h1.parts[2], fr(st, T.whole)); E.op(h2, fr(st, T.jb)); E.op(h3.g, fr(st, T.eq)); E.op(h3.parts[3], fr(st, T.mat)); E.op(h3.parts[4], fr(st, T.js));
          E.op(gI, fr(st, T.name66 - 0.1));
          E.op(i1.g, fr(st, T.rep)); E.op(i1b, fr(st, T.rep)); E.op(i2.g, fr(st, T.tw)); E.op(i2.parts[1], fr(st, T.adT));
          E.op(i3.g, fr(st, T.ll)); E.op(i3.parts[1], fr(st, T.pos2)); E.op(i4.g, fr(st, T.step1)); E.op(i5.g, fr(st, T.diag)); E.op(i5.parts[1], fr(st, T.axes)); E.op(i6.g, fr(st, T.axes + 0.6));
          E.op(gK, fr(st, T.rev - 0.1));
          E.op(k1.g, fr(st, T.rev)); E.op(k1.parts[1], fr(st, T.back)); E.op(k2, fr(st, T.adj2)); E.op(k3.g, fr(st, T.undo));
        }
      };
    }
  };
})();
