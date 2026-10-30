// 007 第 1 幕：位姿——朝向 + 位置。左边俯视地图（坐标网格，同 003 第 2 幕），右栏两格：位置 p、朝向 R。
// 句 1「平面」地图淡入，「地图」写出「俯视地图」，「扫地机」扫地机在 (3, 1) 弹出（车头朝上），「两样东西」右栏两个虚线空格（各一个问号）；
// 句 2「位置」左格亮起，「原点」原点鼓一下、写「原点」，「中心」扫地机中心套的白环鼓一下，「往右三格」「往上一格」各画一条紫色虚线分量（标 3、1），
//    「这就是它的位置」紫色箭头从原点画到扫地机，左格写出列向量 (3, 1) 和「位置 p」；
// 句 3「朝向」右格亮起，「往前的一根」红色小箭头从扫地机中心长出来（写「前」），「往左的一根」绿色小箭头（写「左」）；
// 句 4（坐标系 = 原点 + 轴）「中心」扫地机中心的白环鼓一下，「小箭头」两根小箭头鼓一下，「坐标系」扫地机连同两根小箭头框一圈虚线、框顶写「扫地机坐标系」（念到坐标前收起），「地图的原点」原点鼓一下，「两根轴」地图的两根轴亮一下，「世界坐标系」y 轴顶上右边写「世界坐标系」；
// 句 5「车头朝上」扫地机亮一下，「世界坐标系」两根轴再亮一下，「往前那根」「零、一」红色尖上写 (0, 1)，「往左那根」「负一、零」绿色尖上写 (−1, 0)；
// 句 6「竖着排成两列」右格的矩阵两列先后弹出（同 005 第 1 幕：在矩阵里放大弹出，网格上对应的坐标鼓一下），「旋转矩阵」写出「朝向 R」，
//    「转九十度」扫地机上画一段从朝右转到朝上的虚线弧，写「90°」（平面里的转动，蓝），矩阵下写「转 90°（第 3 集）」；
// 句 7「合起来叫位姿」两格外面框一个亮黄框，写「位姿 = 位置 + 朝向」，「停在什么位置」紫箭头亮一下，「车头朝着什么方向」两根小箭头亮一下，「说清楚」打勾。
// 时刻取自配音的逐词时间（分镜时间 = 念到这个词的时刻），见 ../ep007.py。
(function () {
  var E = window.EP007;
  var D = Math.PI / 180;
  var O = { ox: 160, oy: 300, u: 60, x0: -2, x1: 4, y0: -1, y1: 4 };
  var VX = 3, VY = 1;                                         // 扫地机停在 (3, 1)，车头朝上
  var SX = [535, 702], SY = 190;                              // 右栏两格：位置 p、朝向 R 的中心
  var T = {
    map: 1.89, bot: 3.04, slots: 5.92,
    pos: 8.45, origin: 9.35, center1: 10.56, g1: [11.35, 11.75], g2: [12.17, 12.7], p: [13.36, 14.11], pvec: 13.93,
    ori: 15.95, fwd: 19.49, left: 20.55,
    bodyO: 23.59, arrowsHi: 24.67, bodyF: 27.63, worldO: 28.93, axesHi: 29.48, worldF: 30.46,
    nose: 32.76, inWorld: 33.91, e1: 35.22, l1: [36.43, 36.87], e2: 37.36, l2: [38.55, 39.2],
    cols: [41.91, 42.64], name: 43.58, ep3: 45.66, turn: [46.49, 47.21], labOut: 44.92,
    sum: [48.81, 49.47], pose: 50.59, where: 51.45, dir: 52.77, ok: 54.37
  };

  E.scenes[0] = {
    title: '位姿：朝向 + 位置',
    dur: 56.05,
    cues: [
      { at: 0, s: '扫地机停在地图上：说清它停成什么样，要 2 样东西' },
      { at: 7.8, s: '**位置**：从原点到扫地机中心，往右 3 格、往上 1 格' },
      { at: 15.14, s: '**朝向**：扫地机自己的 2 根小箭头，往前 1 根、往左 1 根' },
      { at: 22.36, s: '扫地机中心 + 2 根小箭头 = **扫地机坐标系**；地图原点 + 2 根轴 = **世界坐标系**' },
      { at: 32.2, s: '车头朝上，在世界坐标系里：往前那根 → (0, 1)，往左那根 → (−1, 0)' },
      { at: 40.26, s: '竖着排成 2 列 = 旋转矩阵 $R$（第 5 集），正是第 3 集转 90° 的矩阵' },
      { at: 48.71, s: '**位姿** = **位置** + 朝向：停在哪儿、车头朝哪，都说清了' }
    ],
    build: function (K) {
      var s = K.sceneSvg('俯视地图上停着一台扫地机：它的位置是从原点往右 3 格、往上 1 格，记作 p = (3, 1)；它的朝向是自己的两根小箭头，也就是扫地机坐标系（地图的两根轴是世界坐标系），车头朝上时在世界坐标系里往前那根指向 (0, 1)、往左那根指向 (−1, 0)，竖着排成两列就是旋转矩阵 R，正是第 3 集转 90 度的矩阵；位置加朝向合起来叫位姿');

      /* ══ 左：俯视地图 ══ */
      var pl = E.plane(s, O), P = pl.P, box = pl.box;
      var mapLab = E.txt(s, (box[0] + box[2]) / 2, box[3] + 34, '俯视地图', 20, E.INK2, 'middle', 700);
      var o0 = P(0, 0);
      var oRing = K.paint(E.el('circle', { cx: o0[0], cy: o0[1], r: 8, fill: 'none', 'stroke-width': 3 }, s), null, E.VIOLET);
      var oLab = E.halo(E.txt(s, o0[0] - 12, o0[1] + 32, '原点', 20, E.INK2, 'end', 700));
      // 两条虚线分量（往右 3、往上 1）
      function guide(a, b, lab, lx, ly) {
        var g = E.el('g', {}, s), pa = P(a[0], a[1]), pb = P(b[0], b[1]);
        var l = K.paint(E.el('path', { d: 'M ' + pa[0] + ' ' + pa[1] + ' L ' + pb[0] + ' ' + pb[1], fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '8 6', 'stroke-linecap': 'round' }, g), null, E.VIOLET);
        var t = E.halo(E.txt(g, lx, ly, lab, 24, E.VIOLET, 'middle', 900));
        return { g: g, l: l, t: t };
      }
      var gx = guide([0, 0], [3, 0], '3', P(1.5, 0)[0], O.oy + 30);
      var gy = guide([3, 0], [3, 1], '1', P(3, 0)[0] + 22, P(0, 0.25)[1] + 9);   // 贴着 x 轴：再往上会碰到扫地机

      var vac = E.vac(s, P, 0.6);
      vac.set(VX, VY, 90 * D);
      // 位置 p：原点指到扫地机中心的紫箭头，夹在扫地机和它自己的两根小箭头中间
      var pArrow = E.vec(vac.g, E.VIOLET, 5, { head: 16 });
      vac.g.insertBefore(pArrow.g, vac.arrowsG);
      var pc = P(VX, VY);
      pArrow.set(o0[0], o0[1], pc[0], pc[1]);
      var pLab = E.halo(E.txt(s, P(1, 1 / 3)[0] - 6, P(1, 1 / 3)[1] - 16, 'p', 28, E.VIOLET, 'middle', 900));
      pLab.style.fontFamily = 'KaTeX_Math, serif'; pLab.style.fontStyle = 'italic';
      var tips = vac.tips();
      var fwdLab = E.halo(E.txt(s, tips[0][0] - 16, tips[0][1] + 6, '前', 20, E.X, 'end', 900));
      var leftLab = E.halo(E.txt(s, tips[1][0] - 14, tips[1][1] + 7, '左', 20, E.Y, 'end', 900));
      var lab1 = E.halo(E.txt(s, tips[0][0] + 12, tips[0][1] - 10, '(0, 1)', 22, E.X, 'start', 900));
      var lab2 = E.halo(E.txt(s, tips[1][0] - 8, tips[1][1] - 16, '(−1, 0)', 22, E.Y, 'end', 900));
      // 两个坐标系的名字：扫地机坐标系 = 扫地机连同两根小箭头外面一圈虚线框，名字写在框顶上（尖上要写坐标，念到坐标前就收起）；
      // 世界坐标系写在 y 轴顶上右边（这一幕那里空着），一直留着
      var bb = [P(1.3, 2.3), P(3.55, 0.55)];   // 左边让出「左」字
      var bodyG = E.el('g', {}, s);
      K.paint(E.el('path', { d: E.rrect(bb[0][0], bb[0][1], bb[1][0] - bb[0][0], bb[1][1] - bb[0][1], 14), fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '7 6' }, bodyG), null, E.INK);
      var bodyLab = E.halo(E.txt(bodyG, (bb[0][0] + bb[1][0]) / 2, bb[0][1] - 10, '扫地机坐标系', 20, E.INK, 'middle', 900));
      var worldLab = E.halo(E.txt(s, O.ox + 12, P(0, 3.4)[1] + 7, '世界坐标系', 20, E.INK2, 'start', 900));
      // 「转九十度」：从朝右转到朝上的虚线弧
      var AR = 0.82 * O.u, arcPts = [];
      for (var k = 0; k <= 24; k++) { var a = 90 * D * k / 24; arcPts.push([pc[0] + AR * Math.cos(a), pc[1] - AR * Math.sin(a)]); }
      var turnArc = E.arrow(s, arcPts, E.Z, 3, { head: 11, dash: '6 5' });   // 平面里的转动是绕 z 轴转：蓝
      // 「中心」：扫地机中心套一圈白环鼓一下（坐标系 = 原点 + 轴，《Robotic Systems》第 3 章术语表）
      var cRing = K.paint(E.el('circle', { cx: P(VX, VY)[0], cy: P(VX, VY)[1], r: 8, fill: 'none', 'stroke-width': 3 }, s), null, E.INK);
      var turnLab = E.halo(E.txt(s, pc[0] + 1.05 * O.u, pc[1] - 0.62 * O.u, '90°', 22, E.Z, 'start', 900));

      /* ══ 右栏：位置 p、朝向 R 两格 ══ */
      var slots = SX.map(function (x, i) {
        var g = E.el('g', {}, s), w = i ? 170 : 110;
        var r = K.paint(E.el('rect', { x: x - w / 2, y: SY - 66, width: w, height: 132, rx: 14, 'stroke-width': 2.5, 'stroke-dasharray': '8 6' }, g), 'none', E.BORDER);
        var q = E.txt(g, x, SY + 16, '?', 44, E.MUT, 'middle', 900);
        var t = E.txt(g, x, SY - 92, i ? '朝向' : '位置', 24, i ? E.INK : E.VIOLET, 'middle', 900);
        return { g: g, r: r, q: q, t: t };
      });
      var pv = E.matN(s, SX[0], SY, [[3], [1]], { size: 34, dy: 46, color: function () { return E.VIOLET; } });
      var pName = E.txt(s, SX[0], SY + 96, 'p', 30, E.VIOLET, 'middle', 900);
      pName.style.fontFamily = 'KaTeX_Math, serif'; pName.style.fontStyle = 'italic';
      var RO = { size: 34, dx: 64, dy: 46, color: function (i, j) { return j ? E.Y : E.X; } }, re = E.matExt([[0, -1], [1, 0]], RO);
      var Rm = E.matN(s, SX[1] + (re[0] - re[1]) / 2, SY, [[0, -1], [1, 0]], RO);   // 「−1」那列宽：整块挪一下，括号在格子里居中
      var rName = E.txt(s, SX[1], SY + 96, 'R', 30, E.INK, 'middle', 900);
      rName.style.fontFamily = 'KaTeX_Math, serif'; rName.style.fontStyle = 'italic';
      var ep3 = E.txt(s, SX[1], SY + 132, '转 90°（第 3 集）', 20, E.INK2, 'middle', 700);
      // 位姿：两格外面的亮黄框 + 名字 + 勾
      var poseG = E.el('g', {}, s);
      var poseBox = K.paint(E.el('path', { d: E.rrect(462, 70, 330, 280, 18), fill: 'none', 'stroke-width': 3 }, poseG), null, E.HI);
      var poseLab = E.txt(poseG, 620, 396, '位姿 = 位置 + 朝向', 28, E.HI, 'middle', 900);
      var ok = E.check(poseG, 752, 374, 26, E.GOOD, 4.5);

      return {
        el: s,
        draw: function (st, t) {
          var f = E.fade, bump = E.bump;
          E.op(mapLab, f(st, T.map, 0.4));
          E.op(vac.body, f(st, T.bot, 0.4));
          var nose = bump(st, T.nose, 0.8);
          vac.body.style.filter = nose > 0.05 ? 'drop-shadow(0 0 ' + (8 * nose).toFixed(1) + 'px var(--hi))' : '';

          /* 句 2：原点、两条分量、位置箭头 */
          var ob = Math.max(bump(st, T.origin, 0.8), bump(st, T.worldO, 0.8));
          var cb = Math.max(bump(st, T.center1, 0.8), bump(st, T.bodyO, 0.8));
          cRing.setAttribute('r', (8 + 8 * cb).toFixed(2)); E.op(cRing, cb > 0.02 ? 1 : 0);
          E.op(oRing, f(st, T.origin, 0.3)); oRing.setAttribute('r', (8 + 6 * ob).toFixed(2));
          E.op(oLab, f(st, T.origin, 0.3));
          E.drawOn(gx.l, f(st, T.g1[0], T.g1[1] - T.g1[0] + 0.2)); E.op(gx.t, f(st, T.g1[1], 0.3));
          E.drawOn(gy.l, f(st, T.g2[0], T.g2[1] - T.g2[0] + 0.1)); E.op(gy.t, f(st, T.g2[1], 0.3));
          var gOut = 1 - f(st, T.p[1] + 0.4, 0.5);                // 位置箭头画完，两条分量退淡
          E.op(gx.g, 0.35 + 0.65 * gOut); E.op(gy.g, 0.35 + 0.65 * gOut);
          var pp = f(st, T.p[0], T.p[1] - T.p[0]), q = [o0[0] + (pc[0] - o0[0]) * pp, o0[1] + (pc[1] - o0[1]) * pp];
          pArrow.set(o0[0], o0[1], q[0], q[1]);
          var pw = bump(st, T.where, 0.9);
          pArrow.line.style.strokeWidth = 5 + 3 * pw; E.glow(pArrow.g, E.VIOLET, pw > 0.05);
          E.op(pLab, f(st, T.p[1], 0.3));

          /* 句 3、4：两根小箭头长出来，尖上写坐标 */
          var ef = f(st, T.fwd, 0.5), el2 = f(st, T.left, 0.5);
          var c = P(VX, VY);
          vac.arrows[0].set(c[0], c[1], c[0] + (tips[0][0] - c[0]) * ef, c[1] + (tips[0][1] - c[1]) * ef);
          vac.arrows[1].set(c[0], c[1], c[0] + (tips[1][0] - c[0]) * el2, c[1] + (tips[1][1] - c[1]) * el2);
          var ah = bump(st, T.arrowsHi, 0.8);
          var b1 = Math.max(bump(st, T.e1, 0.7), bump(st, T.dir, 0.9), ah), b2 = Math.max(bump(st, T.e2, 0.7), bump(st, T.dir, 0.9), ah);
          vac.arrows[0].line.style.strokeWidth = 5 + 3 * b1; vac.arrows[1].line.style.strokeWidth = 5 + 3 * b2;
          E.glow(vac.arrows[0].g, E.X, b1 > 0.05); E.glow(vac.arrows[1].g, E.Y, b2 > 0.05);
          E.op(fwdLab, f(st, T.fwd + 0.3, 0.3)); E.op(leftLab, f(st, T.left + 0.3, 0.3));
          /* 句 4：两个坐标系 */
          E.op(bodyG, f(st, T.bodyF, 0.3) * (1 - f(st, T.l1[0] - 0.6, 0.4)));
          E.scaleAt(bodyLab, (bb[0][0] + bb[1][0]) / 2, bb[0][1] - 16, 1 + 0.15 * bump(st, T.bodyF + 0.1, 0.6));
          E.op(worldLab, f(st, T.worldF, 0.3)); E.scaleAt(worldLab, O.ox + 60, P(0, 3.4)[1], 1 + 0.15 * bump(st, T.worldF + 0.1, 0.6));
          var mb = Math.max(bump(st, T.axesHi, 0.9), bump(st, T.inWorld, 0.9));
          pl.axes.style.filter = mb > 0.05 ? 'drop-shadow(0 0 ' + (6 * mb).toFixed(1) + 'px #fff)' : '';
          var labOut = 1 - f(st, T.labOut, 0.4);
          lab1.textContent = st < T.l1[1] ? '(0,' : '(0, 1)';
          lab2.textContent = st < T.l2[1] ? '(−1,' : '(−1, 0)';
          E.op(lab1, f(st, T.l1[0], 0.3) * labOut); E.op(lab2, f(st, T.l2[0], 0.3) * labOut);
          E.scaleAt(lab1, tips[0][0] + 12, tips[0][1] - 18, 1 + 0.2 * bump(st, T.cols[0], 0.7));
          E.scaleAt(lab2, tips[1][0] - 8, tips[1][1] - 24, 1 + 0.2 * bump(st, T.cols[1], 0.7));
          E.drawOn(turnArc.line, f(st, T.turn[0], 0.7)); E.op(turnArc.head, f(st, T.turn[0] + 0.6, 0.2));
          E.op(turnLab, f(st, T.turn[0] + 0.5, 0.3));

          /* 右栏：两个空格 → 位置 p、朝向 R */
          slots.forEach(function (sl, i) {
            E.op(sl.g, f(st, T.slots + 0.3 * i, 0.4));
            var on = f(st, i ? T.ori : T.pos, 0.4);
            sl.r.style.stroke = on > 0.5 ? (i ? E.INK2 : E.VIOLET) : E.BORDER;
            E.op(sl.q, 1 - f(st, i ? T.cols[0] - 0.3 : T.pvec - 0.3, 0.3));
          });
          E.pop(pv.g, SX[0], SY, f(st, T.pvec, 0.4), 0.8);
          E.op(pName, f(st, T.pvec + 0.2, 0.3));
          E.op(Rm.frame, f(st, T.cols[0], 0.3));
          Rm.cells.forEach(function (row, i) {
            row.forEach(function (n, j) {
              var pc2 = f(st, T.cols[j], 0.4), q2 = Rm.pos(i, j);
              E.op(n, pc2); E.scaleAt(n, q2[0], q2[1], 1 + 0.35 * (1 - pc2));
            });
          });
          Rm.hl('c0', 0, 0, 1, 0, E.X, f(st, T.cols[0] + 0.3, 0.3) * (1 - f(st, T.name + 0.6, 0.4)));   // 数弹完再框
          Rm.hl('c1', 0, 1, 1, 1, E.Y, f(st, T.cols[1] + 0.3, 0.3) * (1 - f(st, T.name + 0.6, 0.4)));
          E.pop(rName, SX[1], SY + 86, f(st, T.name, 0.35), 0.8);
          E.op(ep3, f(st, T.ep3, 0.3));
          var sb = bump(st, T.sum[0], 0.7), rb = bump(st, T.sum[1], 0.7);
          E.scaleAt(pv.g, SX[0], SY, 1 + 0.1 * sb); E.scaleAt(Rm.g, SX[1], SY, 1 + 0.1 * rb);

          /* 句 6：位姿框 */
          E.drawOn(poseBox, f(st, T.pose - 0.3, 0.6));
          E.op(poseLab, f(st, T.pose, 0.4));
          E.drawOn(ok, f(st, T.ok, 0.4));
        }
      };
    }
  };
})();
