// 007 第 4 幕：方向——只转，不平移（路线 §L0 自测题 2 的后一半：方向 v' = Rv）。左边接着用俯视地图，右栏分几段换着写。
// 句 1「扫地机往前开」扫地机身后拖出三道速度线，红色小箭头（往前）亮一下、写「往前开」，「一、零」右栏写「开的方向（扫地机坐标系）」和列向量 (1, 0)；
// 句 2 讲清转的是什么：「先转」右栏写「① 只转」和 R 乘 (1, 0)；「扫地机坐标系」扫地机的两根小箭头（往前红、往左绿）亮一下，
//    「世界坐标系」地图的两根轴亮一下；讲「转」要看得见转：「一、零」扫地机中心先按车头朝右画一支红色虚线箭头 (1, 0)（写 (1, 0)，扫地机自己的红箭头先退淡），
//    「跟着车头转九十度」这支箭头绕扫地机中心转 90°，箭头尖描出一段蓝弧（写 90°），正好落在扫地机朝上的红箭头上；
// 句 3「第一列拿一份」第一列框红，「零、一」写出 (0, 1)、下面写「世界坐标系」，「指向上方」红箭头亮一下、尖旁写 (0, 1)，「没错」打勾；
// 句 4「再加上位置」右栏写「要是再加上位置」和 (0, 1) + (3, 1)，「三、二」写出橙色 (3, 2)，「箭头斜着指向右上」从原点画橙箭头到 (3, 2)（红箭头先退淡），
//    「明明是朝上」红箭头回来、亮一下，橙箭头和橙算式打叉；
// 句 5 橙箭头收起，「没有起点」右栏写「方向没有起点」，「停在这里」扫地机连同红箭头滑到 (1, 3)，「停在那里」再滑到 (−1, 0.5)，每停一处留下一支淡红箭头，
//    「都是朝上」三支红箭头一起亮，扫地机滑回 (3, 1)；「只转，不平移」右栏框出「方向：只转，不平移」；
// 句 6（自由向量）右栏上半写「只管长短和方向、不管起点：课本上叫」，「箭头」地图上三支红箭头再亮一下，「自由向量」弹出「自由向量（free vector）」牌子；
// 句 7 右栏换成 T 乘 (1, 0, 0)：「补零」最后的 0 框亮黄，「第三列拿零份」第三列框紫、写「× 0」，「没加进来」写出 (0, 1, 0)；
// 句 8「指到袜子的这支箭头」地图上从扫地机画亮黄虚线箭头到袜子，「也是方向」旁边写「方向」，「两个点相减」右栏写 (2, 3, 1) − (3, 1, 1)，
//    「最后一个数」两个 1 框亮黄，「零」写出 (−1, 2, 0)、最后的 0 框亮黄；
// 句 9 右栏换成总结：「速度」「力」两块小牌，「点后面补一」「方向后面补零」两行依次亮起。
// 时刻取自配音的逐词时间（分镜时间 = 念到这个词的时刻），见 ../ep007.py。
(function () {
  var E = window.EP007;
  var D = Math.PI / 180;
  var O = { ox: 160, oy: 300, u: 60, x0: -2, x1: 4, y0: -1, y1: 4 };
  var VX = 3, VY = 1, SOCK = [2, 3];
  var STOPS = [[VX, VY], [1, 3], [-1, 0.5], [VX, VY]];        // 句 4：扫地机滑过的几处（最后滑回来）
  var T = {
    drive: 2.11, self: 3.1, a1: [5.34, 5.77],
    turn: 6.94, frameB: 10.29, frameW: 11.99, ghost: 13.69, rot: [15.34, 16.24], k1: 17.04, r1: [18.49, 18.97], up: 19.86, ok: 20.54,
    plusP: 22.32, r2: [23.74, 24.24], red: [24.73, 25.5], still: 27.05, phase1Out: 28.73,
    noStart: 29.02, slide: [[31.03, 31.63], [32.28, 32.88], [34.73, 35.53]], same: 33.96, rule: 35.77, freeA: 38.67, arrows2: 41.08, free: 42.63, phase2Out: 43.86,
    homo: 44.5, zero: 46.47, k3: 47.1, res3: 48.39,
    toSock: 51.68, isDir: 53.6, sub: 54.94, lasts: 55.7, res4: [56.82, 58.32], phase4Out: 59.16,
    speed: 59.45, force: 60.22, pt1: 62.96, dir0: 64.16
  };
  var RX = 452;

  E.scenes[3] = {
    title: '方向：只转，不平移',
    dur: 66.24,
    cues: [
      { at: 0, s: '扫地机往前开：在**扫地机坐标系**里，开的**方向**是 (1, 0)' },
      { at: 6.84, s: '先转：乘 $R$，就是把**扫地机坐标系**里的数换成**世界坐标系**里的数——(1, 0) 跟着车头转 90°' },
      { at: 16.94, s: '第 1 列拿 1 份：$R$ 乘 (1, 0) = (0, 1)，朝上，没错' },
      { at: 21.76, s: '要是再加上**位置**：(3, 2)，箭头斜着指向右上——**算错**了' },
      { at: 28.7, s: '**方向**没有起点：停在哪儿，往前开都朝上——只转，**不平移**' },
      { at: 38.25, s: '只管长短和方向、不管起点：课本上叫**自由向量**（free vector）' },
      { at: 44.18, s: '齐次变换：**方向**后面补 0，第 3 列拿 0 份，**位置**没加进来' },
      { at: 50.83, s: '2 个点相减是**方向**：最后一个数 1 − 1 = 0' },
      { at: 59.35, s: '速度、力都是**方向**：点后面补 1，**方向**后面补 0' }
    ],
    build: function (K) {
      var s = K.sceneSvg('扫地机往前开，在扫地机坐标系里开的方向是 (1, 0)；先转：乘旋转矩阵，就是把扫地机坐标系里的数换成世界坐标系里的数，(1, 0) 先按车头朝右画出来，再跟着车头转 90 度，得到世界坐标系里的 (0, 1)，朝上；要是再加上位置就成了 (3, 2)，箭头斜着指向右上，算错了；方向没有起点，扫地机停在哪里往前开都是朝上，所以方向只转、不平移，这种只管长短和方向、不管起点的箭头课本上叫自由向量；用齐次变换算，方向后面补 0，位置一点也没加进来；两个点相减是方向，最后一个数 1 − 1 = 0；速度、力都是方向');
      function vecN(parent, x, y, v, color, size) {
        return E.matN(parent, x, y, v.map(function (a) { return [a]; }), { size: size || 24, color: function (i) { return Array.isArray(color) ? color[i] : color; } });
      }

      /* ══ 左：俯视地图 ══ */
      var pl = E.plane(s, O), P = pl.P, box = pl.box;
      E.txt(s, (box[0] + box[2]) / 2, box[3] + 34, '俯视地图', 20, E.INK2, 'middle', 700);
      var o0 = P(0, 0), sk = P(SOCK[0], SOCK[1]);
      E.halo(E.txt(s, o0[0] - 12, o0[1] + 32, '原点', 20, E.INK2, 'end', 700));
      var sock = E.sock(s, sk[0], sk[1], 30);
      var ghosts = [1, 2].map(function () { var a = E.vec(s, E.X, 4.5, { head: 14 }); return a; });
      var redA = E.vec(s, E.ERR, 5, { head: 16 });
      var redLab = E.halo(E.txt(s, P(3, 2)[0] + 14, P(3, 2)[1] - 8, '(3, 2)', 22, E.ERR, 'start', 900));
      var redX = E.cross(s, P(1.5, 1)[0] - 4, P(1.5, 1)[1] - 22, 10, E.ERR, 4.5);
      var speed = E.el('g', {}, s);
      [-14, 0, 14].forEach(function (dx, i) {
        K.paint(E.el('line', { x1: 0, y1: 0, x2: 0, y2: 22 - 6 * Math.abs(i - 1), 'stroke-width': 3, 'stroke-linecap': 'round', transform: 'translate(' + dx + ' ' + (36 + 4 * Math.abs(i - 1)) + ')' }, speed), null, E.INK2);
      });
      var vac = E.vac(s, P, 0.6);
      vac.set(VX, VY, 90 * D);
      E.op(vac.arrows[1].g, 0);
      var fwdLab = E.halo(E.txt(s, 0, 0, '往前开', 20, E.X, 'start', 900));
      // 句 2：(1, 0) 先按车头朝右画在扫地机中心，再跟着车头转 90°（箭头尖描蓝弧）
      var vcp = P(VX, VY), ghost = E.vec(s, E.X, 4.5, { head: 15, dash: '7 5' });
      var ghostLab = E.halo(E.txt(s, vcp[0] + O.u, vcp[1] - 18, '(1, 0)', 22, E.X, 'middle', 900));
      var gArc = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '6 5' }, s), null, E.Z);   // 平面里的转动：蓝
      var gArcLab = E.halo(E.txt(s, vcp[0] + 0.95 * O.u, vcp[1] - 0.95 * O.u, '90°', 22, E.Z, 'start', 900));
      var upLab = E.halo(E.txt(s, 0, 0, '(0, 1)', 22, E.X, 'end', 900));
      var toSock = E.arrow(s, [P(VX, VY), [sk[0] + 8, sk[1] + 16]], E.HI, 3.5, { head: 14, dash: '8 6' });
      var dirLab = E.halo(E.txt(s, P(2.45, 2.1)[0] - 12, P(2.45, 2.1)[1] + 6, '方向', 20, E.HI, 'end', 900));

      /* ══ 右栏 ══ */
      // 第一段（句 1–3）
      var ph1 = E.el('g', {}, s);
      // 一行一行从左往右排（E.flow：按括号的真实宽度留空）
      function col(v) { return v.map(function (a) { return [a]; }); }
      function vecO(color) { return { size: 24, color: function (i) { return Array.isArray(color) ? color[i] : color; } }; }
      var rowA = E.el('g', {}, ph1), fA = E.flow(RX - 6, 56, 10);
      fA.txt(rowA, '开的方向（扫地机坐标系）', 21, E.INK2, 700, 0);
      var vA = fA.mat(rowA, col([1, 0]), vecO([E.X, E.Y]));
      var rowB = E.el('g', {}, ph1), YB = 150, fB = E.flow(RX - 6, YB, 10);
      fB.txt(rowB, '① 只转', 22, E.GOOD, 900, 0);
      var Rm = fB.mat(rowB, [[0, -1], [1, 0]], { size: 24, dx: 44, color: function (i, j) { return j ? E.Y : E.X; } });
      var vB = fB.mat(rowB, col([1, 0]), vecO([E.X, E.Y]), 6);
      var resB = E.el('g', {}, rowB);
      fB.sym(resB, '=', 28);
      var rB = fB.mat(resB, col([0, 1]), vecO(E.X));   // 第一列
      var okB = E.check(rowB, fB.x + 10, YB - 12, 24, E.GOOD, 4.5);
      var worldTag = E.txt(rowB, (rB.box[0] + rB.box[2]) / 2, rB.box[3] + 30, '世界坐标系', 18, E.INK2, 'middle', 700);   // 结果是世界坐标系里的数
      var rowC = E.el('g', {}, ph1), YC = 262, fC = E.flow(RX - 6, YC, 10);
      E.txt(rowC, RX, YC - 44, '要是再加上位置', 21, E.ERR, 'start', 700);
      fC.mat(rowC, col([0, 1]), vecO(E.X), 0);
      fC.sym(rowC, '+', 28);
      fC.mat(rowC, col([3, 1]), vecO(E.VIOLET));
      var resC = E.el('g', {}, rowC);
      fC.sym(resC, '=', 28);
      var rC = fC.mat(resC, col([3, 2]), vecO(E.ERR));
      var xC = E.cross(rowC, fC.x + 22, YC, 11, E.ERR, 4.5);
      // 第二段（句 4）：方向没有起点 → 只转，不平移
      var ph2 = E.el('g', {}, s);
      E.txt(ph2, 620, 352, '方向没有起点', 24, E.INK, 'middle', 900);
      var ruleG = E.el('g', {}, ph2);
      K.paint(E.el('rect', { x: RX + 4, y: 368, width: 328, height: 46, rx: 12, 'stroke-width': 2.5 }, ruleG), E.tint(E.GOOD, 12, 'var(--demo-surface)'), E.GOOD);
      E.txt(ruleG, 620, 399, '方向：只转，不平移', 24, E.GOOD, 'middle', 900);
      // 句 6：一手资料里的叫法（《Modern Robotics》第 3 章 free vector）；上面一、二行（开的方向、只转）这时还在，写在它们和「方向没有起点」中间
      var freeA = E.txt(ph2, 620, 258, '只管长短和方向、不管起点：课本上叫', 20, E.INK2, 'middle', 700);
      var freeG = E.el('g', {}, ph2);
      K.paint(E.el('rect', { x: RX + 4, y: 270, width: 328, height: 46, rx: 12, 'stroke-width': 2.5 }, freeG), E.tint(E.HI, 12, 'var(--demo-surface)'), E.HI);
      E.txt(freeG, 620, 301, '自由向量（free vector）', 24, E.HI, 'middle', 900);
      // 第三段（句 5）：T 乘 (1, 0, 0)
      var ph3 = E.el('g', {}, s), Y3 = 110, f3 = E.flow(RX - 6, Y3, 10);
      E.txt(ph3, RX, 40, '用齐次变换算', 21, E.INK2, 'start', 700);
      var Tm = f3.mat(ph3, [[0, -1, 3], [1, 0, 1], [0, 0, 1]], { size: 24, dx: 44,
        color: function (i, j) { return i === 2 ? E.MUT : j === 0 ? E.X : j === 1 ? E.Y : E.VIOLET; } }, 0);
      var v3 = f3.mat(ph3, col([1, 0, 0]), vecO([E.X, E.Y, E.HI]), 6);
      var res3 = E.el('g', {}, ph3);
      f3.sym(res3, '=', 28);
      var r3 = f3.mat(res3, col([0, 1, 0]), vecO([E.X, E.X, E.HI]));   // 第一列
      var times0 = E.halo(E.txt(ph3, Tm.pos(2, 2)[0], Tm.box[3] + 32, '× 0', 22, E.VIOLET, 'middle', 900));
      var note3 = E.txt(ph3, 620, 226, '第 3 列拿 0 份：位置没加进来', 21, E.INK, 'middle', 700);
      // 第四段（句 6）：两个点相减（标签放高一点，不贴三排的括号）
      var ph4 = E.el('g', {}, s), Y4 = 334, f4 = E.flow(RX - 6, Y4, 10);
      E.txt(ph4, RX, 262, '2 个点相减', 21, E.HI, 'start', 900);
      var a4 = f4.mat(ph4, col([2, 3, 1]), vecO(E.HI), 0);
      f4.sym(ph4, '−', 28);
      var b4 = f4.mat(ph4, col([3, 1, 1]), vecO(E.VIOLET));
      var res4 = E.el('g', {}, ph4);
      f4.sym(res4, '=', 28);
      var r4 = f4.mat(res4, col([-1, 2, 0]), vecO(E.HI));
      var dirTag = E.txt(ph4, r4.pos(2, 0)[0], r4.box[3] + 24, '方向', 22, E.INK, 'middle', 900);   // 写在结果下面：右边没地方
      // 第五段（句 7）：总结
      var ph5 = E.el('g', {}, s);
      var chips = [['速度', 0], ['力', 1]].map(function (d) { return E.tag(ph5, 470 + d[1] * 104, 40, 90, 38, d[0], E.INK2, { size: 22, weight: 900 }); });
      var chipTo = E.txt(ph5, 682, 66, '→ 方向', 22, E.INK, 'start', 900);
      var rows5 = [{ s: '点', v: ['x', 'y', 1], c: E.HI, y: 190 }, { s: '方向', v: ['x', 'y', 0], c: E.INK, y: 330 }].map(function (d) {
        var g = E.el('g', {}, ph5);
        var r = K.paint(E.el('rect', { x: RX + 4, y: d.y - 66, width: 328, height: 132, rx: 16, 'stroke-width': 2.5 }, g), E.tint(d.c, 10, 'var(--demo-surface)'), d.c);
        E.txt(g, 540, d.y + 9, d.s, 30, d.c, 'middle', 900);
        var m = vecN(g, 660, d.y, d.v, [E.INK, E.INK, E.HI], 26);
        [0, 1].forEach(function (i) { m.cells[i][0].style.fontFamily = 'KaTeX_Math, serif'; m.cells[i][0].style.fontStyle = 'italic'; });
        m.hl('last', 2, 0, 2, 0, E.HI, 1);
        var tail = E.txt(g, 714, d.y + 30, d.v[2] ? '补 1' : '补 0', 22, E.HI, 'start', 900);
        return { g: g, r: r, m: m, tail: tail };
      });

      function placeVac(st) {
        var pos = STOPS[0];
        for (var i = 0; i < 3; i++) {
          var a = T.slide[i], q = E.fade(st, a[0], a[1] - a[0]);
          if (q > 0) pos = [STOPS[i][0] + (STOPS[i + 1][0] - STOPS[i][0]) * q, STOPS[i][1] + (STOPS[i + 1][1] - STOPS[i][1]) * q];
        }
        return pos;
      }

      return {
        el: s,
        draw: function (st, t) {
          var f = E.fade, bump = E.bump;

          /* ══ 地图 ══ */
          var pos = placeVac(st);
          vac.set(pos[0], pos[1], 90 * D);
          var showL = f(st, T.frameB, 0.3) * (1 - f(st, T.ghost - 0.5, 0.3));   // 「扫地机坐标系」：往左那根也亮出来一下
          E.op(vac.arrows[1].g, showL);
          var fb = bump(st, T.frameB, 0.9);
          vac.arrows[1].line.style.strokeWidth = 5 + 3 * fb; E.glow(vac.arrows[1].g, E.Y, fb > 0.05);
          var wb = bump(st, T.frameW, 0.9);
          pl.axes.style.filter = wb > 0.05 ? 'drop-shadow(0 0 ' + (6 * wb).toFixed(1) + 'px #fff)' : '';
          var c = P(pos[0], pos[1]), tip = vac.tips()[0];
          speed.setAttribute('transform', 'translate(' + c[0].toFixed(1) + ' ' + c[1].toFixed(1) + ')');
          var drv = f(st, T.drive, 0.3) * (1 - f(st, T.turn - 0.6, 0.5));
          E.op(speed, drv * (0.7 + 0.3 * Math.sin(2 * Math.PI * t / 0.5)));
          // 红箭头（往前）：句 3 画橙箭头时退淡，「明明是朝上」回来
          // 句 2 转的那一段、句 3 画橙箭头时，扫地机自己的红箭头先退淡
          var gDim = 1 - 0.75 * Math.max(f(st, T.ghost - 0.2, 0.3) * (1 - f(st, T.rot[1], 0.3)), f(st, T.red[0], 0.3) * (1 - f(st, T.still, 0.3)));
          var gth = 90 * D * f(st, T.rot[0], T.rot[1] - T.rot[0]), gt = [vcp[0] + O.u * Math.cos(gth), vcp[1] - O.u * Math.sin(gth)];
          ghost.set(vcp[0], vcp[1], gt[0], gt[1]);
          var gOn = f(st, T.ghost, 0.3) * (1 - f(st, T.rot[1] + 0.1, 0.3));
          E.op(ghost.g, gOn); E.op(ghostLab, gOn * (1 - f(st, T.rot[0], 0.3)));
          if (gth > 0.01) gArc.setAttribute('d', 'M ' + (vcp[0] + O.u).toFixed(2) + ' ' + vcp[1].toFixed(2) + ' A ' + O.u + ' ' + O.u + ' 0 0 0 ' + gt[0].toFixed(2) + ' ' + gt[1].toFixed(2));
          else gArc.setAttribute('d', '');
          var gaOut = 1 - f(st, T.up + 0.6, 0.4);
          E.op(gArc, gaOut); E.op(gArcLab, f(st, T.rot[1], 0.3) * gaOut);
          E.op(vac.arrows[0].g, gDim);
          var gb = Math.max(bump(st, T.drive, 0.8), fb, bump(st, T.up, 0.9), bump(st, T.still, 0.9), bump(st, T.same, 1.0), bump(st, T.arrows2, 0.9));
          vac.arrows[0].line.style.strokeWidth = 5 + 3 * gb; E.glow(vac.arrows[0].g, E.X, gb > 0.05);
          fwdLab.setAttribute('x', (tip[0] + 12).toFixed(1)); fwdLab.setAttribute('y', (tip[1] + 4).toFixed(1));
          E.op(fwdLab, f(st, T.drive, 0.3) * (1 - f(st, T.turn, 0.3)));
          upLab.setAttribute('x', (tip[0] - 12).toFixed(1)); upLab.setAttribute('y', (tip[1] - 8).toFixed(1));   // 往上挪：句 3 的红箭头从左下方过来
          E.op(upLab, f(st, T.up, 0.3) * (1 - f(st, T.phase1Out, 0.3)));
          // 句 3：算错的橙箭头
          var rp = f(st, T.red[0], T.red[1] - T.red[0] + 0.2), r32 = P(3 * rp, 2 * rp);
          redA.set(o0[0], o0[1], r32[0], r32[1]);
          var redOut = 1 - f(st, T.phase1Out, 0.4);
          E.op(redA.g, (rp > 0 ? 1 : 0) * redOut); E.op(redLab, f(st, T.red[1], 0.3) * redOut);
          E.drawOn(redX, f(st, T.still + 0.3, 0.4)); E.op(redX, (st >= T.still + 0.3 ? 1 : 0) * redOut);
          // 句 4：停过的两处留下淡红箭头
          ghosts.forEach(function (g, i) {
            var p0 = P(STOPS[i + 1][0], STOPS[i + 1][1]), p1 = P(STOPS[i + 1][0], STOPS[i + 1][1] + 1);
            g.set(p0[0], p0[1], p1[0], p1[1]);
            var on = f(st, T.slide[i][1], 0.2) * (1 - f(st, T.phase2Out, 0.4));
            var sb = Math.max(bump(st, T.same, 1.0), bump(st, T.arrows2, 0.9));
            E.op(g.g, on * (0.45 + 0.55 * sb)); g.line.style.strokeWidth = 4.5 + 2.5 * sb;
          });
          // 句 6：扫地机指到袜子的箭头
          E.op(sock, 0.45 + 0.55 * f(st, T.toSock, 0.4));
          E.drawOn(toSock.line, f(st, T.toSock, 0.9)); E.op(toSock.head, f(st, T.toSock + 0.8, 0.2));
          var tOut = 1 - f(st, T.phase4Out, 0.4);
          E.op(toSock.g, tOut); E.op(dirLab, f(st, T.isDir, 0.3) * tOut);

          /* ══ 右栏 ══ */
          E.op(ph1, 1 - f(st, T.phase2Out, 0.4));
          E.op(rowA, f(st, T.a1[0] - 0.2, 0.4));
          vA.cells[0][0].style.opacity = f(st, T.a1[0], 0.3); vA.cells[1][0].style.opacity = f(st, T.a1[1], 0.3);
          E.op(rowB, f(st, T.turn, 0.4));
          Rm.hl('c0', 0, 0, 1, 0, E.X, f(st, T.k1, 0.3) * (1 - f(st, T.up, 0.3)));
          vB.hl('a', 0, 0, 0, 0, E.X, f(st, T.k1, 0.3) * (1 - f(st, T.up, 0.3)));
          E.op(resB, f(st, T.r1[0], 0.3));
          rB.cells[0][0].style.opacity = f(st, T.r1[0], 0.3); rB.cells[1][0].style.opacity = f(st, T.r1[1], 0.3);
          E.drawOn(okB, f(st, T.ok, 0.4));
          E.op(worldTag, f(st, T.r1[0], 0.3));
          E.op(rowC, f(st, T.plusP, 0.4) * (1 - f(st, T.phase1Out, 0.4)));
          E.op(resC, f(st, T.r2[0], 0.3));
          rC.cells[0][0].style.opacity = f(st, T.r2[0], 0.3); rC.cells[1][0].style.opacity = f(st, T.r2[1], 0.3);
          E.drawOn(xC, f(st, T.still + 0.3, 0.4));
          E.op(ph2, f(st, T.noStart, 0.4) * (1 - f(st, T.phase2Out, 0.4)));
          E.pop(ruleG, 620, 391, f(st, T.rule, 0.4), 0.85);
          E.op(freeA, f(st, T.freeA, 0.4));
          E.pop(freeG, 620, 293, f(st, T.free, 0.4), 0.85);
          E.op(ph3, f(st, T.homo, 0.4) * (1 - f(st, T.phase4Out, 0.4)));
          var z = f(st, T.zero, 0.35), qz = v3.pos(2, 0);
          E.op(v3.cells[2][0], z); E.scaleAt(v3.cells[2][0], qz[0], qz[1], 1 + 0.6 * (1 - z));
          v3.hl('z', 2, 0, 2, 0, E.HI, f(st, T.zero + 0.3, 0.3));   // 「0」放大弹完再框，弹的时候不碰框边
          Tm.hl('c2', 0, 2, 2, 2, E.VIOLET, f(st, T.k3, 0.3));
          E.op(times0, f(st, T.k3 + 0.3, 0.3));
          E.op(res3, f(st, T.res3, 0.35)); E.op(note3, f(st, T.res3 + 0.3, 0.4));
          r3.hl('z', 2, 0, 2, 0, E.HI, f(st, T.res3, 0.35));
          E.op(ph4, f(st, T.sub, 0.4) * (1 - f(st, T.phase4Out, 0.4)));
          a4.hl('l', 2, 0, 2, 0, E.HI, f(st, T.lasts, 0.3)); b4.hl('l', 2, 0, 2, 0, E.HI, f(st, T.lasts, 0.3));
          E.op(res4, f(st, T.res4[0], 0.3));
          r4.hl('l', 2, 0, 2, 0, E.HI, f(st, T.res4[1], 0.3));
          r4.cells[2][0].style.opacity = f(st, T.res4[1], 0.3);
          E.op(dirTag, f(st, T.res4[1] + 0.3, 0.3));
          E.op(ph5, f(st, T.speed - 0.2, 0.4));
          chips.forEach(function (ch, i) { E.pop(ch.g, 515 + i * 104, 59, f(st, i ? T.force : T.speed, 0.35), 0.8); });
          E.op(chipTo, f(st, T.force + 0.3, 0.3));
          rows5.forEach(function (r, i) {
            var on = f(st, i ? T.dir0 : T.pt1, 0.4);
            E.op(r.g, 0.25 + 0.75 * on);
            E.op(r.tail, on);
          });
        }
      };
    }
  };
})();
