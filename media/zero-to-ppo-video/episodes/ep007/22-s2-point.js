// 007 第 2 幕：点——先转，再平移（路线 §L0 自测题 2 的前一半：点 v' = Rv + p）。左边接着第 1 幕的俯视地图，右栏一行行写算式。
// 句 1「摄像头」扫地机前方张开一片亮黄视野，「袜子」袜子在 (2, 3) 弹出，「正前方两格」从扫地机中心沿红箭头画红色虚线（写「前 2」），
//    「往左一格」再沿绿箭头方向画绿色虚线（写「左 1」），右栏顶上写「扫地机坐标系」和列向量 (2, 1)；
// 句 2「扫地机坐标系」两条虚线鼓一下，「世界坐标系」地图的两根轴亮一下，「什么位置」袜子旁边打问号；
// 句 3 讲「转」要看得见转：「先转」右栏写「① 先转」；「二」「一」「画在原点」先按车头朝右的样子把 (2, 1) 画在原点：往右 2 格红线、往上 1 格绿线（「2」标在红线外侧，「1」标在绿线箭头尖外面，到「加起来」退场）、
//    亮黄箭头指到 (2, 1)（扫地机自己的绿箭头尖正好也在 (2, 1)，这段先退淡）；「扫地机坐标系比世界坐标系转了九十度」扫地机上画一段从朝右转到朝上的蓝虚线弧（写 90°）；
//    「跟着转九十度」原点那一套（红线、绿线、亮黄箭头）绕原点一起转 90°，箭头尖描出一段蓝弧（写 90°），停在 (−1, 2)：红线朝上、绿线朝左；
//    「旋转矩阵」「乘上二、一」右栏出 R 乘 (2, 1)；
// 句 4（新拆出来的一句）「第一列拿两份」R 的第一列框红、朝上的红线亮一下，下一行出 2 (0, 1)；「第二列拿一份」第二列框绿、朝左的绿线亮一下，出 + 1 (−1, 0)；
//    「负一、二」写出结果和 (−1, 2)；
// 句 5「前」「左」扫地机旁的两条虚线依次鼓一下，「上」「左」原点出发的红线、绿线依次鼓一下，「箭头」亮黄箭头亮一下，「原点」原点鼓一下；
// 句 6「再平移」右栏写「② 再平移」，「扫地机的位置」紫箭头从原点画到扫地机，「三、一」出 + (3, 1)，「得到」亮黄箭头整支平移、起点从原点挪到扫地机，
//    箭头尖沿一支紫色虚线从 (−1, 2) 挪到 (2, 3)，「二、三」写出结果；「你看」袜子外一圈亮环，「落在袜子上」打勾；
// 句 7「世界原点」原点鼓一下，「画到袜子」从原点画一支亮黄箭头到袜子（紫色位置箭头、平移过去的亮黄箭头退淡成辅助线），「位置」这支箭头亮一下，
//    「二、三」袜子旁的 (2, 3) 和右栏结果 (2, 3) 鼓一下；
// 句 8 右栏底部框出「点：v′ = Rv + p」，「乘」「旋转矩阵」Rv 亮一下，「加上」「位置」+ p 亮一下。
// 时刻取自配音的逐词时间（分镜时间 = 念到这个词的时刻），见 ../ep007.py。
(function () {
  var E = window.EP007;
  var D = Math.PI / 180;
  var O = { ox: 160, oy: 300, u: 60, x0: -2, x1: 4, y0: -1, y1: 4 };
  var VX = 3, VY = 1, SOCK = [2, 3], RV = [-1, 2];
  var T = {
    cam: 0.83, sock: 2.09, camOut: 2.8, g1: [3.21, 4.04], g2: [4.45, 4.9], seen: 3.9,
    self: 7.37, map: 9.49, ask: 11.18,
    s1: 13.2, draw: [14.28, 14.72, 14.95], vacTurn: 18.61, turn: [20.62, 21.76], rv: 22.65, rvv: 23.71,
    c1: 25.66, two: 26.23, c2: 26.85, one: 27.44, add: 27.91, res: [28.81, 29.5],
    front: 32.22, left: 32.67, up: 34.23, left2: 34.67, arrow: 35.82, origin: 37.05,
    s2: 39.37, p: [40.16, 41.27], pv: 42.01, slide: [42.98, 44.18], res2: [43.4, 43.85], look: 44.65, land: 45.87,
    wOrigin: 47.98, wDraw: [48.7, 49.36], wPos: 52.8, wRes: [53.59, 54.13],
    sum: 56.32, mul: 59.81, plus: 61.18
  };
  var RX = 452;                                              // 右栏左边界

  function clipPoly(pts, b) {   // Sutherland–Hodgman：多边形裁到矩形 b = [x0, y0, x1, y1]
    [[0, 1, b[0]], [0, -1, b[2]], [1, 1, b[1]], [1, -1, b[3]]].forEach(function (c) {
      var k = c[0], sgn = c[1], lim = c[2], out = [];
      function inside(p) { return sgn > 0 ? p[k] >= lim : p[k] <= lim; }
      function cut(a, q) { var t = (lim - a[k]) / (q[k] - a[k]); return [a[0] + (q[0] - a[0]) * t, a[1] + (q[1] - a[1]) * t]; }
      pts.forEach(function (p, i) {
        var q = pts[(i + 1) % pts.length];
        if (inside(p)) { out.push(p); if (!inside(q)) out.push(cut(p, q)); } else if (inside(q)) out.push(cut(p, q));
      });
      pts = out;
    });
    return pts;
  }

  E.scenes[1] = {
    title: '点：先转，再平移',
    dur: 62.76,
    cues: [
      { at: 0, s: '扫地机看到一只袜子：正前方 2 格、往左 1 格 → (2, 1)' },
      { at: 6.14, s: '(2, 1) 是**扫地机坐标系**里的说法：换到**世界坐标系**，袜子在什么位置？' },
      { at: 12.74, s: '① 先转：(2, 1) 先画在原点；扫地机坐标系比世界坐标系转了 90°，箭头也跟着转 90°：$R$ 乘 (2, 1)' },
      { at: 25.56, s: '第 1 列拿 2 份 + 第 2 列拿 1 份 = (−1, 2)' },
      { at: 30.6, s: '扫地机的前、左，换成了地图的上、左——可箭头还从原点出发' },
      { at: 38.86, s: '② 再平移：加上**位置** (3, 1)，得到 (2, 3)，正好落在袜子上' },
      { at: 47.64, s: '世界原点画到袜子的箭头 = 袜子在**世界坐标系**里的位置 (2, 3)' },
      { at: 55.27, s: '**点**从扫地机坐标系换到世界坐标系：先乘 $R$，再加 $p$，即 $v\' = Rv + p$' }
    ],
    build: function (K) {
      var s = K.sceneSvg('扫地机看到袜子在它正前方 2 格、往左 1 格，即 (2, 1)，这是扫地机坐标系里的说法；换到世界坐标系，先转：先把 (2, 1) 按车头朝右画在原点，车头转了 90 度，箭头也跟着绕原点转 90 度，就是旋转矩阵乘 (2, 1)，第一列拿 2 份、第二列拿 1 份，得到 (−1, 2)，箭头还从原点出发；再平移：加上扫地机的位置 (3, 1)，得到 (2, 3)，正好落在袜子上；从世界原点画到袜子的箭头就是袜子在世界坐标系里的位置 (2, 3)；所以点从扫地机坐标系换到世界坐标系是先乘 R、再加 p');

      /* ══ 左：俯视地图 ══ */
      var pl = E.plane(s, O), P = pl.P, box = pl.box;
      E.txt(s, (box[0] + box[2]) / 2, box[3] + 34, '俯视地图', 20, E.INK2, 'middle', 700);
      var o0 = P(0, 0), vc = P(VX, VY), sk = P(SOCK[0], SOCK[1]);
      var oRing = K.paint(E.el('circle', { cx: o0[0], cy: o0[1], r: 8, fill: 'none', 'stroke-width': 3 }, s), null, E.VIOLET);
      E.halo(E.txt(s, o0[0] - 12, o0[1] + 32, '原点', 20, E.INK2, 'end', 700));
      // 摄像头的视野：从车头往上张开 ±35°，裁在地图里
      var A = [VX, VY + 0.3], cone = [A];
      for (var k = 0; k <= 12; k++) { var a = (55 + 70 * k / 12) * D; cone.push([A[0] + 2.3 * Math.cos(a), A[1] + 2.3 * Math.sin(a)]); }
      var conePts = clipPoly(cone.map(function (q) { return P(q[0], q[1]); }), box);
      var coneP = K.paint(E.el('path', { d: 'M ' + conePts.map(function (q) { return q[0].toFixed(1) + ' ' + q[1].toFixed(1); }).join(' L ') + ' Z', 'stroke-width': 2, 'stroke-dasharray': '6 6' }, s),
        E.tint(E.HI, 10, 'var(--demo-surface)'), E.HI);
      coneP.style.fillOpacity = 0.8;
      var sock = E.sock(s, sk[0], sk[1], 30);
      var skRing = K.paint(E.el('circle', { cx: sk[0], cy: sk[1], r: 22, fill: 'none', 'stroke-width': 3 }, s), null, E.HI);
      var ask = E.halo(E.txt(s, sk[0] - 30, sk[1] - 14, '?', 34, E.HI, 'middle', 900));
      var okSock = E.check(s, sk[0] - 52, sk[1] - 20, 24, E.GOOD, 4.5);
      // 扫地机看到的：前 2（红虚线）、左 1（绿虚线）
      function dash(a, b, color) {
        var pa = P(a[0], a[1]), pb = P(b[0], b[1]);
        return K.paint(E.el('path', { d: 'M ' + pa[0] + ' ' + pa[1] + ' L ' + pb[0] + ' ' + pb[1], fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '8 6', 'stroke-linecap': 'round' }, s), null, color);
      }
      var gF = dash([VX, VY], [VX, VY + 2], E.X), gL = dash([VX, VY + 2], SOCK, E.Y);
      var gFl = E.halo(E.txt(s, P(VX, 2)[0] + 14, P(VX, 2)[1] + 8, '前 2', 20, E.X, 'start', 900));
      var gLl = E.halo(E.txt(s, P(2.5, 3)[0] + 4, P(2.5, 3)[1] - 14, '左 1', 20, E.Y, 'middle', 900));
      // 原点那一套：先按车头朝右画 (2, 1)——往右 2（红）、往上 1（绿）、亮黄箭头；跟着车头转 90° 以后就是两份 (0, 1) 和一份 (−1, 0)
      var seg1 = E.vec(s, E.X, 5, { head: 14 }), seg2 = E.vec(s, E.Y, 5, { head: 14 });
      var segL = [E.halo(E.txt(s, 0, 0, '2', 22, E.X, 'middle', 900)), E.halo(E.txt(s, 0, 0, '1', 22, E.Y, 'middle', 900))];
      var R5 = Math.sqrt(5), A0 = Math.atan2(1, 2);
      var tipArc = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '6 5' }, s), null, E.Z);   // 平面里的转动：蓝
      var tipArcLab = E.halo(E.txt(s, P(2.6 * Math.cos(A0 + 45 * D), 0)[0], P(0, 2.6 * Math.sin(A0 + 45 * D))[1], '90°', 22, E.Z, 'middle', 900));
      // 扫地机上：车头从朝右转到朝上的蓝虚线弧（同第 1 幕）
      var vcp = P(VX, VY), VR = 0.82 * O.u, vArcPts = [];
      for (var q = 0; q <= 24; q++) { var aq = 90 * D * q / 24; vArcPts.push([vcp[0] + VR * Math.cos(aq), vcp[1] - VR * Math.sin(aq)]); }
      var vArc = E.arrow(s, vArcPts, E.Z, 3, { head: 11, dash: '6 5' });
      var vArcLab = E.halo(E.txt(s, vcp[0] + 1.05 * O.u, vcp[1] - 0.62 * O.u, '90°', 22, E.Z, 'start', 900));
      var vac = E.vac(s, P, 0.6);
      vac.set(VX, VY, 90 * D);
      var pArrow = E.vec(vac.g, E.VIOLET, 5, { head: 16 });
      vac.g.insertBefore(pArrow.g, vac.arrowsG);
      var slideG = E.el('g', {}, s);
      var tipPath = E.arrow(slideG, [P(RV[0], RV[1]), sk], E.VIOLET, 2.5, { head: 11, dash: '6 5' });
      var big = E.vec(s, E.HI, 5, { head: 16 });
      big.line.style.filter = 'drop-shadow(0 0 2px #000)';
      var rvLab = E.halo(E.txt(s, P(RV[0], RV[1])[0] - 14, P(RV[0], RV[1])[1] - 14, '(−1, 2)', 22, E.HI, 'middle', 900));
      var skLab = E.halo(E.txt(s, sk[0] + 8, sk[1] - 50, '(2, 3)', 22, E.HI, 'middle', 900));
      // 句 7：世界原点 → 袜子（停在袜子边上，不压袜子）= 袜子在世界坐标系里的位置
      var wd = [sk[0] - o0[0], sk[1] - o0[1]], wl = Math.hypot(wd[0], wd[1]), wEnd = [sk[0] - wd[0] / wl * 20, sk[1] - wd[1] / wl * 20];
      var wArrow = E.vec(s, E.HI, 5, { head: 16 });
      wArrow.line.style.filter = 'drop-shadow(0 0 2px #000)';

      /* ══ 右栏：一行行的算式 ══ */
      // 一行一行从左往右排（E.flow：按括号的真实宽度留空，括号和 =、+ 之间留 10）
      function vecO(color) { return { size: 26, color: function (i) { return Array.isArray(color) ? color[i] : color; } }; }
      function col(v) { return v.map(function (a) { return [a]; }); }
      // A：扫地机坐标系里的 (2, 1)
      var rowA = E.el('g', {}, s), fA = E.flow(RX - 6, 58, 10);
      fA.txt(rowA, '扫地机坐标系', 22, E.INK2, 700, 0);
      fA.mat(rowA, col([2, 1]), vecO([E.X, E.Y]));
      E.sock(rowA, fA.x + 24, 64, 26);
      // B：① 先转——R 乘 (2, 1) =，下一行 2 (0, 1) + 1 (−1, 0) = (−1, 2)
      var rowB = E.el('g', {}, s), YB = 150, YB2 = 234, fB = E.flow(RX - 6, YB, 10);
      fB.txt(rowB, '① 先转', 22, E.HI, 900, 0);
      var Rm = fB.mat(rowB, [[0, -1], [1, 0]], { size: 26, dx: 48, color: function (i, j) { return j ? E.Y : E.X; } });
      var Rv = fB.mat(rowB, col([2, 1]), vecO([E.X, E.Y]), 6);
      var eq1 = fB.sym(rowB, '=', 28);
      var termG = [E.el('g', {}, rowB), E.el('g', {}, rowB)], resG = E.el('g', {}, rowB), fB2 = E.flow(RX, YB2, 10);
      fB2.sym(termG[0], '2', 28, E.X, 0);
      fB2.mat(termG[0], col([0, 1]), vecO(E.X), 5);
      fB2.sym(termG[1], '+', 28); fB2.sym(termG[1], '1', 28, E.Y, 8);
      fB2.mat(termG[1], col([-1, 0]), vecO(E.Y), 5);
      fB2.sym(resG, '=', 28);
      var r1 = fB2.mat(resG, col([-1, 2]), vecO(E.HI));
      // C：② 再平移——(−1, 2) + (3, 1) = (2, 3)
      var rowC = E.el('g', {}, s), YC = 320, fC = E.flow(RX - 6, YC, 10);
      fC.txt(rowC, '② 再平移', 22, E.HI, 900, 0);
      fC.mat(rowC, col([-1, 2]), vecO(E.HI), 8);
      var plusG = E.el('g', {}, rowC), res2G = E.el('g', {}, rowC);
      fC.sym(plusG, '+', 28);
      fC.mat(plusG, col([3, 1]), vecO(E.VIOLET));
      fC.sym(res2G, '=', 28);
      var c3 = fC.mat(res2G, col([2, 3]), vecO(E.HI));
      // D：点：v′ = Rv + p
      var rowD = E.el('g', {}, s);
      K.paint(E.el('rect', { x: RX - 6, y: 372, width: 340, height: 46, rx: 12, 'stroke-width': 2.5 }, rowD), E.tint(E.HI, 10, 'var(--demo-surface)'), E.HI);
      E.txt(rowD, RX + 12, 404, '点：', 24, E.HI, 'start', 900);
      var fD = E.tex(rowD, 636, 404, '', 28, E.INK, 'middle', { w: 220, h: 44 });
      // 念到「乘 / 旋转矩阵」时 Rv 亮黄，念到「加上 / 位置」时 + p 亮黄
      function texD(hiR, hiP) {
        return "v'=" + (hiR ? '\\textcolor{' + E.HEX.hi + '}{R\\,v}' : 'R\\,v') + '+' + (hiP ? '\\textcolor{' + E.HEX.hi + '}{p}' : '\\textcolor{' + E.HEX.violet + '}{p}');
      }

      return {
        el: s,
        draw: function (st, t) {
          var f = E.fade, bump = E.bump;

          /* 句 1、2：视野、袜子、两条虚线、问号 */
          E.op(coneP, f(st, T.cam, 0.5) * (1 - f(st, T.camOut, 0.4)));   // 虚线分量画出来之前收起，不压「前 2」「左 1」
          E.pop(sock, sk[0], sk[1], f(st, T.sock, 0.35), 0.6);
          var gOut = 0.3 + 0.7 * (1 - f(st, T.slide[0], 0.4));      // 平移开始后两条虚线退淡
          E.drawOn(gF, f(st, T.g1[0], T.g1[1] - T.g1[0] + 0.2)); E.op(gFl, f(st, T.g1[1], 0.3) * gOut);
          E.drawOn(gL, f(st, T.g2[0], T.g2[1] - T.g2[0] + 0.1)); E.op(gLl, f(st, T.g2[1], 0.3) * gOut);
          gF.style.strokeOpacity = gOut; gL.style.strokeOpacity = gOut;
          var bf = Math.max(bump(st, T.self, 0.8), bump(st, T.front, 0.7)), bl = Math.max(bump(st, T.self, 0.8), bump(st, T.left, 0.7));
          gF.style.strokeWidth = 3 + 2.5 * bf; gL.style.strokeWidth = 3 + 2.5 * bl;
          var mb = bump(st, T.map, 0.9);
          pl.axes.style.filter = mb > 0.05 ? 'drop-shadow(0 0 ' + (6 * mb).toFixed(1) + 'px #fff)' : '';
          E.op(ask, f(st, T.ask, 0.3) * (1 - f(st, T.slide[1] - 0.2, 0.3)));

          /* 句 3：原点那一套先按车头朝右画出来，再跟着车头绕原点转 90°；句 6 亮黄箭头整支平移到扫地机 */
          var th = 90 * D * f(st, T.turn[0], T.turn[1] - T.turn[0]), c = Math.cos(th), sn = Math.sin(th);
          function rot(x, y) { return [c * x - sn * y, sn * x + c * y]; }
          var d1 = f(st, T.draw[0], 0.4), d2 = f(st, T.draw[1], 0.35);
          var e1 = rot(2 * d1, 0), e2 = rot(2, d2), q1 = P(e1[0], e1[1]), q2 = P(rot(2, 0)[0], rot(2, 0)[1]), q3 = P(e2[0], e2[1]);
          seg1.set(o0[0], o0[1], q1[0], q1[1]);
          var sOut = 1 - f(st, T.p[0], 0.5);
          E.op(seg1.g, (d1 > 0 ? 1 : 0) * sOut);
          seg2.set(q2[0], q2[1], q3[0], q3[1]);
          E.op(seg2.g, (d2 > 0 ? 1 : 0) * sOut);
          [[1, -0.32], [2, 1.4]].forEach(function (l, i) {   // 「2」写在红线外侧、「1」写在绿线箭头尖外面（绿线和黄箭头中间太挤），跟着一起转；「加起来」就退场，给 (−1, 2) 让位
            var r = rot(l[0], l[1]), pr = P(r[0], r[1]);
            segL[i].setAttribute('x', pr[0].toFixed(1)); segL[i].setAttribute('y', (pr[1] + 8).toFixed(1));
            E.op(segL[i], f(st, T.draw[i] + 0.2, 0.3) * (1 - f(st, T.add, 0.4)));
          });
          var bu = Math.max(bump(st, T.up, 0.7), bump(st, T.two, 0.8)), bl2 = Math.max(bump(st, T.left2, 0.7), bump(st, T.one, 0.8));
          seg1.line.style.strokeWidth = 5 + 3 * bu; seg2.line.style.strokeWidth = 5 + 3 * bl2;
          // 两段蓝弧：扫地机车头转 90°；箭头尖绕原点转 90°（边转边描）
          E.drawOn(vArc.line, f(st, T.vacTurn, 0.8)); E.op(vArc.head, f(st, T.vacTurn + 0.7, 0.2));
          var arcOut = 1 - f(st, T.add, 0.5);
          E.op(vArc.g, arcOut); E.op(vArcLab, f(st, T.vacTurn + 0.6, 0.3) * arcOut);
          if (th > 0.01) {
            var ae = P(R5 * Math.cos(A0 + th), R5 * Math.sin(A0 + th)), as = P(R5 * Math.cos(A0), R5 * Math.sin(A0));
            tipArc.setAttribute('d', 'M ' + as[0].toFixed(2) + ' ' + as[1].toFixed(2) + ' A ' + (R5 * O.u).toFixed(2) + ' ' + (R5 * O.u).toFixed(2) + ' 0 0 0 ' + ae[0].toFixed(2) + ' ' + ae[1].toFixed(2));
          } else tipArc.setAttribute('d', '');
          E.op(tipArc, arcOut); E.op(tipArcLab, f(st, T.turn[1], 0.3) * arcOut);
          // 扫地机自己的绿箭头尖正好在 (2, 1)：原点那一套没转完之前先退淡，免得看成一回事
          E.op(vac.arrows[1].g, 1 - 0.85 * f(st, T.draw[0] - 0.3, 0.3) * (1 - f(st, T.turn[1], 0.4)));
          var pa = f(st, T.draw[2], 0.4), mv = f(st, T.slide[0], T.slide[1] - T.slide[0]);
          var rv = rot(2, 1), start = [mv * VX, mv * VY], tip = [start[0] + rv[0] * pa, start[1] + rv[1] * pa];
          var ps = P(start[0], start[1]), pt = P(tip[0], tip[1]);
          big.set(ps[0], ps[1], pt[0], pt[1]);
          var aux = 1 - 0.65 * f(st, T.wDraw[0], 0.4);                     // 句 7：画世界原点 → 袜子时，两支箭头退成辅助线
          E.op(big.g, (pa > 0 ? 1 : 0) * aux);
          var ba = bump(st, T.arrow, 0.9);
          big.line.style.strokeWidth = 5 + 3 * ba; E.glow(big.g, E.HI, ba > 0.05);
          E.op(rvLab, f(st, T.res[0], 0.3) * (1 - f(st, T.slide[0], 0.3)));
          E.op(slideG, f(st, T.slide[0] - 0.2, 0.3) * (1 - f(st, T.land + 0.6, 0.5)));
          E.drawOn(tipPath.line, f(st, T.slide[0], T.slide[1] - T.slide[0]));
          E.op(tipPath.head, f(st, T.slide[1] - 0.1, 0.2));
          E.op(skLab, f(st, T.res2[1], 0.3));
          var wr = Math.max(bump(st, T.wRes[0], 0.8), bump(st, T.wRes[1], 0.8));
          E.scaleAt(skLab, sk[0] + 8, sk[1] - 58, 1 + 0.2 * wr);
          var wp = f(st, T.wDraw[0], T.wDraw[1] - T.wDraw[0]);
          wArrow.set(o0[0], o0[1], o0[0] + (wEnd[0] - o0[0]) * wp, o0[1] + (wEnd[1] - o0[1]) * wp);
          E.op(wArrow.g, wp > 0 ? 1 : 0);
          var wb = bump(st, T.wPos, 0.9);
          wArrow.line.style.strokeWidth = 5 + 3 * wb; E.glow(wArrow.g, E.HI, wb > 0.05);
          var ob = Math.max(bump(st, T.origin, 0.8), bump(st, T.wOrigin, 0.8));
          oRing.setAttribute('r', (8 + 6 * ob).toFixed(2)); E.op(oRing, ob > 0.02 ? 1 : 0);

          /* 句 5：位置箭头；袜子亮环、勾 */
          var pp = f(st, T.p[0], T.p[1] - T.p[0]);
          pArrow.set(o0[0], o0[1], o0[0] + (vc[0] - o0[0]) * pp, o0[1] + (vc[1] - o0[1]) * pp);
          E.op(pArrow.g, (pp > 0 ? 1 : 0) * aux);
          var lk = K.seg(st, T.look, T.look + 1.0);
          skRing.setAttribute('r', (18 + 16 * lk).toFixed(2)); E.op(skRing, lk > 0 && lk < 1 ? 1 - lk : 0);
          E.drawOn(okSock, f(st, T.land, 0.4));

          /* 右栏 */
          E.op(rowA, f(st, T.seen, 0.4));
          E.op(rowB, f(st, T.s1, 0.4));
          E.op(Rm.g, f(st, T.rv, 0.3)); E.op(Rv.g, f(st, T.rvv, 0.3)); E.op(eq1, f(st, T.c1 - 0.2, 0.3));
          Rm.hl('c0', 0, 0, 1, 0, E.X, f(st, T.c1, 0.3) * (1 - f(st, T.c2, 0.3)));
          Rm.hl('c1', 0, 1, 1, 1, E.Y, f(st, T.c2, 0.3) * (1 - f(st, T.add, 0.3)));
          Rv.hl('a', 0, 0, 0, 0, E.X, f(st, T.c1, 0.3) * (1 - f(st, T.c2, 0.3)));
          Rv.hl('b', 1, 0, 1, 0, E.Y, f(st, T.c2, 0.3) * (1 - f(st, T.add, 0.3)));
          E.op(termG[0], f(st, T.two, 0.3)); E.op(termG[1], f(st, T.one, 0.3));
          E.op(resG, f(st, T.res[0], 0.3));
          r1.cells[0][0].style.opacity = f(st, T.res[0], 0.3); r1.cells[1][0].style.opacity = f(st, T.res[1], 0.3);
          E.op(rowC, f(st, T.s2, 0.4));
          E.op(plusG, f(st, T.pv - 0.4, 0.3));
          E.op(res2G, f(st, T.res2[0], 0.3));
          c3.cells[0][0].style.opacity = f(st, T.res2[0], 0.3); c3.cells[1][0].style.opacity = f(st, T.res2[1], 0.3);
          c3.hl('w', 0, 0, 1, 0, E.HI, wr);
          E.op(rowD, f(st, T.sum, 0.4));
          fD.setTex(texD(bump(st, T.mul, 1.0) > 0.3, bump(st, T.plus, 0.9) > 0.3));   // setTex 内容没变就不重排
        }
      };
    }
  };
})();
