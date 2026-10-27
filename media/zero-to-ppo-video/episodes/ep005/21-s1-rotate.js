// 005 第 1 幕：旋转矩阵——转动只改朝向。左边一张坐标网格（灰色原网格 + 紫色变形网格，同 003 第 3 幕），右栏放矩阵与要点。
// 句 1 回忆 003：「矩阵」右栏出单位矩阵，「列」两列依次框亮，「小箭头」往右一格（红）、往上一格（绿）各鼓一下，「落到」两个箭头尖标出 (1, 0)、(0, 1)；
// 句 2「不转的时候」「两根小箭头」「留在原地」两根小箭头鼓一下、尖上的坐标亮一下，「一、零和零、一」矩阵里四个数按念到的顺序框一下；
// 句 3 单位矩阵：「左上」「右下」两个 1 框亮黄，「别处都是零」两个 0 框灰，「单位矩阵」矩阵下写名字，「拿它乘谁」右栏写 I p = p、网格上一个点 p，
//    「不变」p 原地亮一下、写「乘谁，谁都不变」；句 4 起淡出；
// 句 4「转起来」紫色网格连同两根小箭头一起转（转一圈多，最后停在 36.87°），箭头尖走过的地方描出一个虚线圆，「半径」写出半径 1；
// 句 5「这里」停住；「零点八」「零点六」往右那根的尖标出坐标、量出横 0.8 竖 0.6；「负零点六」「零点八」往上那根同样标出；
// 句 6「竖着排成两列」右栏换成旋转矩阵：红列先、绿列后，在矩阵里放大弹出，同时网格上对应的坐标鼓一下（不让数飞过网格，同 004 第 2 幕的改法）；
//    「矩阵」加紫框，「旋转矩阵」写出名字；
// 句 7「格子」两根小箭头围成的小方块涂亮黄、原来的小方块画成虚线，「正方形」原点标直角；右栏「只改朝向 / 不改长短 / 不改角度」三行依次打勾，
//    念到「长短」两根小箭头上写「1」，「角度」直角亮一下；
// 句 8「三维」网格换成斜着看的三维小坐标架（x 红、y 绿），「再加一根」长出 z（蓝），「三乘三」右栏换成 3×3 矩阵，
//    「三列」坐标架绕一根斜轴转起来，三列的数跟着实时变、各自框亮。
// 时刻取自配音的逐词时间（分镜时间 = 念到这个词的时刻），见 ../ep005.py。
(function () {
  var E = window.EP005, K = E.K;
  var D = Math.PI / 180, ANG = Math.atan2(0.6, 0.8);         // 36.87°：cos = 0.8、sin = 0.6
  var OX = 200, OY = 220, U = 92;                           // 网格：原点、一格多少 px（x、y 都是 −2…2）
  var MX = 622, MY = 140;                                   // 右栏矩阵中心
  var T = {
    mat: 1.95, col: [2.45, 2.73], arrows: [3.44, 3.6], link: 4.56, out1: 21.2,
    // 句 2、3：单位矩阵——「不转的时候」「两根小箭头」「留在原地」「一 零 零 一」，「左上」「右下」「斜线」「别处」「单位矩阵」「拿它乘谁」「不变」
    still: 6.38, both: 7.62, stay: 8.53, nums: [9.79, 10.19, 10.63, 11.12], tl: 12.77, br: 13.26, diag: 14.23, zero: 15.49,
    unit: 17.32, mul: 18.34, same0: 19.66,
    spin: [22.46, 29.64], tips: 24.99, radius: 26.47, circle: 27.32,
    e1: 30.58, l1: [31.83, 32.66], e2: 33.53, l2: [34.95, 35.91],
    pop: [38.16, 38.54], frame: 40.4, name: 41.3,
    square: 44.03, ghost: 44.33, same: 45.37, right: 46, rows: [46.91, 48.32, 49.47], len: 48.53, ang: 49.71,
    to3: 51.32, z: 52.79, m3: 54.61, cols3: 55.48, turn3: [55.7, 58.2]
  };
  var AX3 = [0.3, 0.25, 1], TURN3 = 48 * D;              // 三维：绕这根斜轴转 48°

  E.scenes[0] = {
    title: '旋转矩阵：转动只改朝向',
    dur: 59.04,
    cues: [
      { at: 0, s: '**矩阵**的每一列 = 一根小箭头最后落到的位置' },
      { at: 6.29, s: '不转的时候：两列是 (1, 0) 和 (0, 1)' },
      { at: 12.43, s: '**单位矩阵**：斜线上是 1、别处是 0，乘谁谁都不变' },
      { at: 21.09, s: '转动：两根小箭头的尖，一直在半径 1 的圆上走' },
      { at: 28.99, s: '往右那根 → (0.8, 0.6)，往上那根 → (−0.6, 0.8)' },
      { at: 37.63, s: '竖着排成两列：这次转动的**旋转矩阵**' },
      { at: 43.25, s: '格子还是一样大的正方形：只改朝向，不改长短和角度' },
      { at: 51.22, s: '三维：三根小箭头 → 3×3 的**旋转矩阵**' }
    ],
    build: function (K) {
      var s = K.sceneSvg('两根小箭头随网格一起转动，箭头尖一直在半径 1 的圆上；转到 36.87 度时，往右那根落在 (0.8, 0.6)，往上那根落在 (−0.6, 0.8)，竖着排成两列就是旋转矩阵；转动后格子还是一样大的正方形，只改朝向、不改长短和角度；三维时有三根小箭头，旋转矩阵是 3×3');

      /* ══ 二维：网格 + 两根小箭头 ══ */
      var flat = E.el('g', {}, s);
      var pl = E.plane(flat, { ox: OX, oy: OY, u: U, x0: -2, x1: 2, y0: -2, y1: 2 });
      var P = pl.P;
      var circle = K.paint(E.el('path', { fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '7 6' }, flat), null, E.HI);
      var radius = E.el('g', {}, flat);
      var rp = P(Math.cos(-50 * D), Math.sin(-50 * D));
      K.paint(E.el('line', { x1: OX, y1: OY, x2: rp[0], y2: rp[1], 'stroke-width': 2.5 }, radius), null, E.HI);
      var rl = P(1.2 * Math.cos(-50 * D), 1.2 * Math.sin(-50 * D));   // 「1」写在圆外：转动的小箭头尖不会扫过它
      E.halo(E.txt(radius, rl[0], rl[1] + 8, '1', 22, E.HI, 'middle', 900));
      // 句 5：两根小箭头围成的小方块（亮黄）、原来的小方块（虚线）、直角、两根上的「1」
      var ghostSq = K.paint(E.el('path', { fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '6 5' }, flat), null, E.INK2);
      var sq = K.paint(E.el('path', { 'stroke-width': 2.5 }, flat), E.tint(E.HI, 26, 'var(--demo-surface)'), E.HI);
      var legs = K.paint(E.el('path', { fill: 'none', 'stroke-width': 2.2, 'stroke-dasharray': '5 5' }, flat), null, E.X);
      var legX = E.halo(E.txt(flat, 0, 0, '0.8', 20, E.X, 'middle', 700)), legY = E.halo(E.txt(flat, 0, 0, '0.6', 20, E.X, 'start', 700));
      var e1 = E.vec(flat, E.X, 5, { head: 16 }), e2 = E.vec(flat, E.Y, 5, { head: 16 });
      var corner = K.paint(E.el('path', { fill: 'none', 'stroke-width': 2.5 }, flat), null, E.INK);
      var ones = [E.X, E.Y].map(function (c) { return E.halo(E.txt(flat, 0, 0, '1', 22, c, 'middle', 900)); });
      var tipDots = [E.X, E.Y].map(function (c) { return K.paint(E.el('circle', { r: 7 }, flat), c); });
      var lab1 = E.halo(E.txt(flat, 0, 0, '', 22, E.X, 'start', 900)), lab2 = E.halo(E.txt(flat, 0, 0, '', 22, E.Y, 'end', 900));

      /* ── 右栏：矩阵（句 1 是单位矩阵，句 4 起是旋转矩阵）、名字、句 5 三行 ── */
      var frame = K.paint(E.el('rect', { x: MX - 112, y: MY - 66, width: 224, height: 132, rx: 14, 'stroke-width': 3 }, flat), E.tint(E.VIOLET, 10, 'var(--demo-surface)'), E.VIOLET);
      var m = E.mat(flat, MX, MY, [1, 0, 0, 1], { size: 34, dx: 90, dy: 50, pad: 50, cw: 80, colors: [E.X, E.Y, E.X, E.Y] });
      var name = E.txt(flat, MX, MY + 104, '旋转矩阵', 26, E.VIOLET, 'middle', 900);
      // 单位矩阵：名字、I p = p、网格上不动的点 p
      var unitLab = E.txt(flat, MX, MY + 104, '单位矩阵 I', 26, E.HI, 'middle', 900);
      var ipp = E.tex(flat, MX, 316, 'I\\,\\textcolor{' + E.HEX.acc + '}{p}=\\textcolor{' + E.HEX.acc + '}{p}', 32, E.INK, 'middle', { w: 220, h: 54 });
      var same0 = E.txt(flat, MX, 372, '乘谁，谁都不变', 24, E.INK, 'middle', 900);
      var PP = P(1.2, 1.3);
      var pDot = K.paint(E.el('circle', { cx: PP[0], cy: PP[1], r: 7 }, flat), E.ACC);
      var pRing = K.paint(E.el('circle', { cx: PP[0], cy: PP[1], r: 10, fill: 'none', 'stroke-width': 3 }, flat), null, E.ACC);
      var pLab = E.halo(E.txt(flat, PP[0] + 14, PP[1] - 10, 'p', 24, E.ACC, 'start', 900));
      var rows = ['只改朝向', '不改长短', '不改角度'].map(function (t, i) {
        var g = E.el('g', {}, flat), y = 300 + 44 * i;
        return { g: g, mark: E.check(g, MX - 92, y - 16, 22, E.GOOD, 4), text: E.txt(g, MX - 58, y + 3, t, 24, E.INK, 'start', 700) };
      });

      /* ══ 三维：斜着看的小坐标架 + 3×3 矩阵 ══ */
      var solid = E.el('g', {}, s);
      var proj = E.view3(128, 24);
      var gz = E.gizmo(solid, 210, 250, 120, proj);
      var m3 = E.mat3(solid, MX, 190, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], { size: 26, dx: 84, dy: 40, colors: [E.X, E.Y, E.Z] });
      var m3lab = E.txt(solid, MX, 300, '3×3 旋转矩阵', 26, E.VIOLET, 'middle', 900);

      function sqPath(a, b) {   // 两根小箭头 a、b（数学坐标）围成的平行四边形
        var p0 = P(0, 0), p1 = P(a[0], a[1]), p2 = P(a[0] + b[0], a[1] + b[1]), p3 = P(b[0], b[1]);
        return 'M ' + [p0, p1, p2, p3].map(function (p) { return p[0].toFixed(1) + ' ' + p[1].toFixed(1); }).join(' L ') + ' Z';
      }
      function bump(st, a, d) { var u = K.seg(st, a, a + d); return u > 0 && u < 1 ? Math.sin(Math.PI * u) : 0; }

      return {
        el: s,
        draw: function (st, t) {
          var f = E.fade;

          /* ── 二维部分：句 6 开头淡出 ── */
          E.op(flat, 1 - f(st, T.to3, 0.5));
          var th = (360 * D + ANG) * f(st, T.spin[0], T.spin[1] - T.spin[0]);
          var R = E.rot(th);
          pl.warp.set(R);
          E.op(pl.warp.g, (0.6 * f(st, T.spin[0] - 0.4, 0.5)).toFixed(3));
          var a = [R[0], R[2]], b = [R[1], R[3]], o = P(0, 0), q1 = P(a[0], a[1]), q2 = P(b[0], b[1]);
          e1.set(o[0], o[1], q1[0], q1[1]); e2.set(o[0], o[1], q2[0], q2[1]);
          var b1 = Math.max(bump(st, T.arrows[0], 0.6), bump(st, T.e1, 0.6), bump(st, T.both, 0.6)), b2 = Math.max(bump(st, T.arrows[1], 0.6), bump(st, T.e2, 0.6), bump(st, T.both, 0.6));
          e1.line.style.strokeWidth = 5 + 3 * b1; e2.line.style.strokeWidth = 5 + 3 * b2;
          E.glow(e1.g, E.X, b1 > 0.05); E.glow(e2.g, E.Y, b2 > 0.05);

          // 句 2：箭头尖走过的地方描出虚线圆（转满一圈后整圈都在）；「尖」两个尖上的圆点；「半径」半径线
          var sweep = Math.min(th, 2 * Math.PI - 1e-4);
          if (sweep > 0.01) {
            var e = P(Math.cos(sweep), Math.sin(sweep));
            circle.setAttribute('d', 'M ' + (OX + U) + ' ' + OY + ' A ' + U + ' ' + U + ' 0 ' + (sweep > Math.PI ? 1 : 0) + ' 0 ' + e[0].toFixed(2) + ' ' + e[1].toFixed(2));
          } else circle.setAttribute('d', '');
          E.op(circle, f(st, T.spin[0], 0.3));
          [q1, q2].forEach(function (q, i) {
            tipDots[i].setAttribute('cx', q[0].toFixed(1)); tipDots[i].setAttribute('cy', q[1].toFixed(1));
            tipDots[i].setAttribute('r', (6 + 2 * Math.sin(2 * Math.PI * t / 1.2)).toFixed(2));
            E.op(tipDots[i], f(st, T.tips, 0.3) * (1 - f(st, T.e1 - 0.3, 0.4)));
          });
          E.op(radius, f(st, T.radius, 0.4) * (1 - f(st, T.e1 - 0.3, 0.4)));

          // 句 1「落到」两个尖标出 (1, 0)、(0, 1)，转起来时淡出；句 3：坐标标签逐个数写出来；往右那根量出横 0.8、竖 0.6
          var pre = st < T.spin[0];
          lab1.textContent = pre ? '(1, 0)' : st < T.l1[1] ? '(0.8,' : '(0.8, 0.6)';
          lab1.setAttribute('x', q1[0] + 14); lab1.setAttribute('y', q1[1] - 10);
          lab2.textContent = pre ? '(0, 1)' : st < T.l2[1] ? '(−0.6,' : '(−0.6, 0.8)';
          lab2.setAttribute('x', q2[0] - 14); lab2.setAttribute('y', q2[1] - 10);
          var preOn = f(st, T.link, 0.3) * (1 - f(st, T.out1, 0.4));
          var labOut = 1 - f(st, T.square - 0.3, 0.4);   // 句 5 的小方块出来前收掉，不压虚线方块
          E.op(lab1, pre ? preOn : f(st, T.l1[0], 0.3) * labOut); E.op(lab2, pre ? preOn : f(st, T.l2[0], 0.3) * labOut);
          var sb = bump(st, T.stay, 0.7);   // 「留在原地」两个尖上的坐标鼓一下；句 6 哪一列弹进矩阵，哪个坐标鼓一下
          E.scaleAt(lab1, q1[0] + 14, q1[1] - 18, 1 + 0.2 * Math.max(sb, bump(st, T.pop[0], 0.7)));
          E.scaleAt(lab2, q2[0] - 14, q2[1] - 18, 1 + 0.2 * Math.max(sb, bump(st, T.pop[1], 0.7)));
          var foot = P(a[0], 0);
          legs.setAttribute('d', 'M ' + q1[0].toFixed(1) + ' ' + q1[1].toFixed(1) + ' L ' + foot[0].toFixed(1) + ' ' + foot[1].toFixed(1));
          legX.setAttribute('x', ((o[0] + foot[0]) / 2).toFixed(1)); legX.setAttribute('y', (OY + 26).toFixed(1));
          // 竖边右边、圆外面
          legY.setAttribute('x', (foot[0] + 20).toFixed(1)); legY.setAttribute('y', ((OY + q1[1]) / 2 + 8).toFixed(1));
          var pl3 = f(st, T.l1[0], 0.4) * (1 - f(st, T.square - 0.3, 0.4));   // 量出来的横竖两边留到句 7
          E.op(legs, pl3); E.op(legX, f(st, T.l1[0], 0.3) * (1 - f(st, T.square - 0.3, 0.4))); E.op(legY, f(st, T.l1[1], 0.3) * (1 - f(st, T.square - 0.3, 0.4)));

          /* ── 右栏矩阵：句 1 单位矩阵（「列」两列框亮），6.4 起淡出；句 4 两个坐标飞进来，换成旋转矩阵 ── */
          var rot = st >= T.pop[0];
          m.set(rot ? [0.8, -0.6, 0.6, 0.8] : [1, 0, 0, 1]);
          var mOn = rot ? 1 : f(st, T.mat, 0.35) * (1 - f(st, T.out1, 0.4));
          E.op(m.g, mOn);
          m.cells.forEach(function (c, k) {   // 句 6：两列先后在矩阵里放大弹出
            var pc = rot ? f(st, T.pop[k % 2], 0.4) : 1, q = m.pos(k);
            E.op(c, pc); E.scaleAt(c, q[0], q[1], 1 + 0.6 * (1 - pc));
          });
          m.colBox(0, E.X, rot ? f(st, T.pop[0], 0.3) : f(st, T.col[0], 0.3) * (1 - f(st, T.still, 0.3)));
          m.colBox(1, E.Y, rot ? f(st, T.pop[1], 0.3) : f(st, T.col[1], 0.3) * (1 - f(st, T.still, 0.3)));
          // 单位矩阵：念到的数各框一下；「左上」「右下」两个 1 留着亮黄框，「别处」两个 0 留着灰框（矩阵里 [左上, 右上, 左下, 右下]）
          var uOut = 1 - f(st, T.out1, 0.4);
          [[0, T.nums[0], T.tl, E.HI], [1, T.nums[2], T.zero, E.MUT], [2, T.nums[1], T.zero, E.MUT], [3, T.nums[3], T.br, E.HI]].forEach(function (d) {
            m.cellBox(d[0], d[3], rot ? 0 : Math.max(bump(st, d[1], 0.6), f(st, d[2], 0.3)) * uOut);
          });
          E.pop(unitLab, MX, MY + 96, f(st, T.unit, 0.35) * uOut, 0.8);
          E.pop(ipp, MX, 306, f(st, T.mul, 0.35) * uOut, 0.85);
          E.op(same0, f(st, T.same0, 0.3) * uOut);
          E.op(pDot, f(st, T.mul, 0.3) * uOut); E.op(pLab, f(st, T.mul, 0.3) * uOut);
          var pu = K.seg(st, T.same0, T.same0 + 0.9);
          pRing.setAttribute('r', (8 + 16 * pu).toFixed(2)); E.op(pRing, pu > 0 && pu < 1 ? (1 - pu) * uOut : 0);
          E.op(frame, f(st, T.frame, 0.35));
          E.pop(name, MX, MY + 96, f(st, T.name, 0.35), 0.8);

          /* ── 句 5：小方块、原来的小方块、直角、「1」、三行打勾 ── */
          sq.setAttribute('d', sqPath(a, b));
          ghostSq.setAttribute('d', sqPath([1, 0], [0, 1]));
          var same = bump(st, T.same, 0.8);
          E.op(sq, f(st, T.square, 0.4) * (0.85 + 0.15 * same));
          E.op(ghostSq, f(st, T.ghost, 0.4));
          var c1 = P(a[0] * 0.2, a[1] * 0.2), c2 = P(a[0] * 0.2 + b[0] * 0.2, a[1] * 0.2 + b[1] * 0.2), c3 = P(b[0] * 0.2, b[1] * 0.2);
          corner.setAttribute('d', 'M ' + c1[0].toFixed(1) + ' ' + c1[1].toFixed(1) + ' L ' + c2[0].toFixed(1) + ' ' + c2[1].toFixed(1) + ' L ' + c3[0].toFixed(1) + ' ' + c3[1].toFixed(1));
          var angB = bump(st, T.ang, 0.9);
          E.op(corner, f(st, T.right, 0.3));
          corner.style.strokeWidth = 2.5 + 2 * angB; E.glow(corner, E.HI, angB > 0.05);
          [a, b].forEach(function (v, i) {   // 「1」写在箭头中段、小方块外面那一侧（数学坐标里的法向，画面上 y 反过来）
            var mid = P(v[0] * 0.55, v[1] * 0.55), n = i ? [-v[1], v[0]] : [v[1], -v[0]];
            ones[i].setAttribute('x', (mid[0] + 18 * n[0]).toFixed(1));
            ones[i].setAttribute('y', (mid[1] - 18 * n[1] + 8).toFixed(1));
            E.op(ones[i], f(st, T.len, 0.3));
          });
          rows.forEach(function (r, i) {
            var p = f(st, T.rows[i], 0.35);
            E.op(r.g, Math.min(1, p * 2)); E.drawOn(r.mark, p);
          });

          /* ── 句 6：三维小坐标架 + 3×3 矩阵 ── */
          var p3 = f(st, T.to3 + 0.2, 0.5);
          E.op(solid, p3);
          var R3 = E.rot3(AX3, TURN3 * f(st, T.turn3[0], T.turn3[1] - T.turn3[0]));
          gz.set(R3);
          E.op(gz.arrows[2].g, f(st, T.z, 0.4)); E.op(gz.labels[2], f(st, T.z, 0.4));
          E.op(gz.box, 0.7 * f(st, T.z + 0.3, 0.5));
          m3.set(R3);
          E.pop(m3.g, MX, 190, f(st, T.m3, 0.35), 0.85);
          [E.X, E.Y, E.Z].forEach(function (c, j) { m3.colBox(j, c, f(st, T.cols3 + 0.2 * j, 0.3)); });
          E.op(m3lab, f(st, T.m3 + 0.2, 0.35));
        }
      };
    }
  };
})();
