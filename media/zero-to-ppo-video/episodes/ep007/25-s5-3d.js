// 007 第 5 幕：三维——4×4 的齐次变换矩阵。左边斜着看的房间：地面网格、地图原点的三根灰轴、站在地图上的 G1（骨盆朝地图的 −y，正对镜头偏左）；右栏放骨盆的 4×4。
// 句 1「三维」地面和三根轴淡入，「人形机器人」G1 淡入，「骨盆」骨盆亮一下，「朝向」骨盆上长出三根小箭头（前 绿、左 琥珀、上 蓝），
//    「三乘三的旋转矩阵」右栏出 3×3（三列绿、琥珀、蓝，自带括号），「位置」「三个数」紫色虚线箭头从地图原点画到骨盆，右栏出位置那一列（紫，自带括号）；
// 句 2「拼起来」两对括号换成一对、位置那一列贴过来，「四乘四」左边写「4×4」，「右边一列是位置」第四列框紫，「最下面一排」最下面一排框亮黄，
//    「零、零、零、一」四个数依次弹出（灰）；
// 句 3「点」右栏写「点 (x, y, z, 1)」（1 框亮黄），「摄像头」从头上的摄像头（d435）画一条亮黄虚线视线，「杯子」地上的杯子弹出；
//    「方向」写「方向 (x, y, z, 0)」，「往前走的速度」脚前地上一支白箭头（往前）写「速度」，「往下的重力」机器人右边一支白箭头往下写「重力」；
// 句 4「所有这样的四乘四矩阵」矩阵外框亮一下，「特殊欧氏群」矩阵下面写「SE(3)：特殊欧氏群」；
// 句 5 右栏换成连乘：「接着乘」写「接着乘」，「骨盆」「大腿」「小腿」「脚」G1 的骨盆、左大腿、左小腿、左脚依次上色，每一节长出自己的三根小箭头，右栏对应出一块「4×4」；
// 句 6「按顺序乘起来」几块之间的 × 依次亮，「脚的位姿」脚上的小箭头放大、写「= 脚的位姿」，「正向运动学」出「正运动学」，「第十一集」补「第 11 集」。
// 时刻取自配音的逐词时间（分镜时间 = 念到这个词的时刻），见 ../ep007.py。
(function () {
  var E = window.EP007;
  var D = Math.PI / 180;
  var ROOM = { ox: 70, oy: 352, scale: 230, az: 40, el: 18, floor: [-0.15, 1.1, -0.15, 0.85], step: 0.25, at: [0.6, 0.5], yaw: -90 * D, axis: 0.3 };
  var MX = 622, MY = 158;                                     // 右栏 4×4 的中心
  var CUP = [0.52, 0.1], VEL = [[1.08, 0.72], [1.08, 0.3]], GRAV = [1.05, 0.8, [1.0, 0.6]];
  var LINKS = [['pelvis', 'pelvis', E.VIOLET, '骨盆'], ['left_hip_pitch_link', 'leftThigh', E.ACC, '大腿'], ['left_knee_link', 'leftShin', E.GOOD, '小腿'], ['left_ankle_roll_link', 'leftFoot', E.HI, '脚']];
  var T = {
    room: 0.1, bot: 1.75, pelvis: 2.75, frame: 3.4, rot: [3.99, 4.75], pos: 5.78, pcol: 6.33,
    join: [7.76, 8.4], four: 9.1, col4: 10.51, row: 11.97, zeros: [13.16, 13.61, 14.09, 14.6],
    pt: 15.75, one: 17.13, cam: 18.41, cup: 19.72, dir: 20.44, zero: 21.12, vel: 22.7, grav: 23.54,
    all: 25.33, se3: 28.45, phaseA: 30.2,
    chain: 33.74, links: [36.85, 37.68, 38.34, 39.07], each: 39.61,
    mul: [42.46, 42.7, 42.94], foot: 43.8, fk: 46.09, ep11: 47.25
  };

  E.scenes[4] = {
    title: '三维：4×4 的齐次变换矩阵',
    dur: 49.28,
    cues: [
      { at: 0, s: '三维也一样：骨盆的朝向是 3×3 的旋转矩阵，**位置**是 3 个数' },
      { at: 7.66, s: '拼成 4×4：右边一列是**位置**，最下面一排 0、0、0、1' },
      { at: 15.65, s: '点后面补 1（摄像头看到的杯子）；**方向**后面补 0（速度、重力）' },
      { at: 25.23, s: '所有这样的 4×4 矩阵：特殊欧氏群 SE(3)' },
      { at: 30.39, s: '能接着乘：骨盆、大腿、小腿、脚，每一节一个 4×4' },
      { at: 42.1, s: '按顺序乘起来 = 脚的位姿：正运动学（第 11 集）' }
    ],
    build: function (K) {
      var s = K.sceneSvg('三维里人形机器人骨盆的朝向是 3×3 的旋转矩阵，位置是 3 个数，拼起来是 4×4 的齐次变换矩阵：右边一列是位置，最下面一排是 0、0、0、1；点后面补 1，比如头上摄像头看到的杯子；方向后面补 0，比如往前走的速度、往下的重力；所有这样的 4×4 矩阵叫特殊欧氏群 SE(3)；骨盆、大腿、小腿、脚每一节都有一个 4×4，按顺序乘起来就是脚的位姿，这就是正运动学');
      function sym(parent, x, y, t, size, color) { var n = E.txt(parent, x, y + (size || 26) * 0.34, t, size || 26, color || E.INK, 'middle', 400); n.style.fontFamily = 'KaTeX_Main, serif'; return n; }

      /* ══ 左：三维房间 ══ */
      var room = E.room(s, ROOM), P = room.P, v = room.robot;
      var O0 = P([0, 0, 0]), PEL = P(room.pelvis);
      var pArrow = E.arrow(s, [O0, PEL], E.VIOLET, 3.5, { head: 14, dash: '9 6' });
      var pLab = E.halo(E.txt(s, (O0[0] + PEL[0]) / 2 + 2, (O0[1] + PEL[1]) / 2 - 16, 'p', 28, E.VIOLET, 'middle', 900));
      pLab.style.fontFamily = 'KaTeX_Math, serif'; pLab.style.fontStyle = 'italic';
      var fr = E.frame3(s, 4.5, 13);
      fr.set(room.frame(0.32));
      // 句 3：杯子（点）、视线；速度、重力（方向）
      var camAt = room.at('d435_link'), cupAt = P([CUP[0], CUP[1], 0]);
      var sight = K.paint(E.el('path', { d: 'M ' + camAt[0].toFixed(1) + ' ' + camAt[1].toFixed(1) + ' L ' + cupAt[0].toFixed(1) + ' ' + (cupAt[1] - 14).toFixed(1), fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '6 6' }, s), null, E.HI);
      var cup = E.cup(s, cupAt[0], cupAt[1], 30);
      var cupLab = E.halo(E.txt(s, cupAt[0], cupAt[1] + 28, '杯子', 20, E.HI, 'middle', 900));   // 写在杯子下面：右上方有视线穿过
      var v0 = P([VEL[0][0], VEL[0][1], 0]), v1 = P([VEL[1][0], VEL[1][1], 0]);
      var vel = E.vec(s, E.INK, 4.5, { head: 15 }); vel.set(v0[0], v0[1], v1[0], v1[1]);
      var velLab = E.halo(E.txt(s, v1[0] - 12, v1[1] + 7, '速度', 20, E.INK, 'end', 900));
      var g0 = P([GRAV[0], GRAV[1], GRAV[2][0]]), g1 = P([GRAV[0], GRAV[1], GRAV[2][1]]);
      var grav = E.vec(s, E.INK, 4.5, { head: 15 }); grav.set(g0[0], g0[1], g1[0], g1[1]);
      var gravLab = E.halo(E.txt(s, g0[0] + 12, (g0[1] + g1[1]) / 2 + 7, '重力', 20, E.INK, 'start', 900));
      // 句 5：一节一节的小坐标架
      var minis = LINKS.map(function (l) { var f3 = E.frame3(s, 3, 9); f3.set(room.linkFrame(l[0], 0.13)); return f3; });

      /* ══ 右栏 ══ */
      var phA = E.el('g', {}, s);
      E.txt(phA, MX, 44, '骨盆的位姿', 22, E.INK2, 'middle', 900);
      var Hs = Math.round(room.H * 100) / 100;
      var Tm = E.matN(phA, MX, MY, [[0, 1, 0, ROOM.at[0]], [-1, 0, 0, ROOM.at[1]], [0, 0, 1, Hs], [0, 0, 0, 1]], { size: 24, dx: 54, dy: 34, pad: 36, cw: 58,
        color: function (i, j) { return i === 3 ? E.MUT : j === 0 ? E.GOOD : j === 1 ? E.WARN : j === 2 ? E.ACC : E.VIOLET; } });
      var bR = Tm.brk(0, 0, 2, 2), bP = Tm.brk(0, 3, 2, 3, E.VIOLET);
      var labR = E.txt(phA, (Tm.pos(0, 0)[0] + Tm.pos(0, 2)[0]) / 2, 82, '旋转矩阵 R', 20, E.INK2, 'middle', 900);
      var labP = E.txt(phA, Tm.pos(0, 3)[0], 82, '位置 p', 20, E.VIOLET, 'middle', 900);
      var four = E.txt(phA, Tm.box[0] - 14, MY + 8, '4×4', 24, E.HI, 'end', 900);
      var rowPt = E.el('g', {}, phA), rowDir = E.el('g', {}, phA);
      E.txt(rowPt, 470, 296, '点', 24, E.HI, 'start', 900);
      E.tex(rowPt, 590, 296, '(x,\\,y,\\,z,\\,\\textcolor{' + E.HEX.hi + '}{1})', 26, E.INK, 'middle', { w: 150, h: 40 });
      E.txt(rowPt, 676, 296, '杯子', 20, E.INK2, 'start', 700);
      E.txt(rowDir, 470, 344, '方向', 24, E.GOOD, 'start', 900);
      E.tex(rowDir, 590, 344, '(x,\\,y,\\,z,\\,\\textcolor{' + E.HEX.hi + '}{0})', 26, E.INK, 'middle', { w: 150, h: 40 });
      var dirEx = E.txt(rowDir, 676, 344, '速度、重力', 20, E.INK2, 'start', 700);
      var se3 = E.el('g', {}, phA);
      E.txt(se3, MX, 398, 'SE(3)：特殊欧氏群', 24, E.HI, 'middle', 900);
      // 句 5、6：连乘
      var phB = E.el('g', {}, s);
      var chainLab = E.txt(phB, MX, 60, '一节一节接着乘', 22, E.INK2, 'middle', 900);
      var chips = LINKS.map(function (l, i) {
        var g = E.el('g', {}, phB), x = 488 + i * 88;
        K.paint(E.el('rect', { x: x - 29, y: 104, width: 58, height: 56, rx: 10, 'stroke-width': 2.5 }, g), E.tint(l[2], 16, 'var(--demo-surface)'), l[2]);
        E.txt(g, x, 139, '4×4', 20, E.INK, 'middle', 900);
        E.txt(g, x, 190, l[3], 22, l[2], 'middle', 900);
        return { g: g, x: x };
      });
      var timesN = [0, 1, 2].map(function (i) { return sym(phB, 532 + i * 88, 132, '×', 22); });
      var footRes = E.txt(phB, MX, 252, '= 脚的位姿', 28, E.HI, 'middle', 900);
      var fkG = E.el('g', {}, phB);
      E.tag(fkG, MX - 110, 290, 150, 44, '正运动学', E.ACC, { size: 24, weight: 900 });
      var ep11 = E.txt(fkG, MX + 54, 320, '第 11 集', 22, E.INK2, 'start', 700);

      return {
        el: s,
        draw: function (st, t) {
          var f = E.fade, bump = E.bump;
          var A = 1 - f(st, T.phaseA, 0.4);

          /* 左：房间、G1 */
          E.op(room.floor, f(st, T.room, 0.5)); E.op(room.axes, f(st, T.room + 0.2, 0.5));
          E.op(room.bot, f(st, T.bot, 0.5));
          v.clearTint();
          var pb = bump(st, T.pelvis, 0.9);
          if (pb > 0.02) v.tint('pelvis', E.HI, 0.8 * pb);
          LINKS.forEach(function (l, i) {
            var on = f(st, T.links[i], 0.4);
            if (on > 0) v.tint(l[1], l[2], 0.7 * on);
            var mf = f(st, T.links[i] + 0.1, 0.35);
            E.op(minis[i].g, mf);
          });
          var fb = bump(st, T.foot, 1.0);
          minis[3].arrows.forEach(function (a) { a.line.style.strokeWidth = 3 + 2.5 * fb; });
          E.glow(minis[3].g, E.HI, fb > 0.05);
          E.op(fr.g, f(st, T.frame, 0.4) * (1 - f(st, T.chain, 0.4)));
          E.drawOn(pArrow.line, f(st, T.pos, 0.7)); E.op(pArrow.head, f(st, T.pos + 0.6, 0.2) * A);
          E.op(pArrow.line, (st >= T.pos ? 1 : 0) * A); E.op(pLab, f(st, T.pos + 0.5, 0.3) * A);
          var pw = bump(st, T.col4, 0.9);
          pArrow.line.style.strokeWidth = 3.5 + 2.5 * pw;
          E.drawOn(sight, f(st, T.cam, 0.6)); E.op(sight, (st >= T.cam ? 1 : 0) * A);
          E.pop(cup, cupAt[0], cupAt[1] - 12, f(st, T.cup, 0.35) * A, 0.6); E.op(cupLab, f(st, T.cup + 0.2, 0.3) * A);
          E.op(vel.g, f(st, T.vel, 0.35) * A); E.op(velLab, f(st, T.vel + 0.2, 0.3) * A);
          E.op(grav.g, f(st, T.grav, 0.35) * A); E.op(gravLab, f(st, T.grav + 0.2, 0.3) * A);

          /* 右栏 A：4×4 拼起来，点 / 方向，SE(3) */
          E.op(phA, A);
          var rOn = f(st, T.rot[0], 0.4), pOn = f(st, T.pcol, 0.4), jn = f(st, T.join[0], T.join[1] - T.join[0]);
          var shift = 40 * (1 - jn);
          for (var i = 0; i < 4; i++) for (var j = 0; j < 4; j++) {
            var c = Tm.cells[i][j], q = Tm.pos(i, j);
            if (i === 3) { var z = f(st, T.zeros[j], 0.35); E.op(c, z); E.scaleAt(c, q[0], q[1], 1 + 0.6 * (1 - z)); continue; }
            if (j === 3) { E.op(c, pOn); c.setAttribute('x', (q[0] + shift).toFixed(1)); continue; }
            E.op(c, rOn);
          }
          // 两对小括号先收起，位置那一列再贴过来，最后出一对大括号
          E.op(bR, rOn * (1 - f(st, T.join[0] - 0.25, 0.25)));
          bP.setAttribute('transform', 'translate(' + shift.toFixed(1) + ' 0)');
          E.op(bP, pOn * (1 - f(st, T.join[0] - 0.25, 0.25)));
          E.op(Tm.frame, f(st, T.join[1], 0.3));
          var ab = bump(st, T.all, 1.2);
          Tm.frame.style.filter = ab > 0.05 ? 'drop-shadow(0 0 ' + (8 * ab).toFixed(1) + 'px var(--hi))' : '';
          E.op(labR, rOn); E.op(labP, pOn); labP.setAttribute('x', (Tm.pos(0, 3)[0] + shift).toFixed(1));
          E.op(four, f(st, T.four, 0.3));
          Tm.hl('c4', 0, 3, 2, 3, E.VIOLET, f(st, T.col4, 0.3) * (1 - f(st, T.row, 0.3)));
          Tm.hl('row', 3, 0, 3, 3, E.HI, f(st, T.row, 0.3) * (1 - f(st, T.pt, 0.3)));
          E.op(rowPt, f(st, T.pt, 0.4)); E.op(rowDir, f(st, T.dir, 0.4)); E.op(dirEx, f(st, T.vel, 0.3));
          E.op(se3, f(st, T.se3, 0.4));

          /* 右栏 B：连乘 */
          E.op(phB, f(st, T.chain, 0.4));
          E.op(chainLab, f(st, T.chain, 0.4));
          chips.forEach(function (ch, i) { E.pop(ch.g, ch.x, 140, f(st, T.links[i], 0.35), 0.8); });
          timesN.forEach(function (n, i) {
            var b = bump(st, T.mul[i], 0.6), q = [532 + i * 88, 132];
            E.op(n, f(st, T.links[i + 1], 0.3)); E.scaleAt(n, q[0], q[1], 1 + 0.12 * b); n.style.fill = b > 0.1 ? E.HI : E.INK;
          });
          E.op(footRes, f(st, T.foot, 0.4));
          E.op(fkG, f(st, T.fk, 0.4)); E.op(ep11, f(st, T.ep11, 0.3));
        }
      };
    }
  };
})();
