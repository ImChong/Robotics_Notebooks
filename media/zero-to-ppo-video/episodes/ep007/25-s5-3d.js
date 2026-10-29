// 007 第 5 幕：三维——4×4 的齐次变换矩阵。左边斜着看的房间（同 005、006 的 G1 视角：从右前方看，地图的 x 往左下、y 往右下）：
// 地面网格、地图原点的三根灰轴、站在地图上的 G1；右栏放骨盆的 4×4。
// 句 1「三维」地面和三根轴淡入，「人形机器人」G1 淡入（骨盆上三根小箭头：前 红、左 绿、上 蓝），「往左转九十度」G1 原地往左转 90°（同 006 第 1 幕），
//    「骨盆」骨盆亮一下，「朝向」三根小箭头亮一下，「上一集」右栏写「第 6 集：往左转 90°」，「三乘三的旋转矩阵」右栏出 3×3（三列红、绿、蓝，自带括号），
//    「位置」「三个数」紫色虚线箭头从地图原点画到骨盆，右栏出位置那一列（紫，自带括号）；
// 句 2「拼起来」两对括号先收起、位置那一列贴过来、换成一对大括号，「四乘四」左边写「4×4」，「右边一列是位置」第四列框紫，「最下面一排」最下面一排框亮黄，
//    「零、零、零、一」四个数依次弹出（灰）；
// 句 3「点」右栏写「点 (x, y, z, 1)」，「摄像头」从头上的摄像头（d435）画一条亮黄虚线视线，「杯子」地上的杯子（三维的圆柱杯，和房间用同一个投影，见 00-common.js 的 E.cup3）弹出；
//    「向量」写「向量 (x, y, z, 0)」，「往前走的速度」脚前地上一支白箭头（往前）写「速度」，「往下的重力」机器人旁边一支白箭头往下写「重力」；
// 句 4「所有这样的四乘四矩阵」矩阵外框亮一下，「特殊欧氏群」矩阵下面写「SE(3)：特殊欧氏群」，「旋转再加上平移」补「= 旋转 + 平移」；
// 句 5 右栏换成连乘：「接着乘」写「一节一节接着乘」，「骨盆」「大腿」「小腿」「脚」G1 的骨盆、左大腿、左小腿、左脚依次涂亮黄，每一节长出自己的三根小箭头，右栏对应出一块「4×4」；
// 句 6「按顺序乘起来」几块之间的 × 依次亮，「脚的位姿」脚上的小箭头放大、写「= 脚的位姿」，「正向运动学」出「正运动学」，「第十一集」补「第 11 集」。
// 时刻取自配音的逐词时间（分镜时间 = 念到这个词的时刻），见 ../ep007.py。
(function () {
  var E = window.EP007;
  var D = Math.PI / 180;
  // G1 站在 (0.3, 1)，转完朝地图的 +y（往左转了 90°），R 正是 006 的 [[0, −1, 0], [1, 0, 0], [0, 0, 1]]；margin：原地转身时画布框放宽
  var ROOM = { ox: 110, oy: 300, scale: 230, az: 135, el: 18, floor: [-0.1, 0.4, -0.1, 1.7], step: 0.25, at: [0.3, 1.0], yaw: 90 * D, axis: 0.3, margin: 40 };
  var MX = 620, MY = 158;                                     // 右栏 4×4 的中心
  var CUP = [-0.05, 1.4], VEL = [[0.45, 1.1], [0.45, 1.55]], GRAV = [-0.1, 1.2, [1.05, 0.65]];
  // 大腿取膝关节的父连杆 left_hip_yaw_link（G1 URDF：hip pitch → roll → yaw 三个小连杆，yaw 那节一直到膝盖）
  var LINKS = [['pelvis', 'pelvis', '骨盆'], ['left_hip_yaw_link', 'leftThigh', '大腿'], ['left_knee_link', 'leftShin', '小腿'], ['left_ankle_roll_link', 'leftFoot', '脚']];
  var T = {
    room: 0.1, bot: 1.75, turn: [3.16, 4.3], pelvis: 4.69, ori: 5.17, ep6: 6.08, rot: 6.91, pos: 9.1, pcol: 9.71,
    join: [11.14, 11.78], four: 12.48, col4: 13.89, row: 15.35, zeros: [16.54, 16.99, 17.47, 17.98], ep6Out: 18.78,
    pt: 19.13, cam: 21.77, cup: 23.07, dir: 23.79, vel: 26.07, grav: 26.9,
    all: 28.68, se3: 32.39, plus: 33.25, phaseA: 35.24,
    chain: 38.94, links: [42.16, 42.97, 43.64, 44.35],
    mul: [48.13, 48.36, 48.59], foot: 49.47, fk: 51.36, ep11: 52.52
  };

  E.scenes[4] = {
    title: '三维：4×4 的齐次变换矩阵',
    dur: 54.55,
    cues: [
      { at: 0, s: '三维也一样：G1 往左转 90°，骨盆的朝向就是第 6 集那个 3×3，**位置**是 3 个数' },
      { at: 11.04, s: '拼成 4×4：右边一列是**位置**，最下面一排 0、0、0、1' },
      { at: 19.03, s: '点后面补 1（摄像头看到的杯子）；**向量**后面补 0（速度、重力）' },
      { at: 28.58, s: '所有这样的 4×4 矩阵：特殊欧氏群 SE(3)，旋转再加上平移' },
      { at: 35.44, s: '能接着乘：比如骨盆、大腿、小腿、脚，每一节一个 4×4' },
      { at: 47.37, s: '按顺序乘起来 = 脚的位姿：正运动学（第 11 集）' }
    ],
    build: function (K) {
      var s = K.sceneSvg('三维里人形机器人原地往左转 90 度，骨盆的朝向就是第 6 集那个 3×3 的旋转矩阵，位置是 3 个数，拼起来是 4×4 的齐次变换矩阵：右边一列是位置，最下面一排是 0、0、0、1；点后面补 1，比如头上摄像头看到的杯子；向量后面补 0，比如往前走的速度、往下的重力；所有这样的 4×4 矩阵叫特殊欧氏群 SE(3)，是旋转再加上平移；骨盆、大腿、小腿、脚每一节都有一个 4×4，按顺序乘起来就是脚的位姿，这就是正运动学');
      function sym(parent, x, y, t, size, color) { var n = E.txt(parent, x, y + (size || 26) * 0.34, t, size || 26, color || E.INK, 'middle', 400); n.style.fontFamily = 'KaTeX_Main, serif'; return n; }

      /* ══ 左：三维房间（按转完的姿势算好静态的东西，draw 里再按时间转身） ══ */
      var room = E.room(s, ROOM), P = room.P, v = room.robot;
      var O0 = P([0, 0, 0]), PEL = P(room.pelvis);
      var pArrow = E.arrow(s, [O0, PEL], E.VIOLET, 3.5, { head: 14, dash: '9 6' });
      var pLab = E.halo(E.txt(s, (O0[0] + PEL[0]) / 2 - 16, (O0[1] + PEL[1]) / 2 - 4, 'p', 28, E.VIOLET, 'middle', 900));
      pLab.style.fontFamily = 'KaTeX_Math, serif'; pLab.style.fontStyle = 'italic';
      var fr = E.frame3(s, 4.5, 13);
      // 句 3：杯子（点）、视线；速度、重力（向量）
      var camAt = room.at('d435_link'), cupAt = P([CUP[0], CUP[1], 0]);
      var cup = E.cup3(s, P, CUP), cupMid = P([CUP[0], CUP[1], 0.065]);
      var sight = K.paint(E.el('path', { d: 'M ' + camAt[0].toFixed(1) + ' ' + camAt[1].toFixed(1) + ' L ' + cup.top[0].toFixed(1) + ' ' + cup.top[1].toFixed(1), fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '6 6' }, s), null, E.HI);
      s.insertBefore(sight, cup.g);   // 视线压在杯子下面，停在杯口
      var cupLab = E.halo(E.txt(s, cupAt[0] + 22, cup.top[1] - 8, '杯子', 20, E.HI, 'start', 900));   // 写在杯口右上（视线从左上方来），和下面的「SE(3)」那行隔开
      var v0 = P([VEL[0][0], VEL[0][1], 0]), v1 = P([VEL[1][0], VEL[1][1], 0]);
      var vel = E.vec(s, E.INK, 4.5, { head: 15 }); vel.set(v0[0], v0[1], v1[0], v1[1]);
      var velLab = E.halo(E.txt(s, v1[0] + 14, v1[1] + 6, '速度', 20, E.INK, 'start', 900));   // 写在箭头尖右边：左边是脚
      var g0 = P([GRAV[0], GRAV[1], GRAV[2][0]]), g1 = P([GRAV[0], GRAV[1], GRAV[2][1]]);
      var grav = E.vec(s, E.INK, 4.5, { head: 15 }); grav.set(g0[0], g0[1], g1[0], g1[1]);
      var gravLab = E.halo(E.txt(s, g0[0] + 12, (g0[1] + g1[1]) / 2 + 7, '重力', 20, E.INK, 'start', 900));
      // 句 5：一节一节的小坐标架（转完的姿势）
      var minis = LINKS.map(function (l) { var f3 = E.frame3(s, 3, 9); f3.set(room.linkFrame(l[0], 0.13)); return f3; });

      /* ══ 右栏 ══ */
      var phA = E.el('g', {}, s);
      E.txt(phA, MX, 44, '骨盆的位姿', 22, E.INK2, 'middle', 900);
      var Hs = Math.round(room.H * 100) / 100;
      var Tm = E.matN(phA, MX, MY, [[0, -1, 0, ROOM.at[0]], [1, 0, 0, ROOM.at[1]], [0, 0, 1, Hs], [0, 0, 0, 1]], { size: 24, dx: 50, dy: 34,
        color: function (i, j) { return i === 3 ? E.MUT : [E.X, E.Y, E.Z, E.VIOLET][j]; } });
      var bR = Tm.brk(0, 0, 2, 2), bP = Tm.brk(0, 3, 2, 3, E.VIOLET);
      var labR = E.txt(phA, Tm.pos(0, 1)[0], 70, '旋转矩阵 R', 20, E.INK2, 'middle', 900);
      var labP = E.txt(phA, Tm.pos(0, 3)[0], 70, '位置 p', 20, E.VIOLET, 'middle', 900);
      var four = E.txt(phA, Tm.box[0] - 14, MY + 8, '4×4', 24, E.HI, 'end', 900);
      var ep6 = E.txt(phA, MX, 260, '第 6 集：往左转 90° 的矩阵', 20, E.INK2, 'middle', 700);
      var rowPt = E.el('g', {}, phA), rowDir = E.el('g', {}, phA);
      E.txt(rowPt, 470, 296, '点', 24, E.HI, 'start', 900);
      E.tex(rowPt, 590, 296, '(x,\\,y,\\,z,\\,\\textcolor{' + E.HEX.hi + '}{1})', 26, E.INK, 'middle', { w: 150, h: 40 });
      E.txt(rowPt, 676, 296, '杯子', 20, E.INK2, 'start', 700);
      E.txt(rowDir, 470, 344, '向量', 24, E.INK, 'start', 900);
      E.tex(rowDir, 590, 344, '(x,\\,y,\\,z,\\,\\textcolor{' + E.HEX.hi + '}{0})', 26, E.INK, 'middle', { w: 150, h: 40 });
      var dirEx = E.txt(rowDir, 676, 344, '速度、重力', 20, E.INK2, 'start', 700);
      var se3 = E.txt(phA, MX - 4, 398, 'SE(3)：特殊欧氏群', 24, E.HI, 'end', 900);
      var se3b = E.txt(phA, MX + 8, 398, '= 旋转 + 平移', 22, E.INK, 'start', 700);
      // 句 5、6：连乘
      var phB = E.el('g', {}, s);
      E.txt(phB, MX, 60, '一节一节接着乘', 22, E.INK2, 'middle', 900);
      var chips = LINKS.map(function (l, i) {
        var g = E.el('g', {}, phB), x = 488 + i * 88;
        // 小牌比「4×4」宽出两边各 12 px 左右（同 006「字离框边留足空」），× 号夹在两块中间
        K.paint(E.el('rect', { x: x - 32, y: 104, width: 64, height: 56, rx: 10, 'stroke-width': 2.5 }, g), E.tint(E.HI, 14, 'var(--demo-surface)'), E.HI);
        E.txt(g, x, 138, '4×4', 18, E.INK, 'middle', 900);
        E.txt(g, x, 190, l[2], 22, E.HI, 'middle', 900);
        return { g: g, x: x };
      });
      var timesN = [0, 1, 2].map(function (i) { return sym(phB, 532 + i * 88, 132, '×', 22); });
      var footRes = E.txt(phB, MX, 252, '= 脚的位姿', 28, E.HI, 'middle', 900);
      var fkG = E.el('g', {}, phB);
      E.tag(fkG, MX - 110, 290, 150, 44, '正运动学', E.HI, { size: 24, weight: 900 });
      var ep11 = E.txt(fkG, MX + 54, 320, '第 11 集', 22, E.INK2, 'start', 700);

      return {
        el: s,
        draw: function (st, t) {
          var f = E.fade, bump = E.bump;
          var A = 1 - f(st, T.phaseA, 0.4);

          /* 左：房间、G1（句 1 原地往左转 90°） */
          E.op(room.floor, f(st, T.room, 0.5)); E.op(room.axes, f(st, T.room + 0.2, 0.5));
          E.op(room.bot, f(st, T.bot, 0.5));
          room.setYaw(90 * D * f(st, T.turn[0], T.turn[1] - T.turn[0]));
          fr.set(room.frame(0.32));
          v.clearTint();
          var pb = bump(st, T.pelvis, 0.9);
          if (pb > 0.02) v.tint('pelvis', E.HI, 0.8 * pb);
          LINKS.forEach(function (l, i) {
            var on = f(st, T.links[i], 0.4);
            if (on > 0) v.tint(l[1], E.HI, 0.6 * on);
            E.op(minis[i].g, f(st, T.links[i] + 0.1, 0.35));
          });
          var fb = bump(st, T.foot, 1.0);
          minis[3].arrows.forEach(function (a) { a.line.style.strokeWidth = 3 + 2.5 * fb; });
          E.glow(minis[3].g, E.HI, fb > 0.05);
          E.op(fr.g, f(st, T.bot + 0.3, 0.4) * (1 - f(st, T.chain, 0.4)));
          var ob = bump(st, T.ori, 0.9);
          fr.arrows.forEach(function (a) { a.line.style.strokeWidth = 4.5 + 2.5 * ob; });
          E.drawOn(pArrow.line, f(st, T.pos, 0.7)); E.op(pArrow.head, f(st, T.pos + 0.6, 0.2) * A);
          E.op(pArrow.line, (st >= T.pos ? 1 : 0) * A); E.op(pLab, f(st, T.pos + 0.5, 0.3) * A);
          var pw = bump(st, T.col4, 0.9);
          pArrow.line.style.strokeWidth = 3.5 + 2.5 * pw;
          E.drawOn(sight, f(st, T.cam, 0.6)); E.op(sight, (st >= T.cam ? 1 : 0) * A);
          E.pop(cup.g, cupMid[0], cupMid[1], f(st, T.cup, 0.35) * A, 0.6); E.op(cupLab, f(st, T.cup + 0.2, 0.3) * A);
          E.op(vel.g, f(st, T.vel, 0.35) * A); E.op(velLab, f(st, T.vel + 0.2, 0.3) * A);
          E.op(grav.g, f(st, T.grav, 0.35) * A); E.op(gravLab, f(st, T.grav + 0.2, 0.3) * A);

          /* 右栏 A：4×4 拼起来，点 / 向量，SE(3) */
          E.op(phA, A);
          var rOn = f(st, T.rot, 0.4), pOn = f(st, T.pcol, 0.4), jn = f(st, T.join[0], T.join[1] - T.join[0]);
          var shift = 40 * (1 - jn);
          for (var i = 0; i < 4; i++) for (var j = 0; j < 4; j++) {
            var c = Tm.cells[i][j], q = Tm.pos(i, j);
            if (i === 3) { var z = f(st, T.zeros[j], 0.35); E.op(c, z); E.scaleAt(c, q[0], q[1], 1 + 0.35 * (1 - z)); continue; }
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
          E.op(se3, f(st, T.se3, 0.4)); E.op(se3b, f(st, T.plus, 0.4));
          E.op(ep6, f(st, T.ep6, 0.4) * (1 - f(st, T.ep6Out, 0.4)));

          /* 右栏 B：连乘 */
          E.op(phB, f(st, T.chain, 0.4));
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
