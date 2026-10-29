// 片头主图（800×420）：三句旁白各一段——
// ① 左边斜着看的房间里站着 G1（同 005、006 的视角），「朝向」G1 原地往左转 90°（同 006 第 1 幕），骨盆上三根小箭头（前 红、左 绿、上 蓝）跟着转，
//    右栏「朝向 R」的 3×3 跟着实时变、转完是 006 的 [[0, −1, 0], [1, 0, 0], [0, 0, 1]]，顶上写「第 5、6 集：朝向」；
//    「位置」地面网格和地图原点的三根灰轴淡入，紫色虚线箭头从原点画到骨盆，「数字」右栏「位置 p」一列三个数弹出；
// ② 接 006 片尾的预告：「朝向」「位置」右栏两块依次鼓一下，「四乘四的矩阵」两块下面写「拼成 4×4？」，
//    「一个点」地上的杯子（三维的圆柱杯，和房间用同一个投影，见 00-common.js 的 E.cup3）弹出、头上摄像头画一条亮黄虚线视线、杯子旁写「点 ?」，「一个方向」脚边地上一支白箭头（往前）、旁写「方向 ?」，「不同」两个问号鼓一下；
// ③ 「点要先转、再平移」右栏写「点：先转、再平移」，「方向只转、不平移」写「方向：只转、不平移」，问号换成「点」「方向」；
//    「拼成一个四乘四」两对小括号先收起、位置那一列贴到 R 右边、换成一对大括号，最下面一排 0、0、0、1 弹出，「点后面补一」「方向后面补零」各补一行 (x, y, z, 1 / 0)。
// 时刻取自配音的逐词时间（../ep007.py 的 [from, to] 等于每句配音实长，分镜时间 = 念到这个词的时刻）。
(function () {
  var E = window.EP007, H = E.HEX;
  var D = Math.PI / 180;
  var ROOM = { ox: 110, oy: 300, scale: 230, az: 135, el: 18, floor: [-0.1, 0.4, -0.1, 1.7], step: 0.25, at: [0.3, 1.0], yaw: 90 * D, axis: 0.3, margin: 40 };
  var CUP = [-0.05, 1.5], VEL = [[0.45, 1.1], [0.45, 1.55]];
  var MX = 620, MY = 150;
  var T = {
    bot: 0.1, turn: [0.9, 2.4], tag: 1.3, ori: 2.06, room: 7.42, pos: [7.97, 8.6], nums: 9.73,
    both: [13.38, 14.14], ask4: 15.72, cup: 18.41, dir: 19.16, diff: 20.13,
    ptAns: 22.59, dirAns: 24.47, join: [28.06, 28.7], zeros: [28.76, 28.92, 29.08, 29.24], row1: 30.2, row0: 31.38
  };

  E.hero = function (K) {
    var s = K.sceneSvg('片头主图：人形机器人原地往左转 90 度，朝向是一个 3×3 的旋转矩阵，位置是从地图原点指到骨盆的箭头；朝向加上位置怎么拼成一个 4×4 的矩阵，用它换算一个点（摄像头看到的杯子）和一个方向（往前走）有什么不同：点要先转、再平移，方向只转、不平移；拼成 4×4 以后，点后面补 1，方向后面补 0');

    /* ── 左：房间（静态的东西按转完的姿势算） ── */
    var room = E.room(s, ROOM), P = room.P;
    var O0 = P([0, 0, 0]), PEL = P(room.pelvis);
    var pArrow = E.arrow(s, [O0, PEL], E.VIOLET, 3.5, { head: 14, dash: '9 6' });
    var pLab = E.halo(E.txt(s, (O0[0] + PEL[0]) / 2 - 16, (O0[1] + PEL[1]) / 2 - 4, 'p', 28, E.VIOLET, 'middle', 900));
    pLab.style.fontFamily = 'KaTeX_Math, serif'; pLab.style.fontStyle = 'italic';
    var fr = E.frame3(s, 4.5, 13);
    var camAt = room.at('d435_link'), cupAt = P([CUP[0], CUP[1], 0]);
    var cup = E.cup3(s, P, CUP), cupMid = P([CUP[0], CUP[1], 0.065]);
    var sight = K.paint(E.el('path', { d: 'M ' + camAt[0].toFixed(1) + ' ' + camAt[1].toFixed(1) + ' L ' + cup.top[0].toFixed(1) + ' ' + cup.top[1].toFixed(1), fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '6 6' }, s), null, E.HI);
    s.insertBefore(sight, cup.g);   // 视线压在杯子下面，停在杯口
    var v0 = P([VEL[0][0], VEL[0][1], 0]), v1 = P([VEL[1][0], VEL[1][1], 0]);
    var vel = E.vec(s, E.INK, 4.5, { head: 15 }); vel.set(v0[0], v0[1], v1[0], v1[1]);
    var askPt = E.halo(E.txt(s, cupAt[0] + 36, cupAt[1] - 2, '点 ?', 22, E.HI, 'start', 900));
    var askDir = E.halo(E.txt(s, v1[0] + 14, v1[1] + 6, '方向 ?', 22, E.INK, 'start', 900));   // 写在箭头尖右边：左边是脚
    var nPt = E.halo(E.txt(s, cupAt[0] + 36, cupAt[1] - 2, '点', 22, E.HI, 'start', 900));
    var nDir = E.halo(E.txt(s, v1[0] + 14, v1[1] + 6, '方向', 22, E.INK, 'start', 900));

    /* ── 右：朝向 R、位置 p → 4×4（转身时 R 的数留一位小数，按最宽的那组留空） ── */
    var Hs = Math.round(room.H * 100) / 100;
    var Tm = E.matN(s, MX, MY, [[1, 0, 0, ROOM.at[0]], [0, 1, 0, ROOM.at[1]], [0, 0, 1, Hs], [0, 0, 0, 1]], { size: 24, dx: 52, dy: 34,
      wide: [['0.7', '−0.7', '0', Hs], ['0.7', '0.7', '0', 1], ['0', '0', '1', Hs], ['0', '0', '0', '1']],
      color: function (i, j) { return i === 3 ? E.MUT : [E.X, E.Y, E.Z, E.VIOLET][j]; } });
    var bR = Tm.brk(0, 0, 2, 2), bP = Tm.brk(0, 3, 2, 3, E.VIOLET);
    var tag56 = E.txt(s, MX, 36, '第 5、6 集：朝向', 20, E.INK2, 'middle', 700);
    var labR = E.txt(s, Tm.pos(0, 1)[0], 64, '朝向 R', 22, E.INK, 'middle', 900);
    var labP = E.txt(s, Tm.pos(0, 3)[0], 64, '位置 p', 22, E.VIOLET, 'middle', 900);
    var four = E.txt(s, Tm.box[0] - 14, MY + 8, '4×4', 24, E.HI, 'end', 900);
    var ask4 = E.txt(s, MX, 240, '拼成 4×4？', 22, E.HI, 'middle', 900);
    // 下面两段：「点：先转、再平移」+ (x, y, z, 1)，「方向：只转、不平移」+ (x, y, z, 0)
    var rows = [['点：先转、再平移', E.HI, 1, 266], ['方向：只转、不平移', E.INK, 0, 348]].map(function (d) {
      var g = E.el('g', {}, s), h = E.el('g', {}, g), m = E.el('g', {}, g);
      E.txt(h, 488, d[3] + 8, d[0], 24, d[1], 'start', 900);
      E.txt(m, 500, d[3] + 47, '补' + d[2], 22, E.HI, 'start', 900);
      E.tex(m, 640, d[3] + 46, '(x,\\,y,\\,z,\\,\\textcolor{' + H.hi + '}{' + d[2] + '})', 26, E.INK, 'middle', { w: 160, h: 40 });
      return { g: g, h: h, m: m };
    });

    return {
      el: s,
      draw: function (t) {
        var f = E.fade, bump = E.bump;

        /* ① 原地往左转 90°；R 的三列跟着变（转的途中留一位小数） */
        var tp = f(t, T.turn[0], T.turn[1] - T.turn[0]), yaw = 90 * D * tp;
        room.setYaw(yaw);
        fr.set(room.frame(0.32));
        E.op(room.bot, f(t, T.bot, 0.5)); E.op(fr.g, f(t, T.bot + 0.3, 0.4));
        var c = Math.cos(yaw), sn = Math.sin(yaw), rm = [[c, -sn, 0], [sn, c, 0], [0, 0, 1]];
        for (var i = 0; i < 3; i++) for (var j = 0; j < 3; j++) Tm.cells[i][j].textContent = E.num(rm[i][j], tp < 1 ? 1 : 2);
        E.op(tag56, f(t, T.tag, 0.4));
        E.op(room.floor, f(t, T.room, 0.5)); E.op(room.axes, f(t, T.room + 0.2, 0.5));
        E.drawOn(pArrow.line, f(t, T.pos[0], T.pos[1] - T.pos[0] + 0.2)); E.op(pArrow.head, f(t, T.pos[1], 0.2)); E.op(pLab, f(t, T.pos[1], 0.3));

        /* ② 杯子、视线、往前的方向 */
        E.drawOn(sight, f(t, T.cup, 0.6));
        E.pop(cup.g, cupMid[0], cupMid[1], f(t, T.cup, 0.35), 0.6);
        E.op(vel.g, f(t, T.dir, 0.35));
        var aOut1 = 1 - f(t, T.ptAns, 0.3), aOut2 = 1 - f(t, T.dirAns, 0.3), qb = bump(t, T.diff, 0.8);
        E.op(askPt, f(t, T.cup + 0.3, 0.3) * aOut1); E.op(askDir, f(t, T.dir + 0.3, 0.3) * aOut2);
        E.scaleAt(askPt, cupAt[0] + 36, cupAt[1] - 10, 1 + 0.2 * qb); E.scaleAt(askDir, v1[0] + 14, v1[1] - 2, 1 + 0.2 * qb);
        E.op(nPt, f(t, T.ptAns, 0.3)); E.op(nDir, f(t, T.dirAns, 0.3));

        /* 右栏：R 与 p 分开 → 拼成 4×4 */
        var rOn = f(t, T.ori, 0.4), pOn = f(t, T.nums, 0.4), jn = f(t, T.join[0], T.join[1] - T.join[0]);
        var shift = 40 * (1 - jn);
        for (i = 0; i < 4; i++) for (j = 0; j < 4; j++) {
          var cell = Tm.cells[i][j], q = Tm.pos(i, j);
          if (i === 3) { var z = f(t, T.zeros[j], 0.3); E.op(cell, z); E.scaleAt(cell, q[0], q[1], 1 + 0.35 * (1 - z)); continue; }
          if (j === 3) { E.op(cell, pOn); cell.setAttribute('x', (q[0] + shift).toFixed(1)); continue; }
          E.op(cell, rOn);
        }
        // 两对小括号先收起，位置那一列再贴过来，最后出一对大括号（小括号留着会压到挪过来的数）
        E.op(bR, rOn * (1 - f(t, T.join[0] - 0.25, 0.25)));
        bP.setAttribute('transform', 'translate(' + shift.toFixed(1) + ' 0)');
        E.op(bP, f(t, T.pos[1], 0.4) * (1 - f(t, T.join[0] - 0.25, 0.25)));
        E.op(Tm.frame, f(t, T.join[1], 0.3));
        E.op(labR, rOn); E.op(labP, f(t, T.pos[1], 0.4));
        labP.setAttribute('x', (Tm.pos(0, 3)[0] + shift).toFixed(1));
        Tm.hl('r', 0, 0, 2, 2, E.INK2, bump(t, T.both[0], 0.7));
        Tm.hl('p', 0, 3, 2, 3, E.VIOLET, bump(t, T.both[1], 0.7)).setAttribute('transform', 'translate(' + shift.toFixed(1) + ' 0)');   // 框跟着还没贴过来的那一列
        E.op(four, f(t, T.zeros[0], 0.3));
        E.op(ask4, f(t, T.ask4, 0.3) * (1 - f(t, T.join[0], 0.3)));
        E.op(rows[0].h, f(t, T.ptAns, 0.4)); E.op(rows[1].h, f(t, T.dirAns, 0.4));
        E.op(rows[0].m, f(t, T.row1, 0.4)); E.op(rows[1].m, f(t, T.row0, 0.4));
      }
    };
  };
})();
