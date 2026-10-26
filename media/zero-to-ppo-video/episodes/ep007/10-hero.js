// 片头主图（800×420）：三句旁白各一段——
// ① 左边斜着看的房间里站着 G1，「朝向」骨盆上三根小箭头（前 绿、左 琥珀、上 蓝）跟着 G1 原地转身，右栏「朝向 R」的 3×3 跟着实时变；
//    「位置」地面网格和地图原点的三根灰轴淡入，紫色虚线箭头从原点画到骨盆，「数字」右栏「位置 p」一列三个数弹出；
// ② 「朝向和位置」右栏两块鼓一下，「一个点」地上的杯子弹出、头上摄像头画一条亮黄虚线视线、杯子旁打问号，「一个方向」脚边地上一支白箭头（往前）、旁边打问号，
//    「地图」三根灰轴亮一下；
// ③ 「点要先转、再平移」杯子旁的问号换成「点：先转、再平移」，「方向只转、不平移」白箭头旁换成「方向：只转」，
//    「拼成一个四乘四」位置那一列贴到 R 右边、两对括号换成一对、最下面一排 0、0、0、1 弹出，「点后面补一」「方向后面补零」右栏下面两行依次写出。
// 时刻取自配音的逐词时间（../ep007.py 的 [from, to] 等于每句配音实长，分镜时间 = 念到这个词的时刻）。
(function () {
  var E = window.EP007, H = E.HEX;
  var D = Math.PI / 180;
  var ROOM = { ox: 70, oy: 352, scale: 230, az: 40, el: 18, floor: [-0.15, 1.1, -0.15, 0.85], step: 0.25, at: [0.6, 0.5], yaw: -90 * D, axis: 0.3, margin: 40 };
  var CUP = [0.52, 0.1], VEL = [[1.08, 0.72], [1.08, 0.3]];
  var MX = 615, MY = 150;
  var T = {
    bot: 0.1, turn: [0.6, 2.4], ori: 1.38, room: 4.49, pos: [5.05, 5.67], nums: 6.8,
    both: [11.52, 12.15], cup: 12.98, dir: 14.2, map: 15.4,
    ptAns: 18.27, dirAns: 20.15, join: [23.74, 24.3], zeros: [24.44, 24.6, 24.76, 24.92], row1: 25.88, row0: 27.06
  };

  E.hero = function (K) {
    var s = K.sceneSvg('片头主图：人形机器人站在地图上，朝向是三根小箭头，位置是从地图原点指到骨盆的箭头；它看到的一个点（杯子）和一个方向（往前走），换到地图上：点要先转、再平移，方向只转、不平移；朝向和位置拼成四乘四的矩阵，点后面补 1，方向后面补 0');

    /* ── 左：房间 ── */
    var room = E.room(s, ROOM), P = room.P;
    var O0 = P([0, 0, 0]), PEL = P(room.pelvis);
    var pArrow = E.arrow(s, [O0, PEL], E.VIOLET, 3.5, { head: 14, dash: '9 6' });
    var pLab = E.halo(E.txt(s, (O0[0] + PEL[0]) / 2 + 2, (O0[1] + PEL[1]) / 2 - 16, 'p', 28, E.VIOLET, 'middle', 900));
    pLab.style.fontFamily = 'KaTeX_Math, serif'; pLab.style.fontStyle = 'italic';
    var fr = E.frame3(s, 4.5, 13);
    var cupAt = P([CUP[0], CUP[1], 0]);
    var sight = K.paint(E.el('path', { fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '6 6' }, s), null, E.HI);
    var cup = E.cup(s, cupAt[0], cupAt[1], 30);
    var v0 = P([VEL[0][0], VEL[0][1], 0]), v1 = P([VEL[1][0], VEL[1][1], 0]);
    var vel = E.vec(s, E.INK, 4.5, { head: 15 }); vel.set(v0[0], v0[1], v1[0], v1[1]);
    var askPt = E.halo(E.txt(s, cupAt[0], cupAt[1] + 28, '点 ?', 22, E.HI, 'middle', 900));   // 写在杯子下面：右上方有视线穿过
    var askDir = E.halo(E.txt(s, v0[0] + 10, v0[1] - 6, '方向 ?', 22, E.INK, 'start', 900));
    var nPt = E.halo(E.txt(s, cupAt[0], cupAt[1] + 28, '点', 22, E.HI, 'middle', 900));
    var nDir = E.halo(E.txt(s, v0[0] + 10, v0[1] - 6, '方向', 22, E.GOOD, 'start', 900));

    /* ── 右：朝向 R、位置 p → 4×4 ── */
    var Tm = E.matN(s, MX, MY, [[1, 0, 0, ROOM.at[0]], [0, 1, 0, ROOM.at[1]], [0, 0, 1, Math.round(room.H * 100) / 100], [0, 0, 0, 1]], { size: 24, dx: 54, dy: 34, pad: 36, cw: 58,
      color: function (i, j) { return i === 3 ? E.MUT : j === 0 ? E.GOOD : j === 1 ? E.WARN : j === 2 ? E.ACC : E.VIOLET; } });
    var bR = Tm.brk(0, 0, 2, 2), bP = Tm.brk(0, 3, 2, 3, E.VIOLET);
    var labR = E.txt(s, Tm.pos(0, 1)[0], 72, '朝向 R', 22, E.INK, 'middle', 900);
    var labP = E.txt(s, Tm.pos(0, 3)[0], 72, '位置 p', 22, E.VIOLET, 'middle', 900);
    var four = E.txt(s, Tm.box[0] - 14, MY + 8, '4×4', 24, E.HI, 'end', 900);
    // 下面两段：「点：先转、再平移」+ (x, y, z, 1)，「方向：只转、不平移」+ (x, y, z, 0)
    var rows = [['点：先转、再平移', E.HI, 1, 266], ['方向：只转、不平移', E.GOOD, 0, 348]].map(function (d) {
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

        /* ① 转身：yaw 从 −40° 转到 −90°；R 的三列跟着变 */
        var yaw = (-40 - 50 * f(t, T.turn[0], T.turn[1] - T.turn[0])) * D;
        room.setYaw(yaw);
        fr.set(room.frame(0.32));
        E.op(room.bot, f(t, T.bot, 0.5)); E.op(fr.g, f(t, T.ori - 0.3, 0.4));
        var c = Math.cos(yaw), sn = Math.sin(yaw), rm = [[c, -sn, 0], [sn, c, 0], [0, 0, 1]];
        for (var i = 0; i < 3; i++) for (var j = 0; j < 3; j++) Tm.cells[i][j].textContent = E.num(rm[i][j]);
        E.op(room.floor, f(t, T.room, 0.5)); E.op(room.axes, f(t, T.room + 0.2, 0.5));
        var mb = bump(t, T.map, 1.0);
        room.axes.style.filter = mb > 0.05 ? 'drop-shadow(0 0 ' + (6 * mb).toFixed(1) + 'px #fff)' : '';
        E.drawOn(pArrow.line, f(t, T.pos[0], T.pos[1] - T.pos[0] + 0.2)); E.op(pArrow.head, f(t, T.pos[1], 0.2)); E.op(pLab, f(t, T.pos[1], 0.3));

        /* ② 杯子、视线、往前的方向 */
        var cam = room.at('d435_link');
        sight.setAttribute('d', 'M ' + cam[0].toFixed(1) + ' ' + cam[1].toFixed(1) + ' L ' + cupAt[0].toFixed(1) + ' ' + (cupAt[1] - 14).toFixed(1));
        E.drawOn(sight, f(t, T.cup, 0.6));
        E.pop(cup, cupAt[0], cupAt[1] - 12, f(t, T.cup, 0.35), 0.6);
        E.op(vel.g, f(t, T.dir, 0.35));
        var aOut1 = 1 - f(t, T.ptAns, 0.3), aOut2 = 1 - f(t, T.dirAns, 0.3);
        E.op(askPt, f(t, T.cup + 0.3, 0.3) * aOut1); E.op(askDir, f(t, T.dir + 0.3, 0.3) * aOut2);
        E.op(nPt, f(t, T.ptAns, 0.3)); E.op(nDir, f(t, T.dirAns, 0.3));

        /* 右栏：R 与 p 分开 → 拼成 4×4 */
        var rOn = f(t, T.ori, 0.4), pOn = f(t, T.nums, 0.4), jn = f(t, T.join[0], T.join[1] - T.join[0]);
        var shift = 40 * (1 - jn), both = Math.max(bump(t, T.both[0], 0.7), 0), bothP = bump(t, T.both[1], 0.7);
        for (i = 0; i < 4; i++) for (j = 0; j < 4; j++) {
          var cell = Tm.cells[i][j], q = Tm.pos(i, j);
          if (i === 3) { var z = f(t, T.zeros[j], 0.3); E.op(cell, z); E.scaleAt(cell, q[0], q[1], 1 + 0.6 * (1 - z)); continue; }
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
        Tm.hl('r', 0, 0, 2, 2, E.INK2, both);
        Tm.hl('p', 0, 3, 2, 3, E.VIOLET, bothP).setAttribute('transform', 'translate(' + shift.toFixed(1) + ' 0)');   // 框跟着还没贴过来的那一列
        E.op(four, f(t, T.zeros[0], 0.3));
        E.op(rows[0].h, f(t, T.ptAns, 0.4)); E.op(rows[1].h, f(t, T.dirAns, 0.4));
        E.op(rows[0].m, f(t, T.row1, 0.4)); E.op(rows[1].m, f(t, T.row0, 0.4));
      }
    };
  };
})();
