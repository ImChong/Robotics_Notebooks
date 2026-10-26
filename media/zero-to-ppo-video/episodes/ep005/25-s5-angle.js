// 005 第 5 幕：中间的朝向——平均转过的角度。左边接着上一幕的地面网格（x 前、y 左），右栏放做法、矩阵、G1 与三维示意。
// 句 1 网格上 0° 的两根小箭头和 90° 的虚线小箭头；右栏「别平均矩阵」打叉、「平均转过的角度」打勾，「零度到九十度」单位圆上从 0° 到 90° 的弧亮起，
//    「四十五」两根小箭头沿圆转到 45°（长度不变），右栏写出「0° 到 90°：中间是 45°」；
// 句 2 右栏出转 45° 的旋转矩阵，四个数按念到的顺序写出，网格上两个箭头尖标坐标；「长度」「垂直」「航列式」三块小牌依次打勾，网格上同时写「1」、标直角、涂小方块；
// 句 3「每一个中间朝向」0°…90° 每隔 15° 一根淡绿小箭头依次亮起，尖都落在圆上；「机器人」右栏换成 G1，从 0° 一路转到 90°（每一刻都是真正的转动），「不变形」打勾；
// 句 4「三维」网格换成斜着看的三维小坐标架，「转动的轴」画出一根斜着的转轴，坐标架绕它来回转；右栏顶上弯曲球面示意图（蓝弧沿着面走），
//    「四元数」「矩阵指数」两块小牌，念到「第六集」「第十集」各补上集数。
// 时刻取自配音的逐词时间（分镜时间 = 念到这个词的时刻），见 ../ep005.py。
(function () {
  var E = window.EP005, K = E.K;
  var D = Math.PI / 180;
  var OX = 200, OY = 214, U = 82;
  var MX = 622, R45 = [0.71, -0.71, 0.71, 0.71];
  var T = {
    no: 3.0, yes: 4.09, arc: 5.6, mid: 7.33, turn: [7.33, 1.0], txt: [5.6, 6.13, 6.85],
    mat: 9.98, cells: [11.42, 13.49, 12.32, 14.42], chk: [15.31, 16.29, 17.26],
    fan: 20.25, along: 21.73, robot: 22.68, spin: [23.0, 2.2], ok: 25.07,
    to3: 26.66, axis: 27.48, tilt: 28.35, slerp: 30.08, quat: 32.27, mexp: 33.3, ep6: 35.35, ep10: 37.13
  };
  var AXIS = [1, -0.5, 1];                                  // 和三根小箭头都隔 48° 以上，画面里也够长

  function bump(st, a, d) { var u = K.seg(st, a, a + d); return u > 0 && u < 1 ? Math.sin(Math.PI * u) : 0; }

  E.scenes[4] = {
    title: '中间的朝向：平均转过的角度',
    dur: 38.7,
    cues: [
      { at: 0, s: '别平均矩阵，**平均角度**：0° 到 90°，中间是 45°' },
      { at: 9.02, s: '转 45° 的旋转矩阵：长度 1、互相垂直、行列式 1' },
      { at: 19.36, s: '每个中间朝向都沿着圆走：转身的每一刻，身子都不变形' },
      { at: 26.56, s: '三维：沿着转动取中间值，用四元数（第 6 集）或矩阵指数（第 10 集）' }
    ],
    build: function (K) {
      var s = K.sceneSvg('中间的朝向要平均转过的角度，不要平均矩阵：零度到九十度，中间是四十五度；转 45 度的旋转矩阵两列是 (0.71, 0.71) 和 (−0.71, 0.71)，长度 1、互相垂直、行列式 1；每个中间朝向都沿着圆走，机器人转身每一刻都不变形；三维里转轴不一定竖直，沿着转动取中间值要用四元数（第 6 集）或矩阵指数（第 10 集）');

      /* ══ 二维 ══ */
      var flat = E.el('g', {}, s);
      var pl = E.plane(flat, { ox: OX, oy: OY, u: U, x0: -2, x1: 2, y0: -2, y1: 2, labels: false });
      var P = pl.P;
      E.halo(E.txt(flat, pl.box[2] + 22, OY + 8, 'x 前', 22, E.INK2, 'start', 700));
      E.halo(E.txt(flat, OX + 12, pl.box[1] - 4, 'y 左', 22, E.INK2, 'start', 700));
      E.op(pl.warp.g, 0.6);
      var unit = K.paint(E.el('circle', { cx: OX, cy: OY, r: U, fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '6 6' }, flat), null, E.HI);
      unit.style.strokeOpacity = 0.6;
      var arc = K.paint(E.el('path', { fill: 'none', 'stroke-width': 5, 'stroke-linecap': 'round' }, flat), null, E.ACC);
      var sq = K.paint(E.el('path', { 'stroke-width': 2.5 }, flat), E.tint(E.HI, 26, 'var(--demo-surface)'), E.HI);
      var fan = [];
      for (var a = 0; a <= 90; a += 15) fan.push({ a: a * D, v: E.vec(flat, E.GOOD, 3, { head: 11 }), dot: K.paint(E.el('circle', { r: 5 }, flat), E.ACC) });
      var ghost = [E.vec(flat, E.GOOD, 3.5, { head: 13, dash: '6 5' }), E.vec(flat, E.WARN, 3.5, { head: 13, dash: '6 5' })];
      var e1 = E.vec(flat, E.GOOD, 5, { head: 16 }), e2 = E.vec(flat, E.WARN, 5, { head: 16 });
      var corner = K.paint(E.el('path', { fill: 'none', 'stroke-width': 2.5 }, flat), null, E.INK);
      var lab45 = E.halo(E.txt(flat, 0, 0, '45°', 22, E.ACC, 'middle', 900));
      var tip1 = E.halo(E.txt(flat, 0, 0, '(0.71, 0.71)', 20, E.GOOD, 'start', 900)), tip2 = E.halo(E.txt(flat, 0, 0, '(−0.71, 0.71)', 20, E.WARN, 'end', 900));

      /* 右栏 ①：做法 + 矩阵 + 三块小牌 */
      var r1 = E.el('g', {}, flat);
      var noG = E.el('g', {}, r1), yesG = E.el('g', {}, r1);
      var noX = E.cross(noG, 482, 40, 10, E.BAD, 4.5); E.txt(noG, 506, 48, '别平均矩阵', 24, E.BAD, 'start', 900);
      var yesC = E.check(yesG, 470, 76, 24, E.GOOD, 4.5); E.txt(yesG, 506, 96, '平均转过的角度', 24, E.GOOD, 'start', 900);
      var ang = E.txt(r1, 476, 150, '', 26, E.INK, 'start', 900);
      var m = E.mat(r1, MX, 248, R45, { size: 28, dx: 84, dy: 42, pad: 46, cw: 76, colors: [E.GOOD, E.WARN, E.GOOD, E.WARN] });
      var chips = ['长度 1', '互相垂直', '行列式 1'].map(function (t, i) {
        var g = E.el('g', {}, r1), x = 456 + 114 * i;
        var tg = E.tag(g, x, 318, 106, 40, t, E.GOOD, { size: 20, ink: E.INK, weight: 900, rx: 10 });
        return { g: g, tg: tg, cx: x + 53 };
      });
      var ones = [E.GOOD, E.WARN].map(function (c) { return E.halo(E.txt(flat, 0, 0, '1', 22, c, 'middle', 900)); });

      /* 右栏 ②：G1 转身 */
      var rob = E.el('g', {}, flat);
      var g1 = E.g1(rob, MX, 374, 238, { az: 135, el: 20 });
      var fr = E.frame3(rob, 4.5);
      var okG = E.el('g', {}, rob);
      E.check(okG, MX - 92, 396, 22, E.GOOD, 4);
      E.halo(E.txt(okG, MX + 14, 414, '每一刻都不变形', 22, E.GOOD, 'middle', 900));

      /* ══ 三维 ══ */
      var solid = E.el('g', {}, s);
      var proj = E.view3(128, 24);
      var gz = E.gizmo(solid, 210, 250, 112, proj);
      var axisL = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '9 7' }, solid), null, E.HI);
      var axisLab = E.halo(E.txt(solid, 0, 0, '转动的轴', 22, E.HI, 'middle', 900));
      var sph = E.el('g', {}, solid);
      K.paint(E.el('circle', { cx: MX, cy: 124, r: 78, 'stroke-width': 2.5 }, sph), E.tint(E.VIOLET, 16, 'var(--demo-surface)'), E.VIOLET);
      K.paint(E.el('ellipse', { cx: MX, cy: 160, rx: 68, ry: 17, fill: 'none', 'stroke-width': 1.8, 'stroke-dasharray': '6 6' }, sph), null, E.VIOLET);
      var sArc = K.paint(E.el('path', { d: 'M 574 106 Q 622 46 670 106', fill: 'none', 'stroke-width': 4 }, sph), null, E.ACC);
      [574, 670].forEach(function (x) { K.paint(E.el('circle', { cx: x, cy: 106, r: 6 }, sph), E.INK); });
      E.txt(sph, MX, 230, '沿着转动走', 22, E.ACC, 'middle', 900);
      var tools = [{ s: '四元数', ep: '第 6 集', c: E.ACC }, { s: '矩阵指数', ep: '第 10 集', c: E.VIOLET }].map(function (d, i) {
        var g = E.el('g', {}, solid), y = 280 + 58 * i;
        var tg = E.tag(g, 470, y, 300, 46, '', d.c, { size: 24, ink: E.INK, weight: 900, rx: 12 });
        return { g: g, tg: tg, d: d, y: y };
      });

      function sqPath(a, b) {
        var p0 = P(0, 0), p1 = P(a[0], a[1]), p2 = P(a[0] + b[0], a[1] + b[1]), p3 = P(b[0], b[1]);
        return 'M ' + [p0, p1, p2, p3].map(function (p) { return p[0].toFixed(1) + ' ' + p[1].toFixed(1); }).join(' L ') + ' Z';
      }

      return {
        el: s,
        draw: function (st, t) {
          var f = E.fade, o = P(0, 0);
          var out2 = 1 - f(st, T.to3, 0.5);
          E.op(flat, out2);

          /* 两根小箭头：0° → 45°（沿着圆转，长度不变） */
          var th = 45 * D * f(st, T.turn[0], T.turn[1]);
          var R = E.rot(th), a2 = [R[0], R[2]], b2 = [R[1], R[3]], q1 = P(a2[0], a2[1]), q2 = P(b2[0], b2[1]);
          pl.warp.set(R);
          e1.set(o[0], o[1], q1[0], q1[1]); e2.set(o[0], o[1], q2[0], q2[1]);
          var g90 = E.rot(Math.PI / 2), gp1 = P(g90[0], g90[2]), gp2 = P(g90[1], g90[3]);
          ghost[0].set(o[0], o[1], gp1[0], gp1[1]); ghost[1].set(o[0], o[1], gp2[0], gp2[1]);
          ghost.forEach(function (g) { E.op(g.g, 0.75 * (1 - f(st, T.fan, 0.4))); });
          // 0° → 90° 的弧（单位圆上）
          var ap = f(st, T.arc, 0.8), e = P(Math.cos(Math.PI / 2 * ap), Math.sin(Math.PI / 2 * ap));
          arc.setAttribute('d', ap > 0.01 ? 'M ' + (OX + U) + ' ' + OY + ' A ' + U + ' ' + U + ' 0 0 0 ' + e[0].toFixed(1) + ' ' + e[1].toFixed(1) : '');
          E.op(arc, 0.85);
          var l45 = P(1.32 * Math.cos(Math.PI / 4), 1.32 * Math.sin(Math.PI / 4));
          lab45.setAttribute('x', l45[0] + 6); lab45.setAttribute('y', l45[1] + 8);
          E.pop(lab45, l45[0] + 6, l45[1], f(st, T.mid, 0.3) * (1 - f(st, T.cells[0] - 0.2, 0.3)), 0.6);
          // 句 2：坐标、「1」、直角、小方块
          tip1.setAttribute('x', q1[0] + 12); tip1.setAttribute('y', q1[1] - 10);
          tip2.setAttribute('x', q2[0] - 12); tip2.setAttribute('y', q2[1] - 10);
          E.op(tip1, f(st, T.cells[0], 0.3) * (1 - f(st, T.fan, 0.4))); E.op(tip2, f(st, T.cells[1], 0.3) * (1 - f(st, T.fan, 0.4)));
          [a2, b2].forEach(function (v, i) {
            var mid = P(v[0] * 0.55, v[1] * 0.55), n = i ? [-v[1], v[0]] : [v[1], -v[0]];
            ones[i].setAttribute('x', (mid[0] + 18 * n[0]).toFixed(1)); ones[i].setAttribute('y', (mid[1] - 18 * n[1] + 8).toFixed(1));
            E.op(ones[i], f(st, T.chk[0], 0.3) * (1 - f(st, T.fan, 0.4)));
          });
          var c1 = P(a2[0] * 0.2, a2[1] * 0.2), c2 = P(a2[0] * 0.2 + b2[0] * 0.2, a2[1] * 0.2 + b2[1] * 0.2), c3 = P(b2[0] * 0.2, b2[1] * 0.2);
          corner.setAttribute('d', 'M ' + c1[0].toFixed(1) + ' ' + c1[1].toFixed(1) + ' L ' + c2[0].toFixed(1) + ' ' + c2[1].toFixed(1) + ' L ' + c3[0].toFixed(1) + ' ' + c3[1].toFixed(1));
          E.op(corner, f(st, T.chk[1], 0.3) * (1 - f(st, T.fan, 0.4)));
          sq.setAttribute('d', sqPath(a2, b2));
          E.op(sq, f(st, T.chk[2], 0.4) * (1 - f(st, T.fan, 0.4)));
          // 句 3：0°…90° 每 15° 一根，尖落在圆上
          fan.forEach(function (fv, i) {
            var p = P(Math.cos(fv.a), Math.sin(fv.a));
            fv.v.set(o[0], o[1], p[0], p[1]); fv.dot.setAttribute('cx', p[0].toFixed(1)); fv.dot.setAttribute('cy', p[1].toFixed(1));
            var on = f(st, T.fan + 0.22 * i, 0.3);
            E.op(fv.v.g, 0.55 * on); E.op(fv.dot, on);
          });
          var ab = bump(st, T.along, 1.0);
          unit.style.strokeOpacity = 0.6 + 0.4 * ab; E.glow(unit, E.HI, ab > 0.05);
          E.op(e1.g, 1 - 0.6 * f(st, T.fan, 0.4)); E.op(e2.g, 1 - 0.6 * f(st, T.fan, 0.4));

          /* 右栏 ① */
          E.op(r1, 1 - f(st, T.robot - 0.3, 0.4));
          E.op(noG, f(st, T.no, 0.3)); E.drawOn(noX, f(st, T.no + 0.2, 0.35));
          E.op(yesG, f(st, T.yes, 0.3)); E.drawOn(yesC, f(st, T.yes + 0.6, 0.35));
          ang.textContent = st < T.txt[1] ? '0°' : st < T.txt[2] ? '0° 到 90°' : '0° 到 90°：中间是 45°';
          E.op(ang, f(st, T.txt[0], 0.3));
          m.cells.forEach(function (c, k) { E.op(c, f(st, T.cells[k], 0.25)); });
          m.brackets.forEach(function (b) { E.op(b, f(st, T.mat, 0.3)); });
          chips.forEach(function (c, i) {
            E.pop(c.g, c.cx, 338, f(st, T.chk[i], 0.3), 0.8);
          });

          /* 右栏 ②：G1 从 0° 转到 90°，不变形 */
          E.op(rob, f(st, T.robot, 0.4));
          var yaw = 90 * D * f(st, T.spin[0], T.spin[1]);
          g1.set(yaw, 1); fr.set(g1.frame(0.5));
          E.op(okG, f(st, T.ok, 0.3));

          /* ══ 三维 ══ */
          E.op(solid, f(st, T.to3 + 0.2, 0.5));
          var sw = 0.5 - 0.5 * Math.cos(2 * Math.PI * Math.max(0, st - T.slerp) / 4);   // 「沿着转动」起绕转轴来回转
          gz.set(E.rot3(AXIS, (15 + 30 * sw) * D));   // 15°…45°：这一段里三根小箭头在画面上都和转轴隔 36° 以上，不叠在转轴上
          var AL = Math.hypot(AXIS[0], AXIS[1], AXIS[2]);
          var A0 = gz.P(AXIS.map(function (x) { return -1.25 * x / AL; })), A1 = gz.P(AXIS.map(function (x) { return 2 * x / AL; }));   // 转轴伸到小箭头够不着的地方再写字
          axisL.setAttribute('d', 'M ' + A0[0].toFixed(1) + ' ' + A0[1].toFixed(1) + ' L ' + A1[0].toFixed(1) + ' ' + A1[1].toFixed(1));
          E.drawOn(axisL, f(st, T.axis, 0.6));
          axisLab.setAttribute('x', A1[0] + 4); axisLab.setAttribute('y', A1[1] - 12);
          E.op(axisLab, f(st, T.tilt, 0.3));
          E.op(sph, f(st, T.slerp, 0.4)); E.drawOn(sArc, f(st, T.slerp + 0.3, 0.7));
          tools.forEach(function (tl, i) {
            tl.tg.text.textContent = tl.d.s + (st >= (i ? T.ep10 : T.ep6) ? ' → ' + tl.d.ep : '');
            E.pop(tl.g, 620, tl.y + 23, f(st, i ? T.mexp : T.quat, 0.3), 0.85);
          });
        }
      };
    }
  };
})();
