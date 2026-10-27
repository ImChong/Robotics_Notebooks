// 006 第 2 幕：欧拉角——三个角，按顺序转。左边 G1（身子绕骨盆转，不画地面），右边三行「偏航 / 俯仰 / 横滚」，底下放例子和「顺序不能换」。
// 句 1「欧拉角」右边出标题「按顺序转三次」，「三次」三行的编号 ①②③ 依次出来，「每次绕一根轴」「转一个角度」各鼓一下；
// 句 2「偏航」第一行写出名字和「绕竖着的轴」，「竖着的轴」骨盆上画一根蓝色竖虚线、绕它一圈，「原地转身」G1 转身 45°，行尾的数从 0° 数到 45°；
// 句 3「俯仰」第二行，「左右方向的轴」画绿虚线（身子转完以后的左右，y）和一圈，「往前倾」G1 前倾 25°；
// 句 4「横滚」第三行，「前后方向的轴」画红虚线（身子此刻的前后，x）和一圈，「侧歪」G1 侧歪 20°；
// 句 5「写成三个数」底下出 (45°, 25°, 20°)，念到哪个数哪个数和对应的行一起亮；「人一看就懂」G1 回到原样，再跟着「转身」「前倾」「侧歪」按顺序重做一遍；
// 句 6「直观」「只用三个数」底下两块绿牌依次打勾；
// 句 7「顺序不能换」底下换成两台小 G1：左边先转身 90° 再前倾 45°，右边先前倾 45° 再转身 90°，「第三集」右上出小牌，
//    「转动接着做」左边那台按顺序做，「换了顺序」右边那台反着做，「不一样」中间打「≠」，「写清楚转的顺序」底下写出约定。
// 时刻取自配音的逐词时间（分镜时间 = 念到这个词的时刻），见 ../ep006.py。
(function () {
  var E = window.EP006, K = E.K;
  var D = Math.PI / 180;
  var T = {
    head: 0.1, nums: 3.28, axisB: 4.41, angB: 5.23,
    rows: [7.82, 12.21, 17.05], axes: [8.53, 12.88, 17.82], spins: [9.66, 14.65, 19.64], spinD: 1.0,
    triple: 21.41, flash: [22.75, 23.39, 24.04], reset: [24.79, 0.8], redo: [25.98, 27.33, 28.65], redoD: 0.8,
    good: [30.43, 31.37], best: 33.29,
    order: 36.01, ep3: 37.56, runA: 38.67, runB: 39.79, runD: 0.45, neq: 40.74, conv: 43.87, axesOut: 21.2
  };
  var ANG = [45 * D, 25 * D, 20 * D];
  var GX = 170, GY = 362, GH = 290;
  var RX = 372, ROW = [{ y: 52 }, { y: 120 }, { y: 188 }];
  var COLS = [E.AX[2], E.AX[1], E.AX[0]];   // 偏航绕 z（蓝）、俯仰绕 y（绿）、横滚绕 x（红）

  function bump(st, a, d) { var u = K.seg(st, a, a + d); return u > 0 && u < 1 ? Math.sin(Math.PI * u) : 0; }

  E.scenes[1] = {
    title: '欧拉角：三个角，按顺序转',
    dur: 46.0,
    cues: [
      { at: 0, s: '**欧拉角**：拆成按顺序的三次转动，每次绕一根轴、转一个角度' },
      { at: 6.96, s: '① 偏航：绕竖着的轴，原地转身' },
      { at: 11.3, s: '② 俯仰：绕身子左右方向的轴，往前倾' },
      { at: 16.15, s: '③ 横滚：绕身子前后方向的轴，往一边侧歪' },
      { at: 21.31, s: '(45°, 25°, 20°)：转身 45°，前倾 25°，侧歪 20°' },
      { at: 30.33, s: '好处：**直观**，又只用 3 个数' },
      { at: 35.13, s: '顺序不能换（第 3 集）：一定要写清楚转的顺序' }
    ],
    build: function (K) {
      var s = K.sceneSvg('欧拉角把一个朝向拆成按顺序的三次转动：先偏航，绕竖轴原地转身 45 度；再俯仰，绕身子左右方向的轴前倾 25 度；最后横滚，绕身子前后方向的轴侧歪 20 度。写成三个数 (45°, 25°, 20°)，直观，只用三个数。顺序不能换：先转身 90 度再前倾 45 度，和先前倾再转身，结果不一样，所以要写清楚转的顺序');

      /* ══ 左：G1 + 三根转轴（虚线）和绕轴的一圈 ══ */
      var rob = E.el('g', {}, s);
      var back = E.el('g', {}, rob);
      var g1 = E.g1(rob, GX, GY, GH, { az: 100, el: 18, w: 230, up: 20 });
      var axG = E.el('g', {}, rob), front = E.el('g', {}, rob);
      var fr = E.frame3(rob, 5);
      var axLines = COLS.map(function (c) { return K.paint(E.el('path', { fill: 'none', 'stroke-width': 3.5, 'stroke-dasharray': '10 7', 'stroke-linecap': 'round' }, axG), null, c); });
      var rings = COLS.map(function (c) { return E.ring3(back, front, c, 4); });
      var P = function (v) { return g1.project(v); };
      P.toward = E.view3(100, 18).toward;

      /* ══ 右：标题 + 三行 ══ */
      var head = E.txt(s, RX, 30, '按顺序转三次：', 24, E.INK, 'start', 900);
      var rows = [
        { name: '偏航', how: '绕竖着的轴：转身' },
        { name: '俯仰', how: '绕身子左右的轴：前倾' },
        { name: '横滚', how: '绕身子前后的轴：侧歪' }
      ].map(function (d, i) {
        var y = ROW[i].y, g = E.el('g', {}, s);
        var box = K.paint(E.el('rect', { x: RX, y: y, width: 418, height: 58, rx: 12, 'stroke-width': 2.5 }, g), E.tint(COLS[i], 8, 'var(--demo-surface)'), COLS[i]);
        var badge = E.el('g', {}, s);
        K.paint(E.el('circle', { cx: RX + 26, cy: y + 29, r: 16 }, badge), COLS[i]);
        E.txt(badge, RX + 26, y + 36, String(i + 1), 20, 'var(--bg)', 'middle', 900);
        var nm = E.txt(g, RX + 54, y + 38, d.name, 26, COLS[i], 'start', 900);
        var how = E.txt(g, RX + 116, y + 37, d.how, 20, E.INK2, 'start', 700);
        var val = E.txt(g, RX + 404, y + 39, '0°', 28, E.INK, 'end', 900);
        return { g: g, box: box, badge: badge, nm: nm, how: how, val: val, y: y };
      });

      /* ══ 右下：例子、好处 ══ */
      var tri = E.el('g', {}, s);
      var TXS = [462, 546, 630];
      [[420, '('], [500, ','], [584, ','], [670, ')']].forEach(function (d) { E.txt(tri, d[0], 300, d[1], 34, E.INK, 'middle', 700); });
      var triN = ['45°', '25°', '20°'].map(function (v, i) { return E.txt(tri, TXS[i], 300, v, 34, COLS[i], 'middle', 900); });
      var easy = E.txt(tri, 556, 344, '人一看就懂', 22, E.INK2, 'middle', 700);
      var goods = ['直观', '只用 3 个数'].map(function (t, i) {
        var g = E.el('g', {}, s), x = 400 + 196 * i;
        var tg = E.tag(g, x, 368, 180, 42, '', E.GOOD, { size: 22, ink: E.GOOD, weight: 900, rx: 12 });
        tg.text.textContent = t;
        tg.text.setAttribute('x', x + 104);
        var ck = E.check(g, x + 16, 380, 20, E.GOOD, 4);
        return { g: g, tg: tg, ck: ck, cx: x + 90 };
      });

      /* ══ 右下 ②：顺序不能换——两台小 G1 ══ */
      var ord = E.el('g', {}, s);
      var mini = [{ x: 470, lab: ['先转身 90°', '再前倾 45°'] }, { x: 690, lab: ['先前倾 45°', '再转身 90°'] }].map(function (d) {
        var g = E.el('g', {}, ord);
        var r = E.g1(g, d.x, 392, 200, { az: 135, el: 20, w: 200, up: 10 });   // 斜着看：往左倒、往前倒都看得出来
        var f3 = E.frame3(g, 4);
        var l1 = E.txt(g, d.x, 80, d.lab[0], 21, E.INK, 'middle', 900), l2 = E.txt(g, d.x, 106, d.lab[1], 21, E.INK, 'middle', 900);
        return { g: g, r: r, f3: f3, l1: l1, l2: l2, x: d.x };
      });
      var neq = E.txt(ord, 580, 300, '≠', 52, E.BAD, 'middle', 900);
      var ep3 = E.tag(ord, 478, 8, 204, 36, '第 3 集：顺序不能换', E.VIOLET, { size: 19, ink: E.INK, weight: 900, rx: 10 });
      var conv = E.halo(E.txt(ord, 580, 34, '写清楚顺序：先偏航，再俯仰，最后横滚', 21, E.HI, 'middle', 900));

      return {
        el: s,
        draw: function (st) {
          var f = E.fade;

          /* G1：转身 → 前倾 → 侧歪；「人一看就懂」回到原样再按顺序做一遍 */
          var redo = st >= T.reset[0];
          var back0 = redo ? 1 - f(st, T.reset[0], T.reset[1]) : 1;   // 回原样
          var a = [0, 1, 2].map(function (i) {
            var p = redo ? (st < T.redo[0] ? back0 : f(st, T.redo[i], T.redoD)) : f(st, T.spins[i], T.spinD);
            return ANG[i] * p;
          });
          var M = E.zyx(a[0], a[1], a[2]);
          g1.setR(M);
          fr.set(g1.frame(0.42));

          /* 三根转轴：偏航绕世界竖轴；俯仰绕转身以后的左右；横滚绕再前倾以后的前后 */
          var F0 = E.Rz(a[0]), F1 = E.mul3(F0, E.Ry(a[1]));
          var axs = [[0, 0, 1], E.col3(F0, 1), E.col3(F1, 0)];
          var planes = [[[1, 0, 0], [0, 1, 0]], [E.col3(F0, 0), [0, 0, 1]], [E.col3(F1, 1), E.col3(F1, 2)]];
          var axOut = 1 - f(st, T.axesOut, 0.4);
          axs.forEach(function (v, i) {
            var p = P([-0.62 * v[0], -0.62 * v[1], -0.62 * v[2]]), q = P([0.62 * v[0], 0.62 * v[1], 0.62 * v[2]]);
            axLines[i].setAttribute('d', 'M ' + p[0].toFixed(1) + ' ' + p[1].toFixed(1) + ' L ' + q[0].toFixed(1) + ' ' + q[1].toFixed(1));
            var on = f(st, T.axes[i], 0.4) * axOut, act = i < 2 ? 1 - f(st, T.axes[i + 1], 0.4) * 0.65 : 1;   // 后一根出来时，前一根淡一些
            E.op(axLines[i], on * act);
            rings[i].set([0, 0, 0], planes[i][0], planes[i][1], 0.26, P, P);
            E.op(rings[i].front, on * act); E.op(rings[i].back, on * act);
          });

          /* 右：标题、三行 */
          E.op(head, f(st, T.head, 0.4) * (1 - f(st, T.order, 0.4)));
          rows.forEach(function (r, i) {
            var bb = bump(st, T.axisB, 0.6) + bump(st, T.angB, 0.6);
            E.scaleAt(r.badge, RX + 26, r.y + 29, f(st, T.nums + 0.18 * i, 0.3) * (1 + 0.25 * bb));
            E.op(r.g, f(st, T.rows[i], 0.35) * (1 - f(st, T.order, 0.4)));
          E.op(r.badge, f(st, T.nums + 0.18 * i, 0.3) * (1 - f(st, T.order, 0.4)));
            r.val.textContent = Math.round(a[i] / D) + '°';
            var fl = bump(st, T.flash[i], 0.8);
            r.box.style.strokeWidth = 2.5 + 3 * fl;
            E.glow(r.box, COLS[i], fl > 0.05);
          });

          /* 右下：(45°, 25°, 20°)、人一看就懂、两块绿牌；「顺序不能换」起换成两台小 G1 */
          var ex = f(st, T.triple, 0.35) * (1 - f(st, T.order, 0.4));
          E.op(tri, ex);
          triN.forEach(function (n, i) { E.scaleAt(n, TXS[i], 290, 1 + 0.14 * bump(st, T.flash[i], 0.8)); });   // 鼓一下不碰到两边的逗号
          E.op(easy, f(st, T.reset[0], 0.3));
          goods.forEach(function (g, i) {
            E.pop(g.g, g.cx, 389, f(st, T.good[i], 0.3) * (1 - f(st, T.order, 0.4)), 0.85);
            E.drawOn(g.ck, f(st, T.good[i] + 0.2, 0.35));
            g.tg.rect.style.strokeWidth = 2 + 2.5 * bump(st, T.best, 0.9);
          });

          // 两台小 G1：左边 Rz(90°)·Ry(45°)（先转身、再绕身子的左右前倾），右边 Ry(45°)·Rz(90°)（先前倾、再绕身子的竖轴转身）
          E.op(ord, f(st, T.order, 0.4));
          var A1 = f(st, T.runA, T.runD), A2 = f(st, T.runA + T.runD + 0.1, T.runD);
          var B1 = f(st, T.runB, T.runD), B2 = f(st, T.runB + T.runD + 0.1, T.runD);
          mini[0].r.setR(E.mul3(E.Rz(90 * D * A1), E.Ry(45 * D * A2)));
          mini[1].r.setR(E.mul3(E.Ry(45 * D * B1), E.Rz(90 * D * B2)));
          mini.forEach(function (m, i) {
            m.f3.set(m.r.frame(0.36));
            var a1 = i ? B1 : A1, a2 = i ? B2 : A2, s0 = i ? T.runB : T.runA;
            m.l1.style.fill = st >= s0 && a1 < 1 ? 'var(--hi)' : '';
            m.l2.style.fill = a1 >= 1 && a2 > 0 && a2 < 1 ? 'var(--hi)' : '';
          });
          E.pop(neq, 580, 284, f(st, T.neq, 0.3), 0.5);
          E.pop(ep3.g, 580, 26, f(st, T.ep3, 0.3) * (1 - f(st, T.conv - 0.3, 0.3)), 0.85);
          E.op(conv, f(st, T.conv, 0.35));
        }
      };
    }
  };
})();
