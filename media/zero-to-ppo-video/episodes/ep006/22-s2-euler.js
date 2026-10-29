// 006 第 2 幕：欧拉角——三个角，按顺序转。先按机器人里常用的 RPY：先横滚（绕 x）、再俯仰（绕 y）、最后偏航（绕 z），绕的都是不动的轴（URDF 的 rpy 就这么定义），
// 这样转严格叫「固定角」（X-Y-Z 固定角）；再反过来绕跟着身子转动的轴，先偏航、再俯仰、最后横滚，摆出来一模一样，这才叫「欧拉角」（Z-Y-X 欧拉角）——
// 两种转法是同一个矩阵 Rz·Ry·Rx（Lynch & Park《Modern Robotics》附录 B.2）。三个数按 (横滚, 俯仰, 偏航) 写。左边 G1（身子绕骨盆转，不画地面），右边三行「横滚 / 俯仰 / 偏航」，底下放例子和「顺序不能换」。
// 句 1「欧拉角」右边出标题「按顺序转三次」，「三次」三行的编号 ①②③ 依次出来，「每次绕一根轴」「转一个角度」各鼓一下；
// 句 2「横滚」第一行写出名字和「绕前后的轴」，「前后方向的轴」骨盆上画一根红色虚线（x）、绕它一圈，「侧歪」G1 侧歪 20°，行尾的数从 0° 数到 20°；
// 句 3「俯仰」第二行，「左右方向的轴」画绿虚线（y）和一圈，「往前倾」G1 前倾 25°；
// 句 4「偏航」第三行，「竖着的轴」画蓝虚线（z）和一圈，「原地转身」G1 转身 45°；
// 句 5「三根轴都是不动的」三根虚线一起亮一下，「固定角」顶上标题换成「固定角：绕不动的轴，① → ② → ③」，「机器人里最常用」底下写 RPY = Roll 横滚 · Pitch 俯仰 · Yaw 偏航，念到哪个哪个亮；
// 句 6「换一种转法」G1 回到原样，「跟着身子」骨盆上三根小箭头鼓一下，「反过来」标题换成「欧拉角：绕身子的轴，③ → ② → ①」、底下出固定角 / 欧拉角两行对照，
//    「偏航」「俯仰」「横滚」G1 依次绕身子自己的 z、y、x 转（那根小箭头变粗，沿它画虚线和一圈），「一模一样」打勾；
// 句 7「写成三个数」底下出 (20°, 25°, 45°)，念到哪个数哪个数和对应的行一起亮；「人一看就懂」G1 回到原样，再跟着「侧歪」「前倾」「转身」按顺序重做一遍；
// 句 8「直观」「只用三个数」底下两块绿牌依次打勾；
// 句 9「顺序不能换」底下换成两台小 G1：左边先绕竖轴转 90° 再绕左右轴转 45°，右边反过来（都绕不动的轴），「第三集」右上出小牌，
//    「转动接着做」左边那台按顺序做，「换了顺序」右边那台做，「不一样」中间打「≠」，「固定角和欧拉角各有十二种」顶上写出「各 12 种：写清楚是哪一种」。
// 时刻取自配音的逐词时间（分镜时间 = 念到这个词的时刻），见 ../ep006.py。
(function () {
  var E = window.EP006, K = E.K;
  var D = Math.PI / 180;
  var T = {
    head: 0.1, nums: 3.28, axisB: 4.41, angB: 5.23,
    rows: [7.78, 12.64, 17.27], axes: [8.53, 13.3, 18.0], spins: [10.02, 14.8, 19.14], spinD: 1.0,
    fixed: 20.86, fixName: 24.36, rpy: [25.47, 28.4, 29.13, 29.75], axesOut: 30.9,
    eu0: 31.23, body: 32.96, rev: 34.47, eu: [37.04, 36.06, 35.28], euD: 0.6, same: 38.23, euName: 39.74,   // eu[i]：绕身子的第 i 根轴（x 横滚、y 俯仰、z 偏航）开始转
    triple: 41.21, flash: [42.53, 43.16, 43.85], reset: [44.65, 0.8], redo: [45.85, 46.95, 48.22], redoD: 0.8,
    good: [50.33, 51.27], best: 53.19,
    order: 55.91, ep3: 57.46, runA: 58.57, runB: 59.69, runD: 0.45, neq: 60.64, conv: 63.58
  };
  var ANG = [20 * D, 25 * D, 45 * D];                         // (横滚, 俯仰, 偏航)
  var GX = 170, GY = 362, GH = 290;
  var RX = 372, ROW = [{ y: 52 }, { y: 120 }, { y: 188 }];
  var COLS = [E.X, E.Y, E.Z];                                  // 横滚绕 x（红）、俯仰绕 y（绿）、偏航绕 z（蓝）
  var AXES = [[1, 0, 0], [0, 1, 0], [0, 0, 1]], PLANES = [[[0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0]]];
  var PL = [[1, 2], [0, 2], [0, 1]];   // 绕第 i 根轴转的那一圈，在另外两根轴张成的平面里

  function bump(st, a, d) { var u = K.seg(st, a, a + d); return u > 0 && u < 1 ? Math.sin(Math.PI * u) : 0; }

  E.scenes[1] = {
    title: '欧拉角：三个角，按顺序转',
    dur: 69.55,
    cues: [
      { at: 0, s: '**欧拉角**：拆成按顺序的三次转动，每次绕一根轴、转一个角度' },
      { at: 6.96, s: '① 横滚：绕前后方向的轴（x），往一边侧歪' },
      { at: 11.74, s: '② 俯仰：绕左右方向的轴（y），往前倾' },
      { at: 16.3, s: '③ 偏航：绕竖着的轴（z），原地转身' },
      { at: 20.76, s: '三根轴都不动：严格叫**固定角**；机器人里最常用这个顺序 RPY（横滚、俯仰、偏航）' },
      { at: 31.13, s: '绕跟着身子转动的轴，反过来先偏航、再俯仰、最后横滚：摆出来一样，这才叫**欧拉角**' },
      { at: 41.11, s: '(20°, 25°, 45°)：侧歪 20°，前倾 25°，转身 45°' },
      { at: 50.23, s: '好处：**直观**，又只用 3 个数' },
      { at: 55.03, s: '顺序不能换；固定角、欧拉角各 12 种，要写清楚是哪一种' }
    ],
    build: function (K) {
      var s = K.sceneSvg('欧拉角把一个朝向拆成按顺序的三次转动，机器人里最常用 RPY：先横滚，绕前后方向的轴侧歪 20 度；再俯仰，绕左右方向的轴前倾 25 度；最后偏航，绕竖着的轴原地转身 45 度，三根轴都不动，严格叫固定角。反过来绕跟着身子转动的轴，先偏航、再俯仰、最后横滚，摆出来一模一样，这才叫欧拉角。写成三个数 (20°, 25°, 45°)，直观，只用三个数。顺序不能换：先绕竖轴转 90 度再绕左右轴转 45 度，和反过来，结果不一样；固定角和欧拉角各有 12 种，要写清楚是哪一种');

      /* ══ 左：G1 + 三根不动的转轴（虚线）和绕轴的一圈 ══ */
      var rob = E.el('g', {}, s);
      var back = E.el('g', {}, rob);
      var g1 = E.g1(rob, GX, GY, GH, { az: 100, el: 18, w: 230, up: 20 });
      var axG = E.el('g', {}, rob), front = E.el('g', {}, rob);
      var fr = E.frame3(rob, 5);
      var axLines = COLS.map(function (c) { return K.paint(E.el('path', { fill: 'none', 'stroke-width': 3.5, 'stroke-dasharray': '10 7', 'stroke-linecap': 'round' }, axG), null, c); });
      var rings = COLS.map(function (c) { return E.ring3(back, front, c, 4); });
      // 欧拉角那一遍：沿身子自己的轴画虚线和一圈（轴跟着身子，每帧重算）
      var euLines = COLS.map(function (c) { return K.paint(E.el('path', { fill: 'none', 'stroke-width': 3.5, 'stroke-dasharray': '10 7', 'stroke-linecap': 'round' }, axG), null, c); });
      var euRings = COLS.map(function (c) { return E.ring3(back, front, c, 4); });
      var P = function (v) { return g1.project(v); };
      P.toward = E.view3(100, 18).toward;
      AXES.forEach(function (v, i) {
        var p = P([-0.62 * v[0], -0.62 * v[1], -0.62 * v[2]]), q = P([0.62 * v[0], 0.62 * v[1], 0.62 * v[2]]);
        axLines[i].setAttribute('d', 'M ' + p[0].toFixed(1) + ' ' + p[1].toFixed(1) + ' L ' + q[0].toFixed(1) + ' ' + q[1].toFixed(1));
        rings[i].set([0, 0, 0], PLANES[i][0], PLANES[i][1], 0.26, P, P);
      });

      /* ══ 右：标题 + 三行 ══ */
      var head = E.txt(s, RX, 30, '按顺序转三次：', 24, E.INK, 'start', 900);
      var rows = [
        { name: '横滚', how: '绕前后的轴 x：侧歪' },
        { name: '俯仰', how: '绕左右的轴 y：前倾' },
        { name: '偏航', how: '绕竖着的轴 z：转身' }
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

      /* ══ 右下：RPY 的名字（每个英文首字母用那根轴的颜色） ══ */
      var rpy = E.el('g', {}, s);
      E.txt(rpy, RX, 284, '机器人里最常用的顺序：', 21, E.INK, 'start', 900);
      var rpyParts = [['R', 'oll 横滚'], ['P', 'itch 俯仰'], ['Y', 'aw 偏航']].map(function (d, i) {
        var x = RX + 6 + 140 * i;
        var t = E.txt(rpy, x, 330, '', 22, E.INK2, 'start', 700);
        var b = document.createElementNS('http://www.w3.org/2000/svg', 'tspan'); b.textContent = d[0]; b.style.fill = COLS[i]; b.style.fontWeight = 900; b.style.fontSize = '28px';
        var r = document.createElementNS('http://www.w3.org/2000/svg', 'tspan'); r.textContent = d[1];
        t.appendChild(b); t.appendChild(r);
        return { t: t, x: x };
      });

      /* ══ 右下：固定角 / 欧拉角对照（句 6）：轴的名字用那根轴的颜色 ══ */
      var cmp = E.el('g', {}, s);
      function seqLine(y, lead, names) {
        var t = E.txt(cmp, RX, y, '', 20, E.INK2, 'start', 700);
        function sp(str, col, w) { var n = document.createElementNS('http://www.w3.org/2000/svg', 'tspan'); n.textContent = str; if (col) n.style.fill = col; if (w) n.style.fontWeight = w; t.appendChild(n); }
        sp(lead[0], E.INK, 900); sp(lead[1]);
        names.forEach(function (k, j) { if (j) sp(' → '); sp(['横滚', '俯仰', '偏航'][k], COLS[k], 900); });
        return t;
      }
      var cmpA = seqLine(282, ['固定角', '（轴不动）：'], [0, 1, 2]);
      var cmpB = seqLine(322, ['欧拉角', '（轴跟着身子）：'], [2, 1, 0]);
      var cmpOk = E.el('g', {}, cmp);
      E.check(cmpOk, RX + 2, 352, 20, E.GOOD, 4);
      E.txt(cmpOk, RX + 36, 370, '摆出来一模一样', 21, E.GOOD, 'start', 900);

      /* ══ 右下：例子、好处 ══ */
      var tri = E.el('g', {}, s);
      var TXS = [462, 546, 630];
      [[420, '('], [500, ','], [584, ','], [670, ')']].forEach(function (d) { E.txt(tri, d[0], 300, d[1], 34, E.INK, 'middle', 700); });
      var triN = ['20°', '25°', '45°'].map(function (v, i) { return E.txt(tri, TXS[i], 300, v, 34, COLS[i], 'middle', 900); });
      var easy = E.txt(tri, 556, 344, '人一看就懂', 22, E.INK2, 'middle', 700);
      var goods = ['直观', '只用 3 个数'].map(function (t, i) {
        var g = E.el('g', {}, s), x = 400 + 196 * i;
        var tg = E.tag(g, x, 368, 180, 42, '', E.GOOD, { size: 22, ink: E.GOOD, weight: 900, rx: 12 });
        tg.text.textContent = t;
        tg.text.setAttribute('x', x + 104);
        var ck = E.check(g, x + 16, 380, 20, E.GOOD, 4);
        return { g: g, tg: tg, ck: ck, cx: x + 90 };
      });

      /* ══ 右下 ②：顺序不能换——两台小 G1（都绕不动的轴转） ══ */
      var ord = E.el('g', {}, s);
      var mini = [{ x: 470, lab: ['先绕竖轴转 90°', '再绕左右轴转 45°'] }, { x: 690, lab: ['先绕左右轴转 45°', '再绕竖轴转 90°'] }].map(function (d) {
        var g = E.el('g', {}, ord);
        var r = E.g1(g, d.x, 392, 200, { az: 135, el: 20, w: 200, up: 10 });   // 斜着看：往左倒、往前倒都看得出来
        var f3 = E.frame3(g, 4);
        var l1 = E.txt(g, d.x, 80, d.lab[0], 20, E.INK, 'middle', 900), l2 = E.txt(g, d.x, 106, d.lab[1], 20, E.INK, 'middle', 900);
        return { g: g, r: r, f3: f3, l1: l1, l2: l2, x: d.x };
      });
      var neq = E.txt(ord, 580, 300, '≠', 52, E.ERR, 'middle', 900);
      var ep3 = E.tag(ord, 478, 8, 204, 36, '第 3 集：顺序不能换', E.VIOLET, { size: 19, ink: E.INK, weight: 900, rx: 10 });
      var conv = E.halo(E.txt(ord, 580, 34, '固定角、欧拉角各 12 种：写清楚是哪一种', 20, E.HI, 'middle', 900));

      return {
        el: s,
        draw: function (st) {
          var f = E.fade;

          /* G1：侧歪 → 前倾 → 转身（绕不动的 x、y、z）；「人一看就懂」回到原样再按顺序做一遍 */
          // 句 6「换一种转法」：回到原样，再按欧拉角的顺序做一遍——式子还是 Rz·Ry·Rx，只是偏航先走完、横滚最后走：
          // 先绕 z 转，再绕转过之后身子的 y，最后绕身子的 x（绕身子的轴转是右乘）
          var redo = st >= T.reset[0], euler = st >= T.eu0 && !redo;
          var back0 = redo ? 1 - f(st, T.reset[0], T.reset[1]) : euler ? 1 - f(st, T.eu0, 0.8) : 1;   // 回原样
          var a = [0, 1, 2].map(function (i) {
            var p = redo ? (st < T.redo[0] ? back0 : f(st, T.redo[i], T.redoD))
              : euler ? (st < T.eu[2] ? back0 : f(st, T.eu[i], T.euD)) : f(st, T.spins[i], T.spinD);
            return ANG[i] * p;
          });
          g1.setR(E.zyx(a[2], a[1], a[0]));   // 先 Rx(横滚)，再左乘 Ry(俯仰)，最后左乘 Rz(偏航)：Rz·Ry·Rx
          fr.set(g1.frame(0.42));
          // 欧拉角那一遍：正在转的那根身子的轴——小箭头变粗，沿它画虚线和一圈；「跟着身子」三根小箭头一起鼓一下
          var bodyB = bump(st, T.body, 1.0);
          var before = [E.mul3(E.Rz(ANG[2]), E.Ry(ANG[1])), E.Rz(ANG[2]), [[1, 0, 0], [0, 1, 0], [0, 0, 1]]];   // 绕身子 x / y / z 转之前身子的朝向
          var nextAt = [T.same + 0.6, T.eu[0], T.eu[1]];
          [0, 1, 2].forEach(function (i) {
            var on = euler ? f(st, T.eu[i] - 0.3, 0.3) * (1 - f(st, nextAt[i], 0.3)) : 0;
            fr.arrows[i].line.style.strokeWidth = 5 + 3 * Math.max(on, bodyB);
            E.op(euLines[i], on); euRings[i].op(on);
            if (on <= 0) return;
            var ax = E.col3(before[i], i), pa = E.col3(before[i], PL[i][0]), pb = E.col3(before[i], PL[i][1]);
            var p = P(ax.map(function (x) { return -0.62 * x; })), q = P(ax.map(function (x) { return 0.62 * x; }));
            euLines[i].setAttribute('d', 'M ' + p[0].toFixed(1) + ' ' + p[1].toFixed(1) + ' L ' + q[0].toFixed(1) + ' ' + q[1].toFixed(1));
            euRings[i].set([0, 0, 0], pa, pb, 0.26, P, P);
          });

          /* 三根转轴：世界的 x、y、z，不跟着身子动；「三根轴都是不动的」一起亮一下 */
          var axOut = 1 - f(st, T.axesOut, 0.4), fb = bump(st, T.fixed, 1.6);
          AXES.forEach(function (v, i) {
            var on = f(st, T.axes[i], 0.4) * axOut;
            var act = i < 2 ? 1 - f(st, T.axes[i + 1], 0.4) * 0.65 * (1 - f(st, T.fixed, 0.3)) : 1;   // 后一根出来时，前一根淡一些；「不动」时三根一起亮
            E.op(axLines[i], on * act);
            axLines[i].style.strokeWidth = 3.5 + 2.5 * fb;
            rings[i].op(on * act);
          });

          /* 右：标题、三行 */
          // 标题：「固定角」「反过来」「写成三个数」各换一次字，换字时先淡下去再亮回来
          head.textContent = st < T.fixName ? '按顺序转三次：' : st < T.rev ? '固定角：绕不动的轴，① → ② → ③' : st < T.triple ? '欧拉角：绕身子的轴，③ → ② → ①' : '按顺序转三次：';
          var dip = [T.fixName, T.rev, T.triple].reduce(function (m, a) { return Math.max(m, bump(st, a - 0.2, 0.4)); }, 0);
          E.op(head, f(st, T.head, 0.4) * (1 - f(st, T.order, 0.4)) * (1 - 0.9 * dip));
          rows.forEach(function (r, i) {
            var bb = bump(st, T.axisB, 0.6) + bump(st, T.angB, 0.6);
            E.scaleAt(r.badge, RX + 26, r.y + 29, f(st, T.nums + 0.18 * i, 0.3) * (1 + 0.25 * bb));
            E.op(r.g, f(st, T.rows[i], 0.35) * (1 - f(st, T.order, 0.4)));
            E.op(r.badge, f(st, T.nums + 0.18 * i, 0.3) * (1 - f(st, T.order, 0.4)));
            r.val.textContent = Math.round(a[i] / D) + '°';
            var fl = bump(st, T.flash[i], 0.8) + bump(st, T.rpy[i + 1], 0.7) + bump(st, T.eu[i] - 0.1, 0.8);
            r.box.style.strokeWidth = 2.5 + 3 * Math.min(1, fl);
            E.glow(r.box, COLS[i], fl > 0.05);
          });

          /* 右下：RPY（「机器人里最常用」到「写成三个数」） */
          E.op(rpy, f(st, T.rpy[0], 0.35) * (1 - f(st, T.eu0, 0.3)));
          var cmpOut = 1 - f(st, T.triple - 0.2, 0.3);
          E.op(cmpA, f(st, T.rev, 0.3) * cmpOut); E.op(cmpB, f(st, T.eu[2], 0.3) * cmpOut);
          E.op(cmpOk, f(st, T.same, 0.3) * cmpOut);
          E.scaleAt(cmpB, RX, 316, 1 + 0.06 * bump(st, T.euName, 0.8));   // 「欧拉角」念到时这一行鼓一下
          rpyParts.forEach(function (p, i) {
            p.t.style.fill = f(st, T.rpy[i + 1], 0.3) > 0.5 ? E.INK : '';
            E.scaleAt(p.t, p.x + 40, 322, 1 + 0.12 * bump(st, T.rpy[i + 1], 0.7));
          });

          /* 右下：(20°, 25°, 45°)、人一看就懂、两块绿牌；「顺序不能换」起换成两台小 G1 */
          var ex = f(st, T.triple, 0.35) * (1 - f(st, T.order, 0.4));
          E.op(tri, ex);
          triN.forEach(function (n, i) { E.scaleAt(n, TXS[i], 290, 1 + 0.14 * bump(st, T.flash[i], 0.8)); });   // 鼓一下不碰到两边的逗号
          E.op(easy, f(st, T.reset[0], 0.3));
          goods.forEach(function (g, i) {
            E.pop(g.g, g.cx, 389, f(st, T.good[i], 0.3) * (1 - f(st, T.order, 0.4)), 0.85);
            E.drawOn(g.ck, f(st, T.good[i] + 0.2, 0.35));
            g.tg.rect.style.strokeWidth = 2 + 2.5 * bump(st, T.best, 0.9);
          });

          // 两台小 G1，都绕不动的轴转（后做的左乘）：左边先 Rz(90°) 再 Ry(45°) → Ry·Rz；右边先 Ry(45°) 再 Rz(90°) → Rz·Ry
          E.op(ord, f(st, T.order, 0.4));
          var A1 = f(st, T.runA, T.runD), A2 = f(st, T.runA + T.runD + 0.1, T.runD);
          var B1 = f(st, T.runB, T.runD), B2 = f(st, T.runB + T.runD + 0.1, T.runD);
          mini[0].r.setR(E.mul3(E.Ry(45 * D * A2), E.Rz(90 * D * A1)));
          mini[1].r.setR(E.mul3(E.Rz(90 * D * B2), E.Ry(45 * D * B1)));
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
