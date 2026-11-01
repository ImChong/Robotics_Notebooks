// 008 第 3 幕：约束——不能越过的线。左半边接着上一幕：小跑道 + 代价图（紫色的碗，亮黄小球停在碗底 (2, 8)）；右栏上面是代价的式子。
// 句 1「最多」小跑道和代价图在 1 格处立起橙色的墙，墙右边是橙色斜线区，右栏出约束牌「约束：u ≤ 1」，「约束」牌子鼓一下；
// 句 2「超过一格」斜线区加深、碗在墙外的那段变灰，「碗底」小球亮一下，「出界」小球上打橙 ×、写「出界」；
// 句 3「能选的范围」横轴 0–1 加粗成绿色、写「能选（可行集）」，「代价最小」小球顺着碗壁滚到 (1, 10)，「十分」写「10 分」，「贴」墙和小球一起亮一下；
// 句 4「迈三格」墙挪到 3 格，约束牌改成 u ≤ 3，「碗底」小球滚回 (2, 8) 打勾，「不起作用」右栏写「约束不起作用」，「两格」小球亮一下；
// 句 5「至少迈四格」4 格处再立一道墙，左边全盖上斜线区（4 ≤ u ≤ 3），「没有」横轴上能选的那段消失、小球淡下去，「无解」碗下写「无解」、
//    右栏写「可行集是空的」「无解（不可行）」（Boyd & Vandenberghe 4.1.1 节：没有可行点叫 infeasible）；
// 句 6 左半边和右栏收起，三块牌按念到的词弹出，每块一台侧视 G1（URDF）：「关节」大腿抬起，小腿在膝关节的限位 −5°…165°
//    （URDF 的 lower / upper）之间来回转、碰到橙色挡块就停；「电机」G1 一下一下往下蹲，膝关节力矩的竖条跟着涨、到橙线「上限」
//    （URDF 的 effort 139 N·m）就顶住；「脚」G1 站着，「推」绿箭头从地面往上顶脚底（✓），「拉」橙虚线箭头往下拽（×）。
// 时刻取自配音的逐词时间（分镜时间 = 念到这个词的时刻），见 ../ep008.py。
(function () {
  var E = window.EP008, K = E.K, H = E.HEX;
  var RX = 470;
  var T = {
    wall: 1.72, card: 2.3, word: 5.48,
    over: 7.34, gray: 8.6, bottom: 9.59, out: 11.07,
    feas: 13.01, roll1: [14.3, 0.9], ten: 15.78, edge: 17.92,
    move: [21.3, 0.8], roll2: [22.3, 0.8], ok: 23.4, noUse: 24.84, two: 26.24,
    infe: 28.56, add: 30.25, lo: 31.5, none: 33.15, word: 34.65,
    leave: 36.25, cards: [38.39, 40.66, 42.12], push: 42.7, pull: 43.67
  };

  E.scenes[2] = {
    title: '约束：不能越过的线',
    dur: 45.05,
    cues: [
      { at: 0, s: '一步最多迈 1 格：必须守住的限制叫**约束**，$u \\le 1$' },
      { at: 6.84, s: '超过 1 格都不能选：**碗底**（2 格）**出界**了' },
      { at: 12.7, s: '能选的里面，1 格最好（10 分）：贴在约束的边上' },
      { at: 20.21, s: '要是最多迈 3 格：碗底在范围里，约束不起作用' },
      { at: 27.75, s: '约束打架：最多 3 格又至少 4 格 → 没有能选的，**无解**（不可行）' },
      { at: 36.15, s: '机器人的约束：关节有范围、电机的力有上限、脚只能推不能拉' }
    ],
    build: function (K) {
      var s = K.sceneSvg('约束：一步最多只能迈一格，超过一格的迈法都不能选，碗底两格出界了；能选的范围里代价最小的是一格，十分，最好的解贴在约束的边上；要是最多能迈三格，碗底在范围里，约束不起作用，答案还是两格；机器人里的约束：关节只能在一定范围里转，电机的力有上限，脚只能推地面不能拉地面');

      /* ══ 左：小跑道 + 代价图 + 墙 ══ */
      var left = E.el('g', {}, s);
      var tr = E.track(left, E.MINI);
      var ch = E.chart(left, { ox: E.CH.ox, oy: E.CH.oy, su: E.CH.su, sc: E.CH.sc, xLab: '迈几格 u', yLab: '扣分' });
      var P = ch.P;
      var bowlFill = E.el('path', { d: ch.path(E.cost, 0, 4) + ' L ' + P(4, 16.6)[0] + ' ' + P(0, 16.6)[1] + ' L ' + P(0, 16.6)[0] + ' ' + P(0, 16.6)[1] + ' Z' }, left);
      K.paint(bowlFill, E.tint(E.VIOLET, 22, 'transparent'));
      E.op(bowlFill, 0.9);
      var c3 = ch.curve(E.cost, E.VIOLET, 5);
      var gray = K.paint(E.el('path', { fill: 'none', 'stroke-width': 6, 'stroke-linecap': 'round' }, left), null, E.MUT);
      var feas = K.paint(E.el('line', { y1: E.CH.oy, y2: E.CH.oy, 'stroke-width': 8, 'stroke-linecap': 'round' }, left), null, E.GOOD);
      var feasLab = E.halo(E.txt(left, 0, E.CH.oy - 14, '能选（可行集）', 20, E.GOOD, 'start', 900));
      var wallC = E.wall(left, s, P(0, 17.4)[1], E.CH.oy, '');
      var wallT = E.wall(left, s, E.MINI.y - 50, E.MINI.y + 14, '');
      // 句 5：再加一条「至少 4 格」——4 格左边全是斜线区，墙在 4 格
      var loC = E.wall(left, s, P(0, 17.4)[1], E.CH.oy, ''), loT = E.wall(left, s, E.MINI.y - 50, E.MINI.y + 14, '');
      var loLab = E.halo(E.txt(left, 0, P(0, 12.4)[1], '至少 4 格', 20, E.BAD, 'end', 900));
      var noneLab = E.halo(E.txt(left, P(2, 0)[0], P(0, 4)[1], '无解', 30, E.BAD, 'middle', 900));
      var wallLab = E.halo(E.txt(left, 0, P(0, 16.4)[1], '', 20, E.BAD, 'start', 900));   // 写在墙右边的斜线区顶上，不压竖轴的「扣分」
      s.appendChild(tr.foot); s.appendChild(tr.step);
      var ring = K.paint(E.el('circle', { r: 17, fill: 'none', 'stroke-width': 3.5 }, left), null, E.HI);
      var ball = K.paint(E.el('circle', { r: 11, 'stroke-width': 3 }, left), E.HI, 'var(--demo-surface)');
      var outX = E.cross(left, 0, 0, 11, E.BAD, 5);
      var outLab = E.halo(E.txt(left, P(2, 0)[0], P(0, 8)[1] + 46, '出界', 22, E.BAD, 'middle', 900));
      var ballLab = E.halo(E.txt(left, 0, 0, '', 22, E.HI, 'middle', 900));
      var okC = E.check(left, 0, 0, 20, E.HI, 4.5);

      /* ══ 右栏：式子、约束牌、两行结论 ══ */
      var right = E.el('g', {}, s);
      E.txt(right, RX + 6, 40, '代价', 26, E.VIOLET, 'start', 900);
      E.tex(right, RX + 62, 38, '=' + E.ctex(H.acc, '(4-u)^2') + '+' + E.ctex(H.warn, 'u^2'), 26, E.INK, 'start', { w: 240, h: 46 });
      var card = E.el('g', {}, right);
      var cardR = K.paint(E.el('rect', { x: RX, y: 70, width: 320, height: 62, rx: 14, 'stroke-width': 3 }, card), E.tint(E.BAD, 12, 'var(--demo-surface)'), E.BAD);
      var cardTxt = E.txt(card, RX + 16, 110, '约束：一步最多迈', 22, E.INK, 'start', 900);
      var cardTex = E.tex(card, RX + 304, 109, '', 30, E.BAD, 'end', { w: 160, h: 50 });
      var line1 = E.txt(right, RX + 6, 196, '', 24, E.INK, 'start', 900);
      var line2 = E.txt(right, RX + 6, 246, '', 24, E.INK, 'start', 900);
      var line3 = E.txt(right, RX + 6, 296, '', 24, E.INK, 'start', 900);

      /* ══ 句 5：三块牌，每块一台侧视 G1（URDF），数取自 unitree_ros 的 g1_29dof_rev_1_0.urdf ══ */
      var GY = 292, SC = 1.05;                              // 牌里的地面线、G1 缩放（第三块的地面高一些，脚底下要画箭头）
      var cards = [0, 1, 2].map(function (i) {
        var g = E.el('g', {}, s), x = 28 + 252 * i, cx = x + 118, gy = i === 2 ? 254 : GY, sc = i === 2 ? 0.92 : SC;
        K.paint(E.el('rect', { x: x, y: 40, width: 236, height: 344, rx: 18, 'stroke-width': 2.5 }, g), E.tint(E.BAD, 8, 'var(--demo-surface)'), E.BAD);
        K.paint(E.el('line', { x1: x + 22, y1: gy, x2: x + 214, y2: gy, 'stroke-width': 3, 'stroke-linecap': 'round' }, g), null, E.BORDER);
        var r = window.Robot3D.g1Side(g, 0);
        var rx = cx - (i === 1 ? 36 : 22);
        r.place({ x: rx, scale: sc, dy: gy - 388 });
        // G1 本地坐标 → 牌上的坐标（同 g1Side 的 place：以脚底 y = 388 为心缩放）
        function sv(p) { return [r.state.x + sc * p[0], r.state.dy + 388 + sc * (p[1] - 388)]; }
        return { g: g, x: x, cx: cx, r: r, sv: sv, gy: gy };
      });
      function caption(c, title, l1, l2) {
        E.txt(c.g, c.cx, 82, title, 26, E.INK, 'middle', 900);
        E.txt(c.g, c.cx, 336, l1, 22, E.INK2, 'middle', 700);
        E.txt(c.g, c.cx, 366, l2, 22, E.INK2, 'middle', 700);
      }
      // ① 膝关节：大腿往前抬 55°，小腿在 URDF 的限位 −5°（lower −0.087 rad）到 165°（upper 2.8798 rad）之间来回转，碰到就停
      var KN = { a: 55, lo: -5, hi: 165 };
      var jg = cards[0].g;
      var kArc = K.paint(E.el('path', { fill: 'none', 'stroke-width': 5, 'stroke-linecap': 'round' }, jg), null, E.GOOD);
      kArc.style.strokeOpacity = 0.85;
      var kStops = [0, 1].map(function () { return K.paint(E.el('line', { 'stroke-width': 6, 'stroke-linecap': 'round' }, jg), null, E.BAD); });
      var kLab = E.halo(E.txt(jg, 0, 0, '', 20, E.INK, 'middle', 900));
      caption(cards[0], '关节', '膝关节只能在', '−5° 到 165° 里转');
      // ② 电机：一下一下往下蹲，膝关节要的力矩跟着涨，到 URDF 的上限（effort 139 N·m）就顶住
      var mg = cards[1].g, M = { x: cards[1].x + 196, y0: 280, y1: 112, cap: 150 };
      K.paint(E.el('rect', { x: M.x - 16, y: M.y1, width: 32, height: M.y0 - M.y1, rx: 8, 'stroke-width': 2.5 }, mg), E.S2, E.BORDER);
      var bar = K.paint(E.el('rect', { x: M.x - 11, width: 22, rx: 5 }, mg), E.tint(E.WARN, 70, 'var(--demo-surface)'));
      K.paint(E.el('line', { x1: M.x - 26, y1: M.cap, x2: M.x + 26, y2: M.cap, 'stroke-width': 4, 'stroke-linecap': 'round' }, mg), null, E.BAD);
      E.halo(E.txt(mg, M.x, M.cap - 12, '上限', 20, E.BAD, 'middle', 900));
      var A1X = cards[1].r.pts().ankle[0];                  // 站直时踝的本地 x：往下蹲时踝不挪
      var link2 = K.paint(E.el('line', { 'stroke-width': 2, 'stroke-dasharray': '4 5' }, mg), null, E.WARN);
      caption(cards[1], '电机', '膝关节的力矩', '上限 139 N·m');
      // ③ 脚：站着；地面往上推脚底（绿）可以，往下拉（橙虚线）不行
      var fg = cards[2].g, FX = cards[2].sv([14, 388])[0], FY = cards[2].gy;
      for (var k = 0; k < 7; k++) K.paint(E.el('line', { x1: cards[2].x + 34 + 26 * k, y1: FY + 2, x2: cards[2].x + 22 + 26 * k, y2: FY + 14, 'stroke-width': 2 }, fg), null, E.BORDER);
      var push = E.el('g', {}, fg);
      E.arrow(push, [[FX - 8, FY + 56], [FX - 8, FY + 3]], E.GOOD, 5, { head: 15 });
      E.halo(E.txt(push, FX - 24, FY + 48, '推', 22, E.GOOD, 'end', 900));
      E.check(push, FX - 70, FY + 30, 18, E.GOOD, 4);
      var pull = E.el('g', {}, fg);
      E.arrow(pull, [[FX + 40, FY + 3], [FX + 40, FY + 56]], E.BAD, 5, { head: 15, dash: '8 7' });
      E.halo(E.txt(pull, FX + 56, FY + 48, '拉', 22, E.BAD, 'start', 900));
      E.cross(pull, FX + 94, FY + 40, 9, E.BAD, 4);
      caption(cards[2], '脚', '只能推地面', '不能拉地面');

      return {
        el: s,
        draw: function (st, t) {
          var f = E.fade;
          var keep = 1 - f(st, T.leave, 0.5);
          E.op(left, keep); E.op(right, keep);
          var dimB = 1 - 0.75 * f(st, T.none, 0.4);              // 无解时小球、脚印淡下去
          E.op(tr.foot, keep * dimB); E.op(tr.step, keep * dimB);

          /* 约束的上限：1 → 3（句 4） */
          var lim = 1 + 2 * f(st, T.move[0], T.move[1]);
          var wOn = f(st, T.wall, 0.4), zOn = 0.55 + 0.45 * f(st, T.over, 0.4);
          wallC.set(P(lim, 0)[0], P(4.3, 0)[0]); wallT.set(tr.X(lim), tr.X(4) + 30);
          E.op(wallC.g, wOn); E.op(wallT.g, wOn);
          wallC.zone.style.opacity = wallT.zone.style.opacity = zOn;
          wallLab.textContent = '最多 ' + (st < T.move[0] + T.move[1] / 2 ? 1 : 3) + ' 格';
          wallLab.setAttribute('x', (P(lim, 0)[0] + 10).toFixed(1));
          E.op(wallLab, wOn);
          var eb = E.bump(st, T.edge, 1.0);
          wallC.line.style.strokeWidth = 4 + 3 * eb; E.glow(wallC.line, E.BAD, eb > 0.05);

          /* 碗在墙外的那段变灰；能选的横轴段 */
          gray.setAttribute('d', ch.path(E.cost, lim, 4, 40));
          E.op(gray, 0.85 * f(st, T.gray, 0.4));
          feas.setAttribute('x1', P(0, 0)[0]); feas.setAttribute('x2', P(lim, 0)[0].toFixed(1));
          var noneP = f(st, T.none, 0.4);                         // 句 5：没有能选的了
          E.op(feas, f(st, T.feas, 0.4) * (1 - noneP));
          feasLab.setAttribute('x', (P(0, 0)[0] + 8).toFixed(1));
          E.op(feasLab, f(st, T.feas, 0.4) * (1 - noneP));
          // 「至少 4 格」的墙：线在 4 格，斜线区盖住 0–4 格
          loC.set(P(0, 0)[0], P(4, 0)[0]); loT.set(tr.X(0) - 6, tr.X(4));
          [loC, loT].forEach(function (w) { var x = w === loC ? P(4, 0)[0] : tr.X(4); w.line.setAttribute('x1', x.toFixed(1)); w.line.setAttribute('x2', x.toFixed(1)); });
          var loOn = f(st, T.lo, 0.4);
          E.op(loC.g, loOn); E.op(loT.g, loOn);
          loLab.setAttribute('x', (P(4, 0)[0] - 10).toFixed(1)); E.op(loLab, loOn);
          E.pop(noneLab, P(2, 0)[0], P(0, 4)[1] - 10, f(st, T.word, 0.35), 0.7);

          /* 小球：碗底 → 1 格（贴着墙）→ 回到碗底 */
          var ub = E.steps(st, [[0, 2], [T.roll1[0], 1], [T.roll2[0], 2]], 0.85);
          var bp = P(ub, E.cost(ub));
          ball.setAttribute('cx', bp[0].toFixed(1)); ball.setAttribute('cy', bp[1].toFixed(1));
          var bb = E.bump(st, T.bottom, 0.8) + E.bump(st, T.edge, 1.0) + E.bump(st, T.two, 0.8);
          ball.setAttribute('r', (11 + 4 * Math.min(1, bb)).toFixed(2)); E.glow(ball, E.HI, bb > 0.05);
          E.op(ball, dimB);
          ring.setAttribute('cx', bp[0].toFixed(1)); ring.setAttribute('cy', bp[1].toFixed(1));
          E.op(ring, E.bump(st, T.edge, 1.0));
          tr.setU(ub);
          var ox = 1 - f(st, T.roll1[0], 0.3);
          outX.setAttribute('transform', 'translate(' + P(2, 8)[0] + ' ' + P(2, 8)[1] + ')');
          E.drawOn(outX, f(st, T.out, 0.3) * ox);
          E.op(outLab, f(st, T.out, 0.3) * ox);
          ballLab.textContent = st < T.move[0] ? '10 分' : '8 分';
          ballLab.setAttribute('x', (bp[0] + (st < T.move[0] ? -18 : 0)).toFixed(1)); ballLab.setAttribute('y', (bp[1] + (st < T.move[0] ? 36 : 44)).toFixed(1));
          ballLab.setAttribute('text-anchor', st < T.move[0] ? 'end' : 'middle');
          E.op(ballLab, f(st, T.ten, 0.3) * (1 - f(st, T.move[0], 0.3)) + f(st, T.two, 0.3) * (1 - f(st, T.infe, 0.3)));
          okC.setAttribute('transform', 'translate(' + (P(2, 8)[0] + 22) + ' ' + (P(2, 8)[1] - 34) + ')');
          E.drawOn(okC, f(st, T.ok, 0.4) * (1 - f(st, T.infe, 0.3)));

          /* 右栏 */
          E.pop(card, RX + 160, 101, f(st, T.card, 0.35), 0.85);
          cardTex.setTex(st >= T.lo ? '4\\le u\\le 3' : 'u\\le ' + (st < T.move[0] + T.move[1] / 2 ? 1 : 3));
          cardTxt.textContent = st >= T.lo ? '约束：同时要' : '约束：一步最多迈';
          var wb = E.bump(st, T.word, 0.8);
          cardR.style.strokeWidth = 3 + 2 * wb; E.glow(cardR, E.BAD, wb > 0.05);
          var A = st < T.move[0], Z = st >= T.infe;              // A：最多 1 格；Z：句 5 约束打架
          var zOut = 1 - f(st, T.infe - 0.3, 0.3);
          line1.textContent = Z ? '至少 4 格、最多 3 格' : A ? '碗底 u = 2：出界' : '碗底 u = 2：在范围里';
          line1.style.fill = Z ? E.INK : A ? E.BAD : E.GOOD;
          E.op(line1, Z ? f(st, T.lo, 0.3) : A ? f(st, T.out, 0.3) * (1 - f(st, T.move[0] - 0.3, 0.3)) : f(st, T.ok, 0.3) * zOut);
          line2.textContent = Z ? '可行集是空的' : A ? '可行集：0 ≤ u ≤ 1' : '约束不起作用';
          line2.style.fill = Z ? E.BAD : A ? E.GOOD : E.MUT;
          E.op(line2, Z ? f(st, T.none, 0.3) : A ? f(st, T.feas, 0.3) * (1 - f(st, T.move[0] - 0.3, 0.3)) : f(st, T.noUse, 0.3) * zOut);
          line3.textContent = Z ? '无解（不可行）' : A ? '最好：u = 1，代价 10' : '答案：还是 u = 2';
          line3.style.fill = Z ? E.BAD : E.HI;
          E.op(line3, Z ? f(st, T.word, 0.3) : A ? f(st, T.ten, 0.3) * (1 - f(st, T.move[0] - 0.3, 0.3)) : f(st, T.two, 0.3) * zOut);

          /* 句 5：三块牌 */
          cards.forEach(function (c, i) { E.pop(c.g, c.cx, 212, f(st, T.cards[i], 0.35), 0.85); });
          // ① 膝关节：想转到的角度一来一回，超出 −5°…165° 的部分被限位挡住
          var kWant = 80 + 105 * Math.sin(2 * Math.PI * (t - T.cards[0]) / 3.2), kk = Math.max(KN.lo, Math.min(KN.hi, kWant));
          var r0 = cards[0].r;
          r0.legs([KN.a, KN.a - kk], [-4, 0]);
          var P0 = r0.pts(), kn = cards[0].sv(P0.knee), an = cards[0].sv(P0.ankle);
          // 小腿实际方向和名义角（0 = 竖直朝下、往前为正）差一个常数：URDF 的小腿不是正竖着的
          var phiReal = Math.atan2(an[0] - kn[0], an[1] - kn[1]) * 180 / Math.PI, off = phiReal - (KN.a - kk);
          function dirAt(phi) { var q = (phi + off) * Math.PI / 180; return [Math.sin(q), Math.cos(q)]; }
          var R = 30, pts = [];
          for (var j = 0; j <= 40; j++) { var d = dirAt(KN.a - KN.lo - (KN.hi - KN.lo) * j / 40); pts.push([kn[0] + R * d[0], kn[1] + R * d[1]]); }
          kArc.setAttribute('d', K.polyPath(pts));
          [KN.lo, KN.hi].forEach(function (lim, i) {
            var d = dirAt(KN.a - lim);
            kStops[i].setAttribute('x1', (kn[0] + (R - 9) * d[0]).toFixed(1)); kStops[i].setAttribute('y1', (kn[1] + (R - 9) * d[1]).toFixed(1));
            kStops[i].setAttribute('x2', (kn[0] + (R + 11) * d[0]).toFixed(1)); kStops[i].setAttribute('y2', (kn[1] + (R + 11) * d[1]).toFixed(1));
            kStops[i].style.strokeWidth = Math.abs(kk - lim) < 0.01 ? 9 : 6;
          });
          kLab.textContent = String(Math.round(kk)).replace('-', '−') + '°';
          kLab.setAttribute('x', (kn[0] - 66).toFixed(1)); kLab.setAttribute('y', (kn[1] - 30).toFixed(1));
          kLab.style.fill = kk === KN.lo || kk === KN.hi ? E.BAD : E.INK;
          // ② 往下蹲：蹲得越深，膝关节要的力矩越大；超过上限的部分顶住（示意，不按真实质量算）
          var r1 = cards[1].r, dep = 24 * (0.5 - 0.5 * Math.cos(2 * Math.PI * (t - T.cards[1]) / 3.0));
          var lg = r1.legIK([0, 302], [A1X, r1.ankleY - dep]);
          r1.legs(lg, lg);
          r1.place({ dy: GY - 388 + SC * dep });
          r1.kneeDot.style.fill = E.WARN;
          var want = 0.3 + 1.0 * dep / 24, capH = M.y0 - M.cap, top = M.y0 - capH * want;
          var y = Math.max(M.cap, top);
          bar.setAttribute('y', y.toFixed(1)); bar.setAttribute('height', (M.y0 - 3 - y).toFixed(1));
          K.paint(bar, top <= M.cap ? E.tint(E.BAD, 70, 'var(--demo-surface)') : E.tint(E.WARN, 70, 'var(--demo-surface)'));
          var kp = cards[1].sv(r1.pts().knee);
          link2.setAttribute('x1', (kp[0] + 8).toFixed(1)); link2.setAttribute('y1', kp[1].toFixed(1));
          link2.setAttribute('x2', (M.x - 18).toFixed(1)); link2.setAttribute('y2', ((M.y0 + y) / 2).toFixed(1));
          E.op(push, f(st, T.push, 0.3)); E.op(pull, f(st, T.pull, 0.3));
        }
      };
    }
  };
})();
