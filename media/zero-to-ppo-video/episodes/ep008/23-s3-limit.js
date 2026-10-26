// 008 第 3 幕：约束——不能越过的线。左半边接着上一幕：小跑道 + 代价图（紫色的碗，亮黄小球停在碗底 (2, 8)）；右栏上面是代价的式子。
// 句 1「最多」小跑道和代价图在 1 格处立起红墙，墙右边是红色斜线区，右栏出约束牌「约束：u ≤ 1」，「约束」牌子鼓一下；
// 句 2「超过一格」斜线区加深、碗在墙外的那段变灰，「碗底」小球亮一下，「出界」小球上打红 ×、写「出界」；
// 句 3「能选的范围」横轴 0–1 加粗成绿色、写「能选」，「代价最小」小球顺着碗壁滚到 (1, 10)，「十分」写「10 分」，「贴」红墙和小球一起亮一下；
// 句 4「迈三格」红墙挪到 3 格，约束牌改成 u ≤ 3，「碗底」小球滚回 (2, 8) 打勾，「不起作用」右栏写「约束不起作用」，「两格」小球亮一下；
// 句 5 左半边和右栏收起，三块牌按念到的词弹出：「关节」一根连杆在两道红色挡块之间来回摆、碰到就停；
//    「电机」力的竖条往上涨、到红线「上限」就顶住；「脚」脚掌踩在地面上，「推」绿箭头从地面往上顶（✓），「拉」红虚线箭头往下拽（×）。
// 时刻取自配音的逐词时间（分镜时间 = 念到这个词的时刻），见 ../ep008.py。
(function () {
  var E = window.EP008, K = E.K, H = E.HEX;
  var RX = 470;
  var T = {
    wall: 1.72, card: 2.3, word: 5.48,
    over: 7.34, gray: 8.6, bottom: 9.59, out: 11.07,
    feas: 13.01, roll1: [14.3, 0.9], ten: 15.78, edge: 17.92,
    move: [21.3, 0.8], roll2: [22.3, 0.8], ok: 23.4, noUse: 24.84, two: 26.24,
    leave: 27.85, cards: [29.99, 32.26, 33.72], push: 34.3, pull: 35.27
  };

  E.scenes[2] = {
    title: '约束：不能越过的线',
    dur: 36.65,
    cues: [
      { at: 0, s: '一步最多迈 1 格：必须守住的限制叫**约束**，$u \\le 1$' },
      { at: 6.84, s: '超过 1 格都不能选：**碗底**（2 格）**出界**了' },
      { at: 12.7, s: '能选的范围里最好的是 1 格（10 分）：贴在约束的边上' },
      { at: 20.21, s: '要是最多迈 3 格：碗底在范围里，约束不起作用' },
      { at: 27.75, s: '机器人的约束：关节有范围、电机的力有上限、脚只能推不能拉' }
    ],
    build: function (K) {
      var s = K.sceneSvg('约束：一步最多只能迈一格，超过一格的迈法都不能选，碗底两格出界了；能选的范围里代价最小的是一格，十分，最好的解贴在约束的边上；要是最多能迈三格，碗底在范围里，约束不起作用，答案还是两格；机器人里的约束：关节只能在一定范围里转，电机的力有上限，脚只能推地面不能拉地面');

      /* ══ 左：小跑道 + 代价图 + 红墙 ══ */
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
      var feasLab = E.halo(E.txt(left, 0, E.CH.oy - 14, '能选', 20, E.GOOD, 'middle', 900));
      var wallC = E.wall(left, s, P(0, 17.4)[1], E.CH.oy, '');
      var wallT = E.wall(left, s, E.MINI.y - 50, E.MINI.y + 14, '');
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
      E.txt(card, RX + 16, 110, '约束：一步最多迈', 22, E.INK, 'start', 900);
      var cardTex = E.tex(card, RX + 304, 109, '', 30, E.BAD, 'end', { w: 110, h: 50 });
      var line1 = E.txt(right, RX + 6, 196, '', 24, E.INK, 'start', 900);
      var line2 = E.txt(right, RX + 6, 246, '', 24, E.INK, 'start', 900);
      var line3 = E.txt(right, RX + 6, 296, '', 24, E.INK, 'start', 900);

      /* ══ 句 5：三块牌 ══ */
      var cards = [0, 1, 2].map(function (i) {
        var g = E.el('g', {}, s), x = 28 + 252 * i;
        K.paint(E.el('rect', { x: x, y: 40, width: 236, height: 344, rx: 18, 'stroke-width': 2.5 }, g), E.tint(E.BAD, 8, 'var(--demo-surface)'), E.BAD);
        return { g: g, x: x, cx: x + 118 };
      });
      // ① 关节：转轴 + 连杆，绿色的能转范围（±50°），两道红色挡块
      var J = { x: cards[0].cx, y: 250, r: 96, lim: 50 };
      var jg = cards[0].g;
      function polar(a, r) { var q = a * Math.PI / 180; return [J.x + r * Math.sin(q), J.y - r * Math.cos(q)]; }
      var a0 = polar(-J.lim, J.r + 18), a1 = polar(J.lim, J.r + 18);
      K.paint(E.el('path', { d: 'M ' + a0[0].toFixed(1) + ' ' + a0[1].toFixed(1) + ' A ' + (J.r + 18) + ' ' + (J.r + 18) + ' 0 0 1 ' + a1[0].toFixed(1) + ' ' + a1[1].toFixed(1),
        fill: 'none', 'stroke-width': 6, 'stroke-linecap': 'round' }, jg), null, E.GOOD);
      [-1, 1].forEach(function (sg) {
        var p = polar(sg * (J.lim + 4), J.r - 6), q = polar(sg * (J.lim + 4), J.r + 34);
        K.paint(E.el('line', { x1: p[0], y1: p[1], x2: q[0], y2: q[1], 'stroke-width': 7, 'stroke-linecap': 'round' }, jg), null, E.BAD);
      });
      var link = K.paint(E.el('line', { x1: J.x, y1: J.y, 'stroke-width': 16, 'stroke-linecap': 'round' }, jg), null, E.INK2);
      K.paint(E.el('circle', { cx: J.x, cy: J.y, r: 15, 'stroke-width': 4 }, jg), E.S2, E.INK);
      E.txt(jg, J.x, 86, '关节', 26, E.INK, 'middle', 900);
      E.txt(jg, J.x, 330, '只能在一定', 22, E.INK2, 'middle', 700);
      E.txt(jg, J.x, 360, '范围里转', 22, E.INK2, 'middle', 700);
      // ② 电机：力的竖条，到红线就顶住
      var M = { x: cards[1].cx, y0: 300, y1: 112, cap: 150 };
      var mg = cards[1].g;
      K.paint(E.el('rect', { x: M.x - 26, y: M.y1, width: 52, height: M.y0 - M.y1, rx: 10, 'stroke-width': 2.5 }, mg), E.S2, E.BORDER);
      var bar = K.paint(E.el('rect', { x: M.x - 20, width: 40, rx: 6 }, mg), E.tint(E.WARN, 70, 'var(--demo-surface)'));
      K.paint(E.el('line', { x1: M.x - 40, y1: M.cap, x2: M.x + 40, y2: M.cap, 'stroke-width': 4, 'stroke-linecap': 'round' }, mg), null, E.BAD);
      E.halo(E.txt(mg, M.x + 46, M.cap + 7, '上限', 20, E.BAD, 'start', 900));
      E.txt(mg, M.x, 86, '电机', 26, E.INK, 'middle', 900);
      E.txt(mg, M.x, 330, '出的力', 22, E.INK2, 'middle', 700);
      E.txt(mg, M.x, 360, '有上限', 22, E.INK2, 'middle', 700);
      // ③ 脚：脚掌踩在地面上，推（绿、往上）可以，拉（红、往下）不行
      var F = { x: cards[2].cx, y: 220 };
      var fg = cards[2].g;
      K.paint(E.el('line', { x1: F.x - 92, y1: F.y, x2: F.x + 92, y2: F.y, 'stroke-width': 3 }, fg), null, E.BORDER);
      for (var k = 0; k < 7; k++) K.paint(E.el('line', { x1: F.x - 84 + 26 * k, y1: F.y + 2, x2: F.x - 96 + 26 * k, y2: F.y + 14, 'stroke-width': 2 }, fg), null, E.BORDER);
      K.paint(E.el('path', { d: E.rrect(F.x - 46, F.y - 30, 92, 30, 10), 'stroke-width': 2.5 }, fg), E.S2, E.INK);
      K.paint(E.el('rect', { x: F.x - 22, y: F.y - 92, width: 30, height: 64, rx: 8, 'stroke-width': 2.5 }, fg), E.S2, E.INK);
      var push = E.el('g', {}, fg);
      E.arrow(push, [[F.x + 66, F.y + 60], [F.x + 66, F.y - 6]], E.GOOD, 5, { head: 15 });
      E.halo(E.txt(push, F.x + 50, F.y + 52, '推', 22, E.GOOD, 'end', 900));
      E.check(push, F.x + 78, F.y + 36, 18, E.GOOD, 4);
      var pull = E.el('g', {}, fg);
      E.arrow(pull, [[F.x - 66, F.y - 20], [F.x - 66, F.y + 62]], E.BAD, 5, { head: 15, dash: '8 7' });
      E.halo(E.txt(pull, F.x - 50, F.y + 52, '拉', 22, E.BAD, 'start', 900));
      E.cross(pull, F.x - 88, F.y + 44, 9, E.BAD, 4);
      E.txt(fg, F.x, 86, '脚', 26, E.INK, 'middle', 900);
      E.txt(fg, F.x, 330, '只能推地面', 22, E.INK2, 'middle', 700);
      E.txt(fg, F.x, 360, '不能拉地面', 22, E.INK2, 'middle', 700);

      return {
        el: s,
        draw: function (st, t) {
          var f = E.fade;
          var keep = 1 - f(st, T.leave, 0.5);
          E.op(left, keep); E.op(right, keep);
          E.op(tr.foot, keep); E.op(tr.step, keep);

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
          E.op(feas, f(st, T.feas, 0.4));
          feasLab.setAttribute('x', P(lim / 2, 0)[0].toFixed(1));
          E.op(feasLab, f(st, T.feas, 0.4));

          /* 小球：碗底 → 1 格（贴着墙）→ 回到碗底 */
          var ub = E.steps(st, [[0, 2], [T.roll1[0], 1], [T.roll2[0], 2]], 0.85);
          var bp = P(ub, E.cost(ub));
          ball.setAttribute('cx', bp[0].toFixed(1)); ball.setAttribute('cy', bp[1].toFixed(1));
          var bb = E.bump(st, T.bottom, 0.8) + E.bump(st, T.edge, 1.0) + E.bump(st, T.two, 0.8);
          ball.setAttribute('r', (11 + 4 * Math.min(1, bb)).toFixed(2)); E.glow(ball, E.HI, bb > 0.05);
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
          E.op(ballLab, f(st, T.ten, 0.3) * (1 - f(st, T.move[0], 0.3)) + f(st, T.two, 0.3));
          okC.setAttribute('transform', 'translate(' + (P(2, 8)[0] + 22) + ' ' + (P(2, 8)[1] - 34) + ')');
          E.drawOn(okC, f(st, T.ok, 0.4));

          /* 右栏 */
          E.pop(card, RX + 160, 101, f(st, T.card, 0.35), 0.85);
          cardTex.setTex('u\\le ' + (st < T.move[0] + T.move[1] / 2 ? 1 : 3));
          var wb = E.bump(st, T.word, 0.8);
          cardR.style.strokeWidth = 3 + 2 * wb; E.glow(cardR, E.BAD, wb > 0.05);
          var A = st < T.move[0];
          line1.textContent = A ? '碗底 u = 2：出界' : '碗底 u = 2：在范围里';
          line1.style.fill = A ? E.BAD : E.GOOD;
          E.op(line1, A ? f(st, T.out, 0.3) * (1 - f(st, T.move[0] - 0.3, 0.3)) : f(st, T.ok, 0.3));
          line2.textContent = A ? '能选：0 ≤ u ≤ 1' : '约束不起作用';
          line2.style.fill = A ? E.GOOD : E.MUT;
          E.op(line2, A ? f(st, T.feas, 0.3) * (1 - f(st, T.move[0] - 0.3, 0.3)) : f(st, T.noUse, 0.3));
          line3.textContent = A ? '最好：u = 1，代价 10' : '答案：还是 u = 2';
          line3.style.fill = E.HI;
          E.op(line3, A ? f(st, T.ten, 0.3) * (1 - f(st, T.move[0] - 0.3, 0.3)) : f(st, T.two, 0.3));

          /* 句 5：三块牌 */
          cards.forEach(function (c, i) { E.pop(c.g, c.cx, 212, f(st, T.cards[i], 0.35), 0.85); });
          // 连杆在 ±58° 之间摆，碰到 ±50° 的挡块就停住
          var ang = Math.max(-J.lim, Math.min(J.lim, 64 * Math.sin(2 * Math.PI * (t - T.cards[0]) / 2.6)));
          var tip = polar(ang, J.r);
          link.setAttribute('x2', tip[0].toFixed(1)); link.setAttribute('y2', tip[1].toFixed(1));
          K.paint(link, null, Math.abs(ang) >= J.lim - 0.01 ? E.BAD : E.INK2);
          // 力的竖条：想要的力一涨一落，超过上限的部分顶住
          var want = 0.5 + 0.62 * Math.sin(2 * Math.PI * (t - T.cards[1]) / 2.4), top = M.y0 - (M.y0 - M.y1 - 6) * want;
          var y = Math.max(M.cap, Math.min(M.y0 - 6, top));
          bar.setAttribute('y', y.toFixed(1)); bar.setAttribute('height', (M.y0 - 3 - y).toFixed(1));
          K.paint(bar, top <= M.cap ? E.tint(E.BAD, 70, 'var(--demo-surface)') : E.tint(E.WARN, 70, 'var(--demo-surface)'));
          E.op(push, f(st, T.push, 0.3)); E.op(pull, f(st, T.pull, 0.3));
        }
      };
    }
  };
})();
