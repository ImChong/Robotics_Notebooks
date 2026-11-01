// 008 第 2 幕：目标——给每一种迈法算代价。左上是小跑道（和代价图横轴对齐：迈 u 格的脚印正好在图上 u 的正上方），左下是代价图，右栏放式子和表。
// 句 1「打个分」代价图淡入，「越少越好」竖轴旁出「越少越好」和向下的箭头；
// 句 2「第一项」右上出第一块牌「① 还剩几格的平方 (4 − u)²」（蓝），「还剩几格」小跑道上出「还剩 … 格」尺寸线；
// 句 3 脚印依次停在 0、1、4 格，图上打出蓝点 (0, 16)、(1, 9)、(4, 0)，句末蓝色曲线画出来；
// 句 4「平方」右栏出四个正方形（边长 1、2、3、4 格 → 面积 1、4、9、16），「狠」最大的那个鼓一下；
// 句 5「迈四格最好」蓝线最低点 (4, 0) 写「最好？」，「费力」弧变粗，「摔」小 G1 往前一栽再站回来，「第二项」出第二块牌「② 步子的平方 u²」（琥珀），
//    「平方」琥珀色曲线画出来；
// 句 6「两项加起来」两块牌换成一行式子「代价 = (4 − u)² + u²」，「代价」鼓一下，「式子」下面括出「目标函数」；
// 句 7 右栏出表：迈 0…4 格，按念到的数一格格填「还剩² + 步子² = 代价」，每行算完图上打一个紫点，脚印跟着停到那一格；
// 句 8「迈两格」第 3 行亮黄框，「代价最小」紫点 (2, 8) 套亮黄圈，「两项」蓝线、琥珀线上 u = 2 的两点（各 4 分）鼓一下；
// 句 9「连起来」紫点连成曲线（蓝、琥珀两条淡下去），「碗」曲线下垫一层淡紫，「碗底」亮黄小球落到 (2, 8) 并写「碗底（最优解）」，「最好」打勾。
// 时刻取自配音的逐词时间（分镜时间 = 念到这个词的时刻），见 ../ep008.py。
(function () {
  var E = window.EP008, K = E.K, H = E.HEX;
  var RX = 470;                                             // 右栏左缘
  var T = {
    chart: 0.6, less: 3.56,
    card1: 5.8, rem: 7.64,
    pts1: [[11.49, 0], [14.27, 1], [16.94, 4]], val1: [13.38, 16.23, 18.68], curve1: [19.0, 1.1],
    sq: 20.53, sqOn: 21.6, sqLab: 22.44, hard: 22.98, sqOut: 24.3,
    best4: [25.8, 26.36], tired: 27.89, fall: [29.3, 30.4], card2: 30.98, curve2: [32.2, 1.1],
    sum: 34.12, cost: 37.07, func: [38.34, 39.07],
    rows: [41.92, 44.59, 46.7, 48.9, 51.14],
    cells: [[42.64, 43.11, 43.68], [45.24, 45.46, 46.02], [47.44, 47.7, 48.2], [49.68, 49.89, 50.47], [51.94, 52.26, 53.21]],
    pick: 55.02, min: 55.81, eight: 57.08, both: 57.45,
    dim: 60.6, join: [61.4, 1.0], bowl: 62.6, ball: [63.36, 0.5], best: 64.43
  };

  E.scenes[1] = {
    title: '目标：给每种迈法算代价',
    dur: 66.24,
    cues: [
      { at: 0, s: '给每种迈法打个分：扣的分越少越好' },
      { at: 5.71, s: '第一项：还剩几格，扣几格的平方 $(4-u)^2$' },
      { at: 11.4, s: '迈 0 格扣 16，迈 1 格扣 9，迈 4 格到线、不扣分' },
      { at: 20.28, s: '用平方：剩得越多，扣得越狠' },
      { at: 24.26, s: '只看第一项：迈 4 格最好？' },
      { at: 27.3, s: '步子越大越费力、越容易摔 → 第二项：$u^2$' },
      { at: 34.03, s: '两项相加 = **代价**；算代价的式子 = **目标函数**' },
      { at: 41.01, s: '算一遍：迈 0、1、2、3、4 格 → 16、10、8、10、16' },
      { at: 54.93, s: '迈 2 格代价最小，只有 8 分：两项都照顾到了' },
      { at: 59.83, s: '代价连起来是一只碗：**碗底**就是最好的选择' }
    ],
    build: function (K) {
      var s = K.sceneSvg('目标：给每一种迈法打分，第一项扣还剩几格的平方，第二项扣步子的平方，两项加起来叫代价，算代价的式子叫目标函数；迈零、一、二、三、四格的代价是十六、十、八、十、十六，迈两格最小；把代价连起来是一只碗，碗底就是最好的选择');

      /* ══ 左：小跑道 + 代价图 ══ */
      var tr = E.track(s, E.MINI);
      var ch = E.chart(s, { ox: E.CH.ox, oy: E.CH.oy, su: E.CH.su, sc: E.CH.sc, xLab: '迈几格 u', yLab: '扣分' });
      var P = ch.P;
      var less = E.el('g', {}, s);
      E.arrow(less, [[P(4.2, 0)[0] - 4, P(0, 13)[1]], [P(4.2, 0)[0] - 4, P(0, 6)[1]]], E.HI, 3.5, { head: 13 });
      E.halo(E.txt(less, P(4.2, 0)[0] - 4, P(0, 14.4)[1], '越少越好', 20, E.HI, 'middle', 900));
      var bowlFill = E.el('path', { d: ch.path(E.cost, 0, 4) + ' L ' + P(4, 16.6)[0] + ' ' + P(0, 16.6)[1] + ' L ' + P(0, 16.6)[0] + ' ' + P(0, 16.6)[1] + ' Z' }, s);
      K.paint(bowlFill, E.tint(E.VIOLET, 22, 'transparent'));
      var c1 = ch.curve(E.rem, E.ACC, 4), c2 = ch.curve(E.eff, E.WARN, 4), c3 = ch.curve(E.cost, E.VIOLET, 5);
      var d1 = T.pts1.map(function (p) { return ch.dot(E.ACC, 7.5).set(p[1], E.rem(p[1])); });
      var v1 = T.pts1.map(function (p) {
        var q = P(p[1], E.rem(p[1]));
        return E.halo(E.txt(s, q[0] + (p[1] === 4 ? -16 : 16), q[1] + (p[1] === 4 ? -12 : 6), String(E.rem(p[1])), 20, E.ACC, p[1] === 4 ? 'end' : 'start', 900));
      });
      var best4 = E.halo(E.txt(s, P(4, 0)[0] - 16, P(0, 2.6)[1] - 4, '最好？', 22, E.ACC, 'end', 900));
      // u = 2 处的一根叠起来的柱子：下半截蓝（还剩² = 4）、上半截琥珀（步子² = 4），合起来 8
      var both = E.el('g', {}, s);
      [[0, 4, E.ACC], [4, 8, E.WARN]].forEach(function (b) {
        var p0 = P(2, b[0]), p1 = P(2, b[1]);
        K.paint(E.el('rect', { x: p0[0] - 9, y: p1[1], width: 18, height: p0[1] - p1[1], rx: 3, 'stroke-width': 2 }, both), E.tint(b[2], 55, 'var(--demo-surface)'), b[2]);
        E.halo(E.txt(both, p0[0] - 16, (p0[1] + p1[1]) / 2 + 7, '4', 20, b[2], 'end', 900));
      });
      var d3 = [0, 1, 2, 3, 4].map(function (u) { return ch.dot(E.VIOLET, 8).set(u, E.cost(u)); });
      var ring = K.paint(E.el('circle', { cx: P(2, 8)[0], cy: P(2, 8)[1], r: 15, fill: 'none', 'stroke-width': 3.5 }, s), null, E.HI);
      var ball = K.paint(E.el('circle', { r: 11, 'stroke-width': 3 }, s), E.HI, 'var(--demo-surface)');
      var bottom = E.halo(E.txt(s, P(2, 8)[0], P(0, 8)[1] + 44, '碗底（最优解）', 22, E.HI, 'middle', 900));   // 最优解：Boyd & Vandenberghe 4.1.1 节的 optimal point
      var okBest = E.check(s, P(2, 8)[0] + 88, P(0, 8)[1] + 28, 20, E.HI, 4.5);

      /* ══ 右栏：两块牌 → 一行式子；四个正方形；表 ══ */
      function card(y, n, tex, color) {
        var g = E.el('g', {}, s);
        K.paint(E.el('rect', { x: RX, y: y, width: 320, height: 58, rx: 14, 'stroke-width': 2.5 }, g), E.tint(color, 12, 'var(--demo-surface)'), color);
        E.txt(g, RX + 16, y + 37, n, 22, E.INK, 'start', 900);
        E.tex(g, RX + 306, y + 38, tex, 28, color, 'end', { w: 120, h: 50 });
        return g;
      }
      var card1 = card(8, '① 还剩几格的平方', '(4-u)^2', E.ACC);
      var card2 = card(76, '② 步子的平方', 'u^2', E.WARN);

      // 四个正方形：边长 1、2、3、4 格（一格 11 px），底边对齐
      var sqG = E.el('g', {}, s), sqs = [];
      var SQB = 304, SQ = 15, SQX = [RX + 20, RX + 64, RX + 126, RX + 210];
      [1, 2, 3, 4].forEach(function (k, i) {
        var g = E.el('g', {}, sqG), a = SQ * k;
        for (var r = 0; r < k; r++) for (var c = 0; c < k; c++) {
          K.paint(E.el('rect', { x: SQX[i] + SQ * c, y: SQB - a + SQ * r, width: SQ, height: SQ, 'stroke-width': 1.2 }, g), E.tint(E.ACC, 30, 'var(--demo-surface)'), E.ACC);
        }
        var lab = E.txt(g, SQX[i] + a / 2, SQB + 28, '剩 ' + k, 18, E.INK2, 'middle', 700);
        var val = E.txt(g, SQX[i] + a / 2, SQB - a - 10, String(k * k), 22, E.ACC, 'middle', 900);
        sqs.push({ g: g, cx: SQX[i] + a / 2, a: a, lab: lab, val: val });
      });
      E.txt(sqG, RX + 160, 178, '扣的分 = 还剩格数的平方', 20, E.INK2, 'middle', 700);

      // 一行式子：代价 = (4 − u)² + u²，下面括出「目标函数」
      var form = E.el('g', {}, s);
      var costWord = E.txt(form, RX + 6, 44, '代价', 30, E.VIOLET, 'start', 900);
      E.tex(form, RX + 70, 42, '=' + E.ctex(H.acc, '(4-u)^2') + '+' + E.ctex(H.warn, 'u^2'), 30, E.INK, 'start', { w: 250, h: 52 });
      var brace = K.paint(E.el('path', { d: 'M ' + (RX + 4) + ' 62 V 72 H ' + (RX + 300) + ' V 62 M ' + (RX + 152) + ' 72 V 80', fill: 'none', 'stroke-width': 2.5 }, form), null, E.VIOLET);
      var funcTag = E.tag(form, RX + 92, 82, 120, 34, '目标函数', E.VIOLET, { size: 20, ink: E.VIOLET, weight: 900 });

      // 表：迈 u 格 | 还剩² | + | 步子² | = | 代价
      var COL = [RX + 34, RX + 112, RX + 158, RX + 200, RX + 244, RX + 292], RY0 = 196, RH = 40;
      var table = E.el('g', {}, s);
      var hl = K.paint(E.el('rect', { x: RX + 2, y: RY0 + 2 * RH - 28, width: 318, height: 38, rx: 10, 'stroke-width': 3 }, table), E.tint(E.HI, 14, 'var(--demo-surface)'), E.HI);
      [['迈', E.GOOD], ['还剩²', E.ACC], ['', E.INK], ['步子²', E.WARN], ['', E.INK], ['代价', E.VIOLET]].forEach(function (h, i) {
        if (h[0]) E.txt(table, COL[i], RY0 - 38, h[0], 20, h[1], 'middle', 900);
      });
      K.paint(E.el('line', { x1: RX + 4, y1: RY0 - 27, x2: RX + 318, y2: RY0 - 27, 'stroke-width': 1.5 }, table), null, E.BORDER);
      var rows = [0, 1, 2, 3, 4].map(function (u, k) {
        var y = RY0 + RH * k, g = E.el('g', {}, table);
        function cell(i, s1, c) { var n = E.txt(g, COL[i], y, s1, 24, c, 'middle', i === 5 ? 900 : 700); n.style.fontFamily = 'KaTeX_Main, serif'; return n; }
        var cu = E.txt(g, COL[0], y, u + ' 格', 22, E.GOOD, 'middle', 900);
        return { g: g, y: y, u: cu, cells: [[cell(1, String(E.rem(u)), E.ACC)], [cell(2, '+', E.INK2), cell(3, String(E.eff(u)), E.WARN)], [cell(4, '=', E.INK2), cell(5, String(E.cost(u)), E.VIOLET)]] };
      });


      return {
        el: s,
        draw: function (st, t) {
          var f = E.fade;
          E.op(ch.g, f(st, T.chart, 0.5));
          E.op(less, f(st, T.less, 0.4) * (1 - f(st, T.card1 - 0.3, 0.4)));

          /* 脚印：句 3 停在 0、1、4；句 5 迈四格；句 7 跟着表一行行停；句 8 回到 2 */
          var keys = [[0, 2]].concat(T.pts1.map(function (p) { return [p[0], p[1]]; }), [[T.best4[0], 4]],
            T.rows.map(function (a, k) { return [a, k]; }), [[T.pick, 2]]);
          var u = E.steps(st, keys, 0.45);
          tr.setU(u);
          E.op(tr.foot, 1); E.op(tr.step, 1);
          E.op(tr.rem, f(st, T.rem, 0.4));
          var tired = f(st, T.tired, 0.4) * (1 - f(st, T.card2, 0.5));
          tr.step.firstChild.style.strokeWidth = 3.5 + 3 * tired;
          K.paint(tr.step.firstChild, null, tired > 0.5 ? E.WARN : E.GOOD);
          // 「摔」：小 G1 往前一栽再站回来（绕脚尖转）
          var fall = E.bump(st, T.fall[0], T.fall[1] - T.fall[0]);
          if (tr.robot) tr.robot.place({ rot: 16 * fall, pivot: [28, 388] });

          /* 句 2、3：第一项的蓝点和蓝线 */
          E.pop(card1, RX + 160, 37, f(st, T.card1, 0.35) * (1 - f(st, T.sum, 0.4)), 0.85);
          var dimOld = 1 - 0.75 * f(st, T.dim, 0.6);
          d1.forEach(function (d, i) {
            E.pop(d.g, P(T.pts1[i][1], 0)[0], P(0, E.rem(T.pts1[i][1]))[1], f(st, T.pts1[i][0] + 0.3, 0.3) * (1 - f(st, T.curve2[0], 0.5)), 0.4);
            E.op(v1[i], f(st, T.val1[i], 0.3) * (1 - f(st, T.curve2[0], 0.5)));
          });
          E.drawOn(c1, f(st, T.curve1[0], T.curve1[1])); c1.style.opacity = dimOld;
          E.op(best4, f(st, T.best4[1], 0.3) * (1 - f(st, T.card2, 0.4)));

          /* 句 4：四个正方形 */
          E.op(sqG, f(st, T.sq, 0.4) * (1 - f(st, T.sqOut, 0.4)));
          sqs.forEach(function (q, i) {
            E.pop(q.g, q.cx, SQB - q.a / 2, f(st, T.sqOn + 0.18 * i, 0.3), 0.3);
            E.op(q.val, f(st, T.sqLab + 0.12 * i, 0.3));
          });
          var hb = E.bump(st, T.hard, 0.9);
          E.scaleAt(sqs[3].g, sqs[3].cx, SQB - sqs[3].a / 2, 1 + 0.1 * hb);
          E.glow(sqs[3].g, E.ACC, hb > 0.05);

          /* 句 5：第二项 */
          E.pop(card2, RX + 160, 105, f(st, T.card2, 0.35) * (1 - f(st, T.sum, 0.4)), 0.85);
          E.drawOn(c2, f(st, T.curve2[0], T.curve2[1])); c2.style.opacity = dimOld;

          /* 句 6：式子、目标函数 */
          E.op(form, f(st, T.sum + 0.2, 0.4));
          var cb = E.bump(st, T.cost, 0.8);
          E.scaleAt(costWord, RX + 36, 34, 1 + 0.15 * cb);
          E.drawOn(brace, f(st, T.func[0], 0.4));
          E.pop(funcTag.g, RX + 152, 99, f(st, T.func[1], 0.3), 0.8);

          /* 句 7：表 */
          E.op(table, f(st, T.rows[0] - 0.4, 0.4));
          rows.forEach(function (r, k) {
            E.op(r.u, f(st, T.rows[k], 0.3));
            r.cells.forEach(function (cs, j) { cs.forEach(function (c) { E.op(c, f(st, T.cells[k][j], 0.25)); }); });
            E.pop(d3[k].g, P(k, 0)[0], P(0, E.cost(k))[1], f(st, T.cells[k][2], 0.3), 0.3);
          });

          /* 句 8：迈两格最小 */
          E.op(hl, f(st, T.pick, 0.3));
          E.op(ring, f(st, T.min, 0.3) * (1 - f(st, T.ball[0], 0.3)));
          var bb = E.bump(st, T.eight, 0.8);
          E.scaleAt(rows[2].cells[2][1], COL[5], rows[2].y - 8, 1 + 0.3 * bb);
          var b2 = E.bump(st, T.both, 1.2);
          E.op(both, f(st, T.both, 0.3) * (1 - f(st, T.dim, 0.4)));
          [rows[2].cells[0][0], rows[2].cells[1][1]].forEach(function (c) { E.scaleAt(c, +c.getAttribute('x'), rows[2].y - 8, 1 + 0.25 * b2); });

          /* 句 9：连成一只碗、碗底 */
          E.drawOn(c3, f(st, T.join[0], T.join[1]));
          E.op(bowlFill, 0.9 * f(st, T.bowl, 0.6));
          var bp = P(2, 8);
          ball.setAttribute('cx', bp[0]); ball.setAttribute('cy', (bp[1] - 60 * (1 - K.ease(K.seg(st, T.ball[0], T.ball[0] + T.ball[1])))).toFixed(1));
          E.op(ball, f(st, T.ball[0], 0.2));
          E.op(bottom, f(st, T.ball[0] + 0.3, 0.3));
          E.drawOn(okBest, f(st, T.best, 0.4));
        }
      };
    }
  };
})();
