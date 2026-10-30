// 片头主图（800×420）：三句旁白各一段——
// ① 回顾：顶上出小牌「前面七集：位置、朝向」；侧视 G1 站在跑道起点，4 格外是终点线；
// ② 「优化」顶上弹出「优化」大牌；「问题」跑道上方出问号，「一大堆做法」跑道上依次冒出一串虚线落脚点（迈 0.5、1、…、4 格），
//    「最好」其中迈 2 格的那个变成亮黄、画出弧形箭头并打勾，别的淡下去；
// ③ 「三样东西」顶上换成三块牌：「变量」（迈几格）、「目标」（扣分最少）、「约束」（最多迈几格）按念到的词弹出；
//    「交给电脑去找」右上角出一只小碗（代价曲线），小球顺着碗壁滚到碗底。
// 时刻取自配音的逐词时间（../ep008.py 的 [from, to] 等于每句配音实长，分镜时间 = 念到这个词的时刻）。
(function () {
  var E = window.EP008, K = E.K;
  var PREV = [0.6, 4.6], OPT = [9.2, 16.3], Q = 10.75, MANY = 12.37, BEST = 14.0;
  var THREE = [18.61, 19.27, 19.83], BOWL = 21.11, ROLL = [21.5, 22.5];
  var X0 = 262, Y = 340, W = 106;
  var CAND = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4];

  E.hero = function (K) {
    var s = K.sceneSvg('片头主图：机器人离终点线还有四格，这一步迈多远？可以迈的远近有一大堆，要挑出最好的那一个；写清楚变量、目标、约束三样东西，剩下的交给电脑，像小球滚到碗底一样找到最好的选择');

    var tr = E.track(s, { x0: X0, y: Y, w: W, scale: 1.5, arcH: 50 });
    E.op(tr.rem, 0);

    /* ② 一串虚线落脚点 + 各自的细弧；最好的那个（2 格）用跑道自带的绿脚印、弧和标签 */
    var cand = CAND.map(function (u) {
      var g = E.el('g', {}, s);
      var x = tr.X(u);
      var arc = K.paint(E.el('path', { d: 'M ' + X0 + ' ' + (Y - 8) + ' Q ' + ((X0 + x) / 2) + ' ' + (Y - 8 - 2 * Math.min(50, 18 + 0.5 * (x - X0))) + ' ' + x + ' ' + (Y - 10),
        fill: 'none', 'stroke-width': 2, 'stroke-dasharray': '4 6' }, g), null, E.MUT);
      var e = K.paint(E.el('ellipse', { cx: x, cy: Y - 2, rx: 18, ry: 5.5, 'stroke-width': 2, 'stroke-dasharray': '4 4' }, g), 'var(--demo-surface)', E.INK2);
      return { g: g, arc: arc, e: e, u: u };
    });
    s.appendChild(tr.step); s.appendChild(tr.foot);   // 选中的那个压在虚线上面
    var q = E.halo(E.txt(s, tr.X(2), 222, '?', 54, E.HI, 'middle', 900));
    var ok = E.check(s, tr.X(2) + 26, Y - 46, 22, E.HI, 4.5);

    /* 顶上的小牌：前面七集 → 优化 → 三样东西 */
    var prev = E.tag(s, 270, 6, 260, 38, '前面七集：位置、朝向', E.INK2, { size: 20, ink: E.INK2, weight: 700, fill: E.S2 });
    var opt = E.tag(s, 330, 2, 140, 52, '优化', E.HI, { size: 32, ink: E.HI, weight: 900, rx: 14, sw: 3 });
    var subs = ['迈几格', '扣分最少', '最多迈几格'];
    var three = E.THREE.map(function (d, i) {
      var g = E.el('g', {}, s), x = 40 + 160 * i;
      E.tag(g, x, 4, 130, 46, d.s, d.c, { size: 26, ink: d.c, weight: 900, rx: 14, sw: 3 });
      E.txt(g, x + 65, 80, subs[i], 20, E.INK2, 'middle', 700);
      return { g: g, cx: x + 65 };
    });

    /* ③ 右上角的小碗：代价曲线 (4 − u)² + u²，小球从 u = 0 滚到碗底 u = 2 */
    var bowl = E.el('g', {}, s);
    var B = { ox: 560, oy: 150, su: 52, sc: 7.2 };
    function bp(u) { return [B.ox + B.su * u, B.oy - B.sc * E.cost(u)]; }
    var pts = [];
    for (var k = 0; k <= 40; k++) pts.push(bp(4 * k / 40));
    var curve = K.paint(E.el('path', { d: K.polyPath(pts), fill: 'none', 'stroke-width': 4, 'stroke-linecap': 'round' }, bowl), null, E.VIOLET);
    var ball = K.paint(E.el('circle', { r: 9, 'stroke-width': 2.5 }, bowl), E.HI, 'var(--demo-surface)');
    var bottom = E.halo(E.txt(bowl, bp(2)[0], bp(2)[1] + 30, '碗底', 18, E.HI, 'middle', 900));
    E.halo(E.txt(bowl, B.ox + 2 * B.su, 18, '交给电脑去找', 20, E.INK, 'middle', 900));

    return {
      el: s,
      draw: function (t) {
        var f = E.fade;
        E.op(tr.g, f(t, 0, 0.5));
        E.pop(prev.g, 400, 25, f(t, PREV[0], 0.35) * (1 - f(t, PREV[1], 0.4)), 0.85);
        E.pop(opt.g, 400, 28, f(t, OPT[0], 0.35) * (1 - f(t, OPT[1], 0.4)), 0.7);

        /* ② 问号、一串虚线落脚点、选中 2 格 */
        E.pop(q, tr.X(2), 205, f(t, Q, 0.3) * (1 - f(t, BEST, 0.3)), 0.6);
        var pick = f(t, BEST, 0.4);
        cand.forEach(function (c, i) {
          var on = f(t, MANY + 0.09 * i, 0.25);
          E.op(c.g, on * (1 - 0.7 * pick * (c.u === 2 ? 0 : 1)) * (c.u === 2 ? 1 - pick : 1));
        });
        tr.setU(2, '迈 2 格');
        E.op(tr.foot, pick); E.op(tr.step, pick);
        E.drawOn(ok, f(t, BEST + 0.6, 0.4));

        /* ③ 三块牌、小碗与小球 */
        three.forEach(function (d, i) { E.pop(d.g, d.cx, 40, f(t, THREE[i], 0.3), 0.8); });
        E.pop(bowl, B.ox + 2 * B.su, 90, f(t, BOWL, 0.4), 0.85);
        var r = K.ease(K.seg(t, ROLL[0], ROLL[1])), u = 2 * r, p = bp(u);
        ball.setAttribute('cx', p[0].toFixed(1)); ball.setAttribute('cy', (p[1] - 9).toFixed(1));
        E.op(bottom, f(t, ROLL[1], 0.3));
        curve.style.strokeWidth = 4;
      }
    };
  };
})();
