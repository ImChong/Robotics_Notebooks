// 片头主图（800×420）：左边 G1，右边一列三张写法卡片（欧拉角 / 四元数 / 旋转矩阵）。三句旁白各一段——
// ① 「上一集」左上小牌；「旋转矩阵」G1 原地往左转 90°，最下面的矩阵卡弹出、九个数跟着转身实时变，「九个数」「两条规矩」卡上小字依次写出；
//    「同一个朝向」G1 脚下的圆和骨盆上的三根小箭头亮一下；「三个角」欧拉角卡弹出 (90°, 0°, 0°)，「四个数」四元数卡弹出 (0.71, 0, 0, 0.71)；
// ② 「欧拉角」「四元数」「旋转矩阵」三张卡的名字依次鼓一下；「坑」三张卡右边各出一个问号；
// ③ 「万向锁」「两套数」「冗余」问号依次换成红牌，「给人看」「内部计算」「连乘」红牌下面依次出绿牌。
// 时刻取自配音的逐词时间（../ep006.py 的 [from, to] 等于每句配音实长，分镜时间 = 念到这个词的时刻）。
(function () {
  var E = window.EP006, K = E.K;
  var D = Math.PI / 180;
  var T = {
    prev: [0.1, 4.4], turn: [1.03, 1.0], mat: 1.03, nine: 2.15, rules: 3.33, same: 4.76, euler: 6.53, quat: 7.62,
    names: [10.71, 11.46, 12.25], ask: 14.66,
    bad: [18.15, 20.79, 22.62], use: [24.67, 26.2, 27.96]
  };
  var GX = 150, GY = 370, GH = 316;   // 脚下的圆不出画面
  var CX = 300, CW = 490, TX = 640, TW = 138;
  var ROWS = [{ y: 18, h: 100 }, { y: 130, h: 100 }, { y: 242, h: 150 }];

  function bump(st, a, d) { var u = K.seg(st, a, a + d); return u > 0 && u < 1 ? Math.sin(Math.PI * u) : 0; }

  E.hero = function (K) {
    var s = K.sceneSvg('片头主图：机器人往左转九十度，同一个朝向可以写成欧拉角三个数、四元数四个数、旋转矩阵九个数；欧拉角会万向锁、只给人看，四元数一个朝向有两套数、做内部计算，旋转矩阵冗余、用来连乘');

    /* ── 左：G1、脚下的圆、骨盆上的三根小箭头 ── */
    var rob = E.el('g', {}, s);
    var ring = K.paint(E.el('path', { fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '6 6' }, rob), null, E.BORDER);
    var g1 = E.g1(rob, GX, GY, GH);
    var fr = E.frame3(rob, 5);
    ring.setAttribute('d', E.ptsPath(g1.ring(0.45), true));
    var prev = E.tag(s, 20, 6, 214, 36, '上一集：旋转矩阵', E.INK2, { size: 20, ink: E.INK2, weight: 700, fill: E.S2 });

    /* ── 右：三张卡片 ── */
    var defs = [
      { name: '欧拉角', c: E.WARN, n: '3 个数', bad: '坑：万向锁', use: '用：给人看' },
      { name: '四元数', c: E.ACC, n: '4 个数', bad: '坑：两套数', use: '用：内部计算' },
      { name: '旋转矩阵', c: E.VIOLET, n: '9 个数', bad: '坑：冗余', use: '用：连乘' }
    ];
    var cards = defs.map(function (d, i) {
      var r = ROWS[i], cd = E.card(s, CX, r.y, CW, r.h, d.name, d.c, 26);
      var cnt = E.txt(cd.g, CX + 22, r.y + 72, d.n, 20, E.INK2, 'start', 700);
      var ask = E.txt(cd.g, TX + TW / 2, r.y + r.h / 2 + 16, '?', 44, E.INK2, 'middle', 900);
      var bad = E.tag(cd.g, TX, r.y + r.h / 2 - 40, TW, 36, d.bad, E.BAD, { size: 20, ink: E.BAD, weight: 900, rx: 10 });
      var use = E.tag(cd.g, TX, r.y + r.h / 2 + 4, TW, 36, d.use, E.GOOD, { size: 20, ink: E.GOOD, weight: 900, rx: 10 });
      return { g: cd.g, name: cd.name, cnt: cnt, ask: ask, bad: bad, use: use, cy: r.y + r.h / 2 };
    });
    E.txt(cards[0].g, 516, ROWS[0].y + 62, '(90°, 0°, 0°)', 30, E.INK, 'middle', 700);   // 度数用正文字体：KaTeX_Main 的 ° 排得像上标小圈
    E.txt(cards[1].g, 524, ROWS[1].y + 62, '(0.71, 0, 0, 0.71)', 28, E.INK, 'middle', 700).style.fontFamily = 'KaTeX_Main, serif';
    var rules = E.txt(cards[2].g, CX + 22, ROWS[2].y + 100, '2 条规矩', 20, E.INK2, 'start', 700);
    var m3 = E.grid(cards[2].g, 548, ROWS[2].y + 76, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], { size: 22, dx: 56, dy: 30, colors: [E.GOOD, E.WARN, E.ACC] });

    return {
      el: s,
      draw: function (t) {
        var f = E.fade;

        /* G1 往左转 90°；「同一个朝向」脚下的圆和小箭头亮一下 */
        var yaw = 90 * D * f(t, T.turn[0], T.turn[1]);
        g1.set(yaw); fr.set(g1.frame(0.5));
        var sb = bump(t, T.same, 0.9);
        ring.style.stroke = sb > 0.05 ? 'var(--hi)' : ''; ring.style.strokeOpacity = 0.6 + 0.4 * sb;
        fr.arrows.forEach(function (a) { a.line.style.strokeWidth = 5 + 3 * sb; });
        E.pop(prev.g, 127, 24, f(t, T.prev[0], 0.35) * (1 - f(t, T.prev[1], 0.4)), 0.85);

        /* 卡片：矩阵 → 欧拉角 → 四元数 */
        var c = Math.cos(yaw), sn = Math.sin(yaw);
        m3.set([[c, -sn, 0], [sn, c, 0], [0, 0, 1]]);
        E.op(cards[2].cnt, f(t, T.nine, 0.3)); E.op(rules, f(t, T.rules, 0.3));
        [T.euler, T.quat, T.mat].forEach(function (a, i) { E.pop(cards[i].g, CX + CW / 2, cards[i].cy, f(t, a, 0.35), 0.9); });
        E.op(cards[0].cnt, f(t, T.euler + 0.2, 0.3)); E.op(cards[1].cnt, f(t, T.quat + 0.2, 0.3));

        cards.forEach(function (cd, i) {
          var nb = bump(t, T.names[i], 0.8);
          E.scaleAt(cd.name, CX + 22, cd.cy - 30, 1 + 0.18 * nb);
          E.op(cd.ask, f(t, T.ask + 0.15 * i, 0.3) * (1 - f(t, T.bad[i] - 0.1, 0.25)));
          E.pop(cd.bad.g, TX + TW / 2, cd.cy - 22, f(t, T.bad[i], 0.3), 0.8);
          E.pop(cd.use.g, TX + TW / 2, cd.cy + 22, f(t, T.use[i], 0.3), 0.8);
        });
      }
    };
  };
})();
