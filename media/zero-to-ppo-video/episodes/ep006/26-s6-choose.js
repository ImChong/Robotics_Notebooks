// 006 第 6 幕：怎么选——按用途换着用。上面一行标题跟着旁白换；中间三张卡片（欧拉角 / 四元数 / 旋转矩阵）并排；底下一条换算流程。
// 句 1「三种写法」三张卡片出来，「各有各的坑」三块红牌依次出来，「十全十美」标题写「没有十全十美的一种」；
// 句 2「三个自由度」标题换成「转动只有 3 个自由度」，「只用三个数」「出毛病」欧拉角卡写红字；「多用几个数」「加上规矩」四元数、矩阵卡写出 4 − 1 = 3、9 − 6 = 3；
// 句 3「按用途换着用」标题换，「欧拉角」「给人看」「配置文件」「屏幕上」欧拉角卡写用途、亮框；
// 句 4「四元数」「存朝向」「取中间值」「仿真器和传感器」四元数卡写用途、亮框；
// 句 5「旋转矩阵」「串起来」「正运动学」「第十一集」「矩阵指数」「第十集」矩阵卡写用途、亮框；
// 句 6 底下流程：人输入（欧拉角）→ 内部（四元数）→ 要连乘（矩阵）→ 显示给人看（欧拉角），按念到的顺序一格格出来。
// 时刻取自配音的逐词时间（分镜时间 = 念到这个词的时刻），见 ../ep006.py。
(function () {
  var E = window.EP006, K = E.K;
  var T = {
    cards: 0.1, bad: [1.31, 1.55, 1.77], perfect: 3.6,
    dof: 6.84, three: 7.95, trouble: 8.99, more: 14.04, rules: 15.05,
    useHead: 17.99, use: [[19.8, 21.14, 22.37, 23.82], [25.89, 29.65, 30.45, 31.51], [34.91, 37.33, 39.93, 43.46]],
    flow: [46.19, 47.69, 49.57, 51.84]
  };
  var CW = 250, CY = 40, CH = 262, XS = [14, 275, 536];
  var DEFS = [
    { name: '欧拉角', c: E.WARN, n: '3 个数', minus: '', bad: '坑：万向锁', use: '只给人看', ex: ['配置文件里写的', '屏幕上显示的'] },
    { name: '四元数', c: E.ACC, n: '4 个数', minus: '− 1 条规矩 = 3', bad: '坑：两套数', use: '存朝向、取中间值', ex: ['仿真器、传感器', '大多用它'] },
    { name: '旋转矩阵', c: E.VIOLET, n: '9 个数', minus: '− 6 个等式 = 3', bad: '坑：冗余', use: '连乘：串起转动', ex: ['正运动学（第 11 集）', '矩阵指数（第 10 集）'] }
  ];

  function bump(st, a, d) { var u = K.seg(st, a, a + d); return u > 0 && u < 1 ? Math.sin(Math.PI * u) : 0; }

  E.scenes[5] = {
    title: '怎么选：按用途换着用',
    dur: 55.21,
    cues: [
      { at: 0, s: '三种写法各有各的坑：没有十全十美的一种' },
      { at: 5.78, s: '只用 3 个数，总有姿势出毛病；不想出毛病，就多用几个数、加上规矩' },
      { at: 16.77, s: '**欧拉角**最直观：只给人看（配置文件、屏幕显示）' },
      { at: 25.79, s: '**四元数**：没有万向锁、1 条规矩，存朝向、取中间值' },
      { at: 34.81, s: '**旋转矩阵**：连乘最方便，正运动学靠它（第 11 集）' },
      { at: 46.09, s: '输入欧拉角 → 四元数 → 要连乘换矩阵 → 显示时换回欧拉角' }
    ],
    build: function (K) {
      var s = K.sceneSvg('三种写法各有各的坑，没有十全十美的一种：转动只有三个自由度，只用三个数（欧拉角）总有姿势出毛病；四元数四个数减一条规矩、旋转矩阵九个数减六个等式，剩下的都是三。工程上按用途换着用：欧拉角只给人看，配置文件和屏幕显示；四元数存朝向、取中间值，仿真器和传感器大多用它；旋转矩阵连乘，正运动学靠它。人输入欧拉角，进来换成四元数，要连乘时换成矩阵，显示给人看时再换回欧拉角');

      var head = E.txt(s, 400, 26, '', 22, E.INK, 'middle', 900);
      var cards = DEFS.map(function (d, i) {
        var x = XS[i], cd = E.card(s, x, CY, CW, CH, d.name, d.c, 26);
        var n = E.txt(cd.g, x + 22, CY + 76, d.n, 21, E.INK, 'start', 900);
        var minus = E.txt(cd.g, x + 22, CY + 106, d.minus, 20, E.GOOD, 'start', 900);
        var trouble = i === 0 ? E.txt(cd.g, x + 22, CY + 106, '总有姿势出毛病', 20, E.ERR, 'start', 900) : null;
        var bad = E.tag(cd.g, x + 18, CY + 124, 140, 34, d.bad, E.ERR, { size: 19, ink: E.ERR, weight: 900, rx: 10 });
        var use = E.txt(cd.g, x + 22, CY + 196, '用：' + d.use, 21, E.GOOD, 'start', 900);
        var ex = d.ex.map(function (t, k) { return E.txt(cd.g, x + 22, CY + 226 + 26 * k, t, 18, E.INK2, 'start', 700); });
        return { g: cd.g, rect: cd.rect, n: n, minus: minus, trouble: trouble, bad: bad, use: use, ex: ex, cx: x + CW / 2 };
      });

      /* 底下：换算流程 */
      var flow = E.el('g', {}, s);
      var boxes = [['人输入', '欧拉角', E.WARN], ['内部计算', '四元数', E.ACC], ['要连乘', '旋转矩阵', E.VIOLET], ['显示给人看', '欧拉角', E.WARN]].map(function (d, i) {
        var g = E.el('g', {}, flow), x = 14 + 204 * i;
        K.paint(E.el('rect', { x: x, y: 322, width: 166, height: 88, rx: 14, 'stroke-width': 2.5 }, g), E.tint(d[2], 10, 'var(--demo-surface)'), d[2]);
        E.txt(g, x + 83, 354, d[0], 19, E.INK2, 'middle', 700);
        E.txt(g, x + 83, 392, d[1], 26, d[2], 'middle', 900);
        var ar = i < 3 ? E.arrow(flow, [[x + 170, 366], [x + 200, 366]], E.INK2, 3, { head: 11 }) : null;
        return { g: g, ar: ar, cx: x + 83 };
      });

      return {
        el: s,
        draw: function (st) {
          var f = E.fade;
          head.textContent = st < T.perfect ? '' : st < T.dof ? '没有十全十美的一种' : st < T.useHead ? '转动只有 3 个自由度' : '按用途，换着用';
          head.style.fill = st >= T.useHead ? 'var(--demo-good)' : '';
          E.op(head, Math.max(f(st, T.perfect, 0.3) * (1 - f(st, T.dof - 0.3, 0.3)), f(st, T.dof, 0.3)));
          cards.forEach(function (c, i) {
            E.pop(c.g, c.cx, CY + CH / 2, f(st, T.cards + 0.15 * i, 0.35), 0.92);
            E.pop(c.bad.g, XS[i] + 88, CY + 141, f(st, T.bad[i], 0.3), 0.8);
            var nb = bump(st, i ? T.more : T.three, 0.8);
            E.scaleAt(c.n, XS[i] + 22, CY + 70, 1 + 0.2 * nb);
            if (c.trouble) E.op(c.trouble, f(st, T.trouble, 0.3));
            else E.op(c.minus, f(st, T.rules, 0.3));
            var u = T.use[i];
            E.op(c.use, f(st, u[1], 0.3));
            c.ex.forEach(function (e, k) { E.op(e, f(st, u[2 + k], 0.3)); });
            var hl = f(st, u[0], 0.3) * (1 - f(st, i < 2 ? T.use[i + 1][0] : T.flow[0], 0.3));
            c.rect.style.strokeWidth = 2.5 + 2.5 * hl;
            E.glow(c.rect, DEFS[i].c, hl > 0.05);
          });
          boxes.forEach(function (b, i) {
            E.pop(b.g, b.cx, 366, f(st, T.flow[i], 0.3), 0.85);
            if (b.ar) { E.drawOn(b.ar.line, f(st, T.flow[i + 1] - 0.2, 0.3)); E.op(b.ar.head, f(st, T.flow[i + 1], 0.2)); }
          });
        }
      };
    }
  };
})();
