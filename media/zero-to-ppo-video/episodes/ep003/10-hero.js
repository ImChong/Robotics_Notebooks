// 片头主图（800×420）：三句旁白各一段——
// ① 回顾 002「人形难控」：正面 G1 站在中间，「用数说清楚」之后，念到哪一样就在它身上画出哪一样并弹出问句小牌：
//    手在哪（抬起的手上一点）、身子朝哪（绕躯干一圈弯箭头）、动多快（骨盆旁一支速度箭头）、受多大的力（画面左脚底一支朝上的推力）；
// ② 10.2 起小牌淡出，G1 退到左边，右边是这一集的问题：「数学打底」→「线性代数」→「在机器人里到底怎么用？」；
// ③ 16.66 起问题淡出，「位置是向量」时从两脚中间（原点）到手画一支蓝箭头、右边弹出列向量卡，「变换是矩阵」时弹出矩阵卡。
// 时刻取自配音的逐词时间（../ep003.py 的 [from, to] 等于每句配音实长，分镜时间 ≈ 念到的时刻）。
(function () {
  var E = window.EP003, K = E.K;
  var TAU = Math.PI * 2;
  var HX = 400, GY = 404, S0 = 1.12;                  // 人形：两脚中间的地面点、缩放
  var LX = 150, S1 = 0.95, MOVE = 10.25;              // ② 起退到左边
  var ARM = [[58, 96], [10, 6]];                      // 画面左手抬起（外张、小臂上弯），方便指「手在哪」
  // 四个问句：念到的时刻、小牌中心、颜色
  var ASK = [
    { s: '手在哪？', at: 5.84, c: [128, 118], color: E.ACC },
    { s: '身子朝哪？', at: 6.58, c: [672, 118], color: E.VIOLET },
    { s: '动多快？', at: 7.59, c: [672, 236], color: E.GOOD },
    { s: '受多大的力？', at: 8.33, c: [128, 316], color: E.WARN }
  ];
  var Q_AT = { tag: 11.09, big: 13.15, sub: 14.07, ask: 15.38 }, ANS = 16.66, VEC = 18.15, MAT = 19.38;

  E.hero = function (K) {
    var s = K.sceneSvg('片头主图：要控制人形机器人，先要用数说清楚手在哪、身子朝哪、动多快、受多大的力；线性代数在机器人里怎么用？位置是向量，变换是矩阵');

    /* ── 人形（外面一层 g 管退到左边） ── */
    var humG = E.el('g', {}, s);
    var ground = K.paint(E.el('line', { x1: -120, y1: 388, x2: 120, y2: 388, 'stroke-width': 3, 'stroke-linecap': 'round' }, humG), null, E.BORDER);
    var hum = E.front(humG);
    hum.place(0, 1);
    var prev = E.tag(s, 300, 14, 200, 36, '上一集：人形难控', E.INK2, { size: 20, ink: E.INK2, weight: 700, fill: E.S2 });

    /* ── ① 四样东西：画在人形身上的标记（跟着人形走，所以放在 humG 里）+ 问句小牌（固定在画面上） ── */
    var handDot = K.paint(E.el('circle', { r: 9, 'stroke-width': 3 }, humG), E.tint(E.ACC, 30), E.ACC);
    // 绕躯干的椭圆弧（中心 (0, 190)）：从左边经过身前（下沿）转到右上，逆时针约 200°
    var turnArc = E.curveArrow(humG, 'M -72.9 193.8 A 74 22 0 1 0 64.1 179', E.VIOLET, 4, -153, { head: 14 });
    var velo = E.vec(humG, E.GOOD, 5, { head: 16 });
    var force = E.vec(humG, E.WARN, 5, { head: 16 });
    var asks = ASK.map(function (a) {
      var link = E.el('path', { fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '6 5', 'stroke-linecap': 'round' }, s);
      K.paint(link, null, a.color);
      var w = a.s.length * 26 + 30;
      var tg = E.tag(s, a.c[0] - w / 2, a.c[1] - 22, w, 44, a.s, a.color, { size: 24, ink: a.color, weight: 900, rx: 12, sw: 2.5 });
      return { a: a, link: link, g: tg.g, w: w };
    });

    /* ── ② 这一集的问题 ── */
    var q = E.el('g', {}, s);
    var qTag = E.tag(q, 460, 70, 160, 38, '数学打底', E.HI, { size: 22, ink: E.HI, weight: 900 });
    var qBig = E.txt(q, 540, 200, '线性代数', 68, E.INK, 'middle', 900);
    var qSub = E.txt(q, 540, 262, '在机器人里到底怎么用？', 30, E.INK2, 'middle', 700);
    var qAsk = E.txt(q, 718, 196, '?', 72, E.HI, 'middle', 900);

    /* ── ③ 答案：两张卡 ── */
    function card(x, title, color, tex) {
      var g = E.el('g', {}, s);
      K.paint(E.el('rect', { x: x, y: 96, width: 222, height: 236, rx: 18, 'stroke-width': 2.5 }, g), E.tint(color, 10, 'var(--demo-surface)'), color);
      E.txt(g, x + 111, 140, title, 30, color, 'middle', 900);
      E.tex(g, x + 111, 240, tex, 50, E.INK, 'middle', { w: 210, h: 140 });
      return g;
    }
    var vecCard = card(300, '位置是向量', E.ACC, E.col([3, 2], E.HEX.acc));
    var matCard = card(546, '变换是矩阵', E.VIOLET, E.mat2([0, -1, 1, 0], [E.HEX.good, E.HEX.warn]));
    var posVec = E.vec(humG, E.ACC, 5, { head: 16 });
    var origin = K.paint(E.el('circle', { cx: 0, cy: 388, r: 6 }, humG), E.INK);

    return {
      el: s,
      draw: function (t, still, rt) {
        if (rt === undefined) rt = t;
        var fade = E.fade;

        /* 人形：站着轻晃；② 起退到左边、缩小 */
        var mv = fade(t, MOVE, 0.8), x = HX + (LX - HX) * mv, sc = S0 + (S1 - S0) * mv;
        humG.setAttribute('transform', 'translate(' + x.toFixed(2) + ' ' + (GY - 388 * sc).toFixed(2) + ') scale(' + sc.toFixed(4) + ')');
        E.op(humG, fade(t, 0, 0.5));
        var w = Math.sin(TAU * rt / 3.2);
        hum.pose({ shift: 2.5 * w, arm: [[ARM[0][0] + 3 * w, ARM[0][1] + 3 * w], [ARM[1][0] - 2 * w, ARM[1][1] - 2 * w]] });
        E.pop(prev.g, 400, 32, fade(t, 1.0, 0.35) * (1 - fade(t, 4.3, 0.4)), 0.85);

        /* 人形身上各部位（本地坐标）：手 = 画面左手（机器人右手），骨盆，画面左脚 */
        var hand = hum.world(hum.view.at('right_rubber_hand')), pel = hum.world(hum.view.at('pelvis'));
        var foot = hum.world(hum.view.at('right_ankle_roll_link'));
        handDot.setAttribute('cx', hand[0].toFixed(1)); handDot.setAttribute('cy', hand[1].toFixed(1));

        /* ① 5.84 起念到哪一样画哪一样、弹哪块小牌；10.2 起淡出 */
        var out = 1 - fade(t, MOVE - 0.05, 0.4);
        var pa = ASK.map(function (a) { return fade(t, a.at, 0.3) * out; });
        E.op(handDot, pa[0]);
        handDot.setAttribute('r', (9 + 2 * Math.sin(TAU * rt / 1.2)).toFixed(2));
        E.op(turnArc.g, pa[1]);
        velo.set(pel[0] + 66, pel[1] + 4, pel[0] + 66 + 78 * fade(t, ASK[2].at, 0.5), pel[1] + 4); E.op(velo.g, pa[2]);
        force.set(foot[0] - 30, 388, foot[0] - 30, 388 - 92 * fade(t, ASK[3].at, 0.5)); E.op(force.g, pa[3]);   // 画在脚外侧，不和小腿叠在一起
        // 小牌连到人形身上（画面坐标：人形这时还在正中，按 humG 的变换换算）
        var targets = [hand, [76, 184], [pel[0] + 144, pel[1] + 4], [foot[0] - 30, 330]];
        asks.forEach(function (o, i) {
          var p = targets[i], X = x + p[0] * sc, Y = GY - 388 * sc + p[1] * sc;
          var left = o.a.c[0] < 400, x0 = o.a.c[0] + (left ? o.w / 2 + 4 : -o.w / 2 - 4);
          o.link.setAttribute('d', 'M ' + x0 + ' ' + o.a.c[1] + ' L ' + (X + (left ? -14 : 14)).toFixed(1) + ' ' + Y.toFixed(1));
          E.pop(o.g, o.a.c[0], o.a.c[1], pa[i], 0.7);
          E.op(o.link, pa[i]);
        });

        /* ② 11.09「数学打底」、13.15「线性代数」、14.07「在机器人里…」、15.38「怎么用」「?」；16.66 起淡出 */
        var qo = 1 - fade(t, ANS, 0.4);
        E.pop(qTag.g, 540, 89, fade(t, Q_AT.tag, 0.3) * qo, 0.8);
        E.pop(qBig, 540, 180, fade(t, Q_AT.big, 0.35) * qo, 0.85);
        E.op(qSub, fade(t, Q_AT.sub, 0.4) * qo);
        E.pop(qAsk, 718, 172, fade(t, Q_AT.ask, 0.3) * qo, 0.5);

        /* ③ 18.15「位置是向量」：原点到手一支蓝箭头 + 向量卡；19.38「变换是矩阵」：矩阵卡 */
        var pv = fade(t, VEC, 0.5);
        E.op(origin, pv);
        posVec.set(0, 388, hand[0] * pv, 388 + (hand[1] - 388) * pv); E.op(posVec.g, pv > 0 ? 1 : 0);
        E.pop(vecCard, 411, 214, fade(t, VEC, 0.35), 0.85);
        E.pop(matCard, 657, 214, fade(t, MAT, 0.35), 0.85);
      }
    };
  };
})();
