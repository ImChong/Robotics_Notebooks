// 片头主图（800×420）：三句旁白各一段——
// ① 回顾 001：四步盒子一开头就淡淡摆着（不让主图空等两秒多），念到哪一步哪一盒亮起 → 反馈虚线连成圈 → 「只钻控制」时其余三盒变暗、控制盒亮边 + 「★ 本系列只钻控制」；
// ② 8.1 起四步圈淡出，机械臂 / 轮式底盘 / 四足 / 人形按念到的名字弹出；「为什么偏偏」时其余三台变暗，人形放大、垫聚光圆、头上一个「?」；
// ③ 16.9 起「?」和其余三台淡出，人形挪到正中，四个难点小牌在「一样不落」时依次弹出在它两侧（虚线连到人形），「摆出来」时依次鼓一下。
//    小牌不排在四台机器人底下：排成一行会让人以为「关节多」说的是机械臂。
// 时刻取自配音的逐词时间（../ep002.py 的 [from, to] 等于每句配音实长，分镜时间 ≈ 念到的时刻）。
(function () {
  var E = window.EP002, K = E.K;
  var TAU = Math.PI * 2;
  var BOX_X = [40, 230, 420, 610], BOX_W = 150, BOX_Y = 150, BOX_H = 70, MID_Y = 185;
  var STEP_AT = [2.6, 3.2, 3.8, 4.35];                       // 感知 2.68、规划 3.25、控制 3.84、执行 4.41
  var LOOP_OUT = 8.1;
  var BOTS = [
    { x: 105, at: 8.35, name: '机械臂', c: [130, 196] },
    { x: 300, at: 9.2, name: '轮式底盘', c: [300, 240] },
    { x: 480, at: 10.3, name: '四足', c: [484, 208] },
    { x: 660, at: 11.1, name: '人形', c: [660, 180] }
  ];
  var FEET_Y = 272, DIM = 13.05, ASK = 13.5;
  var MOVE = 16.9, HX = 400;                                 // 人形挪到 x = 400
  var CHIP = [[180, 140], [180, 236], [620, 140], [620, 236]], CHIP_W = 176, CHIP_H = 52, CHIP_AT = 19.05, SHOW = 23.5;

  function bump(st, a, d) { var u = K.seg(st, a, a + d); return u > 0 && u < 1 ? Math.sin(Math.PI * u) : 0; }

  E.hero = function (K) {
    var s = K.sceneSvg('片头主图：上一集的四步圈只钻控制；机械臂、轮式底盘、四足、人形里偏偏挑人形，因为它把控制里最难的四件事全占了');

    /* ── ① 四步圈 ── */
    var loop = E.el('g', {}, s);
    var links = [0, 1, 2].map(function (i) {
      return E.arrow(loop, [[BOX_X[i] + BOX_W + 4, MID_Y], [BOX_X[i + 1] - 4, MID_Y]], E.INK2, 3, { head: 10 });
    });
    var fb = E.arrow(loop, [[685, 220], [685, 256], [115, 256], [115, 226]], E.INK2, 2.5, { dash: '8 6', head: 10 });
    var boxes = E.STEPS.map(function (stp, i) {
      var g = E.el('g', {}, loop), x = BOX_X[i];
      var rect = K.paint(E.el('rect', { x: x, y: BOX_Y, width: BOX_W, height: BOX_H, rx: 14, 'stroke-width': 3 }, g), E.S2, stp.color);
      E.txt(g, x + BOX_W / 2, MID_Y + 10, stp.name, 28, stp.color, 'middle', 900);
      return { g: g, rect: rect, cx: x + BOX_W / 2 };
    });
    var only = E.tag(loop, 390, 96, 210, 34, '★ 本系列只钻控制', E.ACC, { size: 20, ink: E.ACC, weight: 700 });

    /* ── ② 四台机器人 ── */
    var spot = K.paint(E.el('circle', { cx: 660, cy: 182, r: 104 }, s), E.ACC);
    E.op(spot, 0);
    var bots = BOTS.map(function (b, i) {
      var g = E.el('g', {}, s), inner = E.el('g', {}, g);
      // 人形用正面机器人（火柴人缩到这么小只剩一根杆）：E.front 的脚底在 y 388，外面再套一层挪到 FEET_Y
      var r = i === 0 ? E.arm(inner, b.x, FEET_Y, 1.2) : i === 1 ? E.wheeled(inner, b.x, FEET_Y, 1.4)
        : i === 2 ? E.quadruped(inner, b.x, FEET_Y, 1.45) : E.front(E.el('g', { transform: 'translate(0 ' + (FEET_Y - 388) + ')' }, inner));
      if (i === 3) r.place(b.x, 0.6);
      var label = E.txt(g, b.x, 308, b.name, 24, E.INK2, 'middle', 700);
      return { g: g, inner: inner, r: r, label: label, b: b };
    });
    var hum = bots[3];
    var ask = E.txt(s, 728, 132, '?', 56, 'var(--hi)', 'middle', 900);

    /* ── ③ 四个难点：左右各两块，虚线连到人形身上（左边连到肩 / 髋的高度，右边同理） ── */
    var chips = E.DIFF.map(function (d, i) {
      var c = CHIP[i], left = c[0] < HX, x0 = c[0] + (left ? CHIP_W / 2 + 6 : -CHIP_W / 2 - 6), x1 = HX + (left ? -52 : 52);
      var link = E.el('path', { d: 'M ' + x0 + ' ' + c[1] + ' H ' + x1, fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '6 5', 'stroke-linecap': 'round' }, s);
      K.paint(link, null, d.color);
      // 外层 g 管「鼓一下」，里层（tag 自己的 g）管弹出
      var og = E.el('g', {}, s);
      var tg = E.tag(og, c[0] - CHIP_W / 2, c[1] - CHIP_H / 2, CHIP_W, CHIP_H, d.name, d.color, { size: 26, ink: d.color, weight: 900, rx: 12, sw: 2.5 });
      return { og: og, g: tg.g, link: link, c: c };
    });

    return {
      el: s,
      draw: function (t, still, rt) {
        if (rt === undefined) rt = t;
        var fade = E.fade;

        /* ① 四步圈：按念到的词弹出；4.9「这个圈」反馈虚线画出；6.3「只钻控制」其余三盒变暗；8.1 起整组淡出 */
        var focus = fade(t, 6.3, 0.5), out = 1 - fade(t, LOOP_OUT, 0.4);
        var p0 = fade(t, 0.15, 0.5);
        boxes.forEach(function (b, i) {
          var p = fade(t, STEP_AT[i], 0.3), o = p0 * (0.3 + 0.7 * p);
          E.scaleAt(b.g, b.cx, MID_Y, 0.94 + 0.06 * p + 0.06 * Math.sin(Math.PI * p));
          if (i) E.op(links[i - 1].g, (o * (1 - 0.7 * focus)).toFixed(3));
          E.op(b.g, (i === 2 ? o : o * (1 - 0.7 * focus)).toFixed(3));
        });
        var br = 0.5 + 0.5 * Math.sin(TAU * rt / 1.2);
        boxes[2].rect.style.strokeWidth = (3 + 1.5 * focus * br).toFixed(2);
        boxes[2].rect.style.filter = focus > 0 ? 'drop-shadow(0 0 ' + (focus * (3 + 7 * br)).toFixed(1) + 'px ' + E.ACC + ')' : '';
        K.paint(boxes[2].rect, focus > 0 ? E.tint(E.ACC, Math.round(22 * focus)) : E.S2);
        var pf = fade(t, 4.85, 0.5);
        E.drawOn(fb.line, pf);
        E.op(fb.head, K.seg(pf, 0.9, 1));
        fb.g.style.opacity = (1 - 0.7 * focus).toFixed(3);
        E.pop(only.g, 495, 113, fade(t, 6.6, 0.3), 0.8);
        E.op(loop, out);

        /* ② 四台机器人按念到的名字弹出；13.05「为什么偏偏」其余三台变暗，人形放大 1.1、垫聚光圆；13.5「?」 */
        var dim = fade(t, DIM, 0.5), mv = fade(t, MOVE, 0.7), dx = (HX - 660) * mv;
        bots.forEach(function (o, i) {
          var p = fade(t, o.b.at, 0.3);
          E.pop(o.inner, o.b.c[0], o.b.c[1], p);
          E.op(o.label, p);
          if (i < 3) E.op(o.g, ((1 - 0.65 * dim) * (1 - fade(t, MOVE, 0.4))).toFixed(3));
        });
        hum.g.setAttribute('transform', 'translate(' + dx.toFixed(2) + ' 0) translate(660 ' + FEET_Y + ') scale(' + (1 + 0.1 * dim).toFixed(4) + ') translate(-660 ' + (-FEET_Y) + ')');
        hum.label.style.fill = dim > 0.5 ? E.INK : E.INK2;
        spot.setAttribute('cx', (660 + dx).toFixed(2));
        E.op(spot, (0.1 * dim).toFixed(3));
        var w = Math.sin(TAU * rt / 3.2);                    // 人形站着轻晃：上身横移、两臂小幅摆
        hum.r.pose({ shift: 3 * w, arm: [[8 + 3 * w, 4 + 4 * w], [8 - 3 * w, 4 - 4 * w]] });
        bots[0].r.set(10 + 6 * Math.sin(TAU * rt / 2.6), 55);
        E.pop(ask, 728, 112, fade(t, ASK, 0.3) * (1 - fade(t, 16.9, 0.4)), 0.6);

        /* ③ 19.05「人形一样不落」四个难点依次弹出、虚线连到人形；23.5「摆出来」依次鼓一下 */
        chips.forEach(function (c, i) {
          var p = fade(t, CHIP_AT + 0.25 * i, 0.3);
          E.pop(c.g, c.c[0], c.c[1], p, 0.7);
          E.scaleAt(c.og, c.c[0], c.c[1], 1 + 0.1 * bump(t, SHOW + 0.12 * i, 0.4));
          E.drawOn(c.link, fade(t, CHIP_AT + 0.25 * i + 0.15, 0.3));
        });
      }
    };
  };
})();
