// 001 第 8 幕：本系列只钻「控制」这一盒。由 svg/loop4.svg 改编（缩放到 800 宽）：四个大盒子 + 盒间箭头 + 反馈虚线；
// 3.0 其余三盒和箭头退到 0.3，控制盒放大 1.1 倍、描边呼吸发光，后面垫聚光圆，上方「★ 本系列主战场」→
// 底部一行「要去哪儿」→「每个关节这一刻怎么出力」→ 三个暗盒轻闪一下 → 暗盒上方「用到才讲」。
(function () {
  var E = window.EP001, K = E.K;
  var BOX_X = [25, 220, 415, 610], BOX_W = 170, BOX_Y = 110, BOX_H = 150, MID_Y = 185;
  var LINES = [
    ['我在哪、世界啥样', '≈ 眼耳和身体感觉'],
    ['去哪、走哪条路', '≈ 心里想好'],
    ['每个关节出多少力', '≈ 给肌肉派活'],
    ['把命令变成动作', '≈ 肌肉骨骼动']
  ];
  var CTRL = 2, CX = 500;                 // 控制盒及其中心 x（放大中心 (500, 185)）
  var FOCUS = 3.0, FOCUS_D = 0.6, DIM = 0.3, ZOOM = 0.1;
  var GROW = BOX_W * ZOOM / 2;            // 控制盒放大后左右各外扩 8.5
  var TAU = Math.PI * 2;

  // a 起 d 秒内鼓一下：0 → 1 → 0
  function bump(st, a, d) { var u = K.seg(st, a, a + d); return u > 0 && u < 1 ? Math.sin(Math.PI * u) : 0; }

  // 盒间水平箭头（线 + 三角头）；set(x0, x1) 改两端：控制盒放大后，箭头缩回去贴着它，免得箭头尖被盖住
  function hArrow(parent, y, color, w, head) {
    var g = E.el('g', {}, parent);
    var line = K.paint(E.el('path', { fill: 'none', 'stroke-width': w, 'stroke-linecap': 'round' }, g), null, color);
    var tip = K.paint(E.el('path', {}, g), color);
    function set(x0, x1) {
      line.setAttribute('d', 'M ' + x0.toFixed(1) + ' ' + y + ' H ' + (x1 - head * 0.5).toFixed(1));
      tip.setAttribute('d', 'M ' + x1.toFixed(1) + ' ' + y + ' L ' + (x1 - head).toFixed(1) + ' ' + (y - head * 0.6).toFixed(1) +
        ' L ' + (x1 - head).toFixed(1) + ' ' + (y + head * 0.6).toFixed(1) + ' Z');
    }
    return { g: g, set: set };
  }

  E.scenes[7] = {
    title: '本系列只钻「控制」这一盒',
    dur: 15.9,
    cues: [
      { at: 0, s: '四个盒子，每个都是一门大学问' },
      { at: 3, s: '本系列只钻第三盒：**控制**' },
      { at: 5.6, s: '控制：「要去哪儿」→「每个关节怎么出力」' },
      { at: 11.3, s: '另外三盒：只讲控制离不开的那部分' }
    ],
    build: function (K) {
      var s = K.sceneSvg('感知、规划、控制、执行四个盒子，每个都是一门大学问；本系列只钻控制这一盒：把要去哪儿翻译成每个关节这一刻怎么出力。另外三盒只讲控制离不开的那部分');

      /* ── 聚光圆（最底层） ── */
      var spot = K.paint(E.el('circle', { cx: CX, cy: MID_Y, r: 130 }, s), E.ACC);
      E.op(spot, 0);

      /* ── 盒间箭头 + 反馈虚线（压在盒子下面：控制盒放大时盖住箭头根部） ── */
      var links = [0, 1, 2].map(function (i) {
        var a = hArrow(s, MID_Y, E.INK2, 3, 9);
        a.x0 = BOX_X[i] + BOX_W + 3; a.x1 = BOX_X[i + 1] - 3;
        a.set(a.x0, a.x1);
        return a;
      });
      var fbG = E.el('g', {}, s);
      var fb = E.arrow(fbG, [[695, 260], [695, 300], [110, 300], [110, 266]], E.INK2, 2.5, { dash: '8 6', head: 10 });
      var fbWord = E.txt(fbG, 402, 322, '反馈', 20, E.INK2);

      /* ── 四个大盒子（控制盒最后画，放大后在最上层） ── */
      var boxes = E.STEPS.map(function (stp, i) {
        var g = E.el('g', {}, s), x = BOX_X[i], cx = x + BOX_W / 2;
        var rect = K.paint(E.el('rect', { x: x, y: BOX_Y, width: BOX_W, height: BOX_H, rx: 16, 'stroke-width': 3 }, g), E.S2, stp.color);
        E.txt(g, cx, 155, stp.name, 28, stp.color, 'middle', 700);
        E.txt(g, cx, 190, LINES[i][0], 18, E.INK);
        E.txt(g, cx, 218, LINES[i][1], 18, E.INK2);
        return { g: g, rect: rect, cx: cx };
      });
      s.appendChild(boxes[CTRL].g);
      var ctrlRect = boxes[CTRL].rect;

      /* ── 文字：主战场 · 底部一行（两段同一个 text，整行居中在 400） · 用到才讲 ── */
      var star = E.txt(s, CX, 92, '★ 本系列主战场', 22, E.ACC, 'middle', 700);
      var bottom = E.txt(s, 400, 366, '', 22, E.VIOLET);   // 规格 20px；这是全幕的结论句，按约定「重点字 22–30」放到 22
      var goal = E.el('tspan', {}, bottom);
      goal.textContent = '「要去哪儿」';
      var how = E.el('tspan', {}, bottom);
      how.textContent = '→「每个关节这一刻怎么出力」';
      how.style.fill = E.ACC;
      var tags = [0, 1, 3].map(function (i) {
        var cx = boxes[i].cx;
        var tg = E.tag(s, cx - 52, 80, 104, 24, '用到才讲', E.MUT, { size: 18, ink: E.MUT, sw: 1.5 });   // 宽 104：字宽 72，两端各留 16
        return { g: tg.g, cx: cx };
      });

      return {
        el: s,
        draw: function (st, t) {
          var fade = E.fade;
          var f = fade(st, FOCUS, FOCUS_D);
          // 暗盒透明度：3.0 起 1 → 0.3；11.3 轻闪一下 0.3 → 0.5 → 0.3（0.8 s）
          var dimTo = DIM + 0.2 * bump(st, 11.3, 0.8);
          var dim = 1 - (1 - dimTo) * f;

          /* 0–0.6 四个盒子依次弹出（间隔 0.15 s，每个 0.15 s）；控制盒再按 f 放大到 1.1 */
          boxes.forEach(function (b, i) {
            var p = fade(st, 0.15 * i, 0.15);
            var sc = (0.8 + 0.2 * p) * (i === CTRL ? 1 + ZOOM * f : 1);
            E.scaleAt(b.g, b.cx, MID_Y, sc);
            E.op(b.g, (p * (i === CTRL ? 1 : dim)).toFixed(3));
          });

          /* 控制盒描边按 t 呼吸发光（周期 1.2 s） */
          var br = 0.5 + 0.5 * Math.sin(TAU * t / 1.2);
          ctrlRect.style.strokeWidth = (3 + 1.2 * f * br).toFixed(2);
          ctrlRect.style.filter = f > 0 ? 'drop-shadow(0 0 ' + (f * (3 + 7 * br)).toFixed(1) + 'px ' + E.ACC + ')' : '';
          E.op(spot, (0.08 * f).toFixed(4));

          /* 0.6 起盒间箭头依次出现；控制盒放大时，进控制盒的箭头尖、出控制盒的箭头根各缩回 8.5 */
          links.forEach(function (a, i) {
            var x0 = a.x0 + (i === CTRL ? GROW * f : 0), x1 = a.x1 - (i === CTRL - 1 ? GROW * f : 0);
            a.set(x0, x1);
            E.op(a.g, (fade(st, 0.6 + 0.1 * i, 0.2) * dim).toFixed(3));
          });

          /* 0.8–1.4 反馈虚线画出，箭头头在末尾出现；1.2「反馈」 */
          var pf = fade(st, 0.8, 0.6);
          E.drawOn(fb.line, pf);
          E.op(fb.head, K.seg(pf, 0.9, 1));
          E.op(fbWord, fade(st, 1.2, 0.3));
          E.op(fbG, dim.toFixed(3));

          /* 3.3「★ 本系列主战场」 */
          E.pop(star, CX, 85, fade(st, 3.3, 0.3), 0.8);

          /* 5.6「要去哪儿」；7.3 接上「→ 每个关节这一刻怎么出力」 */
          E.pop(bottom, 400, 359, fade(st, 5.6, 0.3), 0.9);
          how.style.fillOpacity = fade(st, 7.3, 0.4).toFixed(3);

          /* 14.9 三个暗盒上方「用到才讲」 */
          tags.forEach(function (tg, k) { E.pop(tg.g, tg.cx, 92, fade(st, 14.9 + 0.08 * k, 0.3), 0.8); });
        }
      };
    }
  };
})();
