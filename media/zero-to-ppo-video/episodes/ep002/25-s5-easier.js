// 002 第 5 幕：学会人形，换别的都是降难度。左边人形 + 四个难点小牌（全占）；右边三张卡（机械臂 / 四足 / 轮式底盘）按念到的名字弹出，
// 每张一句「少了什么」，并把它没有的难点小牌划掉——只划旁白里说到的：机械臂「不浮也不倒」划浮动基、强不稳定；
// 四足「站着就稳」划强不稳定；轮式底盘「推一把不翻」划强不稳定、「不用换脚」划接触切换。
// 「降难度」时人形到三张卡的箭头依次画出 +「↓ 降难度」；「反过来就不行」机械臂卡回指人形的红虚线打叉，念到浮动基 / 换脚 / 摔倒时
// 人形这边对应的三块小牌依次闪红，「从头补」；最后「教学载体」人形头上「★ 最好的教学载体」。
// 时刻取自配音的逐词时间（分镜时间 ≈ 念到的时刻），见 ../ep002.py。
(function () {
  var E = window.EP002, K = E.K;
  var TAU = Math.PI * 2;
  var HUM = [100, 348, 0.68];                               // 人形：中心 x、脚底 y、缩放
  var CHIP_X = 165, CHIP_W = 140, CHIP_H = 40, CHIP_Y = [150, 200, 250, 300];
  var CARD_X = 420, CARD_W = 360, CARD_H = 116, CARD_Y = [24, 152, 280];
  var CARDS = [
    { name: '机械臂', at: 6.85, txt: '底座拧死：不浮也不倒', txtAt: 7.45, cut: [[1, 8.4], [3, 9.0]] },
    { name: '四足', at: 10.3, txt: '四条腿撑地：站着就稳', txtAt: 11.1, cut: [[3, 12.4]] },
    { name: '轮式底盘', at: 13.15, txt: '推一把不翻，轮子一直着地', txtAt: 13.95, cut: [[3, 14.45], [2, 16.75]] }
  ];
  var ARROW_AT = [21.25, 22.0, 22.75], EASY = 24.6;
  var BACK = 26.2, BLAME = { 1: 29.35, 2: 30.05, 3: 30.65 }, REDO = 33.1, STAR = 38.8;   // 念到浮动基 / 换脚 / 摔倒，对应第 1、2、3 块小牌

  function bell(st, a, b) { var u = K.seg(st, a, b); return u > 0 && u < 1 ? Math.sin(Math.PI * u) : 0; }

  E.scenes[4] = {
    title: '学会人形，换别的都是降难度',
    dur: 40.5,
    cues: [
      { at: 0, s: '四个难点：人形全占' },
      { at: 4.1, s: '机械臂、四足、轮式底盘：难点少一些' },
      { at: 18.4, s: '学会人形 → 换别的几乎都是**降难度**' },
      { at: 26.1, s: '反过来不行：没碰过的难点得从头补' },
      { at: 34.5, s: '所以从人形学起：眼下最好的教学载体' }
    ],
    build: function (K) {
      var s = K.sceneSvg('四个难点人形全占；机械臂底座拧死不浮也不倒，四足四条腿撑地站着就稳，轮式底盘推一把不翻、轮子一直着地不用换脚；所以学会人形再去控制它们几乎都是降难度，反过来只练过机械臂换到人形，浮动基、换脚和摔倒都得从头补；人形是眼下最好的教学载体');

      /* ── 左：人形 + 四个难点 ── */
      var humG = E.el('g', {}, s);
      var spot = K.paint(E.el('circle', { cx: HUM[0], cy: 250, r: 82 }, humG), 'var(--hi)');
      E.op(spot, 0);
      var hum = E.front(E.el('g', { transform: 'translate(0 ' + (HUM[1] - 388) + ')' }, humG));
      hum.place(HUM[0], HUM[2]);
      var humLabel = E.txt(humG, HUM[0], 378, '人形', 24, E.INK, 'middle', 900);
      var chips = E.DIFF.map(function (d, i) {
        var og = E.el('g', {}, s);
        var tg = E.tag(og, CHIP_X, CHIP_Y[i] - CHIP_H / 2, CHIP_W, CHIP_H, d.name, d.color, { size: 22, ink: d.color, weight: 900, rx: 10, sw: 2.5 });
        return { og: og, g: tg.g, rect: tg.rect, color: d.color, cy: CHIP_Y[i] };
      });
      var all = E.txt(s, CHIP_X + CHIP_W / 2, 348, '四个全占', 20, E.INK2, 'middle', 700);
      var redo = E.tag(s, CHIP_X + 8, 330, CHIP_W - 16, 32, '从头补', E.BAD, { size: 20, ink: E.BAD, weight: 900 });
      var star = E.tag(s, 26, 66, 264, 38, '★ 最好的教学载体', 'var(--hi)', { size: 22, ink: 'var(--hi)', weight: 900 });

      /* ── 右：三张卡 ── */
      var cards = CARDS.map(function (c, i) {
        var y = CARD_Y[i], g = E.el('g', {}, s);
        var rect = K.paint(E.el('rect', { x: CARD_X, y: y, width: CARD_W, height: CARD_H, rx: 14, 'stroke-width': 2 }, g), E.S1, E.BORDER);
        if (i === 0) E.arm(g, 452, y + 104, 0.82).set(16, 62);
        else if (i === 1) E.quadruped(g, 466, y + 96, 0.72);
        else E.wheeled(g, 468, y + 92, 0.72);
        E.txt(g, 530, y + 34, c.name, 22, E.INK, 'start', 900);
        var line = E.txt(g, 530, y + 62, c.txt, 19, E.GOOD, 'start', 700);
        var pills = c.cut.map(function (cu, k) {
          var d = E.DIFF[cu[0]], x = 530 + k * 108, pg = E.el('g', {}, g);
          E.tag(pg, x, y + 76, 100, 28, d.name, d.color, { size: 18, ink: d.color, weight: 700, rx: 8, sw: 1.5 });
          var strike = K.paint(E.el('path', { d: 'M ' + (x + 6) + ' ' + (y + 90) + ' H ' + (x + 94), fill: 'none', 'stroke-width': 3, 'stroke-linecap': 'round' }, pg), null, E.BAD);
          return { g: pg, strike: strike, at: cu[1], x: x + 50, y: y + 90 };
        });
        return { g: g, rect: rect, line: line, pills: pills, cy: y + CARD_H / 2, c: c };
      });

      /* ── 降难度箭头 + 反过来的红虚线 ── */
      var arrows = CARD_Y.map(function (y) { return E.arrow(s, [[CHIP_X + CHIP_W + 8, 225], [CARD_X - 6, y + CARD_H / 2]], E.GOOD, 3, { head: 12 }); });
      var easy = E.txt(s, 346, 342, '↓ 降难度', 20, E.GOOD, 'middle', 900);   // 放在三支箭头下方，不压线
      var back = E.curveArrow(s, 'M ' + (CARD_X - 4) + ' 46 Q 330 22 252 124', E.BAD, 3, 128, { dash: '7 6', head: 12 });
      var backX = E.cross(s, 332, 50, 12, E.BAD, 4);
      var backTxt = E.txt(s, 332, 24, '反过来不行', 20, E.BAD, 'middle', 900);

      return {
        el: s,
        draw: function (st, t) {
          var fade = E.fade;

          /* 0.3 人形；1.2「人形」起四块小牌依次弹出；2.5「全占了」；26 后「四个全占」退场 */
          E.op(humG, fade(st, 0.2, 0.4));
          var w = Math.sin(TAU * t / 3.2);
          hum.pose({ shift: 2.5 * w, arm: [[8 + 3 * w, 4 + 4 * w], [8 - 3 * w, 4 - 4 * w]] });
          chips.forEach(function (c, i) { E.pop(c.g, CHIP_X + CHIP_W / 2, c.cy, fade(st, 1.2 + 0.2 * i, 0.3), 0.7); });
          E.op(all, fade(st, 2.5, 0.3) * (1 - fade(st, 26.0, 0.3)));

          /* 三张卡：念到名字弹出，「少了什么」一句，再按念到的词划掉小牌；28.05「只练过机械臂」机械臂卡描红 */
          cards.forEach(function (cd, i) {
            E.pop(cd.g, CARD_X + CARD_W / 2, cd.cy, fade(st, cd.c.at, 0.35), 0.9);
            E.op(cd.line, fade(st, cd.c.txtAt, 0.3));
            cd.pills.forEach(function (p) {
              E.op(p.g, fade(st, p.at - 0.25, 0.25));
              E.drawOn(p.strike, fade(st, p.at, 0.3));
            });
          });
          var armHi = E.win(st, 28.0, 33.6, 0.3);
          K.paint(cards[0].rect, null, armHi > 0.5 ? E.BAD : E.BORDER);
          cards[0].rect.style.strokeWidth = armHi > 0.5 ? 3 : 2;

          /* 21.25 起三支箭头依次画出；24.6「降难度」 */
          arrows.forEach(function (a, i) {
            var p = fade(st, ARROW_AT[i], 0.45);
            E.drawOn(a.line, p); E.op(a.head, K.seg(p, 0.9, 1));
            E.op(a.g, (1 - 0.6 * fade(st, BACK, 0.4)).toFixed(3));
          });
          E.pop(easy, 346, 335, fade(st, EASY, 0.3), 0.7);
          E.op(easy, fade(st, EASY, 0.3) * (1 - 0.6 * fade(st, BACK, 0.4)));

          /* 26.2「反过来」红虚线；26.85「不行」打叉；念到浮动基 / 换脚 / 摔倒时对应小牌闪红、抖一下；33.1「从头补」 */
          var pb = fade(st, BACK, 0.6), gone = 1 - fade(st, 34.4, 0.5);   // 34.4 最后一句开始时红虚线退场，给「★」让位
          E.drawOn(back.line, pb); E.op(back.head, K.seg(pb, 0.9, 1));
          E.op(back.g, gone);
          E.pop(backX, 332, 50, fade(st, 26.85, 0.25) * gone, 0.4);
          E.op(backTxt, fade(st, 26.85, 0.3) * gone);
          chips.forEach(function (c, i) {
            var bt = BLAME[i], hit = bt ? fade(st, bt, 0.2) * (1 - fade(st, 35.8, 0.4)) : 0;
            var dx = bt ? 5 * Math.sin(6 * Math.PI * K.seg(st, bt, bt + 0.4)) * bell(st, bt, bt + 0.4) : 0;
            K.paint(c.rect, null, hit > 0.5 ? E.BAD : c.color);
            c.rect.style.strokeWidth = hit > 0.5 ? 3.5 : 2.5;
            c.og.setAttribute('transform', 'translate(' + dx.toFixed(2) + ' 0)');
          });
          E.pop(redo.g, CHIP_X + CHIP_W / 2, 346, E.win(st, REDO, 35.8, 0.3), 0.8);

          /* 36.0「从人形学起」人形垫一圈亮黄；38.8「教学载体」★ */
          E.op(spot, (0.12 * fade(st, 36.0, 0.5)).toFixed(3));
          humLabel.style.fill = st >= 36.0 ? 'var(--hi)' : E.INK;
          E.pop(star.g, 158, 85, fade(st, STAR, 0.3), 0.7);
        }
      };
    }
  };
})();
