// 002 第 4 幕：难点四·强不稳定。左边一张四条腿的桌子（斜着画，看得见四只脚围成的框），右边正面人形；两边各一个重心标记。
// 「画一条竖线」时从重心往地上画虚线；「四只脚围成的大框」时桌脚下的框亮绿，「稳」时打勾；
// 「重心高」量一下高度；「这一小块」两只脚底下的窄条亮绿。
// 「稍微一歪」人绕右脚外沿转 17°（重心竖线越过右脚外沿要转 14.4°），竖线跑出窄条就变红、「要倒！」；回正后「被推一下」再歪一次；
// 「抬一下脚」左脚离地、窄条只剩右脚那一块，竖线落在外面。「一刻不停地调」人原地小幅左右晃、竖线在窄条里来回，头顶「控制」牌一下下闪；
// 「手心里立一根扫把」右边弹出手托扫把，手跟着扫把挪；「一松手」手撤走、扫把倒下；「控制一停」控制牌打叉，人绕右脚外沿倒地。
// 时刻取自配音的逐词时间（分镜时间 ≈ 念到的时刻），见 ../ep002.py。
(function () {
  var E = window.EP002, K = E.K;
  var TAU = Math.PI * 2;
  var RX = 440, EDGE = 35, IN = 18;                        // 人形中心 x；右脚（画面右）外沿 / 内沿离中心多远——建好 G1 后按它脚的实际宽度改
  var TCOM = [170, 288], TFOOT = [170, 376];                // 桌面中心与它正下方的地面点（框的中心）
  var TIP = 17, FALL0 = 34.4, FALL1 = 35.4;
  var BROOM = [680, 300];                                   // 扫把立在手心 (680, 300)

  E.scenes[3] = {
    title: '难点四：强不稳定，一松就倒',
    dur: 36.5,
    cues: [
      { at: 0, s: '难点四：**强不稳定**' },
      { at: 3.0, s: '重心的竖线落在脚围成的框里：站得住' },
      { at: 11.7, s: '人形：重心高，脚下撑住它的只有一小块' },
      { at: 18.0, s: '稍一歪、被推、抬脚：竖线就跑出去，要倒' },
      { at: 26.2, s: '站着不动，**控制**也得一刻不停地调' },
      { at: 32.5, s: '控制一停，它就倒' }
    ],
    build: function (K) {
      var s = K.sceneSvg('难点四强不稳定：从重心往地上画竖线，四条腿的桌子竖线落在四只脚围成的大框里，放着就稳；人形重心高，脚下只有两只脚那一小块，稍一歪、被推一下、抬一下脚竖线就跑出去；所以站着不动控制也得一刻不停地调，像手心里立扫把，控制一停它就倒');

      K.paint(E.el('line', { x1: 20, y1: 388, x2: 780, y2: 388, 'stroke-width': 3 }, s), null, E.BORDER);

      /* ── 左：桌子（斜视）+ 桌脚围成的框 ── */
      var table = E.el('g', {}, s);
      var quad = K.paint(E.el('path', { d: 'M 70 388 L 230 388 L 270 364 L 110 364 Z', 'stroke-width': 2.5, 'stroke-linejoin': 'round' }, table), E.tint(E.GOOD, 30, 'transparent'), E.GOOD);
      [[110, 286, 364], [270, 286, 364]].forEach(function (l) {
        var n = K.paint(E.el('line', { x1: l[0], y1: l[1], x2: l[0], y2: l[2], 'stroke-width': 8, 'stroke-linecap': 'round' }, table), null, E.INK2);
        E.op(n, 0.55);
      });
      // 前面两条腿先画、上端伸进桌面里：桌面压住腿，而不是腿的圆头压在桌面正面的下沿上
      [70, 230].forEach(function (x) { K.paint(E.el('line', { x1: x, y1: 304, x2: x, y2: 388, 'stroke-width': 8, 'stroke-linecap': 'round' }, table), null, E.INK2); });
      var top = K.paint(E.el('path', { d: 'M 58 300 L 242 300 L 282 276 L 98 276 Z', 'stroke-width': 2.5, 'stroke-linejoin': 'round' }, table), E.S2, E.INK2);
      K.paint(E.el('rect', { x: 58, y: 300, width: 184, height: 10, 'stroke-width': 2.5 }, table), E.S2, E.INK2);
      // 桌面的右侧面（斜着看得见的那一面，厚 10，和正面一样）
      K.paint(E.el('path', { d: 'M 242 300 L 282 276 L 282 286 L 242 310 Z', 'stroke-width': 2.5, 'stroke-linejoin': 'round' }, table), E.S1, E.INK2);
      var tLine = K.paint(E.el('path', { d: 'M ' + TCOM[0] + ' ' + (TCOM[1] + 12) + ' V ' + TFOOT[1], fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '6 5' }, s), null, E.WARN);
      var tDot = K.paint(E.el('circle', { cx: TFOOT[0], cy: TFOOT[1], r: 4.5 }, s), E.WARN);
      var tCom = E.comMark(s, E.WARN, 11); tCom.at(TCOM[0], TCOM[1]);
      var steady = E.el('g', {}, s);
      E.check(steady, 104, 228, 22, E.GOOD, 4);
      E.txt(steady, 134, 248, '放着就稳', 22, E.GOOD, 'start', 700);

      /* ── 右：正面人形 + 脚下窄条 + 重心竖线 ── */
      var band = K.paint(E.el('rect', { y: 384, height: 10, rx: 3, 'stroke-width': 2 }, s), E.tint(E.GOOD, 30, 'transparent'), E.GOOD);
      var bandLabel = E.txt(s, RX, 416, '只有这一小块', 18, E.GOOD, 'middle', 700);
      // 重心用 G1 各连杆的质量与质心算（E.front 的 com()），脚宽按它的脚掌外形；「重心高」的量尺量到站直时的重心
      var hum = E.front(s);
      hum.place(RX, 1);
      EDGE = hum.footSpan[1][1]; IN = hum.footSpan[1][0];
      var C0 = hum.com();
      var hBar = E.el('g', {}, s);
      s.insertBefore(hBar, hum.g);
      K.paint(E.el('path', { d: 'M 352 388 V ' + C0[1].toFixed(1) + ' M 344 ' + C0[1].toFixed(1) + ' H 360 M 344 388 H 360', fill: 'none', 'stroke-width': 2.5, 'stroke-linecap': 'round' }, hBar), null, E.WARN);
      E.txt(hBar, 340, (C0[1] + 8).toFixed(1), '重心高', 22, E.WARN, 'end', 700);   // 标在量尺顶端（重心那头），放中间会贴着桌面右角
      var hLine = K.paint(E.el('line', { 'stroke-width': 2.5, 'stroke-dasharray': '6 5' }, s), null, E.WARN);
      var hDot = K.paint(E.el('circle', { r: 4.5 }, s), E.WARN);
      var hCom = E.comMark(s, E.WARN, 11);
      var fallTag = E.tag(s, RX - 56, 40, 112, 36, '要倒！', E.BAD, { size: 22, ink: E.BAD, weight: 900 });
      var fellTag = E.tag(s, 340, 300, 92, 36, '倒了', E.BAD, { size: 22, ink: E.BAD, weight: 900 });   // 人往右倒，牌子放在支点左边，不被扫过
      var push = E.arrow(s, [[300, 160], [384, 160]], E.BAD, 5, { head: 14 });
      var pushTxt = E.txt(s, 300, 146, '推一下', 20, E.BAD, 'start', 700);
      var liftTxt = E.txt(s, 392, 336, '抬脚', 20, E.BAD, 'end', 700);

      /* 控制牌 + 两只脚边上的「调」小弧 */
      var ctrl = E.el('g', {}, s);
      var ctrlRect = K.paint(E.el('rect', { x: RX - 96, y: 30, width: 192, height: 40, rx: 12, 'stroke-width': 2.5 }, ctrl), E.tint(E.ACC, 20), E.ACC);
      var ctrlTxt = E.txt(ctrl, RX, 57, '控制 · 一刻不停地调', 20, E.ACC, 'middle', 700);
      var ctrlX = E.cross(s, RX + 112, 50, 12, E.BAD, 4);
      var adj = [-1, 1].map(function (sg) {
        return K.paint(E.el('path', { d: 'M ' + (RX + sg * 62) + ' 360 A 20 20 0 0 ' + (sg > 0 ? 0 : 1) + ' ' + (RX + sg * 52) + ' 338', fill: 'none', 'stroke-width': 3, 'stroke-linecap': 'round' }, s), null, E.ACC);
      });

      /* 手托扫把（扫把头朝上），在自己的组里绕手心转 */
      var broomG = E.el('g', {}, s);
      var palm = E.el('g', {}, broomG);
      K.paint(E.el('path', { d: 'M -4 14 L -34 62', fill: 'none', 'stroke-width': 14, 'stroke-linecap': 'round' }, palm), null, E.INK2);   // 小臂
      K.paint(E.el('path', { d: 'M -26 -2 Q 0 16 26 0', fill: 'none', 'stroke-width': 9, 'stroke-linecap': 'round' }, palm), null, E.INK2);   // 摊开的手掌
      K.paint(E.el('path', { d: 'M -26 -2 L -30 -14', fill: 'none', 'stroke-width': 7, 'stroke-linecap': 'round' }, palm), null, E.INK2);   // 大拇指
      var stick = E.el('g', {}, broomG);
      K.paint(E.el('line', { x1: 0, y1: 0, x2: 0, y2: -140, 'stroke-width': 5, 'stroke-linecap': 'round' }, stick), null, E.WARN);
      K.paint(E.el('path', { d: 'M -6 -138 L 6 -138 L 20 -176 L -20 -176 Z', 'stroke-width': 2.5, 'stroke-linejoin': 'round' }, stick), E.S2, E.WARN);

      function setBand(x0, x1, bad) {
        band.setAttribute('x', x0.toFixed(1)); band.setAttribute('width', (x1 - x0).toFixed(1));
        K.paint(band, E.tint(bad ? E.BAD : E.GOOD, 30, 'transparent'), bad ? E.BAD : E.GOOD);
      }

      return {
        el: s,
        draw: function (st, t) {
          var fade = E.fade;
          E.op(table, fade(st, 0, 0.6));
          E.op(hum.g, fade(st, 0, 0.6));

          /* 人的姿势：歪（18.95–20.4 绕右脚外沿转到 17°，21.9 回正）→ 推（22.55 再歪，23.1 回正）→ 抬左脚（23.45–24.95）
             → 26.3 起原地小晃（控制在调）→ 34.4 控制停，绕右脚外沿倒地 */
          var tipA = TIP * K.ease(K.seg(st, 18.95, 20.4)) * (1 - fade(st, 21.9, 0.4));
          var tipB = TIP * fade(st, 22.55, 0.45) * (1 - fade(st, 23.05, 0.35));
          var lift = 28 * fade(st, 23.45, 0.35) * (1 - fade(st, 24.6, 0.4));
          var sway = 2.2 * Math.sin(TAU * t / 1.4) * fade(st, 26.3, 0.6) * (1 - fade(st, FALL0, 0.2));
          // 倒地转到 86°，右臂甩过头顶：绕右脚外沿转 90° 时右臂和右肩会压进地面以下（左臂在上面，不用动）
          var u = K.seg(st, FALL0, FALL1), fall = 86 * u * u, fl = fade(st, FALL0 + 0.45, 0.45);   // 倒到一半才甩臂，免得手先戳到头顶的控制牌
          var tilt = Math.max(tipA, tipB) + fall;
          var pivot = tilt > 0.01 ? [EDGE, 388] : [0, 388];
          var armL = 8 + 10 * (tipA + tipB) / TIP;
          hum.pose({ tilt: tilt > 0.01 ? tilt : sway, pivot: pivot, lift: [lift, 0],
            arm: [[armL, 4], [8 + 162 * fl, 4 + 171 * fl]] });

          /* 重心标记与竖线：3.1 标记弹出，4.15「画一条竖线」画到地上；竖线落在窄条外就变红 */
          var c = hum.world(hum.com());
          hCom.at(c[0], c[1]);
          var pm = fade(st, 3.1, 0.3);
          E.op(tCom.g, pm);
          var pl = fade(st, 4.15, 0.5);
          E.drawOn(tLine, pl); E.op(tDot, K.seg(pl, 0.9, 1));
          var bx0 = lift > 1 ? RX + IN : RX - EDGE, bx1 = RX + EDGE;
          var out = c[0] < bx0 - 0.5 || c[0] > bx1 + 0.5;
          var lineC = out && fade(st, 15.6, 0.3) > 0 ? E.BAD : E.WARN;
          hLine.setAttribute('x1', c[0].toFixed(1)); hLine.setAttribute('x2', c[0].toFixed(1));
          hLine.setAttribute('y1', (c[1] + 12).toFixed(1)); hLine.setAttribute('y2', (c[1] + 12 + (388 - c[1] - 12) * pl).toFixed(1));
          hLine.style.stroke = lineC; hDot.style.fill = lineC;
          hDot.setAttribute('cx', c[0].toFixed(1)); hDot.setAttribute('cy', 388);
          E.op(hLine, pl > 0 ? 1 : 0); E.op(hDot, K.seg(pl, 0.9, 1));
          E.op(hCom.g, pm);

          /* 桌子：6.35「桌子」亮一下；7.6「围成的大框」框亮出来；10.5「稳」打勾 */
          top.style.filter = E.win(st, 6.3, 7.4, 0.3) > 0.5 ? 'drop-shadow(0 0 6px ' + E.INK + ')' : '';
          E.op(quad, fade(st, 7.6, 0.6));
          E.pop(steady, 170, 240, fade(st, 10.5, 0.3), 0.7);

          /* 12.6「重心高」量高度；15.3「两只脚」窄条亮出来，16.5「这一小块」 */
          E.op(hBar, fade(st, 12.6, 0.4) * (1 - fade(st, 17.8, 0.4)));
          setBand(bx0, bx1, out && st >= 15.6);
          E.op(band, fade(st, 15.3, 0.5));
          E.op(bandLabel, fade(st, 16.4, 0.3) * (1 - fade(st, 18.0, 0.3)));

          /* 21.5「要倒」；22.3「推」箭头；23.45「抬脚」 */
          E.pop(fallTag.g, RX, 58, E.win(st, 21.45, 22.2, 0.25), 0.7);
          E.pop(fellTag.g, 386, 318, fade(st, 35.15, 0.25), 0.7);
          var pp = E.win(st, 22.25, 23.2, 0.25);
          E.drawOn(push.line, fade(st, 22.25, 0.3)); E.op(push.g, pp); E.op(push.head, K.seg(fade(st, 22.25, 0.3), 0.9, 1));
          E.op(pushTxt, pp);
          E.op(liftTxt, E.win(st, 23.45, 24.8, 0.25));

          /* 26.3 起：控制牌（每 0.7 s 闪一下）+ 两脚边的小弧跟着晃的方向轮流亮；34.2「停」打叉、变灰 */
          var cp = fade(st, 27.8, 0.3), stop = fade(st, 34.2, 0.2);
          E.pop(ctrl, RX, 50, cp, 0.8);
          var pulse = 0.5 + 0.5 * Math.cos(TAU * t / 0.7);
          ctrlRect.style.filter = cp > 0 && stop < 0.5 ? 'drop-shadow(0 0 ' + (3 + 6 * pulse).toFixed(1) + 'px ' + E.ACC + ')' : '';
          K.paint(ctrlRect, stop > 0.5 ? E.S2 : E.tint(E.ACC, 20), stop > 0.5 ? E.MUT : E.ACC);
          ctrlTxt.style.fill = stop > 0.5 ? E.MUT : E.ACC;
          E.pop(ctrlX, RX + 112, 50, stop, 0.4);
          var dir = Math.cos(TAU * t / 1.4);                 // 晃的速度方向：往右晃时左边的弧（把人往回拉）亮，反之亦然
          adj.forEach(function (a, k) { E.op(a, (cp * (1 - stop) * Math.max(0, k ? -dir : dir)).toFixed(3)); });

          /* 29.9「手心里立一根扫把」：扫把小幅晃、手跟着往同一边挪；32.7「松手」手往下撤走，扫把倒下（33.6 后淡出） */
          var bp = fade(st, 29.9, 0.35) * (1 - fade(st, 33.7, 0.4));
          var wob = 7 * Math.sin(TAU * t / 1.6), drop = fade(st, 32.7, 0.4);
          var bu = K.seg(st, 32.75, 33.55), ang = wob * (1 - drop) + 85 * bu * bu;
          var hx = BROOM[0] + 10 * Math.sin(TAU * t / 1.6 + 0.6) * (1 - drop);
          palm.setAttribute('transform', 'translate(' + hx.toFixed(1) + ' ' + (BROOM[1] + 40 * drop).toFixed(1) + ')');
          E.op(palm, 1 - drop);
          stick.setAttribute('transform', 'translate(' + hx.toFixed(1) + ' ' + BROOM[1] + ') rotate(' + ang.toFixed(1) + ')');
          E.op(broomG, bp);
          broomG.setAttribute('transform', 'translate(0 ' + (6 * (1 - fade(st, 29.9, 0.35))).toFixed(1) + ')');
        }
      };
    }
  };
})();
