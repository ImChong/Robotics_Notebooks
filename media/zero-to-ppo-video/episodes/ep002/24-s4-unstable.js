// 002 第 4 幕：难点四·强不稳定。整幕一个斜视的三维视角：地面网格、左边四条腿的桌子、右边人形（URDF 用同一个相机渲染），
// 桌子、地面上的框、重心竖线都是三维点投下来的；两边各一个重心标记。
// 「画一条竖线」时从重心往地上画虚线；「四只脚围成的大框」时桌脚下的框亮绿，「稳」时打勾；
// 「重心高」量一下高度；「这一小块」两只脚掌围成的地面框亮绿。
// 「稍微一歪」人绕画面右脚的外沿（三维的滚转）转 17°，竖线跑出脚下那一块就变红、「要倒！」；回正后「被推一下」再歪一次；
// 「抬一下脚」画面左脚离地、脚下只剩画面右脚那一块，竖线落在外面。「一刻不停地调」人原地小幅左右晃、竖线在那一块里来回，头顶「控制」牌一下下闪；
// 「手心里立一根扫把」右边弹出手托扫把，手跟着扫把挪；「一松手」手撤走、扫把倒下；「控制一停」控制牌打叉，人绕右脚外沿倒地。
// 时刻取自配音的逐词时间（分镜时间 ≈ 念到的时刻），见 ../ep002.py。
(function () {
  var E = window.EP002, K = E.K;
  var TAU = Math.PI * 2;
  // 人形两脚中间的地面点放在画面里的 (RX, GY)、整个三维场景缩到 HS：人往自己左边倒，在这个视角里是朝镜头倒下来，
  // 地面比别的幕提高、人小一点，倒地后才不出画框。EDGE：支点（画面右脚外沿）离两脚中间多远，建好 G1 后按脚掌算
  var RX = 440, GY = 356, HS = 0.9, EDGE = 35;
  // 斜视相机（人形、桌子、地面网格同一个）：从机器人左前上方看，桌子看得见顶面、正面和右侧面
  var AZ = 120, EL = 20;
  // 桌子在机器人坐标系里（米；机器人朝 +x，左手 +y）：中心、宽（沿 y）、深（沿 x）、高、桌面厚、桌腿离桌角
  var TAB = { x: 0.47, y: -1.12, w: 0.82, d: 0.45, h: 0.39, t: 0.045, inset: 0.04 };
  var GRID = 0.25, FLOOR = [24, 278, 776, 414];             // 地面网格间距（米）；网格只画在画框里这一块地面上
  var TIP = 17, FALL0 = 34.4, FALL1 = 35.4;
  var BROOM = [646, 300], HAND = 1.25;                      // 扫把立在手心 (646, 300)，手放大 1.25 倍；手和扫把画在右边一张小卡片里

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

      /* ── 人形（斜视）先建：桌子、地面、各种标注都用它的相机投影。P：机器人坐标系里的三维点（米）→ 分镜坐标 ── */
      var hum = E.front(s, { az: AZ, el: EL, box: [-170, 0, 560, 432] });
      hum.place(RX, HS, GY);
      var Z0 = hum.ground, P = hum.proj;
      function poly(list) {
        return 'M ' + list.map(function (p) { var q = P(p); return q[0].toFixed(1) + ' ' + q[1].toFixed(1); }).join(' L ') + ' Z';
      }
      function under(n) { s.insertBefore(n, hum.g); return n; }

      /* ── 地面网格：贴地的线投下来，裁到 FLOOR 这一块（不用 clipPath：fitStory 量取景范围时会把裁掉的部分也算进去） ── */
      var grid = under(E.el('g', {}, s));
      function gridLine(p0, p1) {
        var a = P(p0), b = P(p1), t0 = 0, t1 = 1, dx = b[0] - a[0], dy = b[1] - a[1];
        var tests = [[-dx, a[0] - FLOOR[0]], [dx, FLOOR[2] - a[0]], [-dy, a[1] - FLOOR[1]], [dy, FLOOR[3] - a[1]]];
        for (var k = 0; k < 4; k++) {
          var pp = tests[k][0], q = tests[k][1];
          if (pp === 0) { if (q < 0) return; continue; }
          var r = q / pp;
          if (pp < 0) t0 = Math.max(t0, r); else t1 = Math.min(t1, r);
        }
        if (t1 <= t0) return;
        K.paint(E.el('line', { x1: (a[0] + dx * t0).toFixed(1), y1: (a[1] + dy * t0).toFixed(1), x2: (a[0] + dx * t1).toFixed(1), y2: (a[1] + dy * t1).toFixed(1),
          'stroke-width': 1.5 }, grid), null, E.INK2);
      }
      for (var gx = -1.5; gx <= 1.0 + 1e-9; gx += GRID) gridLine([gx, -2.75, Z0], [gx, 2.0, Z0]);
      for (var gy = -2.75; gy <= 2.0 + 1e-9; gy += GRID) gridLine([-1.5, gy, Z0], [1.0, gy, Z0]);

      /* ── 左：桌子（三维，同一个相机）+ 桌脚围成的框 ── */
      var table = under(E.el('g', {}, s));
      var x0 = TAB.x - TAB.d / 2, x1 = TAB.x + TAB.d / 2, y0 = TAB.y - TAB.w / 2, y1 = TAB.y + TAB.w / 2, zt = Z0 + TAB.h, zb = zt - TAB.t;
      var lx = [x0 + TAB.inset, x1 - TAB.inset], ly = [y0 + TAB.inset, y1 - TAB.inset];
      var quad = K.paint(E.el('path', { d: poly([[lx[1], ly[0], Z0], [lx[1], ly[1], Z0], [lx[0], ly[1], Z0], [lx[0], ly[0], Z0]]),
        'stroke-width': 2.5, 'stroke-linejoin': 'round' }, table), E.tint(E.GOOD, 30, 'transparent'), E.GOOD);
      // 桌腿：后面两条（淡一点）先画，再画前面两条；上端都伸到桌面底下，由桌面压住
      [[lx[0], ly[0]], [lx[0], ly[1]], [lx[1], ly[0]], [lx[1], ly[1]]].forEach(function (l, k) {
        var a = P([l[0], l[1], Z0]), b = P([l[0], l[1], zb + 0.01]);
        var n = K.paint(E.el('line', { x1: a[0].toFixed(1), y1: a[1].toFixed(1), x2: b[0].toFixed(1), y2: b[1].toFixed(1), 'stroke-width': 8, 'stroke-linecap': 'round' }, table), null, E.INK2);
        if (k < 2) E.op(n, 0.55);
      });
      var top = K.paint(E.el('path', { d: poly([[x1, y0, zt], [x1, y1, zt], [x0, y1, zt], [x0, y0, zt]]), 'stroke-width': 2.5, 'stroke-linejoin': 'round' }, table), E.S2, E.INK2);
      K.paint(E.el('path', { d: poly([[x1, y0, zt], [x1, y1, zt], [x1, y1, zb], [x1, y0, zb]]), 'stroke-width': 2.5, 'stroke-linejoin': 'round' }, table), E.S2, E.INK2);   // 正面
      K.paint(E.el('path', { d: poly([[x1, y1, zt], [x0, y1, zt], [x0, y1, zb], [x1, y1, zb]]), 'stroke-width': 2.5, 'stroke-linejoin': 'round' }, table), E.S1, E.INK2);   // 右侧面
      // 桌子的重心（桌面中心）和它正下方的地面点
      var TC = P([TAB.x, TAB.y, zt - TAB.t / 2]), TF = P([TAB.x, TAB.y, Z0]), TT = P([TAB.x, TAB.y, zt]);
      var tLine = K.paint(E.el('path', { d: 'M ' + TC[0].toFixed(1) + ' ' + (TC[1] + 12).toFixed(1) + ' V ' + TF[1].toFixed(1), fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '6 5' }, s), null, E.WARN);
      var tDot = K.paint(E.el('circle', { cx: TF[0].toFixed(1), cy: TF[1].toFixed(1), r: 4.5 }, s), E.WARN);
      var tCom = E.comMark(s, E.WARN, 11); tCom.at(TC[0], TC[1]);
      var steady = E.el('g', {}, s);
      E.check(steady, TT[0] - 66, TT[1] - 60, 22, E.GOOD, 4);
      E.txt(steady, TT[0] - 36, TT[1] - 40, '放着就稳', 22, E.GOOD, 'start', 700);

      /* ── 右：人形 + 脚下那一小块（两只脚掌围成的地面框）+ 重心竖线 ── */
      // 重心用 G1 各连杆的质量与质心算（三维），脚掌按它的外形；「重心高」的量尺量到站直时的重心
      var soleR = hum.sole('right'), soleL = hum.sole('left');
      var YR = Math.min(soleR[0][1], soleR[1][1]), YL = Math.max(soleL[0][1], soleL[1][1]), YLin = Math.min(soleL[0][1], soleL[1][1]);
      var FX0 = soleR[2][0], FX1 = soleR[0][0];               // 脚跟、脚尖的 x
      EDGE = YL * hum.scale;                                    // 侧倒的支点：画面右脚（机器人左脚）的外沿
      var band = under(K.paint(E.el('path', { 'stroke-width': 2, 'stroke-linejoin': 'round' }, s), E.tint(E.GOOD, 30, 'transparent'), E.GOOD));
      // 压在地面网格上的字描一圈底色，网格线不从字里穿过
      function halo(n) { n.style.paintOrder = 'stroke'; n.style.stroke = E.S1; n.style.strokeWidth = '5px'; n.style.strokeLinejoin = 'round'; return n; }
      var BL = P([FX1, (YR + YL) / 2, Z0]);
      var bandLabel = halo(E.txt(s, BL[0], BL[1] + 30, '只有这一小块', 18, E.GOOD, 'middle', 700));
      var C0 = hum.com3();
      var hBar = under(E.el('g', {}, s));
      var HB = P([0.15, -0.42, Z0]), HT = P([0.15, -0.42, C0.z]);
      K.paint(E.el('path', { d: 'M ' + HB[0].toFixed(1) + ' ' + HB[1].toFixed(1) + ' L ' + HT[0].toFixed(1) + ' ' + HT[1].toFixed(1) +
        ' M ' + (HT[0] - 8).toFixed(1) + ' ' + HT[1].toFixed(1) + ' H ' + (HT[0] + 8).toFixed(1) + ' M ' + (HB[0] - 8).toFixed(1) + ' ' + HB[1].toFixed(1) + ' H ' + (HB[0] + 8).toFixed(1),
        fill: 'none', 'stroke-width': 2.5, 'stroke-linecap': 'round' }, hBar), null, E.WARN);
      E.txt(hBar, HT[0] - 12, HT[1] + 8, '重心高', 22, E.WARN, 'end', 700);   // 标在量尺顶端（重心那头）
      var hLine = K.paint(E.el('line', { 'stroke-width': 2.5, 'stroke-dasharray': '6 5' }, s), null, E.WARN);
      var hDot = K.paint(E.el('circle', { r: 4.5 }, s), E.WARN);
      var hCom = E.comMark(s, E.WARN, 11);
      var fallTag = E.tag(s, RX - 56, 40, 112, 36, '要倒！', E.BAD, { size: 22, ink: E.BAD, weight: 900 });
      var fellTag = E.tag(s, 340, 300, 92, 36, '倒了', E.BAD, { size: 22, ink: E.BAD, weight: 900 });   // 人往右倒，牌子放在支点左边，不被扫过
      var push = E.arrow(s, [[300, 160], [384, 160]], E.BAD, 5, { head: 14 });
      var pushTxt = E.txt(s, 300, 146, '推一下', 20, E.BAD, 'start', 700);
      var liftTxt = halo(E.txt(s, 392, 336, '抬脚', 20, E.BAD, 'end', 700));

      /* 控制牌 + 两只脚边上的「调」小弧 */
      var ctrl = E.el('g', {}, s);
      var ctrlRect = K.paint(E.el('rect', { x: RX - 96, y: 30, width: 192, height: 40, rx: 12, 'stroke-width': 2.5 }, ctrl), E.tint(E.ACC, 20), E.ACC);
      var ctrlTxt = E.txt(ctrl, RX, 57, '控制 · 一刻不停地调', 20, E.ACC, 'middle', 700);
      var ctrlX = E.cross(s, RX + 112, 50, 12, E.BAD, 4);
      // 两只脚外侧的小弧（脚踝高度、脚掌外沿再往外一点）
      var adj = [-1, 1].map(function (sg) {
        var a = P([0.02, sg < 0 ? YR - 0.07 : YL + 0.07, Z0 + 0.12]);
        return K.paint(E.el('path', { d: 'M ' + (a[0] + sg * 5).toFixed(1) + ' ' + (a[1] + 11).toFixed(1) + ' A 20 20 0 0 ' + (sg > 0 ? 0 : 1) + ' ' + (a[0] - sg * 5).toFixed(1) + ' ' + (a[1] - 11).toFixed(1),
          fill: 'none', 'stroke-width': 3, 'stroke-linecap': 'round' }, s), null, E.ACC);
      });

      /* 手托扫把：一个比喻，放在右边一张小卡片里（不站在三维地面上）。手是侧面、掌心朝上的一整只手——小臂、袖口、
         并拢的四指（两道指缝）、翘起的大拇指，填成和机器人一样的浅灰；扫把头朝上，扫把在自己的组里绕手心转 */
      var broomG = E.el('g', {}, s);
      K.paint(E.el('rect', { x: BROOM[0] - 82, y: 128, width: 218, height: 254, rx: 14, 'stroke-width': 2 }, broomG), E.S1, E.BORDER);
      var palm = E.el('g', {}, broomG);
      var skin = E.tint(E.INK, 72, 'var(--demo-surface)');
      function part(d, fill) { return K.paint(E.el('path', { d: d, 'stroke-width': 2.5, 'stroke-linejoin': 'round' }, palm), fill, E.INK2); }
      part('M -27 -1 L -21 14 L -40 50 L -55 41 Z', skin);                     // 小臂
      part('M -40 50 L -55 41 L -61 51 L -46 60 Z', E.S2);                     // 袖口
      part('M -24 -1 C -10 3 8 3 20 -2 C 30 -6 44 -8 50 -4 Q 56 0 50 5 C 40 9 26 10 16 12 C 4 15 -12 16 -22 14 Z', skin);   // 手掌 + 四指
      K.paint(E.el('path', { d: 'M 22 4 Q 34 2 47 0 M 20 8 Q 30 7 41 6', fill: 'none', 'stroke-width': 1.5, 'stroke-linecap': 'round' }, palm), null, E.INK2);   // 指缝
      part('M -22 0 C -25 -8 -19 -17 -13 -17 Q -8 -16 -10 -10 C -12 -5 -13 -3 -14 0 Z', skin);   // 大拇指
      var stick = E.el('g', {}, broomG);
      K.paint(E.el('line', { x1: 0, y1: 0, x2: 0, y2: -110, 'stroke-width': 5, 'stroke-linecap': 'round' }, stick), null, E.WARN);
      K.paint(E.el('path', { d: 'M -6 -108 L 6 -108 L 20 -146 L -20 -146 Z', 'stroke-width': 2.5, 'stroke-linejoin': 'round' }, stick), E.S2, E.WARN);
      K.paint(E.el('path', { d: 'M -9 -113 L -13 -141 M -3 -113 L -4 -141 M 3 -113 L 4 -141 M 9 -113 L 13 -141', fill: 'none', 'stroke-width': 1.5, 'stroke-linecap': 'round' }, stick), null, E.WARN);   // 扫把苗

      // 脚下的地面框：两只脚都踩着时是两只脚掌外沿围成的一块，抬起右脚（画面左脚）后只剩左脚那一块
      function setBand(lifted, bad) {
        var ya = lifted ? YLin : YR;
        band.setAttribute('d', poly([[FX1, ya, Z0], [FX1, YL, Z0], [FX0, YL, Z0], [FX0, ya, Z0]]));
        K.paint(band, E.tint(bad ? E.BAD : E.GOOD, 30, 'transparent'), bad ? E.BAD : E.GOOD);
      }

      return {
        el: s,
        draw: function (st, t) {
          var fade = E.fade;
          E.op(table, fade(st, 0, 0.6));
          E.op(grid, (0.22 * fade(st, 0, 0.6)).toFixed(3));
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
          var c = hum.world(hum.com()), c3 = hum.com3(), gp = P([c3.x, c3.y, Z0]);   // 重心、它正下方的地面点
          hCom.at(c[0], c[1]);
          var pm = fade(st, 3.1, 0.3);
          E.op(tCom.g, pm);
          var pl = fade(st, 4.15, 0.5);
          E.drawOn(tLine, pl); E.op(tDot, K.seg(pl, 0.9, 1));
          // 竖线落没落在脚下那一块里：看重心的左右（机器人 y）在不在着地脚掌的外沿之间
          var lifted = lift > 1, out = c3.y < (lifted ? YLin : YR) - 0.002 || c3.y > YL + 0.002;
          var lineC = out && fade(st, 15.6, 0.3) > 0 ? E.BAD : E.WARN;
          hLine.setAttribute('x1', c[0].toFixed(1)); hLine.setAttribute('x2', (c[0] + (gp[0] - c[0]) * pl).toFixed(1));
          hLine.setAttribute('y1', (c[1] + 12).toFixed(1)); hLine.setAttribute('y2', (c[1] + 12 + (gp[1] - c[1] - 12) * pl).toFixed(1));
          hLine.style.stroke = lineC; hDot.style.fill = lineC;
          hDot.setAttribute('cx', gp[0].toFixed(1)); hDot.setAttribute('cy', gp[1].toFixed(1));
          E.op(hLine, pl > 0 ? 1 : 0); E.op(hDot, K.seg(pl, 0.9, 1));
          E.op(hCom.g, pm);

          /* 桌子：6.35「桌子」亮一下；7.6「围成的大框」框亮出来；10.5「稳」打勾 */
          top.style.filter = E.win(st, 6.3, 7.4, 0.3) > 0.5 ? 'drop-shadow(0 0 6px ' + E.INK + ')' : '';
          E.op(quad, fade(st, 7.6, 0.6));
          E.pop(steady, TT[0], TT[1] - 48, fade(st, 10.5, 0.3), 0.7);

          /* 12.6「重心高」量高度；15.3「两只脚」窄条亮出来，16.5「这一小块」 */
          E.op(hBar, fade(st, 12.6, 0.4) * (1 - fade(st, 17.8, 0.4)));
          setBand(lifted, out && st >= 15.6);
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

          /* 29.9「手心里立一根扫把」：扫把小幅晃、手跟着往同一边挪；32.7「松手」手往下撤走，扫把倒下（倒 60°，不甩出卡片；33.6 后淡出） */
          var bp = fade(st, 29.9, 0.35) * (1 - fade(st, 33.7, 0.4));
          var wob = 7 * Math.sin(TAU * t / 1.6), drop = fade(st, 32.7, 0.4);
          var bu = K.seg(st, 32.75, 33.55), ang = wob * (1 - drop) + 60 * bu * bu;
          var hx = BROOM[0] + 10 * Math.sin(TAU * t / 1.6 + 0.6) * (1 - drop);
          palm.setAttribute('transform', 'translate(' + hx.toFixed(1) + ' ' + (BROOM[1] + 24 * drop).toFixed(1) + ') scale(' + HAND + ')');
          E.op(palm, 1 - drop);
          stick.setAttribute('transform', 'translate(' + hx.toFixed(1) + ' ' + BROOM[1] + ') rotate(' + ang.toFixed(1) + ')');
          E.op(broomG, bp);
          broomG.setAttribute('transform', 'translate(0 ' + (6 * (1 - fade(st, 29.9, 0.35))).toFixed(1) + ')');
        }
      };
    }
  };
})();
