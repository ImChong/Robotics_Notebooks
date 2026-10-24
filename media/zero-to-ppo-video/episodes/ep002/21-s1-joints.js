// 002 第 1 幕：难点一·关节多。左边 UR5（6 个转动关节的位置依次点亮 → 「6–7 个关节」），右边正面 G1：
// 29 个转动关节按念到的部位点亮（腿 12 → 腰 3 → 胳膊 14），「二十多个」时弹出「29」——数自它的 URDF（G1 没有脖子关节，头是固定的）。
// 「迈一步」时 G1 两腿侧摆把重心挪到右脚、抬左脚，腿 / 腰 / 胳膊各在念到时鼓一圈光；「配合」时虚线框住全身 +「一起配合」。
// 20.3 起机械臂退场、换成「控制」盒，「一起派活」时从控制盒连线到全部 29 个关节。
// 时刻取自配音的逐词时间（分镜时间 ≈ 念到的时刻），见 ../ep002.py。
(function () {
  var E = window.EP002, K = E.K;
  var TAU = Math.PI * 2;
  var ARM = [150, 360, 1.35];                              // UR5 底座 (x, y) 与缩放
  var HX = 540;                                            // 正面 G1 中心 x
  var GROUP_AT = { legs: 8.45, waist: 9.05, arms: 9.85 };
  var LABEL = { arms: ['胳膊', 612, 170], waist: ['腰', 612, 216], legs: ['腿', 612, 314] };
  var STEP = 15.45, PULSE = { legs: 16.6, waist: 17.4, arms: 17.75 }, TOGETHER = 18.9;
  var CTRL = 20.3, TICK = 21.2, FAN = 23.0, SYNC = 24.7;

  function bell(st, a, b) { var u = K.seg(st, a, b); return u > 0 && u < 1 ? Math.sin(Math.PI * u) : 0; }

  E.scenes[0] = {
    title: '难点一：关节多，要一起配合',
    dur: 26.3,
    cues: [
      { at: 0, s: '难点一：**关节多**' },
      { at: 2.6, s: '常见机械臂：六七个关节（UR5 是 6 个）' },
      { at: 6.2, s: '人形：腿、腰、胳膊，常见 **20+** 个关节（G1 是 29 个）' },
      { at: 13.1, s: '迈一步：腿、腰、胳膊得一起配合' },
      { at: 20.3, s: '**控制**：每一刻给 20 多个关节一起派活' }
    ],
    build: function (K) {
      var s = K.sceneSvg('难点一关节多：常见机械臂六七个关节（UR5 是 6 个），人形两条腿、腰、两条胳膊常见二十多个关节（G1 是 29 个）；迈一步时腿、腰、胳膊要一起配合，控制每一刻都要给这二十多个关节一起派活');

      /* ── 左：UR5 ── */
      var left = E.el('g', {}, s);
      E.txt(left, 150, 60, '机械臂', 24, E.INK2, 'middle', 700);
      K.paint(E.el('line', { x1: 70, y1: ARM[1], x2: 230, y2: ARM[1], 'stroke-width': 3 }, left), null, E.BORDER);
      var arm = E.arm(left, ARM[0], ARM[1], ARM[2]);
      var armDots = [0, 1, 2, 3, 4, 5].map(function () { return K.paint(E.el('circle', { r: 7, 'stroke-width': 2 }, left), E.ACC, E.S1); });
      var armTally = E.el('g', {}, left);
      E.txt(armTally, 150, 140, '6–7', 48, E.ACC, 'middle', 900);
      E.txt(armTally, 150, 172, '个关节', 20, E.INK2);

      /* ── 右：正面 G1 + 29 个关节点 ── */
      var together = E.el('path', { d: E.rrect(458, 80, 164, 318, 30), fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '8 6' }, s);
      K.paint(together, null, E.ACC);
      var hum = E.front(s, { dots: true });
      hum.place(HX, 1);
      var rings = hum.all.map(function () { return K.paint(E.el('circle', { r: 8, fill: 'none', 'stroke-width': 2.5 }, s), null, E.ACC); });
      var labels = {};
      Object.keys(LABEL).forEach(function (k) { labels[k] = E.txt(s, LABEL[k][1], LABEL[k][2], LABEL[k][0], 22, E.INK2, 'start', 700); });
      var tally = E.el('g', {}, s);
      E.txt(tally, 734, 112, '29', 48, E.ACC, 'middle', 900);
      E.txt(tally, 734, 140, '个关节', 20, E.INK2);
      var togetherTag = E.tag(s, 480, 50, 120, 32, '一起配合', E.ACC, { size: 20, ink: E.ACC, weight: 700 });

      /* ── 控制盒 + 连到每个关节点的线（人形此时站直不动，终点按站直的姿势算一次，在第一次画时） ── */
      var ctrl = E.el('g', {}, s);
      var ctrlRect = K.paint(E.el('rect', { x: 70, y: 190, width: 160, height: 64, rx: 14, 'stroke-width': 3 }, ctrl), E.tint(E.ACC, 20), E.ACC);
      E.txt(ctrl, 150, 232, '控制', 30, E.ACC, 'middle', 900);
      var tick = K.paint(E.el('rect', { x: 70, y: 190, width: 160, height: 64, rx: 14, fill: 'none', 'stroke-width': 2 }, s), null, E.ACC);
      var fanG = E.el('g', {}, s), fans = null;
      s.appendChild(hum.g);                                   // 连线压在人形下面
      rings.forEach(function (r) { s.appendChild(r); });

      return {
        el: s,
        draw: function (st, t) {
          var fade = E.fade;

          /* 0–0.6 两边先淡淡出现；2.7「机械臂」、6.3「人形机器人」各自亮起；13.2 机械臂退暗，20.3 退场 */
          var armOn = 0.35 + 0.65 * fade(st, 2.7, 0.4);
          E.op(left, (fade(st, 0, 0.6) * armOn * (1 - 0.65 * fade(st, 13.2, 0.5)) * (1 - fade(st, CTRL - 0.3, 0.4))).toFixed(3));
          E.op(hum.g, (fade(st, 0, 0.6) * (0.35 + 0.65 * fade(st, 6.3, 0.4))).toFixed(3));

          /* UR5 慢慢动；4.4「六七个」6 个关节点依次亮，4.95「关节」弹出 6–7 */
          arm.set(15 + 8 * Math.sin(TAU * t / 4), 45 + 10 * Math.sin(TAU * t / 3.1));
          arm.joints().forEach(function (p, k) {
            armDots[k].setAttribute('cx', p[0].toFixed(1)); armDots[k].setAttribute('cy', p[1].toFixed(1));
            E.op(armDots[k], fade(st, 4.4 + 0.1 * k, 0.15));
          });
          E.pop(armTally, 150, 150, fade(st, 4.95, 0.3), 0.7);

          /* G1 的姿势：15.45「迈一步」重心右移、抬左脚；17.3 上身往支撑脚一侧偏；17.7 两臂张开保持平衡；19.3 前收回 */
          var sh = 12 * fade(st, STEP, 0.6) * (1 - fade(st, 18.7, 0.6));
          var lift = 34 * bell(st, STEP + 0.6, 18.3);
          var lean = 4 * fade(st, 17.3, 0.5) * (1 - fade(st, 18.4, 0.6));
          var ab = bell(st, 17.7, 19.1);
          hum.pose({ shift: sh, lean: lean, lift: [lift, 0], arm: [[8 + 26 * ab, 4 + 34 * ab], [8 + 8 * ab, 4 + 10 * ab]] });

          /* 按念到的部位点亮关节点（变 accent、放大），右侧标签同时变色；关节点平时是淡淡的白点 */
          Object.keys(hum.groups).forEach(function (k) {
            var p = fade(st, GROUP_AT[k], 0.3);
            hum.groups[k].forEach(function (d, j) {
              var q = fade(st, GROUP_AT[k] + 0.03 * j, 0.25);
              d.style.fill = q > 0.5 ? E.ACC : E.INK;
              d.setAttribute('r', (4.5 + 2 * q).toFixed(2));
              E.op(d, (0.45 + 0.55 * q).toFixed(3));
            });
            E.op(labels[k], (0.35 + 0.65 * fade(st, 6.3, 0.4)).toFixed(3));
            labels[k].style.fill = p > 0.5 ? E.ACC : E.INK2;
          });
          E.pop(tally, 734, 110, fade(st, 11.15, 0.3), 0.6);

          /* 每个关节点的光圈：迈一步时腿 / 腰 / 胳膊各鼓一圈；24.7「配合」全身再一起鼓一圈 */
          hum.view.pre();
          hum.all.forEach(function (d, j) {
            var k = j < hum.groups.legs.length ? 'legs' : j < hum.groups.legs.length + hum.groups.waist.length ? 'waist' : 'arms';
            var u = Math.max(K.seg(st, PULSE[k], PULSE[k] + 0.7), K.seg(st, SYNC, SYNC + 0.7));
            var p = hum.locate(d);
            rings[j].setAttribute('cx', p[0].toFixed(1)); rings[j].setAttribute('cy', p[1].toFixed(1));
            rings[j].setAttribute('r', (8 + 14 * u).toFixed(2));
            E.op(rings[j], u > 0 && u < 1 ? (1 - u).toFixed(3) : 0);
          });

          /* 18.9「配合」虚线框住全身 +「一起配合」 */
          E.drawOn(together, fade(st, TOGETHER, 0.6));
          E.pop(togetherTag.g, 540, 66, fade(st, TOGETHER + 0.2, 0.3), 0.8);

          /* 20.3「控制」盒；21.2「每一刻」起每 0.5 s 外扩一圈；23.0「一起派活」连线画到每个关节点 */
          E.pop(ctrl, 150, 222, fade(st, CTRL, 0.4), 0.8);
          var u = st < TICK ? -1 : ((t - TICK) / 0.5) % 1;
          if (st >= TICK && u >= 0) {
            tick.setAttribute('transform', 'translate(150 222) scale(' + (1 + 0.25 * u).toFixed(3) + ') translate(-150 -222)');
            E.op(tick, (0.8 * (1 - u)).toFixed(3));
          } else E.op(tick, 0);
          ctrlRect.style.filter = st >= CTRL ? 'drop-shadow(0 0 6px ' + E.ACC + ')' : '';
          if (!fans && st >= FAN - 1 && fanG.isConnected) {
            fans = hum.all.map(function (d) {
              var p = hum.locate(d);
              return K.paint(E.el('path', { d: 'M 232 222 L ' + (p[0] - 7).toFixed(1) + ' ' + p[1].toFixed(1), fill: 'none', 'stroke-width': 1.6, 'stroke-linecap': 'round' }, fanG), null, E.ACC);
            });
          }
          if (fans) fans.forEach(function (l, j) { E.drawOn(l, fade(st, FAN + 0.015 * j, 0.5)); });
        }
      };
    }
  };
})();
