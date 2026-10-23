// 002 第 1 幕：难点一·关节多。左边机械臂（6 个关节点 → 「6–7 个关节」），右边正面人形（腿 → 腰 → 胳膊 → 脖子按念到的词点亮，
// 「二十多个」时弹出「20+」）。「迈一步」时人形横移重心、抬左脚，腿 / 腰 / 胳膊各在念到时鼓一圈光；「配合」时虚线框住全身 +「一起配合」。
// 21.5 起机械臂退场、换成「控制」盒，「一起派活」时从控制盒连线到全部 14 个关节点。人形上一个点代表一处关节，不是一个自由度。
// 时刻取自配音的逐词时间（分镜时间 ≈ 念到的时刻），见 ../ep002.py。
(function () {
  var E = window.EP002, K = E.K;
  var R = Math.PI / 180, TAU = Math.PI * 2;
  var ARM = [150, 360, 1.35];                              // 机械臂底座 (x, y) 与缩放
  var HX = 540;                                            // 正面人形中心 x
  var GROUP_AT = { legs: 8.3, waist: 9.05, arms: 9.8, neck: 11.1 };
  var LABEL = { neck: ['脖子', 636, 146], arms: ['胳膊', 636, 206], waist: ['腰', 636, 250], legs: ['腿', 636, 326] };
  var STEP = 16.65, PULSE = { legs: 17.8, waist: 18.6, arms: 18.95 }, TOGETHER = 20.1;
  var CTRL = 21.5, FAN = 24.2;

  function bell(st, a, b) { var u = K.seg(st, a, b); return u > 0 && u < 1 ? Math.sin(Math.PI * u) : 0; }
  // 机械臂关节点（本地坐标）：底座转台、肩、肘、手腕三个
  function armJoints(a1, a2) {
    var p0 = [0, -18], p1 = [p0[0] + 64 * Math.sin(a1 * R), p0[1] - 64 * Math.cos(a1 * R)];
    var p2 = [p1[0] + 54 * Math.sin((a1 + a2) * R), p1[1] - 54 * Math.cos((a1 + a2) * R)];
    function lerp(k) { return [p1[0] + (p2[0] - p1[0]) * k, p1[1] + (p2[1] - p1[1]) * k]; }
    return [[0, -8], p0, p1, lerp(0.6), lerp(0.8), p2];
  }

  E.scenes[0] = {
    title: '难点一：关节多，要一起配合',
    dur: 27.5,
    cues: [
      { at: 0, s: '难点一：**关节多**' },
      { at: 2.6, s: '常见机械臂：六七个关节' },
      { at: 6.2, s: '人形：腿、腰、胳膊、脖子，常见 **20+** 个关节' },
      { at: 14.3, s: '迈一步：腿、腰、胳膊得一起配合' },
      { at: 21.5, s: '**控制**：每一刻给 20 多个关节一起派活' }
    ],
    build: function (K) {
      var s = K.sceneSvg('难点一关节多：常见机械臂六七个关节，人形两条腿、腰、两条胳膊加脖子常见二十多个关节；迈一步时腿、腰、胳膊要一起配合，控制每一刻都要给这二十多个关节一起派活');

      /* ── 左：机械臂 ── */
      var left = E.el('g', {}, s);
      E.txt(left, 150, 60, '机械臂', 24, E.INK2, 'middle', 700);
      K.paint(E.el('line', { x1: 70, y1: ARM[1], x2: 230, y2: ARM[1], 'stroke-width': 3 }, left), null, E.BORDER);
      var arm = E.arm(left, ARM[0], ARM[1], ARM[2]);
      var armDots = [0, 1, 2, 3, 4, 5].map(function () { return K.paint(E.el('circle', { r: 7, 'stroke-width': 2 }, left), E.ACC, E.S1); });
      var armTally = E.el('g', {}, left);
      E.txt(armTally, 150, 140, '6–7', 48, E.ACC, 'middle', 900);
      E.txt(armTally, 150, 172, '个关节', 20, E.INK2);

      /* ── 右：正面人形 ── */
      var together = E.el('path', { d: E.rrect(458, 80, 164, 318, 30), fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '8 6' }, s);
      K.paint(together, null, E.ACC);
      var hum = E.front(s);
      hum.place(HX, 1);
      var rings = hum.all.map(function () { return K.paint(E.el('circle', { r: 8, fill: 'none', 'stroke-width': 2.5 }, s), null, E.ACC); });
      var labels = {};
      Object.keys(LABEL).forEach(function (k) { labels[k] = E.txt(s, LABEL[k][1], LABEL[k][2], LABEL[k][0], 22, E.INK2, 'start', 700); });
      var tally = E.el('g', {}, s);
      E.txt(tally, 734, 112, '20+', 48, E.ACC, 'middle', 900);
      E.txt(tally, 734, 140, '个关节', 20, E.INK2);
      var togetherTag = E.tag(s, 480, 50, 120, 32, '一起配合', E.ACC, { size: 20, ink: E.ACC, weight: 700 });

      /* ── 控制盒 + 连到每个关节点的线（人形此时站直不动，终点按站直的姿势算一次） ── */
      var ctrl = E.el('g', {}, s);
      var ctrlRect = K.paint(E.el('rect', { x: 70, y: 190, width: 160, height: 64, rx: 14, 'stroke-width': 3 }, ctrl), E.tint(E.ACC, 20), E.ACC);
      E.txt(ctrl, 150, 232, '控制', 30, E.ACC, 'middle', 900);
      var tick = K.paint(E.el('rect', { x: 70, y: 190, width: 160, height: 64, rx: 14, fill: 'none', 'stroke-width': 2 }, s), null, E.ACC);
      var fans = hum.all.map(function (d) {
        var p = hum.locate(d);
        var l = E.el('path', { d: 'M 232 222 L ' + (p[0] - 8).toFixed(1) + ' ' + p[1].toFixed(1), fill: 'none', 'stroke-width': 1.8, 'stroke-linecap': 'round' }, s);
        K.paint(l, null, E.ACC);
        return l;
      });
      s.appendChild(hum.g);                                   // 连线压在人形下面
      rings.forEach(function (r) { s.appendChild(r); });

      return {
        el: s,
        draw: function (st, t) {
          var fade = E.fade;

          /* 0–0.6 两边先淡淡出现；2.7「机械臂」、6.3「人形机器人」各自亮起；14.4 机械臂退暗，21.5 退场 */
          var armOn = 0.35 + 0.65 * fade(st, 2.7, 0.4);
          E.op(left, (fade(st, 0, 0.6) * armOn * (1 - 0.65 * fade(st, 14.4, 0.5)) * (1 - fade(st, CTRL - 0.3, 0.4))).toFixed(3));
          E.op(hum.g, (fade(st, 0, 0.6) * (0.35 + 0.65 * fade(st, 6.3, 0.4))).toFixed(3));

          /* 机械臂慢慢动；4.4「六七个」关节点依次亮，4.95「关节」弹出 6–7 */
          var a1 = 15 + 8 * Math.sin(TAU * t / 4), a2 = 45 + 10 * Math.sin(TAU * t / 3.1);
          arm.set(a1, a2);
          armJoints(a1, a2).forEach(function (p, k) {
            armDots[k].setAttribute('cx', (ARM[0] + ARM[2] * p[0]).toFixed(1));
            armDots[k].setAttribute('cy', (ARM[1] + ARM[2] * p[1]).toFixed(1));
            E.op(armDots[k], fade(st, 4.4 + 0.1 * k, 0.15));
          });
          E.pop(armTally, 150, 150, fade(st, 4.95, 0.3), 0.7);

          /* 人形的姿势：16.65「迈一步」重心右移、抬左脚；18.6 上身往支撑脚一侧偏；18.95 两臂张开保持平衡；20.5 前收回 */
          var sh = 12 * fade(st, STEP, 0.6) * (1 - fade(st, 19.9, 0.6));
          var lift = 34 * bell(st, STEP + 0.6, 19.5);
          var lean = 4 * fade(st, 18.5, 0.5) * (1 - fade(st, 19.6, 0.6));
          var ab = bell(st, 18.9, 20.3);
          hum.pose({ shift: sh, lean: lean, lift: [lift, 0], arm: [[8 + 26 * ab, 4 + 34 * ab], [8 + 8 * ab, 4 + 10 * ab]] });

          /* 按念到的部位点亮关节点（变 accent、放大），右侧标签同时变色 */
          Object.keys(hum.groups).forEach(function (k) {
            var p = fade(st, GROUP_AT[k], 0.3);
            hum.groups[k].forEach(function (d, j) {
              var q = fade(st, GROUP_AT[k] + 0.05 * j, 0.25);
              d.style.fill = q > 0.5 ? E.ACC : E.INK;
              d.setAttribute('r', (6.5 + 1.5 * q).toFixed(2));
            });
            E.op(labels[k], (0.35 + 0.65 * fade(st, 6.3, 0.4)).toFixed(3));
            labels[k].style.fill = p > 0.5 ? E.ACC : E.INK2;
          });
          E.pop(tally, 734, 110, fade(st, 12.35, 0.3), 0.6);

          /* 每个关节点的光圈：迈一步时腿 / 腰 / 胳膊各鼓一圈；25.9「配合」全身再一起鼓一圈 */
          hum.all.forEach(function (d, j) {
            var k = j < 6 ? 'legs' : j < 7 ? 'waist' : j < 13 ? 'arms' : 'neck';
            var u = Math.max(K.seg(st, PULSE[k] || 99, (PULSE[k] || 99) + 0.7), K.seg(st, 25.9, 26.6));
            var p = hum.locate(d);
            rings[j].setAttribute('cx', p[0].toFixed(1)); rings[j].setAttribute('cy', p[1].toFixed(1));
            rings[j].setAttribute('r', (8 + 14 * u).toFixed(2));
            E.op(rings[j], u > 0 && u < 1 ? (1 - u).toFixed(3) : 0);
          });

          /* 20.1「配合」虚线框住全身 +「一起配合」 */
          E.drawOn(together, fade(st, TOGETHER, 0.6));
          E.pop(togetherTag.g, 540, 66, fade(st, TOGETHER + 0.2, 0.3), 0.8);

          /* 21.5「控制」盒；22.4「每一刻」起每 0.5 s 外扩一圈；24.2「一起派活」连线画到每个关节点 */
          E.pop(ctrl, 150, 222, fade(st, CTRL, 0.4), 0.8);
          var u = st < 22.4 ? -1 : ((t - 22.4) / 0.5) % 1;
          if (st >= 22.4 && u >= 0) {
            var g = 1 + 0.25 * u;
            tick.setAttribute('transform', 'translate(150 222) scale(' + g.toFixed(3) + ') translate(-150 -222)');
            E.op(tick, (0.8 * (1 - u)).toFixed(3));
          } else E.op(tick, 0);
          ctrlRect.style.filter = st >= CTRL ? 'drop-shadow(0 0 6px ' + E.ACC + ')' : '';
          fans.forEach(function (l, j) { E.drawOn(l, fade(st, FAN + 0.02 * j, 0.5)); });
        }
      };
    }
  };
})();
