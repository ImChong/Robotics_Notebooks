// 002 第 2 幕：难点二·浮动基。左边桌上的机械臂：「拧死」时两颗螺栓拧上，手臂摆动底座不动，「固定点」处一个原点 + 虚线箭头指到夹爪；
// 右边侧视人形：躯干上一个随身的小坐标架（紫），「没钉在任何地方」时「没钉住」，「到处跑」时往前走一小段、拖出一条轨迹，
// 「浮动基」标签；「在哪 / 朝哪 / 歪没歪」右侧三行依次出现（地上原点连到躯干、躯干朝向加粗、身子前倾 6°），「要控制的量」时三行变色；
// 「没有哪个电机能直接推躯干」：电机推躯干的箭头打叉；「脚踩地 / 借地面推回来的力」：脚亮起、地面往上的推力箭头；「扶正」时回正。
// 时刻取自配音的逐词时间（分镜时间 ≈ 念到的时刻），见 ../ep002.py。
(function () {
  var E = window.EP002, K = E.K;
  var TAU = Math.PI * 2;
  var X0 = 540, WALK = 63, W0 = 16.4, W1 = 18.2;           // 人形从 x 540 走 63（poseWalk 半个周期）到 603
  var X1 = X0 + WALK, TORSO_Y = 262, SC = 1.15;            // 人形放大 1.15 倍（以脚底为心）；TW 是躯干坐标架原点的画面 y
  var TW = 388 - (388 - TORSO_Y) * SC;
  var Q = [['在哪？', 20.55], ['朝哪？', 21.3], ['歪没歪？', 21.75]], Q_X = 690, Q_Y = [226, 260, 294];
  var TILT = 6;

  function mix(a, b, k) { return [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k]; }

  E.scenes[1] = {
    title: '难点二：浮动基，身子没固定',
    dur: 33.6,
    cues: [
      { at: 0, s: '难点二：**浮动基**（基 = 机器人的身子、底座）' },
      { at: 5.9, s: '机械臂：底座拧死，算什么都从这个固定点出发' },
      { at: 13.5, s: '人形：躯干没钉住，跟着身体到处跑' },
      { at: 20.2, s: '躯干在哪、朝哪、歪没歪：都要控制' },
      { at: 25.3, s: '没有电机能直接推躯干：只能脚踩地，借地面的力' }
    ],
    build: function (K) {
      var s = K.sceneSvg('难点二浮动基：机械臂底座拧死在桌上，算什么都从这个固定点出发；人形的躯干没钉在任何地方，在哪、朝哪、歪没歪都要控制，而且没有电机能直接推躯干，只能靠脚踩地借地面推回来的力');

      K.paint(E.el('line', { x1: 20, y1: 388, x2: 780, y2: 388, 'stroke-width': 3 }, s), null, E.BORDER);

      /* ── 左：桌子 + 机械臂 ── */
      var left = E.el('g', {}, s);
      K.paint(E.el('rect', { x: 72, y: 302, width: 12, height: 86, rx: 2, 'stroke-width': 2 }, left), E.S2, E.INK2);
      K.paint(E.el('rect', { x: 276, y: 302, width: 12, height: 86, rx: 2, 'stroke-width': 2 }, left), E.S2, E.INK2);
      K.paint(E.el('rect', { x: 60, y: 290, width: 240, height: 12, rx: 3, 'stroke-width': 2.5 }, left), E.S2, E.INK2);
      var arm = E.arm(left, 180, 290, 1);
      var baseHi = K.paint(E.el('path', { d: 'M 146 290 L 214 290 L 202 272 L 158 272 Z', fill: 'none', 'stroke-width': 3.5, 'stroke-linejoin': 'round' }, left), null, E.VIOLET);
      var bolts = [156, 204].map(function (x) {
        var g = E.el('g', {}, left);
        K.paint(E.el('circle', { cx: 0, cy: 0, r: 5.5, 'stroke-width': 2 }, g), E.S1, E.INK);
        var x1 = K.paint(E.el('line', { x1: -3.5, y1: 0, x2: 3.5, y2: 0, 'stroke-width': 2 }, g), null, E.INK);
        return { g: g, x: x, slot: x1 };
      });
      var origin = K.paint(E.el('circle', { cx: 180, cy: 272, r: 7, 'stroke-width': 2 }, left), E.ACC, E.S1);
      var fixedTag = E.tag(left, 128, 318, 104, 32, '固定点', E.ACC, { size: 20, ink: E.ACC, weight: 700 });
      var reach = E.arrow(left, [[186, 267], [250, 216]], E.ACC, 2.5, { dash: '6 5', head: 10 });

      /* ── 右：侧视人形 + 躯干上的小坐标架 ── */
      var noPin = E.tag(s, 412, 228, 96, 32, '没钉住', E.BAD, { size: 20, ink: E.BAD, weight: 700 });
      var trail = K.paint(E.el('path', { d: 'M ' + X0 + ' ' + TW + ' H ' + X1, fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '3 6', 'stroke-linecap': 'round' }, s), null, E.VIOLET);
      var trailStart = K.paint(E.el('circle', { cx: X0, cy: TW, r: 4 }, s), E.VIOLET);
      var where = K.paint(E.el('path', { d: 'M 450 388 L ' + X1 + ' ' + TW.toFixed(1), fill: 'none', 'stroke-width': 2, 'stroke-dasharray': '5 5' }, s), null, E.INK2);
      var whereO = K.paint(E.el('circle', { cx: 450, cy: 388, r: 5 }, s), E.INK2);
      var r = E.robot(s, X0);
      var frame = E.el('g', {}, r.body);
      E.arrow(frame, [[0, TORSO_Y], [30, TORSO_Y]], E.VIOLET, 3, { head: 9 });
      E.arrow(frame, [[0, TORSO_Y], [0, TORSO_Y - 30]], E.VIOLET, 3, { head: 9 });
      var heading = E.arrow(frame, [[30, TORSO_Y], [54, TORSO_Y]], E.ACC, 3, { head: 10, dash: '5 4' });
      K.paint(E.el('circle', { cx: 0, cy: TORSO_Y, r: 4.5 }, frame), E.VIOLET);
      var tiltArc = K.paint(E.el('path', { d: 'M 0 ' + (TORSO_Y - 44) + ' A 44 44 0 0 1 ' + (44 * Math.sin(TILT * Math.PI / 180)).toFixed(1) + ' ' + (TORSO_Y - 44 * Math.cos(TILT * Math.PI / 180)).toFixed(1),
        fill: 'none', 'stroke-width': 3, 'stroke-linecap': 'round' }, frame), null, E.ACC);
      var floatTag = E.tag(s, X1 - 52, 112, 104, 34, '浮动基', E.VIOLET, { size: 22, ink: E.VIOLET, weight: 900 });

      // 右侧三行 + 「要控制的量」
      var qs = Q.map(function (q, i) { return E.txt(s, Q_X, Q_Y[i], q[0], 22, E.INK, 'start', 700); });
      var ctrlTag = E.tag(s, Q_X - 6, 312, 112, 30, '要控制的量', E.ACC, { size: 18, ink: E.ACC, weight: 700 });

      // 电机 → 躯干（打叉）；地面推力
      var motor = E.el('g', {}, s);
      K.paint(E.el('circle', { cx: 450, cy: 186, r: 20, 'stroke-width': 3 }, motor), E.S2, E.INK2);
      K.paint(E.el('path', { d: 'M 442 194 V 178 L 450 188 L 458 178 V 194', fill: 'none', 'stroke-width': 2.5, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, motor), null, E.INK2);
      E.txt(motor, 450, 230, '电机', 18, E.INK2);
      var pushG = E.el('g', {}, s);
      var push = E.arrow(pushG, [[474, 192], [576, 236]], E.INK2, 3, { head: 11 });
      var noPush = E.cross(s, 525, 214, 12, E.BAD, 4);
      var grf = E.arrow(s, [[X1 + 9, 416], [X1 + 9, 394]], E.GOOD, 4, { head: 12 });
      var grfLabel = E.txt(s, X1 + 24, 414, '地面推回来的力', 18, E.GOOD, 'start', 700);

      return {
        el: s,
        draw: function (st, t) {
          var fade = E.fade;

          /* 0–0.6 两边淡入；13.5 起左边退暗 */
          E.op(left, (fade(st, 0, 0.6) * (1 - 0.6 * fade(st, 13.4, 0.5))).toFixed(3));
          E.op(r.g, fade(st, 0, 0.6));

          /* 3.89「身子」：躯干亮紫边；4.54「底座」：机械臂底座亮紫边；5.9 前都退回去。14.16「躯干」起躯干一直亮着 */
          var body = Math.max(E.win(st, 3.85, 5.8, 0.3), fade(st, 14.1, 0.3));
          r.torso.style.stroke = body > 0.5 ? E.VIOLET : E.INK;
          E.glow(r.torso, E.VIOLET, body > 0.5);
          E.op(baseHi, E.win(st, 4.5, 5.8, 0.3));

          /* 7.18「拧死」：两颗螺栓弹出、拧两圈 */
          bolts.forEach(function (b, i) {
            var p = fade(st, 7.15 + 0.12 * i, 0.3), rot = 720 * K.ease(K.seg(st, 7.15 + 0.12 * i, 7.75 + 0.12 * i));
            b.g.setAttribute('transform', 'translate(' + b.x + ' 284) scale(' + (0.6 + 0.4 * p).toFixed(3) + ')');
            b.slot.setAttribute('transform', 'rotate(' + rot.toFixed(1) + ')');
            E.op(b.g, p);
          });

          /* 8.28–10.3 手臂摆动、底座不动；11.75「固定点」；12.25「出发」：虚线从固定点画到夹爪 */
          var sw = K.seg(st, 8.25, 10.3), wob = sw > 0 && sw < 1 ? Math.sin(TAU * sw) : 0;
          arm.set(25 + 22 * wob, 60 - 18 * wob);
          E.pop(origin, 180, 272, fade(st, 11.7, 0.3), 0.5);
          E.pop(fixedTag.g, 180, 334, fade(st, 11.75, 0.3), 0.8);
          E.drawOn(reach.line, fade(st, 12.2, 0.5));
          E.op(reach.head, K.seg(fade(st, 12.2, 0.5), 0.9, 1));

          /* 15.1「钉」：「没钉住」；16.4–18.2 往前走 63、拖出轨迹；18.2–18.6 收成站姿 */
          E.pop(noPin.g, 460, 244, E.win(st, 15.1, W0, 0.3), 0.8);
          var u = K.seg(st, W0, W1), X = X0 + WALK * K.ease(u);
          var w = K.poseWalk((X - X0) / 126), k = fade(st, W1, 0.4);
          r.legs(mix(w.legA, [4, 0], k), mix(w.legB, [-4, 0], k));
          r.arm(mix([w.armA[0] * 0.8, w.armA[1] * 0.8], [8, 4], k));
          if (st < W0) r.stand();
          // 19–20：身子前倾 6°（绕脚底）；32.1–32.6「扶正」回 0
          var tilt = TILT * fade(st, 21.75, 0.5) * (1 - fade(st, 32.1, 0.5));
          r.place({ x: X, rot: tilt, pivot: [0, 388], scale: SC });
          trail.setAttribute('d', 'M ' + X0 + ' ' + TW.toFixed(1) + ' H ' + Math.max(X0 + 0.01, X).toFixed(2));
          E.op(trail, fade(st, W0, 0.2)); E.op(trailStart, fade(st, W0, 0.2));
          E.pop(floatTag.g, X1, 129, fade(st, 18.7, 0.3), 0.8);

          /* 小坐标架：14.2 出现；21.3「朝哪」朝向箭头；21.75「歪」弧线 */
          E.pop(frame, 0, TORSO_Y, fade(st, 14.2, 0.3), 0.6);
          E.drawOn(heading.line, fade(st, 21.3, 0.4));
          E.op(heading.head, K.seg(fade(st, 21.3, 0.4), 0.9, 1));
          E.drawOn(tiltArc, fade(st, 21.9, 0.4));

          /* 右侧三行；20.55「在哪」地上原点连到躯干；23.65「要控制的量」三行变 accent */
          var ctl = fade(st, 23.65, 0.3);
          qs.forEach(function (q, i) {
            var p = fade(st, Q[i][1], 0.3);
            E.op(q, p);
            q.setAttribute('transform', 'translate(' + (12 * (1 - p)).toFixed(1) + ' 0)');
            q.style.fill = ctl > 0.5 ? E.ACC : E.INK;
          });
          E.drawOn(where, fade(st, 20.55, 0.5));
          E.op(whereO, fade(st, 20.55, 0.2));
          E.pop(ctrlTag.g, Q_X + 50, 327, ctl, 0.8);

          /* 26.1「电机」；27.2「推」箭头；27.5「躯干」打叉；28.7「脚踩地」脚亮；29.95「借地面推回来的力」推力箭头 */
          E.op(motor, fade(st, 26.1, 0.3) * (1 - 0.6 * fade(st, 28.6, 0.4)));
          E.drawOn(push.line, fade(st, 26.6, 0.5));
          E.op(push.head, K.seg(fade(st, 26.6, 0.5), 0.9, 1));
          E.op(pushG, (1 - 0.6 * fade(st, 28.6, 0.4)).toFixed(3));
          E.pop(noPush, 525, 214, fade(st, 27.45, 0.25), 0.4);
          var foot = fade(st, 28.65, 0.3);
          r.foot.style.stroke = foot > 0.5 ? E.GOOD : E.INK;
          E.glow(r.foot, E.GOOD, foot > 0.5);
          E.drawOn(grf.line, fade(st, 29.95, 0.4));
          E.op(grf.head, K.seg(fade(st, 29.95, 0.4), 0.9, 1));
          E.op(grfLabel, fade(st, 30.3, 0.3));
        }
      };
    }
  };
})();
