// 001 第 6 幕：绕回第一步：反馈。四步条「执行」为当前步、控制盒 ★ 保留；客厅 + 机器人（X=150）同 S2，右上角小地图接 S3 结尾：
// 反馈虚线变亮、往感知方向流 → 传感器各处冒扩散圈「新读数」→「感知」又成当前步 →
// 四个盒子轮流亮（示意）；踩偏前倾 → 下一圈感知发现（感知盒、小盒子闪红）→ 回正「扶正了 ✓」→
// 快进：一圈接一圈，从纸箱后面绕过去走到门口（小地图白点同步）→ 门发光「到了 ✓」→
// 这条线叫「反馈」：虚线加粗、光点从执行流向感知，四个盒子一起亮。
(function () {
  var E = window.EP001, K = E.K;
  var X0 = 150, X1 = 672;                              // 站位；走到门口（停在门前，摄像头不碰门框）
  var TOE = [18, 388];                                 // 前脚尖（本地坐标）= 画面 (168, 388)，前倾支点
  var FB = [[700, 52], [700, 62], [115, 62], [115, 54]];   // 四步条反馈虚线（执行 → 感知）
  var STEP_U = 45 / 580, FOOT_U0 = 0.117;              // S3 的 6 个小脚印
  var WALK0 = 16.6, WALK1 = 19.4, RAMP = 0.35;         // 走路：起步 / 停步各加减速 0.35 s
  var BLINK = 0.6;                                     // 轮流亮：每个盒子 0.6 s（真实时间 t）
  var TAU = Math.PI * 2;
  // 步长：E.robot 腿长 42 + 42，poseWalk 髋角 ±22°，支撑脚相对髋每半周期后移 2·84·sin22° ≈ 63px，
  // 一个周期 ≈ 126px 时支撑脚净位移为 0（00-common.js 注释里的「/ 90」会让支撑脚每步往后滑约 18px）
  var STRIDE = 4 * 84 * Math.sin(22 * Math.PI / 180);
  // 小地图以右上角 (780, 70) 为心缩到 0.9：面板底边 196 → 183，机器人站直时头顶 ≈ 188.5，走到右半边不再顶住面板

  function mix(c, p, base) {
    return p <= 0 ? base : p >= 1 ? c : 'color-mix(in srgb, ' + c + ' ' + (p * 100).toFixed(1) + '%, ' + base + ')';
  }
  function lerp(a, b, w) { return [a[0] + (b[0] - a[0]) * w, a[1] + (b[1] - a[1]) * w]; }
  // 一次性闪 n 下（每下 per 秒），返回 0…1；其余时刻 0
  function flash(st, a, n, per) {
    var u = (st - a) / per;
    return u < 0 || u >= n ? 0 : Math.sin(Math.PI * (u - Math.floor(u)));
  }
  // a 起 d 秒内鼓一下：0 → 1 → 0
  function bump(st, a, d) { var u = K.seg(st, a, a + d); return u > 0 && u < 1 ? Math.sin(Math.PI * u) : 0; }
  function scaleOr(n, cx, cy, s) { if (Math.abs(s - 1) > 0.0005) E.scaleAt(n, cx, cy, s); else n.removeAttribute('transform'); }

  // 小地图脚印（同 S3：绕椭圆自己的中心转到路线方向）
  function footprint(mm, u, side, rx, color) {
    var p = mm.at(u), q = mm.at(Math.min(1, u + 0.01)), a = Math.atan2(q[1] - p[1], q[0] - p[0]);
    var cx = p[0] - side * 5 * Math.sin(a), cy = p[1] + side * 5 * Math.cos(a);
    var e = E.el('ellipse', { cx: cx.toFixed(1), cy: cy.toFixed(1), rx: rx, ry: 2.5,
      transform: 'rotate(' + (a * 180 / Math.PI).toFixed(1) + ' ' + cx.toFixed(1) + ' ' + cy.toFixed(1) + ')' }, mm.prints);
    K.paint(e, color);
    return e;
  }

  /* 走路：已走距离（按 st，梯形速度：0.35 s 加速、匀速、0.35 s 减速） */
  var D = X1 - X0, VMAX = D / (WALK1 - WALK0 - RAMP), ACC = VMAX / RAMP;
  function walked(st) {
    var tau = st - WALK0, T = WALK1 - WALK0;
    if (tau <= 0) return 0;
    if (tau >= T) return D;
    if (tau < RAMP) return 0.5 * ACC * tau * tau;
    if (tau > T - RAMP) return D - 0.5 * ACC * (T - tau) * (T - tau);
    return 0.5 * VMAX * RAMP + VMAX * (tau - RAMP);
  }
  // 从纸箱后面走过：机器人始终踩在地面上、大小不变，纸箱画在它前面挡住腿（以前缩小上移会让脚悬空，违反物理）
  // 步态相位 = ∫ dX / STRIDE：脚不打滑。逐像素累加成表
  var PH = (function () {
    var a = [0];
    for (var i = 1; i <= D; i++) a.push(a[i - 1] + 1 / STRIDE);
    return a;
  })();
  function phase(d) { var i = Math.min(D - 1, Math.floor(d)), k = d - i; return PH[i] + (PH[i + 1] - PH[i]) * k; }

  E.scenes[5] = {
    title: '绕回第一步：反馈',
    dur: 25.8,
    cues: [
      { at: 0, s: '动作改变了位置和姿势 → 传感器再测 → 回到感知' },
      { at: 7.8, s: '身子一歪？下一圈感知发现，命令跟着调整' },
      { at: 16.5, s: '一圈接一圈：绕过纸箱，走到门口' },
      { at: 20.4, s: '执行绕回感知的这条线叫「**反馈**」：有了它才成圈' }
    ],
    build: function (K) {
      var s = K.sceneSvg('反馈：机器人迈完一步，位置和姿势变了，传感器马上再测，又回到感知；踩偏身子一歪，下一圈感知发现、命令跟着把身子扶正；一圈接一圈绕过纸箱走到门口。从执行绕回感知的这条线叫反馈，有了它四步才连成圈');

      /* ── 四步条（★ 常驻）；反馈线上的光点 ── */
      var bar = E.stepBar(s);
      E.op(bar.star, 1);
      var fbDot = K.paint(E.el('circle', { cx: 700, cy: 52, r: 6 }, bar.g), E.INK);

      /* ── 客厅：地面 → 门 → 小地图 → 机器人 → 扩散圈 → 纸箱（最上层，遮住机器人） ── */
      var room = E.room(s);
      var mapG = E.el('g', {}, s);
      var mm = E.miniMap(mapG);
      mm.setReveal(650);
      // 地图的位置和缩放由 E.miniMap 统一定（放在门左边，右缘 700）
      var prints = null;   // 脚印要沿路线量长度：draw 里挂到页面后再建

      var bot = E.robot(s, X0);
      var P = bot.pts();
      // 小盒子 = 骨盆里的 IMU（imu_in_pelvis），侧视时被大腿根和手挡住：闪红时在上面再画一个同位置的小盒子（跟着身子转）
      var imuHi = K.paint(E.el('rect', { x: P.imu[0] - 5, y: P.imu[1] - 5, width: 10, height: 10, rx: 2 }, bot.body), E.BAD);
      E.op(imuHi, 0);

      // 传感器扩散圈：摄像头、小盒子、髋 / 膝 / 踝、脚底（画面坐标，此时机器人站在 X0 不动）
      var ringG = E.el('g', {}, s);
      var rings = [[X0 + 11, 203], [X0 + P.imu[0], P.imu[1]], [X0 + P.hip[0], P.hip[1]], [X0 + P.knee[0], P.knee[1]], [X0 + P.ankle[0], P.ankle[1]], [X0 + 8, 388]]
        .map(function (c) { return K.paint(E.el('circle', { cx: c[0].toFixed(1), cy: c[1].toFixed(1), r: 4, fill: 'none', 'stroke-width': 2.5 }, ringG), null, E.GOOD); });
      s.appendChild(room.boxG);

      /* ── 文字 ── */
      var newRead = E.txt(s, 210, 176, '新读数', 20, E.GOOD);
      var note = E.txt(s, 24, 90, '示意：真机上四步同时跑，快慢不同', 18, E.MUT, 'start');
      var ffG = E.el('g', {}, s);
      E.txt(ffG, 24, 116, '快进', 18, E.MUT, 'start');
      K.paint(E.el('path', { d: 'M 66 102 L 77 109 L 66 116 Z M 78 102 L 89 109 L 78 116 Z' }, ffG), E.MUT);
      var tilt = E.txt(s, 270, 190, '身子往前歪！', 20, E.BAD);
      var okG = E.el('g', {}, s);
      E.txt(okG, 262, 190, '扶正了', 20, E.GOOD);
      var okCheck = E.check(okG, 298, 176, 18, E.GOOD);
      var arrG = E.el('g', {}, s);
      // 「到了 ✓」接替门上方的「门口」（门顶 y 148）
      E.txt(arrG, 735, 138, '到了', 20, E.ACC);
      var arrCheck = E.check(arrG, 759, 124, 16, E.ACC, 3);
      var fbWord = E.txt(s, 408, 96, '反馈', 24, E.INK, 'middle', 700);   // 离虚线远一点：光点沿虚线走过时不压字

      // 四步条盒子：k = 0 为「已讲过」，k = 1 为「当前步」样式；so 为描边不透明度
      function box(i, k, c, so, glow) {
        var r = bar.rects[i];
        K.paint(r, k > 0 ? E.tint(c, (22 * k).toFixed(1)) : E.S2, c);
        r.style.strokeWidth = (2 + k).toFixed(2);
        r.style.strokeOpacity = so.toFixed(3);
        bar.labels[i].style.fill = mix(c, k, E.INK2);
        E.glow(r, c, glow);
      }

      return {
        el: s,
        draw: function (st, t) {
          var fade = E.fade;

          /* ── 四步条 ──
             0–6.0「执行」当前步；6.0「感知」当前步（鼓一下）；7.8–20.4 按 t 轮流亮（每个 0.6 s，开头 0.4 s 从当前步过渡过去）；
             10.8「感知」描边闪两次 bad，压过轮流亮 0.6 s（其余三个同时退暗）；20.4 停；24.1 四个一起亮 */
          var breath = 0.6 + 0.4 * Math.sin(TAU * t);
          var sw = fade(st, 6.0, 0.3), curW = 1 - fade(st, 7.8, 0.4);
          var cur = [sw, 0, 0, 1 - sw];
          var rotW = Math.min(fade(st, 7.8, 0.4), 1 - fade(st, 20.4, 0.3));
          var allOn = fade(st, 24.1, 0.35);
          var fl = st >= 10.8 && st < 11.4 ? flash(st, 10.8, 2, 0.3) : -1;
          rotW *= 1 - 0.65 * E.win(st, 10.75, 11.35, 0.1);   // 闪红时其余三个盒子退暗，红闪压过轮流亮
          for (var i = 0; i < 4; i++) {
            var u = (((t / BLINK - i) % 4) + 4) % 4;
            var lit = u < 0.08 ? u / 0.08 : u < 1 ? 1 : u < 1.3 ? 1 - (u - 1) / 0.3 : 0;   // 亮 0.6 s，0.18 s 拖尾
            var k = Math.min(1, curW * cur[i] + rotW * lit + allOn);
            var B = breath + (1 - breath) * rotW;
            if (i === 0 && fl >= 0) box(0, fl, E.BAD, 0.5 + 0.5 * fl, fl > 0.5);
            else box(i, k, E.STEPS[i].color, 0.5 + k * (B - 0.5), k > 0.6 && (rotW > 0.5 || allOn > 0.5));
          }
          var b0 = bump(st, 6.0, 0.5);
          scaleOr(bar.rects[0], 115, 32, 1 + 0.08 * b0);
          scaleOr(bar.labels[0], 115, 32, 1 + 0.08 * b0);

          /* ── 反馈虚线：1.7 变亮（ink 2.5），dash 按 t 往感知方向滚；20.4 加粗到 4（虚线期间改平头，免得线帽把空隙填满） ── */
          var fp = fade(st, 1.7, 0.3), fw = fade(st, 20.4, 0.4);
          var fc = mix(E.INK, fp, E.MUT);
          bar.fb.line.style.stroke = fc; bar.fb.head.style.fill = fc;
          bar.fb.line.style.strokeWidth = (2 + 0.5 * fp + 1.5 * fw).toFixed(2);
          bar.fb.line.setAttribute('stroke-linecap', fw > 0 ? 'butt' : 'round');
          bar.fb.line.style.strokeDashoffset = st >= 1.7 ? (-((30 * t) % 11)).toFixed(2) : 0;
          scaleOr(bar.fb.head, 115, 54, 1 + 0.35 * fw);
          // 光点：20.4 起按 t 每 1.2 s 从执行流到感知
          var du = ((t / 1.2) % 1 + 1) % 1, dp = K.pointOn(FB, du);
          fbDot.setAttribute('cx', dp[0].toFixed(1)); fbDot.setAttribute('cy', dp[1].toFixed(1));
          E.op(fbDot, (fade(st, 20.4, 0.4) * Math.min(1, du / 0.06, (1 - du) / 0.06)).toFixed(3));
          E.glow(fbDot, E.INK, true);
          E.pop(fbWord, 408, 88, fade(st, 22.8, 0.35), 0.8);

          /* ── 机器人：9.5 绕前脚尖前倾 10°（0.6 s），13.2–14.0 回正；16.6–19.4 走到 X=672 ── */
          var d = walked(st), X = X0 + d;
          var lean = 10 * fade(st, 9.5, 0.6) * (1 - fade(st, 13.2, 0.8));
          bot.place({ x: X, rot: lean, pivot: TOE, scale: 1, dy: 0 });
          var w = Math.min(fade(st, WALK0, 0.25), 1 - fade(st, WALK1 - 0.25, 0.25));
          if (w > 0) {
            var g = K.poseWalk(phase(d));
            bot.legs(lerp([4, 0], g.legA, w), lerp([-4, 0], g.legB, w));
            bot.arm(lerp([8, 4], bot.walkArm(phase(d)), w));
          } else bot.stand();

          /* 4.5 传感器各处冒扩散圈（半径 4 → 18、淡出，0.8 s）+「新读数」 */
          var rp = K.seg(st, 4.5, 5.3), re = 1 - (1 - rp) * (1 - rp);
          rings.forEach(function (c) { c.setAttribute('r', (4 + 14 * re).toFixed(2)); E.op(c, rp > 0 && rp < 1 ? 1 - rp : 0); });
          E.pop(newRead, 210, 170, Math.min(fade(st, 4.5, 0.3), 1 - fade(st, 7.8, 0.3)), 0.8);

          /* 10.8 小盒子和「感知」盒同时闪两次 bad */
          var fi = Math.max(0, fl);
          E.op(imuHi, fi.toFixed(3));
          E.glow(imuHi, E.BAD, fi > 0.4);

          /* 9.5「身子往前歪！」→ 15.4「扶正了 ✓」；16.5 淡出（机器人要从这里走过） */
          // 两句在同一个位置：先让「身子往前歪！」退掉，「扶正了」再出来，不叠在一起
          E.op(tilt, Math.min(fade(st, 9.5, 0.3), 1 - fade(st, 15.2, 0.2)));
          E.pop(okG, 280, 184, Math.min(fade(st, 15.4, 0.3), 1 - fade(st, 16.5, 0.3)), 0.8);
          E.drawOn(okCheck, K.seg(st, 15.5, 15.8));

          /* 7.8「示意」；16.5「快进 ▶▶」；20.4–20.8 淡出 */
          var out = 1 - fade(st, 20.4, 0.4);
          E.op(note, Math.min(fade(st, 7.8, 0.4), out));
          E.op(ffG, Math.min(fade(st, 16.5, 0.3), out));

          /* ── 小地图：0–0.5 淡入，接 S3 结尾（revealX 650、路线 + 6 个脚印，第一个变 warn）；白点跟着 X 走；20.4 淡出 ── */
          E.op(mapG, Math.min(fade(st, 0, 0.5), out));
          if (!prints && mm.route.isConnected) {
            prints = [];
            for (var j = 0; j < 6; j++) prints.push(footprint(mm, FOOT_U0 + j * STEP_U, j % 2 ? -1 : 1, j ? 4.5 : 6, j ? E.VIOLET : E.WARN));
          }
          var mp = mm.dot((X - 110) / (X1 - 110));   // 走到门前时白点正好到地图上的门
          // 迷雾不退回：开场已散到 650，白点 + 90 超过 650 后才跟着往右散
          mm.setReveal(Math.max(650, Math.min(768, mp[0] + 90)));

          /* 19.4 门描边发 accent 光（20.4 后不再脉动）；「门口」→「到了 ✓」 */
          var ap = fade(st, 19.4, 0.3), pulse = 1 - fade(st, 20.4, 0.4);
          room.door.style.stroke = mix(E.ACC, ap, E.INK2);
          room.door.style.strokeWidth = (2.5 + ap).toFixed(2);
          var gr = ap * (6 + 4 * pulse * Math.sin(TAU * t / 1.2));
          room.door.style.filter = ap > 0 ? 'drop-shadow(0 0 ' + gr.toFixed(1) + 'px ' + E.ACC + ')' : '';
          // 「门口」先退、「到了 ✓」再出来（同一个位置，各占一半）
          E.op(room.doorLabel, 1 - Math.min(1, 2 * ap));
          E.pop(arrG, 745, 224, Math.max(0, 2 * ap - 1), 0.8);
          E.drawOn(arrCheck, K.seg(st, 19.5, 19.8));
        }
      };
    }
  };
})();
