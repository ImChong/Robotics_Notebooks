// 001 第 5 幕：第四步：执行。四步条「执行」为当前步、控制盒 ★ 保留；客厅 + 机器人（X=110）同 S2，地面留着 S3 的紫色脚印：
// 命令小圆点从控制盒流到膝盖（3 趟，每趟膝盖闪 warn）→ 放大镜从膝盖弹出：电机 ≈ 肌肉（快 · 劲小）→ 接上减速器（慢 · 劲大）→
// 放大镜缩回膝盖；近腿关节 + 骨架变 warn 发光 → 腿抬起前摆、一脚踩进脚印（冲击线）→ 躯干前移、后腿跟上，停在 X=150。
(function () {
  var E = window.EP001, K = E.K;
  var X0 = 110, X1 = 150;
  var R = Math.PI / 180;
  var FP = [178, 388];                                          // S3 的脚印
  var MAG = [330, 200], MAG_R = 100;                            // 放大镜
  var MOTOR = [270, 195], PINION = [310, 195], GEAR = [350, 195];
  var CMD = [[505, 52], [505, 200], [300, 300], [113, 344]];    // 命令走的三次贝塞尔：控制盒底 → 膝盖
  var TRIP = 1.2, TRIP0 = 0.6, TRIPS = 3;                       // 每趟 1.2 s，0.6 起共 3 趟，到达 1.8 / 3.0 / 4.2
  // 电机每秒 1.5 圈（真实时间 t）。规格写 2 圈，但成片 30 fps 下小齿轮每帧转 24°，超过半个齿距 22.5°，齿线看起来倒着转；
  // 1.5 圈时小齿轮每帧 18°、大齿轮每帧 8°（半齿距 10°），转向都看得清
  var MOTOR_RPS = 1.5;
  var GEAR_RATIO = 8 / 18;                                      // 大齿轮转速 = 小齿轮 × 齿数比（规格 12/28 ≈ 0.429，差 3.7%），齿才始终交错、不打架

  /* 走一步（16.6–18.5）。腿长 42 + 42；站姿 [4°, 0°] 的踝在 y = 302 + 84·cos4°，把这个高度当作「踩在地上」 */
  var S4 = Math.sin(4 * R), GA = 302 + 84 * Math.cos(4 * R);
  var NEAR_T = FP[0] - 7;                                       // 171：脚线是 [踝 − 4, 踝 + 18]，中点对准脚印
  var FAR0 = X0 + 4 - 84 * S4, FAR1 = X1 + 4 - 84 * S4;        // 远脚踝：站在 110 时 / 停在 150 后（回到站姿下的相对位置）
  var LIFT = [25, -15];                                         // 髋前屈 25°、膝弯 40°（膝角是全局角：25 − 40）
  var LIFT_A = [X0 + 42 * Math.sin(LIFT[0] * R) + 42 * Math.sin(LIFT[1] * R), 302 + 42 * Math.cos(LIFT[0] * R) + 42 * Math.cos(LIFT[1] * R)];

  function mix(c, p, base) {
    return p <= 0 ? base : p >= 1 ? c : 'color-mix(in srgb, ' + c + ' ' + (p * 100).toFixed(1) + '%, ' + base + ')';
  }
  // a 起 d 秒内鼓一下：0 → 1 → 0
  function bump(st, a, d) { var u = K.seg(st, a, a + d); return u > 0 && u < 1 ? Math.sin(Math.PI * u) : 0; }
  // 弹出带一点回弹
  function backOut(x) { var c = 1.6; return x <= 0 ? 0 : x >= 1 ? 1 : 1 + (c + 1) * Math.pow(x - 1, 3) + c * Math.pow(x - 1, 2); }
  function scaleOr(n, cx, cy, s) { if (Math.abs(s - 1) > 0.0005) E.scaleAt(n, cx, cy, Math.max(s, 0.001)); else n.removeAttribute('transform'); }
  function line(parent, color, w) {
    var l = E.el('line', { 'stroke-width': w, 'stroke-linecap': 'round' }, parent);
    K.paint(l, null, color);
    return l;
  }
  function put(l, a, b) {
    l.setAttribute('x1', a[0].toFixed(1)); l.setAttribute('y1', a[1].toFixed(1));
    l.setAttribute('x2', b[0].toFixed(1)); l.setAttribute('y2', b[1].toFixed(1));
  }
  // 文字淡入并上浮 6px
  function rise(n, p) { E.op(n, p); n.setAttribute('transform', p < 1 ? 'translate(0 ' + (6 * (1 - p)).toFixed(2) + ')' : ''); }

  /* 命令曲线按弧长取点（解析算，不用等节点挂到页面） */
  function bez(u) {
    var v = 1 - u, a = v * v * v, b = 3 * v * v * u, c = 3 * v * u * u, d = u * u * u;
    return [a * CMD[0][0] + b * CMD[1][0] + c * CMD[2][0] + d * CMD[3][0], a * CMD[0][1] + b * CMD[1][1] + c * CMD[2][1] + d * CMD[3][1]];
  }
  var LUT = (function () {
    var pts = [bez(0)], len = [0];
    for (var i = 1; i <= 160; i++) { pts.push(bez(i / 160)); len.push(len[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])); }
    return { pts: pts, len: len };
  })();
  function cmdAt(s) {
    var want = E.clamp(s, 0, 1) * LUT.len[160], i = 1;
    while (i < 160 && LUT.len[i] < want) i++;
    var k = (want - LUT.len[i - 1]) / (LUT.len[i] - LUT.len[i - 1]), a = LUT.pts[i - 1], b = LUT.pts[i];
    return [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k];
  }

  /* 两段腿 IK：髋 h → 踝 a（画面坐标），膝朝前；返回 [髋角, 膝角]（度，全局，0 = 竖直向下，同 E.robot） */
  function ik(h, a) {
    var dx = a[0] - h[0], dy = a[1] - h[1], d = Math.min(83.999, Math.hypot(dx, dy));
    var phi = Math.atan2(dx, dy) / R, al = Math.acos(d / 84) / R;
    return [phi + al, phi - al];
  }
  // 髋在 hx、踝在 a 时，髋至少要从 302 降多少才够得着
  function need(hx, a) { var dx = a[0] - hx; return Math.max(0, (a[1] - 302) - Math.sqrt(Math.max(0, 84 * 84 - dx * dx))); }
  // 机器人在分镜时间 st 的位置与两腿角度
  function motion(st) {
    var x = X0 + (X1 - X0) * K.ease(K.seg(st, 17.5, 18.5));
    if (st < 17.5) {                    // 16.6–17.5 抬腿前摆
      var u = K.ease(K.seg(st, 16.6, 17.5));
      return { x: x, dy: 0, near: [4 + (LIFT[0] - 4) * u, LIFT[1] * u], far: [-4, 0] };
    }
    var na, fa;
    if (st < 18.1) {                    // 远脚撑地；近脚落向脚印：x 先快后慢，最后竖直踩下
      var v = K.seg(st, 17.5, 18.1);
      na = [LIFT_A[0] + (NEAR_T - LIFT_A[0]) * (1 - (1 - v) * (1 - v)), LIFT_A[1] + (GA - LIFT_A[1]) * v * v * v - 6 * Math.sin(Math.PI * v)];
      fa = [FAR0, GA];
    } else {                            // 近脚踩住脚印不动；后腿抬起跟上
      var w = K.seg(st, 18.1, 18.5);
      na = [NEAR_T, GA];
      fa = [FAR0 + (FAR1 - FAR0) * K.ease(w), GA - 8 * Math.sin(Math.PI * w)];
    }
    // 躯干前移时两腿都够不着就把髋降一点（踩下那一刻约 7px），脚不打滑
    var dy = Math.max(need(x, na), need(x + 4, fa)), hip = 302 + dy;
    return { x: x, dy: dy, near: ik([x, hip], na), far: ik([x + 4, hip], fa) };
  }

  /* 齿轮：圆 + 外圈 n 根短齿线（长 5）+ 轮辐（看得出转向：30 fps 下齿线转太快会像倒转） */
  function gear(parent, c, r, n, spokes) {
    var outer = E.el('g', {}, parent), g = E.el('g', {}, outer);
    for (var i = 0; i < n; i++) {
      var a = 360 / n * i * R;
      put(line(g, E.WARN, 3), [c[0] + r * Math.cos(a), c[1] + r * Math.sin(a)], [c[0] + (r + 5) * Math.cos(a), c[1] + (r + 5) * Math.sin(a)]);
    }
    K.paint(E.el('circle', { cx: c[0], cy: c[1], r: r, 'stroke-width': 3 }, g), E.S2, E.WARN);
    for (var k = 0; k < spokes; k++) {
      var b = (360 / spokes * k - 90) * R;
      put(line(g, E.WARN, 2.5), c, [c[0] + (r - 5) * Math.cos(b), c[1] + (r - 5) * Math.sin(b)]);
    }
    K.paint(E.el('circle', { cx: c[0], cy: c[1], r: 3 }, g), E.WARN);
    return { outer: outer, g: g, spin: function (deg) { g.setAttribute('transform', 'rotate(' + (deg % 360).toFixed(2) + ' ' + c[0] + ' ' + c[1] + ')'); } };
  }

  E.scenes[4] = {
    title: '第四步：执行',
    dur: 18.6,
    cues: [
      { at: 0, s: '第四步 **执行**：命令送进每个关节里的电机' },
      { at: 4.5, s: '电机 ≈ 肌肉：往往转得快、劲儿小' },
      { at: 8.8, s: '减速器：换成转得慢一点、劲儿大一点' },
      { at: 12.9, s: '关节 + 骨架：腿抬起来，一脚踩下去' }
    ],
    build: function (K) {
      var s = K.sceneSvg('执行：控制算出的命令送进膝盖里的电机；放大看，电机转得快、劲儿小，接上减速器变成转得慢、劲儿大；再通过关节和骨架，机器人抬腿、一脚踩进规划好的脚印');

      /* ── 四步条 + 客厅 + 脚印 + 机器人（脚印在机器人脚下；纸箱在最上层，与 S3 / S6 的绘制顺序一致） ── */
      var bar = E.stepBar(s);
      E.op(bar.star, 1);
      var room = E.room(s);
      var print = K.paint(E.el('ellipse', { cx: FP[0], cy: FP[1], rx: 16, ry: 4 }, s), E.VIOLET);
      var bot = E.robot(s, X0);
      s.appendChild(room.boxG);
      var P = bot.pts(), KNEE = [X0 + P.knee[0], P.knee[1]];   // 站姿膝盖 ≈ (113, 344)

      /* ── 落脚冲击线（避开斜下来的小腿：左、上、右各一根） ── */
      var hits = [[[157, 382], [-0.87, -0.5]], [[178, 372], [0, -1]], [[199, 382], [0.87, -0.5]]].map(function (h) {
        return { l: line(s, E.WARN, 3), b: h[0], d: h[1] };
      });

      /* ── 命令走的路线（淡虚线，第一趟小圆点边走边画出来）+ 小圆点 + 「命令」 ── */
      var routeG = E.el('g', {}, s);
      var route = E.el('path', { d: 'M ' + CMD[0].join(' ') + ' C ' + CMD.slice(1).map(function (p) { return p.join(' '); }).join(' '),
        fill: 'none', 'stroke-width': 2, 'stroke-linecap': 'round', 'stroke-dasharray': '4 6' }, routeG);
      K.paint(route, null, E.ACC);
      var dot = K.paint(E.el('circle', { cx: CMD[0][0], cy: CMD[0][1], r: 5 }, s), E.ACC);
      E.glow(dot, E.ACC, true);
      var cmdLabel = E.txt(s, 517, 108, '命令', 18, E.ACC, 'start');

      /* ── 放大镜：整组以膝盖为心缩放（从膝盖弹出、缩回膝盖） ── */
      var mag = E.el('g', {}, s);
      var lead = line(mag, E.WARN, 2);
      put(lead, [262, 272], KNEE);
      K.paint(E.el('circle', { cx: MAG[0], cy: MAG[1], r: MAG_R, 'stroke-width': 3 }, mag), E.S1, E.WARN);
      // 规格 y=128 时首尾两个字压在圆周上，下移到 140 才完整落在圆内
      var magTitle = E.txt(mag, 330, 140, '电机 ≈ 肌肉', 20, E.WARN);
      // 电机：圆 + 三根辐条
      K.paint(E.el('circle', { cx: MOTOR[0], cy: MOTOR[1], r: 26, 'stroke-width': 3 }, mag), E.S2, E.WARN);
      var spokes = E.el('g', {}, mag);
      for (var k = 0; k < 3; k++) {
        var a = (120 * k - 90) * R;
        put(line(spokes, E.WARN, 3), MOTOR, [MOTOR[0] + 20 * Math.cos(a), MOTOR[1] + 20 * Math.sin(a)]);
      }
      K.paint(E.el('circle', { cx: MOTOR[0], cy: MOTOR[1], r: 4 }, mag), E.WARN);
      // 「电机」「减速器」从规格 y=245 下移到 250：大齿轮齿尖低到 y=228，245 时字顶几乎贴着齿
      var motorName = E.txt(mag, 270, 250, '电机', 18, E.INK);
      // 减速器：小齿轮（和电机同速）+ 大齿轮（反向，转速 × 12/28）。大齿轮齿线错开半格，起始时和小齿轮的齿交错
      var big = gear(mag, GEAR, 28, 18, 3), small = gear(mag, PINION, 12, 8, 1);
      var gearName = E.txt(mag, 350, 250, '减速器', 18, E.INK);
      // 圆外下方的两行
      var fastTxt = E.txt(s, 270, 322, '快 · 劲小', 18, E.INK);
      var slowTxt = E.txt(s, 370, 322, '慢 · 劲大', 18, E.WARN);

      /* ── 关节 + 骨架 ── */
      // 规格 (200,300) 居中时左缘约 x=150，机器人停到 X=150 后躯干（右缘 165）会压字：改为左对齐从 x=186 起
      var jointTxt = E.txt(s, 186, 300, '关节 + 骨架', 20, E.WARN, 'start');
      var legParts = [
        { n: bot.hipDot, at: 13.2, dot: true },
        { n: bot.thigh, at: 13.5 },
        { n: bot.kneeDot, at: 13.8, dot: true, knee: true },
        { n: bot.shin, at: 14.1 },
        { n: bot.ankleDot, at: 14.4, dot: true }
      ];

      return {
        el: s,
        draw: function (st, t) {
          var fade = E.fade;
          bar.set(['done', 'done', 'done', 'cur'], t);

          /* 机器人：16.6 前站在 X=110；之后抬腿、落脚、跟步，停在 X=150 */
          var m = motion(st);
          bot.legs(m.near, m.far);
          bot.place({ x: m.x, dy: m.dy });

          /* 0.6–4.2：命令小圆点按 st 走 3 趟；到达 1.8 / 3.0 / 4.2 时膝盖闪 warn */
          var trip = (st - TRIP0) / TRIP, kt = Math.floor(trip);
          if (kt >= 0 && kt < TRIPS) {
            var u = trip - kt, p = cmdAt(u);
            dot.setAttribute('cx', p[0].toFixed(2)); dot.setAttribute('cy', p[1].toFixed(2));
            E.op(dot, Math.min(1, u / 0.06, (1 - u) / 0.04));
          } else E.op(dot, 0);
          E.op(cmdLabel, E.win(st, TRIP0, TRIP0 + TRIP * TRIPS, 0.3));
          E.drawOn(route, E.clamp((st - TRIP0) / TRIP, 0, 1));
          E.op(routeG, 0.5 * (1 - fade(st, TRIP0 + TRIP * TRIPS, 0.3)));
          var kf = 0;
          for (var i = 1; i <= TRIPS; i++) kf = Math.max(kf, bump(st, TRIP0 + TRIP * i, 0.45));

          /* 13.2–15.0：近腿髋 → 大腿 → 膝 → 小腿 → 踝依次变 warn 并发光；13.4「关节 + 骨架」 */
          legParts.forEach(function (lp) {
            var c = fade(st, lp.at, 0.4), f = lp.knee ? Math.max(c, kf) : c;
            if (lp.dot) lp.n.style.fill = mix(E.WARN, f, E.INK); else lp.n.style.stroke = mix(E.WARN, f, E.INK);
            E.glow(lp.n, E.WARN, f > 0.5);
          });
          scaleOr(bot.kneeDot, P.knee[0], P.knee[1], 1 + 0.6 * kf);
          rise(jointTxt, fade(st, 13.4, 0.4));   // 等放大镜缩回膝盖（13.4）再出，免得缩小的圆扫过字

          /* 4.5 放大镜从膝盖弹出（0.4 s）；12.9–13.4 缩回膝盖并淡出 */
          var grow = K.seg(st, 4.5, 4.9), shrink = K.ease(K.seg(st, 12.9, 13.4));
          var ms = backOut(grow) * (1 - shrink);
          if (ms > 0.001) { scaleOr(mag, KNEE[0], KNEE[1], ms); E.op(mag, Math.min(1, grow * 3) * (1 - shrink)); }
          else { E.scaleAt(mag, KNEE[0], KNEE[1], 0.001); E.op(mag, 0); }
          E.op(magTitle, fade(st, 4.7, 0.3));
          // 电机和小齿轮同速（顺时针），大齿轮反向、× 齿数比；大齿轮多转半个齿距（10°）让齿交错
          var deg = 360 * MOTOR_RPS * t;
          spokes.setAttribute('transform', 'rotate(' + (deg % 360).toFixed(2) + ' ' + MOTOR[0] + ' ' + MOTOR[1] + ')');
          small.spin(deg);
          big.spin(10 - deg * GEAR_RATIO);
          // 6.7「电机」；8.8 齿轮 + 「减速器」
          rise(motorName, fade(st, 6.7, 0.3));
          var gp = fade(st, 8.8, 0.4);
          E.pop(small.outer, PINION[0], PINION[1], gp, 0.3);
          E.pop(big.outer, GEAR[0], GEAR[1], fade(st, 8.95, 0.4), 0.3);
          rise(gearName, fade(st, 9.1, 0.3));
          // 圆外：8.2「快 · 劲小」，11.8「慢 · 劲大」，12.9 跟着淡出
          var out = 1 - fade(st, 12.9, 0.25);   // 比放大镜先走，缩回的圆周不扫过这两行字
          rise(fastTxt, fade(st, 8.2, 0.3)); E.op(fastTxt, fade(st, 8.2, 0.3) * out);
          rise(slowTxt, fade(st, 11.8, 0.3)); E.op(slowTxt, fade(st, 11.8, 0.3) * out);

          /* 18.1 踩进脚印：脚印 violet → warn 实心，落点冒出 3 根短冲击线 */
          var land = fade(st, 18.1, 0.15);
          print.style.fill = mix(E.WARN, land, E.VIOLET);
          E.glow(print, E.WARN, land > 0.5);
          var hp = K.ease(K.seg(st, 18.1, 18.3)), ho = 1 - fade(st, 18.35, 0.25);
          hits.forEach(function (h) {
            var a0 = [h.b[0] + h.d[0] * 3 * hp, h.b[1] + h.d[1] * 3 * hp];
            put(h.l, a0, [a0[0] + h.d[0] * 10 * hp, a0[1] + h.d[1] * 10 * hp]);
            E.op(h.l, hp > 0 ? ho : 0);
          });
        }
      };
    }
  };
})();
