// 001 第 2 幕：第一步：感知。四步条 + 客厅 + 机器人站在 X=110：
// 门口插旗、门上脉冲 → 纸箱闪警示 → 四步条「感知」亮起 → 摄像头 ≈ 眼睛（视线锥扫到纸箱）→
// 小盒子 IMU ≈ 内耳（身子轻晃）→ 关节角度传感器（髋 / 膝 / 踝角度弧）→ 标签退后，总结卡：✓ 自己在哪 ✓ 世界长什么样。
(function () {
  var E = window.EP001, K = E.K;
  var X = 110, FOOT = [110, 388];   // 机器人站位；轻晃以脚底为轴
  var TAU = Math.PI * 2, RAD = Math.PI / 180;

  function mix(c, p, base) {
    return p <= 0 ? base : p >= 1 ? c : 'color-mix(in srgb, ' + c + ' ' + (p * 100).toFixed(1) + '%, ' + base + ')';
  }
  // 文字描一圈底色，压在刻度 / 引线上时不糊
  function halo(n) { n.style.paintOrder = 'stroke'; n.style.stroke = E.S1; n.style.strokeWidth = '5px'; n.style.strokeLinejoin = 'round'; return n; }
  function line(parent, color, w) {
    var l = E.el('line', { 'stroke-width': w, 'stroke-linecap': 'round' }, parent);
    K.paint(l, null, color);
    return l;
  }
  function put(l, a, b) {
    l.setAttribute('x1', a[0].toFixed(1)); l.setAttribute('y1', a[1].toFixed(1));
    l.setAttribute('x2', b[0].toFixed(1)); l.setAttribute('y2', b[1].toFixed(1));
  }
  // 画面坐标里的点跟着机器人绕脚底转 deg 度（与 robot.place 的 rot 同向：正值顺时针 = 前倾）
  function rotPt(p, deg) {
    if (!deg) return p;
    var a = deg * RAD, dx = p[0] - FOOT[0], dy = p[1] - FOOT[1];
    return [FOOT[0] + dx * Math.cos(a) - dy * Math.sin(a), FOOT[1] + dx * Math.sin(a) + dy * Math.cos(a)];
  }
  // 角度弧：角度从竖直向下量、朝 +x 为正（同 robot），a0 → a1 递增
  function arcD(c, r, a0, a1) {
    var p0 = [c[0] + r * Math.sin(a0 * RAD), c[1] + r * Math.cos(a0 * RAD)];
    var p1 = [c[0] + r * Math.sin(a1 * RAD), c[1] + r * Math.cos(a1 * RAD)];
    return 'M ' + p0[0].toFixed(2) + ' ' + p0[1].toFixed(2) + ' A ' + r + ' ' + r + ' 0 ' + (a1 - a0 > 180 ? 1 : 0) + ' 0 ' +
      p1[0].toFixed(2) + ' ' + p1[1].toFixed(2);
  }
  // 同一段弧连回圆心的扇形（量角器的样子：一边贴着肢体，一边是扫动的指针）
  function wedgeD(c, r, a0, a1) {
    return 'M ' + c[0].toFixed(2) + ' ' + c[1].toFixed(2) + ' L ' + arcD(c, r, a0, a1).slice(2) + ' Z';
  }
  // 一次性闪 n 下（每下 per 秒），返回 0…1；其余时刻 0
  function flash(st, a, n, per) {
    var u = (st - a) / per;
    return u < 0 || u >= n ? 0 : Math.sin(Math.PI * (u - Math.floor(u)));
  }
  // a 起 d 秒内鼓一下：0 → 1 → 0
  function bump(st, a, d) { var u = K.seg(st, a, a + d); return u > 0 && u < 1 ? Math.sin(Math.PI * u) : 0; }
  function scaleOr(n, cx, cy, s) { if (s > 1.0005) E.scaleAt(n, cx, cy, s); else n.removeAttribute('transform'); }

  E.scenes[1] = {
    title: '第一步：感知',
    dur: 31.4,
    cues: [
      { at: 0, s: '换成机器人：穿过客厅走到门口，路上有个纸箱' },
      { at: 7.1, s: '第一步 **感知**：周围什么样？自己什么状态？' },
      { at: 11.2, s: '摄像头 ≈ 眼睛：看到前面有个纸箱' },
      { at: 14.9, s: '测歪斜和转动的小盒子（行话 IMU）≈ 内耳' },
      { at: 20.3, s: '关节角度传感器（行话：编码器）≈ 闭眼也知道弯了多少' },
      { at: 25.7, s: '传感器合起来回答：我在哪？世界长什么样？' }
    ],
    build: function (K) {
      var s = K.sceneSvg('人形机器人站在客厅一头，门口插着小旗，路中间有个纸箱；摄像头看见纸箱，身体里的小盒子感觉歪没歪，关节传感器知道弯了多少——这就是感知');
      // 机器人先建出来：视线锥要从 G1 头上摄像头（d435）的实际位置发出；它的组稍后挪回原来的绘制顺序（客厅、视线锥之后）
      var robot = E.robot(s, X);
      var P = robot.pts(), CAM = [X + P.cam[0] + 5, P.cam[1]];

      /* ── 客厅；门口的脉冲圈和小旗夹在门与纸箱之间 ── */
      var room = E.room(s);
      var doorFx = E.el('g', {}, s);
      s.insertBefore(doorFx, room.boxG);
      var pulse = K.paint(E.el('circle', { cx: 741, cy: 268, r: 30, fill: 'none', 'stroke-width': 3 }, doorFx), null, E.ACC);
      var pole = line(doorFx, E.ACC, 3);
      var flag = K.paint(E.el('path', { d: 'M 770 120 L 790 127 L 770 134 Z' }, doorFx), E.ACC);
      var boxName = halo(E.txt(s, 413, 412, '纸箱', 18, E.INK2));

      /* ── 视线锥：从摄像头 CAM 发出的扇形（向下 12°–42°，半径 360），越远越淡，远端是圆弧不是硬切边；
         摄像头装在头上，锥只跟着身子绕脚底晃，自己不摆。裁在地面以上 ── */
      var defs = E.el('defs', {}, s);
      var clip = E.el('clipPath', { id: 'ep001-s2-cone-clip' }, defs);
      E.el('rect', { x: 0, y: 0, width: 800, height: 389 }, clip);
      var grad = E.el('radialGradient', { id: 'ep001-s2-cone-grad', gradientUnits: 'userSpaceOnUse', cx: CAM[0], cy: CAM[1], r: 360 }, defs);
      [[0, 0.34], [0.55, 0.16], [1, 0]].forEach(function (g) {
        var st = E.el('stop', { offset: g[0] }, grad); st.style.stopColor = 'var(--demo-good)'; st.style.stopOpacity = g[1];
      });
      var coneClip = E.el('g', { 'clip-path': 'url(#ep001-s2-cone-clip)' }, s);
      var coneOuter = E.el('g', {}, coneClip), coneG = E.el('g', {}, coneOuter);
      var R0 = 360, A0 = 12 * Math.PI / 180, A1 = 42 * Math.PI / 180;
      var cone = E.el('path', { d: 'M' + CAM[0] + ',' + CAM[1] + ' L' + (CAM[0] + R0 * Math.cos(A0)).toFixed(1) + ',' + (CAM[1] + R0 * Math.sin(A0)).toFixed(1) +
        ' A' + R0 + ',' + R0 + ' 0 0 1 ' + (CAM[0] + R0 * Math.cos(A1)).toFixed(1) + ',' + (CAM[1] + R0 * Math.sin(A1)).toFixed(1) + ' Z', fill: 'url(#ep001-s2-cone-grad)' }, coneG);
      var seen = E.txt(s, 413, 330, '前方：纸箱', 20, E.GOOD);

      /* ── 机器人；角度弧、IMU 高亮、IMU 波纹挂在它的身体组里（本地坐标，髋在 (0,302)） ── */
      s.appendChild(robot.g);
      var arcG = E.el('g', {}, robot.body);
      robot.body.insertBefore(arcG, robot.thigh);   // 垫在腿下面：扇形贴着肢体，腿线压住起始边
      var joints = [
        { dot: robot.hipDot, c: P.hip.slice(), a0: 4, at: 20.3 },
        { dot: robot.kneeDot, c: P.knee.slice(), a0: 0, at: 20.7 },
        { dot: robot.ankleDot, c: P.ankle.slice(), a0: 90, at: 21.1 }
      ];
      // r14 的弧在屏幕上只有十几像素：底下垫一块半透明扇形，读起来才像「量角度」
      joints.forEach(function (j) {
        j.wedge = K.paint(E.el('path', { 'stroke-width': 2, 'stroke-linejoin': 'round' }, arcG), E.GOOD, E.GOOD);
        j.wedge.style.fillOpacity = 0.4;
        j.arc = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3.5, 'stroke-linecap': 'round' }, arcG), null, E.GOOD);
      });
      // 近侧上臂正好压在小盒子上：亮起时在手臂上面再画一个同位置的小盒子
      var imuHi = K.paint(E.el('rect', { x: P.imu[0] - 5, y: P.imu[1] - 5, width: 10, height: 10, rx: 2 }, robot.body), E.GOOD);
      var imuRing = K.paint(E.el('circle', { cx: P.imu[0], cy: P.imu[1], r: 6, fill: 'none', 'stroke-width': 2.5 }, robot.body), null, E.GOOD);

      /* ── 三个标签 + 引线（引线终点跟着机器人晃） ── */
      function label(y, w, str, from, to, at) {
        var g = E.el('g', {}, s);
        var lead = line(g, E.GOOD, 2);
        var tg = E.tag(g, 200, y, w, 36, str, E.GOOD);
        tg.text.setAttribute('y', y + 24);
        return { g: g, lead: lead, tag: tg, cx: 200 + w / 2, cy: y + 18, from: from, to: to, at: at };
      }
      var L = [
        label(74, 300, '摄像头 ≈ 眼睛', [214, 110], [CAM[0] + 2, CAM[1] - 4], 11.2),
        label(118, 400, '', [214, 154], [X + P.imu[0] + 3, P.imu[1] - 4], 14.9),
        label(162, 400, '关节角度传感器 ≈ 闭眼也知道弯了多少', [214, 198], [X + P.knee[0] + 6, P.knee[1] - 2], 20.3)
      ];
      var l2 = L[1].tag.text;
      l2.appendChild(document.createTextNode('测歪斜和转动的小盒子（IMU）≈ '));
      var ear = E.el('tspan', {}, l2);
      ear.textContent = '内耳';

      /* ── 四步条 ── */
      var bar = E.stepBar(s);

      /* ── 收尾：灰字 + 总结卡 ── */
      var more = halo(E.txt(s, 214, 226, '有的还配：激光雷达、脚底测力', 18, E.MUT, 'start'));
      // 边走边建图、定位自己也算感知（路线 §L7.1 把 SLAM 归在感知层）；规划只拿这张地图找路
      var slam = halo(E.txt(s, 214, 250, '边走边建图定位，行话 SLAM', 18, E.MUT, 'start'));
      var card = E.el('g', {}, s);
      var cardRect = K.paint(E.el('rect', { x: 480, y: 250, width: 220, height: 86, rx: 12, 'stroke-width': 2.5 }, card), E.S1, E.GOOD);
      var checks = [E.check(card, 500, 269, 18, E.GOOD), E.check(card, 500, 305, 18, E.GOOD)];
      E.txt(card, 528, 284, '自己在哪', 22, E.INK, 'start');
      E.txt(card, 528, 320, '世界长什么样', 22, E.INK, 'start');

      return {
        el: s,
        draw: function (st, t) {
          var fade = E.fade;

          /* 0–0.6：四步条、客厅、机器人淡入 */
          var intro = fade(st, 0, 0.6);
          [bar.g, room.back, room.boxG, robot.g].forEach(function (n) { E.op(n, intro); });

          /* 7.8：「感知」变为当前步（切换时鼓一下） */
          bar.set([st >= 7.8 ? 'cur' : 'off', 'off', 'off', 'off'], t);
          var b0 = bump(st, 7.8, 0.5);
          scaleOr(bar.rects[0], 115, 32, 1 + 0.08 * b0);
          scaleOr(bar.labels[0], 115, 32, 1 + 0.08 * b0);

          /* 1.9：门右上插旗；门中心脉冲圈按 t 每 1.5 s 一次，到 7.1 */
          var pp = fade(st, 1.9, 0.25);
          put(pole, [770, 148], [770, 148 - 28 * pp]);
          E.op(pole, pp > 0 ? 1 : 0);
          E.pop(flag, 770, 127, fade(st, 2.1, 0.25), 0.3);
          var env = E.win(st, 1.9, 6.8, 0.3), ph = ((t / 1.5) % 1 + 1) % 1;
          pulse.setAttribute('r', (30 + 30 * ph).toFixed(2));
          E.op(pulse, env * (1 - ph));

          /* 纸箱：5.4 描边闪两次 warn + 地面下写「纸箱」；13.2 描边变 good */
          var fw = flash(st, 5.4, 2, 0.45), gb = fade(st, 13.2, 0.3);
          room.box.style.stroke = gb > 0 ? mix(E.GOOD, gb, E.MUT) : mix(E.WARN, fw, E.MUT);
          room.box.style.strokeWidth = 3 + fw;
          E.glow(room.box, E.WARN, fw > 0.4);
          E.op(boxName, fade(st, 5.4, 0.4));

          /* 15.2–16.8：以脚底为轴左右轻晃 ±3°，再回正 */
          var su = K.seg(st, 15.2, 16.8), rot = su > 0 && su < 1 ? -3 * Math.sin(TAU * su) : 0;
          robot.place({ rot: rot });

          /* 11.2：摄像头变 good，视线锥从摄像头展开；25.7 变淡 */
          var pc = fade(st, 11.2, 0.4), dim = fade(st, 25.7, 0.4);
          var cp = fade(st, 11.2, 0.25);
          robot.cam.style.fill = mix(E.GOOD, cp, E.INK2);
          E.glow(robot.cam, E.GOOD, cp > 0.5);
          coneOuter.setAttribute('transform', rot ? 'rotate(' + rot.toFixed(3) + ' ' + FOOT[0] + ' ' + FOOT[1] + ')' : '');
          var cs = 0.3 + 0.7 * pc;   // 打开时从摄像头展开一次，之后不动
          coneG.setAttribute('transform', 'translate(' + CAM[0] + ' ' + CAM[1] + ') scale(' + cs.toFixed(4) + ') translate(' + (-CAM[0]) + ' ' + (-CAM[1]) + ')');
          E.op(cone, pc * (1 - 0.55 * dim));

          /* 13.2：箱上「前方：纸箱」 */
          var sp = fade(st, 13.2, 0.35);
          E.op(seen, sp);
          seen.setAttribute('transform', sp < 1 ? 'translate(0 ' + (6 * (1 - sp)).toFixed(2) + ')' : '');

          /* 14.9：小盒子变 good，脉冲 3 次（放大 + 波纹） */
          var ip = fade(st, 14.9, 0.25), ib = flash(st, 14.9, 3, 0.5);
          robot.imu.style.fill = mix(E.GOOD, ip, E.MUT);
          E.op(imuHi, ip);
          scaleOr(imuHi, P.imu[0], P.imu[1], 1 + 0.5 * ib);
          E.glow(imuHi, E.GOOD, ip > 0.5);
          var iu = (st - 14.9) / 0.5;
          if (iu >= 0 && iu < 3) {
            var fr = iu - Math.floor(iu);
            imuRing.setAttribute('r', (6 + 14 * fr).toFixed(2));
            E.op(imuRing, 1 - fr);
          } else E.op(imuRing, 0);

          /* 20.3 / 20.7 / 21.1：髋、膝、踝依次变 good，角度弧按 t 来回扫 */
          joints.forEach(function (j, k) {
            var p = fade(st, j.at, 0.25);
            j.dot.style.fill = mix(E.GOOD, p, E.INK);
            scaleOr(j.dot, j.c[0], j.c[1], 1 + 0.5 * bump(st, j.at, 0.35));
            var grow = fade(st, j.at, 0.4);
            var span = grow * (20 + 50 * (0.5 - 0.5 * Math.cos(TAU * t / 1.8 + 2.1 * k)));
            if (span > 0.5) {
              j.arc.setAttribute('d', arcD(j.c, 14, j.a0, j.a0 + span));
              j.wedge.setAttribute('d', wedgeD(j.c, 14, j.a0, j.a0 + span));
            }
            E.op(j.arc, span > 0.5 ? 1 : 0);
            E.op(j.wedge, span > 0.5 ? 1 : 0);
          });

          /* 标签：弹出，引线从标签画到部件；25.7 一起降到 0.4 */
          L.forEach(function (l) {
            E.pop(l.tag.g, l.cx, l.cy, fade(st, l.at, 0.35), 0.85);
            var lp = K.ease(K.seg(st, l.at + 0.15, l.at + 0.5)), end = rotPt(l.to, rot);
            put(l.lead, l.from, [l.from[0] + (end[0] - l.from[0]) * lp, l.from[1] + (end[1] - l.from[1]) * lp]);
            E.op(l.lead, lp > 0 ? 1 : 0);
            E.op(l.g, 1 - 0.6 * dim);
          });
          /* 16.6：「内耳」变 good 粗体 */
          var ep = fade(st, 16.6, 0.3);
          ear.style.fill = mix(E.GOOD, ep, E.INK);
          ear.style.fontWeight = ep > 0 ? 700 : '';

          /* 25.7：灰字；27.7：总结卡弹出、两个勾画出；29.2：描边闪一次 */
          E.op(more, fade(st, 25.7, 0.5)); E.op(slam, fade(st, 25.7, 0.5));
          E.pop(card, 590, 293, fade(st, 27.7, 0.35), 0.85);
          E.drawOn(checks[0], K.seg(st, 27.85, 28.15));
          E.drawOn(checks[1], K.seg(st, 28.0, 28.3));
          var cb = bump(st, 29.2, 0.6);
          cardRect.style.strokeWidth = 2.5 + 2.5 * cb;
          E.glow(cardRect, E.GOOD, cb > 0.3);
        }
      };
    }
  };
})();
