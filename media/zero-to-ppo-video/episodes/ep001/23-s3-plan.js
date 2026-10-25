// 001 第 3 幕：第二步：规划。四步条「规划」为当前步；客厅和机器人（X=110）同 S2，右上角是小地图（俯视）：
// 扫描线驱散迷雾、白点闪（边画地图边定位，SLAM）→ 直线虚线压过纸箱、打红叉 → 紫色绕行路线画出 →
// 小地图沿路线冒出 6 个小脚印、侧视地面出现「下一脚踩这儿」→ 腿变虚线、关节旁冒「?」：关节怎么用力，规划不管。
(function () {
  var E = window.EP001, K = E.K;
  var X = 110;
  var STEP_U = 45 / 580;          // 脚印间距：半个步态周期（poseWalk 每 90px 一个周期）换算成路线比例
  var FOOT_U0 = 0.117;            // 第一个脚印 u = (178 − 110) / 580，对上侧视地面的脚印
  var PULSE = 1.2;                // 脉冲周期（真实时间 t）

  // 小地图脚印：路线 u 处、左右偏 side·5px，绕椭圆自己的中心转到路线方向。
  //（E.miniMap().footprint 绕路线点 p 旋转，已经沿法线偏好的中心会再被转一个 ang，弯道上脚印偏离法线，所以这里自己画）
  function footprint(mm, parent, u, side, rx, fill) {
    var p = mm.at(u), q = mm.at(Math.min(1, u + 0.01)), a = Math.atan2(q[1] - p[1], q[0] - p[0]);
    var cx = p[0] - side * 5 * Math.sin(a), cy = p[1] + side * 5 * Math.cos(a);
    var e = E.el('ellipse', { cx: cx.toFixed(1), cy: cy.toFixed(1), rx: rx, ry: 2.5,
      transform: 'rotate(' + (a * 180 / Math.PI).toFixed(1) + ' ' + cx.toFixed(1) + ' ' + cy.toFixed(1) + ')' }, parent);
    if (fill) K.paint(e, E.VIOLET); else { e.setAttribute('fill', 'none'); K.paint(e, null, E.VIOLET); }
    return e;
  }
  // 脉冲：外圈椭圆从 (rx, ry) 长到 (rx + grow[0], ry + grow[1]) 同时淡出；k = 0…1 为加强程度
  function ring(node, rx, ry, grow, t, k, alpha) {
    var ph = ((t / PULSE) % 1 + 1) % 1, g = 1 + 0.9 * k;
    node.setAttribute('rx', (rx + grow[0] * g * ph).toFixed(2));
    node.setAttribute('ry', (ry + grow[1] * g * ph).toFixed(2));
    node.style.strokeWidth = (2 + 1.5 * k).toFixed(2);
    E.op(node, (alpha * (1 - ph) * (0.8 + 0.2 * k)).toFixed(3));
  }
  // 填充脚印跟着脉冲微微胀大（只胀不缩，免得比后面的小脚印还小）
  function breathe(t, k) { return 1 + (0.1 + 0.15 * k) * (0.5 - 0.5 * Math.cos(2 * Math.PI * t / PULSE)); }
  // 16.0 加强时光晕随 k 渐亮（E.glow 只有开 / 关，会在 k 过 0.5 时跳一下）
  function softGlow(node, k) { node.style.filter = k > 0.01 ? 'drop-shadow(0 0 ' + (6 * k).toFixed(2) + 'px ' + E.VIOLET + ')' : ''; }

  E.scenes[2] = {
    title: '第二步：规划',
    dur: 22.1,
    cues: [
      { at: 0, s: '第二步 **规划**：决定去哪、走哪条路' },
      { at: 2.6, s: '拿着感知交来的地图：自己在这，门在那' },
      { at: 9.1, s: '直走会撞纸箱 → 改从旁边绕过去' },
      { at: 13.2, s: '还定好「下一脚踩这儿」' },
      { at: 16, s: '这里规划只管到下一脚；关节怎么用力，它不管' }
    ],
    build: function (K) {
      var s = K.sceneSvg('规划：机器人拿着感知交来的地图，发现直走会撞纸箱，改从旁边绕到门口，定好下一脚踩哪；关节怎么用力不归规划管');

      /* ── 四步条 + 客厅 + 机器人（同 S2） ── */
      var bar = E.stepBar(s);
      var room = E.room(s);
      var bot = E.robot(s, X);
      s.appendChild(room.boxG);   // 纸箱在最上层（本幕机器人不走，只为与 S2 / S6 的绘制顺序一致）

      /* ── 侧视：下一脚 ── */
      var stepG = E.el('g', {}, s);
      var stepRing = K.paint(E.el('ellipse', { cx: 178, cy: 388, rx: 16, ry: 4, fill: 'none' }, stepG), null, E.VIOLET);
      var stepFoot = K.paint(E.el('ellipse', { cx: 178, cy: 388, rx: 16, ry: 4 }, stepG), E.VIOLET);
      var stepLabel = E.txt(s, 225, 362, '下一脚踩这儿', 20, E.VIOLET);

      /* ── 侧视：关节旁的「?」和旁注 ── */
      var qs = [304, 346, 374].map(function (y) { return E.txt(s, 136, y, '?', 22, E.MUT, 'start', 700); });   // 最下面那个别贴着「下一脚踩这儿」的椭圆
      var noCare = E.txt(s, 290, 292, '关节怎么用力？规划不管', 20, E.MUT);

      /* ── 小地图：整组从右侧滑入 ── */
      var mapG = E.el('g', {}, s);
      var mm = E.miniMap(mapG);
      mm.setReveal(442);
      // 扫描线：在迷雾之上、门之下
      var scan = E.el('line', { x1: 442, y1: 100, x2: 442, y2: 188, 'stroke-width': 3, 'stroke-linecap': 'round' });
      K.paint(scan, null, E.VIOLET);
      mm.g.insertBefore(scan, mm.door);
      // 直走：白点 → 门的虚线，压过纸箱的一段变红 + 红叉（画在路线之下）
      var rejG = E.el('g', {});
      mm.g.insertBefore(rejG, mm.route);
      var direct = E.el('path', { d: 'M 466 152 L 766 152', fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '6 4' }, rejG);
      K.paint(direct, null, E.INK2);
      // 与主虚线同相位：x = 585 处离起点 119
      var hit = E.el('path', { d: 'M 585 152 L 609 152', fill: 'none', 'stroke-width': 3.6, 'stroke-dasharray': '6 4', 'stroke-dashoffset': 119 }, rejG);
      K.paint(hit, null, E.BAD);
      var crossG = E.el('g', {}, rejG);
      E.cross(crossG, 597, 152, 8, E.BAD, 3);
      // 脚印（路线上、白点下）：draw 里第一次挂到页面后再建，路线长度要量
      var prints = null;
      // 文字在最上层
      var here = E.txt(mm.g, 466, 186, '我在这', 20, E.INK);   // 地图整体缩到 0.91，字补到 20
      var unseen = E.txt(mm.g, 709, 150, '还没看到', 20, E.MUT);
      var slam = E.txt(mapG, 382, 208, '地图由感知交来', 18, E.MUT, 'start');

      function makePrints() {
        prints = [];
        for (var i = 0; i < 6; i++) {
          var u = FOOT_U0 + i * STEP_U, side = i % 2 ? -1 : 1, big = i === 0;
          prints.push({
            ring: big ? footprint(mm, mm.prints, u, side, 6, false) : null,
            e: footprint(mm, mm.prints, u, side, big ? 6 : 4.5, true),
            rx: big ? 6 : 4.5, at: 14.1 + 0.15 * i
          });
        }
      }

      return {
        el: s,
        draw: function (st, t) {
          var fade = E.fade;
          bar.set(['done', 'cur', 'off', 'off'], t);

          /* 0.3–0.8 小地图滑入；3.0–8.0 扫描线 442 → 650，迷雾跟着散开 */
          var mp = fade(st, 0.3, 0.5);
          // 从左边滑入（从右边进来会压过门上的「门口」）；滑入前（完全透明）留在原位：fitStory 会把透明但挪开 40 的面板
          // 也算进取景范围，白白缩小整幕
          mapG.setAttribute('transform', 'translate(' + (mp > 0 ? -40 * (1 - mp) : 0).toFixed(2) + ' 0)');
          E.op(mapG, mp);
          var rx = 442 + 208 * K.seg(st, 3.0, 8.0);
          mm.setReveal(rx);
          E.op(mm.box, rx > 585 ? 1 : 0);   // 迷雾只有 0.92 不透明，纸箱等扫描线到了再出现
          scan.setAttribute('x1', rx.toFixed(2)); scan.setAttribute('x2', rx.toFixed(2));
          E.op(scan, E.win(st, 3.0, 8.0, 0.25));
          E.glow(scan, E.VIOLET, true);

          /* 白点 2.6–9.1 按 t 闪；5.4「我在这」+ SLAM；8.0「还没看到」（9.1 虚线要从它中间穿过，先退场） */
          var blink = E.win(st, 2.6, 9.1, 0.3);
          E.op(mm.dotEl, (1 - 0.7 * blink * (0.5 - 0.5 * Math.cos(2 * Math.PI * t / 0.9))).toFixed(3));
          E.pop(here, 466, 180, fade(st, 5.4, 0.35), 0.85);
          E.op(slam, fade(st, 5.4, 0.4));
          E.op(unseen, E.win(st, 8.0, 9.1, 0.3));

          /* 9.1–9.9 直线虚线；10.0 压过纸箱的一段变红 + 红叉，侧视纸箱描边闪两次红；
             绕行路线画好后整组退到 0.4，13.2 淡出（脚印要沿路线放，第一个正好压在这条虚线上；S6 开场也没有它） */
          E.drawOn(direct, K.seg(st, 9.1, 9.9));
          var hp = fade(st, 10.0, 0.2);
          E.op(hit, hp);
          E.pop(crossG, 597, 152, hp, 0.4);
          E.op(rejG, (1 - 0.6 * fade(st, 12.2, 0.4)) * (1 - fade(st, 13.2, 0.4)));
          var fl = st >= 10.0 && st < 10.9 ? 0.5 - 0.5 * Math.cos(2 * Math.PI * (st - 10.0) / 0.45) : 0;
          room.box.style.stroke = fl > 0 ? E.tint(E.BAD, (100 * fl).toFixed(1), E.MUT) : E.MUT;
          room.tape.style.stroke = room.box.style.stroke;

          /* 10.8–12.2 紫色绕行路线画出 */
          E.drawOn(mm.route, K.ease(K.seg(st, 10.8, 12.2)));

          /* 14.1 起 6 个小脚印依次冒出；第一个和侧视脚印按 t 脉冲，16.0 加强 */
          if (!prints && mm.route.isConnected) makePrints();
          var k = fade(st, 16.0, 0.4);
          if (prints) prints.forEach(function (f, i) {
            var p = fade(st, f.at, 0.15);
            var sc = (0.4 + 0.6 * p) * (i === 0 ? breathe(t, k) : 1);
            f.e.setAttribute('rx', (f.rx * sc).toFixed(2));
            f.e.setAttribute('ry', (2.5 * sc).toFixed(2));
            E.op(f.e, p);
            if (f.ring) { ring(f.ring, 6, 2.5, [6, 3.5], t, k, p); softGlow(f.e, k); }
          });
          var sp = fade(st, 14.1, 0.3), sb = breathe(t, k) * (0.6 + 0.4 * sp);
          stepFoot.setAttribute('rx', (16 * sb).toFixed(2));
          stepFoot.setAttribute('ry', (4 * sb).toFixed(2));
          E.op(stepFoot, sp);
          ring(stepRing, 16, 4, [14, 5], t, k, sp);
          softGlow(stepFoot, k);
          E.op(stepLabel, fade(st, 14.1, 0.4));

          /* 19.3 腿变虚线、降到 0.4；髋膝踝旁冒「?」；旁注 */
          var lp = fade(st, 19.3, 0.3);
          if (st >= 19.3) {
            // 虚线空隙 0 → 4 跟着淡出一起长出来，不在 19.3 一下子从实线跳成虚线
            bot.legStyle(null, '5 ' + (4 * lp).toFixed(2), 1 - 0.6 * lp);
            // 粗线配圆头时 5 4 的虚线会被线帽填满、看着还是实线：虚线期间改平头
            bot.body.querySelectorAll('line').forEach(function (l) {
              if (l.getAttribute('stroke-dasharray')) l.setAttribute('stroke-linecap', 'butt');
            });
          } else {
            bot.legStyle(null, '', 1);
            bot.body.querySelectorAll('line').forEach(function (l) { l.setAttribute('stroke-linecap', 'round'); });
          }
          qs.forEach(function (q, i) { E.pop(q, 142, [296, 338, 378][i], lp, 0.6); });
          E.op(noCare, fade(st, 19.3, 0.4));
        }
      };
    }
  };
})();
