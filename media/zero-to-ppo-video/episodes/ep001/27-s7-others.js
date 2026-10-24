// 001 第 7 幕：换台机器人，还是这四步。不画四步条，换新布局：
// 左侧大面板是扫地机（Create 2 的 URDF，俯视）遇到拖鞋：感知（保险杠闪 + 三道弧扩散）→ 规划（紫色虚线从拖鞋上方绕过）→
// 控制（左右轮速度条：往左拐右轮快、往右拐左轮快）→ 执行（沿紫线绕过去，车身随切线转、轮子条纹滚动），四行步骤标签逐句出现；
// 右侧三张小卡（机械臂 / 自动驾驶汽车 / 人形）按念到的名字弹出，每张卡带一个迷你四步环；
// 15.6「全是」起，三张卡的四个点和左面板四行标签一起按真实时间 t 轮流点亮。
(function () {
  var E = window.EP001, K = E.K;
  var R = Math.PI / 180;

  var VAC = [150, 180];                                   // 扫地机起点（圆心）
  // 紫色虚线（规划路线），从保险杠前沿 (190,180) 出发。扫地机圆心沿「(150,180) 直行到 (190,180) + 这条线」走。
  // 规格是 M190,180 C232,180 248,124 300,124 S348,160 356,180：按它走，0.8 倍的机身（手画版半径 37，两侧轮子伸到 42；Create 2 俯视半径约 38）
  // 上坡时右轮蹭到拖鞋左上角、下坡起机身压住拖鞋右半（终点 (356,180) 时压进 13px）——拖鞋右端 x≈330 到面板右缘 400
  // 只有 70px，放不下直径 77 的机身，回不到 y=180。改为：早点抬头、峰顶抬到 y=108、停在拖鞋右上方 (350,128)，
  // 全程离拖鞋 ≥ 4px，机身不出面板。第二段的控制点让航向一路往右拐到 60°（不在终点前回勾）。
  var PLAN = [[[190, 180], [214, 180], [248, 108], [300, 108]], [[300, 108], [326, 108], [342, 114.1], [350, 128]]];
  var MOVE0 = 9.6, MOVE1 = 11.4;
  var ROW_Y = [296, 326, 356, 386], ROW_AT = [3.5, 6.1, 7.0, 9.6];
  var ROWS = ['感知：看到拖鞋', '规划：决定绕开', '控制：算两个轮子各转多快', '执行：轮子转起来'];
  var CARD_Y = [20, 146, 272], CARD_AT = [12.6, 13.2, 14.5];
  var CARD_NAME = ['机械臂', '自动驾驶汽车', '人形机器人'];
  var DOT_X = [548, 598, 648, 698];
  var CYC = 15.6, SLOT = 0.5;                             // 轮流点亮：每个点 0.5 s
  // 行驶进度：0.25 s 加速、匀速、0.25 s 减速（梯形速度），车身和轮子同起同停
  var RAMP = 0.25 / (MOVE1 - MOVE0), VMAX = 1 / (1 - RAMP);
  function prog(s) {
    if (s <= 0) return 0;
    if (s >= 1) return 1;
    if (s < RAMP) return VMAX * s * s / (2 * RAMP);
    if (s > 1 - RAMP) return 1 - VMAX * (1 - s) * (1 - s) / (2 * RAMP);
    return VMAX * (s - RAMP / 2);
  }
  function speed(s) { return s <= 0 || s >= 1 ? 0 : Math.min(1, s / RAMP, (1 - s) / RAMP); }

  function mix(c, p, base) {
    return p <= 0 ? base : p >= 1 ? c : 'color-mix(in srgb, ' + c + ' ' + (p * 100).toFixed(1) + '%, ' + base + ')';
  }
  // 文字淡入并上浮 6px
  function rise(n, p) { E.op(n, p); n.setAttribute('transform', p < 1 ? 'translate(0 ' + (6 * (1 - p)).toFixed(2) + ')' : ''); }
  function frac(x) { return x - Math.floor(x); }

  /* 扫地机圆心的行驶路线按弧长取点（解析算，不用等节点挂到页面）：直行一段 + PLAN 的两段三次贝塞尔 */
  function bez(c, u) {
    var v = 1 - u, a = v * v * v, b = 3 * v * v * u, d = 3 * v * u * u, e = u * u * u;
    return [a * c[0][0] + b * c[1][0] + d * c[2][0] + e * c[3][0], a * c[0][1] + b * c[1][1] + d * c[2][1] + e * c[3][1]];
  }
  var LUT = (function () {
    var pts = [VAC.slice()], N = 120;
    PLAN.forEach(function (c, k) { for (var i = k ? 1 : 0; i <= N; i++) pts.push(bez(c, i / N)); });
    var len = [0];
    for (var i = 1; i < pts.length; i++) len.push(len[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
    return { pts: pts, len: len };
  })();
  // 路线上比例 s 处：圆心与切线角（度）
  function track(s) {
    var L = LUT.len, n = L.length - 1, want = E.clamp(s, 0, 1) * L[n], i = 1;
    while (i < n && L[i] < want) i++;
    var a = LUT.pts[i - 1], b = LUT.pts[i], k = (want - L[i - 1]) / ((L[i] - L[i - 1]) || 1);
    return { x: a[0] + (b[0] - a[0]) * k, y: a[1] + (b[1] - a[1]) * k, ang: Math.atan2(b[1] - a[1], b[0] - a[0]) / R };
  }
  function planD() {
    return PLAN.map(function (c, k) {
      return (k ? '' : 'M ' + c[0][0] + ' ' + c[0][1] + ' ') + 'C ' + c[1][0] + ' ' + c[1][1] + ' ' + c[2][0] + ' ' + c[2][1] + ' ' + c[3][0] + ' ' + c[3][1];
    }).join(' ');
  }

  // 以 (cx, cy) 为心、半径 r、张角 ±35° 朝 +x 的弧
  function arcD(cx, cy, r) {
    var a = 35 * R, x0 = cx + r * Math.cos(a), y0 = r * Math.sin(a);
    return 'M ' + x0.toFixed(2) + ' ' + (cy - y0).toFixed(2) + ' A ' + r.toFixed(2) + ' ' + r.toFixed(2) + ' 0 0 1 ' + x0.toFixed(2) + ' ' + (cy + y0).toFixed(2);
  }

  /* 轮流点亮：第 k 个点在第 k 格（每格 0.5 s）亮起，前后 0.12 s 交叉淡变；ph < 0 时都不亮 */
  function lit(ph, k) {
    if (ph < 0) return 0;
    var local = ((ph - k) % 4 + 4) % 4;
    if (ph - local < 0) return 0;                         // 这一格还没轮到过（开头不让「执行」先闪一下）
    if (local < 0.24) return local / 0.24;
    if (local < 1) return 1;
    if (local < 1.24) return 1 - (local - 1) / 0.24;
    return 0;
  }

  /* 右侧小卡：卡框 + 图标 + 名字 + 迷你四步环（四点 + 三支短箭头 + 回头虚线） */
  function card(s, i) {
    var y = CARD_Y[i], g = E.el('g', {}, s);
    K.paint(E.el('rect', { x: 420, y: y, width: 360, height: 112, rx: 12, 'stroke-width': 2 }, g), E.S1, E.BORDER);
    var icon = i === 0 ? E.arm(g, 456, y + 102, 0.68) : i === 1 ? E.car(g, 477, y + 92, 0.55) : E.humanoid(g, 470, y + 106, 0.86);
    E.txt(g, 530, y + 44, CARD_NAME[i], 22, E.INK, 'start', 700);
    // 回头虚线带箭头（与四步条的反馈线一样），看得出「执行完又回到感知」；末端切线方向 (548−623, 94−108)
    E.curveArrow(g, 'M 698 ' + (y + 94) + ' Q 623 ' + (y + 108) + ' 548 ' + (y + 94), E.MUT, 2, Math.atan2(-14, -75) / R, { dash: '5 4', head: 8 });
    for (var k = 0; k < 3; k++) E.arrow(g, [[DOT_X[k] + 14, y + 82], [DOT_X[k + 1] - 14, y + 82]], E.INK2, 2, { head: 8 });
    var dots = E.STEPS.map(function (st, k) { return K.paint(E.el('circle', { cx: DOT_X[k], cy: y + 82, r: 10 }, g), st.color); });
    return { g: g, icon: icon, dots: dots, cx: 600, cy: y + 56 };
  }

  E.scenes[6] = {
    title: '换台机器人，还是这四步',
    dur: 18,
    cues: [
      { at: 0, s: '换台机器人，还是这四步' },
      { at: 2.4, s: '扫地机：看到拖鞋 → 决定绕开 → 算两轮各转多快 → 轮子转' },
      { at: 11.5, s: '机械臂、汽车、人形：都是同一个四步循环' }
    ],
    build: function (K) {
      var s = K.sceneSvg('换台机器人还是这四步：扫地机用传感器看到拖鞋、决定从上方绕开、算出左右轮各转多快、轮子转起来绕了过去；机械臂、自动驾驶汽车、人形机器人拆开看也都在转同一个四步圈');

      /* ── 左侧大面板：扫地机 ── */
      var left = E.el('g', {}, s);
      K.paint(E.el('rect', { x: 20, y: 20, width: 380, height: 384, rx: 14, 'stroke-width': 2 }, left), E.S1, E.BORDER);
      E.txt(left, 210, 54, '扫地机', 24, E.INK, 'middle', 700);

      // 拖鞋 + 「拖鞋」
      var slipper = E.el('g', {}, s);
      K.paint(E.el('path', { d: 'M278,168 Q300,160 324,168 Q334,180 324,192 Q300,200 278,192 Q270,180 278,168 Z', 'stroke-width': 2.5 }, slipper), E.S2, E.INK2);
      E.txt(slipper, 301, 222, '拖鞋', 18, E.INK2);

      // 规划路线（紫色虚线），画在扫地机下面
      var plan = E.el('path', { d: planD(), fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '8 6', 'stroke-linecap': 'round' }, s);
      K.paint(plan, null, E.VIOLET);

      // 感知：三道 good 弧，以 (190,180) 为心向拖鞋扩散
      var waves = [0, 1, 2].map(function () {
        var p = E.el('path', { fill: 'none', 'stroke-width': 3, 'stroke-linecap': 'round' }, s);
        return K.paint(p, null, E.GOOD);
      });

      // 扫地机：外层 translate + rotate 跟着路线走，俯视的 Create 2 车头朝 +x。保险杠（前半圈的弧）平时不画，感知时闪；
      // 两个驱动轮藏在机身下面，按真实位置（左右各 0.1175 m）叠一层「透视」的轮子 + 条纹，看得见各转多快
      var bot = E.el('g', {}, s);
      E.vacuum(bot, 0, 0, 0.8, { top: true });
      var bumper = K.paint(E.el('path', { d: 'M 21.2 -31.5 A 38 38 0 0 1 21.2 31.5', fill: 'none', 'stroke-width': 5.6, 'stroke-linecap': 'round' }, bot), null, E.GOOD);
      var wheels = [-30, 22].map(function (wy) {
        K.paint(E.el('rect', { x: -15, y: wy, width: 30, height: 8, rx: 2 }, bot), E.INK2);
        var stripes = [0, 1, 2, 3, 4].map(function () {
          var l = E.el('line', { y1: wy + 1.5, y2: wy + 6.5, 'stroke-width': 2 }, bot);
          return K.paint(l, null, E.S2);
        });
        return stripes;
      });
      function placeStripes(stripes, off) {
        stripes.forEach(function (l, k) {
          var x = (-15 + 2 + frac((k * 6 + off) / 30) * 26).toFixed(2);
          l.setAttribute('x1', x); l.setAttribute('x2', x);
        });
      }

      // 控制：左右轮速度条 + 标签
      var barL = K.paint(E.el('rect', { x: 82, y: 100, width: 0, height: 14, rx: 3 }, s), E.ACC);
      var barR = K.paint(E.el('rect', { x: 82, y: 250, width: 0, height: 14, rx: 3 }, s), E.ACC);
      var lblL = E.txt(s, 40, 112, '左轮', 18, E.INK2, 'start');
      var lblR = E.txt(s, 40, 262, '右轮', 18, E.INK2, 'start');

      // 四行步骤标签
      var rows = ROWS.map(function (str, i) {
        var g = E.el('g', {}, s);
        var dot = K.paint(E.el('circle', { cx: 44, cy: ROW_Y[i] - 7, r: 6 }, g), E.STEPS[i].color);
        var text = E.txt(g, 58, ROW_Y[i], str, 20, E.INK, 'start');
        return { g: g, dot: dot, text: text };
      });

      /* ── 右侧三张小卡 ── */
      var cards = [0, 1, 2].map(function (i) { return card(s, i); });

      var cyc0 = null;                                    // st 越过 15.6 那一刻的真实时间 t（轮流点亮从「感知」开始）

      return {
        el: s,
        draw: function (st, t) {
          var fade = E.fade;

          /* 0–0.6：面板、扫地机、拖鞋淡入 */
          var p0 = fade(st, 0, 0.6);
          [left, slipper, bot].forEach(function (n) { E.op(n, p0); });

          /* 四行标签逐句出现 */
          rows.forEach(function (r, i) { rise(r.g, fade(st, ROW_AT[i], 0.4)); });

          /* 3.5 感知：保险杠闪 good 三下；三道弧按 t 一圈圈扩散（9.6 扫地机开走时淡出） */
          var fu = K.seg(st, 3.5, 4.7), flash = fu > 0 && fu < 1 ? Math.pow(Math.sin(3 * Math.PI * fu), 2) : 0;
          E.op(bumper, flash.toFixed(3));
          E.glow(bumper, E.GOOD, flash > 0.4);
          var wv = E.win(st, 3.5, MOVE0, 0.3);
          waves.forEach(function (w, k) {
            var f = frac(t / 1.2 + k / 3);
            w.setAttribute('d', arcD(190, 180, 12 + 30 * f));
            E.op(w, wv * Math.min(1, f / 0.15, (1 - f) / 0.4));   // 前 60% 亮着，最外一段淡掉
          });

          /* 6.1 规划：紫色虚线画出（0.9 s） */
          E.drawOn(plan, fade(st, 6.1, 0.9));

          /* 9.6–11.4 执行：沿路线开过去，车身随切线转 */
          var sm = K.seg(st, MOVE0, MOVE1), uu = prog(sm), m = track(uu);
          /* 轮速跟着实际转弯方向走（差速转向）：往左拐（航向角减小）右轮快，往右拐左轮快，直行一样快。
             航向变化取前后各 3% 路程，免得一两个取样点的抖动把条推满。开动前是第一段的命令（先直行：一样快）；
             减速时速度条跟着车速一起降到 0，车停下那一刻条也是 0 */
          function lens(u) {
            var k = Math.max(-1, Math.min(1, (track(Math.min(1, u + 0.03)).ang - track(Math.max(0, u - 0.03)).ang) / 20));
            return [58 + 22 * k, 58 - 22 * k];
          }
          var wheelLen = st <= MOVE0 ? lens(0) : st >= MOVE1 ? [0, 0]
            : lens(uu).map(function (v) { return v * (sm > 0.5 ? speed(sm) : 1); });

          /* 7.0 控制：左右轮速度条长出来 */
          var pb = fade(st, 7.0, 0.5);
          barL.setAttribute('width', (wheelLen[0] * pb).toFixed(2));
          barR.setAttribute('width', (wheelLen[1] * pb).toFixed(2));
          E.op(barL, pb > 0 ? 1 : 0); E.op(barR, pb > 0 ? 1 : 0);
          rise(lblL, fade(st, 7.0, 0.3)); rise(lblR, fade(st, 7.0, 0.3));

          bot.setAttribute('transform', 'translate(' + m.x.toFixed(2) + ' ' + m.y.toFixed(2) + ') rotate(' + m.ang.toFixed(2) + ')');
          // 轮子条纹 = 各轮实际转过的量：轮速 × 车速从开动积到现在（只由 st 决定）；车停下条纹就停
          var wheelOff = [0, 0], DT = 1 / 60;
          for (var q = MOVE0 + DT / 2; q < Math.min(st, MOVE1); q += DT) {
            var sq = K.seg(q, MOVE0, MOVE1), lq = lens(prog(sq)), vq = speed(sq) * DT * 0.75;
            wheelOff[0] += lq[0] * vq; wheelOff[1] += lq[1] * vq;
          }
          placeStripes(wheels[0], wheelOff[0]);
          placeStripes(wheels[1], wheelOff[1]);

          /* 右侧小卡按念到的名字弹出；图标的循环动作用 t */
          cards.forEach(function (c, i) { E.pop(c.g, c.cx, c.cy, fade(st, CARD_AT[i], 0.4), 0.9); });
          cards[0].icon.set(20 + 8 * Math.sin(2 * Math.PI * t / 2.4), 60);
          cards[1].icon.spin(0.125);   // 车停在卡片里：轮子不转
          cards[2].icon.idle(t);       // 站着轻晃，不原地迈步

          /* 15.6 起：三张卡的四个点 + 左面板四行标签一起轮流点亮（每个 0.5 s，按 t） */
          if (st < CYC) cyc0 = null; else if (cyc0 === null) cyc0 = t - (st - CYC);
          var ph = st < CYC ? -1 : (t - cyc0) / SLOT, dim = fade(st, CYC, 0.3);
          for (var k = 0; k < 4; k++) {
            var a = lit(ph, k), c = E.STEPS[k].color;
            cards.forEach(function (cd) { cd.dots[k].setAttribute('r', (10 + 2 * a).toFixed(2)); E.op(cd.dots[k], 0.35 + 0.65 * a); });
            var r = rows[k];
            r.dot.setAttribute('r', (6 + 3 * a).toFixed(2));
            r.text.style.fill = mix(c, a, E.INK);
            if (st >= ROW_AT[k] + 0.4) E.op(r.g, 1 - 0.5 * dim * (1 - a));
          }
        }
      };
    }
  };
})();
