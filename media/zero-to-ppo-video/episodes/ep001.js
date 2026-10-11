// 零基础篇 001：机器人到底在干嘛？——感知 → 规划 → 控制 → 执行（路线 §L−1「30 秒看懂"一台机器人在干嘛"」）
// 分镜都画在 800×420 的网格上（stage.html 会把 viewBox 缩到每幕实际画到的范围）。旁白在 ep001.py。
(function () {
  var K = window.PaperDemoKit;
  var C = K.xColors;
  var ACC = C.accent, GOOD = C.good, WARN = C.warn, BAD = C.bad, MUT = C.muted, INK = C.ink, INK2 = C.ink2;
  var S1 = C.surface, S2 = C.surface2, BORDER = C.border, GRID = C.grid;

  function el(tag, attrs, parent) { var n = K.svgEl(tag, attrs); if (parent) parent.appendChild(n); return n; }
  function txt(parent, x, y, s, size, color, anchor, weight) {
    var n = K.svgText(x, y, s, '', size, anchor || 'middle');
    n.style.fill = color || INK;
    if (weight) n.style.fontWeight = weight;
    parent.appendChild(n);
    return n;
  }
  function op(node, v) { node.style.opacity = v; }
  function fade(t, a, d) { return K.ease(K.seg(t, a, a + (d || 0.5))); }

  /* ── 四步循环的四个盒子：颜色、名字、一句白话（路线 §L−1） ── */
  var STEPS = [
    { name: '感知', en: 'Sensor', say: ['我在哪？', '世界长什么样？'], color: GOOD },
    { name: '规划', en: 'Plan', say: ['要去哪？', '走哪条路？'], color: 'var(--violet)' },
    { name: '控制', en: 'Control', say: ['每个关节', '出多少力？'], color: ACC },
    { name: '执行', en: 'Body', say: ['电机转起来', '身体动起来'], color: WARN }
  ];

  /* 一个圆角盒子 + 标题 + 小字；返回 { g, rect, setOn(v) } */
  function stepBox(parent, x, y, w, h, step, opts) {
    var o = opts || {};
    var g = el('g', {}, parent);
    var rect = el('rect', { x: x, y: y, width: w, height: h, rx: 18, 'stroke-width': 3 }, g);
    K.paint(rect, S2, step.color);
    txt(g, x + w / 2, y + (o.sub === false ? h / 2 + 11 : h * 0.36), step.name, o.size || 32, step.color, 'middle', 900);
    if (o.sub !== false) {
      var lines = o.sub || step.say, fs = o.subSize || 18;
      lines.forEach(function (ln, k) { txt(g, x + w / 2, y + h * 0.64 + k * fs * 1.35, ln, fs, INK2); });
    }
    return {
      g: g, rect: rect,
      setOn: function (v) {  // 0…1：亮起时盒子填上一层本色
        rect.style.fill = v > 0 ? 'color-mix(in srgb, ' + step.color + ' ' + Math.round(22 * v) + '%, var(--demo-surface-2))' : S2;
        rect.style.strokeWidth = 3 + 2 * v;
      }
    };
  }

  /* ── 主图：四步循环（片头 + 封面）。四个盒子一字排开，执行 → 感知 一条反馈弧，一颗光点绕圈 ── */
  function loopDiagram(K, opts) {
    var o = opts || {};
    var s = K.sceneSvg('四步循环：感知、规划、控制、执行');
    var mk = K.arrowMarker(s, 'ep001-loop-arrow-' + (o.id || 'hero'), INK2);
    var W = 164, H = 150, Y = 92, xs = [24, 220, 416, 612];
    var boxes = STEPS.map(function (st, i) { return stepBox(s, xs[i], Y, W, H, st); });
    var arrows = [0, 1, 2].map(function (i) {
      var a = el('line', { x1: xs[i] + W + 4, y1: Y + H / 2, x2: xs[i + 1] - 6, y2: Y + H / 2, 'stroke-width': 4, 'marker-end': mk }, s);
      K.paint(a, null, INK2);
      return a;
    });
    // 反馈：执行 → 感知，从盒子底下绕回来
    var fbPts = [[xs[3] + W / 2, Y + H + 6], [xs[3] + W / 2, 330], [xs[0] + W / 2, 330], [xs[0] + W / 2, Y + H + 10]];
    var fb = el('path', { d: K.polyPath(fbPts), fill: 'none', 'stroke-width': 4, 'stroke-dasharray': '10 8', 'marker-end': mk }, s);
    K.paint(fb, null, INK2);
    var fbLabel = txt(s, 400, 362, '反馈：动完再看一眼，循环往复', 22, INK2);
    // 本系列主战场：控制
    var star = txt(s, xs[2] + W / 2, Y - 22, '★ 本系列主战场', 22, ACC, 'middle', 700);
    // 绕圈的光点：沿 感知 → … → 执行 → 反馈弧 → 感知
    var loopPts = [[xs[0] + W / 2, Y + H / 2], [xs[3] + W / 2, Y + H / 2]].concat(fbPts.slice(1, 3)).concat([[xs[0] + W / 2, Y + H / 2]]);
    var dot = el('circle', { r: 11 }, s);
    K.paint(dot, '#fff');
    dot.style.filter = 'drop-shadow(0 0 8px #00d4ff)';
    return {
      el: s, boxes: boxes, arrows: arrows, fb: fb, star: star,
      /* still = 封面：全部画好、不放光点，只让「控制」微亮 */
      draw: function (t, still) {
        // 0–2.4 s 四个盒子依次出现，2.6 s 反馈弧，3.2 s 光点开始绕圈（4 s 一圈）
        boxes.forEach(function (b, i) { op(b.g, fade(t, i * 0.5)); });
        arrows.forEach(function (a, i) { op(a, fade(t, i * 0.5 + 0.3)); });
        op(fb, fade(t, 2.2)); op(fbLabel, fade(t, 2.4)); op(star, fade(t, 2.8));
        var run = still ? -1 : t - 3.2;
        op(dot, run > 0 ? 1 : 0);
        var u = run > 0 ? (run / 4) % 1 : 0;
        var p = K.pointOn(loopPts, u);
        dot.setAttribute('cx', p[0].toFixed(1)); dot.setAttribute('cy', p[1].toFixed(1));
        var lit = run > 0 && u < 0.62 ? Math.min(3, Math.floor(u / 0.62 * 4)) : -1;
        boxes.forEach(function (b, i) { b.setOn(i === lit ? 1 : i === 2 ? 0.35 : 0); });
      }
    };
  }

  /* ── 四台机器人的简笔图标。都以 (x, y) 为脚底中心（扫地机为圆心），s 为缩放 ── */
  function holder(parent, x, y, s) {
    var g = el('g', { transform: 'translate(' + x + ' ' + y + ') scale(' + (s || 1) + ')' }, parent);
    return g;
  }
  // 扫地机（俯视）：圆盘 + 顶上的激光雷达 + 前方保险杠
  function vacuum(parent, x, y, s, color) {
    var g = holder(parent, x, y, s), c = color || INK2;
    K.paint(el('circle', { r: 46, 'stroke-width': 4 }, g), S2, c);
    K.paint(el('circle', { r: 14, 'stroke-width': 3 }, g), S1, c);
    K.paint(el('path', { d: 'M -38 -26 A 46 46 0 0 1 38 -26', fill: 'none', 'stroke-width': 7, 'stroke-linecap': 'round' }, g), null, c);
    return { g: g };
  }
  // 机械臂（侧视）：底座 + 两节连杆 + 夹爪；set(a1, a2) 两个关节角（度，0 = 竖直向上）
  function arm(parent, x, y, s, color) {
    var g = holder(parent, x, y, s), c = color || INK2;
    K.paint(el('path', { d: 'M -34 0 L 34 0 L 22 -18 L -22 -18 Z', 'stroke-width': 3 }, g), S2, c);
    var l1 = el('line', { 'stroke-width': 12, 'stroke-linecap': 'round' }, g), l2 = el('line', { 'stroke-width': 10, 'stroke-linecap': 'round' }, g);
    K.paint(l1, null, c); K.paint(l2, null, c);
    var j = [0, 1, 2].map(function () { return K.paint(el('circle', { r: 7, 'stroke-width': 3 }, g), S1, c); });
    var f1 = el('line', { 'stroke-width': 5, 'stroke-linecap': 'round' }, g), f2 = el('line', { 'stroke-width': 5, 'stroke-linecap': 'round' }, g);
    K.paint(f1, null, c); K.paint(f2, null, c);
    function set(a1, a2) {
      var r = Math.PI / 180, p0 = [0, -18];
      var p1 = [p0[0] + 64 * Math.sin(a1 * r), p0[1] - 64 * Math.cos(a1 * r)];
      var p2 = [p1[0] + 54 * Math.sin((a1 + a2) * r), p1[1] - 54 * Math.cos((a1 + a2) * r)];
      var a = (a1 + a2) * r, ux = Math.sin(a), uy = -Math.cos(a), nx = -uy, ny = ux;
      l1.setAttribute('x1', p0[0]); l1.setAttribute('y1', p0[1]); l1.setAttribute('x2', p1[0]); l1.setAttribute('y2', p1[1]);
      l2.setAttribute('x1', p1[0]); l2.setAttribute('y1', p1[1]); l2.setAttribute('x2', p2[0]); l2.setAttribute('y2', p2[1]);
      [p0, p1, p2].forEach(function (p, k) { j[k].setAttribute('cx', p[0]); j[k].setAttribute('cy', p[1]); });
      [[f1, 1], [f2, -1]].forEach(function (f) {
        f[0].setAttribute('x1', p2[0] + 8 * nx * f[1]); f[0].setAttribute('y1', p2[1] + 8 * ny * f[1]);
        f[0].setAttribute('x2', p2[0] + 8 * nx * f[1] + 16 * ux); f[0].setAttribute('y2', p2[1] + 8 * ny * f[1] + 16 * uy);
      });
      return p2;
    }
    set(25, 60);
    return { g: g, set: set };
  }
  // 自动驾驶汽车（侧视）：车身 + 车窗 + 两个轮子 + 车顶传感器；spin(t) 转动轮辐
  function car(parent, x, y, s, color) {
    var g = holder(parent, x, y, s), c = color || INK2;
    K.paint(el('path', { d: 'M -78 -16 L -78 -40 L -48 -44 L -26 -70 L 30 -70 L 54 -44 L 78 -38 L 78 -16 Z', 'stroke-width': 4, 'stroke-linejoin': 'round' }, g), S2, c);
    K.paint(el('path', { d: 'M -20 -64 L 26 -64 L 44 -46 L -38 -46 Z', 'stroke-width': 2 }, g), S1, c);
    K.paint(el('rect', { x: -10, y: -84, width: 20, height: 12, rx: 4, 'stroke-width': 3 }, g), S1, c);
    var wheels = [-46, 46].map(function (wx) {
      var w = el('g', { transform: 'translate(' + wx + ' -14)' }, g);
      K.paint(el('circle', { r: 16, 'stroke-width': 4 }, w), S1, c);
      var sp = el('line', { x1: -10, y1: 0, x2: 10, y2: 0, 'stroke-width': 3 }, w);
      K.paint(sp, null, c);
      return sp;
    });
    return {
      g: g,
      spin: function (t) { wheels.forEach(function (sp) { sp.setAttribute('transform', 'rotate(' + ((t * 360) % 360).toFixed(1) + ')'); }); }
    };
  }
  // 人形机器人：kit 的火柴人（髋部在 y-48，脚底在 y）；walk(ph) 走一步的相位 0…1
  function humanoid(parent, x, y, s, color) {
    var g = holder(parent, x, y, s);
    var f = K.stickFigure(color || INK2, 5, false, 0.55);
    g.appendChild(f.el);
    function stand() { f.pose(0, -48, { lean: 0, armA: [8, 4], armB: [-8, -4], legA: [4, 0], legB: [-4, 0] }); }
    stand();
    return { g: g, walk: function (ph) { f.pose(0, -48, K.poseWalk(ph)); }, stand: stand, pose: function (P) { f.pose(0, -48, P); } };
  }

  window.EpisodeVideo = {
    ep: '001',
    season: 1,
    badge: 'L−1 · 第 1 季',
    question: '机器人到底在干嘛？',
    pitch: 'PITCH',
    footer: '从零基础讲到 PPO，接上《从头学人形机器人算法》',
    hero: function (K) { return loopDiagram(K, { id: 'hero' }); },
    scenes: [],
    outro: {}
  };
})();
