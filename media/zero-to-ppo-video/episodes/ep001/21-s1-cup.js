// 001 第 1 幕：你拿杯子，也是这四步（不用四步条）。
// 人侧身朝右站在桌前：眼睛看见杯子和挡路的花瓶 → 心里想好绕过花瓶 → 命令沿手臂传到肩 / 肘 / 腕 →
// 右臂沿路线 P 伸出去握住把手 → 杯子抬起，眼睛又看到新位置 → 四步起名字：感知 → 规划 → 控制 → 执行。
(function () {
  var E = window.EP001, K = E.K;

  /* 人体比例（身高 H = 292：头顶 96 到地面 388）：肩高约 0.81H（y 152），胯高约 0.5H（y 243），
     上臂 60、前臂 56、手指 12，肩到指尖约 0.44H。台面按厨房台面高（约 0.5H，台面 y 241），杯子放在手够得着的地方。
     手的路线 P：从自然下垂（236,266）抬过矮花瓶（顶 y 212）上方，落到杯子把手（314,208）。
     解肘用「肘朝下」（正常的屈肘）：全程前臂相对上臂只往上折、不反关节；前臂和手离花瓶顶最近约 5px，不穿模。
     校对点：u=0 手 (236,266) 肘 (223,212)；u=0.6 手 (279,190) 肘 (228,212)；u=1 手 (314,208) 肘 (258,205)。 */
  var P_D = 'M236,266 C246,188 290,170 314,208';
  function bez(u) {
    var a = 1 - u;
    return [a * a * a * 236 + 3 * a * a * u * 246 + 3 * a * u * u * 290 + u * u * u * 314,
      a * a * a * 266 + 3 * a * a * u * 188 + 3 * a * u * u * 170 + u * u * u * 208];
  }
  var SH = [230, 152], UPPER = 60, FORE = 56;
  function elbowOf(H) {
    var dx = H[0] - SH[0], dy = H[1] - SH[1], d = Math.min(Math.hypot(dx, dy), UPPER + FORE - 1e-6);
    var at = Math.atan2(dx, dy), al = Math.acos(Math.max(-1, Math.min(1, (UPPER * UPPER + d * d - FORE * FORE) / (2 * UPPER * d))));
    return [SH[0] + UPPER * Math.sin(at - al), SH[1] + UPPER * Math.cos(at - al)];
  }
  function line(parent, x1, y1, x2, y2, color, w) {
    var l = E.el('line', { x1: x1, y1: y1, x2: x2, y2: y2, 'stroke-width': w, 'stroke-linecap': 'round' }, parent);
    K.paint(l, null, color);
    return l;
  }
  function put(l, a, b) {
    l.setAttribute('x1', a[0].toFixed(1)); l.setAttribute('y1', a[1].toFixed(1));
    l.setAttribute('x2', b[0].toFixed(1)); l.setAttribute('y2', b[1].toFixed(1));
  }
  // 文字描一圈底色，压在虚线上时不糊
  function halo(n) { n.style.paintOrder = 'stroke'; n.style.stroke = E.S1; n.style.strokeWidth = '5px'; n.style.strokeLinejoin = 'round'; }
  function mix(c, pct, base) { return 'color-mix(in srgb, ' + c + ' ' + (pct * 100).toFixed(1) + '%, ' + base + ')'; }

  E.scenes[0] = {
    title: '你拿杯子，也是这四步',
    dur: 27.1,
    cues: [
      { at: 0, s: '拿杯子这件小事，你也在跑四步' },
      { at: 2.8, s: '① 眼睛看见：杯子在哪？什么挡路？' },
      { at: 7.4, s: '② 心里想好：用右手，从花瓶上方绕' },
      { at: 11.3, s: '③ 变成命令：每块肌肉各用多大劲儿' },
      { at: 16.7, s: '④ 肌肉骨骼动起来：手伸了出去' },
      { at: 20.6, s: '手一动，眼睛又看到新位置 → 接着调整' },
      { at: 24.1, s: '这四步就是：感知 → 规划 → 控制 → 执行' }
    ],
    build: function (K) {
      var s = K.sceneSvg('伸手拿杯子：眼睛看见杯子和挡路的花瓶，心里想好从花瓶上方绕，命令传到肩肘腕，手臂伸出去握住把手——感知、规划、控制、执行');
      var world = E.el('g', {}, s);

      /* ── 布景 ── */
      line(world, 40, 388, 760, 388, E.BORDER, 3);
      line(world, 277, 251, 277, 388, E.BORDER, 3);
      line(world, 585, 251, 585, 388, E.BORDER, 3);
      K.paint(E.el('rect', { x: 262, y: 241, width: 338, height: 10, 'stroke-width': 2 }, world), E.S2, E.BORDER);
      // 矮花瓶：瓶身 + 细瓶口，顶在 y 212
      K.paint(E.el('path', { d: 'M276,212 L286,212 L286,218 C296,222 296,240 290,241 L272,241 C266,240 266,222 276,218 Z', 'stroke-width': 2.5, 'stroke-linejoin': 'round' }, world), E.S2, E.INK2);
      var cupG = E.el('g', {}, world);
      K.paint(E.el('rect', { x: 322, y: 193, width: 30, height: 48, rx: 4, 'stroke-width': 2.5 }, cupG), E.S2, E.INK2);
      K.paint(E.el('path', { d: 'M322,198 C309,198 309,220 322,220', fill: 'none', 'stroke-width': 2.5, 'stroke-linecap': 'round' }, cupG), null, E.INK2);

      /* ── ② 紫色路线 P（压在识别框和人之下：标签描了底色，手臂从上面伸过去） ── */
      var route = E.curveArrow(world, P_D, E.VIOLET, 3, Math.atan2(208 - 170, 314 - 290) * 180 / Math.PI, { dash: '8 6', head: 13 });

      /* ── ① 视线 + 识别框（压在布景上、人之下） ── */
      var sightG = E.el('g', {}, world);
      var sight = [[337, 214], [281, 226]].map(function (p) {
        var l = E.el('path', { d: 'M243,114 L' + p[0] + ',' + p[1], fill: 'none', 'stroke-width': 2.5, 'stroke-linecap': 'round', 'stroke-dasharray': '6 5' }, sightG);
        K.paint(l, null, E.GOOD);
        return l;
      });
      function idBox(x, y, w, h, lx, ly, name) {
        var g = E.el('g', {}, world);
        var r = E.el('rect', { x: x, y: y, width: w, height: h, rx: 6, fill: 'none', 'stroke-width': 2.5 }, g);
        K.paint(r, null, E.GOOD);
        var label = E.txt(g, lx, ly, name, 20, E.GOOD, 'middle', 700);
        halo(label);
        return { g: g, label: label };
      }
      // 标签放在不挡路线的地方：杯子标在右边，花瓶标在台面下方
      var cupBox = idBox(304, 185, 54, 62, 388, 222, '杯子');
      cupBox.label.setAttribute('text-anchor', 'start');
      // 「花瓶」从规格的 (326,192) 抬到 (328,164)：路线 P 在 x≈306–348 处 y≈183–190 从花瓶上方经过，原位置正好被它穿过；
      // 再往左就压到看杯子的那条视线
      var vaseBox = idBox(264, 205, 34, 40, 304, 278, '花瓶');

      /* ── 人：侧身朝右 ── */
      var farArm = E.el('path', { d: 'M230,152 L226,212 L224,266', fill: 'none', 'stroke-width': 5, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, world);
      K.paint(farArm, null, E.INK); E.op(farArm, 0.5);
      line(world, 230, 243, 212, 388, E.INK, 5);
      line(world, 230, 243, 248, 388, E.INK, 5);
      line(world, 230, 140, 230, 243, E.INK, 5);
      K.paint(E.el('circle', { cx: 230, cy: 118, r: 22, 'stroke-width': 4 }, world), E.S2, E.INK);
      var eye = K.paint(E.el('circle', { cx: 241, cy: 114, r: 3.5 }, world), E.INK);
      var armG = E.el('g', {}, world);
      var upper = line(armG, 0, 0, 0, 0, E.INK, 6), fore = line(armG, 0, 0, 0, 0, E.INK, 6);
      var fingers = [line(armG, 0, 0, 0, 0, E.INK, 4), line(armG, 0, 0, 0, 0, E.INK, 4)];

      /* ── ② 思考泡（两个小圆比规格的 (205,88) / (220,99) 往左上挪了约 9px：原位置的小圆压在头的描边上） ── */
      var bubble = E.el('g', {}, world);
      [['ellipse', { cx: 130, cy: 50, rx: 112, ry: 30 }], ['circle', { cx: 197, cy: 85, r: 6 }], ['circle', { cx: 208, cy: 96, r: 4 }]].forEach(function (b) {
        b[1]['stroke-width'] = 2.5;
        K.paint(E.el(b[0], b[1], bubble), E.S1, E.VIOLET);
      });
      E.txt(bubble, 130, 57, '用右手，从花瓶上方绕', 20, E.VIOLET);

      /* ── ③ 命令面板：三根竖槽 ── */
      var panel = E.el('g', {}, world);
      K.paint(E.el('rect', { x: 630, y: 92, width: 150, height: 150, rx: 12, 'stroke-width': 2 }, panel), E.S1, E.ACC);
      E.txt(panel, 705, 116, '各用多大劲', 18, E.ACC, 'middle', 700);
      var SX = [656, 696, 736], slots = [], fills = [];
      SX.forEach(function (x, k) {
        slots.push(K.paint(E.el('rect', { x: x, y: 132, width: 18, height: 80, rx: 4, 'stroke-width': 2 }, panel), E.S2, E.BORDER));
        fills.push(K.paint(E.el('rect', { x: x + 2, y: 210, width: 14, height: 0, rx: 2 }, panel), E.ACC));
        E.txt(panel, x + 9, 232, ['肩', '肘', '腕'][k], 18, E.INK2);
      });
      var dots = [0, 1, 2].map(function () { return K.paint(E.el('circle', { r: 5 }, world), E.ACC); });
      var rings = [0, 1, 2].map(function () { return K.paint(E.el('circle', { r: 9, fill: 'none', 'stroke-width': 3 }, world), null, E.ACC); });

      /* ── 再看一眼：杯子 → 眼睛 ── */
      var back = E.curveArrow(world, 'M337,182 C345,110 292,82 252,106', E.INK, 2.5, Math.atan2(106 - 82, 252 - 292) * 180 / Math.PI, { dash: '6 5', head: 11 });
      var backLabel = E.txt(world, 430, 128, '又看到新位置', 18, E.INK);
      halo(backLabel);

      /* ── 起名字：四个盒子 + 箭头 + 反馈虚线 ── */
      var names = E.el('g', {}, s);
      var BX = [25, 220, 415, 610], NAME_AT = [25.4, 25.8, 26.3, 26.7];
      var SUB = ['眼睛看', '心里想好', '给肌肉派活', '肌肉骨骼动'];
      var boxes = BX.map(function (x, i) {
        var g = E.el('g', {}, names);
        K.paint(E.el('rect', { x: x, y: 150, width: 170, height: 116, rx: 16, 'stroke-width': 3 }, g), E.S2, E.STEPS[i].color);
        E.txt(g, x + 85, 198, E.STEPS[i].name, 30, E.STEPS[i].color, 'middle', 900);
        E.txt(g, x + 85, 238, SUB[i], 20, E.INK2);
        return g;
      });
      var links = [0, 1, 2].map(function (i) { return E.arrow(names, [[BX[i] + 173, 208], [BX[i + 1] - 3, 208]], E.INK2, 3, { head: 10 }); });
      var fb = E.arrow(names, [[695, 266], [695, 300], [110, 300], [110, 272]], E.INK2, 2.5, { dash: '8 6', head: 11 });

      return {
        el: s,
        draw: function (st, t) {
          var fade = E.fade;
          E.op(world, fade(st, 0, 0.5) * (1 - 0.88 * fade(st, 24.1, 0.4)));

          /* 手：静止下垂 → 18.9–20.4 沿 P 走到把手 → 21.0 和杯子一起抬 8px */
          var lift = 8 * fade(st, 21.0, 0.4);
          var H = bez(K.ease(K.seg(st, 18.9, 20.4)));
          H = [H[0], H[1] - lift];
          var El = elbowOf(H);
          put(upper, SH, El); put(fore, El, H);
          var f = [(H[0] - El[0]) / FORE, (H[1] - El[1]) / FORE], n = [-f[1], f[0]];
          var gap = 7 - 6.2 * fade(st, 20.4, 0.2);
          [-1, 1].forEach(function (k, i) {
            put(fingers[i], [H[0] + n[0] * 3 * k, H[1] + n[1] * 3 * k], [H[0] + f[0] * 12 + n[0] * gap * k, H[1] + f[1] * 12 + n[1] * gap * k]);
          });
          var warm = fade(st, 16.9, 0.3);
          var armC = warm > 0 ? mix(E.WARN, warm, E.INK) : E.INK;
          [upper, fore, fingers[0], fingers[1]].forEach(function (l) { l.style.stroke = armC; });
          E.glow(armG, E.WARN, warm > 0.5);
          cupG.setAttribute('transform', lift ? 'translate(0 ' + (-lift).toFixed(2) + ')' : '');

          /* ① 眼睛：2.8 变 good 发光（整个 ① 期间），21.3 再闪一次 */
          var g1 = Math.max(E.win(st, 2.8, 7.4, 0.3), E.win(st, 21.3, 22.6, 0.25));
          eye.style.fill = g1 > 0 ? mix(E.GOOD, g1, E.INK) : E.INK;
          E.glow(eye, E.GOOD, g1 > 0.5);
          var annoOut = fade(st, 20.6, 0.4), annoDim = 1 - 0.6 * fade(st, 16.7, 0.4);
          E.drawOn(sight[0], K.seg(st, 3.0, 3.4));
          E.drawOn(sight[1], K.seg(st, 3.4, 3.8));
          E.op(sightG, (1 - 0.7 * fade(st, 7.4, 0.4)) * (1 - annoOut));
          E.pop(cupBox.g, 331, 216, fade(st, 4.5, 0.3), 0.9);
          E.op(cupBox.g, fade(st, 4.5, 0.3) * annoDim * (1 - annoOut));
          E.pop(vaseBox.g, 281, 225, fade(st, 6.9, 0.3), 0.9);
          E.op(vaseBox.g, fade(st, 6.9, 0.3) * annoDim * (1 - annoOut));
          // 18.9 手开始伸：两个名字先淡掉（框留着到 20.6），免得前臂从「花瓶」两个字上扫过
          var labelOut = 1 - fade(st, 18.9, 0.3);
          E.op(cupBox.label, labelOut); E.op(vaseBox.label, labelOut);

          /* ② 思考泡 7.4 弹出、16.7 淡出；路线 9.6–11.0 画出，16.7 降到 0.5 */
          var bp = fade(st, 7.4, 0.35);
          E.pop(bubble, 130, 50, bp, 0.8);
          E.op(bubble, bp * (1 - fade(st, 16.7, 0.4)));
          var rp = K.seg(st, 9.6, 11.0);
          E.drawOn(route.line, rp);
          E.op(route.head, rp >= 0.97 ? 1 : 0);
          E.op(route.g, 1 - 0.5 * fade(st, 16.7, 0.4));

          /* ③ 面板 11.3 淡入；信号点 11.8 起两趟；13.5 / 13.9 / 14.3 肩肘腕闪环；15.4 填充升起 */
          E.op(panel, fade(st, 11.3, 0.4));
          var path = [[230, 140], SH, El, H];
          dots.forEach(function (d, k) {
            var ph = (st - 11.8 - 0.09 * k) / 0.8, on = ph >= 0 && ph < 2;
            var fr = on ? ph - Math.floor(ph) : 0, p = K.pointOn(path, fr);
            d.setAttribute('cx', p[0].toFixed(1)); d.setAttribute('cy', p[1].toFixed(1));
            E.op(d, on ? Math.min(1, fr * 10, (1 - fr) * 10) : 0);
          });
          var RING_AT = [13.5, 13.9, 14.3], ringPt = [SH, El, H];
          var rise = fade(st, 15.4, 0.6), jump = fade(st, 23.0, 0.25);
          var LV0 = [0.55, 0.40, 0.25], LV1 = [0.38, 0.62, 0.40];
          rings.forEach(function (r, k) {
            var T = RING_AT[k], on = st >= T;
            r.setAttribute('cx', ringPt[k][0].toFixed(1)); r.setAttribute('cy', ringPt[k][1].toFixed(1));
            r.setAttribute('r', (5 + 4 * fade(st, T, 0.15)).toFixed(2));
            E.op(r, on ? 1 - fade(st, T + 0.35, 0.5) : 0);
            slots[k].style.stroke = on ? E.ACC : E.BORDER;
            slots[k].style.strokeWidth = on ? 2.5 : 2;
            E.glow(slots[k], E.ACC, on && st < T + 0.8);
            var lv = rise * (LV0[k] + (LV1[k] - LV0[k]) * jump + 0.08 * Math.sin(2 * Math.PI * 0.5 * t + 1.9 * k));
            var h = 76 * E.clamp(lv, 0, 1);
            fills[k].setAttribute('y', (210 - h).toFixed(2)); fills[k].setAttribute('height', h.toFixed(2));
          });

          /* 再看一眼：21.3–22.1 虚线从杯子画回眼睛 */
          var bk = K.seg(st, 21.3, 22.1);
          E.drawOn(back.line, bk);
          E.op(back.head, bk >= 0.97 ? 1 : 0);
          E.op(backLabel, fade(st, 21.5, 0.4));

          /* 起名字：四个盒子依次弹出，箭头跟着后一个盒子；反馈虚线规格写 26.9 起画，这里提前到 26.75–27.0 画完：
             最后一句配音 5.088 s 拉满 [24.1, 27.1]，幕尾只留 0.3 s 就交叉淡出，按 26.9–27.05 画完后全亮只剩约 0.08 s */
          boxes.forEach(function (g, i) { E.pop(g, BX[i] + 85, 208, fade(st, NAME_AT[i], 0.3), 0.85); });
          links.forEach(function (a, i) { E.op(a.g, fade(st, NAME_AT[i + 1], 0.3)); });
          var fp = K.seg(st, 26.75, 27.0);
          E.drawOn(fb.line, fp);
          E.op(fb.head, fp >= 0.97 ? 1 : 0);
        }
      };
    }
  };
})();
