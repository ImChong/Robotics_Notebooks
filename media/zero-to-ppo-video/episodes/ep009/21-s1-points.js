// 009 第 1 幕：一支箭头说不清——刚体上各点的速度不一样。左边网格 + UR5 正侧视（肩关节 = 原点，肘在格点 (1, 2)，同 003），右栏放要点。
// 句 1「刚体」右栏出「刚体：怎么动都不变形」，「小臂」小臂连同手腕、手（肘以外的部分，一起刚性地转）上色；
// 句 2「原点」「肩关节」原点亮一圈、写「原点 = 肩关节」，「肘」肘上亮一个黄点，「绕」肘旁出逆时针的小弯箭头，「往上抬」小臂开始绕肘慢慢往上转（一直转到这一幕结束）；
// 句 3「一点」小臂上亮三个点（离肘 0.6 格、1.3 格、手腕约 2.06 格），「速度」三支速度箭头依次长出来（跟着小臂转，始终垂直于到肘的连线、长度正比于距离），
//    「靠近肘的点」A 鼓一下、右栏「靠近肘：慢」，「往外」B、C 鼓一下、「越往外：越快」；
// 句 4「速度」右栏「刚体的速度 = 一支箭头？」，「写」打红 ×；
// 句 5「规律」右栏换成「规律」，「连线」肘到三个点的虚线画出来，「垂直」三处直角标出来、「① 和到肘的连线垂直」，
//    「离肘越远」「箭头越长」三支箭头依次鼓一下，「正比」「② 长度和距离成正比」；
// 句 6「轴」肘上黄点鼓一下、「转」「多快」小弯箭头鼓一下，「定了」右栏「每一点的速度就都定了」打勾。
// 时刻取自配音的逐词时间（../ep009.py 的 [from, to] 等于每句配音实长，分镜时间 = 念到这个词的时刻）。
(function () {
  var E = window.EP009, K = E.K;
  var D = Math.PI / 180, W = 2;                               // 角速度 2 弧度每秒（逆时针）
  var OX = E.ARM.OX, OY = E.ARM.OY, ARM_S = E.ARM.S, KV = E.ARM.KV, RX = E.ARM.RX;
  var T = {
    rigid: 2.64, noDeform: 4.3, forearm: 6.31,
    origin: 9.1, shoulder: 9.87, elbow: 12.2, around: 13.33, lift: 14.08,
    dots: 16.61, vel: [17.69, 17.91, 18.13], near: 19.29, slow: 19.79, out: 20.5, fast: 21.5,
    one: 24.2, x: 26.29, out1: 27.64,
    rule: 28.76, link: 31.12, perp: 31.46, far: 32.25, longer: [33.21, 33.5, 33.79], ratio: 34.85,
    axis: 39.14, howFast: 39.56, fixed: 41.85
  };
  var LIFT = 40 * D, END = 43.05, ALONG = [0.6, 1.3];         // 前两个点离肘几格（第三个是手腕，约 2.06 格）                             // 小臂从水平往上转 40°，转到这一幕结束

  E.scenes[0] = {
    title: '一支箭头说不清：各点速度不一样',
    dur: END,
    cues: [
      { at: 0, s: '**刚体**：怎么动都不变形，比如机械臂的一节小臂' },
      { at: 7.73, s: '原点选在肩关节；只转动肘关节，小臂绕着肘往上抬' },
      { at: 15.82, s: '小臂上各点速度不一样：靠近肘慢，越往外越快' },
      { at: 23.04, s: '刚体的速度，**不是**一支箭头能写下的' },
      { at: 27.55, s: '规律：每支箭头都和到肘的连线垂直，长度和距离成正比' },
      { at: 37.05, s: '知道绕什么轴、转得多快，每一点的速度就都定了' }
    ],
    build: function (K) {
      var s = K.sceneSvg('UR5 机械臂只转动肘关节，小臂绕着肘往上抬；小臂上靠近肘的点走得慢，越往外越快，所以刚体的速度不能只用一支箭头来写；每支速度箭头都和它到肘的连线垂直，长度和距离成正比，只要知道绕什么轴、转得多快，每一点的速度就都定了');

      /* ══ 左：网格（底层）+ UR5 + 标注（上层） ══ */
      var under = E.el('g', {}, s);
      var arm = E.armAt(s, OX, OY, ARM_S, 330);
      var u = arm.aim(1, 2, 0);
      var pl = E.plane(under, { ox: OX, oy: OY, u: u, x0: -1, x1: 4, y0: -1, y1: 3 });
      var top = E.el('g', {}, s);
      var V = arm.view;
      var RIGID = ['forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link'];

      // 原点：一圈 + 「原点 = 肩关节」
      var oRing = K.paint(E.el('circle', { cx: OX, cy: OY, r: 9, fill: 'none', 'stroke-width': 3 }, top), null, E.HI);
      var oLab = E.halo(E.txt(top, OX - 16, OY - 12, '原点', 22, E.HI, 'end', 900));
      var oLab2 = E.halo(E.txt(top, OX - 16, OY + 16, '= 肩关节', 20, E.INK2, 'end', 700));
      // 肘：黄点 + 「肘」+ 逆时针的小弯箭头（画在肘的左上方，那里没有胳膊）
      var eDot = K.paint(E.el('circle', { r: 7, 'stroke-width': 2.5 }, top), E.HI, E.S1);
      var eLab = E.halo(E.txt(top, 0, 0, '肘', 22, E.HI, 'end', 900));
      var spin = E.arcArrow(top, 0, 0, 30, E.OMEGA, 3.5, { head: 12 });
      // 三个点：A 靠近肘、B 手腕、C 手；虚线连到肘、直角、速度箭头
      var links = [0, 1, 2].map(function () { return K.paint(E.el('path', { fill: 'none', 'stroke-width': 2, 'stroke-dasharray': '5 5' }, top), null, E.HI); });
      var corners = [0, 1, 2].map(function () { return K.paint(E.el('path', { fill: 'none', 'stroke-width': 2.2 }, top), null, E.INK); });
      var vels = [0, 1, 2].map(function () { return E.vec(top, E.VEL, 4.5, { head: 15 }); });
      var dots = [0, 1, 2].map(function () { return K.paint(E.el('circle', { r: 6.5, 'stroke-width': 2.5 }, top), E.VEL, E.S1); });

      /* ══ 右栏 ══ */
      var R = E.el('g', {}, s);
      var rigid = E.el('g', {}, R);
      K.paint(E.el('rect', { x: RX, y: 22, width: 340, height: 82, rx: 14, 'stroke-width': 2.5 }, rigid), E.tint(E.BODY, 10, 'var(--demo-surface)'), E.BODY);
      E.txt(rigid, RX + 22, 56, '刚体', 26, E.BODY, 'start', 900);
      var noDef = E.txt(rigid, RX + 22, 90, '怎么动都不变形', 22, E.INK, 'start', 700);
      // 观察（句 3、4）
      var obs = E.el('g', {}, R);
      var obsRows = [['靠近肘：', '慢'], ['越往外：', '越快']].map(function (d, i) {
        var g = E.el('g', {}, obs), y = 160 + 44 * i;
        E.txt(g, RX + 10, y, d[0], 24, E.INK2, 'start', 700);
        E.txt(g, RX + 10 + 24 * d[0].length, y, d[1], 24, E.VEL, 'start', 900);
        return g;
      });
      var one = E.el('g', {}, obs);
      E.txt(one, RX + 10, 268, '刚体的速度 = 一支箭头？', 24, E.INK, 'start', 900);
      var oneX = E.cross(one, RX + 312, 260, 11, E.ERR, 4.5);
      // 规律（句 5、6）
      var rule = E.el('g', {}, R);
      E.txt(rule, RX + 10, 150, '规律', 24, E.HI, 'start', 900);
      var ruleRows = ['① 和到肘的连线垂直', '② 长度和距离成正比'].map(function (t, i) { return E.txt(rule, RX + 10, 192 + 42 * i, t, 24, E.INK, 'start', 700); });
      var need = E.el('g', {}, rule);
      E.txt(need, RX + 10, 298, '只要知道：绕什么轴、转多快', 23, E.INK, 'start', 900);
      var fixedG = E.el('g', {}, rule);
      var fixedMark = E.check(fixedG, RX + 10, 326, 22, E.GOOD, 4);
      E.txt(fixedG, RX + 42, 343, '每一点的速度就都定了', 23, E.GOOD, 'start', 900);

      function toMath(p) { return [(p[0] - OX) / u, (OY - p[1]) / u]; }

      return {
        el: s,
        draw: function (st, t) {
          var f = E.fade, b = E.bump;

          /* ── 小臂绕肘转：14.08 起慢慢往上转（开头 1 s 缓起，之后匀速） ── */
          var k = E.clamp((st - T.lift) / (END - T.lift), 0, 1), ease0 = 1 / (END - T.lift);
          var phi = LIFT * (k < ease0 ? k * k / (2 * ease0) : k - ease0 / 2) / (1 - ease0 / 2);
          arm.bend(phi);
          V.clearTint();
          var tf = f(st, T.forearm, 0.5);
          if (tf > 0) V.tint(RIGID, E.BODY, 0.4 * tf);

          var e = arm.elbow(), w = arm.wrist();
          var dir = [(w[0] - e[0]), (w[1] - e[1])], dl = Math.hypot(dir[0], dir[1]);
          var pts = ALONG.map(function (d) { return [e[0] + dir[0] / dl * d * u, e[1] + dir[1] / dl * d * u]; }).concat([w]);

          /* ── 原点、肘、小弯箭头 ── */
          E.op(oRing, f(st, T.origin, 0.3)); E.op(oLab, f(st, T.origin, 0.3)); E.op(oLab2, f(st, T.shoulder, 0.3));
          oRing.setAttribute('r', (9 + 4 * b(st, T.origin, 0.7)).toFixed(2));
          eDot.setAttribute('cx', e[0].toFixed(1)); eDot.setAttribute('cy', e[1].toFixed(1));
          var eb = Math.max(b(st, T.elbow, 0.7), b(st, T.axis, 0.7));
          eDot.setAttribute('r', (7 + 4 * eb).toFixed(2));
          E.op(eDot, f(st, T.elbow, 0.3));
          eLab.setAttribute('x', (e[0] - 42).toFixed(1)); eLab.setAttribute('y', (e[1] + 14).toFixed(1));   // 写在肘的左边：左上是小弯箭头，左下是大臂，中间这一小块空着
          E.op(eLab, f(st, T.elbow, 0.3));
          spin.g.setAttribute('transform', 'translate(' + e[0].toFixed(1) + ' ' + e[1].toFixed(1) + ')');
          spin.set(115 * D, 205 * D);
          var sb = b(st, T.howFast, 0.8);
          spin.arc.style.strokeWidth = 3.5 + 2 * sb; E.glow(spin.g, E.OMEGA, sb > 0.05);
          E.op(spin.g, f(st, T.around, 0.4));

          /* ── 三个点、速度箭头、虚线、直角 ── */
          pts.forEach(function (p, i) {
            var r = toMath(p), r0 = toMath(e), rm = [r[0] - r0[0], r[1] - r0[1]], v = E.rot90(rm);
            var tip = [p[0] + KV * u * W * v[0], p[1] - KV * u * W * v[1]];
            var on = f(st, T.vel[i], 0.5);
            // 箭头从零长出来
            vels[i].set(p[0], p[1], p[0] + (tip[0] - p[0]) * on, p[1] + (tip[1] - p[1]) * on);
            E.op(vels[i].g, on > 0 ? 1 : 0);
            var hi = Math.max(i === 0 ? b(st, T.near, 0.8) : 0, i > 0 ? b(st, T.out + 0.25 * (i - 1), 0.8) : 0, b(st, T.longer[i], 0.6));
            vels[i].line.style.strokeWidth = 4.5 + 3 * hi; E.glow(vels[i].g, E.VEL, hi > 0.05);
            dots[i].setAttribute('cx', p[0].toFixed(1)); dots[i].setAttribute('cy', p[1].toFixed(1));
            dots[i].setAttribute('r', (6.5 + 3 * hi).toFixed(2));
            E.op(dots[i], f(st, T.dots + 0.15 * i, 0.3));
            var L = Math.hypot(rm[0], rm[1]) || 1, ux = rm[0] / L, uy = -rm[1] / L, vx = v[0] / L, vy = -v[1] / L;
            links[i].setAttribute('d', 'M ' + e[0].toFixed(1) + ' ' + e[1].toFixed(1) + ' L ' + p[0].toFixed(1) + ' ' + p[1].toFixed(1));
            E.op(links[i], f(st, T.link + 0.12 * i, 0.4));
            var m = 11, c1 = [p[0] - ux * m, p[1] - uy * m], c2 = [c1[0] + vx * m, c1[1] + vy * m], c3 = [p[0] + vx * m, p[1] + vy * m];
            corners[i].setAttribute('d', E.poly([c1, c2, c3]));
            E.op(corners[i], f(st, T.perp + 0.12 * i, 0.3));
          });

          /* ── 右栏 ── */
          E.pop(rigid, RX + 170, 63, f(st, T.rigid, 0.35), 0.9);
          E.op(noDef, f(st, T.noDeform, 0.3));
          var obsOut = 1 - f(st, T.out1, 0.4);
          E.op(obsRows[0], f(st, T.slow, 0.3) * obsOut); E.op(obsRows[1], f(st, T.fast, 0.3) * obsOut);
          E.op(one, f(st, T.one, 0.3) * obsOut); E.drawOn(oneX, f(st, T.x, 0.35));
          E.op(rule, f(st, T.rule, 0.4));
          E.op(ruleRows[0], f(st, T.perp, 0.3)); E.op(ruleRows[1], f(st, T.ratio, 0.3));
          E.op(need, f(st, T.axis - 0.3, 0.3));
          E.op(fixedG, Math.min(1, 2 * f(st, T.fixed - 0.5, 0.4))); E.drawOn(fixedMark, f(st, T.fixed - 0.5, 0.4));
        }
      };
    }
  };
})();
