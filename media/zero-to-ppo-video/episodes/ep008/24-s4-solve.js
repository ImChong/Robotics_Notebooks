// 008 第 4 幕：写成优化问题，交给电脑找。前半是书上的写法，后半是电脑顺着下坡一小步一小步挪到碗底。
// 句 1「三样凑齐」顶上三块牌（变量 / 目标 / 约束）依次飞进来，「优化问题」右边弹出「= 优化问题」；「写」中间写出
//    min（下面一个绿色的 u）、(4 − u)² + u²、s.t. u ≤ 1 三段；
// 句 2「下面」「变量」u 框绿框、下面写「变量」，「目标函数」中间一段框紫框、写「目标函数」，「最小」min 鼓一下，
//    「约束」右边一段框红框、写「约束」和「s.t. = 满足」；「变量多了」下面再写一行一般的写法 min f(x) s.t. g(x) ≤ 0，
//    「一串数字」写 x = (x₁, x₂, …, xₙ)，「写法不变」两行的 min 一起亮一下；
// 句 3 前半收起，顶上留一行小式子，下面是一只大碗（代价图）：「怎么找碗底」碗底出问号，「一个一个试」一串点沿着碗扫过去，「试」右边打红 ×；
// 句 4「常用」小球放在 3.6 格，「往左」往左下的绿色小箭头（变小）、「往右」往右上的红色小箭头（变大），「挪一小步」小球挪到 2.96；
// 句 5 每隔半秒挪一步（u ← u − 0.1 × 坡度），身后留下小点、横轴上方画出每一步的长度，「坡」小球上画出切线，越往下越平，「停在碗底」打勾；
// 句 6「约束」1 格处立起红墙，顶上式子补上 s.t. u ≤ 1，小球回到 0 格重来：0 → 0.8 → 1（顶到墙），「停」打勾；
// 句 7「顺着下坡走」右边写「顺着下坡走」，「第六季」出「第六季：训练策略」小牌和小 G1。
// 时刻取自配音的逐词时间（分镜时间 = 念到这个词的时刻），见 ../ep008.py。
(function () {
  var E = window.EP008, K = E.K, H = E.HEX;
  var FY = 196;                                            // 前半：书上写法这一行的基线
  var CH = { ox: 118, oy: 390, su: 116, sc: 14.5 };        // 后半：大碗
  var T = {
    tags: [0.6, 0.84, 0.97], prob: 2.09, form: [3.21, 4.14],
    sub: 6.13, varBox: 7.1, objBox: 8.47, minB: 9.94, conBox: 11.33,
    gen: 12.25, vecX: 14.08, same: 15.16,
    partB: 16.61, q: 17.5, scan: [18.7, 20.0], noTry: 20.11,
    start: 21.45, left: 22.84, right: 23.96, smaller: 25.26, step1: 26.55,
    steps: [28.57, 29.04, 29.49, 29.99, 30.6, 31.2, 31.6, 32.03, 32.45, 32.85],
    slope: [26.6, 34.0], flat: 31.2, done: 33.83,
    con: 35.77, restart: 36.0, cSteps: [36.39, 36.83], stop: 37.79,
    downhill: 40.11, s6: 41.51
  };

  E.scenes[3] = {
    title: '写成优化问题，交给电脑找',
    dur: 45.1,
    cues: [
      { at: 0, s: '变量、目标、约束凑齐 = 一个**优化问题**' },
      { at: 5.33, s: 'min 下面写**变量**，后面是**目标函数**，s.t. 后面写**约束**' },
      { at: 12.25, s: '变量多了：$u$ 换成一串数 $x$，写法不变' },
      { at: 16.61, s: '电脑怎么找碗底？变量一多，没法一个一个试' },
      { at: 21.36, s: '看往左还是往右代价变小，就往那边挪一小步' },
      { at: 28.3, s: '越靠近碗底，坡越缓，步子越小，最后停在碗底' },
      { at: 35.33, s: '加上约束：挪到 1 格的边上就停下' },
      { at: 39.39, s: '顺着下坡走：第六季训练策略还会用到' }
    ],
    build: function (K) {
      var s = K.sceneSvg('写成优化问题：最小化下面写变量 u，后面是目标函数 (4 − u)² + u²，s.t. 后面写约束 u ≤ 1；变量多了就换成一串数 x，写成 min f(x) s.t. g(x) ≤ 0；电脑找碗底：看往左还是往右代价变小，就往那边挪一小步，越靠近碗底坡越缓、步子越小，最后停在碗底；加上约束，挪到一格的边上就停下；这个顺着下坡走的办法，第六季训练策略还会用到');
      function svgPt(x, y) { var p = s.createSVGPoint(); p.x = x; p.y = y; return p.matrixTransform(s.getScreenCTM().inverse()); }
      // KaTeX 公式实际占的框（svg 坐标）：每帧量一次，字体换好以后自然就准了
      function texBox(fo) {
        var k = fo.querySelector('.katex-html') || fo.texBox, r = k.getBoundingClientRect();
        if (!r.width || !s.getScreenCTM()) return null;
        var a = svgPt(r.left, r.top), b = svgPt(r.right, r.bottom);
        return [a.x, a.y, b.x, b.y];
      }

      /* ══ 前半：书上的写法 ══ */
      var A = E.el('g', {}, s);
      var tags = E.THREE.map(function (d, i) {
        var tg = E.tag(A, 70 + 150 * i, 8, 124, 46, d.s, d.c, { size: 26, ink: d.c, weight: 900, rx: 14, sw: 3 });
        return { g: tg.g, cx: 132 + 150 * i };
      });
      var prob = E.tag(A, 530, 8, 200, 46, '= 优化问题', E.HI, { size: 26, ink: E.HI, weight: 900, rx: 14, sw: 3 });
      var boxes = [E.GOOD, E.VIOLET, E.BAD].map(function (c) {
        return K.paint(E.el('rect', { rx: 10, 'stroke-width': 3 }, A), E.tint(c, 14, 'var(--demo-surface)'), c);
      });
      var fMin = E.tex(A, 112, FY, '\\min', 46, E.INK, 'middle', { w: 120, h: 70 });
      var fU = E.txt(A, 112, FY + 40, 'u', 32, E.GOOD, 'middle', 700);
      fU.style.fontFamily = 'KaTeX_Math, serif'; fU.style.fontStyle = 'italic';
      var fObj = E.tex(A, 176, FY, E.ctex(H.acc, '(4-u)^2') + '+' + E.ctex(H.warn, 'u^2'), 46, E.INK, 'start', { w: 300, h: 80 });
      var fCon = E.tex(A, 500, FY, '\\text{s.t.}\\;\\; u\\le 1', 46, E.INK, 'start', { w: 260, h: 80 });
      var labs = [['变量', E.GOOD], ['目标函数', E.VIOLET], ['约束', E.BAD]].map(function (d) { return E.txt(A, 0, FY + 84, d[0], 24, d[1], 'middle', 900); });
      var stNote = E.txt(A, 0, FY + 112, 's.t. = 满足', 20, E.INK2, 'middle', 700);
      var gen = E.tex(A, 400, 360, '\\min_{' + E.ctex(H.good, 'x') + '}\\; ' + E.ctex(H.violet, 'f(x)') + '\\quad\\text{s.t.}\\;\\; ' + E.ctex(H.bad, 'g(x)\\le 0'), 34, E.INK, 'middle', { w: 520, h: 70 });
      var vecX = E.tex(A, 400, 410, E.ctex(H.good, 'x=(x_1,\\,x_2,\\,\\dots,\\,x_n)'), 26, E.INK, 'middle', { w: 360, h: 46 });

      /* ══ 后半：大碗 ══ */
      var B = E.el('g', {}, s);
      var top = E.tex(B, 400, 30, '', 28, E.INK, 'middle', { w: 560, h: 50 });
      var ch = E.chart(B, { ox: CH.ox, oy: CH.oy, su: CH.su, sc: CH.sc, xLab: 'u', yLab: '代价' });
      var P = ch.P;
      ch.curve(E.cost, E.VIOLET, 5);
      var wall = E.wall(B, s, P(0, 17.2)[1], CH.oy, '');
      var q = E.halo(E.txt(B, P(2, 0)[0], P(0, 8)[1] + 50, '?', 44, E.HI, 'middle', 900));
      var scan = [];
      for (var k = 0; k <= 16; k++) scan.push(ch.dot(E.INK2, 5, B).set(k / 4, E.cost(k / 4)));
      var noTry = E.el('g', {}, B);
      E.halo(E.txt(noTry, 700, 200, '一个一个试', 22, E.INK, 'middle', 900));
      E.cross(noTry, 700, 236, 14, E.BAD, 5);
      var tan = K.paint(E.el('line', { 'stroke-width': 3, 'stroke-dasharray': '7 6', 'stroke-linecap': 'round' }, B), null, E.HI);
      var trail = E.el('g', {}, B), trailDots = [], stepArr = [];
      for (k = 0; k < 14; k++) {
        trailDots.push(K.paint(E.el('circle', { r: 4.5 }, trail), E.HI));
        stepArr.push(E.vec(trail, E.GOOD, 3, { head: 9 }));
      }
      var probeL = E.el('g', {}, B), probeR = E.el('g', {}, B);
      var pl = E.vec(probeL, E.GOOD, 4, { head: 13 }), pr = E.vec(probeR, E.BAD, 4, { head: 13 });
      var plLab = E.halo(E.txt(probeL, 0, 0, '变小', 20, E.GOOD, 'end', 900)), prLab = E.halo(E.txt(probeR, 0, 0, '变大', 20, E.BAD, 'start', 900));
      var ball = K.paint(E.el('circle', { r: 11, 'stroke-width': 3 }, B), E.HI, 'var(--demo-surface)');
      var ok = E.check(B, 0, 0, 22, E.HI, 4.5);
      var okLab = E.halo(E.txt(B, 0, 0, '', 22, E.HI, 'middle', 900));
      var down = E.halo(E.txt(B, 712, 170, '顺着下坡走', 24, E.HI, 'middle', 900));
      var s6 = E.el('g', {}, B);
      var s6tag = E.tag(s6, 628, 196, 168, 40, '第六季：训练策略', E.ACC, { size: 20, ink: E.ACC, weight: 900, rx: 12 });
      var icon = window.Robot3D.g1Icon(s6, 712, 360, 0.9);

      // 两段下坡：从 3.6 出发（不加约束）；从 0 出发、上限 1 格
      var runA = E.descent(3.6, 11), runB = E.descent(0, 2, 1);
      function at(st) {
        if (st < T.restart) {
          var keys = [[0, 3.6], [T.step1, runA[1]]];
          T.steps.forEach(function (a, i) { keys.push([a, runA[i + 2]]); });
          var n = st < T.step1 ? 0 : 1 + T.steps.filter(function (a) { return st >= a; }).length;
          return { u: E.steps(st, keys, 0.3), n: n, run: runA };
        }
        var kb = [[0, 0], [T.cSteps[0], runB[1]], [T.cSteps[1], runB[2]]];
        return { u: E.steps(st, kb, 0.3), n: T.cSteps.filter(function (a) { return st >= a; }).length, run: runB };
      }

      return {
        el: s,
        draw: function (st, t) {
          var f = E.fade;
          var aOn = f(st, 0, 0.4) * (1 - f(st, T.partB, 0.5)), bOn = f(st, T.partB + 0.3, 0.5);
          E.op(A, aOn); E.op(B, bOn);

          /* ── 前半 ── */
          tags.forEach(function (d, i) {
            var p = f(st, T.tags[i], 0.35);
            d.g.setAttribute('transform', 'translate(' + (-60 * (1 - p)).toFixed(1) + ' 0)'); E.op(d.g, p);
          });
          E.pop(prob.g, 630, 31, f(st, T.prob, 0.35), 0.8);
          E.op(fMin, f(st, T.form[0], 0.4)); E.op(fU, f(st, T.form[0], 0.4));
          E.op(fObj, f(st, T.form[0] + 0.4, 0.4)); E.op(fCon, f(st, T.form[1], 0.4));
          var mb = E.bump(st, T.minB, 0.8) + E.bump(st, T.same, 0.9);
          E.scaleAt(fMin, 112, FY - 14, 1 + 0.12 * Math.min(1, mb));
          var rects = [[94, FY + 10, 130, FY + 52], texBox(fObj), texBox(fCon)];
          var on = [f(st, T.varBox, 0.3), f(st, T.objBox, 0.3), f(st, T.conBox, 0.3)];
          rects.forEach(function (r, i) {
            if (!r) return;
            var pad = 8;
            boxes[i].setAttribute('x', (r[0] - pad).toFixed(1)); boxes[i].setAttribute('y', (r[1] - pad).toFixed(1));
            boxes[i].setAttribute('width', (r[2] - r[0] + 2 * pad).toFixed(1)); boxes[i].setAttribute('height', (r[3] - r[1] + 2 * pad).toFixed(1));
            E.op(boxes[i], on[i]);
            labs[i].setAttribute('x', ((r[0] + r[2]) / 2).toFixed(1));
            labs[i].setAttribute('y', (Math.max(FY + 78, r[3] + 34)).toFixed(1));
            E.op(labs[i], on[i]);
          });
          if (rects[2]) { stNote.setAttribute('x', ((rects[2][0] + rects[2][2]) / 2).toFixed(1)); stNote.setAttribute('y', (Math.max(FY + 78, rects[2][3] + 34) + 28).toFixed(1)); }
          E.op(stNote, f(st, T.conBox + 0.3, 0.3));
          E.op(gen, f(st, T.gen, 0.4));
          E.scaleAt(gen, 400, 350, 1 + 0.06 * E.bump(st, T.same, 0.9));
          E.op(vecX, f(st, T.vecX, 0.4));

          /* ── 后半 ── */
          top.setTex('\\min_{' + E.ctex(H.good, 'u') + '}\\;' + E.ctex(H.acc, '(4-u)^2') + '+' + E.ctex(H.warn, 'u^2') +
            (st >= T.con ? '\\quad\\text{s.t.}\\;\\;' + E.ctex(H.bad, 'u\\le 1') : ''));
          E.pop(q, P(2, 0)[0], P(0, 8)[1] + 36, f(st, T.q, 0.3) * (1 - f(st, T.start, 0.3)), 0.6);
          scan.forEach(function (d, i) {
            var a = T.scan[0] + (T.scan[1] - T.scan[0]) * i / 16;
            E.op(d.g, f(st, a, 0.12) * (1 - f(st, T.start - 0.4, 0.4)));
          });
          E.op(noTry, f(st, T.noTry, 0.3) * (1 - f(st, T.start, 0.4)));

          var A2 = at(st), u = A2.u, bp = P(u, E.cost(u));
          // 小球：句 4 出现；句 6 重来前淡出、回到 0 格再淡入
          ball.setAttribute('cx', bp[0].toFixed(1)); ball.setAttribute('cy', bp[1].toFixed(1));
          E.op(ball, f(st, T.start, 0.3) * (1 - f(st, T.restart - 0.3, 0.25)) + f(st, T.restart, 0.25));
          // 身后留下的小点、每一步的长度（横轴上方）
          var reset = st >= T.restart ? 1 : 1 - f(st, T.restart - 0.3, 0.25);
          trailDots.forEach(function (d, i) {
            if (i >= A2.run.length - 1) { E.op(d, 0); E.op(stepArr[i].g, 0); return; }
            var show = i < A2.n, uu = A2.run[i], p = P(uu, E.cost(uu));
            d.setAttribute('cx', p[0].toFixed(1)); d.setAttribute('cy', p[1].toFixed(1));
            E.op(d, show ? 0.7 * reset : 0);
            var a = stepArr[i], y = CH.oy - 16 - (i % 2) * 12, x0 = P(A2.run[i], 0)[0], x1 = P(A2.run[i + 1] === undefined ? uu : A2.run[i + 1], 0)[0];
            a.set(x0, y, x1, y);
            E.op(a.g, show && Math.abs(x1 - x0) > 10 ? reset : 0);   // 太短的就不画（三角比线还长）
          });
          // 往左 / 往右试一下
          var pOn = f(st, T.left, 0.3) * (1 - f(st, T.step1 - 0.2, 0.3));
          var uL = u - 0.55, uR = Math.min(u + 0.45, 4.25), qL = P(uL, E.cost(uL)), qR = P(uR, E.cost(uR));
          pl.set(bp[0] - 6, bp[1] + 4, qL[0], qL[1]); pr.set(bp[0] + 6, bp[1] - 6, qR[0], qR[1]);
          plLab.setAttribute('x', (qL[0] - 10).toFixed(1)); plLab.setAttribute('y', (qL[1] + 24).toFixed(1));
          prLab.setAttribute('x', (qR[0] + 10).toFixed(1)); prLab.setAttribute('y', (qR[1] + 6).toFixed(1));
          E.op(probeL, pOn); E.op(probeR, f(st, T.right, 0.3) * (1 - f(st, T.step1 - 0.2, 0.3)));
          var sb = E.bump(st, T.smaller, 0.8);
          plLab.style.fontSize = (20 + 4 * sb) + 'px';
          // 切线：坡度 4u − 8
          var g = 4 * u - 8, du = 0.55 / Math.sqrt(1 + Math.pow(g * CH.sc / CH.su, 2)) * 1.4;
          var t0 = P(u - du, E.cost(u) - g * du), t1 = P(u + du, E.cost(u) + g * du);
          tan.setAttribute('x1', t0[0].toFixed(1)); tan.setAttribute('y1', t0[1].toFixed(1)); tan.setAttribute('x2', t1[0].toFixed(1)); tan.setAttribute('y2', t1[1].toFixed(1));
          var fl = E.bump(st, T.flat, 1.0);
          tan.style.strokeWidth = 3 + 2 * fl;
          E.op(tan, f(st, T.slope[0], 0.3) * (1 - f(st, T.slope[1], 0.3)));
          // 打勾：碗底 / 顶到墙
          var first = st < T.restart;
          var okAt = first ? P(2, 8) : P(1, 10);
          ok.setAttribute('transform', 'translate(' + (okAt[0] + (first ? 18 : 14)) + ' ' + (okAt[1] - (first ? 40 : 44)) + ')');
          E.drawOn(ok, first ? f(st, T.done, 0.4) * (1 - f(st, T.restart - 0.3, 0.25)) : f(st, T.stop, 0.4));
          okLab.textContent = first ? '碗底' : '停在 1 格';
          okLab.setAttribute('x', (okAt[0] + (first ? 0 : -52)).toFixed(1)); okLab.setAttribute('y', (okAt[1] + 50).toFixed(1));
          E.op(okLab, first ? f(st, T.done, 0.3) * (1 - f(st, T.restart - 0.3, 0.25)) : f(st, T.stop, 0.3));
          // 约束的红墙
          wall.set(P(1, 0)[0], P(4.3, 0)[0]);
          E.op(wall.g, f(st, T.con, 0.4));
          // 顺着下坡走、第六季
          E.op(down, f(st, T.downhill, 0.3));
          E.pop(s6, 712, 280, f(st, T.s6, 0.4), 0.85);
          icon.idle(t);
        }
      };
    }
  };
})();
