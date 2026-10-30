// 008 第 4 幕：写成优化问题，交给电脑找。前半是书上的写法，后半是电脑从初始值出发、顺着下坡一小步一小步挪到碗底，以及初始值怎么选。
// 句 1「三样凑齐」顶上三块牌（变量 / 目标 / 约束）依次飞进来，「优化问题」右边弹出「= 优化问题」；「写」中间写出
//    min（下面一个绿色的 u）、(4 − u)² + u²、s.t. u ≤ 1 三段；
// 句 2「下面」「变量」u 框绿框、下面写「变量」，「目标函数」中间一段框紫框、写「目标函数」，「最小」min 鼓一下，
//    「约束」右边一段框橙框、写「约束」和「s.t. = subject to（满足）」；「变量多了」下面再写一行一般的写法 min f(x) s.t. g(x) ≤ 0，
//    「一串数字」写 x = (x₁, x₂, …, xₙ)，「写法不变」两行的 min 一起亮一下；
// 句 3 前半收起，顶上留一行小式子，下面是一只大碗（代价图）：「怎么找碗底」碗底出问号，「一个一个试」一串点沿着碗扫过去，「试」右边打橙 ×；
// 句 4「出发点」小球放在 3.6 格，「初始值」旁边写「初始值」，「往左」往左下的绿色小箭头（变小）、「往右」往右上的橙色小箭头（变大），「挪」小球挪到 2.96；
// 句 5 每隔约半秒挪一步（u ← u − 0.1 × 坡度），身后留下小点、横轴上方画出每一步的长度，「坡」小球上画出切线，越往下越平，「碗底」打勾、写「碗底」；
// 句 6「从左边出发」绿球从 0 格出发也挪到碗底，「从右边出发」原来的起点写「从右边出发」，「同一个」写「同一个碗底」
//    （碗是凸的：局部最优就是全局最优，Boyd & Vandenberghe 4.2.2 节）；
// 句 7 碗换成两个坑的代价（示意），「坑」两个坑底各写「坑」，「出发」两只球从 0.3 格、2.3 格出发，各自停在一个坑里，
//    「坑里」写「局部最优」「全局最优」，「不一定是最低」左边写「不是最低」（初始值很关键：Boyd 1.4.1 节、Tedrake 第 24 章）；
// 句 8「离答案近」全局最优附近的横轴亮一段，「上一次」曲线上插一面小旗「上一次的答案」，从那儿出发两三步就到最低的坑
//    （上一步的解当初始值：Modern Robotics 6.2 节）；
// 句 9「不知道答案」小旗收起、出「答案在哪？」，「像碗一样」虚线画出一只好算的碗（对两个坑的曲线做最小二乘的抛物线），「碗底」它的碗底亮起，
//    「用它做初始值」碗底正上方、两个坑的曲线上放一只球、写「初始值」，挪到最低的坑（拿简单的凸问题的解做初始值：Boyd 1.4.3 节）；
// 句 10「多选几个初始值」五只球依次出现在 0.15、1.1、1.7、2.3、3.8 格，「各挪一遍」一起往下挪，「留下」停在高坑的三只淡下去，
//    「最小」低坑的两只鼓一下、写「代价最小 ✓」；
// 句 11 换回碗，「约束」1 格处立起橙色的墙，顶上式子补上 s.t. u ≤ 1，「从零格出发」小球从 0 格出发：0 → 0.8 → 1（挪过墙就拉回边上，
//    即 EE364b 讲义的投影次梯度法），「停」打勾；
// 句 12「一小步一小步」右边写「一小步一小步挪」，「第六季」出「第六季：训练策略」小牌和小 G1。
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
    start: 23.43, init: 24.43, left: 25.86, right: 26.68, smaller: 27.97, step1: 29.23,
    steps: [31.1, 31.6, 32.1, 32.55, 33.0, 33.45, 33.9, 34.3, 34.7, 35.1],
    slope: [29.4, 37.9], flat: 34.12, done: 36.54,
    two: 41.7, two2: 42.88, sameBot: 44.88,
    wIn: 47.0, wDip: 48.81, wStart: 49.84, wLab: 52.43, notLow: 52.94,
    near: 57.24, prev: 59.34,
    unk: 62.09, unkQ: 63.42, approx: 66.2, appBot: 68.37, appInit: 69.52, appGo: 70.3,
    multi: 73.01, multiGo: 74.4, keep: 75.68, keepBest: 76.34,
    back: 78.4, con: 78.8, restart: 79.46, cSteps: [80.55, 81.02], stop: 81.93,
    small: 83.94, s6: 85.89
  };
  // 两个坑的代价（示意）：w(u) = 1.6 (u − 0.8)² (u − 3.2)² − (u − 2) + 4，左坑高（局部最优）、右坑低（全局最优）
  function W(u) { return 1.6 * Math.pow(u - 0.8, 2) * Math.pow(u - 3.2, 2) - (u - 2) + 4; }
  // 顺着下坡挪：u ← u − α × 坡度（坡度用数值差分），lim 是约束的上限
  function gd(fn, u0, n, alpha, lim) {
    var us = [u0];
    for (var k = 0; k < n; k++) {
      var u = us[us.length - 1], g = (fn(u + 1e-4) - fn(u - 1e-4)) / 2e-4, v = u - alpha * g;
      us.push(lim === undefined ? v : Math.min(v, lim));
    }
    return us;
  }
  // 按时刻表一步步走：keys[i] 时刻挪到 run[i + 1]
  function walk(st, run, t0, dt) {
    var keys = [[0, run[0]]];
    for (var i = 1; i < run.length; i++) keys.push([t0 + dt * (i - 1), run[i]]);
    return { u: E.steps(st, keys, Math.min(0.3, dt * 0.9)), n: keys.filter(function (k, i) { return i && st >= k[0]; }).length };
  }

  E.scenes[3] = {
    title: '写成优化问题，交给电脑找',
    dur: 89.57,
    cues: [
      { at: 0, s: '变量、目标、约束凑齐 = 一个**优化问题**' },
      { at: 5.33, s: 'min 下：**变量**；后面：**目标函数**；s.t. 后：**约束**' },
      { at: 12.25, s: '变量多了：$u$ 换成一串数 $x$，写法不变' },
      { at: 16.61, s: '电脑怎么找碗底？变量一多，没法一个一个试' },
      { at: 21.36, s: '先选出发点（**初始值**），再看往左还是往右代价变小，就往那边挪一小步' },
      { at: 31.01, s: '越靠近碗底，坡越缓，步子越小，最后停在碗底' },
      { at: 38.04, s: '这只碗只有一个碗底：从左、从右出发，都停在同一个碗底' },
      { at: 46.78, s: '不止一个坑：从不同的地方出发，可能停在不同的坑里（局部最优）' },
      { at: 55.61, s: '初始值选在离答案近的地方，常用上一次算出来的答案' },
      { at: 61.68, s: '不知道答案在哪：先换一个像碗一样好算的问题，用它的碗底做初始值' },
      { at: 71.74, s: '多选几个初始值，各挪一遍，留下代价最小的' },
      { at: 78.34, s: '加上约束：从 0 格出发，挪到 1 格的边上就停下' },
      { at: 83.52, s: '一小步一小步地挪：第六季训练策略还会用到' }
    ],
    build: function (K) {
      var s = K.sceneSvg('写成优化问题：最小化下面写变量 u，后面是目标函数 (4 − u)² + u²，s.t. 后面写约束 u ≤ 1；变量多了就换成一串数 x，写成 min f(x) s.t. g(x) ≤ 0；电脑找碗底：先选一个出发点叫初始值，看往左还是往右代价变小，就往那边挪一小步，越靠近碗底坡越缓、步子越小，最后停在碗底；这只碗只有一个碗底，从左边、右边出发都停在同一个碗底；代价不止一个坑时，从不同的地方出发可能停在不同的坑里，所以初始值要选在离答案近的地方，常用上一次算出来的答案；加上约束，从零格出发挪到一格的边上就停下；这种一小步一小步挪的办法，第六季训练策略还会用到');
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
      var stNote = E.txt(A, 0, FY + 112, 's.t. = subject to（满足）', 20, E.INK2, 'middle', 700);
      var gen = E.tex(A, 400, 360, '\\min_{' + E.ctex(H.good, 'x') + '}\\; ' + E.ctex(H.violet, 'f(x)') + '\\quad\\text{s.t.}\\;\\; ' + E.ctex(H.bad, 'g(x)\\le 0'), 34, E.INK, 'middle', { w: 520, h: 70 });
      var vecX = E.tex(A, 400, 410, E.ctex(H.good, 'x=(x_1,\\,x_2,\\,\\dots,\\,x_n)'), 26, E.INK, 'middle', { w: 360, h: 46 });

      /* ══ 后半：大碗（再换成两个坑的代价、再换回来） ══ */
      var B = E.el('g', {}, s);
      var top = E.tex(B, 400, 30, '', 28, E.INK, 'middle', { w: 560, h: 50 });
      var ch = E.chart(B, { ox: CH.ox, oy: CH.oy, su: CH.su, sc: CH.sc, xLab: 'u', yLab: '代价' });
      var P = ch.P;
      var bowl = ch.curve(E.cost, E.VIOLET, 5), wCurve = ch.curve(W, E.VIOLET, 5);
      var wall = E.wall(B, s, P(0, 17.2)[1], CH.oy, '');
      var q = E.halo(E.txt(B, P(2, 0)[0], P(0, 8)[1] + 50, '?', 44, E.HI, 'middle', 900));
      var scan = [];
      for (var k = 0; k <= 16; k++) scan.push(ch.dot(E.INK2, 5, B).set(k / 4, E.cost(k / 4)));
      var noTry = E.el('g', {}, B);
      E.halo(E.txt(noTry, 700, 200, '一个一个试', 22, E.INK, 'middle', 900));
      E.cross(noTry, 700, 236, 14, E.BAD, 5);
      var tan = K.paint(E.el('line', { 'stroke-width': 3, 'stroke-dasharray': '7 6', 'stroke-linecap': 'round' }, B), null, E.HI);
      // 一个小球 + 身后的小点（+ 横轴上方每一步的长度）
      function runner(color, arrows) {
        var g = E.el('g', {}, B), dots = [], arr = [];
        for (var k = 0; k < 16; k++) {
          dots.push(K.paint(E.el('circle', { r: 4.5 }, g), color));
          if (arrows) arr.push(E.vec(g, E.GOOD, 3, { head: 9 }));
        }
        var ball = K.paint(E.el('circle', { r: 11, 'stroke-width': 3 }, g), color, 'var(--demo-surface)');
        return {
          g: g, ball: ball,
          set: function (fn, run, u, n, trail, arrOn) {
            var bp = P(u, fn(u));
            ball.setAttribute('cx', bp[0].toFixed(1)); ball.setAttribute('cy', bp[1].toFixed(1));
            dots.forEach(function (d, i) {
              if (i >= run.length - 1 || i >= n) { E.op(d, 0); if (arr[i]) E.op(arr[i].g, 0); return; }
              var p = P(run[i], fn(run[i]));
              d.setAttribute('cx', p[0].toFixed(1)); d.setAttribute('cy', p[1].toFixed(1)); E.op(d, 0.7 * trail);
              if (arr[i]) {
                var y = CH.oy - 16 - (i % 2) * 12, x0 = P(run[i], 0)[0], x1 = P(run[i + 1], 0)[0];
                arr[i].set(x0, y, x1, y); E.op(arr[i].g, Math.abs(x1 - x0) > 10 ? arrOn : 0);   // 太短的就不画（三角比线还长）
              }
            });
            return bp;
          }
        };
      }
      var rA = runner(E.HI, true), rL = runner(E.GOOD), rW1 = runner(E.HI), rW2 = runner(E.HI), rP = runner(E.HI), rC = runner(E.HI, true);
      var probeL = E.el('g', {}, B), probeR = E.el('g', {}, B);
      var pl = E.vec(probeL, E.GOOD, 4, { head: 13 }), pr = E.vec(probeR, E.BAD, 4, { head: 13 });
      var plLab = E.halo(E.txt(probeL, 0, 0, '变小', 20, E.GOOD, 'end', 900)), prLab = E.halo(E.txt(probeR, 0, 0, '变大', 20, E.BAD, 'start', 900));
      var initLab = E.halo(E.txt(B, 0, 0, '', 22, E.HI, 'end', 900));
      var leftLab = E.halo(E.txt(B, 0, 0, '从左边出发', 22, E.GOOD, 'start', 900));
      var ok = E.check(B, 0, 0, 22, E.HI, 4.5);
      var okLab = E.halo(E.txt(B, 0, 0, '', 22, E.HI, 'middle', 900));
      // 两个坑：坑底、标签（左：局部最优，右：全局最优）
      var wA = gd(W, 0.3, 14, 0.04), wB = gd(W, 2.3, 14, 0.04), wP = gd(W, 3.0, 6, 0.04);
      var locU = wA[wA.length - 1], gloU = wB[wB.length - 1];
      var locLab = E.halo(E.txt(B, P(locU, 0)[0], P(0, 8.6)[1], '局部最优', 22, E.INK, 'middle', 900));
      var gloLab = E.halo(E.txt(B, P(gloU, 0)[0], P(0, 6)[1], '全局最优', 22, E.HI, 'middle', 900));
      var notLow = E.halo(E.txt(B, P(locU, 0)[0], P(0, 8.6)[1] - 26, '不是最低', 20, E.BAD, 'middle', 900));
      var pits = [locU, gloU].map(function (u) { return E.halo(E.txt(B, P(u, 0)[0], P(0, W(u))[1] + 34, '坑', 22, E.INK2, 'middle', 900)); });
      // 离答案近：全局最优附近横轴上的一段亮黄；上一次的答案：曲线上的小旗
      var near = K.paint(E.el('line', { x1: P(gloU - 0.55, 0)[0], y1: CH.oy, x2: P(gloU + 0.55, 0)[0], y2: CH.oy, 'stroke-width': 9, 'stroke-linecap': 'round' }, B), null, E.HI);
      near.style.strokeOpacity = 0.75;
      var nearLab = E.halo(E.txt(B, P(gloU, 0)[0], CH.oy - 16, '离答案近', 20, E.HI, 'middle', 900));
      var prevG = E.el('g', {}, B), pq = P(3.0, W(3.0));
      K.paint(E.el('line', { x1: pq[0], y1: pq[1], x2: pq[0], y2: pq[1] - 52, 'stroke-width': 2.5 }, prevG), null, E.HI);
      K.paint(E.el('path', { d: 'M ' + pq[0] + ' ' + (pq[1] - 52) + ' L ' + (pq[0] + 20) + ' ' + (pq[1] - 45) + ' L ' + pq[0] + ' ' + (pq[1] - 38) + ' Z' }, prevG), E.HI);
      E.halo(E.txt(prevG, pq[0] - 6, pq[1] - 62, '上一次的答案', 20, E.HI, 'middle', 900));
      // 不知道答案在什么地方：① 换一只像碗一样好算的简单问题——对两个坑的曲线做最小二乘的抛物线（0.905u² − 4.62u + 10.72，
      // 碗底 u ≈ 2.55，在中间那道坎 u ≈ 1.89 的右边），用它的碗底做初始值；② 多选几个初始值各挪一遍，留下代价最小的
      var AP = (function () {
        var S = [0, 0, 0, 0, 0], Tt = [0, 0, 0];
        for (var k = 0; k <= 400; k++) { var u = 4 * k / 400, y = W(u); S[0]++; S[1] += u; S[2] += u * u; S[3] += u * u * u; S[4] += u * u * u * u; Tt[0] += y; Tt[1] += u * y; Tt[2] += u * u * y; }
        var M = [[S[4], S[3], S[2], Tt[2]], [S[3], S[2], S[1], Tt[1]], [S[2], S[1], S[0], Tt[0]]];
        for (var i = 0; i < 3; i++) for (var r = 0; r < 3; r++) if (r !== i) { var fct = M[r][i] / M[i][i]; for (var c = i; c < 4; c++) M[r][c] -= fct * M[i][c]; }
        var a = M[0][3] / M[0][0], b = M[1][3] / M[1][1], cc = M[2][3] / M[2][2];
        return { f: function (u) { return a * u * u + b * u + cc; }, u0: -b / (2 * a) };
      })();
      var appG = E.el('g', {}, B);
      var appC = K.paint(E.el('path', { d: ch.path(AP.f, 0.2, 4, 60), fill: 'none', 'stroke-width': 3.5, 'stroke-dasharray': '9 7', 'stroke-linecap': 'round' }, appG), null, E.ACC);
      var ab = P(AP.u0, AP.f(AP.u0)), aw = P(AP.u0, W(AP.u0));
      E.halo(E.txt(appG, P(2.0, 0)[0], P(0, 3.4)[1], '好算的碗', 20, E.ACC, 'middle', 900));   // 写在碗下面，不压两个坑的曲线
      var appBot = K.paint(E.el('circle', { cx: ab[0], cy: ab[1], r: 8, 'stroke-width': 2.5 }, B), E.ACC, 'var(--demo-surface)');
      var unkQ = E.halo(E.txt(B, P(2.2, 0)[0], P(0, 11.5)[1], '答案在哪？', 24, E.HI, 'middle', 900));
      var rAp = runner(E.HI), wAp = gd(W, AP.u0, 10, 0.04);
      var apLab = E.halo(E.txt(B, aw[0] + 14, aw[1] - 14, '初始值', 20, E.HI, 'start', 900));
      var MS = [0.15, 1.1, 1.7, 2.3, 3.8].map(function (u0, i) {
        var run = gd(W, u0, 14, 0.04);
        return { r: runner(E.HI), run: run, end: run[run.length - 1], i: i };
      });
      var bestMin = Math.min.apply(null, MS.map(function (m) { return W(m.end); }));
      var keepLab = E.halo(E.txt(B, P(gloU, 0)[0], P(0, 6)[1], '代价最小 ✓', 22, E.HI, 'middle', 900));
      var small = E.halo(E.txt(B, 712, 170, '一小步一小步挪', 24, E.HI, 'middle', 900));
      var s6 = E.el('g', {}, B);
      E.tag(s6, 628, 196, 168, 40, '第六季：训练策略', E.ACC, { size: 20, ink: E.ACC, weight: 900, rx: 12 });
      var icon = window.Robot3D.g1Icon(s6, 712, 360, 0.9);

      // 碗上的下坡：从 3.6 出发（不加约束）；从 0 出发；加约束后从 0 出发、上限 1 格
      var runA = E.descent(3.6, 11), runL = E.descent(0, 10), runC = E.descent(0, 2, 1);

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
          top.setTex('\\min_{' + E.ctex(H.good, 'u') + '}\\;' + (st >= T.wIn - 0.2 && st < T.back + 0.2 ? E.ctex(H.violet, '\\text{代价}(u)') :
            E.ctex(H.acc, '(4-u)^2') + '+' + E.ctex(H.warn, 'u^2')) + (st >= T.con ? '\\quad\\text{s.t.}\\;\\;' + E.ctex(H.bad, 'u\\le 1') : ''));
          E.pop(q, P(2, 0)[0], P(0, 8)[1] + 36, f(st, T.q, 0.3) * (1 - f(st, T.start, 0.3)), 0.6);
          scan.forEach(function (d, i) {
            var a = T.scan[0] + (T.scan[1] - T.scan[0]) * i / 16;
            E.op(d.g, f(st, a, 0.12) * (1 - f(st, T.start - 0.4, 0.4)));
          });
          E.op(noTry, f(st, T.noTry, 0.3) * (1 - f(st, T.start, 0.4)));
          // 碗 ↔ 两个坑
          var wOn = f(st, T.wIn, 0.6) * (1 - f(st, T.back, 0.5));   // 两个坑：句 7–10
          E.op(bowl, 1 - wOn); E.op(wCurve, wOn);
          var bowlPart = 1 - f(st, T.wIn - 0.4, 0.4);              // 碗上的这些东西在换成两个坑之前收起

          /* 句 4、5：初始值 3.6 → 碗底（亮黄） */
          var ka = [[0, 3.6], [T.step1, runA[1]]];
          T.steps.forEach(function (a, i) { ka.push([a, runA[i + 2]]); });
          var sa = { u: E.steps(st, ka, 0.3), n: ka.filter(function (k, i) { return i && st >= k[0]; }).length };
          var bp = rA.set(E.cost, runA, sa.u, sa.n, 1, 1 - f(st, T.done + 1.0, 0.4));
          E.op(rA.g, f(st, T.start, 0.3) * bowlPart);
          initLab.textContent = st < T.two2 - 0.3 ? '初始值' : '从右边出发';
          var ip = P(3.6, E.cost(3.6));
          initLab.setAttribute('x', (ip[0] - 18).toFixed(1)); initLab.setAttribute('y', (ip[1] - 8).toFixed(1));
          E.op(initLab, (f(st, T.init, 0.3) * (1 - f(st, T.step1, 0.3)) + f(st, T.two2, 0.3)) * bowlPart);
          // 往左 / 往右试一下
          var u = sa.u;
          var pOn = f(st, T.left, 0.3) * (1 - f(st, T.step1 - 0.2, 0.3));
          var uL = u - 0.55, uR = Math.min(u + 0.45, 4.25), qL = P(uL, E.cost(uL)), qR = P(uR, E.cost(uR));
          pl.set(bp[0] - 6, bp[1] + 4, qL[0], qL[1]); pr.set(bp[0] + 6, bp[1] - 6, qR[0], qR[1]);
          plLab.setAttribute('x', (qL[0] - 10).toFixed(1)); plLab.setAttribute('y', (qL[1] + 24).toFixed(1));
          prLab.setAttribute('x', (qR[0] + 10).toFixed(1)); prLab.setAttribute('y', (qR[1] + 6).toFixed(1));
          E.op(probeL, pOn); E.op(probeR, f(st, T.right, 0.3) * (1 - f(st, T.step1 - 0.2, 0.3)));
          plLab.style.fontSize = (20 + 4 * E.bump(st, T.smaller, 0.8)) + 'px';
          // 切线：坡度 4u − 8
          var g = 4 * u - 8, du = 0.55 / Math.sqrt(1 + Math.pow(g * CH.sc / CH.su, 2)) * 1.4;
          var t0 = P(u - du, E.cost(u) - g * du), t1 = P(u + du, E.cost(u) + g * du);
          tan.setAttribute('x1', t0[0].toFixed(1)); tan.setAttribute('y1', t0[1].toFixed(1)); tan.setAttribute('x2', t1[0].toFixed(1)); tan.setAttribute('y2', t1[1].toFixed(1));
          tan.style.strokeWidth = 3 + 2 * E.bump(st, T.flat, 1.0);
          E.op(tan, f(st, T.slope[0], 0.3) * (1 - f(st, T.slope[1], 0.3)));

          /* 句 6：再从左边 0 格出发（绿），也停在同一个碗底 */
          var sl = walk(st, runL, T.two + 0.5, 0.22);
          rL.set(E.cost, runL, sl.u, sl.n, 1);
          E.op(rL.g, f(st, T.two, 0.3) * bowlPart);
          var lp = P(0, 16);
          leftLab.setAttribute('x', (lp[0] + 20).toFixed(1)); leftLab.setAttribute('y', (lp[1] + 6).toFixed(1));
          E.op(leftLab, f(st, T.two, 0.3) * bowlPart);
          // 打勾：碗底（句 5、6）/ 顶到墙（句 9）
          var cons = st >= T.restart;
          var okAt = cons ? P(1, 10) : P(2, 8);
          ok.setAttribute('transform', 'translate(' + (okAt[0] + (cons ? 14 : 18)) + ' ' + (okAt[1] - (cons ? 44 : 40)) + ')');
          E.drawOn(ok, cons ? f(st, T.stop, 0.4) : f(st, T.done, 0.4) * bowlPart);
          okLab.textContent = cons ? '停在 1 格' : st < T.sameBot ? '碗底' : '同一个碗底';
          okLab.setAttribute('x', (okAt[0] + (cons ? -52 : 0)).toFixed(1)); okLab.setAttribute('y', (okAt[1] + 50).toFixed(1));
          E.op(okLab, cons ? f(st, T.stop, 0.3) : f(st, T.done, 0.3) * bowlPart);

          /* 句 7：两个坑——左边出发停在高的坑（局部最优），右边出发停在低的坑（全局最优） */
          var w1 = walk(st, wA, T.wStart + 0.3, 0.17), w2 = walk(st, wB, T.wStart + 0.3, 0.17);
          rW1.set(W, wA, w1.u, w1.n, 1); rW2.set(W, wB, w2.u, w2.n, 1);
          var wPart = 1 - f(st, T.prev - 0.2, 0.4);
          E.op(rW1.g, f(st, T.wStart, 0.3) * wPart * (1 - f(st, T.back, 0.3))); E.op(rW2.g, f(st, T.wStart, 0.3) * wPart * (1 - f(st, T.back, 0.3)));
          pits.forEach(function (n) { E.op(n, f(st, T.wDip, 0.3) * (1 - f(st, T.wLab, 0.3))); });
          var labOut = 1 - f(st, T.prev - 0.3, 0.3);                 // 「上一次的答案」小旗出来前收起，不叠在一起
          E.op(locLab, f(st, T.wLab, 0.3) * labOut);
          E.op(gloLab, f(st, T.wLab + 0.2, 0.3) * labOut);
          E.op(notLow, f(st, T.notLow, 0.3) * labOut);
          /* 句 8：离答案近、上一次的答案 → 从那儿出发，两三步就到最低的坑 */
          var prevOut = 1 - f(st, T.unk, 0.3);                       // 「不知道答案」时收起
          E.op(near, f(st, T.near, 0.3) * prevOut); E.op(nearLab, f(st, T.near, 0.3) * (1 - f(st, T.prev, 0.3)));
          E.pop(prevG, pq[0], pq[1] - 30, f(st, T.prev, 0.3) * prevOut, 0.6);
          var wp = walk(st, wP, T.prev + 0.6, 0.2);
          rP.set(W, wP, wp.u, wp.n, 1);
          E.op(rP.g, f(st, T.prev + 0.3, 0.3) * prevOut);

          /* 句 9：不知道答案在什么地方 → 好算的碗（虚线），它的碗底落到两个坑的曲线上做初始值，挪到最低的坑 */
          E.pop(unkQ, P(2.2, 0)[0], P(0, 11.5)[1] - 8, f(st, T.unkQ, 0.3) * (1 - f(st, T.approx, 0.3)), 0.7);
          var apOut = 1 - f(st, T.multi - 0.3, 0.3);
          E.drawOn(appC, f(st, T.approx, 0.9)); E.op(appG, f(st, T.approx, 0.3) * apOut);
          E.op(appBot, f(st, T.appBot, 0.3) * apOut);
          var wa = walk(st, wAp, T.appGo, 0.13);
          rAp.set(W, wAp, wa.u, wa.n, 1);
          E.op(rAp.g, f(st, T.appInit, 0.3) * apOut); E.op(apLab, f(st, T.appInit, 0.3) * apOut);
          /* 句 10：多选几个初始值（五个），各挪一遍，留下代价最小的（右坑里的两个） */
          var mOut = 1 - f(st, T.back, 0.3);
          MS.forEach(function (m) {
            var w = walk(st, m.run, T.multiGo + 0.03 * m.i, 0.09);
            m.r.set(W, m.run, w.u, w.n, 1);
            var best = Math.abs(W(m.end) - bestMin) < 1e-3;
            E.op(m.r.g, f(st, T.multi + 0.18 * m.i, 0.25) * mOut * (best ? 1 : 1 - 0.65 * f(st, T.keep, 0.4)));
            if (best) m.r.ball.setAttribute('r', (11 + 4 * E.bump(st, T.keepBest, 0.9)).toFixed(1));
          });
          E.op(keepLab, f(st, T.keepBest, 0.3) * mOut);

          /* 句 9：换回碗，加上约束，从 0 格出发：0 → 0.8 → 1（顶到墙） */
          var kc = [[0, 0], [T.cSteps[0], runC[1]], [T.cSteps[1], runC[2]]];
          var uc = E.steps(st, kc, 0.3), nc = T.cSteps.filter(function (a) { return st >= a; }).length;
          rC.set(E.cost, runC, uc, nc, 1, 1);
          E.op(rC.g, f(st, T.restart, 0.25));
          wall.set(P(1, 0)[0], P(4.3, 0)[0]);
          E.op(wall.g, f(st, T.con, 0.4));
          // 一小步一小步挪、第六季
          E.op(small, f(st, T.small, 0.3));
          E.pop(s6, 712, 280, f(st, T.s6, 0.4), 0.85);
          icon.idle(t);
        }
      };
    }
  };
})();
