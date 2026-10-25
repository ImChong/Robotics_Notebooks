// 003 第 2 幕：向量相加——位移、速度、力都是箭头。
// 左边俯视地图网格（原点 = 起点），Create 2 俯视：「先走一段」沿 (2, 1) 走、身后画出绿箭头，「往右两格、往上一格」时各画一条绿色虚线分量；
// 「再走一段」先原地转向，再沿 (1, 2) 走、琥珀箭头 + 琥珀虚线分量。「在哪」终点冒「?」；「头尾接起来」两支箭头鼓一下、接头处亮一点；
// 「从起点直接指到终点」画蓝色和向量 (0,0) → (3,3)，「答案」时标 (3, 3)。
// 右边算式 [2;1] + [1;2] = [3;3]（数字按绿 / 琥珀 / 蓝上色），「二加一是三」上面一排垫亮黄、地图上两条横的虚线分量再亮一次；「一加二」下面一排、竖的分量。
// 数乘：地图其余变暗，绿箭头 (2, 1) 在「乘二」时拉到 (4, 2) +「×2」，右下算式 2 × [2;1] = [4;2]；「方向不变」过原点的虚线；「两倍长」箭头下两段括号。
// 速度：扫地机掉头往左开，车头一支蓝色速度箭头，「跑多快」时开得更快、箭头跟着变长，停下就收掉；
// 力：右边正面 G1 站在地面线上，「往哪推」亮黄推力箭头从左边推它的肩，「推多大」箭头变长变粗、人往右歪一点。
// 末句：地图淡出；两只脚各一支地面推力（画面左脚绿、右脚琥珀，箭头尖在脚底，方向从脚指向真实重心），「两支箭头一加」两支复制品飞到左边，
// 琥珀接在绿的箭头尖上（头尾相接），「就是」画蓝色合力 +「总力」；「推人」时人脚下两脚中间也画出同一支蓝色总推力。
// 时刻取自配音的逐词时间（分镜时间 ≈ 念到的时刻），见 ../ep003.py。
(function () {
  var E = window.EP003, K = E.K;
  var O = { ox: 100, oy: 330, u: 64, x0: -1, x1: 4, y0: -1, y1: 4, ticks: false };
  var A = [2, 1];
  var ANG_A = Math.atan2(1, 2), ANG_B = Math.atan2(2, 1);
  var LEG1 = [3.55, 5.85], TURN1 = [5.95, 6.45], LEG2 = [6.5, 8.6];
  var GUIDE = { a1: 4.48, a2: 5.27, b1: 7.34, b2: 8.02, out: 9.9 };       // 「右」「往上」各画一条虚线分量
  var ASK = 10.53, JOIN = 12.41, SUM = [13.71, 14.93], PULSE = 15.4, ANS = 16.17;
  var EQ = 17.95, ROWS = 19.04, ROW1 = [20.54, 21.6], ROW2 = [21.73, 22.95];
  var DIM = 23.8, SC_ON = 24.49, SC_GROW = 25.41, SC_EQ2 = 25.9, DIR = 26.02, TWICE = 27.75, CLEAR = 29.5;
  var TURN2 = [30.9, 31.8];
  var SPEED = [[31.98, 0], [32.2, 0.5], [33.19, 0.5], [33.5, 1.5], [34.0, 1.5], [34.6, 0]];   // 格 / 秒：「速度」起步，「跑多快」加速
  var BOT_ON = 33.9, PUSH = 34.42, PUSH_BIG = [35.14, 35.56];
  var S6 = 36.9, FOOT = [38.6, 38.9], FLY = [40.1, 40.35], TOTAL = 41.45, TOTAL_BOT = 42.16, TOTAL_LBL = 42.7;
  var HX = 580, GY = 290, HS = 0.82, FL = 62;                              // G1：两脚中间 x、地面 y、缩放；脚下推力箭头长
  var DIAG = [400, 408];                                                   // 头尾相接的小图：起点

  function bell(st, a, b) { var u = K.seg(st, a, b); return u > 0 && u < 1 ? Math.sin(Math.PI * u) : 0; }
  function lerp(a, b, p) { return [a[0] + (b[0] - a[0]) * p, a[1] + (b[1] - a[1]) * p]; }
  // 按 SPEED 折线积分出走过的格数
  function speedAt(st) {
    for (var i = 1; i < SPEED.length; i++) {
      var a = SPEED[i - 1], b = SPEED[i];
      if (st < b[0]) return st < a[0] ? 0 : a[1] + (b[1] - a[1]) * (st - a[0]) / (b[0] - a[0]);
    }
    return 0;
  }
  function distAt(st) {
    var d = 0;
    for (var i = 1; i < SPEED.length; i++) {
      var a = SPEED[i - 1], b = SPEED[i];
      if (st <= a[0]) break;
      var e = Math.min(st, b[0]), ve = a[1] + (b[1] - a[1]) * (e - a[0]) / (b[0] - a[0]);
      d += (a[1] + ve) / 2 * (e - a[0]);
    }
    return d;
  }

  E.scenes[1] = {
    title: '向量相加：位移、速度、力都是箭头',
    dur: 43.92,
    cues: [
      { at: 0, s: '向量能相加：箭头**头尾相接**' },
      { at: 17.71, s: '对应的数分别相加：$\\begin{bmatrix}2\\\\1\\end{bmatrix}+\\begin{bmatrix}1\\\\2\\end{bmatrix}=\\begin{bmatrix}3\\\\3\\end{bmatrix}$' },
      { at: 23.71, s: '乘一个数：方向不变，长度变倍' },
      { at: 29.45, s: '速度、力也是箭头：方向 + 大小' },
      { at: 36.84, s: '两只脚的推力相加 = 地面推人的总力' }
    ],
    build: function (K) {
      var s = K.sceneSvg('向量能相加：扫地机先走往右两格、往上一格，再走往右一格、往上两格，两支箭头头尾相接，从起点指到终点的就是和，对应的数分别相加；向量乘二方向不变、长度变两倍；速度、力也是箭头，两只脚的地面推力相加就是地面推人的总力');

      /* ── 地图：网格、扫地机、路线 ── */
      var plane = E.plane(s, O), P = plane.P;
      var mapTag = E.txt(s, 230, 40, '俯视地图', 20, E.INK2, 'middle', 700);
      var botG = E.el('g', {}, s), bot = E.el('g', {}, botG);
      E.vacuum(bot, 0, 0, 0.62, { top: true });
      var pathG = E.el('g', {}, s);
      function guide(p, q, color, label, lx, ly) {
        var g = E.el('g', {}, pathG);
        var a = P(p[0], p[1]), b = P(q[0], q[1]);
        var l = E.el('path', { d: 'M ' + a[0] + ' ' + a[1] + ' L ' + b[0] + ' ' + b[1], fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '7 5', 'stroke-linecap': 'round' }, g);
        K.paint(l, null, color);
        E.txt(g, lx, ly, label, 22, color, 'middle', 900);
        return { g: g, l: l };
      }
      var gA1 = guide([0, 0], [2, 0], E.GOOD, '2', P(1, 0)[0], O.oy + 26);
      var gA2 = guide([2, 0], [2, 1], E.GOOD, '1', P(2, 0)[0] - 16, P(0, 0.5)[1] + 8);
      var gB1 = guide([2, 1], [3, 1], E.WARN, '1', P(2.6, 0)[0], P(0, 1)[1] - 10);
      var gB2 = guide([3, 1], [3, 3], E.WARN, '2', P(3, 0)[0] + 16, P(0, 2)[1] + 8);
      var arA = E.vec(pathG, E.GOOD, 5), arB = E.vec(pathG, E.WARN, 5), arS = E.vec(pathG, E.ACC, 5);
      var joint = K.paint(E.el('circle', { r: 7 }, pathG), E.HI);
      var ask = E.txt(pathG, 338, 152, '?', 40, E.HI, 'middle', 900);
      var ans = E.txt(pathG, 328, 150, '(3, 3)', 24, E.ACC, 'start', 900);
      var velG = E.el('g', {}, s), vel = E.vec(velG, E.ACC, 5, { head: 14 });
      var velTxt = E.txt(velG, 0, 0, '速度', 20, E.ACC, 'middle', 700);

      /* ── 数乘：绿箭头拉长、过原点的方向线、两段括号 ── */
      var scG = E.el('g', {}, s);
      var dirLine = E.el('path', { fill: 'none', 'stroke-width': 2, 'stroke-dasharray': '6 6', 'stroke-linecap': 'round' }, scG);
      K.paint(dirLine, null, E.INK2);
      (function () { var c = E.clipSeg(P(-1, -0.5), P(5, 2.5), plane.box); dirLine.setAttribute('d', 'M ' + c[0][0] + ' ' + c[0][1] + ' L ' + c[1][0] + ' ' + c[1][1]); })();
      var dirTxt = E.txt(scG, P(0.95, 0)[0], P(0, -0.6)[1], '方向不变', 20, E.INK2, 'middle', 700);
      var scV = E.vec(scG, E.GOOD, 5);
      var times = E.txt(scG, 0, 0, '×2', 26, E.GOOD, 'middle', 900);
      // 两段括号：箭头右下方各一段（法向 = 方向逆时针转 -90°）
      var nx = Math.sin(ANG_A), ny = Math.cos(ANG_A);                       // 画面坐标里的「右下」法向
      var brackets = [[0, 0, 2, 1], [2, 1, 4, 2]].map(function (q) {
        var a = P(q[0], q[1]), b = P(q[2], q[3]), off = 16, tk = 8;
        var a1 = [a[0] + nx * off, a[1] + ny * off], b1 = [b[0] + nx * off, b[1] + ny * off];
        var p = E.el('path', { d: 'M ' + (a1[0] - nx * tk).toFixed(1) + ' ' + (a1[1] - ny * tk).toFixed(1) + ' L ' + a1[0].toFixed(1) + ' ' + a1[1].toFixed(1) +
          ' L ' + b1[0].toFixed(1) + ' ' + b1[1].toFixed(1) + ' L ' + (b1[0] - nx * tk).toFixed(1) + ' ' + (b1[1] - ny * tk).toFixed(1), fill: 'none', 'stroke-width': 2.5, 'stroke-linejoin': 'round' }, scG);
        return K.paint(p, null, E.GOOD);
      });
      var twiceTxt = E.txt(scG, P(3, 1.5)[0] + nx * 44, P(3, 1.5)[1] + ny * 44, '两倍长', 20, E.GOOD, 'middle', 700);

      /* ── 右边算式 ── */
      var eqG = E.el('g', {}, s), EQY = 165, ROWY = [EQY - 37, EQY + 9];
      var bands = ROWY.map(function (y) { return K.paint(E.el('rect', { x: 444, y: y - 18, width: 290, height: 38, rx: 10 }, eqG), E.tint('var(--hi)', 22, 'var(--demo-surface)')); });
      E.tex(eqG, 476, EQY, E.col([2, 1], E.HEX.good), 40);
      E.tex(eqG, 532, EQY, '+', 40);
      E.tex(eqG, 588, EQY, E.col([1, 2], E.HEX.warn), 40);
      E.tex(eqG, 644, EQY, '=', 40);
      E.tex(eqG, 702, EQY, E.col([3, 3], E.HEX.acc), 40);
      var rowLbl = [E.txt(eqG, 748, ROWY[0] + 7, '往右', 20, E.INK2, 'start', 700), E.txt(eqG, 748, ROWY[1] + 7, '往上', 20, E.INK2, 'start', 700)];
      var eq2G = E.el('g', {}, s), EQ2Y = 320;
      E.tex(eq2G, 500, EQ2Y, '2\\times', 40);
      E.tex(eq2G, 566, EQ2Y, E.col([2, 1], E.HEX.good), 40);
      var eq2R = E.el('g', {}, eq2G);
      E.tex(eq2R, 624, EQ2Y, '=', 40);
      E.tex(eq2R, 684, EQ2Y, E.col([4, 2], E.HEX.good), 40);

      /* ── 右边正面 G1 + 地面 ── */
      var humG = E.el('g', {}, s);
      K.paint(E.el('line', { x1: HX - 115, y1: GY, x2: HX + 115, y2: GY, 'stroke-width': 3, 'stroke-linecap': 'round' }, humG), null, E.BORDER);
      E.txt(humG, HX + 115, GY + 24, '地面', 18, E.MUT, 'end');
      var hum = E.front(humG);
      hum.place(HX, HS, GY);
      var push = E.vec(s, E.HI, 5, { head: 16 });
      var pushTxt = E.txt(s, 0, 0, '推力', 20, E.HI, 'middle', 700);
      // 脚下推力：脚底中心、重心（真实质心）都换到画面坐标；方向从脚指向重心，箭头尖在脚底
      var feet = hum.footSpan.map(function (r) { return hum.world([(r[0] + r[1]) / 2, 388]); });
      var com = hum.world(hum.com());
      var dirs = feet.map(function (f) { var dx = com[0] - f[0], dy = com[1] - f[1], L = Math.hypot(dx, dy); return [dx / L * FL, dy / L * FL]; });
      var fG = E.el('g', {}, s);
      var fL = E.vec(fG, E.GOOD, 5), fR = E.vec(fG, E.WARN, 5);
      var fTxt = E.txt(fG, feet[0][0] - 34, GY + 52, '地面推力', 20, E.INK2, 'end', 700);
      var dG = E.el('g', {}, s);
      var cL = E.vec(dG, E.GOOD, 5), cR = E.vec(dG, E.WARN, 5), cS = E.vec(dG, E.ACC, 5);
      var sumV = [dirs[0][0] + dirs[1][0], dirs[0][1] + dirs[1][1]];
      var totTxt = E.txt(dG, DIAG[0] - 40, DIAG[1] + sumV[1] / 2 + 7, '总力', 22, E.ACC, 'end', 900);
      var mid = [(feet[0][0] + feet[1][0]) / 2, GY];
      var tot = E.vec(s, E.ACC, 6);

      return {
        el: s,
        draw: function (st, t) {
          var fade = E.fade;
          var mapOut = 1 - fade(st, S6, 0.5);

          /* 地图淡入；末句淡出 */
          E.op(plane.g, (fade(st, 0, 0.5) * mapOut).toFixed(3));
          E.op(mapTag, (fade(st, 2.84, 0.4) * mapOut * (1 - 0.6 * fade(st, DIM, 0.4))).toFixed(3));

          /* 扫地机：走 (2, 1) → 转向 → 走 (1, 2)；速度那句掉头往左开 */
          var p1 = K.ease(K.seg(st, LEG1[0], LEG1[1])), p2 = K.ease(K.seg(st, LEG2[0], LEG2[1]));
          var pos = st < LEG2[0] ? lerp([0, 0], A, p1) : lerp(A, [3, 3], p2);
          var ang = ANG_A + (ANG_B - ANG_A) * K.ease(K.seg(st, TURN1[0], TURN1[1]));
          ang += (Math.PI - ANG_B) * K.ease(K.seg(st, TURN2[0], TURN2[1]));
          var d = distAt(st);
          pos = [pos[0] - d, pos[1]];
          var bp = P(pos[0], pos[1]);
          bot.setAttribute('transform', 'translate(' + bp[0].toFixed(2) + ' ' + bp[1].toFixed(2) + ') rotate(' + (-ang * 180 / Math.PI).toFixed(2) + ')');
          var dimBot = 1 - 0.75 * fade(st, DIM, 0.5) * (1 - fade(st, 29.6, 0.5));
          E.op(botG, (fade(st, 0.2, 0.4) * dimBot * mapOut).toFixed(3));

          /* 路线箭头：绿 / 琥珀跟着扫地机长出来；虚线分量在念到「右」「往上」时画出 */
          var o0 = P(0, 0), a1 = P(A[0], A[1]), s3 = P(3, 3);
          if (st < LEG1[0]) { arA.set(0, 0, 0, 0); }
          else { var ea = st < LEG2[0] ? bp : a1; arA.set(o0[0], o0[1], ea[0], ea[1]); }
          if (st < LEG2[0]) arB.set(0, 0, 0, 0);
          else { var eb = st < LEG2[1] ? bp : s3; arB.set(a1[0], a1[1], eb[0], eb[1]); }
          var pathDim = (1 - 0.8 * fade(st, DIM, 0.5)) * (1 - fade(st, CLEAR, 0.4));
          E.op(pathG, (fade(st, 0.2, 0.4) * pathDim).toFixed(3));
          var gw = 1 - fade(st, GUIDE.out, 0.4);
          var hRow = E.win(st, ROW1[0], ROW1[1], 0.25), vRow = E.win(st, ROW2[0], ROW2[1], 0.25);
          [[gA1, GUIDE.a1, hRow], [gA2, GUIDE.a2, vRow], [gB1, GUIDE.b1, hRow], [gB2, GUIDE.b2, vRow]].forEach(function (q) {
            var pd = fade(st, q[1], 0.35);
            E.drawOn(q[0].l, pd);
            E.op(q[0].g, Math.max(pd * gw, q[2]).toFixed(3));
          });

          /* 10.53「在哪」?；12.41「头尾接起来」两支箭头鼓一下、接头亮；13.71 画和向量；16.17「答案」标 (3, 3) */
          E.pop(ask, 338, 140, fade(st, ASK, 0.3) * (1 - fade(st, ANS - 0.2, 0.25)), 0.6);
          var jb = bell(st, JOIN, JOIN + 0.9);
          arA.line.style.strokeWidth = arB.line.style.strokeWidth = (5 + 4 * jb).toFixed(2);
          joint.setAttribute('cx', a1[0]); joint.setAttribute('cy', a1[1]);
          joint.setAttribute('r', (6 + 6 * jb).toFixed(2));
          E.op(joint, Math.max(jb, 0.0).toFixed(3));
          var ps = K.ease(K.seg(st, SUM[0], SUM[1]));
          if (ps > 0) { var e = lerp(o0, s3, ps); arS.set(o0[0], o0[1], e[0], e[1]); } else arS.set(0, 0, 0, 0);
          arS.line.style.strokeWidth = (5 + 3 * bell(st, PULSE, PULSE + 0.6)).toFixed(2);
          E.op(ans, fade(st, ANS, 0.3) * (1 - fade(st, DIM, 0.3)));

          /* 右边算式：17.95 出现；19.04「对应」行标签；两排依次垫亮黄；数乘时变暗 */
          E.op(eqG, (fade(st, EQ, 0.4) * (1 - 0.65 * fade(st, DIM + 0.2, 0.4)) * (1 - fade(st, CLEAR, 0.4))).toFixed(3));
          E.op(bands[0], hRow); E.op(bands[1], vRow);
          rowLbl.forEach(function (l) { E.op(l, fade(st, ROWS, 0.3)); });

          /* 数乘：24.49 绿箭头 (2, 1)；25.41「乘二」拉到 (4, 2) +「×2」；26.02「方向不变」虚线；27.75「两倍长」括号 */
          E.op(scG, (fade(st, SC_ON, 0.3) * (1 - fade(st, CLEAR, 0.4))).toFixed(3));
          var k = 1 + K.ease(K.seg(st, SC_GROW, SC_GROW + 0.6)), tip = P(2 * k, k);
          scV.set(o0[0], o0[1], tip[0], tip[1]);
          times.setAttribute('x', (tip[0] + 6).toFixed(1)); times.setAttribute('y', (tip[1] - 22).toFixed(1));
          E.op(times, fade(st, SC_GROW + 0.2, 0.3));
          E.drawOn(dirLine, fade(st, DIR, 0.5));
          E.op(dirTxt, fade(st, DIR + 0.3, 0.3));
          brackets.forEach(function (b, i) { E.drawOn(b, fade(st, TWICE + 0.15 * i, 0.3)); });
          E.op(twiceTxt, fade(st, TWICE + 0.3, 0.3));
          E.op(eq2G, (fade(st, SC_GROW, 0.3) * (1 - fade(st, CLEAR, 0.4))).toFixed(3));
          E.op(eq2R, fade(st, SC_EQ2, 0.3));

          /* 速度：车头一支箭头，长短跟着开多快；停下就收掉 */
          var v = speedAt(st), front = [bp[0] - 30, bp[1]];
          var vl = v > 0.01 ? 26 + 50 * v / 1.5 : 0;
          vel.set(front[0], front[1], front[0] - vl, front[1]);
          velTxt.setAttribute('x', (front[0] - vl / 2).toFixed(1)); velTxt.setAttribute('y', (front[1] - 18).toFixed(1));
          E.op(velG, Math.min(1, v / 0.3).toFixed(3));

          /* 力：33.9 正面 G1；34.42「往哪推」推力箭头；35.14「推多大」变长变粗、人往右歪；末句收掉 */
          E.op(humG, fade(st, BOT_ON, 0.4));
          var big = K.ease(K.seg(st, PUSH_BIG[0], PUSH_BIG[1]));
          var tilt = 3 * fade(st, PUSH_BIG[0] + 0.2, 0.4) * (1 - fade(st, S6, 0.6));
          hum.pose({ tilt: tilt });
          var sh = hum.world(hum.view.at('right_shoulder_roll_link'));
          var pt = [sh[0] - 30, sh[1] + 6], pl = 46 + 44 * big;
          push.set(pt[0] - pl, pt[1], pt[0], pt[1]);
          push.line.style.strokeWidth = (5 + 2 * big).toFixed(2);
          var pOn = fade(st, PUSH, 0.3) * (1 - fade(st, S6, 0.4));
          E.op(push.g, pOn.toFixed(3));
          pushTxt.setAttribute('x', (pt[0] - pl / 2).toFixed(1)); pushTxt.setAttribute('y', (pt[1] - 18).toFixed(1));
          E.op(pushTxt, pOn.toFixed(3));

          /* 末句：脚下两支地面推力（箭头尖在脚底）；40.1 复制品飞到左边头尾相接；41.45 合力；42.16 脚下总推力 */
          [[fL, 0], [fR, 1]].forEach(function (q) {
            var f = feet[q[1]], dv = dirs[q[1]], p = K.ease(K.seg(st, FOOT[q[1]], FOOT[q[1]] + 0.5));
            var tail = [f[0] - dv[0], f[1] - dv[1]];
            if (p > 0) q[0].set(tail[0], tail[1], tail[0] + dv[0] * p, tail[1] + dv[1] * p); else q[0].set(0, 0, 0, 0);
          });
          E.op(fG, (1 - 0.6 * fade(st, TOTAL_BOT, 0.4)).toFixed(3));
          E.op(fTxt, (fade(st, 39.48, 0.3) * (1 - fade(st, FLY[0], 0.3))).toFixed(3));
          var mL = K.ease(K.seg(st, FLY[0], FLY[0] + 0.55)), mR = K.ease(K.seg(st, FLY[1], FLY[1] + 0.55));
          var tl0 = [feet[0][0] - dirs[0][0], feet[0][1] - dirs[0][1]], tr0 = [feet[1][0] - dirs[1][0], feet[1][1] - dirs[1][1]];
          var tl = lerp(tl0, DIAG, mL), tr = lerp(tr0, [DIAG[0] + dirs[0][0], DIAG[1] + dirs[0][1]], mR);
          if (mL > 0) cL.set(tl[0], tl[1], tl[0] + dirs[0][0], tl[1] + dirs[0][1]); else cL.set(0, 0, 0, 0);
          if (mR > 0) cR.set(tr[0], tr[1], tr[0] + dirs[1][0], tr[1] + dirs[1][1]); else cR.set(0, 0, 0, 0);
          var pS = K.ease(K.seg(st, TOTAL, TOTAL + 0.5));
          if (pS > 0) cS.set(DIAG[0] - 26, DIAG[1], DIAG[0] - 26 + sumV[0] * pS, DIAG[1] + sumV[1] * pS); else cS.set(0, 0, 0, 0);
          E.op(totTxt, fade(st, TOTAL_LBL, 0.3));
          var pT = K.ease(K.seg(st, TOTAL_BOT, TOTAL_BOT + 0.5));
          if (pT > 0) tot.set(mid[0] - sumV[0], mid[1] - sumV[1], mid[0] - sumV[0] * (1 - pT), mid[1] - sumV[1] * (1 - pT)); else tot.set(0, 0, 0, 0);
        }
      };
    }
  };
})();
