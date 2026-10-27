// 006 第 4 幕：四元数——一个朝向，两套数。左边 G1（原地转身），中间一个半径为 1 的圆（横轴 = 第一个数 w，竖轴 = 最后一个数 z），
// 圆上的点转过机器人转身角度的一半；右边是四元数 (w, x, y, z) 的列向量，跟着点实时变。
// 句 1「四元数」右边出四个空格，「四个数」鼓一下；「竖直朝上的轴」G1 上画蓝色竖虚线，「中间两个数一直是零」x、y 填 0，「第一个和最后一个」w、z 框亮；
// 句 2「角度减半」顶上写「转过的角度 ÷ 2」，「圆」画圆和两根轴，「那个点」点出来（此刻 0°，在最右边），「横坐标」「纵坐标」各画投影虚线、w、z 填数；
// 句 3「往左转九十度」G1 转 90°、点跟着转 45°，「四十五度」圆心旁标 45°，「上一集算过」点旁出小牌，「零点七一」两条投影上写 0.71，「所以四元数是」四个数依次框亮；
// 句 4「一条规矩」右下写规矩，「平方加起来等于一」出公式，「零点七一的平方」「零点五」「两个加起来」「正好是一」一行行写出，半径亮一下；
// 句 5「机器人转了一整圈」G1 接着转到 360°、点走到 180°，「回到原来的朝向」G1 旁打勾，「只走了半圈」走过的上半圈亮起来，「负一」四个数依次框亮；
// 句 6「原来是」右边左侧出「原来」(1, 0, 0, 0)，「全变了号」两列之间写「× (−1)」，「一模一样」打勾；「再转一圈」G1 转到 720°、点回到最右边，四个数回到 (1, 0, 0, 0)；
// 句 7「每个朝向」G1 再往左转 90°（点到 45°），「两套数」圆上对面出第二个点 −q，右边两列 q、−q，「全部变号」q 到 −q 画一条过圆心的虚线，「双重覆盖」红牌；
// 句 8「取中间值」G1 回到朝前（点 A 在最右边），淡影子朝左；朝左先写成 −q（225°），「对齐」它沿虚线跳到 q（45°），「靠得近」画绿色短弧，「绕远路」画红色长弧（往右转 270°）；
// 句 9「中间的点」短弧中点出黄点，「零点九二……零点三八」右边四个数依次写出、两条投影标数，「四十五度」G1 转到 45°，「上一集说的」出小牌。
// 时刻取自配音的逐词时间（分镜时间 = 念到这个词的时刻），见 ../ep006.py。
(function () {
  var E = window.EP006, K = E.K, H = E.HEX;
  var D = Math.PI / 180;
  var T = {
    vec: 1.17, four: 2.16, axis: 5.19, zeros: 7.3, ends: [9.2, 9.87],
    half: 12.52, circle: 14.24, point: 15.52, wProj: 16.59, zProj: 18.11,
    turn1: [20.7, 1.2], h45: 22.13, ep5: 23.97, c71: [25.77, 26.7], fill: [28.85, 29.83, 30.31, 30.74],
    rule: 33.11, formula: 34.42, sq: 36.97, half5: 38.24, sum: 39.57, one: 40.41,
    turn2: [42.6, 1.8], back: 44.9, halfC: 46.44, neg: [49.5, 50.23, 50.84, 51.41],
    orig: 52.6, origN: [53.05, 53.42, 53.91, 54.4], flip: 55.18, same: 56.53, turn3: [58.1, 1.6], ones: [59.74, 60.2, 60.76, 61.31],
    turn4: [62.5, 1.0], two: 65.16, minus: 65.97, dbl: 69.11,
    reset: [71.0, 0.9], setup: 72.53, align: [74.4, 0.6], near: 75.7, far: 77.33,
    mid: 80.16, midN: [81.66, 82.72, 83.22, 83.7], turn5: [84.71, 1.2], ep5b: 87.03
  };
  var GX = 112, GY = 372, GH = 290;
  var CX = 404, CY = 214, R = 108;
  var VX = 726, V0X = 600, VY = 128;

  function bump(st, a, d) { var u = K.seg(st, a, a + d); return u > 0 && u < 1 ? Math.sin(Math.PI * u) : 0; }
  // 机器人一共转了多少（度，往左为正）：90 → 360 → 720 → 810（= 朝左）→ 720（朝前）→ 765（45°）
  function yawAt(st) {
    var f = E.fade;
    return 90 * f(st, T.turn1[0], T.turn1[1]) + 270 * f(st, T.turn2[0], T.turn2[1]) + 360 * f(st, T.turn3[0], T.turn3[1])
      + 90 * f(st, T.turn4[0], T.turn4[1]) - 90 * f(st, T.reset[0], T.reset[1]) + 45 * f(st, T.turn5[0], T.turn5[1]);
  }
  function qOf(deg) { var h = deg / 2 * D; return [Math.cos(h), 0, 0, Math.sin(h)]; }

  E.scenes[3] = {
    title: '四元数：一个朝向，两套数',
    dur: 91.07,
    cues: [
      { at: 0, s: '**四元数**：4 个数。原地转身，转轴竖直朝上：中间两个数一直是 0' },
      { at: 11.38, s: '规则：转过的角度**减半**，找圆上的点：横坐标 → 第一个数，纵坐标 → 最后一个数' },
      { at: 20.5, s: '往左转 90°：减半 45°，点是 (0.71, 0.71) → 四元数 (0.71, 0, 0, 0.71)' },
      { at: 32.38, s: '只守一条规矩：四个数的平方和 = 1' },
      { at: 42.08, s: '机器人转一整圈回到原样，点只走半圈：(−1, 0, 0, 0)' },
      { at: 52.5, s: '(1, 0, 0, 0) 和 (−1, 0, 0, 0)：全变了号，朝向一样；再转一圈才回去' },
      { at: 62.36, s: '每个朝向都有两套数：q 和 −q，这叫**双重覆盖**' },
      { at: 70.86, s: '取中间值：先把符号对齐，选靠得近的那一套，不然绕远路' },
      { at: 79.24, s: '沿着圆取中点 (0.92, 0, 0, 0.38)：机器人转 45°，沿着转动取中间值' }
    ],
    build: function (K) {
      var s = K.sceneSvg('四元数用四个数。机器人原地转身，转轴竖直朝上，中间两个数一直是零。把转过的角度减半，在半径为一的圆上找到那个点：横坐标是第一个数，纵坐标是最后一个数。往左转九十度，减半四十五度，四元数是 (0.71, 0, 0, 0.71)，四个数平方和等于一。转一整圈回到原来的朝向，点只走了半圈，四元数变成 (−1, 0, 0, 0)；每个朝向都有两套数 q 和 −q，叫双重覆盖。取中间值要先把符号对齐，选靠得近的那一套，不然会绕远路；对齐以后沿着圆取中点 (0.92, 0, 0, 0.38)，机器人转四十五度');

      /* ══ 左：G1 + 竖轴 + 朝向影子 ══ */
      var rob = E.el('g', {}, s);
      var ring = K.paint(E.el('path', { fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '6 6' }, rob), null, E.BORDER);
      var g1 = E.g1(rob, GX, GY, GH);
      var fr = E.frame3(rob, 5);
      ring.setAttribute('d', E.ptsPath(g1.ring(0.45), true));
      var vAxis = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3.5, 'stroke-dasharray': '10 7', 'stroke-linecap': 'round' }, rob), null, E.ACC);
      (function () { var a = g1.project([0, 0, g1.sole - 0.05]), b = g1.project([0, 0, 0.75]); vAxis.setAttribute('d', 'M ' + a[0].toFixed(1) + ' ' + a[1].toFixed(1) + ' L ' + b[0].toFixed(1) + ' ' + b[1].toFixed(1)); })();
      var okG = E.el('g', {}, s);
      E.check(okG, GX - 74, 392, 20, E.GOOD, 4);
      var okT = E.halo(E.txt(okG, GX + 14, 410, '回到原来的朝向', 20, E.GOOD, 'middle', 900));
      var top = E.txt(s, 20, 26, '', 22, E.INK, 'start', 900);

      /* ══ 中：圆、两根轴、点 ══ */
      var cir = E.el('g', {}, s);
      var axW = E.arrow(cir, [[CX - 1.32 * R, CY], [CX + 1.34 * R, CY]], E.INK2, 2.5, { head: 11 });
      var axZ = E.arrow(cir, [[CX, CY + 1.32 * R], [CX, CY - 1.34 * R]], E.ACC, 2.5, { head: 11 });
      E.halo(E.txt(cir, CX + 1.34 * R, CY + 54, '第一个数 w', 19, E.INK2, 'end', 700));   // 写在横轴下面第二行：第一行留给点的横坐标
      E.halo(E.txt(cir, CX - 10, CY - 1.24 * R, '最后一个数 z', 19, E.ACC, 'end', 700));
      // 圆用 path 画（drawOn 画虚线要 path 的 d）：从最右边开始，逆时针一整圈
      var circle = K.paint(E.el('path', { d: 'M ' + (CX + R) + ' ' + CY + ' A ' + R + ' ' + R + ' 0 1 0 ' + (CX - R) + ' ' + CY + ' A ' + R + ' ' + R + ' 0 1 0 ' + (CX + R) + ' ' + CY,
        fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '6 6' }, cir), null, E.HI);
      var rad1 = E.halo(E.txt(cir, 0, 0, '半径 1', 19, E.HI, 'middle', 900));
      var halfArc = K.paint(E.el('path', { fill: 'none', 'stroke-width': 6, 'stroke-linecap': 'round' }, cir), null, E.HI);
      var halfLab = E.halo(E.txt(cir, CX + 58, CY - R - 12, '只走了半圈', 20, E.HI, 'start', 900));
      var angArc = K.paint(E.el('path', { fill: 'none', 'stroke-width': 2.5 }, cir), null, E.HI);
      var angLab = E.halo(E.txt(cir, 0, 0, '45°', 19, E.HI, 'middle', 900));
      var near = K.paint(E.el('path', { fill: 'none', 'stroke-width': 7, 'stroke-linecap': 'round' }, cir), null, E.GOOD);
      var far = K.paint(E.el('path', { fill: 'none', 'stroke-width': 7, 'stroke-linecap': 'round' }, cir), null, E.ERR);
      var farLab = E.halo(E.txt(cir, CX - 6, CY + R + 38, '绕远路：往右转 270°', 19, E.ERR, 'middle', 900));
      var nearLab = E.halo(E.txt(cir, 0, 0, '近路', 19, E.GOOD, 'start', 900));
      var dia = K.paint(E.el('path', { fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '6 5' }, cir), null, E.INK2);
      var proj = [0, 1].map(function () { return K.paint(E.el('path', { fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '5 5' }, cir), null, E.INK2); });
      var radius = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3 }, cir), null, E.INK);
      var coordW = E.halo(E.txt(cir, 0, 0, '', 19, E.INK, 'middle', 900)), coordZ = E.halo(E.txt(cir, 0, 0, '', 19, E.ACC, 'end', 900));
      var ptB = K.paint(E.el('circle', { r: 9, 'stroke-width': 3 }, cir), E.S1, E.WARN);
      var ptBGhost = K.paint(E.el('circle', { r: 9, 'stroke-width': 2.5, 'stroke-dasharray': '4 3', fill: 'none' }, cir), null, E.ERR);
      var labB = E.halo(E.txt(cir, 0, 0, '', 19, E.WARN, 'middle', 900));
      var ptMid = K.paint(E.el('circle', { r: 8 }, cir), E.HI);
      var pt = K.paint(E.el('circle', { r: 9, 'stroke-width': 3 }, cir), E.HI, E.S1);
      var labA = E.halo(E.txt(cir, 0, 0, '', 19, E.INK, 'start', 900));
      var ep5 = E.tag(cir, 0, 0, 132, 32, '第 5 集算过', E.VIOLET, { size: 18, ink: E.INK, weight: 900, rx: 9 });

      /* ══ 右：四元数（列向量），左边可以再放一列 ══ */
      var vecs = [V0X, VX].map(function (x) {
        var g = E.el('g', {}, s);
        var m = E.grid(g, x, VY, [['?'], ['?'], ['?'], ['?']], { size: 26, dx: 70, dy: 40, pad: 42, cellColors: [E.INK, E.X, E.Y, E.Z] });
        var head = E.txt(g, x, 30, '', 21, E.INK, 'middle', 900);
        return { g: g, m: m, head: head };
      });
      var names = E.el('g', {}, s);
      // 分量名写在右边那列的右边，左边留给第二列
      ['w', 'x', 'y', 'z'].forEach(function (c, k) { var n = E.txt(names, VX + 60, VY - 60 + 40 * k + 7, c, 20, E.MUT, 'middle', 700); n.style.fontFamily = 'KaTeX_Math, serif'; n.style.fontStyle = 'italic'; });
      var times = E.el('g', {}, s);
      E.arrow(times, [[V0X + 45, VY - 60], [VX - 45, VY - 60]], E.HI, 2.5, { head: 9 });
      E.txt(times, (V0X + VX) / 2, 30, '× (−1)', 19, E.HI, 'middle', 900);   // 和两列的标题同一行，夹在中间
      var sameG = E.el('g', {}, s);
      E.check(sameG, 600, 236, 20, E.GOOD, 4);
      E.txt(sameG, 628, 254, '朝向一模一样', 21, E.GOOD, 'start', 900);
      var dbl = E.tag(s, 594, 232, 196, 44, '双重覆盖', E.ERR, { size: 24, ink: E.ERR, weight: 900, rx: 12, sw: 3 });
      var dblSub = E.txt(s, 692, 304, 'q 和 −q：同一个朝向', 19, E.INK2, 'middle', 700);
      var ep5b = E.tag(s, 586, 236, 210, 34, '第 5 集：沿着转动取中间值', E.VIOLET, { size: 17, ink: E.INK, weight: 900, rx: 9 });
      // 规矩：平方和 = 1
      var rule = E.el('g', {}, s);
      E.txt(rule, 690, 236, '一条规矩：', 21, E.INK, 'middle', 900);
      var rl = [E.tex(rule, 690, 270, 'w^2+x^2+y^2+z^2=1', 24, E.INK, 'middle', { w: 220, h: 40 }),
        E.tex(rule, 690, 312, '', 24, E.INK, 'middle', { w: 220, h: 40 }),
        E.tex(rule, 690, 354, '', 24, E.INK, 'middle', { w: 220, h: 40 })];

      function P(a, r) { return [CX + (r || R) * Math.cos(a), CY - (r || R) * Math.sin(a)]; }
      function arcD(a0, a1, r) {   // 逆时针从 a0 到 a1（a1 < a0 时顺时针）
        var p = P(a0, r), q = P(a1, r), big = Math.abs(a1 - a0) > Math.PI ? 1 : 0, sweep = a1 > a0 ? 0 : 1;
        return 'M ' + p[0].toFixed(1) + ' ' + p[1].toFixed(1) + ' A ' + (r || R) + ' ' + (r || R) + ' 0 ' + big + ' ' + sweep + ' ' + q[0].toFixed(1) + ' ' + q[1].toFixed(1);
      }

      return {
        el: s,
        draw: function (st) {
          var f = E.fade;
          var yaw = yawAt(st), q = qOf(yaw), h = yaw / 2 * D;

          /* G1、竖轴、影子、勾 */
          g1.set(yaw * D); fr.set(g1.frame(0.5));
          E.op(vAxis, f(st, T.axis, 0.4) * (1 - f(st, T.half, 0.4)));
          E.op(okG, Math.max(f(st, T.back, 0.3) * (1 - f(st, T.orig, 0.3)), f(st, T.same, 0.3) * (1 - f(st, T.turn3[0], 0.3))));
          okT.textContent = st < T.orig ? '回到原来的朝向' : '朝向一样';
          var turned = Math.round(yaw);
          top.textContent = st < T.half ? '' : st < T.turn1[0] ? '转过的角度 ÷ 2 = 点在圆上转的角度' :
            st < T.turn4[0] ? '机器人转了 ' + turned + '°  →  点转了 ' + E.num(turned / 2, 1) + '°' : st < T.reset[0] ? '朝左：圆上有两个点都是它' :
            st < T.turn5[0] ? '朝前 → 朝左：中间是什么？' : '机器人转了 45°  →  点转了 22.5°';
          E.op(top, f(st, T.half, 0.3));

          /* 圆、轴、半径 */
          E.op(cir, f(st, T.circle, 0.4));
          E.drawOn(circle, f(st, T.circle, 0.8));
          var rb = bump(st, T.one, 1.0);
          circle.style.strokeWidth = 2.5 + 2.5 * rb;
          rad1.setAttribute('x', CX - 0.5 * R * Math.cos(30 * D) - 18); rad1.setAttribute('y', CY + 0.5 * R * Math.sin(30 * D) + 26);
          E.op(rad1, f(st, T.circle + 0.4, 0.3) * (1 - f(st, T.turn2[0], 0.4)) + rb);
          var tip = P(h), mtip = st >= T.mid ? P(22.5 * D) : tip, mq = st >= T.mid ? [Math.cos(22.5 * D), 0, 0, Math.sin(22.5 * D)] : q;   // 句 9 投影、坐标跟着中点
          radius.setAttribute('d', 'M ' + CX + ' ' + CY + ' L ' + tip[0].toFixed(1) + ' ' + tip[1].toFixed(1));
          E.op(radius, f(st, T.point, 0.3));
          pt.setAttribute('cx', tip[0].toFixed(1)); pt.setAttribute('cy', tip[1].toFixed(1));
          E.pop(pt, tip[0], tip[1], f(st, T.point, 0.3), 0.3);
          // 投影虚线与坐标
          proj[0].setAttribute('d', 'M ' + mtip[0].toFixed(1) + ' ' + mtip[1].toFixed(1) + ' L ' + mtip[0].toFixed(1) + ' ' + CY);
          proj[1].setAttribute('d', 'M ' + mtip[0].toFixed(1) + ' ' + mtip[1].toFixed(1) + ' L ' + CX + ' ' + mtip[1].toFixed(1));
          var showC = st < T.two || st >= T.mid;
          E.op(proj[0], (st >= T.mid ? f(st, T.midN[0], 0.3) : f(st, T.wProj, 0.3)) * (showC ? 1 : 0));
          E.op(proj[1], (st >= T.mid ? f(st, T.midN[3], 0.3) : f(st, T.zProj, 0.3)) * (showC ? 1 : 0));
          coordW.textContent = E.num(mq[0]); coordZ.textContent = E.num(mq[3]);
          coordW.setAttribute('x', mtip[0].toFixed(1)); coordW.setAttribute('y', (CY + (mq[3] >= -0.05 ? 26 : -12)).toFixed(1));
          coordZ.setAttribute('x', (CX - 8).toFixed(1)); coordZ.setAttribute('y', (mtip[1] + 7).toFixed(1));
          var cOn = showC ? 1 : 0;
          var wOn = st >= T.mid ? f(st, T.midN[0], 0.3) : Math.max(f(st, T.c71[0], 0.3), bump(st, T.wProj, 1.6));
          var zOn = st >= T.mid ? f(st, T.midN[3], 0.3) : Math.max(f(st, T.c71[1], 0.3), bump(st, T.zProj, 1.6));
          E.op(coordW, wOn * cOn);
          E.op(coordZ, zOn * cOn * (Math.abs(mq[3]) > 0.05 ? 1 : 0));
          // 半角的小弧和读数
          var aOn = f(st, T.h45, 0.3) * (1 - f(st, T.turn2[0], 0.3));
          angArc.setAttribute('d', h > 0.02 && h < Math.PI ? arcD(0, h, 34) : '');
          var lp = P(h / 2, 52);
          angLab.setAttribute('x', lp[0].toFixed(1)); angLab.setAttribute('y', (lp[1] + 6).toFixed(1));
          E.op(angArc, aOn); E.op(angLab, aOn);
          ep5.g.setAttribute('transform', 'translate(' + (tip[0] + 16).toFixed(1) + ' ' + (tip[1] - 46).toFixed(1) + ')');
          E.op(ep5.g, f(st, T.ep5, 0.3) * (1 - f(st, T.c71[0] - 0.2, 0.3)));
          // 只走了半圈
          var hc = f(st, T.halfC, 0.5) * (1 - f(st, T.orig, 0.4));
          halfArc.setAttribute('d', arcD(0, Math.PI - 0.001));
          E.drawOn(halfArc, f(st, T.halfC, 0.6)); E.op(halfArc, hc > 0.01 ? 0.8 * hc : 0); E.op(halfLab, hc);

          /* 句 7：q 和 −q（对面的点、过圆心的虚线） */
          var twoOn = f(st, T.two, 0.3) * (1 - f(st, T.reset[0], 0.3));
          var b225 = P(225 * D);
          dia.setAttribute('d', 'M ' + P(45 * D).join(' ') + ' L ' + b225.join(' '));
          E.op(dia, Math.max(f(st, T.minus, 0.4) * (1 - f(st, T.reset[0], 0.3)), f(st, T.setup, 0.3) * (1 - f(st, T.near, 0.4))));
          // 句 8：B 点（朝左）先在 225°（−q），「对齐」沿虚线跳到 45°（q）
          var al = f(st, T.align[0], T.align[1]);
          var bp = st < T.reset[0] ? b225 : [b225[0] + (P(45 * D)[0] - b225[0]) * al, b225[1] + (P(45 * D)[1] - b225[1]) * al];
          ptB.setAttribute('cx', bp[0].toFixed(1)); ptB.setAttribute('cy', bp[1].toFixed(1));
          E.op(ptB, Math.max(twoOn, f(st, T.setup, 0.3) * (1 - f(st, T.mid, 0.4))));
          labB.textContent = st < T.reset[0] ? '−q' : al < 0.5 ? '朝左 −q' : '朝左 q';
          labB.setAttribute('x', (bp[0] + (al > 0.5 ? 34 : -6)).toFixed(1)); labB.setAttribute('y', (bp[1] + (al > 0.5 ? -12 : 30)).toFixed(1));
          E.op(labB, Math.max(twoOn, f(st, T.setup, 0.3) * (1 - f(st, T.mid, 0.4))));
          ptBGhost.setAttribute('cx', b225[0].toFixed(1)); ptBGhost.setAttribute('cy', b225[1].toFixed(1));
          E.op(ptBGhost, f(st, T.far, 0.3) * (1 - f(st, T.mid, 0.4)));
          labA.textContent = st < T.reset[0] ? 'q' : '朝前';
          labA.setAttribute('x', (tip[0] + 14).toFixed(1)); labA.setAttribute('y', (tip[1] + (st < T.reset[0] ? -10 : 30)).toFixed(1));
          E.op(labA, Math.max(twoOn, f(st, T.setup, 0.3) * (1 - f(st, T.mid, 0.4))));
          near.setAttribute('d', arcD(0, 45 * D));
          E.drawOn(near, f(st, T.near, 0.6)); E.op(near, f(st, T.near, 0.1) * 0.85);
          var np = P(30 * D, R + 18);
          nearLab.setAttribute('x', (np[0] + 4).toFixed(1)); nearLab.setAttribute('y', (np[1] - 2).toFixed(1));
          E.op(nearLab, f(st, T.near + 0.4, 0.3) * (1 - f(st, T.mid, 0.4)));
          far.setAttribute('d', arcD(0, -135 * D));
          E.drawOn(far, f(st, T.far, 0.8)); E.op(far, 0.75 * f(st, T.far, 0.1) * (1 - f(st, T.mid, 0.4)));
          E.op(farLab, f(st, T.far + 0.5, 0.3) * (1 - f(st, T.mid, 0.4)));
          var mp = P(22.5 * D);
          ptMid.setAttribute('cx', mp[0].toFixed(1)); ptMid.setAttribute('cy', mp[1].toFixed(1));
          E.pop(ptMid, mp[0], mp[1], f(st, T.mid + 0.6, 0.3) * (1 - f(st, T.turn5[0] + T.turn5[1], 0.3)), 0.3);

          /* 右：列向量 */
          var V = vecs[1].m, V0 = vecs[0].m;
          var vals;
          if (st < T.mid) vals = q.map(function (x) { return E.num(x); });
          else vals = ['0.92', '0', '0', '0.38'];
          var shown = [st >= T.wProj, st >= T.zeros, st >= T.zeros, st >= T.zProj];
          V.set(vals.map(function (v, k) { return [shown[k] ? v : '?']; }));
          if (st >= T.mid && st < T.turn5[0] + T.turn5[1]) V.cells.forEach(function (c, k) { E.op(c, f(st, T.midN[k], 0.25)); });
          else V.cells.forEach(function (c) { E.op(c, 1); });
          E.op(vecs[1].g, f(st, T.vec, 0.4));
          E.scaleAt(vecs[1].g, VX, VY, 1 + 0.08 * bump(st, T.four, 0.7));
          [0, 1, 2, 3].forEach(function (k) {
            var p = 0;
            if (k === 0 || k === 3) p = Math.max(p, f(st, T.ends[k ? 1 : 0], 0.3) * (1 - f(st, T.half, 0.3)));
            [T.fill, T.neg, T.ones, T.midN].forEach(function (arr) { p = Math.max(p, bump(st, arr[k], 0.9)); });
            V.cellBox(k, k === 0 ? E.HI : [E.X, E.Y, E.Z][k - 1], p);
          });
          vecs[1].head.textContent = st < T.orig ? '四元数' : st < T.turn3[0] ? '现在' : st < T.two ? '四元数' : st < T.reset[0] ? '−q' : '四元数';
          E.op(vecs[1].head, f(st, T.vec, 0.4));
          // 左边一列：句 6「原来」(1, 0, 0, 0)；句 7 q (0.71, 0, 0, 0.71)，右边那列此刻是 −q
          var col0 = st < T.turn4[0] ? ['1', '0', '0', '0'] : ['0.71', '0', '0', '0.71'];
          V0.set(col0.map(function (v) { return [v]; }));
          vecs[0].head.textContent = st < T.turn4[0] ? '原来' : 'q';
          var v0On = st < T.turn4[0] ? f(st, T.orig, 0.3) * (1 - f(st, T.turn3[0], 0.3)) : f(st, T.two, 0.3) * (1 - f(st, T.reset[0], 0.3));
          E.op(vecs[0].g, v0On);
          V0.cells.forEach(function (c, k) { E.op(c, st < T.turn4[0] ? f(st, T.origN[k], 0.25) : 1); });
          if (st >= T.two && st < T.reset[0]) V.set([['−0.71'], ['0'], ['0'], ['−0.71']]);
          E.op(names, f(st, T.vec, 0.4));
          E.op(times, Math.max(f(st, T.flip, 0.3) * (1 - f(st, T.turn3[0], 0.3)), f(st, T.minus, 0.3) * (1 - f(st, T.reset[0], 0.3))));
          E.op(sameG, f(st, T.same, 0.3) * (1 - f(st, T.turn3[0], 0.3)));
          E.pop(dbl.g, 692, 254, f(st, T.dbl, 0.3) * (1 - f(st, T.reset[0], 0.3)), 0.7);
          E.op(dblSub, f(st, T.dbl + 0.3, 0.3) * (1 - f(st, T.reset[0], 0.3)));
          E.pop(ep5b.g, 691, 253, f(st, T.ep5b, 0.3), 0.85);
          // 规矩
          E.op(rule, f(st, T.rule, 0.3) * (1 - f(st, T.turn2[0], 0.4)));
          E.op(rl[0], f(st, T.formula, 0.3));
          rl[1].setTex(st < T.half5 ? '0.71^2' : '0.71^2\\approx 0.5');
          rl[2].setTex(st < T.one ? '0.5+0+0+0.5' : '0.5+0+0+0.5=\\textcolor{' + H.good + '}{1}');
          E.op(rl[1], f(st, T.sq, 0.3)); E.op(rl[2], f(st, T.sum, 0.3));
        }
      };
    }
  };
})();
