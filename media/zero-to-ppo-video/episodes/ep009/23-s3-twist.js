// 009 第 3 幕：运动旋量——角速度 + 线速度，手算一遍。左边同前两幕的网格 + UR5（肘在 (1, 2)，肘上 ⊙ 是朝向你的角速度），右栏一行行写算式。
// 句 1 右栏「已知：角速度 ω」「还缺：随便一个点的速度」「→ 每一点的速度都能算出来」；
// 句 2「原点」原点亮一圈、写「原点 = 肩关节」，右栏「选原点（肩关节）」，「不在小臂上」「怎么办？」；
// 句 3「透明板」一张很大的透明板（紫色淡格子）钉在小臂上，「跟着小臂一起转」小臂连同板从 −14° 转到水平，
//    「盖在原点」板上此刻在原点的那一点亮出来，「速度」那一点长出速度箭头，「v」写上 v；右栏「v：板上此刻在原点那一点的速度」；
// 句 4「肘在一、二」肘旁写 (1, 2)，「从肘指到原点」肘到原点画虚线箭头，右栏「肘 → 原点：(−1, −2)」；
// 句 5「垂直」原点处标直角，「逆时针转九十度」右栏写规则、第三集的例子，「二、负一」写出转完的结果；
// 句 6「再乘以二」右栏「× 2：v = (4, −2)」，左边 v 的标签换成 v = (4, −2)；
// 句 7 右栏换成六维列向量：「角速度」上三格框紫、「v」下三格框蓝、「竖着排成一列」括号出来，「运动旋量」写名字，六个数按念到的顺序弹出；
// 句 8、9 验算：右栏「某一点的速度 = v + 绕原点转的那一份」，「肘在一、二」「负二、一」「负四、二」一步步写出，
//    左边肘上画出紫色的「那一份」(−4, 2) 和蓝色的 v 副本 (4, −2)，「正好是零」两支箭头一起淡掉、肘上写「不动」；
// 句 10「v 不是肩关节的速度」原点的圈鼓一下、写「肩关节：没动」，右栏打红 ×；「透明板上」板描边亮一下、在原点的那一点鼓一下，右栏打勾。
// 时刻取自配音的逐词时间，见 ../ep009.py。
(function () {
  var E = window.EP009, K = E.K, H = E.HEX;
  var D = Math.PI / 180, W = 2;
  var OX = E.ARM.OX, OY = E.ARM.OY, ARM_S = E.ARM.S, KV = E.ARM.KV, RX = E.ARM.RX;
  var T = {
    known: 0.87, lack: 1.74, all: 7.81, out1: 9.2,
    origin: 10.85, shoulder: 12.06, notOn: 13.88, how: 14.92, out2: 16.3,
    plate: 18.22, rot: [19.01, 20.6], cover: 22.02, vel: 23.8, vLab: 24.73, plateTxt: 24.33,
    clear3: 25.9, q: [27.22, 27.95], link: 28.68, rel: [30.04, 30.65],
    perp: 33.45, ccw: 34.25, ep3: [36.1, 37.3, 38.55], res: [40.1, 41.99, 42.45],
    times: 45.45, v: [46.76, 47.25],
    clear6: 48.6, omega: 48.78, vv: 49.94, column: 50.84, name: 52.4, nums: [53.37, 53.85, 54.35, 54.84, 55.43, 56.12],
    clear7: 57.4, any: 59.28, plus: 60.64, part: 61.02, qq: 62.94, r90: [65.34, 65.93], x2: [67.09, 67.67],
    add: [69.09, 69.92], zero: 71.07, still: 71.88,
    notS: 75.37, noMove: 77.86, plateHi: 78.77, here: 80.17
  };
  var PHI0 = -14 * D, END = 82.37;
  var PLATE = [-1.2, -2.6, 2.9, 1.75];                        // 透明板在「小臂水平时」的范围（相对肘，格）；转到 −14° 时右下角也不出画面

  E.scenes[2] = {
    title: '运动旋量：角速度 + 线速度',
    dur: END,
    cues: [
      { at: 0, s: '有了**角速度**，再知道一个点的速度，每一点的速度都能算' },
      { at: 9.34, s: '这个点通常选原点（肩关节），可它不在小臂上' },
      { at: 16.49, s: '小臂想成很大的透明板：板上此刻在原点那一点的速度，记作 **v**' },
      { at: 25.99, s: '肘在 (1, 2)；从肘指到原点：(−1, −2)' },
      { at: 32.09, s: '速度和连线垂直：逆时针转 90°，(−1, −2) → (2, −1)' },
      { at: 43.99, s: '再乘角速度 2：**v** = (4, −2)' },
      { at: 48.69, s: '**运动旋量**：**角速度**三个数 + **v** 三个数 = (0, 0, 2, 4, −2, 0)' },
      { at: 57.43, s: '验算：某点的速度 = **v** + 绕原点转的那一份' },
      { at: 69.0, s: '肘：(4, −2) + (−4, 2) = (0, 0)，转轴本来就不动' },
      { at: 74.4, s: '**v** **不是**肩关节的速度，是透明板上此刻在原点那一点的速度' }
    ],
    build: function (K) {
      var s = K.sceneSvg('有了角速度，再知道一个点的速度，刚体上每一点的速度都能算出来；这个点选在原点，也就是肩关节，把小臂想成一张很大的透明板跟着一起转，板上此刻在原点的那一点的速度记作 v；肘在 (1, 2)，从肘指到原点是 (−1, −2)，逆时针转 90 度是 (2, −1)，乘角速度 2 得 v = (4, −2)；角速度和 v 竖着排成一列，就是运动旋量 (0, 0, 2, 4, −2, 0)；验算：肘的速度 = v + (−4, 2) = 0；v 不是肩关节的速度');

      var under = E.el('g', {}, s);
      var arm = E.armAt(s, OX, OY, ARM_S, 330);
      var u = arm.aim(1, 2, 0);
      var pl = E.plane(under, { ox: OX, oy: OY, u: u, x0: -1, x1: 4, y0: -1, y1: 3 });
      var P = pl.P;
      var top = E.el('g', {}, s);
      var V = arm.view, RIGID = ['forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link'];
      var Q = P(1, 2), O = P(0, 0);

      /* ── 透明板：钉在小臂上，跟着绕肘转 ── */
      var plate = E.el('g', {}, top);
      var plateFill = K.paint(E.el('path', { 'stroke-width': 2.5, 'stroke-dasharray': '8 6' }, plate), E.tint(E.BODY, 14, 'transparent'), E.BODY);
      plateFill.style.fillOpacity = 0.9;
      var plateLines = [];
      for (var gx = Math.ceil(PLATE[0]); gx <= PLATE[2]; gx++) plateLines.push({ a: [gx, PLATE[1]], b: [gx, PLATE[3]] });
      for (var gy = Math.ceil(PLATE[1]); gy <= PLATE[3]; gy++) plateLines.push({ a: [PLATE[0], gy], b: [PLATE[2], gy] });
      plateLines.forEach(function (l) { l.el = K.paint(E.el('path', { fill: 'none', 'stroke-width': 1.2 }, plate), null, E.BODY); l.el.style.opacity = 0.45; });
      var plateLab = E.halo(E.txt(plate, 0, 0, '透明板', 20, E.BODY, 'start', 900));

      // 原点：一圈 + 「原点 = 肩关节」+（句 10）「没动」
      var oRing = K.paint(E.el('circle', { cx: O[0], cy: O[1], r: 10, fill: 'none', 'stroke-width': 3 }, top), null, E.HI);
      var oLab = E.halo(E.txt(top, O[0] - 20, O[1] - 14, '原点', 22, E.HI, 'end', 900));
      var oLab2 = E.halo(E.txt(top, O[0] - 20, O[1] + 14, '= 肩关节', 20, E.INK2, 'end', 700));
      var noMove = E.halo(E.txt(top, O[0] - 20, O[1] - 44, '没动', 22, E.ERR, 'end', 900));   // 写在「原点」上面：下面是底座
      // 肘：⊙（角速度朝向你）+ (1, 2)
      var outOf = E.el('g', {}, top);
      K.paint(E.el('circle', { r: 14, 'stroke-width': 3.5 }, outOf), E.S1, E.OMEGA);
      K.paint(E.el('circle', { r: 4.5 }, outOf), E.OMEGA);
      var qLab = E.halo(E.txt(top, Q[0] + 12, Q[1] - 24, '(1, 2)', 22, E.HI, 'start', 900));   // 肘的右上方：左上是验算时「那一份」的箭头
      var still = E.halo(E.txt(top, Q[0] + 20, Q[1] + 40, '不动', 22, E.GOOD, 'start', 900));
      // 肘 → 原点的虚线箭头；原点处的直角
      var link = E.vec(top, E.HI, 3, { head: 13, dash: '7 5' });
      var corner = K.paint(E.el('path', { fill: 'none', 'stroke-width': 2.5 }, top), null, E.INK);
      // 板上此刻在原点的那一点 + v
      var vArr = E.vec(top, E.VEL, 5, { head: 16 });
      var pDot = K.paint(E.el('circle', { r: 7, 'stroke-width': 2.5 }, top), E.VEL, E.S1);
      var vLab = E.halo(E.txt(top, 0, 0, 'v', 26, E.VEL, 'start', 900));
      // 验算：肘上的两支箭头
      var partArr = E.vec(top, E.OMEGA, 4.5, { head: 15, dash: '7 5' });
      var vCopy = E.vec(top, E.VEL, 4.5, { head: 15 });
      var partLab = E.halo(E.txt(top, 0, 0, '(−4, 2)', 20, E.OMEGA, 'middle', 900));
      var vCopyLab = E.halo(E.txt(top, 0, 0, '(4, −2)', 20, E.VEL, 'start', 900));

      /* ══ 右栏 ══ */
      var R = E.el('g', {}, s);
      // 一行几段不同颜色的字：parts = [[字, 颜色, 粗细], …]，从 RX + 10 起挨着往右排（挂到页面后量宽度再摆，见 layoutRows）
      var rows = [];
      function line(g, y, parts, size) {
        var r = { g: E.el('g', {}, g) };
        r.parts = parts.map(function (pt) { var n = E.txt(r.g, RX + 10, y, pt[0], size || 23, pt[1] || E.INK, 'start', pt[2] || 700); n.style.whiteSpace = 'pre'; return n; });
        rows.push(r);
        return r;
      }
      function layoutRows() {
        rows.forEach(function (r) {
          var x = RX + 10;
          r.parts.forEach(function (n) { n.setAttribute('x', x.toFixed(1)); x += n.getComputedTextLength() + 2; });
        });
      }
      // 句 1、2
      var g1 = E.el('g', {}, R);
      var k1 = line(g1, 60, [['已知：', E.INK2], ['角速度 ω', E.OMEGA, 900]]);
      var k1c = E.check(k1.g, RX + 250, 42, 22, E.GOOD, 4);
      var k2 = line(g1, 104, [['还缺：', E.INK2], ['随便一个点的速度', E.VEL, 900]]);
      var k3 = line(g1, 148, [['→ 每一点的速度都能算出来', E.INK, 900]]);
      var k4 = line(g1, 210, [['这个点：', E.INK2], ['原点（肩关节）', E.HI, 900]]);
      var k5 = line(g1, 254, [['可肩关节', E.INK], ['不在', E.ERR, 900], ['小臂上……', E.INK]]);
      var k6 = line(g1, 298, [['怎么办？', E.INK, 900]]);
      // 句 3
      var g2 = E.el('g', {}, R);
      var k7 = line(g2, 60, [['小臂 → 很大的', E.INK], ['透明板', E.BODY, 900]]);
      var k8 = line(g2, 104, [['跟着小臂一起转', E.INK2]]);
      var k9 = line(g2, 160, [['v', E.VEL, 900], ['：板上此刻在原点', E.INK, 900]]);
      var k9b = line(g2, 200, [['　那一点的速度', E.INK, 900]]);
      // 句 4–6：算式
      var g3 = E.el('g', {}, R);
      var c0 = line(g3, 50, [['算 ', E.INK2], ['v', E.VEL, 900], ['：', E.INK2]], 24);
      var c1 = line(g3, 96, [['肘：', E.INK2], ['(1, 2)', E.HI, 900]]);
      var c2 = line(g3, 140, [['肘 → 原点：', E.INK2], ['(−1, ', E.INK, 900], ['−2)', E.INK, 900]]);
      var c3 = line(g3, 184, [['垂直：', E.INK2], ['逆时针转 90°', E.OMEGA, 900]]);
      var c4 = line(g3, 222, [['（第 3 集：', E.MUT], ['(3, 2)', E.MUT], [' → ', E.MUT], ['(−2, 3)', E.MUT], ['）', E.MUT]], 20);
      var c5 = line(g3, 266, [['(−1, −2)', E.INK, 900], [' → ', E.INK2], ['(2, ', E.INK, 900], ['−1)', E.INK, 900]]);
      var c6 = line(g3, 316, [['× 2：', E.OMEGA, 900], ['v', E.VEL, 900], [' = ', E.INK], ['(4, ', E.VEL, 900], ['−2)', E.VEL, 900]], 26);
      // 句 7：六维列向量
      var g4 = E.el('g', {}, R), CX = RX + 200, CY = 226;
      var colv = E.colvec(g4, CX, CY, [0, 0, 2, 4, -2, 0], { size: 28, dy: 36, colors: [E.OMEGA, E.OMEGA, E.OMEGA, E.VEL, E.VEL, E.VEL], w: 70 });
      var wTag = E.txt(g4, CX + 56, CY - 50, 'ω', 28, E.OMEGA, 'start', 900), vTag = E.txt(g4, CX + 56, CY + 66, 'v', 28, E.VEL, 'start', 900);
      var wTag2 = E.txt(g4, CX + 86, CY - 50, '角速度', 20, E.OMEGA, 'start', 700), vTag2 = E.txt(g4, CX + 82, CY + 66, '线速度', 20, E.VEL, 'start', 700);
      var vName = E.txt(g4, CX - 56, CY + 10, '运动旋量 =', 26, E.HI, 'end', 900);
      // 句 8、9：验算
      var g5 = E.el('g', {}, R);
      var e0 = line(g5, 50, [['验算：', E.INK2]], 24);
      var e1 = line(g5, 96, [['某点的速度 = ', E.INK, 900], ['v', E.VEL, 900], [' + 绕原点转的那一份', E.INK, 900]], 20);   // 20 号：空格照排（white-space: pre）以后，22 号会超出右边
      var e2 = line(g5, 134, [['那一份：位置转 90°，再乘 ω', E.OMEGA, 700]], 20);
      var e3 = line(g5, 184, [['肘 ', E.INK2], ['(1, 2)', E.HI, 900], [' → ', E.INK2], ['(−2, 1)', E.INK, 900], [' → ', E.INK2], ['(−4, 2)', E.OMEGA, 900]]);
      var e4 = line(g5, 236, [['(−4, 2)', E.OMEGA, 900], [' + ', E.INK], ['(4, −2)', E.VEL, 900], [' = ', E.INK], ['(0, 0)', E.GOOD, 900]], 24);
      var e4c = E.check(e4.g, RX + 300, 216, 22, E.GOOD, 4);
      // 句 10
      var g6 = E.el('g', {}, R);
      var n1 = line(g6, 300, [['v', E.VEL, 900], [' 不是肩关节的速度', E.ERR, 900]]);
      var n1x = E.cross(n1.g, RX + 250, 292, 10, E.ERR, 4);
      var n2 = line(g6, 346, [['v', E.VEL, 900], [' 是板上此刻在原点那一点的', E.GOOD, 900]], 22);
      var n2b = line(g6, 384, [['　速度', E.GOOD, 900]], 22);
      var n2c = E.check(n2b.g, RX + 92, 368, 20, E.GOOD, 4);

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) { layoutRows(); laid = true; }
          var f = E.fade, b = E.bump;

          /* ── 小臂（连同透明板）从 −20° 转到水平 ── */
          var phi = PHI0 * (1 - f(st, T.rot[0], T.rot[1] - T.rot[0]));
          arm.bend(phi);
          V.clearTint(); V.tint(RIGID, E.BODY, 0.4);
          var c = Math.cos(phi), sn = Math.sin(phi);
          function plateP(x, y) { return P(1 + c * x - sn * y, 2 + sn * x + c * y); }   // 板上一点（相对肘、小臂水平时的坐标）→ 画面
          plateFill.setAttribute('d', E.poly([plateP(PLATE[0], PLATE[1]), plateP(PLATE[2], PLATE[1]), plateP(PLATE[2], PLATE[3]), plateP(PLATE[0], PLATE[3])], true));
          plateLines.forEach(function (l) { l.el.setAttribute('d', E.poly([plateP(l.a[0], l.a[1]), plateP(l.b[0], l.b[1])])); });
          var pl0 = plateP(PLATE[2] - 1.15, PLATE[3] - 0.55); plateLab.setAttribute('x', pl0[0].toFixed(1)); plateLab.setAttribute('y', pl0[1].toFixed(1));
          var plb = b(st, T.plateHi, 1);
          plateFill.style.strokeWidth = 2.5 + 2.5 * plb; E.glow(plateFill, E.BODY, plb > 0.05);
          E.op(plate, f(st, T.plate, 0.6));

          /* ── 原点、肘 ── */
          // 透明板淡入、转到水平的那一段，板的左边扫过「原点 = 肩关节」：两行字先淡下去，转完再亮回来
          var dim = 1 - 0.85 * E.win(st, T.plate - 0.2, T.rot[1] + 0.1, 0.3);
          E.op(oRing, f(st, T.origin, 0.3)); E.op(oLab, f(st, T.origin, 0.3) * dim); E.op(oLab2, f(st, T.shoulder, 0.3) * dim);
          oRing.setAttribute('r', (10 + 5 * Math.max(b(st, T.origin, 0.7), b(st, T.notS, 0.8))).toFixed(2));
          E.op(noMove, f(st, T.noMove, 0.3));
          var e = arm.elbow();
          outOf.setAttribute('transform', 'translate(' + e[0].toFixed(1) + ' ' + e[1].toFixed(1) + ')');
          E.op(qLab, f(st, T.q[1], 0.3));
          E.op(still, f(st, T.still, 0.3));
          var lk = f(st, T.link, 0.5);
          link.set(Q[0], Q[1], Q[0] + (O[0] - Q[0]) * lk, Q[1] + (O[1] - Q[1]) * lk);
          E.op(link.g, lk > 0 ? 1 - f(st, T.clear6, 0.4) : 0);
          // 直角：原点处，「连线方向（指回肘）」和「速度方向 (2, −1)」
          var rr = [1, 2].map(function (x) { return x / Math.sqrt(5); }), vv = [2 / Math.sqrt(5), -1 / Math.sqrt(5)], m = 13 / u;
          corner.setAttribute('d', E.poly([P(rr[0] * m, rr[1] * m), P(rr[0] * m + vv[0] * m, rr[1] * m + vv[1] * m), P(vv[0] * m, vv[1] * m)]));
          E.op(corner, f(st, T.perp, 0.3) * (1 - f(st, T.clear6, 0.4)));

          /* ── 板上在原点的那一点 + v ── */
          var vOn = f(st, T.vel, 0.5), tip = P(KV * W * 2 * vOn, -KV * W * 1 * vOn);
          vArr.set(O[0], O[1], tip[0], tip[1]); E.op(vArr.g, vOn > 0 ? 1 : 0);
          var here = Math.max(b(st, T.cover, 0.8), b(st, T.here, 0.9));
          pDot.setAttribute('cx', O[0]); pDot.setAttribute('cy', O[1]); pDot.setAttribute('r', (7 + 4 * here).toFixed(2));
          E.op(pDot, f(st, T.cover, 0.3)); E.glow(pDot, E.VEL, here > 0.05);
          var vt = P(KV * W * 2, -KV * W);
          vLab.textContent = st >= T.v[1] ? 'v = (4, −2)' : 'v';
          vLab.setAttribute('x', (vt[0] + 10).toFixed(1)); vLab.setAttribute('y', (vt[1] + 22).toFixed(1));
          E.op(vLab, f(st, T.vLab, 0.3));
          var vb = b(st, T.notS, 0.8);
          vArr.line.style.strokeWidth = 5 + 3 * vb;

          /* ── 验算：肘上「那一份」(−4, 2) 和 v 的副本 (4, −2)，加起来是零 ── */
          var gone = 1 - f(st, T.zero + 0.4, 0.5);
          var pp = f(st, T.x2[0], 0.5), pt = P(1 - KV * W * 2 * pp, 2 + KV * W * pp);
          partArr.set(Q[0], Q[1], pt[0], pt[1]); E.op(partArr.g, pp > 0 ? gone : 0);
          var pe = P(1 - KV * W * 2, 2 + KV * W);
          partLab.setAttribute('x', pe[0].toFixed(1)); partLab.setAttribute('y', (pe[1] - 14).toFixed(1));   // 写在箭头尖正上方（尖在 y 轴左边，往左写会出画面）
          E.op(partLab, f(st, T.x2[1], 0.3) * gone);
          var cp = f(st, T.add[0], 0.5), ct = P(1 + KV * W * 2 * cp, 2 - KV * W * cp);
          vCopy.set(Q[0], Q[1], ct[0], ct[1]); E.op(vCopy.g, cp > 0 ? gone : 0);
          var ce = P(1 + KV * W * 2, 2 - KV * W);
          vCopyLab.setAttribute('x', (ce[0] + 8).toFixed(1)); vCopyLab.setAttribute('y', (ce[1] + 20).toFixed(1));
          E.op(vCopyLab, f(st, T.add[1], 0.3) * gone);

          /* ── 右栏 ── */
          E.op(g1, 1 - f(st, T.out2, 0.4));
          E.op(k1.g, fr(st, T.known)); E.drawOn(k1c, f(st, T.known + 0.2, 0.3));
          E.op(k2.g, fr(st, T.lack)); E.op(k3.g, fr(st, T.all) * (1 - f(st, T.out1, 0.3)) + 0);
          E.op(k4.g, fr(st, T.origin)); E.op(k5.g, fr(st, T.notOn)); E.op(k6.g, fr(st, T.how));
          E.op(g2, f(st, T.plate, 0.4) * (1 - f(st, T.clear3, 0.4)));
          E.op(k7.g, 1); E.op(k8.g, fr(st, T.rot[0])); E.op(k9.g, fr(st, T.plateTxt)); E.op(k9b.g, fr(st, T.plateTxt));
          E.op(g3, f(st, T.clear3 + 0.2, 0.4) * (1 - f(st, T.clear6, 0.4)));
          E.op(c0.g, 1); E.op(c1.g, fr(st, T.q[0]));
          E.op(c2.parts[1], fr(st, T.rel[0])); E.op(c2.parts[2], fr(st, T.rel[1])); E.op(c2.parts[0], fr(st, T.link));
          E.op(c3.g, fr(st, T.perp)); E.op(c3.parts[1], fr(st, T.ccw));
          E.op(c4.parts[0], fr(st, T.ep3[0])); E.op(c4.parts[1], fr(st, T.ep3[1])); E.op(c4.parts[2], fr(st, T.ep3[2])); E.op(c4.parts[3], fr(st, T.ep3[2])); E.op(c4.parts[4], fr(st, T.ep3[2]));
          E.op(c5.parts[0], fr(st, T.res[0])); E.op(c5.parts[1], fr(st, T.res[0])); E.op(c5.parts[2], fr(st, T.res[1])); E.op(c5.parts[3], fr(st, T.res[2]));
          E.op(c6.g, fr(st, T.times)); E.op(c6.parts[3], fr(st, T.v[0])); E.op(c6.parts[4], fr(st, T.v[1]));
          // 六维列向量
          E.op(g4, f(st, T.clear6 + 0.2, 0.4) * (1 - f(st, T.clear7, 0.4)));
          colv.rangeBox(0, 2, E.OMEGA, f(st, T.omega, 0.3)); colv.rangeBox(3, 5, E.VEL, f(st, T.vv, 0.3));
          E.op(wTag, fr(st, T.omega)); E.op(wTag2, fr(st, T.omega)); E.op(vTag, fr(st, T.vv)); E.op(vTag2, fr(st, T.vv));
          E.op(vName, fr(st, T.name));
          colv.cells.forEach(function (cl, i) { E.op(cl, f(st, T.nums[i], 0.3)); });   // 只淡入、不放大：上下两个框紧挨着数
          // 验算
          E.op(g5, f(st, T.clear7 + 0.2, 0.4));
          E.op(e0.g, 1); E.op(e1.g, fr(st, T.any)); E.op(e1.parts[2], fr(st, T.plus)); E.op(e2.g, fr(st, T.part));
          E.op(e3.parts[0], fr(st, T.qq)); E.op(e3.parts[1], fr(st, T.qq));
          E.op(e3.parts[2], fr(st, T.r90[0])); E.op(e3.parts[3], fr(st, T.r90[0]));
          E.op(e3.parts[4], fr(st, T.x2[0])); E.op(e3.parts[5], fr(st, T.x2[0]));
          E.op(e4.g, fr(st, T.add[0])); E.op(e4.parts[3], fr(st, T.zero)); E.op(e4.parts[4], fr(st, T.zero)); E.drawOn(e4c, f(st, T.zero + 0.2, 0.3));
          E.op(g6, 1);
          E.op(n1.g, fr(st, T.notS)); E.drawOn(n1x, f(st, T.noMove, 0.3));
          E.op(n2.g, fr(st, T.plateHi)); E.op(n2b.g, fr(st, T.plateHi)); E.drawOn(n2c, f(st, T.here, 0.3));
        }
      };
    }
  };
})();
