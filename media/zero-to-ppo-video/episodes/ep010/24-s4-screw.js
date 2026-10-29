// 010 第 4 幕：三维——边转边移的位姿。左边是斜着看的三维小场景（z 朝上）：竖直的螺旋轴（琥珀虚线）穿过原点，离轴 0.8 格的小方块带着自己的三根小箭头；
// 右栏上方是 4 × 4 的 T，下方一行行写公式。
// 句 1「四乘四的位姿」T 出来（左上 3 × 3 与右边一列先是问号，最下面一排 0、0、0、1），「左上角三乘三是朝向」框紫，「右边一列是位置」框蓝；
// 句 2「螺旋轴的方向」轴亮一下、顶上出方向箭头 ŝ，「转过这个角度」轴上出转动的小弯箭头、标 θ；「罗德里格斯公式」右栏出公式，「正弦和余弦」下一行写；
// 句 3「位置分两份」右栏出 p = (I − R)q + θhŝ：「轴上的点转完不动」第一份下面标字、轴上的原点亮一下，「角度乘螺距」第二份下面标字、轴旁出往上的箭头；
// 句 4「拧螺丝」画出一圈多（450°）的螺旋线，「竖直向上」「穿过原点」「每转一圈，往上走一格」右栏三行，螺旋线起点到一圈后量出 1 格；
// 句 5「转四分之一圈」方块沿螺旋线走 90°，「转了九十度」T 的左上角填数，「四分之一格」右边一列填 0、0、0.25；
// 句 6「转一整圈」方块接着走到 360°（T 跟着变），「单位矩阵」左上角框亮，「一格」右边一列框亮；
// 句 7「不管转多少」方块接着走到 450°（T 跟着变），「每一列的长度是一、两两垂直」「航列式是一」右栏两行打勾。
// 时刻取自配音的逐词时间，见 ../ep010.py。
(function () {
  var E = window.EP010, K = E.K, H = E.HEX;
  var D = Math.PI / 180, RX = E.ARM.RX;
  var T = {
    mat: 3.36, rot: 4.52, pos: 6.32,
    axis: 9.93, dir: 10.53, turn: 11.1, ang: 11.78, rod: 14.29, rodName: 15.8, sc: 17.26, out2: 19.9,
    two: 20.18, part1: 21.33, still: 23.52, part2: 24.38, hpart: 25.83, out3: 27.8,
    ex: 28.07, up: 29.56, orig: 31.02, pitch: 32.08, turn1: 32.73, one: 34.1,
    q1: [35.59, 36.6], r90: 38.62, p25: 40.94,
    full: [43.03, 44.3], eye: 45.28, p1: 47.99,
    any: [49.88, 53.0], rule1: 54.18, rule1b: 56.56, rule2: 57.31, rule2b: 58.14
  };
  var END = 59.23, HP = 1 / (2 * Math.PI), RAD = 0.8, CUBE = 0.17;
  var proj = E.view3(28, 18), WX = 205, WY = 352, L = 150;
  function S3(v) { var q = proj(v); return [WX + L * q[0], WY - L * q[1]]; }
  function th(st) {
    return Math.PI / 2 * E.fade(st, T.q1[0], T.q1[1] - T.q1[0]) + 1.5 * Math.PI * E.fade(st, T.full[0], T.full[1] - T.full[0]) +
      Math.PI / 2 * E.fade(st, T.any[0], T.any[1] - T.any[0]);
  }

  E.scenes[3] = {
    title: '三维：边转边移的位姿',
    dur: END,
    cues: [
      { at: 0, s: '三维也一样：**矩阵指数**给出 4 × 4 的**位姿**，左上角 3 × 3 是**朝向**，右边一列是**位置**' },
      { at: 8.5, s: '**朝向**：绕**螺旋轴**的方向转过这个**角度**，罗德里格斯公式只用到 sin、cos' },
      { at: 20.09, s: '**位置**两份：轴上的点转完不动 + **角度** × **螺距**（沿着轴走）' },
      { at: 27.98, s: '拧螺丝：**螺旋轴**竖直向上、穿过原点，每转一圈往上走一格' },
      { at: 35.5, s: '转 1/4 圈：**朝向**绕竖直的轴转 90°，**位置**往上 1/4 格' },
      { at: 42.94, s: '转一整圈：**朝向**回到原样（单位矩阵），**位置**往上 1 格' },
      { at: 49.42, s: '不管转多少：每列长度 1、两两垂直，行列式是 1' }
    ],
    build: function (K) {
      var s = K.sceneSvg('三维里矩阵指数给出四乘四的位姿：左上角三乘三是朝向，绕螺旋轴方向转过这个角度，用罗德里格斯公式算；右边一列是位置，分两份：让轴上的点转完不动，再加上角度乘螺距；例子：螺旋轴竖直向上穿过原点，每转一圈往上走一格，转四分之一圈朝向转九十度、往上四分之一格，转一整圈朝向回到单位矩阵、往上一格；算出来的朝向总守着正交、行列式为一两条规矩');

      /* ══ 左：三维 ══ */
      var g3 = E.el('g', {}, s);
      var ground = K.paint(E.el('path', { fill: 'none', 'stroke-width': 1.8, 'stroke-dasharray': '5 6' }, g3), null, E.MUT);
      var gp = []; for (var k = 0; k <= 48; k++) { var a = 2 * Math.PI * k / 48; gp.push(S3([RAD * Math.cos(a), RAD * Math.sin(a), 0])); }
      ground.setAttribute('d', E.poly(gp, true));
      var hBack = K.paint(E.el('path', { fill: 'none', 'stroke-width': 2.2, 'stroke-dasharray': '4 6' }, g3), null, E.ACC);
      hBack.style.opacity = 0.55;
      var a0 = S3([0, 0, -0.12]), a1 = S3([0, 0, 1.95]);
      var axis = K.paint(E.el('path', { d: E.poly([a0, a1]), fill: 'none', 'stroke-width': 3.5, 'stroke-dasharray': '9 7' }, g3), null, E.WARN);
      var sHat = E.vec(g3, E.WARN, 5, { head: 16 }), sh0 = S3([0, 0, 1.55]), sh1 = S3([0, 0, 2.02]);
      sHat.set(sh0[0], sh0[1], sh1[0], sh1[1]);
      var sHatLab = E.halo(E.txt(g3, sh1[0] + 14, sh1[1] + 18, 'ŝ', 26, E.WARN, 'start', 900));
      var axLab = E.halo(E.txt(g3, a1[0] - 16, a1[1] + 14, '螺旋轴', 22, E.WARN, 'end', 900));   // 轴顶左边：方块最高走到 1.25 格
      var hFront = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3.2 }, g3), null, E.ACC);
      var O = S3([0, 0, 0]);
      var oDot = K.paint(E.el('circle', { cx: O[0], cy: O[1], r: 6.5, 'stroke-width': 2.5 }, g3), E.HI, E.S1);
      var oLab = E.halo(E.txt(g3, O[0] - 14, O[1] + 28, '原点', 20, E.HI, 'end', 900));
      // 句 2：绕轴转的小弯箭头（画成轴上一圈椭圆的一段）
      var spin = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3.5 }, g3), null, E.VIOLET), spinHead = K.paint(E.el('path', {}, g3), E.VIOLET);
      var thLab = E.halo(E.txt(g3, 0, 0, 'θ', 26, E.VIOLET, 'start', 900));
      thLab.style.fontFamily = 'KaTeX_Math, serif'; thLab.style.fontStyle = 'italic';
      // 句 3：沿轴往上的箭头
      var rise = E.vec(g3, E.GOOD, 5, { head: 15 });
      var riseLab = E.halo(E.txt(g3, 0, 0, 'θ × 螺距', 20, E.GOOD, 'end', 900));
      // 句 4：一圈后上 1 格的量尺
      var dimG = E.el('g', {}, g3), d0 = S3([RAD, 0, 0]), d1 = S3([RAD, 0, 1]);
      // 量尺离方块远一点：方块起步时它的 x 小箭头朝右伸出 0.36 格
      K.paint(E.el('path', { d: 'M ' + (d0[0] + 46) + ' ' + d0[1].toFixed(1) + ' H ' + (d0[0] + 58) + ' M ' + (d0[0] + 52) + ' ' + d0[1].toFixed(1) + ' V ' + d1[1].toFixed(1) + ' M ' + (d1[0] + 46) + ' ' + d1[1].toFixed(1) + ' H ' + (d1[0] + 58),
        fill: 'none', 'stroke-width': 2.5 }, dimG), null, E.GOOD);
      E.halo(E.txt(dimG, d0[0] + 66, (d0[1] + d1[1]) / 2 + 8, '1 格', 22, E.GOOD, 'start', 900));
      E.halo(E.txt(dimG, d0[0] + 66, (d0[1] + d1[1]) / 2 + 36, '（一圈）', 18, E.INK2, 'start', 700));
      // 小方块 + 自己的三根小箭头
      var block = K.paint(E.el('path', { 'stroke-width': 2.5, 'stroke-linejoin': 'round' }, g3), E.tint(E.VIOLET, 40, 'var(--demo-surface)'), E.VIOLET);
      var giz = E.gizmo(g3, WX, WY, L, proj, { w: 3.5 });
      giz.labels.forEach(function (n) { n.style.display = 'none'; });   // 方块小，字母会压在方块和箭头上：三根小箭头照 005 的配色（x 绿、y 琥珀、z 蓝）

      /* ══ 右栏：T + 公式 ══ */
      var RK = E.rowKit(), Rg = E.el('g', {}, s), X = RX + 10, MX = RX + 206, MY = 96;
      var mG = E.el('g', {}, Rg);
      E.tex(mG, X, MY + 10, 'T=', 30, E.INK, 'start', { w: 60, h: 50 });
      var Q4 = [['?', '?', '?', '?'], ['?', '?', '?', '?'], ['?', '?', '?', '?'], [0, 0, 0, 1]];
      var M = E.matN(mG, MX, MY, Q4, { size: 22, dx: 66, dy: 33, cw: 60, pad: 34, color: function (i, j) { return i === 3 ? E.MUT : j < 3 ? E.VIOLET : E.ACC; } });
      M.cells.forEach(function (row) { row.forEach(function (n) { n.__col = n.style.fill; }); });
      var rotLab = E.txt(mG, MX - 33, MY + 102, '朝向', 22, E.VIOLET, 'middle', 900);
      var posLab = E.txt(mG, MX + 99, MY + 102, '位置', 22, E.ACC, 'middle', 900);
      var ROWS = 236;
      var g2 = E.el('g', {}, Rg);
      var b1 = RK.line(g2, X, ROWS, [['朝向：绕 ', E.INK2], ['ŝ', E.WARN, 900], [' 转过角度 ', E.INK2], ['θ', E.VIOLET, 900]]);
      var b2 = E.el('g', {}, g2);
      E.tex(b2, X, ROWS + 56, 'R=I+\\textcolor{' + H.good + '}{\\sin\\theta}\\,[\\hat s]+(1-\\textcolor{' + H.good + '}{\\cos\\theta})\\,[\\hat s]^2', 21, E.VIOLET, 'start', { w: 320, h: 44 });
      var b3 = RK.line(g2, X, ROWS + 104, [['罗德里格斯公式', E.VIOLET, 900]], 22);
      var b4 = RK.line(g2, X, ROWS + 140, [['只用到 ', E.INK2], ['sin、cos', E.GOOD, 900]], 21);
      var b4b = RK.line(g2, X, ROWS + 170, [['[ŝ]：把 ŝ 的三个数排成 3 × 3', E.MUT, 700]], 17);
      var g3r = E.el('g', {}, Rg);
      var c1 = E.el('g', {}, g3r);
      E.tex(c1, X, ROWS + 16, 'p=\\textcolor{' + H.acc + '}{(I-R)\\,q}+\\textcolor{' + H.good + '}{\\theta\\,h\\,\\hat s}', 28, E.INK, 'start', { w: 300, h: 56 });
      var c2 = RK.line(g3r, X, ROWS + 74, [['① ', E.ACC, 900], ['轴上的点 q 转完不动', E.INK, 900]], 21);
      var c3 = RK.line(g3r, X, ROWS + 112, [['② ', E.GOOD, 900], ['沿着轴走：角度 × 螺距', E.INK, 900]], 21);
      var c4 = RK.line(g3r, X, ROWS + 150, [['原书写作 G(θ)v，展开就是这两份', E.MUT, 700]], 17);
      var g4 = E.el('g', {}, Rg);
      var e1 = RK.line(g4, X, ROWS + 6, [['ŝ', E.WARN, 900], ['：竖直向上', E.INK, 900]]);
      var e2 = RK.line(g4, X, ROWS + 46, [['q', E.HI, 900], ['：原点', E.INK, 900]]);
      var e3 = RK.line(g4, X, ROWS + 86, [['螺距：每转一圈，往上 ', E.INK, 900], ['1 格', E.GOOD, 900]]);
      var e4 = RK.line(g4, X, ROWS + 136, [['转 1/4 圈：', E.INK2], ['转 90°', E.VIOLET, 900], ['，往上 ', E.INK2], ['0.25', E.ACC, 900]]);
      var e5 = RK.line(g4, X, ROWS + 136, [['转一整圈：', E.INK2], ['单位矩阵', E.VIOLET, 900], ['，往上 ', E.INK2], ['1 格', E.ACC, 900]]);
      var g5 = E.el('g', {}, Rg);
      var f1 = RK.line(g5, X, ROWS + 6, [['不管转多少，朝向都守规矩：', E.INK, 900]], 22);
      var f2 = RK.line(g5, X, ROWS + 50, [['① 每一列长度 1、两两垂直', E.INK, 900]], 21);
      var f2ok = E.check(f2.g, X + 286, ROWS + 32, 22, E.GOOD, 4);
      var f3 = RK.line(g5, X, ROWS + 92, [['② 行列式 = 1', E.INK, 900]], 21);
      var f3ok = E.check(f3.g, X + 150, ROWS + 74, 22, E.GOOD, 4);

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) { RK.layout(); laid = true; }
          var f = E.fade, b = E.bump, ang = th(st);

          /* ── 螺旋线：一圈半；句 4 起画出 ── */
          var hp = f(st, T.ex, 1.6), thMax = 2.5 * Math.PI * hp;
          var fr_ = '', bk = '', prev = null, n = Math.max(2, Math.ceil(thMax / 0.05));
          for (var k = 0; k <= n; k++) {
            var aa = thMax * k / n, p = S3([RAD * Math.cos(aa), RAD * Math.sin(aa), HP * aa]), isF = proj.depth([Math.cos(aa), Math.sin(aa), 0]) < 0, sg = p[0].toFixed(1) + ' ' + p[1].toFixed(1);
            if (isF) fr_ += (prev === true ? ' L ' : ' M ') + sg; else bk += (prev === false ? ' L ' : ' M ') + sg;
            if (prev !== null && prev !== isF) { if (isF) bk += ' L ' + sg; else fr_ += ' L ' + sg; }
            prev = isF;
          }
          hFront.setAttribute('d', thMax > 0.02 ? fr_ : ''); hBack.setAttribute('d', thMax > 0.02 ? bk : '');
          E.op(ground, 0.5);
          var ab = Math.max(b(st, T.axis, 0.9), b(st, T.up, 0.9));
          axis.style.strokeWidth = 3.5 + 2.5 * ab; E.glow(axis, E.WARN, ab > 0.05);
          E.op(sHat.g, f(st, T.dir, 0.3)); E.op(sHatLab, f(st, T.dir, 0.3)); E.op(axLab, f(st, T.axis, 0.3));
          oDot.setAttribute('r', (6.5 + 4 * Math.max(b(st, T.still, 0.8), b(st, T.orig, 0.8))).toFixed(2));
          E.op(oLab, f(st, T.still, 0.3));
          E.op(dimG, f(st, T.one, 0.3) * (1 - f(st, T.any[0], 0.4)));

          /* ── 句 2：小弯箭头 θ（轴上 1.25 格高处一圈椭圆的 −30° → 70°） ── */
          var sp = f(st, T.turn, 0.6), z0 = 1.3, r0 = 0.42, s0 = -30 * D, s1 = s0 + 100 * D * sp;
          var spts = []; for (k = 0; k <= 24; k++) { var q = s0 + (s1 - s0) * k / 24; spts.push(S3([r0 * Math.cos(q), r0 * Math.sin(q), z0])); }
          spin.setAttribute('d', sp > 0.02 ? E.poly(spts) : '');
          var tip = spts[24], pre = spts[22], tx = tip[0] - pre[0], ty = tip[1] - pre[1], tl = Math.hypot(tx, ty) || 1;
          tx /= tl; ty /= tl;
          spinHead.setAttribute('d', sp > 0.02 ? 'M ' + (tip[0] + tx * 6).toFixed(1) + ' ' + (tip[1] + ty * 6).toFixed(1) + ' L ' + (tip[0] - tx * 8 - ty * 7).toFixed(1) + ' ' + (tip[1] - ty * 8 + tx * 7).toFixed(1) +
            ' L ' + (tip[0] - tx * 8 + ty * 7).toFixed(1) + ' ' + (tip[1] - ty * 8 - tx * 7).toFixed(1) + ' Z' : '');
          var son = f(st, T.turn, 0.3) * (1 - f(st, T.ex, 0.4));
          E.op(spin, son); E.op(spinHead, son);
          var tlp = S3([0.62, -0.2, z0]); thLab.setAttribute('x', tlp[0].toFixed(1)); thLab.setAttribute('y', (tlp[1] + 8).toFixed(1));
          E.op(thLab, f(st, T.ang, 0.3) * (1 - f(st, T.ex, 0.4)));
          // 句 3：沿轴往上（画在轴左边一点）
          var rp = f(st, T.part2, 0.5), r0p = S3([-0.18, 0.12, 0.35]), r1p = S3([-0.18, 0.12, 0.35 + 0.7 * rp]);
          rise.set(r0p[0], r0p[1], r1p[0], r1p[1]); E.op(rise.g, (rp > 0 ? 1 : 0) * (1 - f(st, T.ex, 0.4)));
          var rl = S3([-0.18, 0.12, 0.7]); riseLab.setAttribute('x', (rl[0] - 14).toFixed(1)); riseLab.setAttribute('y', (rl[1] + 6).toFixed(1));
          E.op(riseLab, f(st, T.hpart, 0.3) * (1 - f(st, T.ex, 0.4)));

          /* ── 小方块：中心 = R (0.8, 0, 0) + (0, 0, h θ)，朝向 R = 绕竖直轴转 θ ── */
          var R = E.rot3([0, 0, 1], ang), c = E.mv3(R, [RAD, 0, 0]); c[2] += HP * ang;
          var cs = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(function (w) { var v = E.mv3(R, [w[0] * CUBE, w[1] * CUBE, 0]); return [c[0] + v[0], c[1] + v[1], c[2]]; });
          var topF = cs.map(function (p) { return S3([p[0], p[1], p[2] + CUBE]); }), botF = cs.map(function (p) { return S3([p[0], p[1], p[2] - CUBE]); });
          var d = '';
          for (k = 0; k < 4; k++) {
            var j = (k + 1) % 4, mid = [(cs[k][0] + cs[j][0]) / 2 - c[0], (cs[k][1] + cs[j][1]) / 2 - c[1], 0];
            if (proj.depth(mid) < 0) d += ' M ' + [topF[k], topF[j], botF[j], botF[k]].map(function (p) { return p[0].toFixed(1) + ' ' + p[1].toFixed(1); }).join(' L ') + ' Z';
          }
          d += ' M ' + topF.map(function (p) { return p[0].toFixed(1) + ' ' + p[1].toFixed(1); }).join(' L ') + ' Z';
          block.setAttribute('d', d);
          giz.set(R.map(function (row) { return row.map(function (x) { return 0.36 * x; }); }), [c[0], c[1], c[2] + CUBE]);

          /* ── 右栏：T ── */
          E.pop(mG, MX, MY, f(st, T.mat, 0.35), 0.9);
          var known = st >= T.r90;
          var Tm = [[R[0][0], R[0][1], R[0][2], 0], [R[1][0], R[1][1], R[1][2], 0], [R[2][0], R[2][1], R[2][2], HP * ang]];
          for (var i = 0; i < 3; i++) for (j = 0; j < 4; j++) {
            var cell = M.cells[i][j], on = j < 3 ? known : st >= T.p25, pq = M.pos(i, j), pc = f(st, j < 3 ? T.r90 : T.p25, 0.3);
            cell.textContent = on ? E.num(Math.abs(Tm[i][j]) < 5e-3 ? 0 : Tm[i][j]) : '?';
            cell.style.fill = on ? cell.__col : E.MUT;
            E.scaleAt(cell, pq[0], pq[1], on ? 1 + 0.2 * (1 - pc) : 1); E.op(cell, on ? pc : 1);
          }
          M.hl('rot', 0, 0, 2, 2, E.VIOLET, f(st, T.rot, 0.3) * (1 - f(st, T.dir, 0.4)) + b(st, T.r90, 0.9) + b(st, T.eye, 1.0) + 0.6 * b(st, T.rule1, 1.0));
          M.hl('pos', 0, 3, 2, 3, E.ACC, f(st, T.pos, 0.3) * (1 - f(st, T.dir, 0.4)) + f(st, T.two, 0.3) * (1 - f(st, T.out3, 0.4)) + b(st, T.p25, 0.9) + b(st, T.p1, 1.0));
          E.op(rotLab, f(st, T.rot, 0.3)); E.op(posLab, f(st, T.pos, 0.3));

          /* ── 右栏：公式 ── */
          E.op(g2, fr(st, 8.59) * (1 - f(st, T.out2, 0.4)));
          E.op(b1.g, 1); E.op(b1.parts[1], fr(st, T.dir)); E.op(b1.parts[2], fr(st, T.turn)); E.op(b1.parts[3], fr(st, T.ang));
          E.op(b2, fr(st, T.rod)); E.op(b3.g, fr(st, T.rodName)); E.op(b4.g, fr(st, T.sc)); E.op(b4b.g, fr(st, T.sc + 0.4));
          E.op(g3r, fr(st, T.two) * (1 - f(st, T.out3, 0.4)));
          E.op(c1, 1); E.op(c2.g, fr(st, T.part1)); E.op(c3.g, fr(st, T.part2)); E.op(c4.g, fr(st, T.hpart + 0.4));
          E.op(g4, fr(st, T.ex) * (1 - f(st, T.any[0] - 0.3, 0.4)));
          E.op(e1.g, fr(st, T.up)); E.op(e2.g, fr(st, T.orig)); E.op(e3.g, fr(st, T.pitch)); E.op(e3.parts[1], fr(st, T.one));
          E.op(e4.g, fr(st, T.q1[0]) * (1 - f(st, T.full[0], 0.3))); E.op(e4.parts[3], fr(st, T.p25));
          E.op(e5.g, fr(st, T.full[0] + 0.3)); E.op(e5.parts[1], fr(st, T.eye)); E.op(e5.parts[3], fr(st, T.p1));
          E.op(g5, fr(st, T.any[0]));
          E.op(f1.g, 1); E.op(f2.g, fr(st, T.rule1)); E.drawOn(f2ok, f(st, T.rule1b, 0.3)); E.op(f3.g, fr(st, T.rule2)); E.drawOn(f3ok, f(st, T.rule2b, 0.3));
        }
      };
    }
  };
})();
