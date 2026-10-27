// 005 第 2 幕：规矩一——每一列长度为 1，两列互相垂直（正交）。左边网格接着上一幕：转 36.87° 的两根小箭头，虚线单位圆垫底；右栏顶上一块规矩牌，下面矩阵和算式。
// 句 1「规矩」出规矩牌，「长度」两根小箭头鼓一下、写上「1」，「垂直」原点标直角；
// 句 2「勾股定理」往右那根下面画出直角三角形（横边 0.8 沿 x 轴、竖边 0.6 虚线），右栏矩阵「第一列」框亮，
//    「零点八的平方」「加零点六的平方」「零点六四」「零点三六」「正好是一」算式一段段写出来，最后打勾；
// 句 3「第二列」往上那根同样画三角形、写一行算式打勾；「夹角」两根之间画角弧，「九十度」写 90°；
// 句 4 网格从旋转变成 003 的「横着拉长」，矩阵换成 [[2, 0], [0, 1]]，「不合格」橙叉，「第一列」框亮，「长度是二」往右那根写橙色「2」，伸出单位圆；
// 句 5 再变成「斜着推一下」[[1, 1], [0, 1]]，橙叉，「垂直」两根之间画 45° 的橙色角弧；
// 句 6 变回旋转矩阵，「正交」规矩牌下出大字「正交」，「书上常写成」一行，「R 的转置乘 R」「等于」式子一段段写出，「单位矩阵」式子右边画出第 1 幕讲过的单位矩阵，
//    「同一件事」规矩牌和式子一起亮一下。
// 时刻取自配音的逐词时间（分镜时间 = 念到这个词的时刻），见 ../ep005.py。
(function () {
  var E = window.EP005, K = E.K, H = E.HEX;
  var OX = 190, OY = 215, U = 80;                           // 网格：x −2…3、y −2…2（横着拉长到 2 倍时往右那根还在格子里）
  var MX = 622, MY = 128, EX = 470;                         // 右栏：矩阵中心、算式左缘
  var Rm = [0.8, -0.6, 0.6, 0.8], S = [2, 0, 0, 1], Hm = [1, 1, 0, 1];
  var T = {
    rule: 1.94, len: 3.29, perp: 5.06,
    tri1: 6.83, col1: 7.79, eq: [8.38, 9.52, 9.74, 10.87, 11.77, 12.88], ok1: 13.49,
    col2: 14.87, eq2: 15.35, triOut: 16.3, arc: 16.84, deg: 17.95,
    toS: [20.35, 0.8], bad1: 21.45, c1: 22.47, two: 24.11,
    toH: [26.25, 0.75], bad2: 27.16, arc2: 27.92, notp: 28.58,
    toR: [30.17, 0.7], ortho: 31.11, book: 32.1, f: [33.5, 34.45, 34.89], unit: 35.17, same: 36.14
  };

  function lerp4(A, B, p) { return A.map(function (a, i) { return a + (B[i] - a) * p; }); }
  function matAt(st) {
    var f = E.fade;
    if (st < T.toS[0]) return Rm;
    if (st < T.toH[0]) return lerp4(Rm, S, f(st, T.toS[0], T.toS[1]));
    if (st < T.toR[0]) return lerp4(S, Hm, f(st, T.toH[0], T.toH[1]));
    return lerp4(Hm, Rm, f(st, T.toR[0], T.toR[1]));
  }
  function bump(st, a, d) { var u = K.seg(st, a, a + d); return u > 0 && u < 1 ? Math.sin(Math.PI * u) : 0; }

  E.scenes[1] = {
    title: '规矩一：长度为 1，互相垂直',
    dur: 38.25,
    cues: [
      { at: 0, s: '规矩一：每一列的长度都是 1，两列互相垂直' },
      { at: 6.48, s: '勾股定理：$0.8^2+0.6^2 = 0.64+0.36 = 1$' },
      { at: 14.78, s: '第二列长度也是 1；两列的夹角正好 90°' },
      { at: 19.58, s: '横着拉长：第一列 (2, 0)，长度是 2 → **不合格**' },
      { at: 26.16, s: '斜着推一下：两列不再垂直 → **不合格**' },
      { at: 30.07, s: '这条规矩叫**正交**，书上写成 $R^\\top R = I$' }
    ],
    build: function (K) {
      var s = K.sceneSvg('旋转矩阵的第一条规矩：每一列长度都是 1，两列互相垂直；用勾股定理验算 0.8 的平方加 0.6 的平方等于 1；横着拉长的矩阵第一列长度是 2，斜着推一下的矩阵两列不垂直，都不合格；这条规矩叫正交，写成 R 的转置乘 R 等于单位矩阵');

      /* ══ 左：网格 ══ */
      var pl = E.plane(s, { ox: OX, oy: OY, u: U, x0: -2, x1: 3, y0: -2, y1: 2 });
      var P = pl.P;
      E.op(pl.warp.g, 0.6);
      var unit = K.paint(E.el('circle', { cx: OX, cy: OY, r: U, fill: 'none', 'stroke-width': 2, 'stroke-dasharray': '6 6' }, s), null, E.HI);
      unit.style.strokeOpacity = 0.55;
      // 直角三角形：t[0] 往右那根、t[1] 往上那根（横边沿 x 轴、竖边虚线、淡色填充）
      var tris = [E.X, E.Y].map(function (c) {
        var g = E.el('g', {}, s);
        var fill = K.paint(E.el('path', {}, g), E.tint(c, 18, 'var(--demo-surface)'));
        var base = K.paint(E.el('path', { 'stroke-width': 6, 'stroke-linecap': 'round' }, g), null, c);
        base.style.strokeOpacity = 0.7;
        var up = K.paint(E.el('path', { 'stroke-width': 2.5, 'stroke-dasharray': '5 5' }, g), null, c);
        var lx = E.halo(E.txt(g, 0, 0, '', 20, c, 'middle', 900)), ly = E.halo(E.txt(g, 0, 0, '', 20, c, 'start', 900));
        return { g: g, fill: fill, base: base, up: up, lx: lx, ly: ly };
      });
      var arc = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3 }, s), null, E.HI);
      var arcLab = E.halo(E.txt(s, 0, 0, '', 22, E.HI, 'middle', 900));
      var e1 = E.vec(s, E.X, 5, { head: 16 }), e2 = E.vec(s, E.Y, 5, { head: 16 });
      var corner = K.paint(E.el('path', { fill: 'none', 'stroke-width': 2.5 }, s), null, E.INK);
      var ones = [E.X, E.Y].map(function (c) { return E.halo(E.txt(s, 0, 0, '1', 22, c, 'middle', 900)); });
      var two = E.halo(E.txt(s, 0, 0, '长度 2', 22, E.ERR, 'middle', 900));

      /* ══ 右栏 ══ */
      var rule = E.tag(s, 452, 12, 340, 44, '规矩一：每列长度 1、互相垂直', E.GOOD, { size: 20, ink: E.INK, weight: 900, rx: 12, sw: 2.5 });
      var m = E.mat(s, MX, MY, Rm, { size: 30, dx: 84, dy: 44, pad: 46, cw: 72, colors: [E.X, E.Y, E.X, E.Y] });
      var mLab = E.txt(s, MX, 200, '', 22, E.INK2, 'middle', 900);
      var badX = E.cross(s, MX + 96, 193, 10, E.ERR, 4.5);
      function eqLine(y) { return E.tex(s, EX, y, '', 28, E.INK, 'start', { w: 320, h: 46 }); }
      var eqA = eqLine(250), eqB = eqLine(296), eqC = eqLine(352);
      var okB = E.check(s, 0, 0, 22, E.GOOD, 4), okC = E.check(s, 0, 0, 22, E.GOOD, 4);
      var rowsBad = [E.txt(s, EX, 262, '', 24, E.INK, 'start', 700), E.txt(s, EX, 306, '', 24, E.ERR, 'start', 900)];
      var big = E.tag(s, MX - 70, 222, 140, 50, '正交', E.GOOD, { size: 30, ink: E.GOOD, weight: 900, rx: 14, sw: 3 });
      var book = E.txt(s, EX, 312, '书上常写成：', 22, E.INK2, 'start', 700);
      var form = E.tex(s, EX + 4, 362, '', 34, E.INK, 'start', { w: 190, h: 56 });
      var unitG = E.el('g', {}, s);                         // 单位矩阵 I（第 1 幕讲过）：斜线上 1、别处 0
      E.mat(unitG, 716, 350, [1, 0, 0, 1], { size: 24, dx: 42, dy: 32, pad: 28, cw: 34, colors: [E.HI, E.MUT, E.MUT, E.HI] });
      E.txt(unitG, 716, 408, '单位矩阵 I', 20, E.HI, 'middle', 900);

      function arcPath(a0, a1, r) {
        var p = P(r * Math.cos(a0), r * Math.sin(a0)), q = P(r * Math.cos(a1), r * Math.sin(a1));
        return 'M ' + p[0].toFixed(1) + ' ' + p[1].toFixed(1) + ' A ' + (r * U).toFixed(1) + ' ' + (r * U).toFixed(1) + ' 0 0 0 ' + q[0].toFixed(1) + ' ' + q[1].toFixed(1);
      }

      return {
        el: s,
        draw: function (st) {
          var f = E.fade, M = matAt(st);
          pl.warp.set(M);
          var a = [M[0], M[2]], b = [M[1], M[3]], o = P(0, 0), q1 = P(a[0], a[1]), q2 = P(b[0], b[1]);
          e1.set(o[0], o[1], q1[0], q1[1]); e2.set(o[0], o[1], q2[0], q2[1]);
          var bl = bump(st, T.len, 0.7);
          e1.line.style.strokeWidth = e2.line.style.strokeWidth = 5 + 3 * bl;
          E.glow(e1.g, E.X, bl > 0.05); E.glow(e2.g, E.Y, bl > 0.05);

          /* 句 1：「1」与直角（只在是旋转矩阵时显示：拉长 / 斜推前淡出，变回旋转后再出来） */
          var isRot = 1 - f(st, T.toS[0] - 0.4, 0.3) + f(st, T.toR[0] + T.toR[1], 0.3);
          [a, b].forEach(function (v, i) {
            var mid = P(v[0] * 0.55, v[1] * 0.55), n = i ? [-v[1], v[0]] : [v[1], -v[0]];
            ones[i].setAttribute('x', (mid[0] + 18 * n[0]).toFixed(1)); ones[i].setAttribute('y', (mid[1] - 18 * n[1] + 8).toFixed(1));
            E.op(ones[i], f(st, T.len, 0.3) * isRot * (1 - f(st, T.tri1 - 0.2, 0.3)) + f(st, T.ortho, 0.3));
          });
          var c1 = P(a[0] * 0.2, a[1] * 0.2), c2 = P(a[0] * 0.2 + b[0] * 0.2, a[1] * 0.2 + b[1] * 0.2), c3 = P(b[0] * 0.2, b[1] * 0.2);
          corner.setAttribute('d', 'M ' + c1[0].toFixed(1) + ' ' + c1[1].toFixed(1) + ' L ' + c2[0].toFixed(1) + ' ' + c2[1].toFixed(1) + ' L ' + c3[0].toFixed(1) + ' ' + c3[1].toFixed(1));
          E.op(corner, f(st, T.perp, 0.3) * isRot);
          var gl = bump(st, T.perp, 0.8) + bump(st, T.ortho, 0.9) + bump(st, T.same, 1.0);
          E.glow(corner, E.HI, gl > 0.05); corner.style.strokeWidth = 2.5 + 2 * Math.min(1, gl);

          /* 句 2、3：两个直角三角形 */
          [a, b].forEach(function (v, i) {
            var tr = tris[i], ft = P(v[0], 0), tp = P(v[0], v[1]);
            tr.fill.setAttribute('d', 'M ' + o[0] + ' ' + o[1] + ' L ' + ft[0].toFixed(1) + ' ' + ft[1].toFixed(1) + ' L ' + tp[0].toFixed(1) + ' ' + tp[1].toFixed(1) + ' Z');
            tr.base.setAttribute('d', 'M ' + o[0] + ' ' + o[1] + ' L ' + ft[0].toFixed(1) + ' ' + ft[1].toFixed(1));
            tr.up.setAttribute('d', 'M ' + ft[0].toFixed(1) + ' ' + ft[1].toFixed(1) + ' L ' + tp[0].toFixed(1) + ' ' + tp[1].toFixed(1));
            tr.lx.textContent = E.num(Math.abs(v[0])); tr.ly.textContent = E.num(Math.abs(v[1]));
            tr.lx.setAttribute('x', ((o[0] + ft[0]) / 2).toFixed(1)); tr.lx.setAttribute('y', (OY + 26).toFixed(1));
            tr.ly.setAttribute('text-anchor', i ? 'end' : 'start');
            // 竖边外侧、单位圆外面
            tr.ly.setAttribute('x', (ft[0] + (i ? -30 : 18)).toFixed(1)); tr.ly.setAttribute('y', ((OY + tp[1]) / 2 + 8).toFixed(1));
            E.op(tr.g, f(st, i ? T.col2 : T.tri1, 0.4) * (1 - f(st, T.triOut, 0.4)));
            var pb = bump(st, i ? T.eq2 : T.eq[0], 0.7), pc = bump(st, i ? T.eq2 + 0.4 : T.eq[2], 0.7);
            E.scaleAt(tr.lx, +tr.lx.getAttribute('x'), OY + 18, 1 + 0.25 * pb);
            E.scaleAt(tr.ly, +tr.ly.getAttribute('x'), +tr.ly.getAttribute('y') - 7, 1 + 0.25 * pc);
          });
          // 夹角：句 3 是 90°（亮黄），句 5 是 45°（橙）
          var A1 = Math.atan2(a[1], a[0]), A2 = Math.atan2(b[1], b[0]), shear = st >= T.toH[0];
          arc.setAttribute('d', arcPath(A1, A2, shear ? 0.42 : 0.36));
          K.paint(arc, null, shear ? E.ERR : E.HI);
          var am = (A1 + A2) / 2, ap = P(0.72 * Math.cos(am), 0.72 * Math.sin(am));
          arcLab.textContent = shear ? '45°' : '90°';
          arcLab.style.fill = shear ? E.ERR : E.HI;
          arcLab.setAttribute('x', ap[0].toFixed(1)); arcLab.setAttribute('y', (ap[1] + 8).toFixed(1));
          var arcOn = shear ? f(st, T.arc2, 0.35) * (1 - f(st, T.toR[0], 0.3)) : f(st, T.arc, 0.35) * (1 - f(st, T.toS[0] - 0.4, 0.3));
          E.op(arc, arcOn); E.op(arcLab, shear ? f(st, T.arc2 + 0.2, 0.3) * (1 - f(st, T.toR[0], 0.3)) : f(st, T.deg, 0.3) * (1 - f(st, T.toS[0] - 0.4, 0.3)));
          // 句 4：拉长后往右那根写橙色「长度 2」
          var tw = P(1.6, 0.22); two.setAttribute('x', tw[0].toFixed(1)); two.setAttribute('y', tw[1].toFixed(1));   // 单位圆外面
          E.op(two, f(st, T.two, 0.3) * (1 - f(st, T.toH[0], 0.3)));

          /* ══ 右栏 ══ */
          E.pop(rule.g, 622, 34, f(st, T.rule, 0.35), 0.85);
          rule.rect.style.strokeWidth = 2.5 + 2 * bump(st, T.same, 1.0);
          E.glow(rule.rect, E.GOOD, bump(st, T.same, 1.0) > 0.05);
          // 矩阵：一直跟着网格的矩阵（只在三段变形之间换数，变的过程中保持前一个）
          var shown = st < T.toS[0] + T.toS[1] / 2 ? Rm : st < T.toH[0] + T.toH[1] / 2 ? S : st < T.toR[0] + T.toR[1] / 2 ? Hm : Rm;
          m.set(shown);
          E.op(m.g, f(st, 0.2, 0.4));
          var colOn = st < T.toS[0] ? [f(st, T.col1, 0.3), f(st, T.col2, 0.3)] : st < T.toH[0] ? [f(st, T.c1, 0.3), 0] : [0, 0];
          m.colBox(0, E.X, colOn[0] * (st < T.toS[0] ? 1 - f(st, T.toS[0] - 0.4, 0.3) : 1));
          m.colBox(1, E.Y, colOn[1] * (1 - f(st, T.toS[0] - 0.4, 0.3)));
          mLab.textContent = st < T.toH[0] ? '横着拉长' : '斜着推一下';
          var labOn = st < T.toH[0] ? f(st, T.toS[0], 0.4) : st < T.toR[0] ? f(st, T.toH[0], 0.4) : 0;
          E.op(mLab, labOn * (1 - f(st, T.toR[0], 0.3)));
          var bx = st < T.toH[0] ? f(st, T.bad1, 0.4) : f(st, T.bad2, 0.4) * (1 - f(st, T.toR[0], 0.3));
          E.drawOn(badX, st < T.toH[0] + 0.3 && st >= T.toH[0] - 0.3 ? 0 : bx);

          // 句 2、3 的算式
          var eqOut = 1 - f(st, T.toS[0] - 0.5, 0.4);
          var A = '\\textcolor{' + E.HEX.x + '}{0.8}^2';
          if (st >= T.eq[1]) A += '+';
          if (st >= T.eq[2]) A += '\\textcolor{' + E.HEX.x + '}{0.6}^2';
          eqA.setTex(A);
          E.op(eqA, f(st, T.eq[0], 0.3) * eqOut);
          var B = '=0.64';
          if (st >= T.eq[4]) B += '+0.36';
          if (st >= T.eq[5]) B += '=\\textcolor{' + E.HEX.x + '}{1}';
          eqB.setTex(B);
          E.op(eqB, f(st, T.eq[3], 0.3) * eqOut);
          eqC.setTex('(\\textcolor{' + E.HEX.y + '}{-0.6})^2+\\textcolor{' + E.HEX.y + '}{0.8}^2=\\textcolor{' + E.HEX.y + '}{1}');
          E.op(eqC, f(st, T.eq2, 0.35) * eqOut);
          okB.setAttribute('transform', 'translate(' + (EX + 236) + ' 278)'); E.drawOn(okB, f(st, T.ok1, 0.4) * eqOut);
          okC.setAttribute('transform', 'translate(' + (EX + 270) + ' 334)'); E.drawOn(okC, f(st, T.eq2 + 0.9, 0.4) * eqOut);

          // 句 4、5 的两行
          var stretch = st < T.toH[0];
          rowsBad[0].textContent = stretch ? '第一列 (2, 0)' : '两列夹角 45°';
          rowsBad[1].textContent = stretch ? '长度 2，不是 1' : '不垂直';
          var r0 = stretch ? f(st, T.c1, 0.3) * (1 - f(st, T.toH[0] - 0.3, 0.3)) : f(st, T.arc2, 0.3) * (1 - f(st, T.toR[0], 0.3));
          var r1 = stretch ? f(st, T.two, 0.3) * (1 - f(st, T.toH[0] - 0.3, 0.3)) : f(st, T.notp, 0.3) * (1 - f(st, T.toR[0], 0.3));
          E.op(rowsBad[0], r0); E.op(rowsBad[1], r1);

          // 句 6：正交、式子
          E.pop(big.g, MX, 247, f(st, T.ortho, 0.35), 0.8);
          E.op(book, f(st, T.book, 0.3));
          var F = '';
          if (st >= T.f[0]) F = 'R^{\\top}';
          if (st >= T.f[1]) F += 'R';
          if (st >= T.f[2]) F += '=I';
          form.setTex(F || '\\,');
          E.op(form, f(st, T.f[0], 0.3));
          var fg = bump(st, T.same, 1.0);
          E.glow(form, E.GOOD, fg > 0.05);
          E.pop(unitG, 716, 360, f(st, T.unit, 0.35), 0.85);
        }
      };
    }
  };
})();
