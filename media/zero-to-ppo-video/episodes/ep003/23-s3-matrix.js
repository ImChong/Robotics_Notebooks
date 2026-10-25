// 003 第 3 幕：矩阵是变换。左边一张坐标网格（灰色原网格 + 一层紫色「变形网格」，即 E.plane 的 warp），右栏放标签与矩阵。
// 句 1「矩阵」「变换」出两块标签；「所有的点」网格交点上的点和一面亮黄小旗（F 形，不对称，转过去看得出来）亮起，「一起挪」时跟着网格轻轻斜一下再回来。
// 句 2 念到哪种变换做哪种：「转九十度」转 90°（插角度，不压扁）→ 回原样 →「拉长到两倍」横向拉到 2 倍 → 回原样 →「斜着推一下」剪切，停住；右栏三块标签，正在做的那块亮。
// 句 3 从剪切过渡到一个一般的线性变换停住：「原点不动」原点亮圈，「直的」「平行的」「间距相等」依次打勾，网格上同时描亮三根变形后的竖线和两段等长的间距；「线性变换」出标签。
// 句 4 网格回原样，「不用管每一个点」点和小旗淡出；「两根小箭头」出现往右一格（绿）、往上一格（琥珀），念到各自鼓一下，右栏列出这两根。
// 句 5「转九十度」网格连同两根小箭头转 90°；「零、一」「负一、零」时在落点和右栏写出坐标。
// 句 6 两个落点坐标「竖着排成两列」飞进右栏的矩阵（第一列绿、第二列琥珀），「矩阵」时加紫框和「转 90° 的矩阵」，
// 「每一列」两列依次框亮、虚线连到对应小箭头尖，「落在哪」两个箭头尖各亮一圈。
// 时刻取自配音的逐词时间（分镜时间 ≈ 念到的时刻），见 ../ep003.py。
(function () {
  var E = window.EP003, K = E.K;
  var HALF = Math.PI / 2;
  var OX = 262, OY = 215, U = 56;                           // 网格：原点在画面里的位置、一格多少 px（x −4…4、y −3…3）
  var I = [1, 0, 0, 1], M_NUDGE = [1.12, 0.32, 0.08, 1], M_STRETCH = [2, 0, 0, 1], M_SHEAR = [1, 1, 0, 1], M_MIX = [1.3, 0.6, 0.3, 0.9];
  var FLAG = [[[1, 0], [1, 2], [2, 2]], [[1, 1], [1.7, 1]]];  // 小旗：不对称，转 90° 前后分得清；各种变换下都留在网格外框里
  var RX = 540;                                             // 右栏左缘
  var TAGS = [{ s: '转 90°', at: 7.78, on: [7.72, 8.74], y: 132 }, { s: '横着拉长 2 倍', at: 9.04, on: [8.82, 10.32], y: 188 },
    { s: '斜着推一下', at: 10.39, on: [10.36, 12.4], y: 244 }];
  var CHECKS = [{ s: '原点不动', at: 14.42 }, { s: '格子线还是直的', at: 16.7 }, { s: '平行的', at: 17.26 }, { s: '间距相等', at: 18.16 }];
  var ROW_Y = [126, 186];                                   // 句 4、5 右栏两根小箭头的行
  var MX = 665, MCY = 200, MS = 44, LABEL_Y = 330;          // 句 6 的矩阵：中心与字号；「转 90° 的矩阵」写在矩阵下面（第一列的连线从 x 轴标签上方过）

  function lerp4(A, B, p) { return A.map(function (a, i) { return a + (B[i] - a) * p; }); }
  function bump(st, a, d) { var u = K.seg(st, a, a + d); return u > 0 && u < 1 ? Math.sin(Math.PI * u) : 0; }

  // 这一刻网格的矩阵（数学坐标里的 [a, b, c, d]）
  function warpAt(st) {
    var f = E.fade;
    if (st < 7.6) return lerp4(I, M_NUDGE, f(st, 5.55, 0.5) * (1 - f(st, 6.3, 0.5)));
    if (st < 8.76) return E.rot(HALF * f(st, 7.72, 0.6) * (1 - f(st, 8.42, 0.32)));
    if (st < 10.34) return lerp4(I, M_STRETCH, f(st, 8.82, 0.75) * (1 - f(st, 10.0, 0.32)));
    if (st < 12.4) return lerp4(I, M_SHEAR, f(st, 10.36, 0.6));
    if (st < 23.1) return lerp4(M_SHEAR, M_MIX, f(st, 12.4, 0.9));
    if (st < 30.1) return lerp4(M_MIX, I, f(st, 23.1, 0.8));
    return E.rot(HALF * f(st, 30.2, 0.9));
  }

  E.scenes[2] = {
    title: '矩阵：把整张网格一起变形',
    dur: 47.98,
    cues: [
      { at: 0, s: '**矩阵**管变换：所有点按同一个规则一起挪' },
      { at: 7.03, s: '转 90°、横着拉长、斜着推' },
      { at: 12.31, s: '原点不动，格子线直、平行、等距：**线性变换**' },
      { at: 23.04, s: '只盯两根小箭头：往右一格、往上一格' },
      { at: 30.1, s: '转 90°：往右那根 → (0, 1)，往上那根 → (−1, 0)' },
      { at: 39.29, s: '每一列 = 一根小箭头最后落在哪' }
    ],
    build: function (K) {
      var s = K.sceneSvg('矩阵管的是变换：把平面上所有的点按同一个规则一起挪，比如转 90 度、横着拉长到两倍、斜着推一下；这几种变换原点不动，格子线变完还是直的、平行的、间距相等，叫线性变换；只要看往右一格和往上一格两根小箭头落在哪：转 90 度后它们落在 (0, 1) 和 (−1, 0)，竖着排成两列就是这个变换的矩阵');
      var pl = E.plane(s, { ox: OX, oy: OY, u: U, x0: -4, x1: 4, y0: -3, y1: 3 });
      var P = pl.P, box = pl.box;
      function at(m, p) { var q = E.apply(m, p); return P(q[0], q[1]); }
      function seg(a, b, m) {                               // 数学坐标里的一段，变形后裁到网格外框里 → path d
        var c = E.clipSeg(at(m, a), at(m, b), box);
        return c ? 'M ' + c[0][0].toFixed(1) + ' ' + c[0][1].toFixed(1) + ' L ' + c[1][0].toFixed(1) + ' ' + c[1][1].toFixed(1) : '';
      }
      function halo(n) { n.style.stroke = E.S1; n.style.strokeWidth = 6; n.style.paintOrder = 'stroke'; n.style.strokeLinejoin = 'round'; return n; }

      /* ── 网格上的东西：句 3 的描亮线、交点上的点、小旗、原点亮圈、句 6 的连线、两根小箭头与落点坐标 ── */
      var hl = [1, 2, 3].map(function () {
        return K.paint(E.el('path', { fill: 'none', 'stroke-width': 4, 'stroke-linecap': 'round' }, s), null, E.HI);
      });
      var gaps = [0, 1].map(function () {
        return K.paint(E.el('path', { fill: 'none', 'stroke-width': 3, 'stroke-linecap': 'round' }, s), null, E.HI);
      });
      var dots = [];
      for (var x = -4; x <= 4; x++) for (var y = -3; y <= 3; y++) {
        dots.push({ p: [x, y], d: Math.hypot(x, y), el: K.paint(E.el('circle', { r: 4 }, s), E.INK) });
      }
      var flag = FLAG.map(function () {
        return K.paint(E.el('path', { fill: 'none', 'stroke-width': 6, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, s), null, E.HI);
      });
      var ring = K.paint(E.el('circle', { cx: OX, cy: OY, r: 9, fill: 'none', 'stroke-width': 3 }, s), null, E.HI);
      var ringDot = K.paint(E.el('circle', { cx: OX, cy: OY, r: 5 }, s), E.HI);
      var links = [E.GOOD, E.WARN].map(function (c) {
        return K.paint(E.el('path', { fill: 'none', 'stroke-width': 2.2, 'stroke-dasharray': '6 5', 'stroke-linecap': 'round' }, s), null, c);
      });
      var e1 = E.vec(s, E.GOOD, 5, { head: 16 }), e2 = E.vec(s, E.WARN, 5, { head: 16 });
      var tips = [E.GOOD, E.WARN].map(function (c) { return K.paint(E.el('circle', { r: 10, fill: 'none', 'stroke-width': 3 }, s), null, c); });
      var p01 = P(0, 1), pm10 = P(-1, 0);
      var lab1 = halo(E.txt(s, p01[0], p01[1] - 16, '(0, 1)', 22, E.GOOD, 'middle', 700));   // 写在箭头尖正上方：连到矩阵第一列的虚线从它下面过
      var lab2 = halo(E.txt(s, pm10[0], pm10[1] - 16, '(−1, 0)', 22, E.WARN, 'middle', 700));

      /* ── 右栏：句 1 的「矩阵 → 变换」、句 2 的三种变换、句 3 的共同点、句 4–5 的两根小箭头、句 6 的矩阵 ── */
      var hMat = E.tag(s, RX + 6, 46, 96, 40, '矩阵', E.VIOLET, { size: 24, ink: E.VIOLET, weight: 900, rx: 10, sw: 2.5 });
      var hArrow = E.arrow(s, [[RX + 112, 66], [RX + 150, 66]], E.INK2, 3, { head: 10 });
      var hTrans = E.tag(s, RX + 158, 46, 96, 40, '变换', E.VIOLET, { size: 24, ink: E.INK, weight: 900, rx: 10, sw: 2.5 });
      var tags = TAGS.map(function (d) { return E.tag(s, RX + 12, d.y - 21, 228, 42, d.s, E.VIOLET, { size: 22, ink: E.INK, weight: 700, rx: 10 }); });

      var chkHead = E.txt(s, RX + 8, 70, '共同点', 22, E.INK2, 'start', 700);
      var checks = CHECKS.map(function (c, i) {
        var y = 112 + 46 * i, g = E.el('g', {}, s);
        return { g: g, mark: E.check(g, RX + 10, y - 16, 22, E.GOOD, 4), text: E.txt(g, RX + 44, y + 2, c.s, 22, E.INK, 'start', 700) };
      });
      var linear = E.tag(s, RX + 22, 300, 200, 48, '线性变换', E.VIOLET, { size: 26, ink: E.VIOLET, weight: 900, rx: 12, sw: 3 });

      var rows = ROW_Y.map(function (y, i) {
        var g = E.el('g', {}, s), icon = E.vec(g, i ? E.WARN : E.GOOD, 4, { head: 12 });
        if (i) icon.set(RX + 22, y + 16, RX + 22, y - 20); else icon.set(RX + 4, y, RX + 42, y);
        E.txt(g, RX + 54, y + 8, i ? '往上一格' : '往右一格', 22, E.INK, 'start', 700);
        var map = E.txt(g, RX + 150, y + 8, i ? '→ (−1, 0)' : '→ (0, 1)', 22, i ? E.WARN : E.GOOD, 'start', 700);
        return { g: g, map: map };
      });

      var frame = K.paint(E.el('rect', { rx: 14, 'stroke-width': 3 }, s), E.tint(E.VIOLET, 10, 'var(--demo-surface)'), E.VIOLET);
      var colHi = [E.GOOD, E.WARN].map(function (c) { return K.paint(E.el('rect', { rx: 8, 'stroke-width': 2.5 }, s), E.tint(c, 18, 'var(--demo-surface)'), c); });
      var mat = E.tex(s, MX, MCY + 0.34 * MS, E.mat2([0, -1, 1, 0], [E.HEX.good, E.HEX.warn]), MS, E.INK);
      var matLabel = E.txt(s, MX, LABEL_Y, '转 90° 的矩阵', 22, E.VIOLET, 'middle', 900);
      // 飞进矩阵的只是两列数字（不带括号、字号同矩阵），落点正好是矩阵里那一列，换成整个矩阵时不会叠出两层
      function numCol(xs, hex) { return '\\begin{matrix}' + xs.map(function (x) { return '\\textcolor{' + hex + '}{' + x + '}'; }).join('\\\\') + '\\end{matrix}'; }
      var fly = [numCol([0, 1], E.HEX.good), numCol([-1, 0], E.HEX.warn)].map(function (tex) { return E.tex(s, MX, MCY, tex, MS); });

      // 画面坐标 ↔ 分镜坐标：KaTeX 排出来的列在哪，要从页面上量（fitStory 改 viewBox 也不影响分镜坐标）
      function toSvg(cx, cy) { var p = s.createSVGPoint(); p.x = cx; p.y = cy; return p.matrixTransform(s.getScreenCTM().inverse()); }
      // KaTeX 的盒子按行高排，外层量不全括号和上下两行：取里面各元素外框的并集（括号、支柱、每个数都算进来）
      function rectOf(nodes) {
        var l = Infinity, tp = Infinity, r = -Infinity, b = -Infinity;
        Array.prototype.forEach.call(nodes, function (n) {
          var q = n.getBoundingClientRect();
          if (q.width < 0.5 || q.height < 0.5) return;
          l = Math.min(l, q.left); tp = Math.min(tp, q.top); r = Math.max(r, q.right); b = Math.max(b, q.bottom);
        });
        if (!(r > l)) return null;
        var a = toSvg(l, tp), c = toSvg(r, b);
        return { x0: a.x, y0: a.y, x1: c.x, y1: c.y };
      }
      function measure() {
        if (!mat.isConnected || !s.getScreenCTM()) return null;
        var cols = mat.querySelectorAll('.col-align-c');
        if (cols.length < 2) return null;
        var m = { cols: [rectOf(cols[0].querySelectorAll('.mord')), rectOf(cols[1].querySelectorAll('.mord'))], all: rectOf(mat.querySelectorAll('.katex-html *')) };
        return m.cols[0] && m.cols[1] && m.all && m.all.x1 - m.all.x0 > 10 ? m : null;
      }
      function center(n) { var b = n.getBBox(); return [b.x + b.width / 2, b.y + b.height / 2]; }
      // 二次曲线的前 p 段（虚线「画出来」：每帧只取到 p，不靠 dashoffset）
      function quad(a, c, b, p) {
        var d = '';
        for (var k = 0; k <= 24; k++) {
          var u = p * k / 24, w = 1 - u;
          d += (k ? ' L ' : 'M ') + (w * w * a[0] + 2 * w * u * c[0] + u * u * b[0]).toFixed(1) + ' ' + (w * w * a[1] + 2 * w * u * c[1] + u * u * b[1]).toFixed(1);
        }
        return p > 0 ? d : '';
      }

      return {
        el: s,
        draw: function (st, t) {
          var f = E.fade, m = warpAt(st);

          /* 网格：2.54「变换」起紫色变形网格出来，之后每帧按 warpAt 变形 */
          pl.warp.set(m);
          E.op(pl.warp.g, (0.62 * f(st, 2.54, 0.5)).toFixed(3));

          /* 3.83「所有的点」交点上的点从原点往外依次亮起、小旗同时出现，跟着网格挪；小旗 15.0 退场，点 24.0「每一个点」淡出 */
          var gone = 1 - f(st, 24.0, 0.5);
          dots.forEach(function (d) {
            var q = at(m, d.p), inside = q[0] >= box[0] - 1 && q[0] <= box[2] + 1 && q[1] >= box[1] - 1 && q[1] <= box[3] + 1;
            d.el.setAttribute('cx', q[0].toFixed(1)); d.el.setAttribute('cy', q[1].toFixed(1));
            E.op(d.el, inside ? (0.85 * f(st, 3.83 + 0.06 * d.d, 0.3) * gone).toFixed(3) : 0);
          });
          flag.forEach(function (path, i) {
            path.setAttribute('d', FLAG[i].map(function (p, k) { var q = at(m, p); return (k ? 'L ' : 'M ') + q[0].toFixed(1) + ' ' + q[1].toFixed(1); }).join(' '));
            E.op(path, (f(st, 4.17, 0.4) * (1 - f(st, 15.0, 0.4))).toFixed(3));   // 15.0 先退场：句 3 描亮的格子线也是亮黄
          });

          /* 句 1：1.07「矩阵」、2.54「变换」；句 2：三块变换标签按念到的词弹出，正在做的那块亮边；12.35 一起退场 */
          var out1 = 1 - f(st, 12.35, 0.35);
          E.pop(hMat.g, RX + 54, 66, f(st, 1.07, 0.3) * out1, 0.8);
          E.op(hArrow.g, (f(st, 2.4, 0.3) * out1).toFixed(3));
          E.pop(hTrans.g, RX + 206, 66, f(st, 2.54, 0.3) * out1, 0.8);
          tags.forEach(function (tg, i) {
            var d = TAGS[i], on = st >= d.on[0] && st < d.on[1];
            E.pop(tg.g, RX + 126, d.y, f(st, d.at - 0.1, 0.3), 0.85);
            E.op(tg.g, (f(st, d.at - 0.1, 0.3) * (on ? 1 : 0.45) * out1).toFixed(3));
            tg.rect.style.strokeWidth = on ? 3.5 : 2;
            E.glow(tg.rect, E.VIOLET, on);
          });

          /* 句 3：13.51「共同点」；14.42「原点不动」原点亮圈；16.7「直」、17.26「平行」、18.16「间距相等」依次打勾并在网格上描亮；21.02「线性变换」；23.1 起退场 */
          var out3 = 1 - f(st, 22.9, 0.35);
          E.op(chkHead, (f(st, 13.51, 0.3) * out3).toFixed(3));
          checks.forEach(function (c, i) {
            var p = f(st, CHECKS[i].at, 0.35);
            E.op(c.g, (Math.min(1, p * 2) * out3).toFixed(3));
            E.drawOn(c.mark, p);
          });
          E.op(ringDot, (f(st, 14.42, 0.3) * out3).toFixed(3));
          ring.setAttribute('r', (9 + 10 * bump(st, 14.42, 0.8)).toFixed(2));
          E.op(ring, (f(st, 14.42, 0.3) * out3).toFixed(3));
          E.glow(ring, E.HI, st >= 14.42 && st < 23.1);
          [16.7, 17.26, 18.16].forEach(function (a, i) {
            hl[i].setAttribute('d', seg([i + 1, -20], [i + 1, 20], m));
            E.op(hl[i], (f(st, a, 0.3) * out3).toFixed(3));
          });
          gaps.forEach(function (g, i) {                    // 变形后 y = −2 那根线上，相邻两根竖线之间的两段：一样长
            var a = at(m, [i + 1, -2]), b = at(m, [i + 2, -2]), dx = b[0] - a[0], dy = b[1] - a[1], L = Math.hypot(dx, dy) || 1;
            var nx = -dy / L * 8, ny = dx / L * 8;
            g.setAttribute('d', 'M ' + (a[0] + nx).toFixed(1) + ' ' + (a[1] + ny).toFixed(1) + ' L ' + (a[0] - nx).toFixed(1) + ' ' + (a[1] - ny).toFixed(1) +
              ' M ' + a[0].toFixed(1) + ' ' + a[1].toFixed(1) + ' L ' + b[0].toFixed(1) + ' ' + b[1].toFixed(1) +
              ' M ' + (b[0] + nx).toFixed(1) + ' ' + (b[1] + ny).toFixed(1) + ' L ' + (b[0] - nx).toFixed(1) + ' ' + (b[1] - ny).toFixed(1));
            E.op(g, (f(st, 18.16, 0.3) * out3).toFixed(3));
          });
          E.pop(linear.g, RX + 122, 324, f(st, 21.02, 0.3) * out3, 0.8);

          /* 句 4–6：两根小箭头随网格变（25.6「两根小箭头」出现）；27.06 / 28.07 各鼓一下，31.47 / 34.85「那根」再鼓一下 */
          var o = P(0, 0), q1 = at(m, [1, 0]), q2 = at(m, [0, 1]), show = f(st, 25.6, 0.4);
          e1.set(o[0], o[1], q1[0], q1[1]); e2.set(o[0], o[1], q2[0], q2[1]);
          var b1 = Math.max(bump(st, 27.0, 0.6), bump(st, 31.47, 0.6)), b2 = Math.max(bump(st, 28.0, 0.6), bump(st, 34.85, 0.6));
          E.op(e1.g, show); E.op(e2.g, show);
          e1.line.style.strokeWidth = 5 + 3 * b1; e2.line.style.strokeWidth = 5 + 3 * b2;
          E.glow(e1.g, E.GOOD, b1 > 0.05); E.glow(e2.g, E.WARN, b2 > 0.05);
          var outRows = 1 - f(st, 39.3, 0.4);
          rows.forEach(function (r, i) {
            E.op(r.g, (f(st, i ? 28.07 : 27.06, 0.3) * outRows).toFixed(3));
            E.op(r.map, f(st, i ? 37.29 : 33.85, 0.3));
          });

          /* 句 5：落点坐标 33.85「零、一」、37.29「负一、零」；句 6 开头 39.5「这两个新位置」鼓一下 */
          var pb = bump(st, 39.45, 0.7);
          E.op(lab1, f(st, 33.75, 0.3)); E.op(lab2, f(st, 37.2, 0.3));
          E.scaleAt(lab1, p01[0], p01[1] - 22, 1 + 0.15 * pb);
          E.scaleAt(lab2, pm10[0], pm10[1] - 22, 1 + 0.15 * pb);

          /* 句 6：40.55「竖着排成两列」两个坐标先后飞进矩阵的两列，41.9 矩阵出来；43.2「矩阵」紫框 +「转 90° 的矩阵」；
             44.25「每一列」第一列框亮、虚线连到往右那根的箭头尖，44.85 第二列；46.5「落在哪」两个箭头尖亮一圈 */
          var M = st >= 39 ? measure() : null;
          var showMat = f(st, 41.9, 0.25);   // 两列都落定以后矩阵（括号）才出来
          E.op(mat, showMat.toFixed(3));
          // 两列先后飞：绿列从右边的 (0, 1) 飞进左列，琥珀列从左边的 (−1, 0) 飞进右列——同时飞会在半路交叉，
          // 所以琥珀列等绿列快落定再走，而且走下面：从网格下半部穿过去，经过绿列下方，再从右下方升进右列（往上绕会出画面）。
          // 路线是三次贝塞尔 a → c1 → c2 → b；绿列往上拱，不划过 x 轴的「x」
          var FLY = [[40.55, 0.7], [41.05, 0.85]];
          fly.forEach(function (n, i) {
            var p = f(st, FLY[i][0], FLY[i][1]), a = center(i ? lab2 : lab1);
            var b = M ? [(M.cols[i].x0 + M.cols[i].x1) / 2, (M.cols[i].y0 + M.cols[i].y1) / 2] : [MX + (i ? 24 : -24), MCY];
            var top = Math.min(a[1], b[1]);
            var c1 = i ? [a[0] + 60, a[1] + 150] : [a[0] + (b[0] - a[0]) / 3, top - 60];
            var c2 = i ? [b[0] + 80, b[1] + 170] : [a[0] + 2 * (b[0] - a[0]) / 3, top - 60];
            var q = 1 - p, w = [q * q * q, 3 * q * q * p, 3 * q * p * p, p * p * p];
            n.setX(w[0] * a[0] + w[1] * c1[0] + w[2] * c2[0] + w[3] * b[0]);
            n.setY(w[0] * a[1] + w[1] * c1[1] + w[2] * c2[1] + w[3] * b[1] + 0.34 * MS);
            // 起飞一小段后才显出来（不和原来的坐标标签叠在一起）；矩阵整个出来以后再收掉
            E.op(n, (f(st, FLY[i][0] + 0.2, 0.2) * (1 - f(st, 42.15, 0.05))).toFixed(3));
          });
          var pf = f(st, 43.2, 0.35);
          if (M) {
            frame.setAttribute('x', (M.all.x0 - 16).toFixed(1)); frame.setAttribute('y', (M.all.y0 - 12).toFixed(1));
            frame.setAttribute('width', (M.all.x1 - M.all.x0 + 32).toFixed(1)); frame.setAttribute('height', (M.all.y1 - M.all.y0 + 24).toFixed(1));
            colHi.forEach(function (r, i) {
              var c = M.cols[i];
              r.setAttribute('x', (c.x0 - 9).toFixed(1)); r.setAttribute('y', (c.y0 - 4).toFixed(1));
              r.setAttribute('width', (c.x1 - c.x0 + 18).toFixed(1)); r.setAttribute('height', (c.y1 - c.y0 + 8).toFixed(1));
            });
            var c1 = M.cols[0], c2 = M.cols[1];
            var s1 = [c1.x0 - 12, (c1.y0 + c1.y1) / 2], s2 = [(c2.x0 + c2.x1) / 2, c2.y1 + 10];
            var p1 = f(st, 44.25, 0.5), p2 = f(st, 44.9, 0.6);
            links[0].setAttribute('d', p1 > 0 ? 'M ' + s1[0].toFixed(1) + ' ' + s1[1].toFixed(1) + ' L ' + (s1[0] + (q1[0] - s1[0]) * p1).toFixed(1) + ' ' + (s1[1] + (q1[1] - s1[1]) * p1).toFixed(1) : '');
            links[1].setAttribute('d', quad(s2, [430, 330], [q2[0] + 4, q2[1] + 8], p2));   // 从第二列底下绕过来，不压第一列和标签
          }
          E.op(frame, pf.toFixed(3));
          E.pop(matLabel, MX, LABEL_Y - 7, pf, 0.8);
          colHi.forEach(function (r, i) { E.op(r, f(st, i ? 44.85 : 44.2, 0.3).toFixed(3)); });
          links.forEach(function (l) { E.op(l, M ? 1 : 0); });
          [q1, q2].forEach(function (q, i) {
            var u = K.seg(st, 46.5 + 0.15 * i, 47.3 + 0.15 * i);
            tips[i].setAttribute('cx', q[0].toFixed(1)); tips[i].setAttribute('cy', q[1].toFixed(1));
            tips[i].setAttribute('r', (8 + 16 * u).toFixed(2));
            E.op(tips[i], u > 0 && u < 1 ? (1 - u).toFixed(3) : 0);
          });
        }
      };
    }
  };
})();
