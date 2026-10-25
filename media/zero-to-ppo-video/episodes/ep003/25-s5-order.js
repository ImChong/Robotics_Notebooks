// 003 第 5 幕：矩阵相乘——两个变换接着做，顺序不能换。左边一张网格（x −4…6、y −1…7），右边两行乘式。
// 记号：「转」R = [[0, −1], [1, 0]]（转 90°），「拉」S = [[2, 0], [0, 1]]（横着拉到 2 倍）；手的那个点沿用上一幕的 (3, 2)。
// ① 先转再拉（蓝）：(3, 2) →转→ (−2, 3) →拉→ (−4, 3)，网格跟着变形，路径留淡迹；念到「转」「拉长」时右边第一行的空位依次出「转」「拉」矩阵
//    （后做的「拉」出在「转」的左边）；「乘起来」「合成」出 = 与合成矩阵 S R = [[0, −2], [1, 0]]；「一步做完」用合成矩阵一跳送到 (−4, 3)；
//    「先做的写在右边」右边的「转」上标「先做」，标签行里从「转」往「拉」画一支小箭头。
// ② 先拉再转（亮黄）：(3, 2) →拉→ (6, 2) →转→ (−2, 6)，第二行出 R S = [[0, −1], [2, 0]]；「完全不同」两个终点各圈一下，中间一个红「≠」。
// ③「讲顺序」两行之间出「拉·转 ≠ 转·拉」，「不能随便交换」交换弧线上打红 ×。
// ④ 25 s 起换成 UR5：「一节连着一节」关节从底座到手腕依次亮、连杆依次上色；「每过一个关节」各关节旁出 T₁、T₂、T₃、T₄T₅T₆；
//    「手在哪」手亮起；「按顺序乘起来」出 手 = T₁T₂T₃T₄T₅T₆ 和一支「按顺序」箭头；「第十一集」出「第 11 集：正运动学」。
// 时刻取自配音的逐词时间（分镜时间 ≈ 念到的时刻），见 ../ep003.py。
(function () {
  var E = window.EP003, K = E.K;
  var TAU = Math.PI * 2;
  var R = [0, -1, 1, 0], S = [2, 0, 0, 1];
  var U = 40, OX = 180, OY = 330;                           // 网格：一格 40 px，原点 (180, 330)
  var START = [3, 2];
  var ROT_A = 2.91, STR_A = 4.19, MUL = 6.94, MERGE = 7.8, JUMP = 9.24, FIRST = 11.21, RIGHT = 11.9;
  var HEAD_B = 14.0, STR_B = 15.24, ROT_B = 16.08, LAND_B = 17.61, DIFF = 18.04, NEQ = 18.48;
  var ORDER = 21.9, SWAP = 22.85;
  var ROBOT = 25.0, CHAIN = 25.59, TAGS = [27.1, 27.45, 27.8, 28.15], HAND = 29.85, BUNCH = 31.57, PROD = 32.36, EP11 = 33.75;
  // 右边两行乘式：「拉」「转」「=」「合成」四列的 x；第一行（先转再拉）中心 y 124，第二行（先拉再转）中心 y 334
  var COL = [522, 612, 670, 738], ROW_A = 124, ROW_B = 334, MS = 30, MID = 644;   // COL 只是初值，layout() 按实际宽度重排
  var LINKS = ['shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link'];

  function bell(st, a, b) { var u = K.seg(st, a, b); return u > 0 && u < 1 ? Math.sin(Math.PI * u) : 0; }

  E.scenes[4] = {
    title: '矩阵相乘：接着做，顺序不能换',
    dur: 35.86,
    cues: [
      { at: 0, s: '先转 90°，再横着拉长 2 倍' },
      { at: 6.05, s: '两个**矩阵**相乘 = 一步做完；先做的写在右边' },
      { at: 13.61, s: '换个顺序：先拉长、再转 → 落点不同' },
      { at: 20.35, s: '矩阵乘法讲**顺序**：一般不能交换' },
      { at: 25.01, s: '每过一个关节一次变换：手在哪 = 按顺序连乘' }
    ],
    build: function (K) {
      var s = K.sceneSvg('两个变换接着做：先转 90° 再横着拉长 2 倍，点 (3, 2) 到 (−4, 3)，两个矩阵相乘合成一个矩阵，先做的写在右边；换个顺序先拉长再转，落到 (−2, 6)，矩阵乘法一般不能交换；机器人每过一个关节一次变换，手在哪就是把这一串矩阵按顺序乘起来');

      /* ══ ①–③ 网格 + 两条路线 ══ */
      var partA = E.el('g', {}, s);
      var pl = E.plane(partA, { ox: OX, oy: OY, u: U, x0: -4, x1: 6, y0: -1, y1: 7, ticks: true });
      E.op(pl.warp.g, 0);
      var P = pl.P;
      function at(v) { return P(v[0], v[1]); }
      function dash(d, color) {
        var p = E.el('path', { d: d, fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '7 6', 'stroke-linecap': 'round' }, partA);
        K.paint(p, null, color);
        p.style.strokeOpacity = 0.6;
        return p;
      }
      function arcD(a, b) {                                   // 绕原点从 a 到 b 的逆时针圆弧（画面上 sweep 0）
        var r = Math.hypot(a[0], a[1]) * U, p = at(a), q = at(b);
        return 'M ' + p[0] + ' ' + p[1] + ' A ' + r.toFixed(2) + ' ' + r.toFixed(2) + ' 0 0 0 ' + q[0].toFixed(2) + ' ' + q[1].toFixed(2);
      }
      function lineD(a, b) { var p = at(a), q = at(b); return 'M ' + p[0] + ' ' + p[1] + ' L ' + q[0] + ' ' + q[1]; }
      var MID_A = E.apply(R, START), END_A = E.apply(S, MID_A);      // (−2, 3)、(−4, 3)
      var MID_B = E.apply(S, START), END_B = E.apply(R, MID_B);      // (6, 2)、(−2, 6)
      var trailA = [dash(arcD(START, MID_A), E.ACC), dash(lineD(MID_A, END_A), E.ACC)];
      var trailB = [dash(lineD(START, MID_B), E.HI), dash(arcD(MID_B, END_B), E.HI)];
      // 「一步做完」：合成矩阵一跳，紫色虚线弧
      var j0 = at(START), j1 = at(END_A), jc = [160, 90];
      var jump = E.curveArrow(partA, 'M ' + j0[0] + ' ' + j0[1] + ' Q ' + jc[0] + ' ' + jc[1] + ' ' + j1[0] + ' ' + j1[1], E.VIOLET, 3,
        Math.atan2(j1[1] - jc[1], j1[0] - jc[0]) * 180 / Math.PI, { dash: '8 6', head: 13 });
      var jumpTxt = E.txt(partA, 128, 146, '一步', 20, E.VIOLET, 'middle', 900);
      var ghost = K.paint(E.el('circle', { r: 7, 'stroke-width': 2 }, partA), E.VIOLET, E.S1);
      var vecB = E.vec(partA, E.HI, 4), vecA = E.vec(partA, E.ACC, 4);
      var dotB = K.paint(E.el('circle', { r: 7, 'stroke-width': 2.5 }, partA), E.HI, E.S1);
      var dotA = K.paint(E.el('circle', { r: 7, 'stroke-width': 2.5 }, partA), E.ACC, E.S1);
      var lab0 = E.txt(partA, at(START)[0] + 12, at(START)[1] - 12, '(3, 2)', 20, E.INK, 'start', 700);
      var labA = E.txt(partA, at(END_A)[0] + 8, at(END_A)[1] - 25, '(−4, 3)', 20, E.ACC, 'start', 900);
      var labB = E.txt(partA, at(END_B)[0] - 22, at(END_B)[1] + 7, '(−2, 6)', 20, E.HI, 'end', 900);   // 放在点左边：右边是转过来的弧线
      var ringA = K.paint(E.el('circle', { cx: at(END_A)[0], cy: at(END_A)[1], r: 17, fill: 'none', 'stroke-width': 3 }, partA), null, E.ACC);
      var ringB = K.paint(E.el('circle', { cx: at(END_B)[0], cy: at(END_B)[1], r: 17, fill: 'none', 'stroke-width': 3 }, partA), null, E.HI);
      var neqPts = E.txt(partA, (at(END_A)[0] + at(END_B)[0]) / 2 - 6, (at(END_A)[1] + at(END_B)[1]) / 2 + 12, '≠', 38, E.BAD, 'middle', 900);

      /* ── 右边两行乘式 ── */
      function mat(cx, cy, tex, color) {
        var g = E.el('g', {}, partA);
        var fo = E.tex(g, cx, cy + 0.34 * MS, tex, MS, color || E.INK);
        return { g: g, fo: fo };
      }
      function row(cy, first, second, result, color, names) {
        var r = {
          cy: cy, x: COL.slice(),
          m: [mat(COL[0], cy, E.mat2(first)), mat(COL[1], cy, E.mat2(second))],
          eq: E.txt(partA, COL[2], cy + 10, '=', 30, E.INK2, 'middle', 700),
          res: mat(COL[3], cy, E.mat2(result), color),
          labels: names.map(function (n, k) { return E.txt(partA, COL[k === 2 ? 3 : k], cy + 62, n, 20, k === 2 ? color : E.VIOLET, 'middle', 700); })
        };
        return r;
      }
      // KaTeX 矩阵宽窄不一（「−1」比「0」宽）：按实际宽度把「矩阵 矩阵 = 合成」一行排在 MID 两边，间距固定；字体没加载完时宽度会变，所以每帧都量
      function layout(r) {
        if (!s.isConnected) return;
        var sr = s.getBoundingClientRect();
        if (!sr.width) return;
        var k = s.viewBox.baseVal.width / sr.width;
        // 先去掉弹出的缩放（pop 每帧在 layout 之后重设），量的是原大
        [r.m[0].g, r.m[1].g, r.res.g].forEach(function (g) { g.removeAttribute('transform'); });
        function w(fo) { var n = fo.texBox.querySelector('.katex') || fo.texBox; return n.getBoundingClientRect().width * k; }
        var ws = [w(r.m[0].fo), w(r.m[1].fo), r.eq.getBBox().width, w(r.res.fo)];
        if (!(ws[0] > 0 && ws[3] > 0)) return;
        var gaps = [4, 8, 8], tot = ws.reduce(function (a, b) { return a + b; }, 0) + 20, x = MID - tot / 2;
        ws.forEach(function (wi, i) { r.x[i] = x + wi / 2; x += wi + (gaps[i] || 0); });
        r.m[0].fo.setX(r.x[0]); r.m[1].fo.setX(r.x[1]); r.eq.setAttribute('x', r.x[2].toFixed(1)); r.res.fo.setX(r.x[3]);
        r.labels[0].setAttribute('x', r.x[0].toFixed(1)); r.labels[1].setAttribute('x', r.x[1].toFixed(1)); r.labels[2].setAttribute('x', r.x[3].toFixed(1));
      }
      var headA = E.txt(partA, MID, 36, '先转、再拉', 22, E.ACC, 'middle', 900);
      var rowA = row(ROW_A, S, R, E.mul(S, R), E.ACC, ['拉', '转', '合成']);
      var firstG = E.el('g', {}, partA);
      var firstTag = E.tag(firstG, -30, -13, 60, 26, '先做', E.VIOLET, { size: 18, ink: E.VIOLET, weight: 700 });
      var order = E.vec(partA, E.VIOLET, 2.5, { head: 9 });
      var headB = E.txt(partA, MID, 292, '先拉、再转', 22, E.HI, 'middle', 900);
      var rowB = row(ROW_B, R, S, E.mul(R, S), E.HI, ['转', '拉', '合成']);
      // 两行之间：拉·转 ≠ 转·拉；交换弧线 + 红 ×
      var neqRow = E.el('g', {}, partA);
      E.txt(neqRow, MID - 68, 230, '拉·转', 24, E.ACC, 'middle', 900);
      E.txt(neqRow, MID, 232, '≠', 32, E.BAD, 'middle', 900);
      E.txt(neqRow, MID + 68, 230, '转·拉', 24, E.HI, 'middle', 900);
      var swap = E.el('path', { d: 'M 574 244 Q 630 270 686 244', fill: 'none', 'stroke-width': 2.5, 'stroke-linecap': 'round' }, partA);
      K.paint(swap, null, E.INK2);
      var swapHeads = E.el('g', {}, partA);
      [[574, 244, 205], [686, 244, -25]].forEach(function (h) {   // 两端各一个小三角：⇄
        var a = h[2] * Math.PI / 180, L = 9, W = 5.4;
        K.paint(E.el('path', { d: 'M ' + h[0] + ' ' + h[1] + ' L ' + (h[0] - L * Math.cos(a) + W * Math.sin(a)).toFixed(1) + ' ' + (h[1] - L * Math.sin(a) - W * Math.cos(a)).toFixed(1) +
          ' L ' + (h[0] - L * Math.cos(a) - W * Math.sin(a)).toFixed(1) + ' ' + (h[1] - L * Math.sin(a) + W * Math.cos(a)).toFixed(1) + ' Z' }, swapHeads), E.INK2);
      });
      var swapX = E.cross(partA, MID, 257, 10, E.BAD, 4);

      /* ══ ④ UR5：一节连一节，每过一个关节一次变换 ══ */
      var partB = E.el('g', {}, s);
      K.paint(E.el('line', { x1: 50, y1: 394, x2: 310, y2: 394, 'stroke-width': 3 }, partB), null, E.BORDER);
      var arm = E.arm(partB, 170, 394, 2.3);
      var jdots = LINKS.map(function () { return K.paint(E.el('circle', { r: 7, 'stroke-width': 2 }, partB), E.INK2, E.S1); });
      var hand = K.paint(E.el('circle', { r: 8, 'stroke-width': 2.5 }, partB), E.ACC, E.S1);
      var handTxt = E.txt(partB, 0, 0, '手', 22, E.ACC, 'start', 900);
      function tTag(tex, w) {
        var g = E.el('g', {}, partB), inner = E.el('g', {}, g);
        var r = K.paint(E.el('rect', { x: -w / 2, y: -16, width: w, height: 32, rx: 8, 'stroke-width': 2 }, inner), E.tint(E.VIOLET, 16, 'var(--demo-surface)'), E.VIOLET);
        E.tex(inner, 0, 0.34 * 22, tex, 22, E.VIOLET);
        return { g: g, inner: inner, rect: r };
      }
      var tags = [tTag('T_1', 50), tTag('T_2', 50), tTag('T_3', 50), tTag('T_4T_5T_6', 96)];
      var prodG = E.el('g', {}, partB), prod = K.svgRich(650, 170 + 0.34 * 32, '手 $= T_1\\,T_2\\,T_3\\,T_4\\,T_5\\,T_6$', { size: 32, anchor: 'middle', w: 380 });
      prod.setTone(E.INK); prodG.appendChild(prod);
      var prodArrow = E.arrow(partB, [[540, 212], [770, 212]], E.VIOLET, 2.5, { head: 11 });
      var prodTxt = E.txt(partB, 655, 240, '按顺序，一个接一个乘', 20, E.VIOLET, 'middle', 700);
      var ep11 = E.tag(partB, 555, 282, 200, 38, '第 11 集：正运动学', E.HI, { size: 20, ink: E.HI, weight: 900 });

      return {
        el: s,
        draw: function (st, t) {
          var fade = E.fade;
          var showA = fade(st, 0, 0.5) * (1 - fade(st, ROBOT - 0.1, 0.4));
          E.op(partA, showA.toFixed(3));
          E.op(partB, fade(st, ROBOT, 0.5).toFixed(3));

          /* ① 先转再拉：2.91「转」转 90°，4.19「拉长」横着拉到 2 倍；网格跟着变形，5.5 后收起 */
          var ra = fade(st, ROT_A, 0.7), sa = fade(st, STR_A, 0.7);
          var mA = E.mul(E.lerpM(S, sa), E.rot(Math.PI / 2 * ra));
          var pA = E.apply(mA, START), qA = at(pA);
          vecA.set(OX, OY, qA[0], qA[1]);
          dotA.setAttribute('cx', qA[0].toFixed(2)); dotA.setAttribute('cy', qA[1].toFixed(2));
          E.drawOn(trailA[0], ra); E.drawOn(trailA[1], sa);
          var wA = E.win(st, ROT_A - 0.2, 5.6, 0.4);
          /* ② 先拉再转：15.24「拉长」、16.08「转」；网格跟着变形，17.9 后收起 */
          var sb = fade(st, STR_B, 0.6), rb = fade(st, ROT_B, 0.7);
          var mB = E.mul(E.rot(Math.PI / 2 * rb), E.lerpM(S, sb));
          var pB = E.apply(mB, START), qB = at(pB), onB = fade(st, STR_B - 0.4, 0.3);
          vecB.set(OX, OY, qB[0], qB[1]);
          dotB.setAttribute('cx', qB[0].toFixed(2)); dotB.setAttribute('cy', qB[1].toFixed(2));
          E.op(vecB.g, onB); E.op(dotB, onB);
          E.drawOn(trailB[0], sb); E.drawOn(trailB[1], rb);
          var wB = E.win(st, STR_B - 0.2, 17.9, 0.4);
          pl.warp.set(wB > 0 ? mB : mA);
          E.op(pl.warp.g, (0.35 * Math.max(wA, wB)).toFixed(3));
          // 第二条路线出来时，第一条退一点（终点还看得清）
          E.op(vecA.g, (1 - 0.35 * fade(st, STR_B - 0.4, 0.4)).toFixed(3));
          E.op(lab0, fade(st, 0.2, 0.4));
          E.op(labA, fade(st, STR_A + 0.6, 0.3));
          E.op(labB, fade(st, LAND_B - 0.2, 0.3));

          /* 9.24「一步做完」：合成矩阵一跳，从 (3, 2) 直接到 (−4, 3)；11.6 后退场 */
          var pj = fade(st, JUMP, 0.55), jOn = E.win(st, JUMP, 12.6, 0.3);
          E.drawOn(jump.line, pj); E.op(jump.head, K.seg(pj, 0.9, 1));
          E.op(jump.g, jOn); E.op(jumpTxt, fade(st, JUMP + 0.2, 0.3) * jOn);
          var u = pj, gx = (1 - u) * (1 - u) * j0[0] + 2 * u * (1 - u) * jc[0] + u * u * j1[0], gy = (1 - u) * (1 - u) * j0[1] + 2 * u * (1 - u) * jc[1] + u * u * j1[1];
          ghost.setAttribute('cx', gx.toFixed(2)); ghost.setAttribute('cy', gy.toFixed(2));
          E.op(ghost, pj > 0 && pj < 1 ? 1 : pj >= 1 ? jOn * 0.6 : 0);

          /* 右边第一行：「转」2.91、「拉」4.19 依次出（拉出在转左边）；6.94「乘」出 =，7.8「合成」出合成矩阵 */
          layout(rowA); layout(rowB);
          E.op(headA, fade(st, 2.45, 0.3));
          E.pop(rowA.m[1].g, rowA.x[1], ROW_A, fade(st, ROT_A, 0.3), 0.7);
          E.pop(rowA.m[0].g, rowA.x[0], ROW_A, fade(st, STR_A, 0.3), 0.7);
          E.op(rowA.labels[1], fade(st, ROT_A, 0.3)); E.op(rowA.labels[0], fade(st, STR_A, 0.3));
          E.op(rowA.eq, fade(st, MUL, 0.3));
          E.pop(rowA.res.g, rowA.x[3], ROW_A, fade(st, MERGE, 0.35), 0.7);
          E.op(rowA.labels[2], fade(st, 8.28, 0.3));
          /* 11.21「先做」标在右边的「转」上，11.9「右边」从转往拉画小箭头 */
          firstG.setAttribute('transform', 'translate(' + rowA.x[1].toFixed(1) + ' 65)');
          E.pop(firstTag.g, 0, 0, fade(st, FIRST, 0.3), 0.7);
          var po = fade(st, RIGHT, 0.5), ox0 = rowA.x[1] - 18, ox1 = rowA.x[0] + 18;
          order.set(ox0, ROW_A + 56, ox0 + (ox1 - ox0) * po, ROW_A + 56); E.op(order.g, po > 0 ? 1 : 0);

          /* 右边第二行：14.0 标题；15.24「拉」出在右边、16.08「转」出在左边；17.61「落到」出 = 和合成矩阵 */
          E.op(headB, fade(st, HEAD_B, 0.3));
          E.pop(rowB.m[1].g, rowB.x[1], ROW_B, fade(st, STR_B, 0.3), 0.7);
          E.pop(rowB.m[0].g, rowB.x[0], ROW_B, fade(st, ROT_B, 0.3), 0.7);
          E.op(rowB.labels[1], fade(st, STR_B, 0.3)); E.op(rowB.labels[0], fade(st, ROT_B, 0.3));
          E.op(rowB.eq, fade(st, LAND_B, 0.3));
          E.pop(rowB.res.g, rowB.x[3], ROW_B, fade(st, LAND_B + 0.2, 0.35), 0.7);
          E.op(rowB.labels[2], fade(st, LAND_B + 0.4, 0.3));

          /* 18.04「完全不同」两个终点各圈一下，18.48「不同」中间红 ≠ */
          var ca = fade(st, DIFF, 0.3), cb = fade(st, DIFF + 0.2, 0.3);
          ringA.setAttribute('r', (17 + 8 * bell(st, DIFF, DIFF + 0.6)).toFixed(2)); E.op(ringA, ca);
          ringB.setAttribute('r', (17 + 8 * bell(st, DIFF + 0.2, DIFF + 0.8)).toFixed(2)); E.op(ringB, cb);
          E.pop(neqPts, (at(END_A)[0] + at(END_B)[0]) / 2 - 6, (at(END_A)[1] + at(END_B)[1]) / 2, fade(st, NEQ, 0.3), 0.5);

          /* ③ 21.9「顺序」两行之间出「拉·转 ≠ 转·拉」；22.85「不能随便交换」交换弧线、23.1 打红 × */
          E.pop(neqRow, MID, 222, fade(st, ORDER, 0.3), 0.8);
          var ps = fade(st, SWAP, 0.4);
          E.drawOn(swap, ps); E.op(swapHeads, K.seg(ps, 0.8, 1));
          E.pop(swapX, MID, 257, fade(st, 23.1, 0.25), 0.4);

          /* ④ 机器人：手臂慢慢动，标注跟着关节走 */
          arm.set(12 + 4 * Math.sin(TAU * t / 4), 64 + 5 * Math.sin(TAU * t / 3.1));
          var J = arm.joints(), tp = arm.tool();
          /* 25.59「一节连着一节」关节从底座到手腕依次亮，连杆依次上一下色 */
          J.forEach(function (p, k) {
            var on = fade(st, CHAIN + 0.13 * k, 0.2);
            jdots[k].setAttribute('cx', p[0].toFixed(1)); jdots[k].setAttribute('cy', p[1].toFixed(1));
            jdots[k].style.fill = on > 0.5 ? E.VIOLET : E.INK2;
            jdots[k].setAttribute('r', (5 + 2 * on + 3 * bell(st, CHAIN + 0.13 * k, CHAIN + 0.13 * k + 0.5)).toFixed(2));
            E.op(jdots[k], (0.6 + 0.4 * on).toFixed(3));
            var w = bell(st, CHAIN + 0.13 * k - 0.1, CHAIN + 0.13 * k + 0.6);
            arm.view.tint(LINKS[k], w > 0.02 ? E.VIOLET : null, w);
          });
          /* 27.1 起各关节旁依次出 T₁、T₂、T₃、T₄T₅T₆；31.57「一串矩阵」一起鼓一下 */
          var spots = [[J[0][0] - 62, J[0][1] + 16], [J[1][0] - 62, J[1][1] - 26], [J[2][0], J[2][1] - 40],
            [(J[3][0] + J[5][0]) / 2 + 80, (J[3][1] + J[5][1]) / 2 - 34]];
          tags.forEach(function (tg, k) {
            tg.g.setAttribute('transform', 'translate(' + spots[k][0].toFixed(1) + ' ' + spots[k][1].toFixed(1) + ')');
            E.pop(tg.inner, 0, 0, fade(st, TAGS[k], 0.3), 0.6);
            var b = bell(st, BUNCH + 0.08 * k, BUNCH + 0.08 * k + 0.45);
            tg.rect.style.strokeWidth = (2 + 2 * b).toFixed(2);
            E.glow(tg.rect, E.VIOLET, b > 0.05);
          });
          /* 29.85「手在哪」手亮起 */
          hand.setAttribute('cx', tp[0].toFixed(1)); hand.setAttribute('cy', tp[1].toFixed(1));
          var ph = fade(st, HAND, 0.3);
          E.op(hand, ph); hand.setAttribute('r', (8 + 5 * bell(st, HAND, HAND + 0.6)).toFixed(2));
          handTxt.setAttribute('x', (tp[0] + 14).toFixed(1)); handTxt.setAttribute('y', (tp[1] + 30).toFixed(1));
          E.op(handTxt, ph);
          /* 32.36「按顺序乘起来」出 手 = T₁…T₆，箭头从左往右画；33.75「第十一集」 */
          E.pop(prodG, 650, 170, fade(st, PROD, 0.35), 0.8);
          var pa = fade(st, PROD + 0.3, 0.8);
          E.drawOn(prodArrow.line, pa); E.op(prodArrow.head, K.seg(pa, 0.9, 1));
          E.op(prodTxt, fade(st, PROD + 0.5, 0.3));
          E.pop(ep11.g, 655, 301, fade(st, EP11, 0.3), 0.7);
        }
      };
    }
  };
})();
