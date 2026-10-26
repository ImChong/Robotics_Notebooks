// 片头主图（800×420）：三句旁白各一段——
// ① 回顾：「上一集」顶上出小牌；G1 站在左边，「身子的朝向」骨盆上长出三根小箭头（前 绿、左 琥珀、上 蓝）并原地往左转一点，
//    「三乘三」右边弹出矩阵卡，三列的数跟着转身实时变，「旋转矩阵」卡下写出名字；
// ② 9.3 起卡片淡出、G1 转回朝前并退到左边：「两个朝向」右边出一台朝左（转 90°）的 G1，「中间值」中间留一个虚线圈和问号，
//    「直接平均」中间上方写出「(R₁ + R₂) ÷ 2 ?」；
// ③ 「两条规矩」顶上两块规矩牌，「直接平均」中间换成按平均出来的矩阵摆的 G1——转了 45°，横向被压成 0.71 倍（瘦了一圈），
//    「破坏」两块规矩牌各打红 ×，「不是旋转」中间下方写「不是旋转」。
// 时刻取自配音的逐词时间（../ep005.py 的 [from, to] 等于每句配音实长，分镜时间 = 念到这个词的时刻）。
(function () {
  var E = window.EP005, K = E.K, H = E.HEX;
  var D = Math.PI / 180;
  var PREV = [0.64, 3.9], FRAME = 5.66, TURN = [5.75, 8.0, 40], CARD = 6.7, NAME = 7.46, OUT1 = 9.3;
  var MOVE = [9.4, 10.4], TWO = 12.47, MID = 13.17, FORM = 15.47;
  var RULES = [19.39, 19.77], AVG = 20.34, SQUASH = [20.75, 21.7], BREAK = 22.0, NOTROT = 23.57;
  var BIG = { x: 250, y: 372, h: 318 }, SLOT = [130, 400, 670], SY = 358, SH = 238;   // 槽位里的脚下虚线圆不碰下面的标签

  E.hero = function (K) {
    var s = K.sceneSvg('片头主图：身子的朝向是一个三乘三的旋转矩阵；机器人从朝前转到朝左，中间的朝向能不能把两个矩阵直接平均？直接平均会破坏旋转矩阵的两条规矩，机器人被压瘦，已经不是旋转');

    function ringPath(pts) { return pts.map(function (p, k) { return (k ? 'L ' : 'M ') + p[0].toFixed(1) + ' ' + p[1].toFixed(1); }).join(' ') + ' Z'; }
    // 一台 G1 + 脚下的虚线圆 + 骨盆上的三根小箭头，整组可以挪、缩放
    function robot(x, y, h) {
      var g = E.el('g', {}, s);
      var ring = K.paint(E.el('path', { fill: 'none', 'stroke-width': 2, 'stroke-dasharray': '6 6' }, g), null, E.BORDER);
      var r = E.g1(g, x, y, h, { az: 135, el: 20 });   // 从前左方斜看：转 0° 偏向画面左、转 90° 偏向画面右，转 45° 正对镜头
      var fr = E.frame3(g, 4.5);
      return {
        g: g, r: r, fr: fr,
        set: function (yaw, sq) { r.set(yaw, sq); ring.setAttribute('d', ringPath(r.ring(0.45))); fr.set(r.frame(0.5)); }
      };
    }
    var big = robot(BIG.x, BIG.y, BIG.h);
    var right = robot(SLOT[2], SY, SH), mid = robot(SLOT[1], SY, SH);
    right.set(90 * D, 1);

    var prev = E.tag(s, 300, 8, 200, 36, '上一集：矩阵代码', E.INK2, { size: 20, ink: E.INK2, weight: 700, fill: E.S2 });

    /* ── ① 矩阵卡 ── */
    var card = E.el('g', {}, s);
    K.paint(E.el('rect', { x: 476, y: 70, width: 300, height: 270, rx: 18, 'stroke-width': 2.5 }, card), E.tint(E.VIOLET, 10, 'var(--demo-surface)'), E.VIOLET);
    E.txt(card, 626, 112, '身子的朝向', 28, E.INK, 'middle', 900);
    var m3 = E.tex(card, 626, 222, '', 30, E.INK, 'middle', { w: 290, h: 150 });
    var name = E.txt(card, 626, 318, '三乘三的旋转矩阵', 24, E.VIOLET, 'middle', 900);

    /* ── ② 三个槽位的标签、中间的问号、平均的式子 ── */
    function label(x, s1, color) { return E.halo(E.txt(s, x, 412, s1, 22, color, 'middle', 900)); }
    var labL = label(SLOT[0], '朝向 R₁', E.INK), labR = label(SLOT[2], '朝向 R₂', E.INK);
    var q = E.el('g', {}, s);
    K.paint(E.el('circle', { cx: SLOT[1], cy: 244, r: 60, fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '8 7' }, q), null, E.HI);
    E.txt(q, SLOT[1], 268, '?', 70, E.HI, 'middle', 900);
    var labM = label(SLOT[1], '中间？', E.HI);
    var form = E.tex(s, SLOT[1], 92, '(R_1+R_2)\\div 2\\;?', 30, E.INK, 'middle', { w: 260, h: 50 });

    /* ── ③ 两块规矩牌 + 打叉、「不是旋转」 ── */
    var rules = [{ x: 40, w: 340, s: '① 每列长度 1、互相垂直' }, { x: 420, w: 340, s: '② 行列式 = 1' }].map(function (d) {
      var g = E.el('g', {}, s);
      var tg = E.tag(g, d.x, 10, d.w, 42, d.s, E.GOOD, { size: 22, ink: E.INK, weight: 900, rx: 12 });
      var x = E.cross(g, d.x + d.w - 24, 31, 10, E.BAD, 4.5);
      return { g: g, tg: tg, x: x, cx: d.x + d.w / 2 };
    });
    var notG = E.el('g', {}, s);                            // 「✗」用 path 画，不靠字体
    E.cross(notG, SLOT[1] - 66, 404, 9, E.BAD, 4);
    E.halo(E.txt(notG, SLOT[1] + 12, 412, '不是旋转', 22, E.BAD, 'middle', 900));

    return {
      el: s,
      draw: function (t) {
        var f = E.fade;

        /* 大 G1：① 站在左边，「朝向」时往左转 40°；② 转回 0° 并退到左边的槽位（整组缩放） */
        var turn = TURN[2] * D * f(t, TURN[0], TURN[1] - TURN[0]) * (1 - f(t, MOVE[0], MOVE[1] - MOVE[0]));
        big.set(turn, 1);
        var mv = f(t, MOVE[0], MOVE[1] - MOVE[0]), sc = 1 + (SH / BIG.h - 1) * mv;
        var X = BIG.x + (SLOT[0] - BIG.x) * mv, Y = BIG.y + (SY - BIG.y) * mv;
        big.g.setAttribute('transform', 'translate(' + X.toFixed(2) + ' ' + Y.toFixed(2) + ') scale(' + sc.toFixed(4) + ') translate(' + (-BIG.x) + ' ' + (-BIG.y) + ')');
        E.op(big.g, f(t, 0, 0.5));
        var pf = f(t, FRAME, 0.5);
        E.op(big.fr.g, pf);
        E.pop(prev.g, 400, 26, f(t, PREV[0], 0.35) * (1 - f(t, PREV[1], 0.4)), 0.85);

        /* ① 矩阵卡：三列跟着转身实时变 */
        var c = Math.cos(turn), sn = Math.sin(turn);
        m3.setTex(E.mat3tex([[c, -sn, 0], [sn, c, 0], [0, 0, 1]], [H.good, H.warn, H.acc]));
        var out1 = 1 - f(t, OUT1, 0.4);
        E.pop(card, 626, 205, f(t, CARD, 0.35) * out1, 0.88);
        E.op(name, f(t, NAME, 0.3));

        /* ② 右边朝左的 G1、中间的问号、式子 */
        E.pop(right.g, SLOT[2], 300, f(t, TWO, 0.4), 0.85);
        E.op(labL, f(t, TWO, 0.3)); E.op(labR, f(t, TWO + 0.1, 0.3));
        var qOut = 1 - f(t, AVG, 0.3);
        E.pop(q, SLOT[1], 244, f(t, MID, 0.35) * qOut, 0.7);
        E.op(labM, f(t, MID, 0.3) * qOut);
        E.pop(form, SLOT[1], 82, f(t, FORM, 0.35), 0.85);

        /* ③ 规矩牌；中间按平均出来的矩阵摆的 G1：转 45°，横向从 1 压到 cos 45° ≈ 0.71 */
        rules.forEach(function (r, i) {
          E.pop(r.g, r.cx, 31, f(t, RULES[i], 0.3), 0.85);
          E.drawOn(r.x, f(t, BREAK + 0.25 * i, 0.4));
          r.tg.rect.style.stroke = t >= BREAK + 0.25 * i ? E.BAD : E.GOOD;
        });
        var sq = 1 + (Math.cos(45 * D) - 1) * f(t, SQUASH[0], SQUASH[1] - SQUASH[0]);
        mid.set(45 * D, sq);
        mid.r.view.clearTint();
        var red = f(t, BREAK, 0.5);
        if (red > 0) mid.r.tint('all', E.BAD, 0.55 * red);
        E.pop(mid.g, SLOT[1], 300, f(t, AVG, 0.4), 0.85);
        E.op(notG, f(t, NOTROT, 0.3));
      }
    };
  };
})();
