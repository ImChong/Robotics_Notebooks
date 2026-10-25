// 片头主图（800×420）：三句旁白各一段——
// ① 回顾 003：左边网格 + UR5 侧视（肩关节 = 原点，手在 (3, 2)），右边「纸和笔」卡；「上一集」顶上出小牌，「纸和笔」卡弹出，
//    「矩阵乘向量」卡里写出 003 的算式（等号右边先空着），「转九十度」机械臂绕肩关节转 90°，「负二、三」落点标 (−2, 3)、算式补上结果；
// ② 10.10 起网格和机械臂淡出，纸和笔卡挪到左边；「动手写代码」右边出 Python 交互窗口（光标在闪），
//    「交给电脑」从卡到窗口画一支亮黄虚线箭头，「一下子算完」箭头上写「一下子算完？」；
// ③ 「向量」打 p = np.array([3, 2])，「矩阵」打 R 的两排，「相乘」打 R、p，「艾特」@ 落进去，「符号」出 array([-2,  3])，和卡上的结果各打 ✓；
//    窗口下面两个小牌：「数组」时出「向量、矩阵 = 数组」，「艾特」时出「相乘 = @」。
// 时刻取自配音的逐词时间（../ep004.py 的 [from, to] 等于每句配音实长，分镜时间 ≈ 念到的时刻）。
(function () {
  var E = window.EP004, K = E.K, H = E.HEX;
  var OX = 196, OY = 356, ARM_S = 1.68, A1 = 30;          // ① 网格原点（= UR5 肩关节）
  var PREV = 0.64, CARD = 1.57, FORM = 3.04, TURN = [5.38, 6.34], LAND = 7.89, OUT1 = 9.93;
  var MOVE = [10.03, 10.83], WIN = 10.96, LINK = 14.69, ASK = 15.47, LINK_OUT = 17.43;
  var P_LINE = 19.72, R_LINE = 20.33, KEYS = [21.72, 21.78, 22.48, 22.52, 22.57], RES = 22.8, OK = 23.05, PILL1 = 21.13, PILL2 = 22.48;
  var LINKS = ['upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link'];
  var CW = 280, CH = 210;                                   // 纸和笔卡的宽、高：① 在右边 (484, 96)，② 起挪到左边 (14, 96)

  E.hero = function (K) {
    var s = K.sceneSvg('片头主图：上一集用纸和笔算了矩阵乘向量，机械臂肩关节转 90 度，手从 (3, 2) 转到 (−2, 3)；这一集把手算的交给电脑：用 Python 的 NumPy，向量和矩阵都是数组，相乘用 @ 符号，一句算完');

    /* ── ① 网格 + UR5 ── */
    var gridG = E.el('g', {}, s);
    var under = E.el('g', {}, gridG);
    var arm = E.armAt(gridG, OX, OY, ARM_S, 260);
    var u = arm.aim(3, 2, A1);
    var pl = E.plane(under, { ox: OX, oy: OY, u: u, x0: -3, x1: 4, y0: -1, y1: 4 });
    var P = pl.P, hp = P(3, 2), tgt = P(-2, 3);
    var ghost = E.vec(gridG, E.ACC, 3, { head: 12, dash: '6 6' });
    var pos = E.vec(gridG, E.ACC, 5, { head: 15 });
    var land = E.txt(gridG, tgt[0] - 4, tgt[1] - 22, '(−2, 3)', 22, E.ACC, 'middle', 900);
    E.halo(land);
    var prev = E.tag(s, 300, 8, 200, 36, '上一集：纸和笔', E.INK2, { size: 20, ink: E.INK2, weight: 700, fill: E.S2 });

    /* ── 纸和笔卡（外面一层 g 管挪位置） ── */
    var cardO = E.el('g', {}, s), card = E.el('g', {}, cardO);
    K.paint(E.el('rect', { x: 0, y: 0, width: CW, height: CH, rx: 16, 'stroke-width': 2.5 }, card), E.S2, E.BORDER);
    // 一支铅笔（笔杆 + 笔尖）
    var pen = E.el('g', {}, card);
    K.paint(E.el('path', { d: 'M 26 40 L 60 14 L 68 24 L 34 50 Z', 'stroke-width': 2, 'stroke-linejoin': 'round' }, pen), E.tint(E.HI, 40, E.S2), E.HI);
    K.paint(E.el('path', { d: 'M 26 40 L 34 50 L 20 56 Z', 'stroke-width': 2, 'stroke-linejoin': 'round' }, pen), E.S2, E.HI);
    E.txt(card, 84, 44, '纸和笔', 26, E.INK, 'start', 900);
    var form = E.tex(card, CW / 2, 134, '', 30, E.INK, 'middle', { w: CW - 16, h: 110 });
    var okCard = E.el('g', {}, card);
    E.check(okCard, CW - 52, 22, 26, E.GOOD, 4);

    /* ── ②③ Python 交互窗口 ── */
    var winG = E.el('g', {}, s);
    var WX = 420, WY = 60, WW = 366, WH = 270;
    var py = E.code(winG, WX, WY, WW, WH, { title: 'Python', size: 20, lh: 30 });
    py.line('>>> ', WIN, { dur: 0 });                       // 第 0 页：空提示符等着；第 1 页起正式打
    py.line('>>> p = np.array([{a:3}, {a:2}])', P_LINE, { dur: 0.5, page: 1 });
    py.line('>>> R = np.array([[{g:0}, {w:-1}],', R_LINE, { dur: 0.4, page: 1 });
    py.line('...               [{g:1},  {w:0}]])', R_LINE + 0.43, { dur: 0.35, page: 1 });
    py.line('>>> R {h:@} p', KEYS[0], { keys: KEYS, page: 1 });
    var L_RES = py.line('array([{a:-2},  {a:3}])', RES, { out: true, page: 1 });
    var okWinO = E.el('g', {}, winG), okWin = E.el('g', {}, okWinO);   // 外层跟着输出那一句走，里层弹出
    E.check(okWin, 0, 0, 22, E.GOOD, 4);
    var pills = [E.tag(s, WX, 350, 200, 40, '向量、矩阵 = 数组', E.ACC, { size: 21, ink: E.ACC, weight: 900 }),
      E.tag(s, WX + 212, 350, 134, 40, '相乘 = @', E.HI, { size: 21, ink: E.HI, weight: 900 })];

    /* ── ② 交给电脑：卡 → 窗口 ── */
    var linkG = E.el('g', {}, s);
    var link = E.arrow(linkG, [[CW + 26, 200], [WX - 12, 200]], E.HI, 3.5, { dash: '8 6', head: 13 });
    var ask = E.txt(linkG, (CW + 14 + WX) / 2, 178, '一下子算完？', 19, E.HI, 'middle', 900);

    return {
      el: s,
      draw: function (t, still, rt) {
        if (rt === undefined) rt = t;
        var fade = E.fade;

        /* ① 网格 + UR5：转 90°，手的位置箭头跟着转 */
        E.op(gridG, fade(t, 0, 0.5) * (1 - fade(t, OUT1, 0.4)));
        E.pop(prev.g, 400, 26, fade(t, PREV, 0.35) * (1 - fade(t, OUT1, 0.4)), 0.85);
        arm.view.clearTint();
        var tw = Math.sin(Math.PI * K.seg(t, TURN[0] - 0.1, TURN[1] + 0.3));
        if (tw > 0) arm.view.tint(LINKS, E.VIOLET, tw);
        var tp = arm.turn(Math.PI / 2 * fade(t, TURN[0], TURN[1] - TURN[0]));
        pos.set(OX, OY, tp[0], tp[1]);
        ghost.set(OX, OY, hp[0], hp[1]);
        E.op(ghost.g, 0.6 * fade(t, TURN[0], 0.3));
        E.op(land, fade(t, LAND, 0.3));

        /* 纸和笔卡：① 在右边，② 挪到左边 */
        var mv = fade(t, MOVE[0], MOVE[1] - MOVE[0]);
        cardO.setAttribute('transform', 'translate(' + (484 + (14 - 484) * mv).toFixed(1) + ' 96)');
        E.pop(card, CW / 2, CH / 2, fade(t, CARD, 0.35), 0.85);
        form.setTex(E.mat2([0, -1, 1, 0], [H.good, H.warn]) + E.col([3, 2], H.acc) + '=' + (t >= LAND ? E.col([-2, 3], H.acc) : '\\,?'));
        E.op(form, fade(t, FORM, 0.4));
        E.pop(okCard, CW - 40, 34, fade(t, OK, 0.3), 0.5);

        /* ②③ 交互窗口 */
        E.pop(winG, WX + WW / 2, WY + WH / 2, fade(t, WIN, 0.35), 0.9);
        py.draw(t, rt);
        okWinO.setAttribute('transform', 'translate(' + (py.x(16) + 8) + ' ' + (py.y(L_RES) - 18) + ')');
        E.pop(okWin, 11, 11, fade(t, OK, 0.3), 0.5);
        py.hl(L_RES, E.ACC, fade(t, RES, 0.3) * 0.8);
        E.pop(pills[0].g, WX + 100, 370, fade(t, PILL1, 0.3), 0.85);
        E.pop(pills[1].g, WX + 279, 370, fade(t, PILL2, 0.3), 0.85);

        /* 卡 → 窗口的箭头 */
        var lp = fade(t, LINK, 0.6);
        E.drawOn(link.line, lp); E.op(link.head, K.seg(lp, 0.85, 1));
        E.op(linkG, 1 - fade(t, LINK_OUT, 0.4));
        E.op(ask, fade(t, ASK, 0.3));
      }
    };
  };
})();
