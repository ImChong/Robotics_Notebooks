// 片头主图（800×420）：三句旁白各一段——
// ① 「第二季」顶上出小牌「第二季 · 机器人学骨架」；「第七集」「位姿」左边斜着看的 UR5 手上长出坐标架，「朝向」坐标架鼓一下，
//    「位置」底座到手画一条虚线，「四乘四的矩阵」右边弹出位姿卡 [R p; 0 1]；
// ② 10.1 起卡片收起：「动起来」UR5 的肘开始转，小臂往上抬，「刚体」小臂上色，「速度」小臂上三个点各长出速度箭头（大小、方向跟着算），
//    「一个量」右边写「速度：一个量？」；
// ③ 「运动旋量」右边换成六维列向量卡：「角速度三个数」上三格框紫，「线速度三个数」下三格框蓝，「六个数」写「6 个数」；
//    「绕着一根轴」左边换成三维的螺旋运动小图：竖直的螺旋轴（琥珀虚线）、绕它走螺旋线的小方块，「边转边移」写出四个字。
// 时刻取自配音的逐词时间（../ep009.py 的 [from, to] 等于每句配音实长，分镜时间 = 念到这个词的时刻）。
(function () {
  var E = window.EP009, K = E.K, H = E.HEX;
  var D = Math.PI / 180;
  var T = {
    season: 2.02, ep7: 4.43, pose: 5.51, orient: 6.06, pos: 6.69, mat: 8.07, out1: 10.1,
    move: 10.93, body: 12.77, vel: 14.82, one: 15.81,
    twist: 18.61, w3: 19.63, v3: 20.94, six: 22.19, any: 23.6, axis: 25.85, screw: 26.86
  };

  E.hero = function (K) {
    var s = K.sceneSvg('片头主图：第二季机器人学骨架；第七集讲过位姿是朝向加位置，写成四乘四的矩阵；这一集让它动起来，机械臂小臂上各点的速度不一样，怎么用一个量说清楚？答案是运动旋量：角速度三个数加线速度三个数，一共六个数；任何刚体的速度都能看成绕一根轴边转边移');

    var season = E.tag(s, 250, 6, 300, 40, '第二季 · 机器人学骨架', E.HI, { size: 22, ink: E.HI, weight: 900, rx: 12 });

    /* ── 左：UR5 斜视 ── */
    var armG = E.el('g', {}, s);
    var ur = window.Robot3D.ur5(armG, 70, 408, 3.0, { az: 32, el: 16 }), V = ur.view, THREE = window.Robot3D.THREE;
    V.camera({ box: [0, 46, 432, 374] });
    var POSE0 = { shoulder_pan_joint: 20 * D, shoulder_lift_joint: -95 * D, elbow_joint: 95 * D, wrist_1_joint: -90 * D, wrist_2_joint: -90 * D, wrist_3_joint: 0 };
    V.joints(POSE0);
    var RIGID = ['forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link'];
    var over = E.el('g', {}, armG);
    var posLine = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '7 6' }, over), null, E.VEL);
    var baseDot = K.paint(E.el('circle', { r: 6 }, over), E.HI);
    var toolAx = [E.X, E.Y, E.Z].map(function (c) { return E.vec(over, c, 4.5, { head: 13 }); });
    var vels = [0, 1, 2].map(function () { return E.vec(over, E.VEL, 4.5, { head: 14 }); });
    var vdots = [0, 1, 2].map(function () { return K.paint(E.el('circle', { r: 6, 'stroke-width': 2.5 }, over), E.VEL, E.S1); });

    /* ── 左（③）：螺旋运动小图 ── */
    var scr = E.el('g', {}, s), proj = E.view3(28, 18), SL = 120, SX = 236, SY = 392, Rr = 0.8, HP = 0.2;
    function S3(v) { var q = proj(v); return [SX + SL * q[0], SY - SL * q[1]]; }
    var a0 = S3([0, 0, -0.15]), a1 = S3([0, 0, 2.6]);
    // 按远近遮挡：远的半圈螺旋线画在轴后面，近的在前面；方块每帧按它和轴谁近放到轴前或轴后
    var hBack = K.paint(E.el('path', { fill: 'none', 'stroke-width': 2.2, 'stroke-dasharray': '4 6' }, scr), null, E.VEL);
    var axis = K.paint(E.el('path', { d: 'M ' + a0[0].toFixed(1) + ' ' + a0[1].toFixed(1) + ' L ' + a1[0].toFixed(1) + ' ' + a1[1].toFixed(1), fill: 'none', 'stroke-width': 3.5, 'stroke-dasharray': '9 7' }, scr), null, E.WARN);
    var hFront = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3.2 }, scr), null, E.VEL);
    hBack.style.opacity = 0.55;
    var block = K.paint(E.el('path', { 'stroke-width': 2.5 }, scr), E.tint(E.BODY, 34, 'var(--demo-surface)'), E.BODY);
    var axLab = E.halo(E.txt(scr, a1[0] + 12, a1[1] + 6, '一根轴', 24, E.WARN, 'start', 900));
    var scrLab = E.halo(E.txt(scr, 122, 402, '边转边移', 28, E.GOOD, 'end', 900));   // 写在左下角：螺旋线贴地那一段在轴的右边，方块走到这一带时已经升高了

    /* ── 右：位姿卡 / 问句 / 六维列向量 ── */
    var RX = 440;
    var card = E.el('g', {}, s);
    K.paint(E.el('rect', { x: RX, y: 70, width: 330, height: 280, rx: 18, 'stroke-width': 2.5 }, card), E.tint(E.VEL, 8, 'var(--demo-surface)'), E.BORDER);
    E.txt(card, RX + 165, 112, '位姿 = 朝向 + 位置', 26, E.INK, 'middle', 900);
    E.tex(card, RX + 165, 230, '\\begin{bmatrix}R & \\textcolor{' + H.violet + '}{p}\\\\ 0 & 1\\end{bmatrix}', 44, E.INK, 'middle', { w: 240, h: 130 });
    E.txt(card, RX + 165, 320, '4 × 4 矩阵（第 7 集）', 22, E.INK2, 'middle', 700);
    var ask = E.el('g', {}, s);
    E.txt(ask, RX + 165, 170, '速度：', 30, E.INK, 'middle', 900);
    E.txt(ask, RX + 165, 230, '一个量？', 44, E.HI, 'middle', 900);
    var tw = E.el('g', {}, s), CX = RX + 150, CY = 232;
    K.paint(E.el('rect', { x: RX, y: 52, width: 330, height: 360, rx: 18, 'stroke-width': 2.5 }, tw), E.tint(E.HI, 6, 'var(--demo-surface)'), E.HI);
    E.txt(tw, RX + 165, 92, '运动旋量', 30, E.HI, 'middle', 900);
    var col = E.colvec(tw, CX, CY, ['ωx', 'ωy', 'ωz', 'vx', 'vy', 'vz'], { size: 30, dy: 42, w: 92, colors: [E.OMEGA, E.OMEGA, E.OMEGA, E.VEL, E.VEL, E.VEL] });
    col.cells.forEach(function (c) { c.style.fontFamily = 'KaTeX_Math, serif'; c.style.fontStyle = 'italic'; });
    var wLab = E.txt(tw, CX + 58, CY - 50, '角速度', 24, E.OMEGA, 'start', 900), vLab = E.txt(tw, CX + 58, CY + 72, '线速度', 24, E.VEL, 'start', 900);
    var sixLab = E.tag(tw, RX + 210, CY - 6, 100, 36, '6 个数', E.HI, { size: 22, ink: E.HI, weight: 900, rx: 10 });

    // 小臂上一点（相对肘、腕的比例 k）在世界里的位置
    function forearmPt(k) {
      var e = V.world('forearm_link'), w = V.world('wrist_1_link');
      return e.clone().lerp(w, k);
    }

    return {
      el: s,
      draw: function (t) {
        var f = E.fade, b = E.bump;
        E.pop(season.g, 400, 26, f(t, T.season, 0.35), 0.85);

        /* ── UR5：动起来后肘慢慢往上转 ── */
        var bend = -30 * D * E.clamp((t - T.move) / 12, 0, 1);
        var pose = Object.assign({}, POSE0); pose.elbow_joint = POSE0.elbow_joint + bend;
        V.joints(pose);
        V.clearTint();
        var bt = f(t, T.body, 0.5);
        if (bt > 0) V.tint(RIGID, E.BODY, 0.4 * bt);
        E.op(armG, f(t, 0, 0.4) * (1 - f(t, T.any, 0.6)));
        // 手上的坐标架、底座到手的虚线
        var tl = V.robot.links.tool0 || V.robot.frames.tool0;
        V.robot.updateMatrixWorld(true);
        var to = new THREE.Vector3().setFromMatrixPosition(tl.matrixWorld), tp0 = V.project(to);
        [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)].forEach(function (ax, i) {
          var tp = V.project(to.clone().addScaledVector(ax.transformDirection(tl.matrixWorld), 0.14));
          toolAx[i].set(tp0[0], tp0[1], tp[0], tp[1]);
          var ob = b(t, T.orient, 0.8);
          toolAx[i].line.style.strokeWidth = 4.5 + 2.5 * ob;
        });
        var frOn = f(t, T.pose, 0.4) * (1 - f(t, T.out1, 0.4));
        toolAx.forEach(function (a) { E.op(a.g, frOn); });
        var bp = V.project(new THREE.Vector3(0, 0, 0));
        baseDot.setAttribute('cx', bp[0].toFixed(1)); baseDot.setAttribute('cy', bp[1].toFixed(1));
        posLine.setAttribute('d', 'M ' + bp[0].toFixed(1) + ' ' + bp[1].toFixed(1) + ' L ' + tp0[0].toFixed(1) + ' ' + tp0[1].toFixed(1));
        var po = f(t, T.pos, 0.4) * (1 - f(t, T.out1, 0.4));
        E.op(posLine, po); E.op(baseDot, po);
        // 小臂上三个点的速度：肘转一点点，看点往哪走（数值求导），按比例画箭头
        var vOn = f(t, T.vel, 0.5);
        var ks = [0.35, 0.7, 1.0], P0 = ks.map(forearmPt);
        pose.elbow_joint -= 0.02; V.joints(pose);
        var P1 = ks.map(forearmPt);
        pose.elbow_joint += 0.02; V.joints(pose);
        P0.forEach(function (p, i) {
          var a = V.project(p), q = V.project(P1[i]), g = 0.32 / 0.02 * vOn;   // 箭头 = 肘转 0.32 弧度时这一点走的位移（只看大小、方向的比例）
          vels[i].set(a[0], a[1], a[0] + (q[0] - a[0]) * g, a[1] + (q[1] - a[1]) * g);
          E.op(vels[i].g, vOn > 0 ? 1 : 0);
          vdots[i].setAttribute('cx', a[0].toFixed(1)); vdots[i].setAttribute('cy', a[1].toFixed(1));
          E.op(vdots[i], f(t, T.vel, 0.3));
        });

        /* ── 右：位姿卡 → 问句 → 运动旋量 ── */
        E.pop(card, RX + 165, 210, f(t, T.mat, 0.35) * (1 - f(t, T.out1, 0.35)), 0.88);
        E.pop(ask, RX + 165, 205, f(t, T.one, 0.35) * (1 - f(t, T.twist - 0.3, 0.3)), 0.85);
        E.pop(tw, RX + 165, 232, f(t, T.twist, 0.4), 0.9);
        col.rangeBox(0, 2, E.OMEGA, f(t, T.w3, 0.3)); col.rangeBox(3, 5, E.VEL, f(t, T.v3, 0.3));
        col.cells.forEach(function (c, i) { E.op(c, f(t, i < 3 ? T.w3 : T.v3, 0.3)); });
        E.op(wLab, f(t, T.w3, 0.3)); E.op(vLab, f(t, T.v3, 0.3));
        E.pop(sixLab.g, RX + 260, CY + 12, f(t, T.six + 0.3, 0.3), 0.8);

        /* ── 左（③）：螺旋运动 ── */
        E.op(scr, f(t, T.any + 0.3, 0.6));
        var th = 1.4 * Math.max(0, t - T.any - 0.3), thMax = 9.6;
        th = th % (thMax + 2);   // 走完一段（约 1.5 圈）停一下再从头走
        th = Math.min(th, thMax);
        function C(a) { return [Rr * Math.cos(a - 0.8), Rr * Math.sin(a - 0.8), 0.15 + HP * a]; }
        var fr = '', bk = '', prev = null, n = Math.max(2, Math.ceil(th / 0.06));
        for (var k = 0; k <= n; k++) {
          var a = th * k / n, p = S3(C(a)), isF = proj.depth([Math.cos(a - 0.8), Math.sin(a - 0.8), 0]) < 0, sg = p[0].toFixed(1) + ' ' + p[1].toFixed(1);
          if (isF) fr += (prev === true ? ' L ' : ' M ') + sg; else bk += (prev === false ? ' L ' : ' M ') + sg;
          if (prev !== null && prev !== isF) { if (isF) bk += ' L ' + sg; else fr += ' L ' + sg; }
          prev = isF;
        }
        hFront.setAttribute('d', th > 0.02 ? fr : ''); hBack.setAttribute('d', th > 0.02 ? bk : '');
        // 小方块：用一个随转角转的扁方块（顶面 + 两个侧面）
        var c = C(th), M = E.rot3([0, 0, 1], th), h = 0.24;
        var cs = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(function (u) { var w = E.mv3(M, [u[0] * h, u[1] * h, 0]); return [c[0] + w[0], c[1] + w[1], c[2]]; });
        var topF = cs.map(function (p) { return S3([p[0], p[1], p[2] + h]); }), botF = cs.map(function (p) { return S3([p[0], p[1], p[2] - h]); });
        var d = 'M ' + topF.map(function (p) { return p[0].toFixed(1) + ' ' + p[1].toFixed(1); }).join(' L ') + ' Z';
        // 两个朝镜头的侧面
        for (k = 0; k < 4; k++) {
          var j = (k + 1) % 4, mid = [(cs[k][0] + cs[j][0]) / 2 - c[0], (cs[k][1] + cs[j][1]) / 2 - c[1], 0];
          if (proj.depth(mid) < 0) d += ' M ' + [topF[k], topF[j], botF[j], botF[k]].map(function (p) { return p[0].toFixed(1) + ' ' + p[1].toFixed(1); }).join(' L ') + ' Z';
        }
        block.setAttribute('d', d);
        if (proj.depth(c) > proj.depth([0, 0, c[2]])) { if (block.nextSibling !== axis) scr.insertBefore(block, axis); }
        else if (block.previousSibling !== hFront) scr.insertBefore(block, hFront.nextSibling);
        var ab = b(t, T.axis, 0.9);
        axis.style.strokeWidth = 3.5 + 2.5 * ab; E.glow(axis, E.WARN, ab > 0.05);
        E.op(axLab, f(t, T.axis, 0.3)); E.op(scrLab, f(t, T.screw, 0.3));
      }
    };
  };
})();
