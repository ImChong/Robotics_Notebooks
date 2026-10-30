// 009 第 4 幕：螺旋运动——边转边移。左边是斜着看的三维小场景（z 朝上），右栏放要点。
// 句 1「平面」左边先放一张俯视的平面小图：一块板绕一个黄点转，角上一点描出圆；「三维」换成三维：一根竖直的转轴（琥珀虚线）、离轴一段的小方块，
//    「边转」方块绕轴转起来，「沿着转轴移动」同时往上走，方块中心描出螺旋线（右栏两行）；
// 句 2「拧螺丝」换成一颗螺丝拧进木板：螺丝头上的槽跟着转，「每转一圈」「往里走一小段」螺纹往下走、整颗往板里沉；
//    「上面的点」螺丝头边上一点亮起来，「螺旋线」它走过的路描成螺旋线；
// 句 3 换回方块：「定理」右栏出定理卡片，方块接着沿螺旋线走，「螺旋轴」轴上写「螺旋轴」；
// 句 4 方块停住：「方向」轴上画出方向箭头 ŝ，「一个点」轴上点出 q，「螺距」「每转一弧度」螺旋线上取相隔 1 弧度的两点，
//    「走多远」量出两点的高度差、标 h（右栏三行依次写出）；
// 句 5「螺距是零」右栏画一个圆圈图标「h = 0：只转不移」，「只移不转」画一支直箭头「h = ∞：只移不转」；
// 句 6 右栏换成公式：运动旋量 = (ω, v)，「后三个数」v 分成两份：「轴不过原点」左边画出原点和原点到 q 的虚线、右栏第一份 −ω × q（标「轴不过原点：透明板」），
//    「沿着轴的移动」h 的量尺鼓一下、右栏第二份 h ω（标「沿轴移动：角速度 × 螺距」）。
// 时刻取自配音的逐词时间，见 ../ep009.py。
(function () {
  var E = window.EP009, K = E.K, H = E.HEX;
  var D = Math.PI / 180, RX = E.ARM.RX;
  var T = {
    plane: 0.63, plane2: 4.22, to3: 4.91, turn: 6.83, rise: 7.59, end3: 9.58,
    screw: 9.88, screwTurn: [11.27, 13.23], more: [13.58, 16.98], head: 14, helix: 15.72, out4: 17.28,
    thm: 18.32, block2: 17.98, any: 18.98, rot: 22.32, mov: 22.69, name: 24.51, stop: 26.08,
    dirS: 29.16, pointQ: 30.24, pitch: 31.31, perRad: 32.63, far: 33.84,
    h0: 36.15, h0t: 36.95, hInf: 38.18, hInfT: 39.92,
    form: 41.65, two: 44.11, part1: 45.09, origin: 45.62, plate: 48.09, part2: 49.36, hPulse: 50.1, times: 50.97
  };
  var END = 53.17;
  // 三维场景：世界原点 O 在画面 (WX, WY)；一格 L px；转轴竖直，穿过 q；方块中心离轴 R，每转 1 弧度往上走 h
  var WX = 78, WY = 372, L = 118, Qw = [1.15, 0.55, 0], R = 0.85, HP = 0.22, A0 = -40 * D, CUBE = 0.27;
  var proj = E.view3(28, 18);
  function S3(v) { var q = proj(v); return [WX + L * q[0], WY - L * q[1]]; }
  // 方块角度随时间：句 1「边转」起转到句 1 结束（2.9 弧度）；句 3 开头起接着转到「螺旋轴」念完（再 5.7 弧度）；之后停住
  function theta(st) {
    var a = 2.9 * E.clamp((st - T.turn) / (T.end3 - T.turn), 0, 1);
    return a + 5.7 * E.clamp((st - T.block2) / (T.stop - T.block2), 0, 1);
  }
  var TH_A = 6.6;                                          // 量螺距用的那一段：从 TH_A 到 TH_A + 1（方块停在 8.6，不升到「螺旋轴」三个字那么高）

  E.scenes[3] = {
    title: '螺旋运动：边转边移',
    dur: END,
    cues: [
      { at: 0, s: '平面里刚体绕一个点转；三维里还能边转、边沿着转轴移动' },
      { at: 9.7, s: '像拧螺丝：每转一圈往里走一小段，上面的点走出螺旋线' },
      { at: 17.38, s: '任何刚体的速度 = 绕某根轴边转边移：这根轴叫**螺旋轴**' },
      { at: 26.16, s: '**螺旋轴**：方向、轴上一点、**螺距**（每转 1 弧度沿轴走多远）' },
      { at: 35.5, s: '**螺距**是 0：只转不移；只移不转：**螺距**看成无穷大' },
      { at: 41.55, s: '**v** 分两份：轴不过原点带出来的 + **角速度** × **螺距**' }
    ],
    build: function (K) {
      var s = K.sceneSvg('平面里刚体绕一个点转；三维里一个小方块绕竖直的转轴转，同时沿着轴往上走，走出螺旋线；就像拧螺丝，每转一圈往里走一小段；任何刚体的速度都能看成绕某根轴边转边移，这根轴叫螺旋轴，由方向、轴上一点和螺距（每转一弧度沿轴走多远）描述；螺距为零只转不移，只移不转时螺距看成无穷大；运动旋量的后三个数 v 分成两份：轴不过原点带出来的 −ω × q，和沿轴移动的 h ω');

      /* ══ ① 平面小图（俯视）：一块板绕黄点转 ══ */
      var flat = E.el('g', {}, s), FC = [215, 215];
      var fPlate = K.paint(E.el('path', { 'stroke-width': 2.5 }, flat), E.tint(E.BODY, 24, 'var(--demo-surface)'), E.BODY);
      var fTrail = K.paint(E.el('path', { fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '6 6' }, flat), null, E.VEL);
      var fCorner = K.paint(E.el('circle', { r: 6.5, 'stroke-width': 2.5 }, flat), E.VEL, E.S1);
      var fPivot = K.paint(E.el('circle', { cx: FC[0], cy: FC[1], r: 8, 'stroke-width': 2.5 }, flat), E.HI, E.S1);
      var fSpin = E.arcArrow(flat, FC[0], FC[1], 34, E.OMEGA, 3.5, { head: 12 });

      /* ══ ② 三维：转轴 + 方块 + 螺旋线 ══ */
      var solid = E.el('g', {}, s);
      var originG = E.el('g', {}, solid);                     // 句 6：世界原点 + 原点到 q 的虚线
      var oAx = [[0.45, 0, 0], [0, 0.45, 0], [0, 0, 0.45]].map(function (v, i) { var a = E.vec(originG, [E.X, E.Y, E.Z][i], 2.5, { head: 10 }), p0 = S3([0, 0, 0]), p1 = S3(v); a.set(p0[0], p0[1], p1[0], p1[1]); return a; });
      var oDot = S3([0, 0, 0]);
      K.paint(E.el('circle', { cx: oDot[0], cy: oDot[1], r: 5 }, originG), E.HI);
      E.halo(E.txt(originG, oDot[0] - 8, oDot[1] + 34, '原点', 20, E.HI, 'middle', 900));
      var oq = S3(Qw);
      var oLine = K.paint(E.el('path', { d: 'M ' + oDot[0] + ' ' + oDot[1] + ' L ' + oq[0].toFixed(1) + ' ' + oq[1].toFixed(1), fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '6 5' }, originG), null, E.HI);
      var axLo = S3([Qw[0], Qw[1], -0.3]), axHi = S3([Qw[0], Qw[1], 2.85]);
      // 按远近遮挡（同 006 的万向节）：远的半圈螺旋线、远的半个小椭圆画在转轴后面，近的画在前面；方块每帧按它和转轴谁近挪到轴前或轴后
      var trailBack = K.paint(E.el('path', { fill: 'none', 'stroke-width': 2.2, 'stroke-dasharray': '4 6' }, solid), null, E.VEL);
      var tArcBack = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3.5, 'stroke-linecap': 'round' }, solid), null, E.OMEGA);
      var axis = K.paint(E.el('path', { d: 'M ' + axLo[0].toFixed(1) + ' ' + axLo[1].toFixed(1) + ' L ' + axHi[0].toFixed(1) + ' ' + axHi[1].toFixed(1), fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '9 7' }, solid), null, E.WARN);
      var trailFront = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3 }, solid), null, E.VEL);
      trailBack.style.opacity = 0.55;
      var upArr = E.vec(solid, E.GOOD, 5, { head: 16 });     // 画在方块下面：方块转到前面时挡住它
      var cubeG = E.el('g', {}, solid);
      var faces = [0, 1, 2, 3, 4, 5].map(function () { return E.el('path', { 'stroke-width': 2, 'stroke-linejoin': 'round' }, cubeG); });
      var cDot = K.paint(E.el('circle', { r: 5 }, cubeG), E.VEL);
      var frontSlot = E.el('g', {}, solid);                    // 方块在转轴前面时放在这里
      var turnArc = E.el('g', {}, solid);                      // 「边转」：转轴上方一圈带箭头的椭圆；「边移」：沿轴往上的箭头
      var tArc = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3.5, 'stroke-linecap': 'round' }, turnArc), null, E.OMEGA);   // 近的半圈
      var tHead = K.paint(E.el('path', {}, turnArc), E.OMEGA);
      var nameLab = E.halo(E.txt(solid, axHi[0] - 12, axHi[1] + 6, '螺旋轴', 24, E.WARN, 'end', 900));   // 轴顶左边：方块最后停在轴的右上方
      // 句 4：方向 ŝ、轴上一点 q、螺距 h
      var sArr = E.vec(solid, E.WARN, 5.5, { head: 18 });
      var sLab = E.halo(E.txt(solid, 0, 0, '方向 ŝ', 22, E.WARN, 'start', 900));
      var qDot = K.paint(E.el('circle', { cx: oq[0], cy: oq[1], r: 7, 'stroke-width': 2.5 }, solid), E.HI, E.S1);
      var qLab = E.halo(E.txt(solid, oq[0] + 14, oq[1] + 22, '轴上一点 q', 20, E.HI, 'start', 900));
      var pitchG = E.el('g', {}, solid);
      var pA = K.paint(E.el('circle', { r: 6, 'stroke-width': 2.5 }, pitchG), E.GOOD, E.S1), pB = K.paint(E.el('circle', { r: 6, 'stroke-width': 2.5 }, pitchG), E.GOOD, E.S1);
      var pArc = K.paint(E.el('path', { fill: 'none', 'stroke-width': 4.5 }, pitchG), null, E.GOOD);
      var hBar = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3 }, pitchG), null, E.GOOD);
      var hGuide = K.paint(E.el('path', { fill: 'none', 'stroke-width': 1.8, 'stroke-dasharray': '4 4' }, pitchG), null, E.GOOD);
      var hLab = E.halo(E.txt(pitchG, 0, 0, '螺距 h', 22, E.GOOD, 'middle', 900));
      var oneRad = E.halo(E.txt(pitchG, 0, 0, '1 弧度', 20, E.GOOD, 'start', 900));

      /* ══ ③ 螺丝拧进木板 ══ */
      var scr = E.el('g', {}, s), SX = 230, SY = 300, SL = 118, SR = 0.2, HR = 0.46, HH = 0.2, LEN = 1.55, PITCH = 0.26;
      var sproj = E.view3(20, 22);
      function SP(v) { var q = sproj(v); return [SX + SL * q[0], SY - SL * q[1]]; }
      var board = E.el('g', {}, scr);
      var bw = 1.25, bd = 0.85, bt = 0.28;
      var bTop = K.paint(E.el('path', { d: E.poly([SP([-bw, -bd, 0]), SP([bw, -bd, 0]), SP([bw, bd, 0]), SP([-bw, bd, 0])], true), 'stroke-width': 2 }, board), '#6b5235', '#8a6b45');
      K.paint(E.el('path', { d: E.poly([SP([-bw, -bd, 0]), SP([bw, -bd, 0]), SP([bw, -bd, -bt]), SP([-bw, -bd, -bt])], true), 'stroke-width': 2 }, board), '#54402a', '#8a6b45');
      K.paint(E.el('path', { d: E.poly([SP([bw, -bd, 0]), SP([bw, bd, 0]), SP([bw, bd, -bt]), SP([bw, -bd, -bt])], true), 'stroke-width': 2 }, board), '#45341f', '#8a6b45');
      var holePts = []; for (var k = 0; k <= 36; k++) { var ha = 2 * Math.PI * k / 36; holePts.push(SP([SR * 1.05 * Math.cos(ha), SR * 1.05 * Math.sin(ha), 0])); }
      K.paint(E.el('path', { d: E.poly(holePts, true) }, board), '#1d150d');
      // 板面以上才看得见：裁剪区域 = 洞口椭圆靠近镜头的那半圈以上
      var clipId = 'ep009-screw-clip';
      var defs = s.querySelector('defs') || E.el('defs', {}, s);
      var clip = E.el('clipPath', { id: clipId }, defs);
      var near = holePts.filter(function (p, i) { var a = 2 * Math.PI * i / 36; return sproj.depth([Math.cos(a), Math.sin(a), 0]) <= 0; }).sort(function (a, b) { return b[0] - a[0]; });
      E.el('path', { d: 'M ' + (near[0][0] + 400) + ' ' + near[0][1].toFixed(1) + ' L ' + near.map(function (p) { return p[0].toFixed(1) + ' ' + p[1].toFixed(1); }).join(' L ') +
        ' L ' + (near[near.length - 1][0] - 400) + ' ' + near[near.length - 1][1].toFixed(1) + ' L ' + (near[near.length - 1][0] - 400) + ' -600 L ' + (near[0][0] + 400) + ' -600 Z' }, clip);
      var screwBody = E.el('g', { 'clip-path': 'url(#' + clipId + ')' }, scr);
      var shaft = K.paint(E.el('path', { 'stroke-width': 2 }, screwBody), '#9aa0a8', '#c8ccd2');
      var threads = K.paint(E.el('path', { fill: 'none', 'stroke-width': 2.2 }, screwBody), null, '#5f656d');
      var headSide = K.paint(E.el('path', { 'stroke-width': 2 }, screwBody), '#8d939b', '#c8ccd2');
      var headTop = K.paint(E.el('path', { 'stroke-width': 2 }, screwBody), '#b9bec5', '#dfe2e6');
      var slot = K.paint(E.el('path', { fill: 'none', 'stroke-width': 6, 'stroke-linecap': 'round' }, screwBody), null, '#4a4f56');
      var sTrail = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3 }, scr), null, E.VEL);
      var sDot = K.paint(E.el('circle', { r: 6.5, 'stroke-width': 2.5 }, scr), E.VEL, E.S1);
      var downArr = E.vec(scr, E.GOOD, 5, { head: 15 });
      var downLab = E.halo(E.txt(scr, 0, 0, '往里走', 22, E.GOOD, 'start', 900));

      /* ══ 右栏 ══ */
      var Rg = E.el('g', {}, s);
      function rtxt(g, y, str, size, color, w) { var n = E.txt(g, RX + 10, y, str, size || 23, color || E.INK, 'start', w || 700); n.style.whiteSpace = 'pre'; return n; }
      var r1 = E.el('g', {}, Rg);
      var a1 = rtxt(r1, 70, '平面：绕一个点转', 24, E.INK, 900);
      var a2 = rtxt(r1, 120, '三维：还能', 24, E.INK, 900);
      var a3 = rtxt(r1, 160, '边转，边沿着转轴移动', 24, E.GOOD, 900);
      var a4 = rtxt(r1, 230, '拧螺丝：', 24, E.INK, 900);
      var a5 = rtxt(r1, 270, '每转一圈，往里走一小段', 22, E.INK2);
      var a6 = rtxt(r1, 310, '上面的点：螺旋线', 22, E.VEL, 900);
      var r2 = E.el('g', {}, Rg);
      var thm = E.el('g', {}, r2);
      K.paint(E.el('rect', { x: RX, y: 24, width: 340, height: 104, rx: 14, 'stroke-width': 2.5 }, thm), E.tint(E.WARN, 10, 'var(--demo-surface)'), E.WARN);
      rtxt(thm, 60, '定理：任何刚体的速度', 23, E.INK, 900);
      var thm2 = rtxt(thm, 104, '都能看成绕某根轴边转边移', 23, E.WARN, 900);
      var b0 = rtxt(r2, 168, '螺旋轴要三样：', 23, E.INK2, 700);
      var b1 = rtxt(r2, 208, '① 方向 ŝ', 23, E.WARN, 900);
      var b2 = rtxt(r2, 246, '② 轴上一点 q', 23, E.HI, 900);
      var b3 = rtxt(r2, 284, '③ 螺距 h：每转 1 弧度沿轴走多远', 21, E.GOOD, 900);
      var cI = E.el('g', {}, r2);                              // h = 0 / h = ∞ 两个小图标
      var ic0 = E.el('g', {}, cI), ic1 = E.el('g', {}, cI);
      var i0 = E.arcArrow(ic0, RX + 26, 340, 15, E.OMEGA, 3, { head: 9 }); i0.set(100 * D, 420 * D);
      E.txt(ic0, RX + 52, 348, 'h = 0：只转不移', 22, E.INK, 'start', 900);
      var i1 = E.vec(ic1, E.GOOD, 3.5, { head: 11 }); i1.set(RX + 10, 380, RX + 42, 380);
      E.txt(ic1, RX + 52, 388, 'h = ∞：只移不转', 22, E.INK, 'start', 900);
      var r3 = E.el('g', {}, Rg);
      var fV = E.tex(r3, RX + 10, 66, '\\mathcal{V}=\\begin{bmatrix}\\textcolor{' + H.acc + '}{\\omega}\\\\ \\textcolor{' + H.violet + '}{v}\\end{bmatrix}', 30, E.INK, 'start', { w: 200, h: 90 });
      var fTwo = E.txt(r3, RX + 150, 70, 'v 分成两份：', 22, E.INK2, 'start', 700);
      var fEq = E.tex(r3, RX + 10, 170, '\\textcolor{' + H.violet + '}{v}=', 32, E.INK, 'start', { w: 60, h: 50 });
      var fP1 = E.tex(r3, RX + 64, 170, '-\\omega\\times q', 32, E.VEL, 'start', { w: 150, h: 50 });
      var fPlus = E.tex(r3, RX + 194, 170, '+', 32, E.INK, 'start', { w: 30, h: 50 });
      var fP2 = E.tex(r3, RX + 228, 170, 'h\\,\\omega', 32, E.GOOD, 'start', { w: 80, h: 50 });
      var br1 = K.paint(E.el('path', { d: 'M ' + (RX + 66) + ' 186 Q ' + (RX + 66) + ' 196 ' + (RX + 76) + ' 196 H ' + (RX + 118) + ' Q ' + (RX + 126) + ' 196 ' + (RX + 128) + ' 204 Q ' + (RX + 130) + ' 196 ' + (RX + 138) + ' 196 H ' + (RX + 180) + ' Q ' + (RX + 190) + ' 196 ' + (RX + 190) + ' 186', fill: 'none', 'stroke-width': 2.5 }, r3), null, E.VEL);
      var br2 = K.paint(E.el('path', { d: 'M ' + (RX + 230) + ' 186 Q ' + (RX + 230) + ' 196 ' + (RX + 240) + ' 196 H ' + (RX + 252) + ' Q ' + (RX + 258) + ' 196 ' + (RX + 260) + ' 204 Q ' + (RX + 262) + ' 196 ' + (RX + 268) + ' 196 H ' + (RX + 280) + ' Q ' + (RX + 290) + ' 196 ' + (RX + 290) + ' 186', fill: 'none', 'stroke-width': 2.5 }, r3), null, E.GOOD);
      var l1 = rtxt(r3, 250, '① 轴不过原点带出来的', 22, E.VEL, 900);
      var l1b = rtxt(r3, 286, '　（就是那张透明板）', 21, E.INK2);
      var l2 = rtxt(r3, 336, '② 沿轴移动：角速度 × 螺距', 22, E.GOOD, 900);

      /* ── 方块：中心、朝向随角度 θ；六个面按深度排序，背面不画 ── */
      var CUBE_C = [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]];
      var FACES = [[0, 1, 2, 3, [0, 0, -1]], [4, 5, 6, 7, [0, 0, 1]], [0, 1, 5, 4, [0, -1, 0]], [2, 3, 7, 6, [0, 1, 0]], [1, 2, 6, 5, [1, 0, 0]], [0, 3, 7, 4, [-1, 0, 0]]];
      var SHADE = { '0,0,1': 34, '0,0,-1': 14, '1,0,0': 26, '-1,0,0': 20, '0,1,0': 18, '0,-1,0': 30 };
      function center(th) { var a = A0 + th; return [Qw[0] + R * Math.cos(a), Qw[1] + R * Math.sin(a), 0.32 + HP * th]; }   // 起点抬高一点，方块底面不出画面
      function drawCube(th) {
        var M = E.rot3([0, 0, 1], th), c = center(th);
        var pts = CUBE_C.map(function (v) { var w = E.mv3(M, [v[0] * CUBE, v[1] * CUBE, v[2] * CUBE]); return [c[0] + w[0], c[1] + w[1], c[2] + w[2]]; });
        var list = FACES.map(function (fc, i) {
          var n = E.mv3(M, fc[4]);
          var mid = [0, 1, 2, 3].reduce(function (acc, k) { var p = pts[fc[k]]; return [acc[0] + p[0] / 4, acc[1] + p[1] / 4, acc[2] + p[2] / 4]; }, [0, 0, 0]);
          return { i: i, vis: proj.depth(n) < 0, z: proj.depth(mid), d: E.poly([0, 1, 2, 3].map(function (k) { return S3(pts[fc[k]]); }), true), shade: SHADE[fc[4].join(',')] };
        }).sort(function (a, b) { return b.z - a.z; });
        list.forEach(function (fc, k) {
          var el = faces[k];
          el.setAttribute('d', fc.vis ? fc.d : '');
          K.paint(el, E.tint(E.BODY, fc.shade + 10, 'var(--demo-surface)'), E.BODY);
        });
        var cp = S3(c); cDot.setAttribute('cx', cp[0].toFixed(1)); cDot.setAttribute('cy', cp[1].toFixed(1));
      }
      // 螺旋线（方块中心走过的路）：靠近镜头的半边实线，远的半边虚线
      function trail(th) {
        var front = '', back = '', n = Math.max(2, Math.ceil(th / 0.06)), prevF = null;
        for (var k = 0; k <= n; k++) {
          var a = th * k / n, c = center(a), p = S3(c), isF = proj.depth([Math.cos(A0 + a), Math.sin(A0 + a), 0]) < 0;
          var seg = p[0].toFixed(1) + ' ' + p[1].toFixed(1);
          if (isF) { front += (prevF === true ? ' L ' : ' M ') + seg; } else { back += (prevF === false ? ' L ' : ' M ') + seg; }
          if (prevF !== null && prevF !== isF) { if (isF) back += ' L ' + seg; else front += ' L ' + seg; }
          prevF = isF;
        }
        trailFront.setAttribute('d', front); trailBack.setAttribute('d', back);
      }

      return {
        el: s,
        draw: function (st, t) {
          var f = E.fade, b = E.bump;

          /* ── ① 平面 ── */
          var fOn = f(st, T.plane - 0.4, 0.4) * (1 - f(st, T.to3, 0.5));
          E.op(flat, fOn);
          var fa = 1.1 * t, rect = [[0.4, -0.55], [2.0, -0.55], [2.0, 0.55], [0.4, 0.55]];
          var FU = 62;
          function fp(p) { var c = Math.cos(fa), sn = Math.sin(fa); return [FC[0] + FU * (c * p[0] - sn * p[1]), FC[1] - FU * (sn * p[0] + c * p[1])]; }
          fPlate.setAttribute('d', E.poly(rect.map(fp), true));
          var cr = fp(rect[2]); fCorner.setAttribute('cx', cr[0].toFixed(1)); fCorner.setAttribute('cy', cr[1].toFixed(1));
          var rr = Math.hypot(2.0, 0.55) * FU;
          fTrail.setAttribute('d', 'M ' + (FC[0] + rr) + ' ' + FC[1] + ' A ' + rr + ' ' + rr + ' 0 1 0 ' + (FC[0] - rr) + ' ' + FC[1] + ' A ' + rr + ' ' + rr + ' 0 1 0 ' + (FC[0] + rr) + ' ' + FC[1]);
          fSpin.set(fa + 2.4, fa + 3.6);

          /* ── ② 三维方块（句 1 结尾、句 3–6） ── */
          var on3 = Math.min(f(st, T.to3 + 0.3, 0.5) * (1 - f(st, T.screw, 0.4)) + f(st, T.block2, 0.5), 1);
          E.op(solid, on3);
          var th = theta(st);
          drawCube(th); trail(Math.max(th, 0.001));
          var cc = center(th);
          if (proj.depth(cc) > proj.depth([Qw[0], Qw[1], cc[2]])) { if (cubeG.nextSibling !== axis) solid.insertBefore(cubeG, axis); }
          else if (cubeG.parentNode !== frontSlot) frontSlot.appendChild(cubeG);
          E.op(trailFront, th > 0.01 ? 1 : 0); E.op(trailBack, th > 0.01 ? 0.55 : 0);
          // 「边转」「边移」：句 1 时出现，句 3 后半收掉
          var cTop = S3([Qw[0], Qw[1], 2.45]), rx = 0.5;
          var ring = []; for (var k = 0; k <= 30; k++) { var aa = (-60 + 280 * k / 30) * D; ring.push(S3([Qw[0] + rx * Math.cos(aa), Qw[1] + rx * Math.sin(aa), 2.45])); }
          // 小椭圆拆成远、近两段（近的半圈压在转轴上，远的半圈被转轴挡住）
          var near = '', far = '', prevN = null;
          ring.forEach(function (p, k) {
            var aa = (-60 + 280 * k / 30) * D, isN = proj.depth([Math.cos(aa), Math.sin(aa), 0]) < 0, sg = p[0].toFixed(1) + ' ' + p[1].toFixed(1);
            if (isN) near += (prevN === true ? ' L ' : ' M ') + sg; else far += (prevN === false ? ' L ' : ' M ') + sg;
            if (prevN !== null && prevN !== isN) { if (isN) far += ' L ' + sg; else near += ' L ' + sg; }
            prevN = isN;
          });
          tArc.setAttribute('d', near); tArcBack.setAttribute('d', far);
          var e1 = ring[ring.length - 1], e0 = ring[ring.length - 3], dx = e1[0] - e0[0], dy = e1[1] - e0[1], dl = Math.hypot(dx, dy) || 1;
          dx /= dl; dy /= dl;
          tHead.setAttribute('d', 'M ' + (e1[0] + dx * 12).toFixed(1) + ' ' + (e1[1] + dy * 12).toFixed(1) + ' L ' + (e1[0] - dy * 7).toFixed(1) + ' ' + (e1[1] + dx * 7).toFixed(1) + ' L ' + (e1[0] + dy * 7).toFixed(1) + ' ' + (e1[1] - dx * 7).toFixed(1) + ' Z');
          var spinOn = Math.max(f(st, T.turn + 0.2, 0.3) * (1 - f(st, T.screw, 0.3)), f(st, T.rot, 0.3) * (1 - f(st, T.name - 0.5, 0.4)));   // 「螺旋轴」三个字出来前收掉，免得压字
          E.op(turnArc, spinOn); E.op(tArcBack, spinOn);
          var upOn = Math.max(f(st, T.rise, 0.4) * (1 - f(st, T.screw, 0.3)), f(st, T.mov, 0.3) * (1 - f(st, T.stop, 0.4)));
          var u0 = S3([Qw[0], Qw[1], 0.55]), u1 = S3([Qw[0], Qw[1], 0.55 + 1.0 * upOn]);   // 贴着转轴、往上：沿着轴移动
          upArr.set(u0[0] + 22, u0[1], u1[0] + 22, u1[1]); E.op(upArr.g, upOn > 0.01 ? 1 : 0);
          E.op(nameLab, f(st, T.name, 0.3));
          var nb = b(st, T.name, 0.9);
          axis.style.strokeWidth = 3 + 2.5 * nb; E.glow(axis, E.WARN, nb > 0.05);
          // 句 4：方向、点、螺距
          var sp = f(st, T.dirS, 0.5), s0 = S3([Qw[0], Qw[1], 1.0]), s1 = S3([Qw[0], Qw[1], 1.0 + 0.9 * sp]);
          sArr.set(s0[0], s0[1], s1[0], s1[1]); E.op(sArr.g, sp > 0.01 ? 1 : 0);
          sLab.setAttribute('x', (s1[0] - 96).toFixed(1)); sLab.setAttribute('y', (s1[1] + 8).toFixed(1)); E.op(sLab, f(st, T.dirS + 0.2, 0.3));
          E.op(qDot, f(st, T.pointQ, 0.3)); E.op(qLab, f(st, T.pointQ, 0.3));
          qDot.setAttribute('r', (7 + 4 * b(st, T.pointQ, 0.7)).toFixed(2));
          var pa = S3(center(TH_A)), pb = S3(center(TH_A + 1));
          pA.setAttribute('cx', pa[0].toFixed(1)); pA.setAttribute('cy', pa[1].toFixed(1)); pB.setAttribute('cx', pb[0].toFixed(1)); pB.setAttribute('cy', pb[1].toFixed(1));
          var arcPts = []; for (k = 0; k <= 20; k++) arcPts.push(S3(center(TH_A + k / 20)));
          pArc.setAttribute('d', E.poly(arcPts));
          E.drawOn(pArc, f(st, T.perRad, 0.6));
          var hx = Math.max(pa[0], pb[0]) + 40;
          hBar.setAttribute('d', 'M ' + (hx - 8) + ' ' + pa[1].toFixed(1) + ' H ' + (hx + 8) + ' M ' + hx + ' ' + pa[1].toFixed(1) + ' V ' + pb[1].toFixed(1) + ' M ' + (hx - 8) + ' ' + pb[1].toFixed(1) + ' H ' + (hx + 8));
          hGuide.setAttribute('d', 'M ' + pa[0].toFixed(1) + ' ' + pa[1].toFixed(1) + ' H ' + (hx - 8) + ' M ' + pb[0].toFixed(1) + ' ' + pb[1].toFixed(1) + ' H ' + (hx - 8));
          hLab.setAttribute('x', hx); hLab.setAttribute('y', (pb[1] - 14).toFixed(1));   // 写在量尺上方：右边紧挨着右栏的公式
          oneRad.setAttribute('x', (pa[0] + 12).toFixed(1)); oneRad.setAttribute('y', (pa[1] + 26).toFixed(1));   // 写在下面那个点的右下方，不压转轴
          E.op(pitchG, f(st, T.pitch, 0.3));
          E.op(pA, f(st, T.pitch, 0.3)); E.op(pB, f(st, T.perRad + 0.5, 0.3)); E.op(oneRad, f(st, T.perRad + 0.3, 0.3));
          var hp = f(st, T.far, 0.4), hb = b(st, T.hPulse, 0.9);
          E.op(hBar, hp); E.op(hGuide, hp); E.op(hLab, hp);
          hBar.style.strokeWidth = 3 + 3 * hb; E.glow(hBar, E.GOOD, hb > 0.05);
          // 句 6：原点、原点到 q
          E.op(originG, f(st, T.origin, 0.4));
          var ob = b(st, T.plate, 0.9);
          oLine.style.strokeWidth = 2.5 + 2.5 * ob; E.glow(oLine, E.HI, ob > 0.05);

          /* ── ③ 螺丝 ── */
          var scOn = f(st, T.screw, 0.4) * (1 - f(st, T.out4, 0.4));
          E.op(scr, scOn);
          var sa = 2 * Math.PI * f(st, T.screwTurn[0], T.screwTurn[1] - T.screwTurn[0]) + 2.6 * Math.PI * K.seg(st, T.more[0], T.more[1]);
          var sink = PITCH * sa / (2 * Math.PI), zb = -0.02 - sink, zt = zb + LEN;   // 螺丝尖起初刚碰到板面
          // 螺杆（轮廓：左右两条竖线）
          var wx = SR * SL;                                      // 竖直圆柱在画面上的半宽（正交投影，竖直方向不缩）
          var cb = SP([0, 0, zb]), ct = SP([0, 0, zt]);
          shaft.setAttribute('d', 'M ' + (cb[0] - wx).toFixed(1) + ' ' + cb[1].toFixed(1) + ' L ' + (ct[0] - wx).toFixed(1) + ' ' + ct[1].toFixed(1) + ' L ' + (ct[0] + wx).toFixed(1) + ' ' + ct[1].toFixed(1) +
            ' L ' + (cb[0] + wx).toFixed(1) + ' ' + cb[1].toFixed(1) + ' L ' + cb[0].toFixed(1) + ' ' + (cb[1] + 14).toFixed(1) + ' Z');
          // 螺纹：杆表面的螺旋线，只画靠近镜头的半圈；跟着转角 sa 往下走（右旋：顺时针拧进去，从上往下看是顺时针，角度取负）
          var td = '', was = false;
          for (k = 0; k <= 360; k++) {
            var ph = k / 360 * (LEN - 0.15) / PITCH * 2 * Math.PI, z = zb + 0.08 + PITCH * ph / (2 * Math.PI), ang = ph - sa;
            var v3 = [SR * Math.cos(ang), SR * Math.sin(ang), z], vis = sproj.depth([Math.cos(ang), Math.sin(ang), 0]) < 0 && z < zt - 0.04;
            if (vis) { var q = SP(v3); td += (was ? ' L ' : ' M ') + q[0].toFixed(1) + ' ' + q[1].toFixed(1); }
            was = vis;
          }
          threads.setAttribute('d', td);
          // 螺丝头：侧面 + 顶面椭圆 + 槽
          var top = [], bot = [];
          for (k = 0; k <= 36; k++) { var hb2 = 2 * Math.PI * k / 36; top.push(SP([HR * Math.cos(hb2), HR * Math.sin(hb2), zt + HH])); bot.push(SP([HR * Math.cos(hb2), HR * Math.sin(hb2), zt])); }
          var lx = top.reduce(function (m, p) { return p[0] < m[0] ? p : m; }), rxp = top.reduce(function (m, p) { return p[0] > m[0] ? p : m; });
          var lb = bot.reduce(function (m, p) { return p[0] < m[0] ? p : m; }), rb = bot.reduce(function (m, p) { return p[0] > m[0] ? p : m; });
          var botNear = bot.filter(function (p, i) { var a = 2 * Math.PI * i / 36; return sproj.depth([Math.cos(a), Math.sin(a), 0]) <= 0; }).sort(function (a, b) { return a[0] - b[0]; });
          headSide.setAttribute('d', 'M ' + lx[0].toFixed(1) + ' ' + lx[1].toFixed(1) + ' L ' + botNear.map(function (p) { return p[0].toFixed(1) + ' ' + p[1].toFixed(1); }).join(' L ') + ' L ' + rxp[0].toFixed(1) + ' ' + rxp[1].toFixed(1) + ' Z');
          headTop.setAttribute('d', E.poly(top, true));
          var s1a = SP([HR * 0.8 * Math.cos(-sa), HR * 0.8 * Math.sin(-sa), zt + HH]), s2a = SP([-HR * 0.8 * Math.cos(-sa), -HR * 0.8 * Math.sin(-sa), zt + HH]);
          slot.setAttribute('d', 'M ' + s1a[0].toFixed(1) + ' ' + s1a[1].toFixed(1) + ' L ' + s2a[0].toFixed(1) + ' ' + s2a[1].toFixed(1));
          // 螺丝头边上的一点和它走过的螺旋线（从 13.42 起描）
          var a0s = -60 * D;
          function rim(a) { var z = -0.02 - PITCH * a / (2 * Math.PI) + LEN + HH; return SP([HR * Math.cos(a0s - a), HR * Math.sin(a0s - a), z]); }
          var dp = rim(sa); sDot.setAttribute('cx', dp[0].toFixed(1)); sDot.setAttribute('cy', dp[1].toFixed(1));
          E.op(sDot, f(st, T.head, 0.3)); sDot.setAttribute('r', (6.5 + 3 * b(st, T.head, 0.7)).toFixed(2));
          var tr = [], a0t = 2 * Math.PI; for (k = 0; k <= 80; k++) { var aa2 = a0t + (sa - a0t) * k / 80; if (sa > a0t) tr.push(rim(aa2)); }
          sTrail.setAttribute('d', tr.length > 1 ? E.poly(tr) : ''); E.op(sTrail, f(st, T.helix - 1.6, 0.3));
          var dn = f(st, T.screwTurn[0] + 1.2, 0.4);
          var d0 = SP([HR + 0.35, 0, LEN - 0.1]), d1 = SP([HR + 0.35, 0, LEN - 0.1 - 0.55 * dn]);
          downArr.set(d0[0], d0[1], d1[0], d1[1]); E.op(downArr.g, dn > 0.01 ? 1 : 0);
          downLab.setAttribute('x', (d0[0] + 14).toFixed(1)); downLab.setAttribute('y', ((d0[1] + d1[1]) / 2 + 8).toFixed(1)); E.op(downLab, dn);

          /* ── 右栏 ── */
          E.op(r1, 1 - f(st, T.out4, 0.4));
          E.op(a1, f(st, T.plane, 0.3)); E.op(a2, f(st, T.to3, 0.3)); E.op(a3, f(st, T.turn, 0.3));
          E.op(a4, f(st, T.screw + 0.3, 0.3)); E.op(a5, f(st, T.screwTurn[0], 0.3)); E.op(a6, f(st, T.helix, 0.3));
          E.op(r2, f(st, T.thm - 0.2, 0.4) * (1 - f(st, T.form - 0.1, 0.4)));
          E.pop(thm, RX + 170, 76, f(st, T.thm, 0.35), 0.92); E.op(thm2, f(st, T.any + 1.3, 0.3));
          E.op(b0, f(st, T.dirS - 1.4, 0.3)); E.op(b1, f(st, T.dirS, 0.3)); E.op(b2, f(st, T.pointQ, 0.3)); E.op(b3, f(st, T.pitch, 0.3));
          E.op(ic0, f(st, T.h0, 0.3)); E.op(ic1, f(st, T.hInf - 0.2, 0.3));
          E.op(r3, f(st, T.form, 0.4));
          E.op(fV, 1); E.op(fTwo, f(st, T.two, 0.3));
          E.op(fEq, f(st, T.two, 0.3)); E.op(fP1, f(st, T.part1, 0.3)); E.op(br1, f(st, T.origin, 0.3)); E.op(l1, f(st, T.origin, 0.3)); E.op(l1b, f(st, T.plate, 0.3));
          E.op(fPlus, f(st, T.part2, 0.3)); E.op(fP2, f(st, T.part2, 0.3)); E.op(br2, f(st, T.hPulse, 0.3)); E.op(l2, f(st, T.times, 0.3));
        }
      };
    }
  };
})();
