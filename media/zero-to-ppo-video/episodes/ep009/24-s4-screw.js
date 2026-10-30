// 009 第 4 幕：螺旋运动——边转边移。左边是斜着看的三维小场景（z 朝上），右栏放要点。
// 句 1「平面」左边先放一张俯视的平面小图：一块板绕一个黄点转，角上一点描出圆；「三维」换成三维：一根竖直的转轴（琥珀虚线）、离轴一段的小方块，
//    「边转」方块绕轴转起来，「沿着转轴移动」同时往上走，方块中心描出螺旋线（右栏两行）；
// 句 2「拧螺丝」换成一颗螺丝拧进木板：螺丝头上的槽跟着转，「每转一圈」「往里走一小段」螺纹往下走、整颗往板里沉；
//    「上面的点」螺丝头边上一点亮起来，「螺旋线」它走过的路描成螺旋线；
// 句 3 换回方块：「定理」右栏出定理卡片，方块接着沿螺旋线走，「螺旋轴」轴上写「螺旋轴」；
// 句 4 方块停住：「方向」轴上画出方向箭头 ŝ，「一个点」轴上点出 q，「螺距」「每转一弧度」螺旋线上取相隔 1 弧度的两点，
//    「走多远」量出两点的高度差、标 h（右栏三行依次写出）；
// 句 5「螺距是零」右栏画一个圆圈图标「h = 0：只转不移」，「只移不转」画一支直箭头「h = ∞：只移不转」；
// 句 6–10 v 的两份，边讲边画（初版只在右栏写 v = −ω × q + h ω，用户反馈「矩阵运算还是展开细节步骤，旋转和平移等」）：
//    句 6「这时」方块、螺旋线、量尺淡出，镜头绕竖直方向转到另一个角度（转轴、q 留着），右栏 𝒱 = (ω, v)、「v 分成两份」；
//    句 7「第一份来自转动」右栏「① 转动」，「透明板」转轴上套一张圆形的透明板（淡黄，带辐条），「原点」原点亮一圈，
//       「跟着板」「绕着螺旋轴转动」板绕轴转、原点那一点的路（紫虚线圆）画出来，转轴上方蓝色小椭圆标出转的方向；
//    句 8（同第 3 幕算 v）「从轴上那一点指到原点」q → 原点画黄箭头，「顺着转动的方向」箭头平移到原点（尾巴在原点），
//       「转九十度」在板面里绕原点转 90°（蓝弧），「乘以角速度」拉长，标「−ω × q」，右栏写出 −ω × q；
//    句 9「第二份来自移动」右栏「② 移动」，「沿着轴走」转轴旁长出往上的绿箭头，「沿着轴」「角速度乘螺距」第一份的尖上往上接一支绿箭头，标「h ω」；
//    句 10「两份加起来」原点到绿箭头尖画出紫色的 v，右栏框出 v = −ω × q + h ω。
//    q 取在和原点一样高的地方：从 q 指到原点的箭头和转轴垂直，「转 90° 再乘」才成立（画面上标出）。
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
    form: 41.64, cam: 41.9, two: 44.1,
    part1: 46.1, turning: 46.79, plate: 49.29, origin: 50.05, around: [51.84, 53.3], same: 54.31,
    qArr: [55.64, 56.95], slide: [57.59, 58.35], rot: [58.83, 59.5], stretch: [59.91, 60.45], p1Lab: 60.22,
    part2: 61.94, along: 63.68, dir2: 67.58, grow: [68.74, 69.9], p2Lab: 69.71,
    sum: [71.24, 71.95], plus: 71.59, vLab: 72.59
  };
  var END = 73.66;
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
      { at: 41.55, s: '运动旋量的后三个数 **v** 分成两份' },
      { at: 46.01, s: '① 转动：把刚体想成透明板，原点那一点绕**螺旋轴**转' },
      { at: 54.22, s: '同第 3 幕算 **v**：q 指到原点，顺着转动的方向转 90°，再 × **角速度**' },
      { at: 61.85, s: '② 移动：沿着轴，大小 = **角速度** × **螺距**' },
      { at: 71.14, s: '两份加起来：**v** = −ω × q + h ω' }
    ],
    build: function (K) {
      var s = K.sceneSvg('平面里刚体绕一个点转；三维里一个小方块绕竖直的转轴转，同时沿着轴往上走，走出螺旋线；就像拧螺丝，每转一圈往里走一小段；任何刚体的速度都能看成绕某根轴边转边移，这根轴叫螺旋轴，由方向、轴上一点和螺距（每转一弧度沿轴走多远）描述；螺距为零只转不移，只移不转时螺距看成无穷大；运动旋量的后三个数 v 分成两份：转动带出来的一份，是从轴上一点 q 指到原点的箭头顺着转动方向转 90° 再乘角速度，写成 −ω × q；移动带出来的一份沿着轴，大小是角速度乘螺距，写成 h ω；两份加起来就是 v');

      /* ══ ① 平面小图（俯视）：一块板绕黄点转 ══ */
      var flat = E.el('g', {}, s), FC = [215, 215];
      var fPlate = K.paint(E.el('path', { 'stroke-width': 2.5 }, flat), E.tint(E.BODY, 24, 'var(--demo-surface)'), E.BODY);
      var fTrail = K.paint(E.el('path', { fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '6 6' }, flat), null, E.VEL);
      var fCorner = K.paint(E.el('circle', { r: 6.5, 'stroke-width': 2.5 }, flat), E.VEL, E.S1);
      var fPivot = K.paint(E.el('circle', { cx: FC[0], cy: FC[1], r: 8, 'stroke-width': 2.5 }, flat), E.HI, E.S1);
      var fSpin = E.arcArrow(flat, FC[0], FC[1], 34, E.OMEGA, 3.5, { head: 12 });

      /* ══ ② 三维：转轴 + 方块 + 螺旋线 ══ */
      var solid = E.el('g', {}, s);
      var oq = S3(Qw);
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


      /* ══ ④ v 的两份（句 6–10）：镜头绕竖直方向转到另一个角度，原点挪到左边中间；转轴上套一张圆形的透明板 ══ */
      var dec = E.el('g', {}, s);
      var CAM0 = { az: 28, el: 18, ox: WX, oy: WY, l: L, z0: -0.3, z1: 2.85 }, CAM1 = { az: -20, el: 32, ox: 163, oy: 290, l: 104, z0: -0.4, z1: 1.75 };
      var RD = 1.6, KW = 1.25, HW = 0.9;                      // 板的半径；第一份拉长的倍数（角速度，示意）；第二份的长度（角速度 × 螺距，示意）
      var RQ = Math.hypot(Qw[0], Qw[1]), AO = Math.atan2(-Qw[1], -Qw[0]), MQ = [-Qw[0], -Qw[1]];   // 原点离轴多远、在轴的哪个方向；从 q 指到原点
      var dAxLo = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '9 7' }, dec), null, E.WARN);   // 板下面那段转轴：被透明板盖住
      var diskG = E.el('g', {}, dec);
      var disk = K.paint(E.el('path', { 'stroke-width': 2.5, 'stroke-dasharray': '8 6' }, diskG), E.tint(E.BODY, 14, 'var(--demo-surface)'), E.BODY);
      var spokeG = E.el('g', { 'data-grid': '1' }, diskG);     // 辐条同格线：查「字压线」时不算
      var spokes = K.paint(E.el('path', { fill: 'none', 'stroke-width': 1.5 }, spokeG), null, E.BODY);
      spokes.style.opacity = 0.5;
      var diskLab = E.halo(E.txt(dec, 0, 0, '透明板', 20, E.BODY, 'middle', 900));
      var dPath = K.paint(E.el('path', { fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '5 6' }, dec), null, E.VEL);
      var dPathHead = K.paint(E.el('path', {}, dec), E.VEL);
      var dSpinFar = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3, 'stroke-linecap': 'round' }, dec), null, E.OMEGA);
      var dAxHi = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '9 7' }, dec), null, E.WARN);
      var dSpin = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3, 'stroke-linecap': 'round' }, dec), null, E.OMEGA);
      var dSpinHead = K.paint(E.el('path', {}, dec), E.OMEGA);
      var upA = E.vec(dec, E.GOOD, 4.5, { head: 14 });
      var dQ = K.paint(E.el('circle', { r: 7, 'stroke-width': 2.5 }, dec), E.HI, E.S1);
      var dQLab = E.halo(E.txt(dec, 0, 0, 'q', 22, E.HI, 'end', 900));
      var dORing = K.paint(E.el('circle', { r: 12, fill: 'none', 'stroke-width': 3 }, dec), null, E.HI);
      var dO = K.paint(E.el('circle', { r: 7, 'stroke-width': 2 }, dec), E.VEL, E.S1);   // 板上此刻在原点的那一点（紫，同第 3 幕）
      var dOLab = E.halo(E.txt(dec, 0, 0, '原点', 20, E.HI, 'end', 900));
      var rArc = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3 }, dec), null, E.OMEGA);
      var rArcHead = K.paint(E.el('path', {}, dec), E.OMEGA);
      var rLab = E.halo(E.txt(dec, 0, 0, '90°', 20, E.OMEGA, 'middle', 900));
      var dA1 = E.vec(dec, E.HI, 4.5, { head: 15 });
      var p1Lab = E.halo(E.txt(dec, 0, 0, '−ω × q', 22, E.HI, 'start', 900));
      var dA2 = E.vec(dec, E.GOOD, 4.5, { head: 15 });
      var p2Lab = E.halo(E.txt(dec, 0, 0, 'h ω', 22, E.GOOD, 'start', 900));
      var dAV = E.vec(dec, E.VEL, 5.5, { head: 17 });
      var vLab2 = E.halo(E.txt(dec, 0, 0, 'v', 26, E.VEL, 'middle', 900));
      // 镜头：c = 0 时同上面的三维场景（转轴、q 和它重合，换场看不出跳），c = 1 时是这一段的角度
      function camAt(c) {
        var m = function (a, b2) { return a + (b2 - a) * c; };
        var pr = E.view3(m(CAM0.az, CAM1.az), m(CAM0.el, CAM1.el)), ox = m(CAM0.ox, CAM1.ox), oy = m(CAM0.oy, CAM1.oy), l = m(CAM0.l, CAM1.l);
        var P = function (v) { var q = pr(v); return [ox + l * q[0], oy - l * q[1]]; };
        P.depth = pr.depth; P.z0 = m(CAM0.z0, CAM1.z0); P.z1 = m(CAM0.z1, CAM1.z1);
        return P;
      }
      function ringPts(P, c, r, a0, aEnd, n) { var out = []; for (var k = 0; k <= n; k++) { var a = a0 + (aEnd - a0) * k / n; out.push(P([c[0] + r * Math.cos(a), c[1] + r * Math.sin(a), c[2]])); } return out; }
      function headAt(el, pts, size) {                         // 折线末端的箭头三角
        var e1 = pts[pts.length - 1], e0 = pts[pts.length - 2], dx = e1[0] - e0[0], dy = e1[1] - e0[1], dl = Math.hypot(dx, dy) || 1;
        dx /= dl; dy /= dl;
        el.setAttribute('d', 'M ' + (e1[0] + dx * size).toFixed(1) + ' ' + (e1[1] + dy * size).toFixed(1) + ' L ' + (e1[0] - dy * size * 0.6).toFixed(1) + ' ' + (e1[1] + dx * size * 0.6).toFixed(1) +
          ' L ' + (e1[0] + dy * size * 0.6).toFixed(1) + ' ' + (e1[1] - dx * size * 0.6).toFixed(1) + ' Z');
      }
      function seg(el, a, b2) { el.setAttribute('d', 'M ' + a[0].toFixed(1) + ' ' + a[1].toFixed(1) + ' L ' + b2[0].toFixed(1) + ' ' + b2[1].toFixed(1)); }

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
      var m1 = rtxt(r3, 152, '① 转动：', 23, E.HI, 900);
      var m1f = E.tex(r3, RX + 118, 152, '-\\textcolor{' + H.acc + '}{\\omega}\\times q', 28, E.HI, 'start', { w: 140, h: 46 });
      var m1a = rtxt(r3, 188, '从 q 指到原点，', 20, E.INK2);
      var m1b = rtxt(r3, 218, '顺着转动的方向转 90°，', 20, E.INK2);
      var m1c = rtxt(r3, 248, '再 × 角速度', 20, E.INK2);
      var m1d = rtxt(r3, 274, '（q 取和原点一样高：箭头和轴垂直）', 17, E.MUT);
      var m2 = rtxt(r3, 308, '② 移动：', 23, E.GOOD, 900);
      var m2f = E.tex(r3, RX + 118, 308, 'h\\,\\textcolor{' + H.acc + '}{\\omega}', 28, E.GOOD, 'start', { w: 80, h: 46 });
      var m2a = rtxt(r3, 340, '沿着轴，大小 = 角速度 × 螺距', 20, E.INK2);
      var sumG = E.el('g', {}, r3);
      K.paint(E.el('rect', { x: RX, y: 356, width: 340, height: 54, rx: 12, 'stroke-width': 2.5 }, sumG), E.tint(E.VEL, 10, 'var(--demo-surface)'), E.VEL);
      E.tex(sumG, RX + 170, 392, '\\textcolor{' + H.violet + '}{v}=\\textcolor{' + H.hi + '}{-\\textcolor{' + H.acc + '}{\\omega}\\times q}+\\textcolor{' + H.good + '}{h\\,\\textcolor{' + H.acc + '}{\\omega}}', 28, E.INK, 'middle', { w: 300, h: 50 });

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
          E.op(solid, on3 * (1 - f(st, T.form, 0.5)));
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


          /* ── ④ v 的两份 ── */
          var dOn = f(st, T.form, 0.5);
          E.op(dec, dOn);
          var Pc = camAt(f(st, T.cam, 1.4));
          var O = Pc([0, 0, 0]), qp = Pc(Qw), q0 = [Qw[0], Qw[1], 0];
          seg(dAxLo, Pc([Qw[0], Qw[1], Pc.z0]), qp); seg(dAxHi, qp, Pc([Qw[0], Qw[1], Pc.z1]));
          // 透明板：绕轴转（辐条跟着转），句 8 开头停住——算的是「这一刻」
          var diskOn = f(st, T.plate, 0.5), db = b(st, T.plate, 0.9);
          disk.setAttribute('d', E.poly(ringPts(Pc, q0, RD, 0, 2 * Math.PI, 60), true));
          var phi = 0.6 * E.clamp(st - T.plate, 0, T.same - T.plate), spd = '';
          for (k = 0; k < 6; k++) { var sE = Pc([Qw[0] + RD * Math.cos(phi + k * Math.PI / 3), Qw[1] + RD * Math.sin(phi + k * Math.PI / 3), 0]); spd += 'M ' + qp[0].toFixed(1) + ' ' + qp[1].toFixed(1) + ' L ' + sE[0].toFixed(1) + ' ' + sE[1].toFixed(1) + ' '; }
          spokes.setAttribute('d', spd);
          E.op(diskG, diskOn); disk.style.strokeWidth = 2.5 + 2 * db; E.glow(disk, E.BODY, db > 0.05);
          // 「透明板」写在板后沿的上方、转轴左边（板里面有紫色的路和辐条）
          diskLab.setAttribute('x', (qp[0] - 70).toFixed(1)); diskLab.setAttribute('y', (qp[1] - 100).toFixed(1)); E.op(diskLab, diskOn);
          // 原点那一点跟着板绕轴走的路（紫虚线圆），画到哪、箭头跟到哪
          var pg = f(st, T.around[0], T.around[1] - T.around[0]);
          if (pg > 0.01) { var pp = ringPts(Pc, q0, RQ, AO + 0.25, AO + 0.25 + (2 * Math.PI - 0.42) * pg, 64); dPath.setAttribute('d', E.poly(pp)); headAt(dPathHead, pp, 12); }
          E.op(dPath, pg > 0.01 ? 1 : 0); E.op(dPathHead, pg > 0.01 ? 1 : 0);
          // 转轴上方的蓝色小椭圆：转的方向（远的半圈压在轴后面）
          var zS = Pc.z1 - 0.38, spinPts = ringPts(Pc, [Qw[0], Qw[1], zS], 0.32, -60 * D, 220 * D, 40), nearD = '', farD = '', prevN = null;
          spinPts.forEach(function (pt, k2) {
            var aa = (-60 + 280 * k2 / 40) * D, isN = Pc.depth([Math.cos(aa), Math.sin(aa), 0]) < 0, sg = pt[0].toFixed(1) + ' ' + pt[1].toFixed(1);
            if (isN) nearD += (prevN === true ? ' L ' : ' M ') + sg; else farD += (prevN === false ? ' L ' : ' M ') + sg;
            if (prevN !== null && prevN !== isN) { if (isN) farD += ' L ' + sg; else nearD += ' L ' + sg; }
            prevN = isN;
          });
          dSpin.setAttribute('d', nearD); dSpinFar.setAttribute('d', farD); headAt(dSpinHead, spinPts, 11);
          var spOn = f(st, T.turning, 0.3);
          E.op(dSpin, spOn); E.op(dSpinFar, spOn); E.op(dSpinHead, spOn);
          // q、原点
          dQ.setAttribute('cx', qp[0].toFixed(1)); dQ.setAttribute('cy', qp[1].toFixed(1));
          dQLab.setAttribute('x', (qp[0] - 14).toFixed(1)); dQLab.setAttribute('y', (qp[1] - 6).toFixed(1));   // 写在 q 左边：右边是沿轴往上的绿箭头
          var oOn = f(st, T.origin, 0.3), ob2 = Math.max(b(st, T.origin, 0.8), b(st, T.around[0], 0.8));
          dO.setAttribute('cx', O[0].toFixed(1)); dO.setAttribute('cy', O[1].toFixed(1)); E.op(dO, oOn);
          dORing.setAttribute('cx', O[0].toFixed(1)); dORing.setAttribute('cy', O[1].toFixed(1)); dORing.setAttribute('r', (12 + 5 * ob2).toFixed(2)); E.op(dORing, oOn);
          // 「原点」写在左边、板的边外面：左下是平移过来的箭头和 90° 的蓝弧
          dOLab.setAttribute('x', (O[0] - 64).toFixed(1)); dOLab.setAttribute('y', (O[1] + 6).toFixed(1)); E.op(dOLab, oOn);
          // 句 8：q → 原点的黄箭头；平移到原点；在板面里绕原点转 90°；拉长（乘角速度）
          var g0 = f(st, T.qArr[0] + 0.3, T.qArr[1] - T.qArr[0]), sl = f(st, T.slide[0], T.slide[1] - T.slide[0]);
          var al = f(st, T.rot[0], T.rot[1] - T.rot[0]), sc = f(st, T.stretch[0], T.stretch[1] - T.stretch[0]);
          var ca = Math.cos(Math.PI / 2 * al), sn = Math.sin(Math.PI / 2 * al), kk = 1 + (KW - 1) * sc;
          var dv = [(ca * MQ[0] - sn * MQ[1]) * kk, (sn * MQ[0] + ca * MQ[1]) * kk];
          var tl0 = [Qw[0] * (1 - sl), Qw[1] * (1 - sl)];
          var tipW = sl > 0 ? [tl0[0] + dv[0], tl0[1] + dv[1], 0] : [Qw[0] + MQ[0] * g0, Qw[1] + MQ[1] * g0, 0];
          var t0p = Pc([tl0[0], tl0[1], 0]), t1p = Pc(tipW);
          dA1.set(t0p[0], t0p[1], t1p[0], t1p[1]); E.op(dA1.g, g0 > 0.01 ? 1 : 0);
          var arcOn = al > 0.02 ? 1 - f(st, T.part2, 0.3) : 0;
          if (al > 0.02) { var ap = ringPts(Pc, [0, 0, 0], 0.6, AO, AO + Math.PI / 2 * al, 24); rArc.setAttribute('d', E.poly(ap)); headAt(rArcHead, ap, 10); }
          E.op(rArc, arcOn); E.op(rArcHead, arcOn);
          var rl = Pc([1.15 * Math.cos(AO + Math.PI / 4), 1.15 * Math.sin(AO + Math.PI / 4), 0]);
          rLab.setAttribute('x', rl[0].toFixed(1)); rLab.setAttribute('y', (rl[1] + 7).toFixed(1));
          E.op(rLab, f(st, T.rot[0] + 0.3, 0.3) * (1 - f(st, T.part2, 0.3)));
          var P1 = [KW * -MQ[1], KW * MQ[0], 0], p1s = Pc(P1);
          p1Lab.setAttribute('x', (p1s[0] + 12).toFixed(1)); p1Lab.setAttribute('y', (p1s[1] + 14).toFixed(1)); E.op(p1Lab, f(st, T.p1Lab, 0.3));
          dA1.line.style.strokeWidth = 4.5 + 2.5 * b(st, T.stretch[1], 0.8);
          // 句 9：沿着轴走（转轴旁的绿箭头）；第一份的尖上往上接第二份
          var ug = f(st, T.along, 0.5), u0 = Pc([Qw[0], Qw[1], 0.2]), u1 = Pc([Qw[0], Qw[1], 0.2 + 0.7 * ug]);
          upA.set(u0[0] + 18, u0[1], u1[0] + 18, u1[1]); E.op(upA.g, ug > 0.01 ? 1 - f(st, T.sum[0], 0.4) : 0);
          var g2 = f(st, T.grow[0], T.grow[1] - T.grow[0]), p2s = Pc([P1[0], P1[1], HW * g2]);
          dA2.set(p1s[0], p1s[1], p2s[0], p2s[1]); E.op(dA2.g, g2 > 0.01 ? 1 : 0);
          p2Lab.setAttribute('x', (p2s[0] + 12).toFixed(1)); p2Lab.setAttribute('y', (p2s[1] + 12).toFixed(1)); E.op(p2Lab, f(st, T.p2Lab, 0.3));
          // 句 10：两份加起来就是 v（原点 → 第二份的尖）
          var gv = f(st, T.sum[0], T.sum[1] - T.sum[0]), vt = Pc([P1[0] * gv, P1[1] * gv, HW * gv]);
          dAV.set(O[0], O[1], vt[0], vt[1]); E.op(dAV.g, gv > 0.01 ? 1 : 0);
          var vb = b(st, T.vLab, 0.9); dAV.line.style.strokeWidth = 5.5 + 3 * vb; E.glow(dAV.line, E.VEL, vb > 0.05);
          vLab2.setAttribute('x', ((O[0] + vt[0]) / 2 - 6).toFixed(1)); vLab2.setAttribute('y', ((O[1] + vt[1]) / 2 - 14).toFixed(1)); E.op(vLab2, f(st, T.vLab, 0.3));

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
          E.op(m1, f(st, T.part1, 0.3)); E.op(m1a, f(st, T.qArr[0], 0.3)); E.op(m1b, f(st, T.slide[0], 0.3)); E.op(m1c, f(st, T.stretch[0], 0.3)); E.op(m1f, f(st, T.p1Lab, 0.3)); E.op(m1d, f(st, T.qArr[0] + 0.6, 0.3));
          E.op(m2, f(st, T.part2, 0.3)); E.op(m2a, f(st, T.dir2, 0.3)); E.op(m2f, f(st, T.p2Lab, 0.3));
          E.pop(sumG, RX + 170, 383, f(st, T.plus, 0.35), 0.92);
        }
      };
    }
  };
})();
