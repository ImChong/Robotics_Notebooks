// 009 第 3 幕：运动旋量——角速度 + 线速度，边讲边画手算一遍。左边网格 + UR5（肘在 (1, 2)，肘上 ⊙ 是朝向你的角速度），右栏一行行写算式；
// 每一步算式，网格上同时画出来（初版只在右栏写算式，左边不动，用户反馈「纯公式推导没有画面观众会看不懂」）。
// 这一幕的速度箭头按 1 : 1 画（1 格每秒画 1 格长），算出来的 (2, −1)、(4, −2) 和箭头一样长；网格放大到 x −2…4、y −2…3，装得下。
// 句 1 右栏「已知：角速度 ω」「还缺：随便一个点的速度」「→ 每一点的速度都能算出来」；
// 句 2「原点」原点亮一圈、写「原点 = 肩关节」，右栏「选原点（肩关节）」，「不在小臂上」「怎么办？」；
// 句 3「透明板」一张很大的透明板（淡黄格子）钉在小臂上，「跟着小臂一起转」小臂连同板从 −14° 转到水平，
//    「盖在原点」板上此刻在原点的那一点亮出来，「v」原点旁写「v = ?」；右栏「v：板上此刻在原点那一点的速度」；
// 计算一步不跳（用户反馈「第三幕的矩阵运算要展开各项矩阵一步一步算，不要只画一个箭头跳过计算步骤」），右栏用 KaTeX 一段一段写出来：
// 句 4「肘在一、二」肘旁写 (1, 2)，「从肘指到原点」肘到原点画出连线箭头（黄），「原点减去肘」右栏 [0; 0] − [1; 2]，
//    「零减一」「零减二」= [0 − 1; 0 − 2]、网格上画出往左 1、往下 2 两段，「得到」= [−1; −2]；
// 句 5「垂直」原点处标直角、画出过原点的垂线，「逆时针转九十度」连线箭头平移到原点（尾巴在原点，指到 (−1, −2)），旁边出蓝色的小弯箭头，
//    「乘这个矩阵」右栏 R = [0 −1; 1 0]（第一列红、第二列绿，同第 3 集），「两列是」两列各写一行；
// 句 6「矩阵乘向量」R [−1; −2]，「第一列取负一份」= (−1)[0; 1]，「是零、负一」= [0; −1]、网格上从原点往下画红的一段，
//    「第二列取负二份」+ (−2)[−1; 0]，「是二、零」+ [2; 0]、接在红的尖上往右画绿的一段，「加起来」黄箭头绕原点转 90°（蓝弧）正好落在绿的尖上，= [2; −1]；
// 句 7「再乘以二」2 [2; −1]，「二乘二」= [2 × 2; 2 × (−1)]、黄箭头拉长两倍变成紫色的 v，「v 等于」= [4; −2]、写 v = (4, −2)；
// 句 8 右栏换成六维列向量（「角速度」⊙ 鼓一下，「v」v 鼓一下）；
// 句 9「等于 v」v 平移一份到肘上；句 10「肘在一、二」原点到肘画黄虚线，「转九十度」R [1; 2]，「第一列取一份」= 1 [0; 1]、「是零、一」= [0; 1] 画红的一段，
//    「第二列取两份」+ 2 [−1; 0]、「是负二、零」+ [−2; 0] 画绿的一段，「加起来」虚线箭头绕原点转 90° 落在绿的尖上，= [−2; 1]；
// 句 11「再乘以二」2 [−2; 1] = [2 × (−2); 2 × 1] = [−4; 2]，(−2, 1) 挪到 v 的箭头尖、拉长两倍成 (−4, 2)，正好回到肘；
// 句 12「加上 v」[−4; 2] + [4; −2] = [−4 + 4; 2 + (−2)] = [0; 0]，「正好是零」两支一起鼓一下、写「合起来 = 0」，「不动」；
// 句 13「v 不是肩关节的速度」原点的圈鼓一下、写「没动」，右栏打 ×；「透明板上」板描边亮一下、在原点的那一点鼓一下，右栏打勾。
// 时刻取自配音的逐词时间，见 ../ep009.py。
(function () {
  var E = window.EP009, K = E.K, H = E.HEX;
  var D = Math.PI / 180, W = 2;
  var OX = 128, OY = 272, ARM_S = 1.82, RX = E.ARM.RX;          // 网格一格约 50 px（第 1、2 幕约 66 px）
  var T = {
    known: 0.87, lack: 1.74, all: 7.81, out1: 9.2,
    origin: 10.85, shoulder: 12.06, notOn: 13.88, how: 14.92, out2: 16.3,
    plate: 18.22, rot: [19.01, 20.6], cover: 22.02, vLab: 24.73, plateTxt: 24.33,
    clear3: 25.9, q: [27.24, 27.93], link: 28.84, sub: 29.95, rel: [31.19, 32.5], res0: 34.01,
    clearB: 36.9, perp: 37.41, ccw: 38.21, hint: 38.74, mat: 42.49, col1: 44.24, col2: 45.1,
    mv: 47.16, take1: 48.27, got1: 49.56, take2: 50.81, got2: 52.19, turn: [53.34, 54.1], res: 54.03,
    clearC: 56.3, times: 57.62, e1: 58.58, stretch: [58.58, 59.6], v: [61.84, 62.31],
    clear6: 63.7, omega: 63.84, vv: 65.0, column: 65.9, name: 67.3, nums: [68.43, 68.91, 69.41, 69.9, 70.49, 71.18],
    clear7: 72.45, any: 73.56, eqv: [74.73, 75.38], plus: 75.56, part: 77.55,
    qq: 81.08, qr: 82.66, qTake1: 83.61, qGot1: 84.68, qTake2: 85.75, qGot2: 86.92, qTurn: [88.22, 89.0], r90: 88.9,
    x2t: 91.16, move: [91.4, 92.3], x2: [91.91, 94.53],
    add: [96.78, 97.6], sum1: 97.59, zero: 100.32, axis: 101.68, still: 102.81, out9: 104.1,
    notS: 105.12, noMove: 107.61, plateHi: 108.52, here: 109.92
  };
  var PHI0 = -14 * D, END = 112.12;
  var PLATE = [-1.2, -2.6, 2.9, 1.75];                        // 透明板在「小臂水平时」的范围（相对肘，格）；转到 −14° 时右下角也不出画面

  E.scenes[2] = {
    title: '运动旋量：角速度 + 线速度',
    dur: END,
    cues: [
      { at: 0, s: '有了**角速度**，再知道一个点的速度，每一点的速度都能算' },
      { at: 9.34, s: '这个点通常选原点（肩关节），可它不在小臂上' },
      { at: 16.49, s: '小臂想成很大的透明板：板上此刻在原点那一点的速度，记作 **v**' },
      { at: 25.99, s: '肘 → 原点 = 原点 − 肘：(0 − 1, 0 − 2) = (−1, −2)' },
      { at: 36.05, s: '垂直：逆时针转 90° = 乘矩阵 R，两列是 (0, 1) 和 (−1, 0)' },
      { at: 47.07, s: '第一列取 −1 份 + 第二列取 −2 份：(0, −1) + (2, 0) = (2, −1)' },
      { at: 56.26, s: '再乘**角速度** 2：(2 × 2, 2 × (−1)) = (4, −2)，就是 **v**' },
      { at: 63.75, s: '**运动旋量**：**角速度**三个数 + **v** 三个数 = (0, 0, 2, 4, −2, 0)' },
      { at: 72.49, s: '验算肘：**v** + 绕原点转的那一份（位置转 90°，再乘**角速度**）' },
      { at: 80.99, s: 'R (1, 2) = 1 × (0, 1) + 2 × (−1, 0) = (0, 1) + (−2, 0) = (−2, 1)' },
      { at: 91.07, s: '再乘 2：(2 × (−2), 2 × 1) = (−4, 2)' },
      { at: 96.69, s: '(−4, 2) + (4, −2) = (−4 + 4, 2 + (−2)) = (0, 0)：肘本来就不动' },
      { at: 104.15, s: '**v** **不是**肩关节的速度，是透明板上此刻在原点那一点的速度' }
    ],
    build: function (K) {
      var s = K.sceneSvg('有了角速度，再知道一个点的速度，刚体上每一点的速度都能算出来；这个点选在原点，也就是肩关节，把小臂想成一张很大的透明板跟着一起转，板上此刻在原点的那一点的速度记作 v；肘在 (1, 2)，从肘指到原点是原点减肘：(0 − 1, 0 − 2) = (−1, −2)；速度和连线垂直，逆时针转 90 度就是乘矩阵 R，两列是 (0, 1) 和 (−1, 0)：第一列取 −1 份是 (0, −1)，第二列取 −2 份是 (2, 0)，加起来是 (2, −1)；再乘角速度 2：(2 × 2, 2 × (−1)) = (4, −2) 就是 v；角速度和 v 竖着排成一列，就是运动旋量 (0, 0, 2, 4, −2, 0)；验算肘：R (1, 2) = (0, 1) + (−2, 0) = (−2, 1)，乘 2 是 (−4, 2)，加上 v (4, −2) 正好是 (0, 0)，肘本来就不动；v 不是肩关节的速度');

      var under = E.el('g', {}, s);
      var arm = E.armAt(s, OX, OY, ARM_S, 300);
      var u = arm.aim(1, 2, 0);
      var pl = E.plane(under, { ox: OX, oy: OY, u: u, x0: -2, x1: 4, y0: -2, y1: 3 });
      var xLab = pl.axes.querySelectorAll('text')[0];
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
      var outRing = K.paint(E.el('circle', { r: 14, 'stroke-width': 3.5 }, outOf), E.S1, E.OMEGA);
      K.paint(E.el('circle', { r: 4.5 }, outOf), E.OMEGA);
      var qLab = E.halo(E.txt(top, Q[0] + 12, Q[1] - 24, '(1, 2)', 22, E.HI, 'start', 900));
      // 「合起来 = 0」「不动」写在 y 轴左边、和肘同高：肘的右下是 v 和 (−4, 2) 两支箭头
      var still = E.halo(E.txt(top, O[0] - 22, Q[1] + 7, '不动', 22, E.GOOD, 'end', 900));
      var zeroLab = E.halo(E.txt(top, O[0] - 22, Q[1] + 7, '合起来 = 0', 20, E.GOOD, 'end', 900));

      /* ── 句 4：肘 → 原点 = 原点 − 肘：往左 1、往下 2 ── */
      function legs(path, x0, y0, x1, y1) {   // 先横后竖：(x0, y0) → (x1, y0) → (x1, y1)
        var a = P(x0, y0), b = P(x1, y0), c = P(x1, y1);
        path.setAttribute('d', E.poly([a, b, c]));
      }
      var rLegs = K.paint(E.el('path', { fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '5 5' }, top), null, E.HI);
      var rL1 = E.halo(E.txt(top, P(0.36, 2)[0], P(0.36, 2)[1] - 10, '−1', 22, E.HI, 'middle', 900));
      var rL2 = E.halo(E.txt(top, P(0, 1)[0] - 12, P(0, 1)[1] + 8, '−2', 22, E.HI, 'end', 900));
      /* ── 句 5、6：垂线、直角；连线平移到原点；乘矩阵 R = 第一列取 −1 份（红）+ 第二列取 −2 份（绿），网格上头尾相接（同第 3 集） ── */
      var perpLine = K.paint(E.el('path', { d: E.poly([P(-0.3, 0.15), P(4, -2)]), fill: 'none', 'stroke-width': 2, 'stroke-dasharray': '3 6' }, top), null, E.INK2);
      var corner = K.paint(E.el('path', { fill: 'none', 'stroke-width': 2.5 }, top), null, E.INK);
      var cA = E.vec(top, E.X, 4.5, { head: 14 }), cB = E.vec(top, E.Y, 4.5, { head: 14 });
      var link = E.vec(top, E.HI, 4, { head: 14 });            // 肘 → 原点的连线，之后平移到原点、转 90°、拉长成 v
      var linkLab = E.halo(E.txt(top, 0, 0, '', 22, E.HI, 'start', 900));
      var turnArc = E.arcArrow(top, O[0], O[1], 0.7 * u, E.OMEGA, 3.5, { head: 12 });
      var turnLab = E.halo(E.txt(top, 0, 0, '90°', 20, E.OMEGA, 'middle', 900));

      /* ── 板上此刻在原点的那一点 + v ── */
      var pDot = K.paint(E.el('circle', { r: 7, 'stroke-width': 2.5 }, top), E.VEL, E.S1);
      var vAsk = E.halo(E.txt(top, O[0] + 34, O[1] - 14, 'v = ?', 24, E.VEL, 'start', 900));
      var vLab = E.halo(E.txt(top, P(4, -2)[0] - 10, P(4, -2)[1] + 30, 'v = (4, −2)', 24, E.VEL, 'end', 900));

      /* ── 验算肘：v 平移到肘；R (1, 2) = 1 份红 + 2 份绿 = (−2, 1)；再乘 2 成 (−4, 2)，挪到 v 的尖，头尾相接回到肘 ── */
      var vCopy = E.vec(top, E.VEL, 4.5, { head: 15 });
      var qA = E.vec(top, E.X, 4, { head: 13 }), qB = E.vec(top, E.Y, 4, { head: 13 });
      var qArr = E.vec(top, E.HI, 3, { head: 13, dash: '7 5' });
      var qTurnArc = E.arcArrow(top, O[0], O[1], 0.75 * u, E.OMEGA, 3, { head: 11 });
      var partArr = E.vec(top, E.OMEGA, 4.5, { head: 15 });
      var partLab = E.halo(E.txt(top, 0, 0, '', 20, E.OMEGA, 'start', 900));

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
      // 算式一段一段写（KaTeX）：y 是基线；横向位置由 flow 量出每段的实际宽度后挨着排
      function tx(g, x, y, tex, w) { return E.tex(g, RX + x, y, tex, 24, E.INK, 'start', { w: w || 140, h: 76 }); }
      var flows = [];
      function flow(x0, parts) { flows.push({ x0: x0, parts: parts }); }   // x0：离 RX 的距离，或者对齐到某一段的左缘
      function layoutFlows() {                                // 每帧都量：KaTeX 的字体晚到也排得对
        flows.forEach(function (fl) {
          var x = typeof fl.x0 === 'number' ? RX + fl.x0 : +fl.x0.getAttribute('x');
          fl.parts.forEach(function (fo) {
            var r = fo.getBoundingClientRect(), k = r.width / +fo.getAttribute('width') || 1, c = fo.texBox.firstElementChild;
            fo.setAttribute('x', x.toFixed(1)); x += (c ? c.getBoundingClientRect().width / k : 0) + 5;
            fl.end = x;
          });
        });
      }
      var C = E.col, cn = E.cnum, HX = H.x, HY = H.y;
      var RM = '\\begin{bmatrix}' + cn(0, HX) + '&' + cn(-1, HY) + '\\\\ ' + cn(1, HX) + '&' + cn(0, HY) + '\\end{bmatrix}';   // 转 90° 的矩阵：两列红、绿
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
      // 句 4：肘 → 原点 = 原点 − 肘
      var g3 = E.el('g', {}, R);
      var c0 = line(g3, 50, [['算 ', E.INK2], ['v', E.VEL, 900], ['：', E.INK2]], 24);
      var c1 = line(g3, 96, [['肘：', E.INK2], ['(1, 2)', E.HI, 900]]);
      var c2 = line(g3, 140, [['肘 → 原点 = 原点 − 肘', E.INK, 900]]);
      var c3a = tx(g3, 0, 214, C([0, 0]) + '-' + C([1, 2], H.hi), 120);
      var c3b = tx(g3, 112, 214, '=\\begin{bmatrix}0-1\\\\ 0-2\\end{bmatrix}', 120);
      var c3c = tx(g3, 222, 214, '=' + C([-1, -2]), 100);
      // 句 5、6：乘矩阵 R，第一列取 −1 份、第二列取 −2 份
      var g3b = E.el('g', {}, R);
      var b0 = line(g3b, 50, [['垂直：', E.INK2], ['逆时针转 90°', E.OMEGA, 900], [' = 乘矩阵 R', E.INK, 900]]);
      var b1 = tx(g3b, 0, 122, 'R=' + RM, 150);
      var b1x = E.txt(g3b, RX + 172, 112, '第一列 (0, 1)', 20, E.X, 'start', 900);
      var b1y = E.txt(g3b, RX + 172, 144, '第二列 (−1, 0)', 20, E.Y, 'start', 900);
      var b2a = tx(g3b, 0, 214, 'R' + C([-1, -2], [HX, HY]), 90);
      var b2b = tx(g3b, 74, 214, '=' + cn('(-1)', HX) + C([0, 1], HX), 130);
      var b2c = tx(g3b, 190, 214, '+' + cn('(-2)', HY) + C([-1, 0], HY), 140);
      var b3a = tx(g3b, 74, 300, '=' + C([0, -1], HX), 90);
      var b3b = tx(g3b, 148, 300, '+' + C([2, 0], HY), 80);
      var b3c = tx(g3b, 214, 300, '=' + C([2, -1], H.hi), 90);
      // 句 7：再乘角速度 2
      var g3c = E.el('g', {}, R);
      var d0 = line(g3c, 50, [['再乘 ', E.INK2], ['角速度 2', E.OMEGA, 900], ['：', E.INK2]]);
      var d1a = tx(g3c, 0, 136, cn(2, H.acc) + C([2, -1]), 80);
      var d1b = tx(g3c, 62, 136, '=\\begin{bmatrix}2\\times 2\\\\ 2\\times(-1)\\end{bmatrix}', 160);
      var d1c = tx(g3c, 222, 136, '=' + C([4, -2], H.violet), 100);
      var d2 = line(g3c, 236, [['v', E.VEL, 900], [' = ', E.INK], ['(4, −2)', E.VEL, 900]], 28);
      // 句 8：六维列向量
      var g4 = E.el('g', {}, R), CX = RX + 200, CY = 226;
      var colv = E.colvec(g4, CX, CY, [0, 0, 2, 4, -2, 0], { size: 28, dy: 36, colors: [E.OMEGA, E.OMEGA, E.OMEGA, E.VEL, E.VEL, E.VEL], w: 70 });
      var wTag = E.txt(g4, CX + 56, CY - 50, 'ω', 28, E.OMEGA, 'start', 900), vTag = E.txt(g4, CX + 56, CY + 66, 'v', 28, E.VEL, 'start', 900);
      var wTag2 = E.txt(g4, CX + 86, CY - 50, '角速度', 20, E.OMEGA, 'start', 700), vTag2 = E.txt(g4, CX + 82, CY + 66, '线速度', 20, E.VEL, 'start', 700);
      var vName = E.txt(g4, CX - 56, CY + 10, '运动旋量 =', 26, E.HI, 'end', 900);
      // 句 9–12：验算
      var g5 = E.el('g', {}, R);
      var e0 = line(g5, 50, [['验算：', E.INK2]], 24);
      var e1 = line(g5, 92, [['某点的速度 = ', E.INK, 900], ['v', E.VEL, 900], [' + 绕原点转的那一份', E.INK, 900]], 20);   // 20 号：空格照排（white-space: pre）以后，22 号会超出右边
      var e2 = line(g5, 128, [['那一份：位置转 90°，再乘 ω', E.OMEGA, 700]], 20);
      var e3a = tx(g5, 0, 204, 'R' + C([1, 2], [HX, HY]), 80);
      var e3b = tx(g5, 64, 204, '=' + cn(1, HX) + C([0, 1], HX), 100);
      var e3c = tx(g5, 152, 204, '+' + cn(2, HY) + C([-1, 0], HY), 110);
      var e4a = tx(g5, 64, 288, '=' + C([0, 1], HX), 80);
      var e4b = tx(g5, 128, 288, '+' + C([-2, 0], HY), 90);
      var e4c = tx(g5, 204, 288, '=' + C([-2, 1], H.hi), 90);
      var e5a = tx(g5, 0, 204, cn(2, H.acc) + C([-2, 1]), 80);   // 句 11：再乘 2（写在 e3、e4 收掉的位置）
      var e5b = tx(g5, 62, 204, '=\\begin{bmatrix}2\\times(-2)\\\\ 2\\times 1\\end{bmatrix}', 160);
      var e5c = tx(g5, 222, 204, '=' + C([-4, 2], H.acc), 100);
      var e6a = tx(g5, 0, 292, C([-4, 2], H.acc) + '+' + C([4, -2], H.violet), 110);   // 句 12：加上 v
      var e6b = tx(g5, 100, 292, '=\\begin{bmatrix}-4+4\\\\ 2+(-2)\\end{bmatrix}', 150);
      var e6c = tx(g5, 246, 292, '=' + C([0, 0], H.good), 80);
      var e6k = E.check(g5, 0, 0, 20, E.GOOD, 4);   // 位置在 layoutFlows 之后跟着 e6 这一行的末尾
      flow(0, [c3a, c3b, c3c]);
      flow(0, [b2a, b2b, b2c]); flow(b2b, [b3a, b3b, b3c]);
      flow(0, [d1a, d1b, d1c]);
      flow(0, [e3a, e3b, e3c]); flow(e3b, [e4a, e4b, e4c]);
      flow(0, [e5a, e5b, e5c]); flow(0, [e6a, e6b, e6c]);
      // 句 13（验算收掉以后写在上面）
      var g6 = E.el('g', {}, R);
      var n1 = line(g6, 110, [['v', E.VEL, 900], [' 不是肩关节的速度', E.ERR, 900]]);
      var n1x = E.cross(n1.g, RX + 250, 102, 10, E.ERR, 4);
      var n2 = line(g6, 156, [['v', E.VEL, 900], [' 是板上此刻在原点那一点的', E.GOOD, 900]], 22);
      var n2b = line(g6, 194, [['　速度', E.GOOD, 900]], 22);
      var n2c = E.check(n2b.g, RX + 92, 178, 20, E.GOOD, 4);

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) { layoutRows(); laid = true; }
          if (s.isConnected) { layoutFlows(); var fe = flows[flows.length - 1].end; e6k.setAttribute('transform', 'translate(' + Math.min(fe + 14, RX + 330).toFixed(1) + ' ' + (fe + 14 > RX + 330 ? 336 : 286) + ')'); }
          var f = E.fade, b = E.bump;

          /* ── 小臂（连同透明板）从 −14° 转到水平 ── */
          var phi = PHI0 * (1 - f(st, T.rot[0], T.rot[1] - T.rot[0]));
          arm.bend(phi);
          V.clearTint(); V.tint(RIGID, E.BODY, 0.4);
          var c = Math.cos(phi), sn = Math.sin(phi);
          function plateP(x, y) { return P(1 + c * x - sn * y, 2 + sn * x + c * y); }   // 板上一点（相对肘、小臂水平时的坐标）→ 画面
          plateFill.setAttribute('d', E.poly([plateP(PLATE[0], PLATE[1]), plateP(PLATE[2], PLATE[1]), plateP(PLATE[2], PLATE[3]), plateP(PLATE[0], PLATE[3])], true));
          plateLines.forEach(function (l) { l.el.setAttribute('d', E.poly([plateP(l.a[0], l.a[1]), plateP(l.b[0], l.b[1])])); });
          var pl0 = plateP(PLATE[2] - 1.55, PLATE[3] - 0.75); plateLab.setAttribute('x', pl0[0].toFixed(1)); plateLab.setAttribute('y', pl0[1].toFixed(1));
          var plb = b(st, T.plateHi, 1);
          plateFill.style.strokeWidth = 2.5 + 2.5 * plb; E.glow(plateFill, E.BODY, plb > 0.05);
          E.op(plate, f(st, T.plate, 0.6));

          /* ── 原点、肘 ── */
          // 透明板淡入、转到水平的那一段，板的左边扫过「原点 = 肩关节」：两行字先淡下去，转完再亮回来
          // 验算时 (1, 2) 绕原点转到 (−2, 1)，扫过这两行字：也先淡下去
          var dim = 1 - 0.85 * Math.max(E.win(st, T.plate - 0.2, T.rot[1] + 0.1, 0.3), E.win(st, T.qTurn[0] - 0.3, T.move[0] + 0.2, 0.3));
          E.op(oRing, f(st, T.origin, 0.3)); E.op(oLab, f(st, T.origin, 0.3) * dim); E.op(oLab2, f(st, T.shoulder, 0.3) * dim);
          oRing.setAttribute('r', (10 + 5 * Math.max(b(st, T.origin, 0.7), b(st, T.notS, 0.8))).toFixed(2));
          E.op(noMove, f(st, T.noMove, 0.3));
          var e = arm.elbow();
          outOf.setAttribute('transform', 'translate(' + e[0].toFixed(1) + ' ' + e[1].toFixed(1) + ')');
          outRing.setAttribute('r', (14 + 5 * Math.max(b(st, T.omega, 0.8), b(st, T.axis, 0.8))).toFixed(2));
          E.op(qLab, f(st, T.q[1], 0.3));
          E.op(still, f(st, T.still, 0.3) * (1 - f(st, T.out9, 0.4)));

          /* ── 句 4：肘 → 原点的连线（黄），往左 1、往下 2 ── */
          var out6 = 1 - f(st, T.clear6, 0.4);
          rLegs.setAttribute('d', ''); legs(rLegs, 1, 2, 0, 0);
          E.op(rLegs, f(st, T.rel[0], 0.4) * (1 - f(st, T.ccw, 0.3)));
          E.op(rL1, f(st, T.rel[0], 0.3) * (1 - f(st, T.ccw, 0.3))); E.op(rL2, f(st, T.rel[1], 0.3) * (1 - f(st, T.ccw, 0.3)));
          // 句 5：垂线、直角
          E.op(perpLine, f(st, T.perp, 0.4) * out6);
          var rr = [1 / Math.sqrt(5), 2 / Math.sqrt(5)], vv = [2 / Math.sqrt(5), -1 / Math.sqrt(5)], m = 13 / u;
          corner.setAttribute('d', E.poly([P(rr[0] * m, rr[1] * m), P(rr[0] * m + vv[0] * m, rr[1] * m + vv[1] * m), P(vv[0] * m, vv[1] * m)]));
          E.op(corner, f(st, T.perp, 0.3) * out6);
          // 连线箭头：画出来（肘 → 原点）→ 平移到原点（原点 → (−1, −2)）→「加起来」时绕原点转 90° 到 (2, −1)，正好落在红、绿两段的尖上 → 拉长两倍成 v (4, −2)
          var lk = f(st, T.link, 0.5), sl = f(st, T.ccw, 0.7), tr = f(st, T.turn[0], T.turn[1] - T.turn[0]), sx = f(st, T.stretch[0], T.stretch[1] - T.stretch[0]);
          var ang = Math.atan2(-2, -1) + Math.PI / 2 * tr, len = Math.sqrt(5) * (1 + sx);
          var tail = [1 - sl, 2 - 2 * sl];                                            // 平移：尾巴从肘挪到原点
          var vec = sl > 0 ? [len * Math.cos(ang), len * Math.sin(ang)] : [-lk, -2 * lk];
          var t0 = P(tail[0], tail[1]), t1 = P(tail[0] + vec[0], tail[1] + vec[1]);
          link.set(t0[0], t0[1], t1[0], t1[1]); E.op(link.g, lk > 0 ? 1 : 0);
          var col = sx > 0.5 ? E.VEL : E.HI;
          link.paint(col);
          var vb = Math.max(b(st, T.vv, 0.8), b(st, T.notS, 0.8), b(st, T.add[0], 0.7), b(st, T.zero, 0.8), b(st, T.res, 0.7));
          link.line.style.strokeWidth = 4 + 1.5 * sx + 3 * vb; E.glow(link.g, col, vb > 0.05);
          // 箭头尖上的坐标：平移完写 (−1, −2)，转完写 (2, −1)（写在尖的正下方：右下是垂线、左上是「90°」）；拉长时收掉（v 的标签另写在下面）
          linkLab.textContent = st >= T.res ? '(2, −1)' : '(−1, −2)';
          var ll = st >= T.res ? [t1[0], t1[1] + 46] : [t1[0] + 14, t1[1] + 10];   // (−1, −2) 写在尖的右边：尖在网格左下角，往左写会出画面
          linkLab.setAttribute('x', ll[0].toFixed(1)); linkLab.setAttribute('y', ll[1].toFixed(1)); linkLab.style.textAnchor = st >= T.res ? 'middle' : 'start';
          E.op(linkLab, (st >= T.res ? f(st, T.res, 0.3) : f(st, T.ccw + 0.6, 0.3) * (1 - f(st, T.turn[0], 0.2))) * (1 - f(st, T.stretch[0], 0.3)));
          // 「逆时针转九十度」：原点旁的小弯箭头；「加起来」它跟着转过去
          var a0 = Math.atan2(-2, -1);
          turnArc.set(a0, a0 + Math.PI / 2 * Math.max(0.35 * f(st, T.hint, 0.4), tr));
          E.op(turnArc.g, f(st, T.hint, 0.3) * (1 - f(st, T.stretch[0], 0.3)));
          var tl = P(1.3 * Math.cos(a0 - 30 * D), 1.3 * Math.sin(a0 - 30 * D));   // 写在平移过来的箭头左边、「= 肩关节」下面：弧的箭头不压字，右下要画红、绿两段
          turnLab.setAttribute('x', tl[0].toFixed(1)); turnLab.setAttribute('y', (tl[1] + 7).toFixed(1));
          E.op(turnLab, f(st, T.hint + 0.2, 0.3) * (1 - f(st, T.stretch[0], 0.3)));
          // 句 6：第一列取 −1 份 = (0, −1)（红，从原点往下），第二列取 −2 份 = (2, 0)（绿，接在红的尖上往右）
          var ga = f(st, T.got1, 0.5), gb = f(st, T.got2, 0.5), cOn = 1 - f(st, T.stretch[0], 0.3);
          var pa0 = P(0, 0), pa1 = P(0, -ga), pb1 = P(2 * gb, -1);
          cA.set(pa0[0], pa0[1], pa1[0], pa1[1]); E.op(cA.g, ga > 0.01 ? cOn : 0);
          cB.set(pa1[0], pa1[1], pb1[0], pb1[1]); E.op(cB.g, gb > 0.01 ? cOn : 0);

          /* ── 板上在原点的那一点、v 的标签 ── */
          var here = Math.max(b(st, T.cover, 0.8), b(st, T.here, 0.9));
          pDot.setAttribute('cx', O[0]); pDot.setAttribute('cy', O[1]); pDot.setAttribute('r', (7 + 4 * here).toFixed(2));
          E.op(pDot, f(st, T.cover, 0.3)); E.glow(pDot, E.VEL, here > 0.05);
          E.op(vAsk, f(st, T.vLab, 0.3) * (1 - f(st, T.link - 0.3, 0.3)));
          vLab.textContent = st >= T.v[1] ? 'v = (4, −2)' : 'v = (4,';
          E.op(vLab, f(st, T.v[0], 0.3));

          /* ── 验算肘：v 平移到肘上；原点到肘的 (1, 2)：1 份红 + 2 份绿 = (−2, 1)，箭头转过去；再拉长两倍成 (−4, 2)、挪到 v 的尖，回到肘 ── */
          var out9 = 1 - f(st, T.out9, 0.4);
          var ev = f(st, T.eqv[0], T.eqv[1] - T.eqv[0]), cv0 = P(ev, 2 * ev), cv1 = P(ev + 4, 2 * ev - 2);
          vCopy.set(cv0[0], cv0[1], cv1[0], cv1[1]); E.op(vCopy.g, f(st, T.eqv[0], 0.2) * out9);
          var qga = f(st, T.qGot1, 0.5), qgb = f(st, T.qGot2, 0.5), qOn = 1 - f(st, T.move[0], 0.3);
          var qa1 = P(0, qga), qb1 = P(-2 * qgb, 1);
          qA.set(O[0], O[1], qa1[0], qa1[1]); E.op(qA.g, qga > 0.01 ? qOn * out9 : 0);
          qB.set(qa1[0], qa1[1], qb1[0], qb1[1]); E.op(qB.g, qgb > 0.01 ? qOn * out9 : 0);
          var qd = f(st, T.qq, 0.5), qt = f(st, T.qTurn[0], T.qTurn[1] - T.qTurn[0]), mv = f(st, T.move[0], T.move[1] - T.move[0]);
          var qa = Math.atan2(2, 1) + Math.PI / 2 * qt, ql = Math.sqrt(5) * qd;
          var q1 = P(ql * Math.cos(qa), ql * Math.sin(qa));
          qArr.set(O[0], O[1], q1[0], q1[1]); E.op(qArr.g, qd > 0 ? out9 * qOn : 0);
          qTurnArc.set(Math.atan2(2, 1), qa); E.op(qTurnArc.g, qt > 0.02 ? out9 * qOn : 0);
          // (−2, 1) 从原点挪到 v 副本的尖 (5, 0)，同时拉长到 (−4, 2)；往上偏 6 px，不和 v 副本叠成一根
          var pS = [5 * mv, 0], pV = [-2 * (1 + mv), 1 + mv], pa = P(pS[0], pS[1]), pb = P(pS[0] + pV[0], pS[1] + pV[1]);
          var nx = 1 / Math.sqrt(5) * 6, ny = -2 / Math.sqrt(5) * 6;
          partArr.set(pa[0] + nx * mv, pa[1] + ny * mv, pb[0] + nx * mv, pb[1] + ny * mv);
          E.op(partArr.g, mv > 0 ? out9 : 0);
          var pb2 = Math.max(b(st, T.zero, 0.8), b(st, T.sum1, 0.6));
          partArr.line.style.strokeWidth = 4.5 + 3 * pb2; vCopy.line.style.strokeWidth = 4.5 + 3 * Math.max(b(st, T.zero, 0.8), b(st, T.add[0], 0.7));
          partLab.textContent = st >= T.move[0] ? '(−4, 2)' : '(−2, 1)';
          // (−4, 2) 写在两支箭头下面、x 轴上面的空当里
          var pm = st >= T.move[0] ? P(2, 0.35) : P(-2, 1);
          partLab.setAttribute('x', (pm[0] + (st >= T.move[0] ? 0 : -6)).toFixed(1)); partLab.setAttribute('y', (pm[1] - (st >= T.move[0] ? 0 : 16)).toFixed(1));
          E.op(partLab, (st >= T.move[0] ? f(st, T.x2[1], 0.3) : f(st, T.r90, 0.3)) * out9);
          E.op(zeroLab, f(st, T.zero, 0.3) * (1 - f(st, T.still - 0.1, 0.3)));
          E.op(xLab, 1 - E.win(st, T.eqv[0], T.out9 + 0.2, 0.3));   // v 副本的尖 (5, 0) 落在 x 轴的「x」上：验算时先淡下去

          /* ── 右栏 ── */
          E.op(g1, 1 - f(st, T.out2, 0.4));
          E.op(k1.g, fr(st, T.known)); E.drawOn(k1c, f(st, T.known + 0.2, 0.3));
          E.op(k2.g, fr(st, T.lack)); E.op(k3.g, fr(st, T.all) * (1 - f(st, T.out1, 0.3)) + 0);
          E.op(k4.g, fr(st, T.origin)); E.op(k5.g, fr(st, T.notOn)); E.op(k6.g, fr(st, T.how));
          E.op(g2, f(st, T.plate, 0.4) * (1 - f(st, T.clear3, 0.4)));
          E.op(k7.g, 1); E.op(k8.g, fr(st, T.rot[0])); E.op(k9.g, fr(st, T.plateTxt)); E.op(k9b.g, fr(st, T.plateTxt));
          // 句 4：原点 − 肘，一项一项减
          E.op(g3, f(st, T.clear3 + 0.2, 0.4) * (1 - f(st, T.clearB, 0.4)));
          E.op(c0.g, 1); E.op(c1.g, fr(st, T.q[0])); E.op(c2.g, fr(st, T.sub));
          E.op(c3a, fr(st, T.sub + 0.4)); E.op(c3b, fr(st, T.rel[0])); E.op(c3c, fr(st, T.res0));
          // 句 5、6：乘矩阵 R，按列取份再相加
          E.op(g3b, f(st, T.clearB + 0.2, 0.4) * (1 - f(st, T.clearC, 0.4)));
          E.op(b0.parts[0], fr(st, T.perp)); E.op(b0.parts[1], fr(st, T.ccw)); E.op(b0.parts[2], fr(st, T.mat));
          E.op(b1, fr(st, T.mat)); E.op(b1x, fr(st, T.col1)); E.op(b1y, fr(st, T.col2));
          E.op(b2a, fr(st, T.mv)); E.op(b2b, fr(st, T.take1)); E.op(b2c, fr(st, T.take2));
          E.op(b3a, fr(st, T.got1)); E.op(b3b, fr(st, T.got2)); E.op(b3c, fr(st, T.res));
          // 句 7：再乘 2，一项一项乘
          E.op(g3c, f(st, T.clearC + 0.2, 0.4) * (1 - f(st, T.clear6, 0.4)));
          E.op(d0.g, 1); E.op(d1a, fr(st, T.times)); E.op(d1b, fr(st, T.e1)); E.op(d1c, fr(st, T.v[0])); E.op(d2.g, fr(st, T.v[1]));
          // 六维列向量
          E.op(g4, f(st, T.clear6 + 0.2, 0.4) * (1 - f(st, T.clear7, 0.4)));
          colv.rangeBox(0, 2, E.OMEGA, f(st, T.omega, 0.3)); colv.rangeBox(3, 5, E.VEL, f(st, T.vv, 0.3));
          E.op(wTag, fr(st, T.omega)); E.op(wTag2, fr(st, T.omega)); E.op(vTag, fr(st, T.vv)); E.op(vTag2, fr(st, T.vv));
          E.op(vName, fr(st, T.name));
          colv.cells.forEach(function (cl, i) { E.op(cl, f(st, T.nums[i], 0.3)); });   // 只淡入、不放大：上下两个框紧挨着数
          // 验算：R (1, 2) → 乘 2 → 加上 v
          E.op(g5, f(st, T.clear7 + 0.2, 0.4) * (1 - f(st, T.out9, 0.4)));
          E.op(e0.g, 1); E.op(e1.g, fr(st, T.any)); E.op(e1.parts[2], fr(st, T.plus)); E.op(e2.g, fr(st, T.part));
          var eOut = 1 - f(st, T.x2t - 0.2, 0.3);
          E.op(e3a, fr(st, T.qr) * eOut); E.op(e3b, fr(st, T.qTake1) * eOut); E.op(e3c, fr(st, T.qTake2) * eOut);
          E.op(e4a, fr(st, T.qGot1) * eOut); E.op(e4b, fr(st, T.qGot2) * eOut); E.op(e4c, fr(st, T.r90) * eOut);
          E.op(e5a, fr(st, T.x2t + 0.1)); E.op(e5b, fr(st, T.x2[0])); E.op(e5c, fr(st, T.x2[1]));
          E.op(e6a, fr(st, T.add[0])); E.op(e6b, fr(st, T.sum1)); E.op(e6c, fr(st, T.zero)); E.drawOn(e6k, f(st, T.zero + 0.2, 0.3));
          E.op(g6, 1);
          E.op(n1.g, fr(st, T.notS)); E.drawOn(n1x, f(st, T.noMove, 0.3));
          E.op(n2.g, fr(st, T.plateHi)); E.op(n2b.g, fr(st, T.plateHi)); E.drawOn(n2c, f(st, T.here, 0.3));
        }
      };
    }
  };
})();
