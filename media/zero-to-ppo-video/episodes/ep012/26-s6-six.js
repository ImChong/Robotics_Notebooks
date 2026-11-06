// 012 第 6 幕：六个关节——最多几组解。UR5 的 8 组解是同一个手的位姿（tool0）用 URDF 正运动学从 300 个初值做数值逆解、去重得到的
// （../ep012.py 文件头；结构同 ur_kinematics：底座 q1 两个 × 手腕 q5 两个 × 肘 q3 两个），每组的 tool0 和参考姿势逐项相同（误差 < 1e-8）。
// 句 1「真实的机械臂」左边一台斜着看的 UR5（参考姿势），「六个关节」六个关节亮圆点，「位置和朝向」手上出三根小箭头（目标位姿，之后一直留着）；
// 句 2「现成的公式」「网上公开、谁都能看的代码」右栏写 ur_kinematics，「三处，各有两个答案」写；
// 句 3「底座那个关节」「朝前伸」→「转过去、反着伸过来」UR5 换成底座反过来的那组，「肘，可以朝上」→「也可以朝下」换成肘下那组，
//    「手腕，也有两种翻法」换成手腕翻过去的那组（每次手都停在同一个位姿）；右栏逐行写；
// 句 4「两种、两种、再两种，一共八种」右栏淡出，整幅铺开 2 × 4 八台 UR5（上排朝前伸、下排反着伸；列：肘上 / 肘下 × 手腕两种翻法），
//    按念到的「两种」依次亮出，「手停在同一个位姿」八台手上的小箭头一起鼓一下；
// 句 5「最多十六组解」八台淡出，右栏写 16；
// 句 6「关节比需要的多」「平面里，手的位置只有 x、y 两个数，却有三个关节」左边画一条三节的平面链，「手不动，前面的关节还能一起挪」末端钉住、三个关节一起来回挪，
// 句 7「人形机器人」换成整个的 G1（URDF，斜着看），「一条胳膊」腿、右臂淡掉、镜头推近到上半身，「七个关节」左臂七个关节亮圆点，
//    「手的位置和朝向」（第 6 集：转动三个自由度，加上位置三个数）手上出三根小箭头，「一共六个数」起手不动、肘绕「肩—手腕」连线转一个来回
//    （描出肘走的虚线弧，写「手不动」），「也多了一个」右栏写「手不动，肘还能绕着转」；
//    2026-11-05 用户要求放大 G1 的胳膊、动一下各关节演示「手固定、解有无穷多组」（原先是整身正面的 G1、胳膊不动）；
// 句 7「手腕转到某些姿势」「万向锁」UR5 手腕中间那个关节转到 0，两根转轴画成平行的蓝线；
// 句 8 总结：左边 UR5 依次摆出 8 组解，右栏写「这台最多 8 组，最一般最多 16 组；肘上、肘下 = 开平方的正负两个」。
// 时刻取自配音的逐词时间，见 ../ep012.py。
(function () {
  var E = window.EP012, K = E.K, H = E.HEX;
  var D = Math.PI / 180, RX = E.ARM.RX;
  var T = {
    real: 0.95, six: 2.16, ori: 3.31,
    ik: 7.2, formula: 8.51, code: 9.71, three: 14.4, two: 15.53,
    base: 17.27, front: 19.12, back: [20.39, 21.57], elbow: 22.12, up: 22.81, down: [23.62, 24.4], wrist: 24.62, flip: [25.31, 26.2], out3: 27.18,
    grid: 27.48, g2: [27.57, 28.26, 28.87], eight: 29.65, same: 30.65, most8: 33.37, out4: 35.86,
    gen: 36.26, proved: 38.38, s16: 39.67,
    more: 42.14, inf: 44.27, planar: 45.52, handXY: 46.27, three3: 49.07, still: 49.97, slide: 50.64, human: 53.32, arm1: 54.56, seven: 55.59, six6: 56.64, swing: 58, extra: 59.1,
    wristS: 62.01, pose7: 63.12, gimbal: 65.36, inf2: 66.59,
    sum: 68.76, finite: 71.59, s8: 73.19, s16b: 75.02, elb: 76.43, sqrt: 79.13
  };
  var END = 81.72;
  // 8 组解（关节顺序：shoulder_pan、shoulder_lift、elbow、wrist_1、wrist_2、wrist_3，弧度），排成 2 × 4：上排底座朝前，下排反着；
  // 每排：肘上（肘在肩—手腕连线上方）·手腕 ①、肘上·手腕 ②、肘下·手腕 ①、肘下·手腕 ②（手腕 ① / ②：wrist_2 为负 / 正）
  var SOL = [
    [0.3000, -1.2500, 1.7000, -2.0000, -1.5700, 0.4000], [0.3000, -0.8121, 1.0166, 1.3871, 1.5700, -2.7416],
    [0.3000, 0.3588, -1.7000, -0.2088, -1.5700, 0.4000], [0.3000, 0.1599, -1.0166, 2.4483, 1.5700, -2.7416],
    [-2.4707, -2.3294, -1.0163, 1.7559, -1.5791, -2.3708], [-2.4707, -1.8918, -1.7002, -1.1395, 1.5791, 0.7708],
    [-2.4707, 2.9821, 1.0163, 0.6949, -1.5791, -2.3708], [-2.4707, 2.7824, 1.7002, -2.9309, 1.5791, 0.7708]
  ];
  // 句 7：G1 左臂 7 个关节的自运动。手（left_rubber_hand）的位置和朝向固定在起始姿势 q0 = (−0.6, 0.45, 0, 0.15, 0, 0, 0) 那里，
  // 肘绕「肩—手腕」连线从 −40° 转到 +62°，每 2° 一组关节角（左臂 7 个：肩 pitch / roll / yaw、肘、腕 roll / pitch / yaw，弧度）；
  // 照 URDF g1_29dof_rev_1_0（腰三个关节取 0）用 Python 沿雅可比的零空间一步一步走、每步用牛顿法拉回同一个手的位姿算出：
  // 手的位姿误差 < 1e-9，相邻两组线性插值手最多漂 0.05 mm，都在关节限位以内。G1E：对应的肘（left_elbow_link 原点）在骨盆坐标系里的位置（米）。
  // 起始姿势肘只弯 9°（G1 的肘在 0° 时小臂已经朝前弯成直角）：肘弯得少、胳膊接近伸直时，肘离「肩—手腕」连线太近，转起来看不出。
  var G1J = ['left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint'];
  var G1SW = -40, G1Q = [
    [-0.699, 0.0374, 0.5022, 0.1518, 0.4826, -0.1084, -0.4031], [-0.6895, 0.0559, 0.4751, 0.1507, 0.4603, -0.0972, -0.3846], [-0.6804, 0.0747, 0.4484, 0.1496, 0.4377, -0.0866, -0.3659], [-0.6719, 0.0938, 0.422, 0.1486, 0.415, -0.0764, -0.347],
    [-0.6638, 0.1132, 0.3959, 0.1478, 0.392, -0.0669, -0.3279], [-0.6562, 0.1329, 0.3701, 0.147, 0.3688, -0.0579, -0.3085], [-0.6491, 0.1528, 0.3445, 0.1463, 0.3453, -0.0495, -0.289], [-0.6425, 0.1729, 0.3192, 0.1457, 0.3217, -0.0417, -0.2692],
    [-0.6363, 0.1933, 0.2941, 0.1452, 0.2979, -0.0345, -0.2493], [-0.6306, 0.2139, 0.2692, 0.1449, 0.2739, -0.0279, -0.2292], [-0.6254, 0.2346, 0.2445, 0.1446, 0.2496, -0.022, -0.2089], [-0.6207, 0.2556, 0.2199, 0.1445, 0.2253, -0.0167, -0.1885],
    [-0.6164, 0.2767, 0.1954, 0.1446, 0.2007, -0.0121, -0.1679], [-0.6127, 0.298, 0.171, 0.1447, 0.176, -0.0082, -0.1472], [-0.6094, 0.3194, 0.1467, 0.145, 0.1512, -0.0049, -0.1264], [-0.6066, 0.3409, 0.1223, 0.1455, 0.1262, -0.0023, -0.1055],
    [-0.6042, 0.3626, 0.098, 0.1461, 0.1011, -0.0004, -0.0845], [-0.6024, 0.3843, 0.0736, 0.1468, 0.076, 0.0007, -0.0635], [-0.6011, 0.4062, 0.0492, 0.1477, 0.0507, 0.0012, -0.0423], [-0.6003, 0.4281, 0.0247, 0.1488, 0.0254, 0.001, -0.0212],
    [-0.6, 0.45, 0, 0.15, 0, 0, 0], [-0.6002, 0.472, -0.0248, 0.1514, -0.0254, -0.0017, 0.0212], [-0.601, 0.494, -0.0498, 0.1529, -0.0509, -0.0041, 0.0423], [-0.6023, 0.5159, -0.0751, 0.1545, -0.0763, -0.0072, 0.0635],
    [-0.6042, 0.5379, -0.1006, 0.1564, -0.1018, -0.011, 0.0845], [-0.6066, 0.5598, -0.1264, 0.1583, -0.1272, -0.0156, 0.1056], [-0.6096, 0.5816, -0.1525, 0.1604, -0.1526, -0.0208, 0.1265], [-0.6132, 0.6034, -0.179, 0.1627, -0.1779, -0.0268, 0.1473],
    [-0.6175, 0.6251, -0.2059, 0.1651, -0.2032, -0.0334, 0.168], [-0.6223, 0.6466, -0.2333, 0.1676, -0.2284, -0.0408, 0.1886], [-0.6278, 0.668, -0.2611, 0.1702, -0.2535, -0.0488, 0.209], [-0.634, 0.6892, -0.2894, 0.173, -0.2785, -0.0575, 0.2292],
    [-0.6408, 0.7103, -0.3183, 0.1758, -0.3034, -0.0668, 0.2493], [-0.6484, 0.7311, -0.3477, 0.1788, -0.3282, -0.0768, 0.2692], [-0.6566, 0.7517, -0.3778, 0.1818, -0.3528, -0.0874, 0.2889], [-0.6656, 0.772, -0.4085, 0.185, -0.3773, -0.0986, 0.3084],
    [-0.6754, 0.792, -0.44, 0.1882, -0.4016, -0.1104, 0.3276], [-0.686, 0.8117, -0.4721, 0.1915, -0.4258, -0.1228, 0.3467], [-0.6973, 0.831, -0.505, 0.1949, -0.4498, -0.1358, 0.3655], [-0.7095, 0.85, -0.5388, 0.1983, -0.4737, -0.1493, 0.384],
    [-0.7226, 0.8686, -0.5733, 0.2018, -0.4974, -0.1634, 0.4024], [-0.7365, 0.8867, -0.6088, 0.2053, -0.5209, -0.178, 0.4204], [-0.7514, 0.9044, -0.6451, 0.2089, -0.5443, -0.1931, 0.4383], [-0.7671, 0.9215, -0.6823, 0.2125, -0.5675, -0.2087, 0.4558],
    [-0.7837, 0.9381, -0.7205, 0.2161, -0.5905, -0.2248, 0.4731], [-0.8013, 0.9542, -0.7597, 0.2198, -0.6133, -0.2414, 0.4902], [-0.8199, 0.9696, -0.7999, 0.2235, -0.636, -0.2584, 0.507], [-0.8394, 0.9843, -0.8411, 0.2271, -0.6585, -0.2759, 0.5235],
    [-0.8598, 0.9984, -0.8833, 0.2309, -0.6809, -0.2937, 0.5398], [-0.8812, 1.0118, -0.9266, 0.2346, -0.703, -0.312, 0.5558], [-0.9036, 1.0243, -0.9708, 0.2383, -0.7251, -0.3307, 0.5716], [-0.9268, 1.0361, -1.016, 0.242, -0.747, -0.3498, 0.5871]
  ], G1E = [
    [0.1364, 0.1462, 0.1585], [0.1355, 0.1495, 0.1579], [0.1346, 0.1528, 0.1574], [0.1337, 0.1562, 0.157], [0.1329, 0.1595, 0.1567], [0.132, 0.163, 0.1565],
    [0.1312, 0.1664, 0.1564], [0.1305, 0.1698, 0.1564], [0.1297, 0.1733, 0.1566], [0.1289, 0.1768, 0.1568], [0.1282, 0.1802, 0.1572], [0.1275, 0.1837, 0.1577],
    [0.1269, 0.1872, 0.1583], [0.1262, 0.1907, 0.1591], [0.1256, 0.1941, 0.1599], [0.125, 0.1975, 0.1609], [0.1244, 0.2009, 0.1619], [0.1239, 0.2043, 0.1631],
    [0.1234, 0.2077, 0.1644], [0.1229, 0.211, 0.1658], [0.1225, 0.2143, 0.1674], [0.1221, 0.2175, 0.169], [0.1217, 0.2207, 0.1707], [0.1213, 0.2238, 0.1726],
    [0.121, 0.2269, 0.1745], [0.1207, 0.2299, 0.1766], [0.1205, 0.2329, 0.1787], [0.1203, 0.2358, 0.181], [0.1201, 0.2386, 0.1833], [0.12, 0.2413, 0.1857],
    [0.1199, 0.244, 0.1882], [0.1198, 0.2465, 0.1908], [0.1198, 0.249, 0.1935], [0.1198, 0.2514, 0.1962], [0.1198, 0.2537, 0.1991], [0.1199, 0.2559, 0.202],
    [0.1201, 0.258, 0.2049], [0.1202, 0.26, 0.2079], [0.1204, 0.2619, 0.211], [0.1207, 0.2636, 0.2141], [0.121, 0.2653, 0.2173], [0.1213, 0.2669, 0.2205],
    [0.1217, 0.2683, 0.2237], [0.1221, 0.2697, 0.227], [0.1225, 0.2709, 0.2303], [0.123, 0.272, 0.2337], [0.1236, 0.2731, 0.2371], [0.1241, 0.2739, 0.2404],
    [0.1247, 0.2747, 0.2438], [0.1254, 0.2754, 0.2472], [0.1261, 0.2759, 0.2507], [0.1268, 0.2763, 0.2541]
  ];
  function g1At(sw) {
    var x = (sw - G1SW) / 2, i = Math.max(0, Math.min(G1Q.length - 2, Math.floor(x))), k = Math.max(0, Math.min(1, x - i));
    return { q: G1Q[i].map(function (v, j) { return v + (G1Q[i + 1][j] - v) * k; }), e: G1E[i].map(function (v, j) { return v + (G1E[i + 1][j] - v) * k; }) };
  }
  var JN = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'];
  function setQ(v, q) { var o = {}; JN.forEach(function (n, i) { o[n] = q[i]; }); v.joints(o); }
  function lerpQ(a, b, k) { return a.map(function (x, i) { var d = b[i] - x; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI; return x + d * k; }); }

  E.scenes[5] = {
    title: '六个关节：最多几组解',
    dur: END,
    cues: [
      { at: 0, s: '真实的机械臂：6 个关节，手的位置和朝向都要对上' },
      { at: 5.95, s: '这台的逆向运动学有现成的公式（公开的代码 ur_kinematics）：3 处各有 2 个答案' },
      { at: 17.18, s: '底座：朝前伸 / 反着伸；肘：朝上 / 朝下；手腕：两种翻法' },
      { at: 27.48, s: '$2\\times 2\\times 2=8$：手停在同一个位姿，最多 **8 组**' },
      { at: 36.17, s: '最一般的 6 个关节：已有人证明，最多 **16 组**' },
      { at: 42.05, s: '关节比需要的多：解有**无穷多组**（平面里手的位置 2 个数，却有 3 个关节）' },
      { at: 53.23, s: '人形一条胳膊 7 个关节，手的位置和朝向一共 6 个数：也是**无穷多组**' },
      { at: 61.92, s: '手腕转到某些姿势（像第 6 集的万向锁）：也有**无穷多组**' },
      { at: 68.66, s: '6 个关节一般有限几组：这台最多 **8 组**，最一般最多 **16 组**；**肘上**、**肘下** = 开根号的正负两个' }
    ],
    build: function (K) {
      var s = K.sceneSvg('真实的机械臂有六个关节，手的位置和朝向都要对上；这台机械臂的逆向运动学有现成的公式，网上公开的代码 ur_kinematics 里就有：一步一步解下来，有三处各有两个答案——底座朝前伸或反着伸，肘朝上或朝下，手腕两种翻法，2 × 2 × 2 = 8，手停在同一个位姿最多摆出八种姿势；最一般的六关节机械臂已有人证明最多十六组解；关节比需要的多，解有无穷多组：平面里手的位置只有两个数却有三个关节，手不动前面的关节还能一起挪，人形机器人一条胳膊七个关节、手的位置和朝向一共六个数，也是这样；手腕转到某些姿势像第六集的万向锁，也有无穷多组解；所以六关节机械臂一般有限几组解：这台最多八组，最一般最多十六组，肘上肘下就是开平方的正负两个');

      /* ── 左：一台大的 UR5 ── */
      var bigG = E.el('g', {}, s);
      var big = window.Robot3D.ur5(bigG, 210, 320, 1.6, { az: 32, el: 16 }), BV = big.view;   // 再大有几组解会伸出视景框右边（check_layout 查过）
      BV.camera({ box: [0, 0, 440, 420] });
      setQ(BV, SOL[0]);
      var top = E.el('g', {}, s);
      var LINKS = ['shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link'];
      var jd = LINKS.map(function () { var c = K.paint(E.el('circle', { r: 7, 'stroke-width': 3 }, top), E.S1, E.HI); return c; });
      function frame(parent, w) { return [E.X, E.Y, E.Z].map(function (c) { return E.vec(parent, c, w || 4, { head: 11 }); }); }
      var bigF = frame(top, 4.5);
      var axesG = E.el('g', {}, top), ax1 = K.paint(E.el('path', { fill: 'none', 'stroke-width': 4, 'stroke-linecap': 'round' }, axesG), null, E.OMEGA), ax3 = K.paint(E.el('path', { fill: 'none', 'stroke-width': 4, 'stroke-linecap': 'round' }, axesG), null, E.OMEGA);
      var axLab = E.halo(E.txt(axesG, 0, 0, '两根轴平行', 20, E.OMEGA, 'end', 900));
      var solLab = E.halo(E.txt(top, 30, 40, '', 22, E.HI, 'start', 900));
      var GOAL = SOL[0];
      // 目标位姿：参考姿势下 tool0 的三根小箭头在画面里的位置（大 UR5）
      function toolArrows(v, arr, L) {
        var p = v.at('tool0');
        [[L, 0, 0], [0, L, 0], [0, 0, L]].forEach(function (d, k) { var q = v.at('tool0', d); arr[k].set(p[0], p[1], q[0], q[1]); });
      }

      /* ── 句 4：2 × 4 八台 ── */
      var gridG = E.el('g', {}, s), tiles = [];
      var heads = ['肘上 · 腕①', '肘上 · 腕②', '肘下 · 腕①', '肘下 · 腕②'];
      SOL.forEach(function (q, i) {
        var col = i % 4, row = Math.floor(i / 4), tx = 10 + col * 196, ty = 20 + row * 192;
        var g = E.el('g', {}, gridG);
        K.paint(E.el('rect', { x: tx + 2, y: ty + 2, width: 188, height: 158, rx: 12, 'stroke-width': 2 }, g), E.tint(row ? E.DOWN : E.UP, 6, 'var(--demo-surface)'), E.BORDER);
        var r = window.Robot3D.ur5(g, tx + 98, ty + 120, 0.8, { az: 32, el: 16 });
        r.view.camera({ box: [tx + 4, ty + 4, 184, 154] });
        setQ(r.view, q);
        var arr = frame(g, 3);
        E.txt(g, tx + 96, ty + 152, heads[col], 17, col < 2 ? E.UP : E.DOWN, 'middle', 900);
        tiles.push({ g: g, v: r.view, arr: arr, row: row });
      });
      var rowLab = [E.txt(gridG, 6, 16, '底座朝前伸', 18, E.INK2, 'start', 900), E.txt(gridG, 6, 206, '底座反着伸', 18, E.INK2, 'start', 900)];
      var eq8 = E.tex(gridG, 400, 402, '2\\times 2\\times 2=' + '\\textcolor{' + H.hi + '}{8}', 30, E.INK, 'middle', { w: 300, h: 50 });

      /* ── 句 6：平面三关节链（末端钉住，前面的关节一起挪）、G1 正面 ── */
      var chainG = E.el('g', {}, s), CU = 52, CO = [70, 300], LL = [1.7, 1.5, 1.3], TGC = [2.7, 1.6];
      function CP(x, y) { return [CO[0] + CU * x, CO[1] - CU * y]; }
      for (var gx = 0; gx <= 6; gx++) K.paint(E.el('line', { x1: CP(gx, 0)[0], y1: CP(0, -1)[1], x2: CP(gx, 0)[0], y2: CP(0, 4)[1], 'stroke-width': 1.2 }, chainG), null, E.GRID);
      for (var gy = -1; gy <= 4; gy++) K.paint(E.el('line', { x1: CP(0, 0)[0], y1: CP(0, gy)[1], x2: CP(6, 0)[0], y2: CP(0, gy)[1], 'stroke-width': 1.2 }, chainG), null, E.GRID);
      var links3 = [0, 1, 2].map(function (k) { return K.paint(E.el('path', { fill: 'none', 'stroke-width': 10, 'stroke-linecap': 'round' }, chainG), null, E.tint(E.INK2, 70, 'var(--demo-surface)')); });
      var jd3 = [0, 1, 2].map(function () { return K.paint(E.el('circle', { r: 8, 'stroke-width': 3 }, chainG), E.S1, E.HI); });
      var tg3 = E.target(chainG, CP(TGC[0], TGC[1])[0], CP(TGC[0], TGC[1])[1], 11, E.TGT);
      var ghosts3 = E.el('g', {}, chainG);
      // 句 7：G1 先整个出来，「一条胳膊」腿、骨盆、右臂淡掉，镜头推近到上半身（先淡掉再推近：没淡完就推，腿会被视景框切掉）；
      // 从右前上方斜着看（az 65、el 35）：这个方向肘走得最远、大臂和小臂也不会被压短（Python 按投影长度挑的）
      var g1G = E.el('g', {}, s);
      var G1CAM = { full: { o: [200, 165], s: 310 }, zoom: { o: [148, 383], s: 720 } };
      var GV = window.Robot3D.view(g1G, { robot: 'g1', origin: G1CAM.full.o, scale: G1CAM.full.s, az: 65, el: 35, box: [0, 0, 440, 420] });
      var g1T = E.el('g', {}, g1G);
      var g1Arc = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3, 'stroke-dasharray': '6 6' }, g1T), null, E.HI);
      var g1F = frame(g1T, 4);
      var g1D = G1J.map(function () { return K.paint(E.el('circle', { r: 6, 'stroke-width': 2.5 }, g1T), E.S1, E.HI); });
      var g1Lab = E.halo(E.txt(g1T, 0, 0, '7 个关节', 22, E.HI, 'start', 900));
      var g1HL = E.halo(E.txt(g1T, 0, 0, '手不动', 22, E.TGT, 'end', 900));

      /* ══ 右栏 ══ */
      var RK = E.rowKit(), Rg = E.el('g', {}, s), X = RX + 10;
      var r1g = E.el('g', {}, Rg);
      var a1 = RK.line(r1g, X, 40, [['真实的机械臂：', E.INK2], ['6 个关节', E.HI, 900]]);
      var a2 = RK.line(r1g, X, 78, [['手的', E.INK2], ['位置', E.TGT, 900], [' + ', E.INK2], ['朝向', E.OMEGA, 900], ['，都要对上', E.INK2]], 21);
      var a3 = RK.line(r1g, X, 134, [['现成的公式：', E.INK2], ['ur_kinematics', E.INK, 900]], 21);
      var a3b = RK.line(r1g, X, 166, [['（网上公开、谁都能看的代码）', E.MUT, 700]], 18);
      var a4 = RK.line(r1g, X, 206, [['一步一步解：', E.INK2], ['3 处', E.INK, 900], ['，各有 ', E.INK2], ['2 个答案', E.HI, 900]], 21);
      var a5 = RK.line(r1g, X, 260, [['① 底座：', E.INK, 900], ['朝前伸', E.INK2], [' / ', E.MUT], ['反着伸', E.INK2]], 21);
      var a6 = RK.line(r1g, X, 300, [['② 肘：', E.INK, 900], ['朝上', E.UP, 900], [' / ', E.MUT], ['朝下', E.DOWN, 900]], 21);
      var a7 = RK.line(r1g, X, 340, [['③ 手腕：', E.INK, 900], ['两种翻法', E.INK2]], 21);
      var r3g = E.el('g', {}, Rg);
      var b1 = RK.line(r3g, X, 44, [['最一般的 6 个关节：', E.INK2]], 21);
      var b2 = RK.line(r3g, X, 80, [['已有人证明，最多 ', E.INK2], ['16 组', E.HI, 900]], 23);
      var b2b = RK.line(r3g, X, 110, [['（Lee & Liang；Raghavan & Roth）', E.MUT, 700]], 17);
      var b3 = RK.line(r3g, X, 150, [['关节比需要的多 → ', E.INK2], ['无穷多组', E.HI, 900]], 21);
      var b4 = RK.line(r3g, X, 186, [['平面：手的位置只有 x、y 2 个数', E.INK2]], 19);
      var b5 = RK.line(r3g, X, 218, [['3 个关节：手不动，前面还能一起挪', E.INK2]], 19);
      var b6 = RK.line(r3g, X, 256, [['人形：一条胳膊 ', E.INK2], ['7 个关节', E.HI, 900]], 20);
      var b6b = RK.line(r3g, X, 288, [['手的位置 + 朝向 6 个数，多了 1 个', E.INK2]], 19);
      var b6c = RK.line(r3g, X, 314, [['→ 手不动，肘还能绕着转', E.HI, 900]], 19);
      var b7 = RK.line(r3g, X, 348, [['手腕某些姿势（像第 6 集的万向锁）', E.INK2]], 19);
      var b8 = RK.line(r3g, X, 380, [['→ 也有 ', E.INK2], ['无穷多组', E.HI, 900]], 21);
      var r4g = E.el('g', {}, Rg);
      var c1 = RK.line(r4g, X, 56, [['6 个关节：一般是', E.INK2], ['有限几组', E.INK, 900]], 21);
      var c2 = RK.line(r4g, X, 104, [['这台（UR5）最多 ', E.INK2], ['8 组', E.HI, 900]], 23);
      var c3 = RK.line(r4g, X, 146, [['最一般的最多 ', E.INK2], ['16 组', E.HI, 900]], 23);
      var c4 = RK.line(r4g, X, 214, [['肘上', E.UP, 900], ['、', E.INK2], ['肘下', E.DOWN, 900], ['：求肘那一步', E.INK2]], 21);
      var c5 = RK.line(r4g, X, 252, [['开根号', E.WARN, 900], ['的正负两个', E.INK2]], 21);

      var laid = false;
      function fr(st, a) { return E.fade(st, a, 0.3); }

      return {
        el: s,
        draw: function (st, t) {
          if (!laid && s.isConnected) { RK.layout(); laid = true; }
          var f = E.fade, b = E.bump;

          /* ── 大 UR5：参考姿势 → 底座反着 → 肘下 → 手腕翻过去；句 7 手腕奇异；句 8 依次摆出 8 组 ── */
          var q = SOL[0];
          var kB = f(st, T.back[0], T.back[1] - T.back[0]), kD = f(st, T.down[0], T.down[1] - T.down[0]), kF = f(st, T.flip[0], T.flip[1] - T.flip[0]);
          if (kB > 0) q = lerpQ(SOL[0], SOL[4], kB);
          if (kD > 0) q = lerpQ(SOL[4], SOL[6], kD);
          if (kF > 0) q = lerpQ(SOL[6], SOL[7], kF);
          // 姿势只在看不见时跳：句 4 等大 UR5 淡出完（grid + 0.4）再换回参考姿势（原先 out3 就换，淡出前一下跳回去）；
          // 句 8「所以」手腕两根轴先淡掉，再慢慢转回参考姿势（原先一下跳回去，淡出中的蓝色转轴跟着甩到竖直，闪一下蓝条）
          if (st >= T.grid + 0.4) q = SOL[0];
          var ws = f(st, T.pose7, 0.8) * (1 - f(st, T.sum + 0.3, 0.8));
          if (st >= T.wristS - 0.5 && st < T.s8) { q = SOL[0].slice(); q[4] = SOL[0][4] * (1 - ws); }
          var kk = -1;
          if (st >= T.s8) { kk = Math.min(7, Math.floor((st - T.s8) / 1.0)); q = SOL[kk]; }
          setQ(BV, q);
          // 大 UR5：句 4 让给八台，句 6 让给三关节链和 G1
          var bigOn = 1 - f(st, T.grid, 0.4);
          if (st >= T.out4) bigOn = f(st, T.out4 + 0.2, 0.4) * (1 - f(st, T.more - 0.4, 0.4));
          if (st >= T.wristS - 0.6) bigOn = f(st, T.wristS - 0.5, 0.4);
          E.op(bigG, bigOn);
          var pts = LINKS.map(function (l) { return BV.at(l); });
          jd.forEach(function (c, k) { c.setAttribute('cx', pts[k][0].toFixed(1)); c.setAttribute('cy', pts[k][1].toFixed(1)); E.op(c, fr(st, T.six) * (1 - f(st, T.ik, 0.3))); });
          toolArrows(BV, bigF, 0.12);
          var fOn = fr(st, T.ori) * bigOn;
          bigF.forEach(function (a) { E.op(a.g, fOn); a.line.style.strokeWidth = 4.5 + 2 * Math.max(b(st, T.front, 0.8), b(st, T.back[1] + 0.2, 0.8), b(st, T.down[1] + 0.2, 0.8), b(st, T.flip[1] + 0.2, 0.8)); });
          // 句 7：手腕两根转轴（wrist_1、wrist_3 的 y 轴）
          var w1a = BV.at('wrist_1_link', [0, -0.16, 0]), w1b = BV.at('wrist_1_link', [0, 0.16, 0]), w3a = BV.at('wrist_3_link', [0, -0.16, 0]), w3b = BV.at('wrist_3_link', [0, 0.16, 0]);
          ax1.setAttribute('d', 'M ' + w1a[0].toFixed(1) + ' ' + w1a[1].toFixed(1) + ' L ' + w1b[0].toFixed(1) + ' ' + w1b[1].toFixed(1));
          ax3.setAttribute('d', 'M ' + w3a[0].toFixed(1) + ' ' + w3a[1].toFixed(1) + ' L ' + w3b[0].toFixed(1) + ' ' + w3b[1].toFixed(1));
          axLab.setAttribute('x', (Math.min(w1a[0], w3a[0], w1b[0], w3b[0]) - 10).toFixed(1)); axLab.setAttribute('y', (Math.max(w1a[1], w3a[1], w1b[1], w3b[1]) + 28).toFixed(1));   // 轴的左下：右边是右栏
          E.op(axesG, fr(st, T.gimbal - 0.6) * (1 - f(st, T.sum, 0.3)));
          solLab.textContent = kk >= 0 ? '第 ' + (kk + 1) + ' / 8 组' : '';
          E.op(solLab, kk >= 0 ? 1 : 0);

          /* ── 句 4：八台 ── */
          var gOn = f(st, T.grid, 0.4) * (1 - f(st, T.out4, 0.4));
          E.op(gridG, gOn);
          tiles.forEach(function (tl, i) {
            // 「两种」：先亮第 1 列（上下两排 = 底座两种），「两种」再亮第 3 列（肘两种），「再两种」亮第 2、4 列（手腕两种）
            var col = i % 4, at = [T.g2[0], T.g2[2], T.g2[1], T.g2[2]][col];
            E.op(tl.g, f(st, Math.min(at, T.eight), 0.3));
            toolArrows(tl.v, tl.arr, 0.12);
            tl.arr.forEach(function (a) { a.line.style.strokeWidth = 3 + 2.5 * b(st, T.same, 1.0); });
          });
          E.op(eq8, fr(st, T.eight));

          /* ── 句 6：三关节链、G1 ── */
          var chOn = f(st, T.planar, 0.4) * (1 - f(st, T.human, 0.4));
          E.op(chainG, chOn);
          if (chOn > 0) {
            var mv = f(st, T.slide, 0.3) * Math.sin((st - T.slide) * 1.6) * 0.32;
            var th1 = 1.15 + mv, P1 = [LL[0] * Math.cos(th1), LL[0] * Math.sin(th1)];
            var sol = E.ik2(TGC[0] - P1[0], TGC[1] - P1[1], true, LL[1], LL[2]) || [0, 0, [0, 0]];
            var P2 = [P1[0] + sol[2][0], P1[1] + sol[2][1]];
            var pp = [[0, 0], P1, P2, TGC];
            links3.forEach(function (ln, k) { var a = CP(pp[k][0], pp[k][1]), b2 = CP(pp[k + 1][0], pp[k + 1][1]); ln.setAttribute('d', 'M ' + a[0].toFixed(1) + ' ' + a[1].toFixed(1) + ' L ' + b2[0].toFixed(1) + ' ' + b2[1].toFixed(1)); });
            jd3.forEach(function (c, k) { var a = CP(pp[k][0], pp[k][1]); c.setAttribute('cx', a[0].toFixed(1)); c.setAttribute('cy', a[1].toFixed(1)); E.op(c, fr(st, T.three3)); });
            E.pop(tg3, CP(TGC[0], TGC[1])[0], CP(TGC[0], TGC[1])[1], 1 + 0 * b(st, T.still, 0.8), 1);
            tg3.setAttribute('transform', 'translate(' + CP(TGC[0], TGC[1])[0] + ' ' + CP(TGC[0], TGC[1])[1] + ') scale(' + (1 + 0.5 * b(st, T.still, 0.9)) + ') translate(' + (-CP(TGC[0], TGC[1])[0]) + ' ' + (-CP(TGC[0], TGC[1])[1]) + ')');
          }
          /* ── 句 7：G1 的左臂。「一共六个数」起手的位姿不动，肘绕「肩—手腕」连线转一个来回：0 → +61° → −37° → 0 ── */
          var g1On = f(st, T.human, 0.4) * (1 - f(st, T.wristS - 0.5, 0.4));
          E.op(g1G, g1On);
          if (g1On > 0) {
            var hz = f(st, T.arm1, 0.3), zk = f(st, T.arm1 + 0.3, 0.8), C0 = G1CAM.full, C1 = G1CAM.zoom;
            GV.camera({ origin: [C0.o[0] + (C1.o[0] - C0.o[0]) * zk, C0.o[1] + (C1.o[1] - C0.o[1]) * zk], scale: C0.s + (C1.s - C0.s) * zk });
            var sw = 0;
            if (st > T.swing) { var wv = 2 * Math.PI * Math.min(1, (st - T.swing) / (T.wristS - 0.6 - T.swing)); sw = 50 * Math.sin(wv) + 10 * (1 - Math.cos(wv)); }
            var ga = g1At(sw), jo = {};
            G1J.forEach(function (n, i) { jo[n] = ga.q[i]; });
            GV.joints(jo);
            GV.clearTint();
            GV.tint(['leftLeg', 'rightLeg', 'pelvis', 'rightArm'], null, 1, 1 - hz);
            GV.tint('leftArm', E.HI, 0.5 * f(st, T.seven, 0.3));
            g1D.forEach(function (c, i) { var p = GV.at(G1J[i].replace(/_joint$/, '_link')); c.setAttribute('cx', p[0].toFixed(1)); c.setAttribute('cy', p[1].toFixed(1)); E.op(c, fr(st, T.seven)); });
            var hp = GV.at('left_rubber_hand');
            [[0.07, 0, 0], [0, 0.07, 0], [0, 0, 0.07]].forEach(function (d, k) { var q = GV.at('left_rubber_hand', d); g1F[k].set(hp[0], hp[1], q[0], q[1]); E.op(g1F[k].g, fr(st, T.six6)); });
            g1Arc.setAttribute('d', G1E.map(function (e, i) { var p = GV.project(e); return (i ? 'L ' : 'M ') + p[0].toFixed(1) + ' ' + p[1].toFixed(1); }).join(' '));
            E.op(g1Arc, 0.85 * fr(st, T.swing));
            var sp = GV.at('left_shoulder_roll_link');
            g1Lab.setAttribute('x', (sp[0] + 22).toFixed(1)); g1Lab.setAttribute('y', (sp[1] - 18).toFixed(1));
            E.op(g1Lab, fr(st, T.seven));
            g1HL.setAttribute('x', (hp[0] - 18).toFixed(1)); g1HL.setAttribute('y', (hp[1] + 36).toFixed(1));
            E.op(g1HL, fr(st, T.swing));
          }

          /* ── 右栏 ── */
          E.op(r1g, 1 - f(st, T.grid - 0.2, 0.3));
          E.op(a1.g, fr(st, T.real)); E.op(a1.parts[1], fr(st, T.six)); E.op(a2.g, fr(st, T.ori));
          E.op(a3.g, fr(st, T.formula)); E.op(a3.parts[1], fr(st, T.code)); E.op(a3b.g, fr(st, T.code)); E.op(a4.g, fr(st, T.three)); E.op(a4.parts[3], fr(st, T.two));
          E.op(a5.g, fr(st, T.base)); E.op(a5.parts[3], fr(st, T.back[0])); E.op(a6.g, fr(st, T.elbow)); E.op(a6.parts[3], fr(st, T.down[0])); E.op(a7.g, fr(st, T.wrist));
          E.op(r3g, f(st, T.gen - 0.2, 0.3) * (1 - f(st, T.sum - 0.2, 0.3)));
          E.op(b1.g, fr(st, T.gen)); E.op(b2.g, fr(st, T.proved)); E.op(b2.parts[1], fr(st, T.s16)); E.op(b2b.g, fr(st, T.proved + 0.3));
          E.op(b3.g, fr(st, T.more)); E.op(b3.parts[1], fr(st, T.inf)); E.op(b4.g, fr(st, T.handXY)); E.op(b5.g, fr(st, T.three3));
          E.op(b6.g, fr(st, T.human)); E.op(b6.parts[1], fr(st, T.seven)); E.op(b6b.g, fr(st, T.six6)); E.op(b6c.g, fr(st, T.extra)); E.op(b7.g, fr(st, T.wristS)); E.op(b8.g, fr(st, T.inf2));
          E.op(r4g, f(st, T.sum, 0.3));
          E.op(c1.g, fr(st, T.sum)); E.op(c1.parts[1], fr(st, T.finite)); E.op(c2.g, fr(st, T.s8)); E.op(c3.g, fr(st, T.s16b - 0.4));
          E.op(c4.g, fr(st, T.elb)); E.op(c5.g, fr(st, T.sqrt));
        }
      };
    }
  };
})();
