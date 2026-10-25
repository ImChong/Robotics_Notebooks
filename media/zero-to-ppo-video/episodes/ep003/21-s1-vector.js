// 003 第 1 幕：位置是向量。左边 UR5 正侧视（肩关节当原点，aim 让手正好落在格点 (3, 2)），右边是说明区：
// ① 「手」时手上一圈亮黄 +「?」；「右上方」冒出含糊的「在右上方？」，「电脑」出一台显示器，「听不懂」打 ×；
// ② 「原点」在肩关节点一个原点，「肩膀」「关节」光圈鼓两下；「往右是 x」「往上是 y」各画一根轴，网格跟着淡入，两轴画完补一个直角记号；
// ③ 「往右三格」虚线沿 x 轴一格一格走，计数 1 → 2 → 3；「往上两格」沿 x = 3 往上走到手，计数 1 → 2；
//    「竖着排成一列」两个计数飞进右边的列向量 [3; 2]（KaTeX，accent），「向量」时下面弹出标签；
// ④ 「箭头」从原点画到手；「方向」画角度弧，「长短」画一条和箭头平行的长度标注；
// ⑤ 「三维」右边换成三维小坐标架（x 右、y 上、z 斜着朝向观众，右手系）+ 一个点的箭头 + 列向量 [x; y; z]，「三个数」时鼓一下；
//    「每个关节」时 UR5 的肘、三个腕关节依次亮起，各有一支从原点出发的淡箭头。
// 时刻取自配音的逐词时间（../ep003.py 的 [from, to] 等于每句配音实长，分镜时间 ≈ 念到的时刻）。
(function () {
  var E = window.EP003, K = E.K;
  var OX = 190, OY = 300, A1 = 25;                         // 肩关节（原点）在画面里的位置；大臂离竖直 25°
  var VX = 650, VY = 205, VS = 48;                         // 右边列向量 [3; 2] 的位置与字号
  var CNT_X = [15.35, 16.25], CNT_Y = [16.34, 16.95];      // 「往右三格」「往上两格」计数虚线的起止
  var FLY = [18.9, 19.65];                                 // 「竖着排成一列」两个数飞进列向量
  var ARROW = [23.55, 25.85], ANG = 27.2, LEN = 27.7;      // 箭头从原点长到手；「方向」「长短」
  var D3 = 31.92, Z_AT = 34.0, V3 = 34.75, THREE = 35.45, JOINTS = 36.77;
  var JOINT_LINKS = ['forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link'];

  function bump(st, a, d) { var u = K.seg(st, a, a + d); return u > 0 && u < 1 ? Math.sin(Math.PI * u) : 0; }
  function lerp(a, b, p) { return a + (b - a) * p; }

  E.scenes[0] = {
    title: '位置：一列数，也是一支箭头',
    dur: 40.23,
    cues: [
      { at: 0, s: '机械臂的手在哪？要用**数**说清楚' },
      { at: 7.03, s: '先选原点和两根轴：往右 x，往上 y' },
      { at: 14.83, s: '往右 3、往上 2，竖着排成一列：**向量** $\\begin{bmatrix}3\\\\2\\end{bmatrix}$' },
      { at: 22.2, s: '也是一支箭头：方向 + 长短 = 位置' },
      { at: 31.01, s: '三维：再加一根 z 轴，向量就是三个数' }
    ],
    build: function (K) {
      var s = K.sceneSvg('位置是向量：以机械臂的肩关节为原点画 x、y 两根轴，手在往右 3 格、往上 2 格，两个数竖着排成一列就是向量 [3; 2]，也可以画成从原点指到手的一支箭头；三维再加一根 z 轴，就是三个数');

      /* ── 网格（在机械臂下面）+ UR5 ── */
      var under = E.el('g', {}, s);
      var arm = E.armAt(s, OX, OY, 2.2, 280);
      var U = arm.aim(3, 2, A1);                             // 一格多少 px：手正好落在格点 (3, 2)
      var pl = E.plane(under, { ox: OX, oy: OY, u: U, x0: -1, x1: 4, y0: -1, y1: 3, axes: false });
      var P = pl.P, H = P(3, 2), box = pl.box;
      var axX = E.vec(under, E.INK2, 2.5, { head: 12 }), axY = E.vec(under, E.INK2, 2.5, { head: 12 });
      var labX = E.txt(under, box[2] + 22, OY + 8, 'x', 24, E.INK2, 'start', 700);
      var labY = E.txt(under, OX, box[1] - 24, 'y', 24, E.INK2, 'middle', 700);
      var right = E.el('path', { d: 'M ' + (OX - 14) + ' ' + OY + ' V ' + (OY - 14) + ' H ' + OX, fill: 'none', 'stroke-width': 2 }, under);
      K.paint(right, null, E.INK2);

      /* ── 叠在机械臂上的标注 ── */
      var over = E.el('g', {}, s);
      // 计数虚线：先沿 x 轴走 3 格，再沿 x = 3 往上走 2 格；每过一个格点落一个小点
      var cntH = E.el('path', { d: 'M ' + OX + ' ' + OY + ' H ' + P(3, 0)[0], fill: 'none', 'stroke-width': 4, 'stroke-dasharray': '9 7', 'stroke-linecap': 'round' }, over);
      var cntV = E.el('path', { d: 'M ' + P(3, 0)[0] + ' ' + OY + ' V ' + H[1], fill: 'none', 'stroke-width': 4, 'stroke-dasharray': '9 7', 'stroke-linecap': 'round' }, over);
      K.paint(cntH, null, E.ACC); K.paint(cntV, null, E.ACC);
      var ticks = [[1, 0], [2, 0], [3, 0], [3, 1], [3, 2]].map(function (q) { return K.paint(E.el('circle', { cx: P(q[0], q[1])[0], cy: P(q[0], q[1])[1], r: 5.5 }, over), E.ACC); });
      // 两个计数（KaTeX 同款字体、按字的中心定位，飞进列向量时和公式里的数叠得上）
      function digit() { var n = E.txt(over, 0, 0, '', 30, E.ACC, 'middle'); n.style.fontFamily = 'KaTeX_Main, serif'; n.setAttribute('dominant-baseline', 'central'); return n; }
      var nX = digit(), nY = digit();
      var SX = [P(1.5, 0)[0], OY + 28], SY = [P(3, 1)[0] + 26, P(3, 1)[1]];   // 计数停在两段虚线的中点旁边

      // 原点、肩关节光圈、手的光圈与「?」
      var org = K.paint(E.el('circle', { cx: OX, cy: OY, r: 7, 'stroke-width': 2.5 }, over), E.HI, E.S1);
      var orgLab = E.txt(over, OX - 34, OY + 36, '原点', 22, E.HI, 'end', 700);
      var shRing = K.paint(E.el('circle', { cx: OX, cy: OY, r: 20, fill: 'none', 'stroke-width': 3 }, over), null, E.ACC);
      var hRing = K.paint(E.el('circle', { cx: H[0], cy: H[1], r: 18, fill: 'none', 'stroke-width': 3 }, over), null, E.HI);
      var ask = E.txt(over, H[0] + 30, H[1] - 24, '?', 40, E.HI, 'middle', 900);

      // 主箭头、角度弧、长度标注
      var main = E.vec(over, E.ACC, 5, { head: 18 });
      var th = Math.atan2(2, 3), R = 54;
      var arc = E.el('path', { d: 'M ' + (OX + R) + ' ' + OY + ' A ' + R + ' ' + R + ' 0 0 0 ' + (OX + R * Math.cos(th)).toFixed(1) + ' ' + (OY - R * Math.sin(th)).toFixed(1), fill: 'none', 'stroke-width': 3, 'stroke-linecap': 'round' }, over);
      K.paint(arc, null, E.HI);
      var arcLab = E.txt(over, OX + R - 4, OY + 28, '方向', 20, E.HI, 'start', 700);   // 放在 x 轴下面，不和箭头下方的长度标注挤
      var nx = 2 / Math.sqrt(13), ny = 3 / Math.sqrt(13), off = 18;   // 箭头往右下方的法向
      var lenG = E.el('g', {}, over);
      var lenLine = E.el('path', { d: 'M ' + (OX + nx * off).toFixed(1) + ' ' + (OY + ny * off).toFixed(1) + ' L ' + (H[0] + nx * off).toFixed(1) + ' ' + (H[1] + ny * off).toFixed(1), fill: 'none', 'stroke-width': 2.5, 'stroke-linecap': 'round' }, lenG);
      K.paint(lenLine, null, E.HI);
      [[OX, OY], H].forEach(function (p) {
        K.paint(E.el('path', { d: 'M ' + (p[0] + nx * (off - 8)).toFixed(1) + ' ' + (p[1] + ny * (off - 8)).toFixed(1) + ' L ' + (p[0] + nx * (off + 8)).toFixed(1) + ' ' + (p[1] + ny * (off + 8)).toFixed(1), fill: 'none', 'stroke-width': 2.5, 'stroke-linecap': 'round' }, lenG), null, E.HI);
      });
      var lm = [lerp(OX, H[0], 0.66) + nx * 44, lerp(OY, H[1], 0.66) + ny * 44];
      var lenLab = E.txt(lenG, lm[0], lm[1] + 7, '长短', 20, E.HI, 'middle', 700);

      // 各关节：从原点出发的淡箭头 + 亮点（「每个关节」时依次亮起）
      var jArrows = JOINT_LINKS.map(function () { return E.vec(over, E.ACC, 2.5, { head: 11 }); });
      var jDots = JOINT_LINKS.map(function () { return K.paint(E.el('circle', { r: 7, 'stroke-width': 2 }, over), E.HI, E.S1); });

      /* ── 右边：含糊的说法 + 显示器 ── */
      var vague = E.tag(s, 548, 62, 200, 46, '在右上方？', E.INK2, { size: 24, ink: E.INK, weight: 700 });
      var vagueX = E.cross(s, 648, 85, 30, E.BAD, 5);
      var pc = E.el('g', {}, s);
      K.paint(E.el('rect', { x: 588, y: 160, width: 124, height: 84, rx: 10, 'stroke-width': 3 }, pc), E.S2, E.INK2);
      K.paint(E.el('path', { d: 'M 650 244 V 266 M 618 268 H 682', fill: 'none', 'stroke-width': 4, 'stroke-linecap': 'round' }, pc), null, E.INK2);
      var pcAsk = E.txt(pc, 650, 216, '?', 40, E.BAD, 'middle', 900);
      var pcLab = E.txt(pc, 650, 306, '电脑听不懂', 22, E.BAD, 'middle', 700);

      /* ── 右边：列向量 [3; 2] ── */
      var vec = E.tex(s, VX, VY, E.col([3, 2], E.HEX.acc), VS, E.HEX.acc);
      // 两个数在 KaTeX 里的中心（先按 bmatrix 两行相距 1.2em 估，挂到页面后从 DOM 量准，换算成分镜坐标）
      var T3 = [VX, VY - 0.34 * VS - 0.6 * VS], T2 = [VX, VY - 0.34 * VS + 0.6 * VS];
      function measure() {
        var ctm = s.getScreenCTM(); if (!ctm) return;
        var inv = ctm.inverse(), walk = document.createTreeWalker(vec, NodeFilter.SHOW_TEXT), range = document.createRange(), pt = s.createSVGPoint();
        for (var n = walk.nextNode(); n; n = walk.nextNode()) {
          if (n.parentElement.closest('.katex-mathml') || (n.nodeValue !== '3' && n.nodeValue !== '2')) continue;
          range.selectNodeContents(n);
          var r = range.getBoundingClientRect(); if (!r.width) continue;
          pt.x = r.left + r.width / 2; pt.y = r.top + r.height / 2;
          var q = pt.matrixTransform(inv);
          if (n.nodeValue === '3') T3 = [q.x, q.y]; else T2 = [q.x, q.y];
        }
      }
      var hintR = E.txt(s, VX + 46, T3[1] + 7, '往右', 20, E.INK2, 'start'), hintU = E.txt(s, VX + 46, T2[1] + 7, '往上', 20, E.INK2, 'start');
      var vecTag = E.tag(s, VX - 50, 286, 100, 40, '向量', E.ACC, { size: 22, ink: E.ACC, weight: 900 });

      /* ── 右边：三维小坐标架 ── */
      var F = [572, 282], ZD = [-0.62, 0.62];
      var g3 = E.el('g', {}, s);
      var a3x = E.vec(g3, E.INK2, 2.5, { head: 12 }), a3y = E.vec(g3, E.INK2, 2.5, { head: 12 }), a3z = E.vec(g3, E.INK2, 2.5, { head: 12 });
      var l3x = E.txt(g3, F[0] + 108, F[1] + 8, 'x', 22, E.INK2, 'start', 700);
      var l3y = E.txt(g3, F[0], F[1] - 116, 'y', 22, E.INK2, 'middle', 700);
      var l3z = E.txt(g3, F[0] + 70 * ZD[0] - 10, F[1] + 70 * ZD[1] + 22, 'z（朝向你）', 20, E.INK2, 'middle', 700);
      var Q = [F[0] + 72 + 46 * ZD[0], F[1] - 62 + 46 * ZD[1]];
      var path3 = E.el('path', { d: 'M ' + F[0] + ' ' + F[1] + ' h 72 l ' + (46 * ZD[0]).toFixed(1) + ' ' + (46 * ZD[1]).toFixed(1) + ' v -62', fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '6 5', 'stroke-linecap': 'round' }, g3);
      K.paint(path3, null, E.ACC);
      var arr3 = E.vec(g3, E.ACC, 4, { head: 14 });
      var dot3 = K.paint(E.el('circle', { cx: Q[0], cy: Q[1], r: 5.5 }, g3), E.ACC);
      var vec3G = E.el('g', {}, s);
      var vec3 = E.tex(vec3G, 742, 210, E.col(['x', 'y', 'z'], E.HEX.acc), 38, E.HEX.acc);
      var vec3Lab = E.txt(vec3G, 742, 304, '三个数', 20, E.ACC, 'middle', 700);

      return {
        el: s,
        draw: function (st, t) {
          var fade = E.fade;

          /* ① 机械臂一开头就淡淡摆着，1.9「机械臂」亮起；2.58「手」光圈鼓一下，2.95「?」；7.3 起「?」退场 */
          E.op(arm.g, (fade(st, 0, 0.6) * (0.4 + 0.6 * fade(st, 1.85, 0.4))).toFixed(3));
          var ringOn = fade(st, 2.55, 0.3) * (1 - fade(st, 7.2, 0.4));
          var pulse = Math.max(bump(st, 2.55, 0.6), bump(st, 25.7, 0.6), bump(st, 29.2, 0.6));
          hRing.setAttribute('r', (18 + 10 * pulse).toFixed(2));
          E.op(hRing, Math.max(ringOn, 0.9 * pulse).toFixed(3));
          E.pop(ask, H[0] + 30, H[1] - 36, fade(st, 2.95, 0.3) * (1 - fade(st, 7.3, 0.4)), 0.6);
          var gone = 1 - fade(st, 6.9, 0.45);
          E.pop(vague.g, 648, 85, fade(st, 4.2, 0.3) * gone, 0.8);
          E.pop(pc, 650, 230, fade(st, 5.0, 0.3) * gone, 0.85);
          E.op(pcAsk, fade(st, 5.4, 0.2));
          E.op(pcLab, fade(st, 5.6, 0.3));
          E.pop(vagueX, 648, 85, fade(st, 5.76, 0.25) * gone, 0.4);

          /* ② 7.76「原点」原点 + 标签；8.75「肩膀」、9.29「关节」光圈各鼓一下；11.96 x 轴、13.0 y 轴；网格随轴淡入；13.6 直角记号 */
          E.pop(org, OX, OY, fade(st, 7.7, 0.3), 0.3);
          E.op(orgLab, fade(st, 7.8, 0.3) * (1 - 0.6 * fade(st, 14.8, 0.4)));
          var sp = Math.max(bump(st, 8.7, 0.6), bump(st, 9.25, 0.6));
          shRing.setAttribute('r', (16 + 14 * sp).toFixed(2));
          E.op(shRing, sp.toFixed(3));
          var px = fade(st, 11.95, 0.6), py = fade(st, 13.0, 0.5);
          if (px > 0) axX.set(OX - (OX - box[0]) * px, OY, OX + (box[2] + 14 - OX) * px, OY); else axX.set(OX, OY, OX, OY);
          if (py > 0) axY.set(OX, OY + (box[3] - OY) * py, OX, OY - (OY - box[1] + 14) * py); else axY.set(OX, OY, OX, OY);
          E.op(labX, fade(st, 12.45, 0.3)); E.op(labY, fade(st, 13.5, 0.3));
          E.op(pl.grid, (0.2 * px + 0.8 * py).toFixed(3));
          E.op(right, fade(st, 13.6, 0.4) * (1 - fade(st, 22.2, 0.4)));

          /* ③ 15.35 起沿 x 轴数 3 格、16.34 起往上数 2 格；18.3「两个数」鼓一下；18.9「竖着排成一列」飞进列向量；
                20.7「向量」标签；22.4 起计数虚线退淡 */
          var ph = K.seg(st, CNT_X[0], CNT_X[1]), pv = K.seg(st, CNT_Y[0], CNT_Y[1]);
          E.drawOn(cntH, ph); E.drawOn(cntV, pv);
          var dim = 1 - 0.75 * fade(st, 22.3, 0.5);
          cntH.style.opacity = ph > 0 ? dim : 0; cntV.style.opacity = pv > 0 ? dim : 0;
          ticks.forEach(function (d, k) {
            var at = k < 3 ? CNT_X[0] + (CNT_X[1] - CNT_X[0]) * (k + 1) / 3 : CNT_Y[0] + (CNT_Y[1] - CNT_Y[0]) * (k - 2) / 2;
            E.pop(d, +d.getAttribute('cx'), +d.getAttribute('cy'), fade(st, at - 0.05, 0.15), 0.3);
            if (st >= at) E.op(d, dim * (1 - fade(st, 31.0, 0.4)));
          });
          var cx = Math.min(3, Math.floor(ph * 3 + 1e-6)), cy = Math.min(2, Math.floor(pv * 2 + 1e-6));
          nX.textContent = cx ? String(cx) : ''; nY.textContent = cy ? String(cy) : '';
          var fly = fade(st, FLY[0], FLY[1] - FLY[0]), gl = Math.max(bump(st, 18.25, 0.5), 0);
          if (st > FLY[0] - 0.5 && st < FLY[1] + 0.5) {
            measure();
            hintR.setAttribute('y', (T3[1] + 7).toFixed(1)); hintU.setAttribute('y', (T2[1] + 7).toFixed(1));
          }
          nX.setAttribute('x', lerp(SX[0], T3[0], fly).toFixed(1)); nX.setAttribute('y', lerp(SX[1], T3[1], fly).toFixed(1));
          nY.setAttribute('x', lerp(SY[0], T2[0], fly).toFixed(1)); nY.setAttribute('y', lerp(SY[1], T2[1], fly).toFixed(1));
          [nX, nY].forEach(function (n) {
            n.style.fontSize = (30 + (VS - 30) * fly + 6 * gl).toFixed(1) + 'px';
            E.op(n, 1 - fade(st, FLY[1], 0.15));
          });
          var vOn = fade(st, FLY[1] - 0.05, 0.15) * (1 - fade(st, D3, 0.4));
          E.op(vec, vOn.toFixed(3));
          E.op(hintR, fade(st, 19.8, 0.3) * (1 - fade(st, D3, 0.4))); E.op(hintU, fade(st, 19.95, 0.3) * (1 - fade(st, D3, 0.4)));
          E.pop(vecTag.g, VX, 306, fade(st, 20.65, 0.3) * (1 - fade(st, D3, 0.4)), 0.7);

          /* ④ 23.55 起箭头从原点长到手（25.85「手上」到）；27.2「方向」角度弧；27.7「长短」长度标注；31.0 起两样标注退场 */
          var pa = fade(st, ARROW[0], ARROW[1] - ARROW[0]);
          if (pa > 0) main.set(OX, OY, lerp(OX, H[0], pa), lerp(OY, H[1], pa)); else main.set(OX, OY, OX, OY);
          var annot = 1 - fade(st, 31.0, 0.4);
          E.drawOn(arc, fade(st, ANG, 0.4)); arc.style.opacity = st >= ANG ? annot : 0;
          E.op(arcLab, fade(st, ANG + 0.1, 0.3) * annot);
          E.drawOn(lenLine, fade(st, LEN, 0.5));
          E.op(lenG, (st >= LEN ? annot : 0).toFixed(3));
          lenG.querySelectorAll('path').forEach(function (n, k) { if (k) E.op(n, fade(st, LEN + 0.45, 0.2)); });
          E.op(lenLab, fade(st, LEN + 0.2, 0.3));

          /* ⑤ 31.92「三维」右边换三维坐标架：x、y 先画，34.0「z 轴」再画 z；34.75「向量」点、虚线、箭头、[x; y; z]；35.45「三个数」鼓一下；
                36.77「每个关节」起肘、三个腕关节依次亮起，各有一支从原点出发的淡箭头 */
          var p3 = fade(st, D3 + 0.1, 0.5), pz = fade(st, Z_AT, 0.45);
          if (p3 > 0) { a3x.set(F[0], F[1], F[0] + 100 * p3, F[1]); a3y.set(F[0], F[1], F[0], F[1] - 108 * p3); } else { a3x.set(F[0], F[1], F[0], F[1]); a3y.set(F[0], F[1], F[0], F[1]); }
          if (pz > 0) a3z.set(F[0], F[1], F[0] + 76 * ZD[0] * pz, F[1] + 76 * ZD[1] * pz); else a3z.set(F[0], F[1], F[0], F[1]);
          E.op(l3x, p3); E.op(l3y, p3); E.op(l3z, fade(st, Z_AT + 0.3, 0.3));
          var pq = fade(st, V3, 0.7);
          E.drawOn(path3, pq);
          if (st >= V3 + 0.5) arr3.set(F[0], F[1], lerp(F[0], Q[0], fade(st, V3 + 0.5, 0.4)), lerp(F[1], Q[1], fade(st, V3 + 0.5, 0.4))); else arr3.set(F[0], F[1], F[0], F[1]);
          E.op(dot3, fade(st, V3 + 0.6, 0.2));
          E.pop(vec3G, 742, 240, fade(st, V3 + 0.1, 0.35), 0.8);
          E.scaleAt(vec3, 742, 196, 1 + 0.12 * bump(st, THREE, 0.5));
          E.op(vec3Lab, fade(st, THREE, 0.3));
          E.op(g3, (st >= D3 ? 1 : 0));

          var joints = st >= JOINTS - 0.5 ? JOINT_LINKS.map(function (l) { return arm.view.at(l); }) : null;
          JOINT_LINKS.forEach(function (l, k) {
            var q = fade(st, JOINTS + 0.28 * k, 0.3);
            if (joints && q > 0) {
              jArrows[k].set(OX, OY, lerp(OX, joints[k][0], q), lerp(OY, joints[k][1], q));
              jDots[k].setAttribute('cx', joints[k][0].toFixed(1)); jDots[k].setAttribute('cy', joints[k][1].toFixed(1));
            } else jArrows[k].set(OX, OY, OX, OY);
            E.op(jArrows[k].g, (0.55 * q).toFixed(3));
            E.pop(jDots[k], joints ? joints[k][0] : 0, joints ? joints[k][1] : 0, q, 0.4);
          });
        }
      };
    }
  };
})();
