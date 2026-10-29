// 008 第 5 幕：机器人里到处都是优化。顶上四个页签（逆运动学 / 最优控制 / 全身控制 / 强化学习），念到哪个，下面就换成哪个：
// 左边一台 G1（Robot Description Gallery Online 收录的 URDF，由 ../../robot3d.js 渲染）演示，右边三行：变量（绿）、目标（紫）、约束（橙）。
// 句 1「到处」四个页签依次弹出；
// 句 2 逆运动学（第 12 集）：侧视 G1，「变量」肩、肘两个关节点亮，「手离」胸前出目标点、手到目标点画虚线并写距离，
//    手臂按阻尼最小二乘一小步一小步挪过去（数值逆运动学，用的就是 URDF 的正运动学），距离降到 0；「约束」肘关节画出 URDF 的限位
//    −60°…120°（lower −1.0472、upper 2.0944 rad）；
// 句 3 最优控制（第四季，LQR、MPC）：侧视 G1 隔一会儿被从背后推一下、往前一歪再站回来，「电机」脚踝亮起，「偏离」画出歪的角度、
//    左上蓝条「偏离²」跟着变，「用了多大的力」琥珀条「用力²」跟着变（同迈步子的两项扣分的颜色），「迈步子」右栏第三行出 (4 − u)² + u²；
// 句 4 全身控制（第五季）：正面 G1，「加速度」「力矩」29 个关节点依次点亮，「脚底」两只脚下出地面往上推的力，「任务」一只手往旁边抬、够到目标点、
//    骨盆下垂一条虚线落在两脚之间（站稳），「摩擦」脚下出摩擦锥，「力矩上限」关节点描一圈橙；
// 句 5 强化学习（第六季）：侧视 G1 原地走，「参数」右上出一排参数小条，「奖励」下面画奖励曲线往上爬，「一小步」参数条一下一下变、曲线跟着长。
// 时刻取自配音的逐词时间（分镜时间 = 念到这个词的时刻），见 ../ep008.py。
(function () {
  var E = window.EP008, K = E.K, H = E.HEX, R3 = function () { return window.Robot3D; };
  var D = Math.PI / 180;
  var CARDS = [
    { name: '逆运动学', sub: '', ep: '第 12 集', on: 3.84, rows: [
      ['变量', '关节角', 4.77], ['目标', '手离目标点越近越好', 6.06], ['约束', '关节的转动范围', 8.25]] },
    { name: '最优控制', sub: 'LQR、MPC', ep: '第四季', on: 11.14, rows: [
      ['变量', '每一步电机出多大的力', 12.16], ['目标', '偏离多少 + 用了多大的力', 15.16], ['同迈步子', '$(4-u)^2+u^2$', 18.41]] },
    { name: '全身控制', sub: 'WBC', ep: '第五季', on: 21.82, rows: [
      ['变量', '加速度、力矩、脚底的力', 22.92], ['目标', '同时完成好几个任务', 26.45], ['约束', '动力学、摩擦、力矩上限', 28.42]] },
    { name: '强化学习', sub: '', ep: '第六季', on: 32.07, rows: [
      ['变量', '策略的参数', 33.23], ['目标', '奖励越多越好', 35.02], ['办法', '一小步一小步地改', 37.6]] }
  ];
  var COLOR = { '变量': E.GOOD, '目标': E.VIOLET, '约束': E.BAD, '同迈步子': E.HI, '办法': E.HI };
  var TABS = 1.68, SAME = 19.91;
  var T = {
    ik: { joints: 5.32, target: 6.54, move: [6.9, 8.0], limit: 8.25, range: 9.74 },
    oc: { push0: 12.3, motor: 13.71, dev: 15.59, effort: 16.9 },
    wbc: { acc: 24.02, tau: 24.73, foot: 25.36, task: 26.84, stand: 27.19, dyn: 28.81, fric: 29.55, cap: 30.22 },
    rl: { param: 34.29, reward: [35.53, 36.9], steps: [37.84, 38.37, 38.9] }
  };
  var BOX = [10, 58, 320, 354];                            // 左边演示框 x, y, w, h

  E.scenes[4] = {
    title: '机器人里到处都是优化',
    dur: 39.89,
    cues: [
      { at: 0, s: '回头看机器人：到处都是**优化问题**' },
      { at: 3.84, s: '逆运动学：变量 关节角，目标 手够到目标点，约束 关节范围' },
      { at: 11.14, s: '最优控制：目标 = 偏离 + 用力，和迈步子的两项扣分一个意思' },
      { at: 21.82, s: '全身控制：变量 关节加速度、力矩和脚底的力，约束 动力学、摩擦、力矩上限' },
      { at: 32.07, s: '强化学习：变量 策略参数，目标 奖励越多越好，一小步一小步地改' }
    ],
    build: function (K) {
      var s = K.sceneSvg('机器人里到处都是优化问题：逆运动学的变量是关节角，目标是手离目标点越近越好，约束是关节的转动范围；最优控制的变量是每一步电机出多大的力，目标是偏离多少加上用了多大的力，和迈步子的两项扣分一个意思；全身控制的变量是关节的加速度、力矩和脚底的力，要同时完成好几个任务，守住动力学、摩擦和力矩上限；强化学习的变量是策略的参数，目标是奖励越多越好，也是一小步一小步地改');

      /* ══ 页签 ══ */
      var tabs = CARDS.map(function (d, i) {
        var tg = E.tag(s, 10 + 196 * i, 6, 188, 40, d.name, E.BORDER, { size: 22, ink: E.INK, weight: 900, rx: 12, fill: E.S1 });
        return { g: tg.g, rect: tg.rect, text: tg.text, cx: 104 + 196 * i };
      });
      var box = K.paint(E.el('rect', { x: BOX[0], y: BOX[1], width: BOX[2], height: BOX[3], rx: 18, 'stroke-width': 2.5 }, s), E.S1, E.BORDER);

      /* ══ 右栏：每个例子一组（名字、出处、三行） ══ */
      var panels = CARDS.map(function (d) {
        var g = E.el('g', {}, s), x = 350;
        E.txt(g, x, 104, d.name, 34, E.INK, 'start', 900);
        if (d.sub) E.txt(g, x + 34 * d.name.length + 14, 102, d.sub, 22, E.MUT, 'start', 700);
        E.tag(g, 682, 76, 108, 38, d.ep, E.INK2, { size: 20, ink: E.INK2, weight: 700, fill: E.S2 });
        var rows = d.rows.map(function (r, k) {
          var rg = E.el('g', {}, g), ry = 184 + 72 * k, c = COLOR[r[0]], lw = r[0].length > 2 ? 116 : 76;
          E.tag(rg, x, ry - 26, lw, 38, r[0], c, { size: 22, ink: c, weight: 900, rx: 10 });
          if (r[1].indexOf('$') === 0) E.tex(rg, x + lw + 14, ry, E.ctex(H.acc, '(4-u)^2') + '+' + E.ctex(H.warn, 'u^2'), 30, E.INK, 'start', { w: 240, h: 50 });
          else E.txt(rg, x + lw + 14, ry, r[1], 26, E.INK, 'start', 700);
          return { g: rg, at: r[2], cy: ry - 7, x: x };
        });
        return { g: g, rows: rows, d: d };
      });

      /* ══ 演示：四组，各自一台 G1 ══ */
      var demos = CARDS.map(function () { return E.el('g', {}, s); });
      // 侧视 G1 的本地坐标 → 分镜坐标（同 g1Side 的 place：以脚底 y = 388 为心缩放）
      function side(parent, x, gy, sc) {
        var r = R3().g1Side(parent, 0);
        r.place({ x: x, scale: sc, dy: gy - 388 });
        r.sv = function (p) { return [r.state.x + sc * p[0], r.state.dy + 388 + sc * (p[1] - 388)]; };
        return r;
      }
      function dot(parent, color, rr) { return K.paint(E.el('circle', { r: rr || 6, 'stroke-width': 2.5 }, parent), color, 'var(--demo-surface)'); }
      function place(n, p) { n.setAttribute('cx', p[0].toFixed(1)); n.setAttribute('cy', p[1].toFixed(1)); }

      /* ── ① 逆运动学：侧视 G1 伸手够胸前的目标点 ── */
      var g1 = demos[0], GY1 = 396, SC1 = 1.5;
      K.paint(E.el('line', { x1: 24, y1: GY1, x2: 316, y2: GY1, 'stroke-width': 3 }, g1), null, E.BORDER);
      var rA = side(g1, 120, GY1, SC1), vA = rA.view;
      var J1 = { lo: -1.0472, hi: 2.0944 };                   // URDF：right_elbow_joint 的 lower / upper（rad）
      function armSet(q) { vA.joints({ right_shoulder_pitch_joint: q[0], right_elbow_joint: q[1] }); }
      function handW(q) { armSet(q); var p = vA.world('right_rubber_hand'); return [p.x, p.z]; }
      function toSvg(name) { return rA.sv(vA.at(name)); }
      // 目标点：手臂在 (−75°, 35°) 时手的位置（够得着）；从手臂垂着、肘微弯（肩 −10°、肘 70°）出发，阻尼最小二乘一步步挪过去
      // （从伸直的 94° 出发会往肘的上限那边弯，卡在 120° 的限位上）
      var q0 = [-10 * D, 70 * D], qT = [-75 * D, 35 * D], tgtW = handW(qT);
      var path = [q0.slice()], q = q0.slice();
      for (var it = 0; it < 22; it++) {
        var h0 = handW(q), e = [tgtW[0] - h0[0], tgtW[1] - h0[1]], J = [[0, 0], [0, 0]];
        for (var j = 0; j < 2; j++) { var qq = q.slice(); qq[j] += 1e-4; var h1 = handW(qq); J[0][j] = (h1[0] - h0[0]) / 1e-4; J[1][j] = (h1[1] - h0[1]) / 1e-4; }
        var l2 = 0.003, A = [[J[0][0] * J[0][0] + J[0][1] * J[0][1] + l2, J[0][0] * J[1][0] + J[0][1] * J[1][1]], [0, J[1][0] * J[1][0] + J[1][1] * J[1][1] + l2]];
        A[1][0] = A[0][1];
        var det = A[0][0] * A[1][1] - A[0][1] * A[1][0], w = [(A[1][1] * e[0] - A[0][1] * e[1]) / det, (-A[1][0] * e[0] + A[0][0] * e[1]) / det];
        q = [q[0] + 0.45 * (J[0][0] * w[0] + J[1][0] * w[1]), Math.max(J1.lo, Math.min(J1.hi, q[1] + 0.45 * (J[0][1] * w[0] + J[1][1] * w[1])))];
        path.push(q.slice());
      }
      // 肘的限位弧：停在最后的姿势，肘从 lower 转到 upper，手指的方向扫出来的扇面
      armSet(path[path.length - 1]);
      var elbowW = vA.world('right_elbow_link'), arcDirs = [];
      for (var k = 0; k <= 30; k++) {
        var hq = handW([path[path.length - 1][0], J1.lo + (J1.hi - J1.lo) * k / 30]);
        arcDirs.push([hq[0] - elbowW.x, hq[1] - elbowW.z]);
      }
      armSet(q0);
      var tgt = E.el('g', {}, g1);
      var tgtC = K.paint(E.el('circle', { r: 11, fill: 'none', 'stroke-width': 3 }, tgt), null, E.HI);
      var tgtX = K.paint(E.el('path', { 'stroke-width': 2.5 }, tgt), null, E.HI);
      var dist = K.paint(E.el('line', { 'stroke-width': 2.5, 'stroke-dasharray': '5 5' }, g1), null, E.VIOLET);
      var distLab = E.halo(E.txt(g1, 0, 0, '', 20, E.VIOLET, 'middle', 900));
      var jA = [dot(g1, E.GOOD), dot(g1, E.GOOD)];
      var limArc = K.paint(E.el('path', { fill: 'none', 'stroke-width': 5, 'stroke-linecap': 'round' }, g1), null, E.GOOD);
      var limStops = [0, 1].map(function () { return K.paint(E.el('line', { 'stroke-width': 6, 'stroke-linecap': 'round' }, g1), null, E.BAD); });
      var limLab = E.halo(E.txt(g1, 0, 0, '肘 −60°～120°', 20, E.BAD, 'start', 900));

      /* ── ② 最优控制：侧视 G1 被推一下、歪了再站回来 ── */
      var g2 = demos[1], GY2 = 396, SC2 = 1.5;
      K.paint(E.el('line', { x1: 24, y1: GY2, x2: 316, y2: GY2, 'stroke-width': 3 }, g2), null, E.BORDER);
      var rB = side(g2, 170, GY2, SC2);
      var plumb = K.paint(E.el('line', { 'stroke-width': 2, 'stroke-dasharray': '5 6' }, g2), null, E.INK2);
      var devArc = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3.5 }, g2), null, E.ACC);
      var pushA = E.vec(g2, E.BAD, 5, { head: 15 });
      var pushLab = E.halo(E.txt(g2, 0, 0, '推', 20, E.BAD, 'end', 900));
      var ankle = dot(g2, E.WARN, 7);
      function barRow(y, name, color) {
        var g = E.el('g', {}, g2);
        E.txt(g, 30, y + 7, name, 20, color, 'start', 900);
        K.paint(E.el('rect', { x: 108, y: y - 9, width: 200, height: 18, rx: 6 }, g), E.S2);
        var b = K.paint(E.el('rect', { x: 108, y: y - 9, height: 18, rx: 6 }, g), color);
        return { g: g, b: b };
      }
      var devBar = barRow(84, '偏离²', E.ACC), effBar = barRow(118, '用力²', E.WARN);
      // 推一下（0.3 s）之后歪的角（度）：再过 τ 秒 θ = 10 · (τ / 0.4) · e^(1 − τ / 0.4)，0.4 s 歪到最大 10°，再慢慢站回来；
      // 力矩取 θ 和它变化快慢的组合（示意，不按真实质量算）
      function tilt(t) {
        var tau = ((t - T.oc.push0) % 3.4 + 3.4) % 3.4, x = Math.max(0, tau - 0.3) / 0.4;
        return { th: 10 * x * Math.exp(1 - x), dth: 10 / 0.4 * (1 - x) * Math.exp(1 - x) * (tau > 0.3 ? 1 : 0), tau: tau };
      }

      /* ── ③ 全身控制：正面 G1，29 个关节点、脚底的力、摩擦锥 ── */
      var g3 = demos[2], GY3 = 366;
      K.paint(E.el('line', { x1: 24, y1: GY3, x2: 316, y2: GY3, 'stroke-width': 3 }, g3), null, E.BORDER);
      var hum = R3().g1Front(g3, { dots: true });
      hum.place(170, 0.9, GY3);
      var cones = E.el('g', {}, g3), forces = E.el('g', {}, g3);
      var feetX = hum.footSpan.map(function (sp) { return hum.world([(sp[0] + sp[1]) / 2, 388])[0]; });
      feetX.forEach(function (fx) {
        K.paint(E.el('path', { d: 'M ' + fx + ' ' + GY3 + ' L ' + (fx - 20) + ' ' + (GY3 + 42) + ' L ' + (fx + 20) + ' ' + (GY3 + 42) + ' Z', 'stroke-width': 2 }, cones),
          E.tint(E.BAD, 25, 'transparent'), E.BAD);
        E.arrow(forces, [[fx, GY3 + 40], [fx, GY3 + 2]], E.GOOD, 5, { head: 13 });
      });
      // 任务：往旁边抬起的那只手要够到的点（手抬到肩高时手的位置，先摆一次姿势量出来）
      hum.pose({ arm: [[80, 84], [8, 4]] }); hum.view.pre();
      var TGT3 = hum.world(hum.view.at('right_rubber_hand'));
      hum.pose({ arm: [[8, 4], [8, 4]] });
      var task = E.el('g', {}, g3);
      var task1 = K.paint(E.el('circle', { r: 10, fill: 'none', 'stroke-width': 3 }, task), null, E.HI);
      var comLine = K.paint(E.el('line', { 'stroke-width': 2.5, 'stroke-dasharray': '5 5' }, g3), null, E.HI);
      var comDot = dot(g3, E.HI, 7);
      var capRings = hum.all.map(function () { return K.paint(E.el('circle', { r: 9, fill: 'none', 'stroke-width': 2.5 }, g3), null, E.BAD); });

      /* ── ④ 强化学习：侧视 G1 原地走，参数小条 + 奖励曲线 ── */
      var g4 = demos[3], GY4 = 270, SC4 = 1.1;
      var tread = K.paint(E.el('line', { x1: 24, y1: GY4, x2: 200, y2: GY4, 'stroke-width': 3, 'stroke-dasharray': '18 10' }, g4), null, E.BORDER);
      var rD = side(g4, 104, GY4, SC4);
      var params = E.el('g', {}, g4);
      K.paint(E.el('rect', { x: 210, y: 80, width: 108, height: 100, rx: 12, 'stroke-width': 2 }, params), E.S2, E.BORDER);
      E.txt(params, 264, 104, '参数', 20, E.GOOD, 'middle', 900);
      var pbars = [0, 1, 2, 3, 4, 5].map(function (i) { return K.paint(E.el('rect', { x: 222 + 15 * i, width: 10, rx: 3 }, params), E.GOOD); });
      var curveG = E.el('g', {}, g4);
      var CX0 = 54, CY0 = 396, CW = 250, CH = 92;
      K.paint(E.el('line', { x1: CX0, y1: CY0, x2: CX0 + CW, y2: CY0, 'stroke-width': 2 }, curveG), null, E.INK2);
      K.paint(E.el('line', { x1: CX0, y1: CY0, x2: CX0, y2: CY0 - CH - 8, 'stroke-width': 2 }, curveG), null, E.INK2);
      E.txt(curveG, CX0 - 8, CY0 - CH + 4, '奖励', 18, E.VIOLET, 'end', 900);
      E.txt(curveG, CX0 + CW, CY0 - 8, '一小步一小步', 18, E.INK2, 'end', 700);
      var rew = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3.5, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, curveG), null, E.VIOLET);
      function rewardAt(x) { return 1 - Math.exp(-3.2 * x) + 0.05 * Math.sin(37 * x) * (1 - x); }   // 示意：越训越高、有起伏

      return {
        el: s,
        draw: function (st, t) {
          var f = E.fade;
          var act = CARDS.map(function (d, i) { var nx = CARDS[i + 1] ? CARDS[i + 1].on : 1e9; return f(st, d.on, 0.35) * (1 - f(st, nx, 0.3)); });
          tabs.forEach(function (tb, i) {
            E.pop(tb.g, tb.cx, 26, f(st, TABS + 0.12 * i, 0.3), 0.85);
            K.paint(tb.rect, null, act[i] > 0.5 ? E.HI : E.BORDER);
            tb.rect.style.strokeWidth = act[i] > 0.5 ? 3.5 : 2;
            tb.text.style.fill = act[i] > 0.5 ? E.HI : st >= CARDS[i].on ? E.INK : E.MUT;
          });
          E.op(box, f(st, CARDS[0].on - 0.3, 0.4));
          panels.forEach(function (p, i) {
            E.op(p.g, act[i]);
            p.rows.forEach(function (r, k) {
              var pr = f(st, r.at, 0.35), sc = 1 + (i === 1 && k === 2 ? 0.08 * E.bump(st, SAME, 0.9) : 0);
              r.g.setAttribute('transform', 'translate(' + (-24 * (1 - pr)).toFixed(1) + ' 0) translate(' + r.x + ' ' + r.cy + ') scale(' + sc.toFixed(3) + ') translate(' + (-r.x) + ' ' + (-r.cy) + ')');
              E.op(r.g, pr);
              E.glow(r.g, E.HI, sc > 1.004);
            });
          });
          demos.forEach(function (g, i) { E.op(g, act[i]); });

          /* ① 逆运动学 */
          var I = T.ik, n = path.length - 1, u = K.ease(K.seg(st, I.move[0], I.move[1])) * n, i0 = Math.floor(u), fr = u - i0;
          var qa = path[Math.min(i0, n)], qb = path[Math.min(i0 + 1, n)];
          armSet([qa[0] + (qb[0] - qa[0]) * fr, qa[1] + (qb[1] - qa[1]) * fr]);
          var tp = rA.sv(vA.project([tgtW[0], vA.world('right_rubber_hand').y, tgtW[1]]));
          var hp = toSvg('right_rubber_hand'), sp = toSvg('right_shoulder_pitch_link'), ep = toSvg('right_elbow_link');
          place(tgtC, tp);
          tgtX.setAttribute('d', 'M ' + (tp[0] - 6) + ' ' + tp[1] + ' H ' + (tp[0] + 6) + ' M ' + tp[0] + ' ' + (tp[1] - 6) + ' V ' + (tp[1] + 6));
          E.pop(tgt, tp[0], tp[1], f(st, I.target, 0.3), 0.5);
          place(jA[0], sp); place(jA[1], ep);
          jA.forEach(function (d, k) { E.op(d, f(st, I.joints + 0.15 * k, 0.3)); });
          var hw = vA.world('right_rubber_hand'), cm = Math.hypot(hw.x - tgtW[0], hw.z - tgtW[1]) * 100;
          dist.setAttribute('x1', hp[0].toFixed(1)); dist.setAttribute('y1', hp[1].toFixed(1)); dist.setAttribute('x2', tp[0].toFixed(1)); dist.setAttribute('y2', tp[1].toFixed(1));
          distLab.textContent = '离目标 ' + (cm < 0.5 ? 0 : Math.round(cm)) + ' cm';
          distLab.setAttribute('x', (tp[0] - 10).toFixed(1)); distLab.setAttribute('y', (tp[1] - 26).toFixed(1));
          var dOn = f(st, I.target + 0.3, 0.3);
          E.op(dist, dOn * (cm < 0.5 ? 0 : 1)); E.op(distLab, dOn);
          // 肘的限位弧（半径 12 cm），两头橙色挡块
          var R = 0.12 * rA.scale * SC1, pts = arcDirs.map(function (d) { var L = Math.hypot(d[0], d[1]); return [ep[0] + R * d[0] / L, ep[1] - R * d[1] / L]; });
          limArc.setAttribute('d', K.polyPath(pts));
          [arcDirs[0], arcDirs[arcDirs.length - 1]].forEach(function (d, k) {
            var L = Math.hypot(d[0], d[1]), ux = d[0] / L, uy = -d[1] / L;
            limStops[k].setAttribute('x1', (ep[0] + (R - 8) * ux).toFixed(1)); limStops[k].setAttribute('y1', (ep[1] + (R - 8) * uy).toFixed(1));
            limStops[k].setAttribute('x2', (ep[0] + (R + 10) * ux).toFixed(1)); limStops[k].setAttribute('y2', (ep[1] + (R + 10) * uy).toFixed(1));
          });
          var lOn = f(st, I.limit, 0.4);
          E.op(limArc, lOn); limStops.forEach(function (l) { E.op(l, lOn); });
          limLab.setAttribute('x', (ep[0] + 30).toFixed(1)); limLab.setAttribute('y', (ep[1] + 44).toFixed(1));
          E.op(limLab, f(st, I.range - 0.6, 0.4));
          limArc.style.strokeWidth = 5 + 2 * E.bump(st, I.range, 0.9);

          /* ② 最优控制：推 → 歪 → 站回来 */
          var O = T.oc, tl = st >= O.push0 ? tilt(t) : { th: 0, dth: 0, tau: 9 };
          rB.place({ rot: tl.th, pivot: [28, 388] });
          var piv = rB.sv([28, 388]);
          plumb.setAttribute('x1', piv[0]); plumb.setAttribute('y1', piv[1]); plumb.setAttribute('x2', piv[0]); plumb.setAttribute('y2', piv[1] - 210);
          var Rd = 150, a = tl.th * D;
          devArc.setAttribute('d', 'M ' + piv[0] + ' ' + (piv[1] - Rd) + ' A ' + Rd + ' ' + Rd + ' 0 0 1 ' + (piv[0] + Rd * Math.sin(a)).toFixed(1) + ' ' + (piv[1] - Rd * Math.cos(a)).toFixed(1));
          E.op(plumb, f(st, O.dev, 0.3)); E.op(devArc, f(st, O.dev, 0.3) * (tl.th > 0.3 ? 1 : 0));
          var pOn = tl.tau < 0.3 ? 1 : 0, py = piv[1] - 150;
          pushA.set(piv[0] - 160, py, piv[0] - 90, py); pushLab.setAttribute('x', piv[0] - 166); pushLab.setAttribute('y', py + 7);
          E.op(pushA.g, pOn * (st >= O.push0 ? 1 : 0)); E.op(pushLab, pOn * (st >= O.push0 ? 1 : 0));
          // 踝：pts() 是转之前的本地坐标，跟着身子绕脚尖转过去
          var ak0 = rB.pts().ankle, ca = Math.cos(a), sa = Math.sin(a);
          var ak = rB.sv([28 + (ak0[0] - 28) * ca - (ak0[1] - 388) * sa, 388 + (ak0[0] - 28) * sa + (ak0[1] - 388) * ca]);
          place(ankle, ak); E.op(ankle, f(st, O.motor, 0.3));
          ankle.setAttribute('r', (7 + 4 * E.bump(st, O.motor, 0.9)).toFixed(1));
          var uu = Math.min(1, Math.abs(0.06 * tl.th + 0.012 * tl.dth));
          devBar.b.setAttribute('width', (200 * Math.min(1, Math.pow(tl.th / 10, 2))).toFixed(1));
          effBar.b.setAttribute('width', (200 * uu * uu).toFixed(1));
          E.op(devBar.g, f(st, O.dev, 0.3)); E.op(effBar.g, f(st, O.effort, 0.3));

          /* ③ 全身控制 */
          var W = T.wbc, nj = 0, raise = f(st, W.task, 0.8);
          hum.pose({ arm: [[8 + 72 * raise, 4 + 80 * raise], [8, 4]] });
          ['legs', 'waist', 'arms'].forEach(function (kk) {
            hum.groups[kk].forEach(function (d) {
              var p = f(st, W.acc + 0.025 * nj++, 0.2);
              d.style.fill = p > 0.5 ? E.GOOD : E.INK; d.setAttribute('r', (4.5 + 1.5 * p).toFixed(2)); E.op(d, 0.4 + 0.6 * p);
            });
          });
          hum.view.pre();
          var cap = E.bump(st, W.cap, 1.2);
          hum.all.forEach(function (d, j) { var p = hum.locate(d); place(capRings[j], p); E.op(capRings[j], cap > 0.05 ? cap : 0); });
          E.op(forces, f(st, W.foot, 0.3)); E.op(cones, f(st, W.fric, 0.4));
          place(task1, TGT3);
          task1.setAttribute('r', (10 + 3 * E.bump(st, W.task + 0.8, 0.8)).toFixed(1));
          E.op(task, f(st, W.task + 0.6, 0.3));
          var com = hum.com(), cw = hum.world(com);
          place(comDot, cw);
          comLine.setAttribute('x1', cw[0].toFixed(1)); comLine.setAttribute('y1', cw[1].toFixed(1)); comLine.setAttribute('x2', cw[0].toFixed(1)); comLine.setAttribute('y2', GY3);
          E.op(comDot, f(st, W.stand, 0.3)); E.op(comLine, f(st, W.stand, 0.3));

          /* ④ 强化学习 */
          var Rl = T.rl;
          rD.walk(((t / 1.1) % 1 + 1) % 1);
          tread.style.strokeDashoffset = (t * 60).toFixed(1);
          E.op(params, f(st, Rl.param, 0.3));
          var nStep = Rl.steps.filter(function (a) { return st >= a; }).length;
          pbars.forEach(function (b, i) {
            var hgt = 26 + 22 * Math.sin(i * 1.7 + 0.9 * nStep + 1.3) + 6 * (i % 2);
            b.setAttribute('y', (168 - hgt).toFixed(1)); b.setAttribute('height', hgt.toFixed(1));
          });
          var pb = Rl.steps.reduce(function (m, a) { return Math.max(m, E.bump(st, a, 0.45)); }, 0);
          E.glow(params, E.GOOD, pb > 0.05);
          E.op(curveG, f(st, Rl.reward[0], 0.3));
          var upto = 0.55 * K.seg(st, Rl.reward[0], Rl.reward[1]) + 0.15 * nStep, ptsR = [];
          for (var k2 = 0; k2 <= 60 * upto; k2++) { var x = k2 / 60; ptsR.push([CX0 + CW * x, CY0 - CH * rewardAt(x)]); }
          rew.setAttribute('d', ptsR.length > 1 ? K.polyPath(ptsR) : '');
        }
      };
    }
  };
})();
