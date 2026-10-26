// 008 第 5 幕：机器人里到处都是优化。2 × 2 四张卡，每张三行：变量（绿）、目标（紫）、约束（红）；念到哪张卡，哪张描亮黄边。
// 句 1「到处」四张空卡一起淡入；
// 句 2 逆运动学（第 12 集）：「变量」关节角、「目标」手离目标点越近越好、「约束」关节转动范围，按念到的词一行行出；
// 句 3 最优控制（第四季，LQR、MPC）：「变量」每一步电机出多大的力、「目标」偏离多少 + 用了多大的力，
//    「迈步子」第三行出亮黄的「同迈步子：(4 − u)² + u²」，「一个意思」这一行鼓一下；
// 句 4 全身控制（第五季）：「变量」关节力矩、脚底的力，「任务」目标 同时完成好几个任务，「守住」约束 动力学、摩擦、力矩上限；
// 句 5 强化学习（第六季）：「变量」策略的参数，「目标」奖励越多越好，「这样」第三行亮黄「办法：一小步一小步地改」。
// 时刻取自配音的逐词时间（分镜时间 = 念到这个词的时刻），见 ../ep008.py。
(function () {
  var E = window.EP008, K = E.K, H = E.HEX;
  var CW = 386, CHH = 198, GAP = 10;
  var CARDS = [
    { name: '逆运动学', sub: '', ep: '第 12 集', on: 3.84, rows: [
      ['变量', '关节角', 4.77], ['目标', '手离目标点越近越好', 6.06], ['约束', '关节的转动范围', 8.25]] },
    { name: '最优控制', sub: 'LQR、MPC', ep: '第四季', on: 11.14, rows: [
      ['变量', '每一步电机出多大的力', 12.16], ['目标', '偏离多少 + 用了多大的力', 15.16], ['同迈步子', '$(4-u)^2+u^2$', 18.41]] },
    { name: '全身控制', sub: 'WBC', ep: '第五季', on: 21.82, rows: [
      ['变量', '关节力矩、脚底的力', 22.92], ['目标', '同时完成好几个任务', 25.95], ['约束', '动力学、摩擦、力矩上限', 27.9]] },
    { name: '强化学习', sub: '', ep: '第六季', on: 31.56, rows: [
      ['变量', '策略的参数', 32.72], ['目标', '奖励越多越好', 34.51], ['办法', '一小步一小步地改', 37.09]] }
  ];
  var COLOR = { '变量': E.GOOD, '目标': E.VIOLET, '约束': E.BAD, '同迈步子': E.HI, '办法': E.HI };
  var FRAMES = 1.68, SAME = 19.91;

  E.scenes[4] = {
    title: '机器人里到处都是优化',
    dur: 39.38,
    cues: [
      { at: 0, s: '回头看机器人：到处都是**优化问题**' },
      { at: 3.84, s: '逆运动学：变量 关节角，目标 手够到目标点，约束 关节范围' },
      { at: 11.14, s: '最优控制：目标 = 偏离 + 用力，和迈步子的两项扣分一个意思' },
      { at: 21.82, s: '全身控制：变量 关节力矩和脚底的力，约束 动力学、摩擦、力矩上限' },
      { at: 31.56, s: '强化学习：变量 策略参数，目标 奖励越多越好，一小步一小步地改' }
    ],
    build: function (K) {
      var s = K.sceneSvg('机器人里到处都是优化问题：逆运动学的变量是关节角，目标是手离目标点越近越好，约束是关节的转动范围；最优控制的变量是每一步电机出多大的力，目标是偏离多少加上用了多大的力，和迈步子的两项扣分一个意思；全身控制的变量是关节力矩和脚底的力，要同时完成好几个任务，守住动力学、摩擦和力矩上限；强化学习的变量是策略的参数，目标是奖励越多越好，也是一小步一小步地改');
      var cards = CARDS.map(function (d, i) {
        var x = 10 + (CW + GAP) * (i % 2), y = 8 + (CHH + GAP) * Math.floor(i / 2);
        var g = E.el('g', {}, s);
        var frame = K.paint(E.el('rect', { x: x, y: y, width: CW, height: CHH, rx: 18, 'stroke-width': 2.5 }, g), E.S1, E.BORDER);
        var head = E.el('g', {}, g);
        var nm = E.txt(head, x + 20, y + 40, d.name, 28, E.INK, 'start', 900);
        if (d.sub) E.txt(head, x + 20 + 28 * d.name.length + 12, y + 39, d.sub, 20, E.MUT, 'start', 700);
        E.tag(head, x + CW - 112, y + 14, 96, 34, d.ep, E.INK2, { size: 20, ink: E.INK2, weight: 700, fill: E.S2 });
        var rows = d.rows.map(function (r, k) {
          var rg = E.el('g', {}, g), ry = y + 80 + 40 * k, c = COLOR[r[0]], lw = r[0].length > 2 ? 104 : 64;
          E.tag(rg, x + 18, ry - 22, lw, 32, r[0], c, { size: 20, ink: c, weight: 900, rx: 9 });
          var tx = r[1].indexOf('$') === 0 ? E.tex(rg, x + 30 + lw, ry, r[1].slice(1, -1).replace('(4-u)^2', E.ctex(H.acc, '(4-u)^2')).replace('u^2', E.ctex(H.warn, 'u^2')), 24, E.INK, 'start', { w: 220, h: 40 })
            : E.txt(rg, x + 30 + lw, ry, r[1], 22, E.INK, 'start', 700);
          return { g: rg, at: r[2], cy: ry - 6, x: x + 18 };
        });
        return { g: g, frame: frame, head: head, rows: rows, d: d, cx: x + CW / 2, cy: y + CHH / 2 };
      });

      return {
        el: s,
        draw: function (st) {
          var f = E.fade;
          cards.forEach(function (c, i) {
            E.op(c.g, f(st, FRAMES + 0.12 * i, 0.4));
            E.op(c.head, 0.3 + 0.7 * f(st, c.d.on, 0.3));
            var next = CARDS[i + 1] ? CARDS[i + 1].on : 1e9;
            var act = f(st, c.d.on, 0.3) * (1 - f(st, next, 0.3));
            K.paint(c.frame, null, act > 0.5 ? E.HI : E.BORDER);
            c.frame.style.strokeWidth = 2.5 + 1.5 * act;
            c.rows.forEach(function (r, k) {
              // 滑进来；最优控制的第三行「同迈步子」在「一个意思」时鼓一下
              var p = f(st, r.at, 0.35), sc = 1 + (i === 1 && k === 2 ? 0.08 * E.bump(st, SAME, 0.9) : 0);
              r.g.setAttribute('transform', 'translate(' + (-24 * (1 - p)).toFixed(1) + ' 0) translate(' + r.x + ' ' + r.cy + ') scale(' + sc.toFixed(3) + ') translate(' + (-r.x) + ' ' + (-r.cy) + ')');
              E.op(r.g, p);
              E.glow(r.g, E.HI, sc > 1.004);
            });
          });
        }
      };
    }
  };
})();
