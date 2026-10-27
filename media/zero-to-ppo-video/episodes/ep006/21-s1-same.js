// 006 第 1 幕：同一个朝向，三种写法。左边 G1，右边一列三张写法卡片（同片头：欧拉角 / 四元数 / 旋转矩阵，从上往下）。
// 句 1「机器人原地往左转」G1 往左转 90°，脚下画出 0° → 90° 的弧（绕竖轴 z 转，蓝），「九十度」弧旁标 90°；
// 句 2「旋转矩阵」矩阵卡弹出，「三列」「三根小箭头」「转完以后的方向」三列依次框亮、骨盆上对应的小箭头同时变粗，「九个数」卡上写「9 个数」；
// 句 3「欧拉角」欧拉角卡弹出 (90°, 0°, 0°)，「三个角」写「3 个数」，「转身九十度」框亮 90°，「另外两个角」框亮两个 0°；
// 句 4「四元数」四元数卡弹出，「四个数」写「4 个数」，四个数按念到的顺序写出，「怎么算出来」右边出问号，「第四幕」问号换成「第 4 幕再算」；
// 句 5「九个数」「三个数」「四个数」三张卡的个数依次鼓一下，「同一个朝向」卡片左边出一道亮黄括线、箭头指回 G1，G1 头顶写「同一个朝向」；
// 句 6「三个自由度」骨盆上绕三根小箭头各出一圈（绕哪根轴就用那根轴的颜色：蓝 z 转身、绿 y 前后倾、红 x 侧歪），「最少三个数」欧拉角卡的「3 个数」亮起来；
// 句 7「欧拉角」「有个坑」欧拉角卡右边红牌「有个坑」，「另外两种」「守规矩」另两张卡右边出「要守规矩」，「下面一个一个看」三张卡右下写第几幕讲。
// 时刻取自配音的逐词时间（分镜时间 = 念到这个词的时刻），见 ../ep006.py。
(function () {
  var E = window.EP006, K = E.K;
  var D = Math.PI / 180;
  var T = {
    robot: 0.6, turn: [2.55, 1.0], deg: 3.17,
    mat: 4.91, cols: 9.95, arrows: 8.96, nine: 11.25,
    eul: 13.36, three: 14.6, first: 15.29, rest: 16.66,
    quat: 19.29, four: 20.5, nums: [21.22, 22.25, 22.74, 23.19], how: 24.91, ep4: 26.18,
    counts: [29.05, 29.76, 28.22], same: 30.92,
    dof: 32.73, rings: [34.23, 34.43, 34.64], least: 35.42, outRings: 38.6,
    eulBad: 40.87, others: 42.13, rules: 44.36, next: 45.55
  };
  var GX = 150, GY = 362, GH = 300;
  var CX = 300, CW = 490, TX = 640, TW = 138;
  var ROWS = [{ y: 18, h: 100 }, { y: 130, h: 100 }, { y: 242, h: 150 }];

  function bump(st, a, d) { var u = K.seg(st, a, a + d); return u > 0 && u < 1 ? Math.sin(Math.PI * u) : 0; }

  E.scenes[0] = {
    title: '同一个朝向，三种写法',
    dur: 47.68,
    cues: [
      { at: 0, s: '一个朝向：机器人原地往左转 90°' },
      { at: 4.58, s: '**旋转矩阵**：3×3，三列 = 三根小箭头转完的方向，9 个数' },
      { at: 13.05, s: '**欧拉角**：3 个角，转身 90°，另外两个是 0' },
      { at: 18.95, s: '**四元数**：4 个数 (0.71, 0, 0, 0.71)，第 4 幕再算' },
      { at: 28.12, s: '9 个、3 个、4 个数，说的是**同一个朝向**' },
      { at: 32.63, s: '转动只有 3 个自由度：最少 3 个数就能说清楚' },
      { at: 38.92, s: '欧拉角数最少，偏偏有坑；另外两种多用几个数，要守规矩' }
    ],
    build: function (K) {
      var s = K.sceneSvg('机器人原地往左转九十度：用旋转矩阵写是九个数，三列是三根小箭头转完的方向；用欧拉角写是三个角 (90°, 0°, 0°)；用四元数写是四个数 (0.71, 0, 0, 0.71)；三种写法说的是同一个朝向。转动只有三个自由度，最少三个数就够；只用三个数的欧拉角偏偏有坑，另外两种多用几个数，要守规矩');

      /* ══ 左：G1 ══ */
      var rob = E.el('g', {}, s);
      var ring = K.paint(E.el('path', { fill: 'none', 'stroke-width': 2.5, 'stroke-dasharray': '6 6' }, rob), null, E.BORDER);
      var arc = E.el('g', {}, rob);
      var arcLine = K.paint(E.el('path', { fill: 'none', 'stroke-width': 4, 'stroke-linecap': 'round' }, arc), null, E.AX[2]);
      var arcHead = K.paint(E.el('path', {}, arc), E.AX[2]);
      var g1 = E.g1(rob, GX, GY, GH);
      var dofBack = E.el('g', {}, rob);
      var fr = E.frame3(rob, 5);
      var dofFront = E.el('g', {}, rob);
      var dofRings = [E.AX[2], E.AX[1], E.AX[0]].map(function (c) { return E.ring3(dofBack, dofFront, c, 4.5); });
      var deg = E.halo(E.txt(rob, 0, 0, '90°', 24, E.AX[2], 'middle', 900));
      ring.setAttribute('d', E.ptsPath(g1.ring(0.45), true));
      var sameLab = E.halo(E.txt(s, GX, 34, '同一个朝向', 26, E.HI, 'middle', 900));
      var dofLab = E.el('g', {}, s);
      E.halo(E.txt(dofLab, GX, 34, '转动：3 个自由度', 24, E.INK, 'middle', 900));

      /* ══ 右：三张卡片 ══ */
      var defs = [
        { name: '欧拉角', c: E.WARN, n: '3 个数', tag: '有个坑', tc: E.BAD, next: '第 2、3 幕' },
        { name: '四元数', c: E.ACC, n: '4 个数', tag: '要守规矩', tc: E.INK2, next: '第 4 幕' },
        { name: '旋转矩阵', c: E.VIOLET, n: '9 个数', tag: '要守规矩', tc: E.INK2, next: '第 5 幕' }
      ];
      var cards = defs.map(function (d, i) {
        var r = ROWS[i], cd = E.card(s, CX, r.y, CW, r.h, d.name, d.c, 26);
        var cnt = E.txt(cd.g, CX + 22, r.y + 72, d.n, 20, E.INK2, 'start', 700);
        var tg = E.tag(cd.g, TX, r.y + r.h / 2 - 40, TW, 36, d.tag, d.tc, { size: 20, ink: d.tc === E.BAD ? E.BAD : E.INK, weight: 900, rx: 10 });
        var nx = E.txt(cd.g, TX + TW / 2, r.y + r.h / 2 + 30, d.next, 20, E.INK2, 'middle', 700);
        return { g: cd.g, rect: cd.rect, cnt: cnt, tag: tg, next: nx, cy: r.y + r.h / 2 };
      });
      // 欧拉角：三个角（度数用正文字体），各带一个框
      var eulG = E.el('g', {}, cards[0].g);
      var eulBox = [0, 1, 2].map(function () { var b = K.paint(E.el('rect', { y: ROWS[0].y + 34, height: 40, rx: 8, 'stroke-width': 2.5 }, eulG), E.tint(E.HI, 18, 'var(--demo-surface)'), E.HI); E.op(b, 0); return b; });
      var EX = [446, 520, 584];   // 三个数各自居中，逗号、括号单独排，框只套住数（框边离逗号、括号都留几 px）
      [[406, '('], [484, ','], [552, ','], [618, ')']].forEach(function (d) { E.txt(eulG, d[0], ROWS[0].y + 64, d[1], 30, E.INK, 'middle', 700); });
      ['90°', '0°', '0°'].forEach(function (v, i) { E.txt(eulG, EX[i], ROWS[0].y + 64, v, 30, E.INK, 'middle', 700); });
      eulBox.forEach(function (b, i) { var w = i ? 40 : 58; b.setAttribute('x', EX[i] - w / 2); b.setAttribute('width', w); });
      // 四元数：四个数
      var qNums = E.spans(cards[1].g, 524, ROWS[1].y + 62, ['(', '0.71', ', ', '0', ', ', '0', ', ', '0.71', ')'], 28, E.INK).filter(function (sp, k) { return k % 2; });
      var qAsk = E.txt(cards[1].g, TX + TW / 2, ROWS[1].y + 66, '?', 40, E.INK2, 'middle', 900);
      var qLater = E.txt(cards[1].g, TX + TW / 2, ROWS[1].y + 58, '第 4 幕再算', 20, E.INK2, 'middle', 700);
      // 旋转矩阵：往左转 90°
      var m3 = E.grid(cards[2].g, 548, ROWS[2].y + 76, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], { size: 24, dx: 58, dy: 32, colors: E.AX });
      // 括线：卡片左边一道竖线，三个小钩接到卡片，中间一支箭头指回 G1
      var brace = E.el('g', {}, s);
      K.paint(E.el('path', { d: 'M 296 ' + (ROWS[0].y + 50) + ' H 284 V ' + (ROWS[2].y + 75) + ' H 296 M 284 ' + (ROWS[1].y + 50) + ' H 296', fill: 'none', 'stroke-width': 3.5, 'stroke-linejoin': 'round' }, brace), null, E.HI);
      var bArrow = E.arrow(brace, [[282, 200], [246, 200]], E.HI, 3.5, { head: 12 });

      return {
        el: s,
        draw: function (st, t) {
          var f = E.fade;

          /* G1 往左转 90° + 脚下的弧 */
          var turn = f(st, T.turn[0], T.turn[1]);
          g1.set(90 * D * turn);
          var fm = g1.frame(0.5);
          fr.set(fm);
          var apts = [];
          for (var k = 0; k <= 24; k++) { var a = 90 * D * turn * k / 24; apts.push(g1.project([0.58 * Math.cos(a), 0.58 * Math.sin(a), g1.sole])); }
          arcLine.setAttribute('d', turn > 0.02 ? E.ptsPath(apts) : '');
          var pa = apts[apts.length - 2], pb = apts[apts.length - 1], ang = Math.atan2(pb[1] - pa[1], pb[0] - pa[0]);
          arcHead.setAttribute('d', turn > 0.1 ? 'M ' + pb[0].toFixed(1) + ' ' + pb[1].toFixed(1) + ' L ' + (pb[0] - 13 * Math.cos(ang) + 7 * Math.sin(ang)).toFixed(1) + ' ' + (pb[1] - 13 * Math.sin(ang) - 7 * Math.cos(ang)).toFixed(1) +
            ' L ' + (pb[0] - 13 * Math.cos(ang) - 7 * Math.sin(ang)).toFixed(1) + ' ' + (pb[1] - 13 * Math.sin(ang) + 7 * Math.cos(ang)).toFixed(1) + ' Z' : '');
          var dp = g1.project([0, 0.58, g1.sole]);   // 弧的终点（朝左）右边
          deg.setAttribute('x', (dp[0] + 34).toFixed(1)); deg.setAttribute('y', (dp[1] + 4).toFixed(1));
          E.op(arc, f(st, T.turn[0], 0.2) * (1 - f(st, T.mat, 0.4)));
          E.op(deg, f(st, T.deg, 0.3) * (1 - f(st, T.mat, 0.4)));
          E.op(rob, f(st, 0, T.robot));

          /* 句 2：三列依次框亮，骨盆上对应的小箭头变粗 */
          [0, 1, 2].forEach(function (j) {
            var on = f(st, T.cols + 0.25 * j, 0.3) * (1 - f(st, T.nine + 0.6, 0.4));
            m3.colBox(j, E.AX[j], on);
            var ab = Math.max(bump(st, T.arrows, 0.8), on);
            fr.arrows[j].line.style.strokeWidth = 5 + 3.5 * ab;
          });

          /* 卡片弹出；个数 */
          E.pop(cards[2].g, CX + CW / 2, cards[2].cy, f(st, T.mat, 0.35), 0.9);
          E.pop(cards[0].g, CX + CW / 2, cards[0].cy, f(st, T.eul, 0.35), 0.9);
          E.pop(cards[1].g, CX + CW / 2, cards[1].cy, f(st, T.quat, 0.35), 0.9);
          [T.three, T.four, T.nine].forEach(function (a, i) {
            E.op(cards[i].cnt, f(st, a, 0.3));
            var b = bump(st, T.counts[i], 0.7), lit = i === 0 ? f(st, T.least, 0.3) * (1 - f(st, T.eulBad, 0.4)) : 0;
            E.scaleAt(cards[i].cnt, CX + 22, cards[i].cy + 15, 1 + 0.35 * Math.max(b, 0.5 * lit));
            cards[i].cnt.style.fill = b > 0.05 || lit > 0.05 ? 'var(--hi)' : '';
          });

          /* 句 3：欧拉角的三个数 */
          E.op(eulBox[0], f(st, T.first, 0.3) * (1 - f(st, T.rest, 0.3)));
          E.op(eulBox[1], f(st, T.rest, 0.3) * (1 - f(st, T.rest + 1.4, 0.3)));
          E.op(eulBox[2], f(st, T.rest + 0.2, 0.3) * (1 - f(st, T.rest + 1.4, 0.3)));

          /* 句 4：四元数四个数依次写出；问号 → 第 4 幕再算 */
          qNums.forEach(function (c, k) { c.style.fillOpacity = f(st, T.nums[k], 0.25); });
          E.op(qAsk, f(st, T.how, 0.3) * (1 - f(st, T.ep4, 0.3)));
          E.op(qLater, f(st, T.ep4, 0.3) * (1 - f(st, T.rules, 0.3)));

          /* 句 5：同一个朝向 */
          var sameOn = f(st, T.same, 0.4) * (1 - f(st, T.dof, 0.4));
          E.op(brace, sameOn); E.drawOn(bArrow.line, f(st, T.same, 0.4)); E.op(sameLab, sameOn);

          /* 句 6：三个自由度——骨盆上绕三根小箭头各一圈 */
          var P = function (v) { return g1.project(v); };
          P.toward = E.view3(135, 20).toward;
          var Rm = E.Rz(90 * D * turn), ax = [E.col3(Rm, 0), E.col3(Rm, 1), E.col3(Rm, 2)];
          // 绕 z：平面里是前、左；绕 y：前、上；绕 x：左、上
          [[ax[0], ax[1]], [ax[0], ax[2]], [ax[1], ax[2]]].forEach(function (pl, i) {
            dofRings[i].set([0, 0, 0], pl[0], pl[1], 0.3, P, P);
            var on = f(st, T.rings[i], 0.3) * (1 - f(st, T.outRings, 0.4));
            E.op(dofRings[i].front, on); E.op(dofRings[i].back, on);
          });
          E.op(dofLab, f(st, T.rings[0], 0.3) * (1 - f(st, T.outRings, 0.4)));

          /* 句 7：坑、守规矩、第几幕 */
          cards.forEach(function (cd, i) {
            var a = i === 0 ? T.eulBad : T.rules + 0.15 * (i - 1);
            E.pop(cd.tag.g, TX + TW / 2, cd.cy - 22, f(st, a, 0.3), 0.8);
            E.op(cd.next, f(st, T.next + 0.25 * i, 0.3));
            var hl = i === 0 ? bump(st, 39.18, 1.2) : bump(st, T.others, 1.2);
            cd.rect.style.strokeWidth = 5 * hl;   // 高亮：外框描一圈同色的边，边框加粗
          });
        }
      };
    }
  };
})();
