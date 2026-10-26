// 006 第 3 幕：万向锁——两根轴转到一起。左边万向节（三个套在一起的环 + 中间身子的三根小箭头），右边三行读数（偏航 · 外环 / 俯仰 · 中环 / 横滚 · 内环），
// 底下一块说明区，按旁白换内容。
// 句 1 右边三行先出来，「第二个角」第二行亮一下；「三次转动」万向节的立柱和小箭头出来，「三个套在一起的环」外、中、内三个环依次出来；
//    「外环管偏航」「中环管俯仰」「内环管横滚」各自来回摆一下，同时对应的行亮；
// 句 2「万向节」说明区写「这种架子：万向节」，「万向锁」下面补一行「万向锁的名字就来自它」；
// 句 3「俯仰转到九十度」中环转到 90°（第二行数到 90°），「前面那根小箭头」绿箭头变粗，「地面」箭头尖旁写「朝下」；
// 句 4「内环的轴」画绿色转轴虚线（已经竖着），「外环的轴」画蓝色转轴虚线，「转到了一起」两根一起亮、说明区写「两根轴重合」；
// 句 5「转外环」外环来回转 30°，「转内环」内环反着来回转 30°——身子的动作一模一样，说明区一行行写出，「效果一样」打等号；
// 句 6「偏航三十」外环转到 30°，说明区写 (30°, 90°, 0°)；「偏航零、横滚负三十」外环转回 0°、内环转到 −30°，身子一动不动，写 (0°, 90°, −30°)，「一模一样」打等号；
// 句 7「绕前后这根轴」画红色虚线（世界的前后）和绕它的红色弯箭头，「做不到」打叉；「只剩两个方向」说明区写出，「万向锁」大红牌；
// 句 8「摔倒」右边换成一台脸朝下趴在地上的 G1（俯仰 90°），「俯仰正好九十度」标出；
// 句 9「没到九十度」右边换成一张小表：身子绕前后转 1°，俯仰 80° / 85° / 88° / 89° 时偏航、横滚各跳 6° / 11° / 27° / 45°，「跳一大截」最后两格变红。
// 时刻取自配音的逐词时间（分镜时间 = 念到这个词的时刻），见 ../ep006.py。
(function () {
  var E = window.EP006, K = E.K;
  var D = Math.PI / 180;
  var T = {
    rows: 0.1, second: 1.71, stand: 3.02, rings: [4.49, 4.8, 5.1], wig: [6.3, 7.6, 8.9], wigD: 1.0,
    jie: 11.82, suo: 12.67,
    pitch: [16.67, 1.4], green: 19.27, down: 20.69,
    xAxis: 23.29, zAxis: 25.27, same: 26.18,
    outerW: [28.6, 1.4], outerTxt: 28.52, innerW: [31.0, 1.4], innerTxt: 30.97, eq: 33.42,
    y30: [35.9, 0.6], tA: 35.88, swap: [38.3, 1.5], tB: 38.27, eq2: 41.54,
    xRed: 44.31, nope: 45.98, two: 49.16, lock: 51.37,
    fall: 54.72, face: 56.21, p90: 57.47,
    table: 60.24, small: 62.34, jump: 65.24
  };
  var GC = [228, 196], GL = 148;
  var RX = 430, RW = 360;
  var COLS = [E.ACC, E.WARN, E.GOOD];
  var JUMP = [[80, 6], [85, 11], [88, 27], [89, 45]];   // 身子绕世界 x 转 1° 时，ZYX 欧拉角的偏航、横滚各变多少（数值算出，见 ../ep006.py）

  function bump(st, a, d) { var u = K.seg(st, a, a + d); return u > 0 && u < 1 ? Math.sin(Math.PI * u) : 0; }
  function wig(st, a, d, amp) { var u = K.seg(st, a, a + d); return u > 0 && u < 1 ? amp * Math.sin(2 * Math.PI * u) : 0; }

  E.scenes[2] = {
    title: '万向锁：两根轴转到一起',
    dur: 66.84,
    cues: [
      { at: 0, s: '三次转动做成三个套在一起的环：外环偏航、中环俯仰、内环横滚' },
      { at: 10.7, s: '这种架子叫**万向节**，「万向锁」的名字就来自它' },
      { at: 15.76, s: '俯仰转到 90°：身子朝下，前面那根小箭头指向地面' },
      { at: 22.46, s: '内环的轴竖了起来，和外环的轴**重合**' },
      { at: 27.96, s: '转外环、转内环，都是绕这根竖线转：效果一样' },
      { at: 35.09, s: '(30°, 90°, 0°) 和 (0°, 90°, −30°)：同一个姿势' },
      { at: 43.13, s: '绕前后的轴，三个环都转不了：只剩 2 个方向 → **万向锁**' },
      { at: 52.87, s: '人形机器人摔倒、脸朝下趴着：俯仰正好 90°' },
      { at: 59.78, s: '靠近 90° 时：身子只转一点点，算出来的角跳一大截' }
    ],
    build: function (K) {
      var s = K.sceneSvg('万向节：外环管偏航、中环管俯仰、内环管横滚。俯仰转到九十度，内环的轴竖起来和外环的轴重合，转外环和转内环效果一样，(30°, 90°, 0°) 和 (0°, 90°, −30°) 是同一个姿势；绕前后的轴，三个环都转不了，只剩两个方向，这就是万向锁。人形机器人摔倒脸朝下趴着，俯仰正好九十度；靠近九十度时，身子转一点点，算出来的角跳一大截');

      /* ══ 左：万向节 ══ */
      var proj = E.view3(150, 22);   // 从左前方看：外环转到 +30° 时也不会侧成一条线（−30° 那一下才侧过来，只在来回摆的一瞬）
      var gb = E.gimbal(s, GC[0], GC[1], GL, proj);
      var P = gb.P;
      var downLab = E.halo(E.txt(s, 0, 0, '朝下', 22, E.GOOD, 'start', 900));
      // 世界的前后（x）：红虚线 + 绕它的弯箭头 + 叉
      var redG = E.el('g', {}, s);
      var xLine = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3.5, 'stroke-dasharray': '10 7', 'stroke-linecap': 'round' }, redG), null, E.BAD);
      var curl = E.el('g', {}, redG);
      var curlLine = K.paint(E.el('path', { fill: 'none', 'stroke-width': 4, 'stroke-linecap': 'round' }, curl), null, E.BAD);
      var curlHead = K.paint(E.el('path', {}, curl), E.BAD);
      var curlX = E.el('g', {}, redG);
      var xLab = E.halo(E.txt(redG, 0, 0, '前后', 22, E.BAD, 'middle', 900));
      (function () {
        var a = P([-1.25, 0, 0]), b = P([1.3, 0, 0]);
        xLine.setAttribute('d', 'M ' + a[0].toFixed(1) + ' ' + a[1].toFixed(1) + ' L ' + b[0].toFixed(1) + ' ' + b[1].toFixed(1));
        var pts = [];
        for (var k = 0; k <= 36; k++) { var t = (-60 + 290 * k / 36) * D; pts.push(P([1.12, 0.26 * Math.cos(t), 0.26 * Math.sin(t)])); }
        curlLine.setAttribute('d', E.ptsPath(pts));
        var p1 = pts[pts.length - 2], p2 = pts[pts.length - 1], an = Math.atan2(p2[1] - p1[1], p2[0] - p1[0]);
        curlHead.setAttribute('d', 'M ' + p2[0].toFixed(1) + ' ' + p2[1].toFixed(1) + ' L ' + (p2[0] - 14 * Math.cos(an) + 8 * Math.sin(an)).toFixed(1) + ' ' + (p2[1] - 14 * Math.sin(an) - 8 * Math.cos(an)).toFixed(1) +
          ' L ' + (p2[0] - 14 * Math.cos(an) - 8 * Math.sin(an)).toFixed(1) + ' ' + (p2[1] - 14 * Math.sin(an) + 8 * Math.cos(an)).toFixed(1) + ' Z');
        var c = P([1.12, 0, 0.62]);   // 叉打在弯箭头上方，不压箭头
        E.cross(curlX, c[0], c[1], 13, E.BAD, 5);
        xLab.setAttribute('x', b[0].toFixed(1)); xLab.setAttribute('y', (b[1] + 30).toFixed(1));
      })();

      /* ══ 右：三行读数 ══ */
      var rows = [['偏航', '外环'], ['俯仰', '中环'], ['横滚', '内环']].map(function (d, i) {
        var y = 14 + 54 * i, g = E.el('g', {}, s);
        var box = K.paint(E.el('rect', { x: RX, y: y, width: RW, height: 46, rx: 12, 'stroke-width': 2.5 }, g), E.tint(COLS[i], 8, 'var(--demo-surface)'), COLS[i]);
        E.txt(g, RX + 18, y + 32, d[0], 24, COLS[i], 'start', 900);
        E.txt(g, RX + 74, y + 31, '· ' + d[1], 20, E.INK2, 'start', 700);
        var val = E.txt(g, RX + RW - 16, y + 33, '0°', 26, E.INK, 'end', 900);
        return { g: g, box: box, val: val };
      });

      /* ══ 右下：说明区（y 190–410），按旁白换内容 ══ */
      function note(lines) {
        var g = E.el('g', {}, s);
        var items = lines.map(function (l) {
          var t = E.txt(g, l.x || RX, l.y, l.s, l.size || 22, l.c || E.INK, l.a || 'start', 900);
          return t;
        });
        return { g: g, items: items };
      }
      var nJie = note([{ s: '这种架子：万向节', y: 236, c: E.HI, size: 26 }, { s: '「万向锁」的名字就来自它', y: 278, c: E.INK2 }]);
      var nSame = note([{ s: '内环的轴 = 外环的轴', y: 236, c: E.BAD, size: 24 }]);
      var nEff = note([{ s: '转外环：绕竖线转', y: 236 }, { s: '转内环：也绕竖线转', y: 276 }, { s: '= 效果一样', y: 322, c: E.HI, size: 24 }]);
      var nTwo = E.el('g', {}, s);
      var tA = E.txt(nTwo, RX + RW / 2, 238, '(30°, 90°, 0°)', 28, E.INK, 'middle', 900);
      var tEq = E.txt(nTwo, RX + RW / 2, 280, '=', 30, E.HI, 'middle', 900);
      var tB = E.txt(nTwo, RX + RW / 2, 322, '(0°, 90°, −30°)', 28, E.INK, 'middle', 900);
      var tSame = E.txt(nTwo, RX + RW / 2, 366, '两组数，同一个姿势', 22, E.HI, 'middle', 900);
      var nLock = note([{ s: '绕前后的轴：转不了', y: 232, c: E.BAD }, { s: '3 个角，只剩 2 个方向', y: 274 }]);
      var lockTag = E.tag(nLock.g, RX + 70, 300, 220, 56, '万向锁', E.BAD, { size: 32, ink: E.BAD, weight: 900, rx: 14, sw: 3 });

      /* ══ 右：脸朝下趴着的 G1（俯仰 90°） ══ */
      var fallG = E.el('g', {}, s);
      K.paint(E.el('path', { d: 'M ' + (RX + 6) + ' 316 H ' + (RX + RW - 6), fill: 'none', 'stroke-width': 3 }, fallG), null, E.BORDER);
      var g1 = E.g1(fallG, RX + RW / 2 - 4, 316, 330, { az: 150, el: 12, w: 356, up: -150, down: 20 });
      var fr = E.frame3(fallG, 4.5);
      g1.setR(E.Ry(90 * D));
      g1.lift(-g1.lowest());   // 趴下以后整台挪到最低点贴地
      fr.set(g1.frame(0.2));
      var fallLab = E.txt(fallG, RX + RW / 2, 168, '摔倒、脸朝下趴着', 24, E.INK, 'middle', 900);
      var p90 = E.txt(fallG, RX + RW / 2, 214, '俯仰 = 90°', 28, E.WARN, 'middle', 900);

      /* ══ 右：靠近 90° 的小表 ══ */
      var tab = E.el('g', {}, s);
      E.txt(tab, RX + RW / 2, 40, '身子绕前后只转 1°：', 22, E.INK, 'middle', 900);
      E.txt(tab, RX + 70, 84, '俯仰', 20, E.INK2, 'middle', 700);
      E.txt(tab, RX + 250, 84, '偏航、横滚各跳', 20, E.INK2, 'middle', 700);
      var bars = JUMP.map(function (j, i) {
        var y = 104 + 64 * i, g = E.el('g', {}, tab);
        E.txt(g, RX + 70, y + 34, j[0] + '°', 26, E.WARN, 'middle', 900);
        var bar = E.el('rect', { x: RX + 140, y: y + 12, width: 0, height: 30, rx: 6 }, g);
        var v = E.txt(g, RX + 150, y + 36, j[1] + '°', 24, E.INK, 'start', 900);
        return { g: g, bar: bar, v: v, w: 150 * j[1] / 45 };
      });

      return {
        el: s,
        draw: function (st) {
          var f = E.fade;

          /* 万向节的三个角 */
          var yaw = wig(st, T.wig[0], T.wigD, 25 * D) + wig(st, T.outerW[0], T.outerW[1], 30 * D)
            + 30 * D * f(st, T.y30[0], T.y30[1]) * (1 - f(st, T.swap[0], T.swap[1]));
          var pitch = wig(st, T.wig[1], T.wigD, 25 * D) + 90 * D * f(st, T.pitch[0], T.pitch[1]);
          var roll = wig(st, T.wig[2], T.wigD, 25 * D) + wig(st, T.innerW[0], T.innerW[1], -30 * D) - 30 * D * f(st, T.swap[0], T.swap[1]);
          gb.set(yaw, pitch, roll);
          E.op(gb.stand, f(st, T.stand, 0.4)); E.op(gb.body[0].g.parentNode, f(st, T.stand, 0.4));
          gb.rings.forEach(function (r, i) {
            var on = f(st, T.rings[i], 0.35);
            E.op(r.front, on); E.op(r.back, on); E.op(gb.axles[i], on);
            E.op(gb.pins[2 * i], on); E.op(gb.pins[2 * i + 1], on);
            var hl = bump(st, T.wig[i] - 0.1, T.wigD + 0.2);
            r.front.style.strokeWidth = 8 + 4 * hl;
          });
          // 绿箭头（身子的前）变粗、写「朝下」
          var gw = Math.max(bump(st, T.green, 1.2), f(st, T.green, 0.3) * (1 - f(st, T.xAxis, 0.4)));
          gb.body[0].line.style.strokeWidth = 6 + 4 * gw;
          var tip = P(E.col3(gb.R, 0).map(function (x) { return x * 0.52; }));
          downLab.setAttribute('x', (tip[0] + 14).toFixed(1)); downLab.setAttribute('y', (tip[1] + 8).toFixed(1));
          E.op(downLab, f(st, T.down, 0.3) * (1 - f(st, T.xAxis, 0.4)));
          // 两根转轴：内环（绿）、外环（蓝）；「转到了一起」一起亮
          var sb = bump(st, T.same, 1.0) + bump(st, T.same + 1.0, 1.0);
          E.op(gb.axes.x, f(st, T.xAxis, 0.4) * (1 - f(st, T.xRed - 0.3, 0.4)));
          E.op(gb.axes.z, f(st, T.zAxis, 0.4) * (1 - f(st, T.xRed - 0.3, 0.4)));
          gb.axes.z.style.strokeWidth = 9 + 3 * sb; gb.axes.x.style.strokeWidth = 3.5 + 2 * sb;   // 两根重合：蓝的粗、垫在下面，绿的细、压在上面，看得出是两根
          // 世界的前后：红虚线、弯箭头、叉
          E.op(redG, f(st, T.xRed, 0.4) * (1 - f(st, T.fall - 0.4, 0.4)));
          E.drawOn(curlLine, f(st, T.xRed + 0.3, 0.6));
          E.op(curlHead, f(st, T.xRed + 0.8, 0.2));
          var cx0 = P([1.12, 0, 0.62]);
          E.pop(curlX, cx0[0], cx0[1], f(st, T.nope, 0.3), 0.5);

          /* 右：三行读数 */
          var vals = [yaw, pitch, roll];
          var rowsOn = f(st, T.rows, 0.4) * (1 - f(st, T.fall - 0.4, 0.4));
          rows.forEach(function (r, i) {
            E.op(r.g, rowsOn);
            r.val.textContent = E.num(Math.round(vals[i] / D)) + '°';
            var hl = bump(st, T.wig[i] - 0.3, T.wigD + 0.6) + (i === 1 ? bump(st, T.second, 1.0) + bump(st, T.pitch[0], T.pitch[1] + 0.4) : 0);
            r.box.style.strokeWidth = 2.5 + 3 * Math.min(1, hl);
            E.glow(r.box, COLS[i], hl > 0.05);
          });

          /* 说明区 */
          E.op(nJie.g, f(st, T.jie, 0.35) * (1 - f(st, T.pitch[0], 0.4)));
          E.op(nJie.items[1], f(st, T.suo, 0.35));
          E.op(nSame.g, f(st, T.same, 0.35) * (1 - f(st, T.outerTxt - 0.2, 0.3)));
          E.op(nEff.g, f(st, T.outerTxt, 0.3) * (1 - f(st, T.tA - 0.3, 0.3)));
          E.op(nEff.items[1], f(st, T.innerTxt, 0.3)); E.op(nEff.items[2], f(st, T.eq, 0.3));
          E.op(nTwo, f(st, T.tA, 0.3) * (1 - f(st, T.xRed - 0.3, 0.3)));
          E.op(tB, f(st, T.tB, 0.3)); E.op(tEq, f(st, T.eq2, 0.3)); E.op(tSame, f(st, T.eq2, 0.3));
          E.op(nLock.g, f(st, T.xRed, 0.3) * (1 - f(st, T.fall - 0.4, 0.4)));
          E.op(nLock.items[0], f(st, T.nope, 0.3)); E.op(nLock.items[1], f(st, T.two, 0.3));
          E.pop(lockTag.g, RX + 180, 328, f(st, T.lock, 0.35), 0.6);

          /* 右：趴着的 G1、小表 */
          E.op(fallG, f(st, T.fall, 0.4) * (1 - f(st, T.table, 0.4)));
          E.op(fallLab, f(st, T.face, 0.3)); E.op(p90, f(st, T.p90, 0.3));
          E.op(tab, f(st, T.table, 0.4));
          bars.forEach(function (b, i) {
            var p = f(st, T.small + 0.35 * i, 0.4);
            b.bar.setAttribute('width', (b.w * p).toFixed(1));
            b.v.setAttribute('x', (RX + 150 + b.w * p).toFixed(1));
            E.op(b.v, p);
            var red = i >= 2 ? f(st, T.jump, 0.3) : 0;
            b.bar.style.fill = red > 0.5 ? E.tint(E.BAD, 60, 'var(--demo-surface)') : E.tint(E.INK2, 45, 'var(--demo-surface)');
            b.v.style.fill = red > 0.5 ? E.BAD : E.INK;
          });
        }
      };
    }
  };
})();
