// 001 第 4 幕：第三步：控制（主战场）。四步条「控制」为当前步、盒里弹出 ★；本幕不画客厅。
// 开场大字「控制 · ★ 本系列主战场」→ 开车类比：导航「前面右转」= 规划，方向盘打多少、油门踩多深 = 控制 →
// 腿部特写：规划只交来「下一脚踩那儿」，控制盒把它翻译成髋 / 膝 / 踝每一刻的力和角度（横条台阶式跳动）→
// 被推、脚下不平 → 重新算 → 秒表：每秒几百～上千次，甚至更多。
(function () {
  var E = window.EP001, K = E.K;
  var RAD = Math.PI / 180, TAU = Math.PI * 2;

  /* 腿：G1 的右腿特写——侧视 G1（髋在本地 (0, 302)、踝 ≈ 381）放大 LEG_K 倍，髋落在 (150, 110)、踝 ≈ (150, 338)，
     和原先手画的腿同一位置；膝屈伸、被推时绕踝晃、踩到鼓包脚掌转动都驱动 G1 的关节 */
  var HIP = [150, 110];
  var LEG_K = 2.87, LEG_X = 150, LEG_DY = HIP[1] - 388 + 86 * LEG_K;
  // 关节名在关节圆左侧（右对齐），跟着关节走。「髋」比规格 (112,115) 低 7px：推的箭头 y = 93 从它头顶擦过，原位会压字
  var LABEL_OFF = [[-38, 22], [-38, 5], [-38, 5]];   // 「髋」再往下 10：「推一下」的箭头横在髋的高度上，别压字

  var ROWS = [{ name: '髋', y: 170, at: 17.2 }, { name: '膝', y: 228, at: 17.8 }, { name: '踝', y: 286, at: 18.2 }];
  var JITTER = 18.7, FAST = 28.9;          // 横条开始跳；改成每 0.04 s 跳
  var PUSH = 23.2, HIT = 23.38;            // 推的箭头飞出；撞上躯干
  var BUMP = 24.3, REDO = 25.2;
  var SLOT_X = 582, SLOT_W = 96;           // 力横槽里的填充（槽 580–680，内缩 2）

  function mix(c, p, base) {
    return p <= 0 ? base : p >= 1 ? c : 'color-mix(in srgb, ' + c + ' ' + (p * 100).toFixed(1) + '%, ' + base + ')';
  }
  // 一次性闪 n 下（每下 per 秒），返回 0…1；其余时刻 0
  function flash(st, a, n, per) {
    var u = (st - a) / per;
    return u < 0 || u >= n ? 0 : Math.sin(Math.PI * (u - Math.floor(u)));
  }
  function line(parent, color, w) {
    var l = E.el('line', { 'stroke-width': w, 'stroke-linecap': 'round' }, parent);
    K.paint(l, null, color);
    return l;
  }
  function put(l, a, b) {
    l.setAttribute('x1', a[0].toFixed(1)); l.setAttribute('y1', a[1].toFixed(1));
    l.setAttribute('x2', b[0].toFixed(1)); l.setAttribute('y2', b[1].toFixed(1));
  }
  function slide(g, dx, p) { g.setAttribute('transform', 'translate(' + (dx * (1 - p)).toFixed(2) + ' 0)'); E.op(g, p); }

  /* 指令值：每 per 秒跳一个新值，基值 0.6 ± 0.2·amp，各行相位错开 1/3 格（按真实时间 t） */
  function hash(n) { var x = Math.sin(n * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); }
  function lvl(i, n, amp) { return 0.6 + 0.2 * amp * (2 * hash(n * 7 + i * 131) - 1); }
  function level(i, t, per, amp) { return lvl(i, Math.floor(t / per + i / 3), amp); }
  // 同一串值的平滑版（上一格 → 这一格缓动过去）：表盘指针、膝角用它，跟着横条但不抽搐
  function levelSmooth(i, t, per, amp) {
    var u = t / per + i / 3, n = Math.floor(u);
    return lvl(i, n - 1, amp) + (lvl(i, n, amp) - lvl(i, n - 1, amp)) * K.ease(u - n);
  }


  E.scenes[3] = {
    title: '第三步：控制（主战场）',
    dur: 30.8,
    cues: [
      { at: 0, s: '★ 第三步 **控制**：本系列的主战场' },
      { at: 2.4, s: '导航说「右转」是规划；方向盘打多少、油门踩多深是控制' },
      { at: 9.8, s: '机器人：规划交来「下一脚踩那儿」这样的目标' },
      { at: 13.9, s: '控制：算出每个关节这一毫秒的力或角度' },
      { at: 21.7, s: '被推、踩到不平 → 马上重新算，一刻不停' },
      { at: 26.7, s: '关节这一层：每秒算几百到上千次' }
    ],
    build: function (K) {
      var s = K.sceneSvg('控制：像开车时方向盘打多少、油门踩多深；机器人的控制把「下一脚踩那儿」翻译成髋、膝、踝每一刻的力和角度，被推或踩到不平就马上重新算，每秒约一千次');

      /* ── 四步条 ── */
      var bar = E.stepBar(s);

      /* ── 开场大字 ── */
      var open = E.el('g', {}, s);
      var bigWord = E.txt(open, 400, 230, '控制', 56, E.ACC, 'middle', 900);
      var subWord = E.txt(open, 400, 280, '★ 本系列主战场', 24, E.ACC);

      /* ── 开车类比：方向盘 + 两根踏板槽 + 导航胶囊 + 括号 ── */
      var drive = E.el('g', {}, s);
      var wheel = E.el('g', {}, drive);
      var ring = E.el('circle', { cx: 480, cy: 210, r: 62, fill: 'none', 'stroke-width': 7 }, wheel);
      // 轮辐 90° / 210° / 330°（SVG 角度：0° 朝右、顺时针，90° 朝下）
      var spokes = [90, 210, 330].map(function (a) {
        var l = line(wheel, E.MUT, 7), c = Math.cos(a * RAD), sn = Math.sin(a * RAD);
        put(l, [480 + 12 * c, 210 + 12 * sn], [480 + 58 * c, 210 + 58 * sn]);
        return l;
      });
      var hub = E.el('circle', { cx: 480, cy: 210, r: 12 }, wheel);
      var gasSlot = K.paint(E.el('rect', { x: 600, y: 160, width: 28, height: 110, rx: 6, 'stroke-width': 2 }, drive), E.S2, E.BORDER);
      var gasFill = K.paint(E.el('rect', { x: 603, y: 267, width: 22, height: 0, rx: 4 }, drive), E.ACC);
      K.paint(E.el('rect', { x: 650, y: 160, width: 28, height: 110, rx: 6, 'stroke-width': 2 }, drive), E.S2, E.BORDER);
      E.txt(drive, 614, 294, '油门', 20, E.INK2);
      E.txt(drive, 664, 294, '刹车', 20, E.INK2);

      var nav = E.el('g', {}, drive);
      K.paint(E.el('rect', { x: 80, y: 96, width: 260, height: 46, rx: 23, 'stroke-width': 2.5 }, nav), E.tint(E.VIOLET, 15), E.VIOLET);
      E.txt(nav, 196, 127, '导航：前面右转', 22, E.VIOLET);
      // 右转小箭头：竖杆上来、拐向右（306–322, 114–128）
      K.paint(E.el('path', { d: 'M 309 128 V 121.5 Q 309 117.5 313 117.5 H 317', fill: 'none', 'stroke-width': 2.5, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, nav), null, E.VIOLET);
      K.paint(E.el('path', { d: 'M 316 113.5 L 322 117.5 L 316 121.5 Z' }, nav), E.VIOLET);
      var isPlan = E.txt(drive, 210, 176, '= 规划', 22, E.VIOLET, 'middle', 700);

      var brace = K.paint(E.el('path', { d: 'M 418 316 V 324 H 690 V 316', fill: 'none', 'stroke-width': 3, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, drive), null, E.ACC);
      var isCtrl = E.txt(drive, 554, 350, '= 控制', 22, E.ACC, 'middle', 700);

      /* ── 脚下不平（先建，压在腿下面）：规格 M100,370 Q150,354 210,370 整体上移 8px，让鼓包顶贴住脚底，不然脚悬在包上方约 12px ── */
      var bump = K.paint(E.el('path', { d: 'M 100 362 Q 150 346 210 362', fill: 'none', 'stroke-width': 3, 'stroke-linecap': 'round' }, s), null, E.BAD);

      /* ── 腿部特写（左）：G1 的右腿（同 S2–S6 的侧视 G1，放大 LEG_K 倍、只画骨盆以下，上缘渐隐），髋落在 (150, 110)。
         髋 / 膝 / 踝套一个空心圈（点名时发光），关节名跟着圈走 ── */
      var leg = E.el('g', {}, s);
      var bot = E.robot(leg, 0, { box: [-34, 286, 84, 106], res: 6, fadeTop: true, hideArms: true });
      var joints = [16, 15, 13].map(function (r) {
        return K.paint(E.el('circle', { r: r, fill: 'none', 'stroke-width': 3 }, leg), null, E.INK2);
      });
      var jointNames = ROWS.map(function (row) { return E.txt(leg, 0, 0, row.name, 20, E.INK2, 'end'); });

      /* ── 输入卡（规划给的）→ 翻译机盒 ── */
      var card = E.el('g', {}, s);
      K.paint(E.el('rect', { x: 250, y: 84, width: 280, height: 44, rx: 12, 'stroke-width': 2.5 }, card), E.tint(E.VIOLET, 12), E.VIOLET);
      E.txt(card, 380, 112, '规划：下一脚踩那儿', 20, E.VIOLET);
      K.paint(E.el('ellipse', { cx: 500, cy: 106, rx: 10, ry: 4 }, card), E.VIOLET);
      var inArrow = E.arrow(s, [[390, 128], [390, 158]], E.VIOLET, 3, { head: 10 });

      var box = E.el('g', {}, s);
      var boxRect = K.paint(E.el('rect', { x: 280, y: 160, width: 220, height: 100, rx: 14, 'stroke-width': 3 }, box), E.tint(E.ACC, 25), E.ACC);
      E.txt(box, 390, 205, '控制', 30, E.ACC, 'middle', 700);
      E.txt(box, 390, 238, '翻译成关节命令', 20, E.INK);
      var redo = E.txt(s, 390, 284, '重新算！', 20, E.ACC);

      /* ── 三行关节指令（右）：名字 · 力 [横槽] · 角度 (表盘)；扇形箭头从盒子右缘引出 ── */
      var rows = ROWS.map(function (row) {
        var y = row.y;
        // 关节名居中在 x = 540（左缘约 530），箭头尖停在 522（规格 534 会扎进字里）
        var fan = E.curveArrow(s, 'M 500 210 C 514 210 508 ' + y + ' 522 ' + y, E.ACC, 3, 0, { head: 10 });
        var g = E.el('g', {}, s);
        E.txt(g, 540, y + 7, row.name, 20, E.INK);
        E.txt(g, 566, y + 6.5, '力', 18, E.INK2);
        K.paint(E.el('rect', { x: 580, y: y - 7, width: 100, height: 14, rx: 7, 'stroke-width': 1.5 }, g), E.S2, E.BORDER);
        var fill = K.paint(E.el('rect', { x: SLOT_X, y: y - 5, width: 0, height: 10, rx: 5 }, g), E.ACC);
        E.txt(g, 704, y + 6.5, '角度', 18, E.INK2);
        K.paint(E.el('circle', { cx: 752, cy: y, r: 16, 'stroke-width': 2 }, g), E.S2, E.INK2);
        var needle = line(g, E.ACC, 3);
        put(needle, [752, y], [764, y]);
        K.paint(E.el('circle', { cx: 752, cy: y, r: 3 }, g), E.ACC);
        return { g: g, fan: fan, fill: fill, needle: needle, y: y, at: row.at };
      });

      /* ── 干扰：推一下（bad 粗箭头）+「不平」标字（鼓包在腿前面建） ── */
      var push = E.el('g', {}, s);
      var pushLine = line(push, E.BAD, 7);
      var pushHead = K.paint(E.el('path', { d: 'M 0 0 L -18 -9 L -18 9 Z' }, push), E.BAD);
      var pushLabel = E.txt(s, 66, 82, '推一下', 20, E.BAD);
      var bumpLabel = E.txt(s, 240, 364, '不平', 20, E.BAD);

      /* ── 有多快：秒表 + 每秒几百～上千次 ── */
      var watch = E.el('g', {}, s);
      K.paint(E.el('rect', { x: 296, y: 313, width: 8, height: 6, rx: 2 }, watch), E.ACC);
      K.paint(E.el('circle', { cx: 300, cy: 340, r: 20, 'stroke-width': 3 }, watch), E.S2, E.ACC);
      var hand = line(watch, E.ACC, 3);
      put(hand, [300, 340], [300, 326]);
      K.paint(E.el('circle', { cx: 300, cy: 340, r: 2.5 }, watch), E.ACC);
      // 「1 秒」用 accent，和秒表归成一组；用 ink2 时和同一水平线上的灰色「出处：…」连成一句
      var oneSec = E.txt(s, 300, 384, '1 秒', 18, E.ACC);   // 离秒表圈远一点（圈底 y 361.5）
      var rate = E.txt(s, 335, 348, '每秒几百～上千次', 28, E.ACC, 'start', 700);
      var source = E.txt(s, 335, 380, '出处：主路线 L4.0 方法谱系表', 18, E.MUT, 'start');
      var more = E.txt(s, 582, 348, '甚至更多', 22, E.ACC, 'start');

      return {
        el: s,
        draw: function (st, t) {
          var fade = E.fade;

          /* 四步条：感知、规划讲过，控制为当前步；0.3 控制盒里弹出 ★ 并一直保留 */
          bar.set(['done', 'done', 'cur', 'off'], t);
          E.pop(bar.star, 566, 33, fade(st, 0.3, 0.35), 0.3);

          /* 0.2 开场两行字弹出；2.4 淡出（0.25 s，大字「控制」和方向盘位置重叠，不交叉叠化） */
          var op0 = fade(st, 0.2, 0.4);
          E.pop(open, 400, 245, op0, 0.85);
          E.op(open, Math.min(op0, 1 - fade(st, 2.4, 0.25)));

          /* 开车类比：2.6 开场字基本淡完后淡入（灰），9.8 整体淡出 */
          E.op(drive, Math.min(fade(st, 2.6, 0.4), 1 - fade(st, 9.8, 0.3)));
          // 3.9 导航胶囊；5.4「= 规划」
          E.pop(nav, 210, 119, fade(st, 3.9, 0.35), 0.8);
          E.op(isPlan, fade(st, 5.4, 0.35));
          // 6.3 方向盘、油门槽变 accent；6.4–7.4 方向盘顺时针转 40°
          var hot = fade(st, 6.3, 0.3), wc = mix(E.ACC, hot, E.MUT);
          ring.style.stroke = wc; hub.style.fill = wc;
          spokes.forEach(function (l) { l.style.stroke = wc; });
          gasSlot.style.stroke = mix(E.ACC, hot, E.BORDER);
          wheel.setAttribute('transform', 'rotate(' + (40 * fade(st, 6.4, 1.0)).toFixed(2) + ' 480 210)');
          // 7.6–8.4 油门槽里的 accent 从底部升到 60%；刹车槽保持空
          var gh = 0.6 * 104 * fade(st, 7.6, 0.8);
          gasFill.setAttribute('y', (267 - gh).toFixed(2));
          gasFill.setAttribute('height', gh.toFixed(2));
          // 8.7 括号画出 +「= 控制」
          E.drawOn(brace, K.ease(K.seg(st, 8.7, 9.05)));
          E.op(isCtrl, fade(st, 8.85, 0.35));

          /* 腿部特写：9.95–10.3 淡入 */
          E.op(leg, fade(st, 9.95, 0.35));

          /* 11.1 输入卡；13.9 翻译机盒弹出 + 箭头，之后按 t 呼吸发光 */
          E.pop(card, 390, 106, fade(st, 11.1, 0.35), 0.85);
          var bp = fade(st, 13.9, 0.35);
          E.pop(box, 390, 210, bp, 0.85);
          var ap = K.seg(st, 13.9, 14.15);
          E.drawOn(inArrow.line, ap);
          E.op(inArrow.head, ap >= 1 ? 1 : 0);
          var gl = 0.5 + 0.5 * Math.sin(TAU * t / 1.6);
          boxRect.style.filter = bp > 0 ? 'drop-shadow(0 0 ' + ((3 + 7 * gl) * bp).toFixed(1) + 'px ' + E.ACC + ')' : '';

          /* 指令值：18.7 起按 t 台阶式跳（每 0.1 s，28.9 起每 0.04 s）；被推后 1.5 s 内幅度 ×1.8 */
          var on = st >= JITTER, per = st >= FAST ? 0.04 : 0.1;
          var amp = 1 + 0.8 * E.win(st, HIT, HIT + 1.5, 0.12);

          /* 17.2 / 17.8 / 18.2 三行依次出现：扇形箭头画出、行滑入、腿上对应关节描边闪 accent */
          rows.forEach(function (r, i) {
            var p = K.seg(st, r.at, r.at + 0.25);
            E.drawOn(r.fan.line, K.ease(p));
            E.op(r.fan.head, p >= 1 ? 1 : 0);
            slide(r.g, -14, fade(st, r.at + 0.1, 0.3));
            var v = on ? level(i, t, per, amp) : 0;
            r.fill.setAttribute('width', (SLOT_W * E.clamp(v, 0, 1)).toFixed(2));
            var vs = on ? levelSmooth(i, t, per, amp) : 0.6;
            var a = (-90 + 55 * (vs - 0.6) / 0.2) * RAD;
            put(r.needle, [752, r.y], [752 + 12 * Math.cos(a), r.y + 12 * Math.sin(a)]);
            var f = flash(st, r.at, 2, 0.25);
            joints[i].style.stroke = mix(E.ACC, f, E.INK2);
            joints[i].style.strokeWidth = 3 + 1.5 * f;
            E.glow(joints[i], E.ACC, f > 0.4);
          });

          /* 腿：膝角跟着「膝」行轻微变化 ±6°（28.9 后横条每 0.04 s 跳，膝仍按 0.1 s 的节奏跟，免得整条腿高频抖）；
             被推后脚踩住不动，整个身子（躯干 + 大腿 + 小腿）像倒立摆一样绕踝晃 ±4°，1.2 s 内衰减——
             以前是绕髋晃，脚会在地上来回滑 30 多像素，违反物理；踩到不平脚倾斜 6° */
          var dKnee = on ? 6 * E.clamp((levelSmooth(1, t, 0.1, amp) - 0.6) / 0.2, -1, 1) : 0;
          var tau = st - HIT, sway = 0;
          if (tau > 0 && tau < 1.2) sway = 4 * Math.pow(1 - tau / 1.2, 2) * Math.sin(TAU * tau / 0.6);
          // G1：膝屈 dKnee，大腿前摆、小腿后摆各一半，踝留在髋正下方；踩到鼓包脚尖朝上 6°；被推时整个人绕踝晃（svg 旋转）
          bot.legs([dKnee / 2, -dKnee / 2], [dKnee / 2 - 3, -dKnee / 2 - 3], -6 * fade(st, BUMP, 0.3));
          var LP = bot.pts(), piv = LP.ankle.slice();
          bot.place({ x: LEG_X, dy: LEG_DY, scale: LEG_K, rot: sway, pivot: piv });
          var sr = sway * RAD, cs = Math.cos(sr), sn = Math.sin(sr);
          var toScene = function (q) {
            var dx = q[0] - piv[0], dy = q[1] - piv[1], r = [piv[0] + dx * cs - dy * sn, piv[1] + dx * sn + dy * cs];
            return [LEG_X + r[0] * LEG_K, 388 + (r[1] - 388) * LEG_K + LEG_DY];
          };
          var P = { hip: toScene(LP.hip), knee: toScene(LP.knee), ankle: toScene(LP.ankle) };
          [P.hip, P.knee, P.ankle].forEach(function (c, i) {
            joints[i].setAttribute('cx', c[0].toFixed(1)); joints[i].setAttribute('cy', c[1].toFixed(1));
            jointNames[i].setAttribute('x', (c[0] + LABEL_OFF[i][0]).toFixed(1)); jointNames[i].setAttribute('y', (c[1] + LABEL_OFF[i][1]).toFixed(1));
          });

          /* 23.2 推的箭头从左边飞出、撞上躯干（尾巴钉在 x = 36，尖从 54 冲到 114） */
          var pp = K.seg(st, PUSH, HIT), tip = 54 + 60 * pp * pp;
          put(pushLine, [36, 93], [tip - 12, 93]);
          pushHead.setAttribute('transform', 'translate(' + tip.toFixed(2) + ' 93)');
          E.op(push, st >= PUSH ? 1 : 0);
          E.op(pushLabel, fade(st, PUSH, 0.3));

          /* 24.3 脚下鼓包画出 +「不平」 */
          E.drawOn(bump, K.ease(K.seg(st, BUMP, BUMP + 0.3)));
          E.op(bumpLabel, fade(st, BUMP + 0.05, 0.3));

          /* 25.2「重新算！」闪两次（亮—灭—亮），之后留着 */
          var ru = (st - REDO) / 0.45;
          E.op(redo, ru <= 0 ? 0 : ru < 1.5 ? 0.5 - 0.5 * Math.cos(TAU * ru) : 1);

          /* 27.8 秒表：指针按 t 一秒一圈；28.9 每秒几百～上千次 + 出处；30.0 甚至更多 */
          E.pop(watch, 300, 338, fade(st, 27.8, 0.35), 0.7);
          hand.setAttribute('transform', 'rotate(' + (360 * (((t % 1) + 1) % 1)).toFixed(1) + ' 300 340)');
          E.op(oneSec, fade(st, 27.9, 0.35));
          slide(rate, -12, fade(st, 28.9, 0.35));
          E.op(source, fade(st, 29.05, 0.35));
          slide(more, -12, fade(st, 30.0, 0.35));
        }
      };
    }
  };
})();
