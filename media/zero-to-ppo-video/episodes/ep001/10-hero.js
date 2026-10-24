// 片头主图（800×420，片头与封面共用）：四台长相天差地别的机器人依次弹出 → 共用括号 → 下面一排四个「?」盒子
// → 反馈虚线把四个盒子连成一个圈，光点绕圈（走进盒子时由盒子亮边接力）→ 「拆开给你看」时四个「?」依次放大。
// 封面静帧（still）：「?」换成 感知 / 规划 / 控制 / 执行，控制盒标「★ 本系列主战场」，弧下文字改成反馈的意思，不画光点。
(function () {
  var E = window.EP001, K = E.K;
  var BOX_X = [24, 220, 416, 612], BOX_W = 164, MID_Y = 263;
  // 光点路线同规格（顺时针，经过四个盒子中心），起点挪到反馈虚线的起点 (694,308)：6.8 虚线刚画完，光点顺着它接着跑
  var LOOP = [[694, 308], [694, 344], [106, 344], [106, 263], [694, 263], [694, 308]];
  var TAU = Math.PI * 2;

  function line(parent, d) {
    var p = E.el('path', { d: d, fill: 'none', 'stroke-width': 2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, parent);
    return K.paint(p, null, E.MUT);
  }

  E.hero = function (K) {
    var s = K.sceneSvg('片头主图：四种机器人，拆开都在转同一个四步圈');

    // 共用括号（M60,192 V200 H740 V192 + 中间刻度），从刻度往两边画出：「四台共用下面这一排」
    var brace = [line(s, 'M 400 200 H 60 V 192'), line(s, 'M 400 200 H 740 V 192'), line(s, 'M 400 200 V 210')];

    // 机器人一排：出场时刻对上旁白念到的名字；c 是弹出缩放的中心（各自画面的中心）。
    // 人形取 2.15 而不是规格的 2.6：配音里「人形机器人」从分镜 2.17 念到 2.81，2.6 弹出时这个词已快念完。
    var bots = [
      { x: 160, at: 0.1, name: '扫地机', c: [160, 100], make: function (g) { return E.vacuum(g, 160, 100, 0.8); } },
      { x: 320, at: 0.7, name: '机械臂', c: [330, 112], make: function (g) { return E.arm(g, 306, 150, 0.9); } },
      { x: 480, at: 1.3, name: '自动驾驶汽车', c: [480, 120], make: function (g) { return E.car(g, 480, 150, 0.8); } },
      { x: 640, at: 2.15, name: '人形机器人', c: [640, 96], make: function (g) { return E.humanoid(g, 640, 152, 1.0); } }
    ];
    bots.forEach(function (b) {
      b.g = E.el('g', {}, s);
      b.r = b.make(b.g);
      b.label = E.txt(s, b.x, 180, b.name, 20, E.INK2);
    });
    var vac = bots[0].r, arm = bots[1].r, car = bots[2].r, hum = bots[3].r;

    // 盒间箭头：第 i 支跟着第 i+1 个盒子出现
    var links = [0, 1, 2].map(function (i) {
      return E.arrow(s, [[BOX_X[i] + BOX_W + 4, MID_Y], [BOX_X[i + 1] - 4, MID_Y]], E.INK2, 3, { head: 10 });
    });

    // 反馈虚线：执行盒底 → 绕到下面 → 回到感知盒底
    var fb = E.arrow(s, [[694, 308], [694, 344], [106, 344], [106, 314]], E.INK2, 3, { dash: '8 6' });
    // 规格写 y=376；下移 6，光点（r9 + 青光）沿下沿走过时和字之间留出空隙
    var caption = E.txt(s, 400, 382, '', 20, E.INK2);

    // 光点：白（--text）+ 青光。画在盒子下面：走进盒子就被盒子盖住、改由那个盒子亮边，
    // 不会从「?」字上穿过去（叠在字前 / 字后都像穿模）
    var dot = K.paint(E.el('circle', { r: 9 }, s), E.INK);
    dot.style.filter = 'drop-shadow(0 0 6px ' + E.ACC + ') drop-shadow(0 0 12px ' + E.ACC + ')';

    // 盒子一排（故意不对准上面的机器人）：「?」与封面用的步骤名叠在同一格，按 still 切换
    var boxes = BOX_X.map(function (x, i) {
      var g = E.el('g', {}, s);
      var rect = E.el('rect', { x: x, y: 218, width: BOX_W, height: 90, rx: 16, 'stroke-width': 2 }, g);
      K.paint(rect, E.S2, E.BORDER);
      return { g: g, rect: rect, x: x, cx: x + 82 };
    });

    // 盒中字。封面步骤名基线取 266、「★」行取 295（规格 258 / 290）：
    // 258 时四个名字都贴在盒子上半部，没有「★」行的三个盒子下半截空着；下移后单行、两行都接近居中，四个名字仍同一基线
    boxes.forEach(function (b, i) {
      b.tg = E.el('g', {}, s);
      b.q = E.txt(b.tg, b.cx, 277, '?', 40, E.MUT, 'middle', 900);
      b.name = E.txt(b.tg, b.cx, 266, E.STEPS[i].name, 30, E.STEPS[i].color, 'middle', 900);
    });
    var star = E.txt(boxes[2].tg, 498, 295, '★ 本系列主战场', 18, E.ACC);

    return {
      el: s,
      // t：片头分镜时间（introAt.hero = 0，所以就是片头的 st）；still：封面静帧；
      // rt：片头真实时间（循环动作用）。stage.html 现在只传 t，没有 rt 时退回用 t。
      draw: function (t, still, rt) {
        if (rt === undefined) rt = t;

        bots.forEach(function (b) {
          var p = E.fade(t, b.at, 0.3);
          E.pop(b.g, b.c[0], b.c[1], p);
          E.op(b.label, p);
        });
        // 循环动作；封面取一个好看的定格姿势
        var dx = still ? 0 : 12 * Math.sin(TAU * rt / 2);
        vac.g.setAttribute('transform', 'translate(' + (160 + dx).toFixed(2) + ' 100) scale(0.8)');
        arm.set(25 + (still ? 0 : 10 * Math.sin(TAU * rt / 2.4)), 60);
        car.spin(0.125);                                   // 车停着：轮子不转
        if (still) hum.stand(); else hum.idle(rt);          // 站着轻晃，不原地迈步

        var pb = E.fade(t, 3.6, 0.5);
        brace.forEach(function (p) { E.drawOn(p, pb); });

        var pd = still ? 0 : E.fade(t, 6.8, 0.3);
        var at = K.pointOn(LOOP, (((rt - 6.8) / 4) % 1 + 1) % 1);
        dot.setAttribute('cx', at[0].toFixed(1));
        dot.setAttribute('cy', at[1].toFixed(1));
        E.op(dot, pd);

        boxes.forEach(function (b, i) {
          var p = E.fade(t, 4.9 + 0.2 * i, 0.3);
          E.pop(b.g, b.cx, MID_Y, p);
          E.pop(b.tg, b.cx, MID_Y, p);
          if (i) E.op(links[i - 1].g, p);
          // 光点走进这个盒子（被盒子盖住）时，盒子亮边 + 青光：k 从光点边缘碰到盒边起 0 → 1
          var depth = Math.min(at[0] - b.x, b.x + BOX_W - at[0], at[1] - 218, 308 - at[1]);
          var k = pd * E.clamp((depth + 9) / 24, 0, 1);
          K.paint(b.rect, null, k > 0 ? E.tint(E.INK, Math.round(85 * k), E.BORDER) : E.BORDER);
          b.rect.style.filter = k > 0 ? 'drop-shadow(0 0 10px ' + E.tint(E.ACC, Math.round(80 * k), 'transparent') + ')' : '';
          // 「拆开给你看」：四个「?」依次 1 → 1.15 → 1。规格写 9.7–10.5，但配音里那里还在念「就把这个圈」，
          // 「拆开给你看」实际在分镜 10.37–11.06，所以整体挪到 10.35–11.15
          var u = K.seg(t, 10.35 + 0.15 * i, 10.7 + 0.15 * i);
          E.scaleAt(b.q, b.cx, MID_Y, 1 + 0.15 * Math.sin(Math.PI * u));
          E.op(b.q, still ? 0 : 1);
          E.op(b.name, still ? 1 : 0);
        });
        K.paint(boxes[2].rect, still ? E.tint(E.ACC, 22) : E.S2);
        E.op(star, still ? 1 : 0);

        var pf = E.fade(t, 6.3, 0.5);
        E.drawOn(fb.line, pf);
        E.op(fb.head, K.seg(pf, 0.9, 1));
        caption.textContent = still ? '反馈：动完再看一眼，循环往复' : '一圈一圈，循环往复';
        E.op(caption, E.fade(t, 6.6, 0.4));
      }
    };
  };
})();
