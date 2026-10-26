// 008 第 1 幕：变量——由我们来选的数。上面是跑道：侧视 G1 站在起点，4 格外是终点线。
// 句 1「终点线」终点线亮一下，「四格」起点到终点线画尺寸线「还有 4 格」；「一步」尺寸线收起，换成弧形箭头 + 虚线脚印，
//    「迈多远」脚印在 1–3 格之间来回挪（还没定）；
// 句 2 脚印按念到的词停到 0、1、2、1.5 格，弧上的字跟着变；「这个」跑道下面出一根滑杆（0–4），绿色滑块就是 u，
//    「选」滑块鼓一下，「变量」左边弹出「变量 u」；
// 句 3 滑杆收起，换成三块牌：「关节」关节转的角度、「电机」电机出的力、「下一脚」下一脚踩哪（这时跑道上的脚印鼓一下）；
// 句 4 跑道和牌收起，「人形」中间出正面 G1，「全身」29 个关节点按腿 → 腰 → 胳膊依次点亮，「三十」右边弹出「29 个关节」，
//    「变量」下面再出「= 29 个变量」。
// 时刻取自配音的逐词时间（分镜时间 = 念到这个词的时刻），见 ../ep008.py。
(function () {
  var E = window.EP008, K = E.K;
  var X0 = 250, Y = 262, W = 115, SY = 344;
  var T = {
    finish: 3.2, four: 3.62, step: 4.43, far: 4.99,
    stop: 6.95, keys: [[0, 0], [7.92, 1], [8.5, 2], [9.52, 1.5]],
    slider: 10.69, pick: 11.54, varTag: 12.75,
    sliderOut: 15.0, chips: [16.19, 17.43, 18.66], footBump: 19.0,
    out: 21.45, hum: 23.02, dots: 23.37, tally: 24.48, vars: 26.18
  };

  E.scenes[0] = {
    title: '变量：由我们来选的数',
    dur: 27.91,
    cues: [
      { at: 0, s: '离终点线 4 格：这一步迈多远最好？' },
      { at: 6.89, s: '可以迈 0、1、2 格，也可以迈 1.5 格' },
      { at: 10.69, s: '由我们来选的数 → **变量**，记作 $u$' },
      { at: 14.33, s: '机器人里的变量：关节转的角度、电机出的力、下一脚踩哪' },
      { at: 21.29, s: '变量不止一个：G1 全身 29 个关节 → 29 个变量' }
    ],
    build: function (K) {
      var s = K.sceneSvg('变量：机器人离终点线还有四格，这一步可以迈零格、一格、两格，也可以迈一格半，这个由我们来选的数叫变量 u；机器人里的变量常常是关节转的角度、电机出的力、下一脚踩在哪；人形全身有 29 个关节，就是 29 个变量');

      /* ══ 跑道 ══ */
      var top = E.el('g', {}, s);
      var tr = E.track(top, { x0: X0, y: Y, w: W, scale: 1.3 });

      /* ══ 滑杆：0–4，绿色滑块 u ══ */
      var sl = E.el('g', {}, top);
      K.paint(E.el('line', { x1: tr.X(0), y1: SY, x2: tr.X(4), y2: SY, 'stroke-width': 8, 'stroke-linecap': 'round' }, sl), null, E.S2);
      var fill = K.paint(E.el('line', { x1: tr.X(0), y1: SY, x2: tr.X(0), y2: SY, 'stroke-width': 8, 'stroke-linecap': 'round' }, sl), null, E.GOOD);
      fill.style.strokeOpacity = 0.6;
      var knob = E.el('g', {}, sl);
      var knobC = K.paint(E.el('circle', { r: 16, 'stroke-width': 3 }, knob), E.GOOD, 'var(--demo-surface)');
      var knobT = E.txt(knob, 0, 0, 'u', 22, 'var(--demo-surface)', 'middle', 900);
      knobT.style.fontStyle = 'italic';
      var varTag = E.tag(sl, 60, SY - 22, 132, 44, '变量 u', E.GOOD, { size: 24, ink: E.GOOD, weight: 900, rx: 14, sw: 3 });

      /* ══ 三块牌 ══ */
      var chipTexts = ['关节转的角度', '电机出的力', '下一脚踩哪'];
      var chips = chipTexts.map(function (c, i) {
        var w = 196, x = 70 + 228 * i;
        var tg = E.tag(top, x, SY - 24, w, 48, c, E.GOOD, { size: 22, ink: E.INK, weight: 900, rx: 14, sw: 2.5 });
        return { g: tg.g, cx: x + w / 2, rect: tg.rect };
      });

      /* ══ 正面 G1 + 29 个关节点，右边计数 ══ */
      var humG = E.el('g', {}, s);
      var hum = window.Robot3D.g1Front(humG, { dots: true });
      hum.place(300, 1.0, 404);
      var tally = E.el('g', {}, s);
      E.txt(tally, 590, 186, '29', 64, E.INK, 'middle', 900);
      E.txt(tally, 590, 222, '个关节', 24, E.INK2, 'middle', 700);
      var vars = E.el('g', {}, s);
      E.arrow(vars, [[590, 240], [590, 282]], E.INK2, 3, { head: 12 });
      E.txt(vars, 590, 322, '29 个变量', 34, E.GOOD, 'middle', 900);

      function osc(t) { return 2 + 0.9 * Math.sin(2 * Math.PI * (t - T.step) / 2.4); }

      return {
        el: s,
        draw: function (st, t) {
          var f = E.fade;
          var topOn = f(st, 0, 0.5) * (1 - f(st, T.out, 0.5));
          E.op(top, topOn);

          /* 句 1：终点线、还有 4 格 */
          var fb = E.bump(st, T.finish, 0.8);
          E.scaleAt(tr.finish, tr.X(4), Y - 40, 1 + 0.12 * fb);
          E.glow(tr.finish, E.HI, fb > 0.05);

          /* 脚印与弧：一步之前不显示；句 1 末尾在 1–3 格之间挪，句 2 按念到的格数停下 */
          var keyed = E.steps(st, T.keys, 0.4);
          var u = E.lerp(osc(st < T.stop ? t : T.stop), keyed, f(st, T.stop, 0.35));
          var unsure = st < T.stop + 0.2;
          tr.setU(u, unsure ? '迈多远？' : undefined, '还有 4 格');
          var stepOn = f(st, T.step, 0.4);
          E.op(tr.step, stepOn); E.op(tr.foot, stepOn);
          tr.footE.style.strokeDasharray = unsure ? '5 4' : '';
          // 还有 4 格：四格时出、一步时收（尺寸线画在起点到终点线之间，不跟脚印走）
          tr.rem.firstChild.setAttribute('d', 'M ' + tr.X(0) + ' ' + (Y - 33) + ' V ' + (Y - 19) + ' M ' + tr.X(0) + ' ' + (Y - 26) + ' H ' + tr.X(4) + ' M ' + tr.X(4) + ' ' + (Y - 33) + ' V ' + (Y - 19));
          tr.remLab.setAttribute('x', ((tr.X(0) + tr.X(4)) / 2).toFixed(1));
          E.op(tr.rem, f(st, T.four, 0.3) * (1 - f(st, T.step - 0.1, 0.3)));
          var fbump = E.bump(st, T.footBump, 0.9);
          tr.footE.setAttribute('rx', (20 + 6 * fbump).toFixed(1));
          E.glow(tr.foot, E.GOOD, fbump > 0.05);

          /* 句 2：滑杆 */
          var slOn = f(st, T.slider, 0.4) * (1 - f(st, T.sliderOut, 0.4));
          E.op(sl, slOn);
          var kx = tr.X(u);
          fill.setAttribute('x2', kx.toFixed(1));
          var kb = E.bump(st, T.pick, 0.8);
          knob.setAttribute('transform', 'translate(' + kx.toFixed(1) + ' ' + SY + ') scale(' + (1 + 0.25 * kb).toFixed(3) + ')');
          knobT.setAttribute('y', '7');
          E.glow(knobC, E.GOOD, kb > 0.05);
          E.pop(varTag.g, 126, SY, f(st, T.varTag, 0.3), 0.8);

          /* 句 3：三块牌 */
          chips.forEach(function (c, i) {
            E.pop(c.g, c.cx, SY, f(st, T.chips[i], 0.35), 0.8);
            c.rect.style.strokeWidth = 2.5 + (i === 2 ? 2 * fbump : 0);
          });

          /* 句 4：正面 G1、29 个关节点 */
          E.op(humG, f(st, T.hum, 0.5));
          var n = 0;
          ['legs', 'waist', 'arms'].forEach(function (k) {
            hum.groups[k].forEach(function (d) {
              var q = f(st, T.dots + 0.035 * n++, 0.2);
              d.style.fill = q > 0.5 ? E.GOOD : E.INK;
              d.setAttribute('r', (4.5 + 2 * q).toFixed(2));
              E.op(d, (0.4 + 0.6 * q).toFixed(3));
            });
          });
          E.pop(tally, 590, 200, f(st, T.tally, 0.35), 0.7);
          E.pop(vars, 590, 300, f(st, T.vars, 0.35), 0.8);
        }
      };
    }
  };
})();
