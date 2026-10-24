// 002 第 3 幕：难点三·接触切换。下面侧视人形从左慢慢走到右（慢放），上面是两只脚的着地时间条（实心 = 着地），竖线跟着当前时刻走。
// 步态自己算：每步 1.95 s，前 30% 双脚着地、后 70% 后脚摆到前面；髋匀速前进、着地的脚钉在地上不打滑（腿用两节 IK 算），
// 像人一样直膝走：撑地的腿近乎伸直，后脚够不着时先抬脚跟，摆动腿先屈膝提脚、落地前伸直（见 L 那行），
// 所以时间条上真有「两只脚都踩着」的一小段。近侧（实线）是右脚、远侧（半透明）是左脚：人朝右走，观众看到的是他的右侧。
// 「两只脚都踩着」时时间条标出所有双脚着地段；「哪只脚踩在地上」着地脚底亮绿；「往上推」地面推力箭头（长短按分到的体重）；
// 「一抬一落」起每次抬脚 / 落脚闪一圈；「规律」起状态牌加上「规律①（两脚撑）/ 规律②（单脚撑）」；「控制」盒连到状态牌，「切换」起每次换状态闪一下。
// 时刻取自配音的逐词时间（分镜时间 ≈ 念到的时刻），见 ../ep002.py。
(function () {
  var E = window.EP002, K = E.K;
  // 2.5 起走 12 步，25.9 走完。起步时刻选成让第 3 步的双脚着地段落在 6.4–7.0（正念「两只脚都踩着」），紧接着单脚段 7.0–8.35（「只剩一只」）
  var WS = 2.5, T = 1.95, D = 0.3, NSTEP = 12, WE = WS + T * NSTEP;
  // 步长（本地单位）、放大、起点。撑地那条腿近乎伸直（膝弯同站姿的 4°）：髋高由前面那只撑地脚定，
  // 刚落地时最低、移到脚正上方时最高（起伏 ≈ 2.6）。后脚离髋最远 0.8 L = 32，直腿够不着就抬脚跟、脚尖着地（FOOT：踝到脚尖）；
  // 摆动腿先屈膝把脚提起（最高点在摆动的前 1/3，膝弯约 65°），落地前伸直、脚尖翘 TOE_UP°，落地后放平
  var L = 40, SC = 1.25, X0 = 110, FOOT = 16, TOE_UP = 6, DEG = Math.PI / 180;
  var CX0 = 120, CX1 = 770, ROW = { far: 49, near: 83 }, ROW_H = 22;
  var PILL = [230, 112, 220, 34];

  function cx(st) { return CX0 + (CX1 - CX0) * (st - WS) / (WE - WS); }
  // 当前时刻的两只脚（本地单位，起点髋在 0）：x、离地高度、是否着地；hip 为髋的 x；ds 为是否双脚着地；stance 为单脚撑时撑地的那只
  function gait(st) {
    var near = { x: L / 2, lift: 0, on: true }, far = { x: -L / 2, lift: 0, on: true };
    if (st < WS) return { near: near, far: far, hip: 0, ds: true };
    if (st >= WE) { near.x = (NSTEP + 0.5) * L; far.x = (NSTEP - 0.5) * L; return { near: near, far: far, hip: NSTEP * L, ds: true }; }
    var k = Math.floor((st - WS) / T), u = (st - WS - k * T) / T;
    var sw = k % 2 ? near : far, stn = k % 2 ? far : near;
    var s = E.clamp((u - D) / (1 - D), 0, 1);
    stn.x = (k + 0.5) * L;
    sw.x = (k - 0.5) * L + 2 * L * K.ease(s);
    var p = E.clamp(s / 0.85, 0, 1);
    sw.lift = s > 0 && s < 1 ? Math.max(13 * Math.sin(Math.PI * p) * (1 - 0.35 * p), 3 * Math.sin(Math.PI * s)) : 0;
    sw.on = !(s > 0 && s < 1);
    sw.s = s;
    return { near: near, far: far, hip: (k + u) * L, ds: sw.on, stance: k % 2 ? 'far' : 'near', k: k };
  }
  // 竖直推力箭头（线 + 三角头），set(x, len, op)：从地面下 392 + len 往上指到 392
  function upArrow(parent) {
    var g = E.el('g', {}, parent);
    var line = K.paint(E.el('line', { 'stroke-width': 5, 'stroke-linecap': 'round' }, g), null, E.GOOD);
    var head = K.paint(E.el('path', {}, g), E.GOOD);
    return {
      g: g,
      set: function (x, len, o) {
        var top = 392, bot = top + len;
        line.setAttribute('x1', x); line.setAttribute('x2', x); line.setAttribute('y1', bot.toFixed(1)); line.setAttribute('y2', top + 8);
        head.setAttribute('d', 'M ' + x + ' ' + top + ' L ' + (x - 9) + ' ' + (top + 13) + ' L ' + (x + 9) + ' ' + (top + 13) + ' Z');
        E.op(g, o);
      }
    };
  }

  E.scenes[2] = {
    title: '难点三：接触切换，脚轮流着地',
    dur: 26.5,
    cues: [
      { at: 0, s: '难点三：**接触切换**' },
      { at: 3.0, s: '两只脚轮流着地：两脚撑 ↔ 一只脚撑' },
      { at: 9.7, s: '哪只脚着地，地面就从哪儿往上推' },
      { at: 16.5, s: '力一变，身体怎么动的规律也换一套' },
      { at: 20.6, s: '**控制**：时刻清楚哪只脚着地，跟着切换' }
    ],
    build: function (K) {
      var s = K.sceneSvg('难点三接触切换：人走路时两只脚轮流着地，一会儿两只脚都踩着，一会儿只剩一只；哪只脚踩地，地面就从哪儿往上推，力一变身体怎么动的规律也换一套，控制要时刻清楚哪只脚着地、跟着切换');

      /* ── 着地时间条 ── */
      var chart = E.el('g', {}, s);
      var dsShade = E.el('g', {}, chart);
      for (var k = 0; k <= NSTEP; k++) {
        var a = WS + k * T, b = Math.min(WE, a + D * T);
        if (a >= WE) break;
        K.paint(E.el('rect', { x: cx(a).toFixed(1), y: ROW.far - ROW_H / 2 - 6, width: (cx(b) - cx(a)).toFixed(1), height: ROW.near - ROW.far + ROW_H + 12, rx: 3 }, dsShade), E.tint(E.GOOD, 22, 'transparent'));
      }
      var rows = {};
      [['far', '左脚'], ['near', '右脚']].forEach(function (rw) {
        var y = ROW[rw[0]];
        K.paint(E.el('rect', { x: CX0, y: y - ROW_H / 2, width: CX1 - CX0, height: ROW_H, rx: 4, 'stroke-width': 1.5 }, chart), E.S2, E.BORDER);
        var bars = E.el('g', {}, chart);
        // 着地段 = 走路区间去掉这只脚的摆动段；远脚（左）在第 0、2、4… 步摆，近脚（右）在第 1、3、5… 步摆
        var cur = WS;
        for (var j = rw[0] === 'far' ? 0 : 1; j < NSTEP; j += 2) {
          var s0 = WS + j * T + D * T, s1 = WS + (j + 1) * T;
          K.paint(E.el('rect', { x: cx(cur).toFixed(1), y: y - ROW_H / 2 + 2, width: Math.max(0, cx(s0) - cx(cur)).toFixed(1), height: ROW_H - 4, rx: 3 }, bars), E.GOOD);
          cur = s1;
        }
        K.paint(E.el('rect', { x: cx(cur).toFixed(1), y: y - ROW_H / 2 + 2, width: (CX1 - cx(cur)).toFixed(1), height: ROW_H - 4, rx: 3 }, bars), E.GOOD);
        rows[rw[0]] = { bars: bars, label: E.txt(chart, CX0 - 12, y + 7, rw[1], 20, rw[0] === 'near' ? E.INK : E.INK2, 'end', 700) };
      });
      var head = K.paint(E.el('line', { y1: ROW.far - ROW_H / 2 - 8, y2: ROW.near + ROW_H / 2 + 8, 'stroke-width': 2.5, 'stroke-linecap': 'round' }, chart), null, E.INK);

      /* ── 状态牌 + 控制盒 ── */
      var pill = E.el('g', {}, s);
      var pillRect = K.paint(E.el('rect', { x: PILL[0], y: PILL[1], width: PILL[2], height: PILL[3], rx: 17, 'stroke-width': 2 }, pill), E.S2, E.GOOD);
      var pillText = E.txt(pill, PILL[0] + PILL[2] / 2, PILL[1] + 24, '', 20, E.INK, 'middle', 700);
      var ctrl = E.el('g', {}, s);
      var ctrlRect = K.paint(E.el('rect', { x: 590, y: 110, width: 140, height: 38, rx: 10, 'stroke-width': 2.5 }, ctrl), E.tint(E.ACC, 20), E.ACC);
      E.txt(ctrl, 660, 136, '控制', 22, E.ACC, 'middle', 900);
      var watch = E.arrow(s, [[586, 129], [PILL[0] + PILL[2] + 8, 129]], E.ACC, 2.5, { dash: '6 5', head: 10 });

      /* ── 地面 + 人形 ── */
      K.paint(E.el('line', { x1: 20, y1: 388, x2: 780, y2: 388, 'stroke-width': 3 }, s), null, E.BORDER);
      var marks = { near: K.paint(E.el('rect', { height: 5, rx: 2 }, s), E.GOOD), far: K.paint(E.el('rect', { height: 5, rx: 2 }, s), E.GOOD) };
      var rings = { near: K.paint(E.el('circle', { fill: 'none', 'stroke-width': 2.5 }, s), null, E.GOOD), far: K.paint(E.el('circle', { fill: 'none', 'stroke-width': 2.5 }, s), null, E.GOOD) };
      var grf = { near: upArrow(s), far: upArrow(s) };
      var r = E.robot(s, X0);

      // 某只脚最近一次抬起 / 落下（摆动段的起止）距 st 多久
      function sinceEvent(st, which) {
        var best = 99;
        for (var j = which === 'far' ? 0 : 1; j < NSTEP; j += 2) {
          [WS + j * T + D * T, WS + (j + 1) * T].forEach(function (ev) { if (st >= ev) best = Math.min(best, st - ev); });
        }
        return best;
      }
      function sinceSwitch(st) {
        if (st < WS) return 99;
        var k = Math.floor((st - WS) / T), u = st - WS - k * T;
        return u < D * T ? u : u - D * T;
      }

      return {
        el: s,
        draw: function (st, t) {
          var fade = E.fade, G = gait(st);

          /* 人形：髋匀速前进，直膝走（见 L 那行），两条腿用 IK 让踝落在算好的位置上；
             手臂跟近腿反着摆——按两条大腿的角差摆，绕站姿 8° 对称 */
          var X = X0 + G.hip * SC;
          // r.ankleY：G1 站直时踝关节的本地 y（≈ 381，踝离脚底 5 cm）；LEG：髋到踝，近乎伸直
          var LEG = r.ankleY - 302, d = { near: G.near.x - G.hip, far: G.far.x - G.hip };
          var lead = G.ds ? Math.max(d.near, d.far) : d[G.stance];
          var V = Math.sqrt(LEG * LEG - lead * lead), DY = LEG - V;         // 髋到踝的竖直距离；身子比站直时低多少
          var walking = st >= WS + T && st < WE, lift = {}, tilt = {}, ang = {};
          ['near', 'far'].forEach(function (w) {
            var f = G[w], dx = d[w];
            // 直腿够不着（后脚、刚离地的摆动脚）就抬脚跟：踝要抬高 heel，脚尖着地，脚掌倾角按它算
            var heel = Math.max(0, V - Math.sqrt(Math.max(0, LEG * LEG - dx * dx)));
            var toeDown = Math.asin(Math.min(1, heel / FOOT)) / DEG;
            lift[w] = Math.max(f.lift, heel);
            if (f.on) {
              // 刚落地的前脚：脚尖还翘着，前半段双脚撑里放平
              var q = walking && dx === lead && G.ds ? E.clamp((L / 2 - dx) / (0.15 * L), 0, 1) : 1;
              tilt[w] = heel > 0 ? toeDown : -TOE_UP * (1 - q);
            } else {
              // 摆动：离地时的脚尖朝下在前 40% 收平，之后慢慢翘起脚尖，准备脚跟落地
              var off = Math.asin(Math.min(1, Math.max(0, Math.sqrt(LEG * LEG - Math.pow((0.5 - D) * L, 2))
                - Math.sqrt(LEG * LEG - Math.pow((0.5 + D) * L, 2))) / FOOT)) / DEG;
              tilt[w] = f.s < 0.4 ? Math.max(toeDown, off * (1 - f.s / 0.4)) : -TOE_UP * (f.s - 0.4) / 0.6;
            }
            ang[w] = r.legIK([0, 302], [dx, r.ankleY - DY - lift[w]]);
          });
          r.legs(ang.near, ang.far, tilt.near, tilt.far);
          var sw = 8 - 0.35 * (ang.near[0] - ang.far[0]);
          r.arm([sw, sw + 12]);
          r.place({ x: X, dy: DY * SC, scale: SC });
          E.op(r.g, fade(st, 0, 0.6));

          /* 时间条：4.0「轮流着地」淡入；6.0「两只脚都踩着」双脚着地段亮出来；竖线跟着 st 走 */
          E.op(chart, fade(st, 0.3, 0.5));
          ['near', 'far'].forEach(function (w) { E.op(rows[w].bars, fade(st, 4.0, 0.6)); });
          E.op(dsShade, fade(st, 6.0, 0.5));
          var hx = Math.max(CX0, Math.min(CX1, cx(st)));
          head.setAttribute('x1', hx.toFixed(1)); head.setAttribute('x2', hx.toFixed(1));
          E.op(head, fade(st, WS, 0.3));

          /* 状态牌：4.0 起显示当前状态；18.2「规律」起加上规律①/②；19.2「换一套」起每次切换闪一下 */
          var law = st >= 18.2;
          pillText.textContent = (G.ds ? '两只脚都着地' : G.stance === 'near' ? '只剩右脚着地' : '只剩左脚着地') + (law ? (G.ds ? ' · 规律①' : ' · 规律②') : '');
          var pw = law ? 260 : 200;
          pillRect.setAttribute('width', pw); pillRect.setAttribute('x', PILL[0] + (PILL[2] - pw) / 2);
          var flash = st >= 19.2 ? Math.max(0, 1 - sinceSwitch(st) / 0.35) : 0;
          pillRect.style.strokeWidth = (2 + 2 * flash).toFixed(2);
          K.paint(pillRect, flash > 0 ? E.tint(E.GOOD, Math.round(30 * flash)) : E.S2, G.ds ? E.GOOD : E.INK2);
          E.op(pill, fade(st, 4.0, 0.4));

          /* 10.2「哪只脚踩在地上」着地脚底亮绿；12.4「往上推」推力箭头（双脚撑时按髋离两脚的远近分体重）；13.2「一抬一落」起闪圈 */
          var span = Math.abs(G.near.x - G.far.x) || 1;
          ['near', 'far'].forEach(function (w) {
            var f = G[w], fx = X0 + f.x * SC;
            var m = marks[w];
            m.setAttribute('x', (fx - 4 * SC).toFixed(1)); m.setAttribute('y', 389); m.setAttribute('width', (22 * SC).toFixed(1));
            E.op(m, f.on ? fade(st, 10.2, 0.3) * (w === 'far' ? 0.6 : 1) : 0);
            var load = !f.on ? 0 : G.ds ? 1 - Math.abs(G.hip - f.x) / span : 1;
            grf[w].set((fx + 7 * SC).toFixed(1), 14 + 14 * load, (f.on ? fade(st, 12.4, 0.4) * (w === 'far' ? 0.6 : 1) : 0).toFixed(3));
            var ev = st >= 13.2 ? sinceEvent(st, w) : 99, ru = ev / 0.5;
            rings[w].setAttribute('cx', (fx + 7 * SC).toFixed(1)); rings[w].setAttribute('cy', (388 - (388 - r.ankleY + lift[w]) * SC).toFixed(1));
            rings[w].setAttribute('r', (6 + 16 * ru).toFixed(1));
            E.op(rings[w], ru < 1 ? (0.9 * (1 - ru)).toFixed(3) : 0);
          });

          /* 21.0「控制」盒；21.5 虚线连到状态牌；24.8「切换」起每次换状态闪一下 */
          E.pop(ctrl, 660, 129, fade(st, 21.0, 0.3), 0.8);
          E.drawOn(watch.line, fade(st, 21.5, 0.5));
          E.op(watch.head, K.seg(fade(st, 21.5, 0.5), 0.9, 1));
          var cf = st >= 24.8 ? Math.max(0, 1 - sinceSwitch(st) / 0.4) : 0;
          ctrlRect.style.filter = cf > 0 ? 'drop-shadow(0 0 ' + (10 * cf).toFixed(1) + 'px ' + E.ACC + ')' : '';
          ctrlRect.style.strokeWidth = (2.5 + 2 * cf).toFixed(2);
        }
      };
    }
  };
})();
