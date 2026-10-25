// 003 第 6 幕：后面的课，全用它们来写。正面 G1 站在中间，左边一栏「向量」、右边一栏「矩阵」，念到一样就从 G1 身上对应的部位连一条虚线到那一栏的卡片：
// ① 「这两样东西」两栏的栏头弹出；
// ② 「手在哪」画面左手一点 → 位置 [x; y; z]，「走多快」骨盆一支速度箭头 → 速度，「受多大的力」画面左脚底一支朝上的力 → 力，「写成向量」三张卡一起亮一下；
// ③ 「身子朝哪」胸口画一个三轴坐标架（x 红、y 绿、z 蓝，x 朝着镜头所以画短、斜着）→ 朝向：3×3 旋转矩阵，「第五集」弹出「第 5 集」；
//    「朝向加上位置」从两脚中间的地面点到坐标架画一支位置箭头 p → 位姿：4×4 矩阵 [R p; 0 1]，「第七集」弹出「第 7 集」；
// ④ 「关节转多快」画面右肩（G1 的左肩侧摆关节）画一段旋转小弧，胳膊匀速往外抬（侧摆轴正好朝着镜头，手在画面里绕肩画圆），
//    「手就走多快」手上一支切向的速度箭头跟着走；「中间也隔着一个矩阵」手到右栏连线，「一个矩阵」弹出雅可比卡，「第十三集」弹出「第 13 集」。
// 时刻取自配音的逐词时间（分镜时间 ≈ 念到的时刻），见 ../ep003.py。
(function () {
  var E = window.EP003, K = E.K;
  var TAU = Math.PI * 2, DEG = Math.PI / 180;
  var HX = 400, HS = 0.95, GY = 362;                       // G1：两脚中间的地面点 (HX, GY)，缩放 HS
  var CW = 250, CH = 104, LX = 10, RX = 540, CY = [64, 176, 288];
  var HEAD_AT = 1.88;                                       // 「两样」
  var VEC_AT = [3.69, 4.57, 5.45], AS_VEC = 6.47;           // 「手」「走」「受」；「写成向量」
  var ORIENT = 8.51, EP5 = 11.84, POSE = 12.81, P_AT = 13.46, EP7 = 15.8;
  var JOINT = 17.76, LIFT = [18.0, 24.4], HANDV = 18.98, MID = 20.25, JCARD = 21.06, JNAME = 22.14, EP13 = 23.0;
  var AB = [12, 40];                                        // 画面右臂外张角（度）：抬臂前后

  function bump(st, a, d) { var u = K.seg(st, a, a + d); return u > 0 && u < 1 ? Math.sin(Math.PI * u) : 0; }

  // 一张卡：标题（24px 粗）+ 一行说明 + 右侧公式；pill 是「第 N 集」小标签（可选）
  function card(parent, x, y, border, title, titleColor, sub, tex, texColor, texSize, pill) {
    var og = E.el('g', {}, parent), g = E.el('g', {}, og);
    var rect = K.paint(E.el('rect', { x: x, y: y, width: CW, height: CH, rx: 14, 'stroke-width': 2.5 }, g), E.S1, border);
    var t = E.txt(g, x + 16, y + 36, title, 24, titleColor, 'start', 900);
    E.txt(g, x + 16, y + 64, sub, 19, E.INK2, 'start', 700);
    var f = tex ? E.tex(g, x + 192, y + 52, tex, texSize || 20, texColor, 'middle', { w: 110, h: 90 }) : null;
    var p = pill ? E.tag(g, x + 16, y + 72, pill.length > 5 ? 86 : 76, 26, pill, E.VIOLET, { size: 18, ink: E.VIOLET, weight: 700 }) : null;
    return { og: og, g: g, rect: rect, title: t, tex: f, pill: p, cx: x + CW / 2, cy: y + CH / 2 };
  }
  // 圆弧箭头：绕 (cx, cy) 半径 r，从角 a0 画到 a1（度，画面坐标 y 朝下；a1 < a0 即画面上逆时针），p 0 → 1 画出来
  function arcArrow(parent, color) {
    var g = E.el('g', {}, parent);
    var line = K.paint(E.el('path', { fill: 'none', 'stroke-width': 3, 'stroke-linecap': 'round' }, g), null, color);
    var head = K.paint(E.el('path', {}, g), color);
    return {
      g: g,
      set: function (cx, cy, r, a0, a1, p) {
        var a = a0 + (a1 - a0) * p, A0 = a0 * DEG, A = a * DEG;
        if (p <= 0.01) { line.setAttribute('d', ''); head.setAttribute('d', ''); return; }
        var x0 = cx + r * Math.cos(A0), y0 = cy + r * Math.sin(A0), x1 = cx + r * Math.cos(A), y1 = cy + r * Math.sin(A);
        line.setAttribute('d', 'M ' + x0.toFixed(1) + ' ' + y0.toFixed(1) + ' A ' + r + ' ' + r + ' 0 0 ' + (a1 < a0 ? 0 : 1) + ' ' + x1.toFixed(1) + ' ' + y1.toFixed(1));
        var dir = a1 < a0 ? -1 : 1, tx = -Math.sin(A) * dir, ty = Math.cos(A) * dir, L = 11, W = 7;   // 前进方向的切线
        head.setAttribute('d', 'M ' + (x1 + tx * 4).toFixed(1) + ' ' + (y1 + ty * 4).toFixed(1) +
          ' L ' + (x1 - tx * L + ty * W).toFixed(1) + ' ' + (y1 - ty * L - tx * W).toFixed(1) +
          ' L ' + (x1 - tx * L - ty * W).toFixed(1) + ' ' + (y1 - ty * L + tx * W).toFixed(1) + ' Z');
      }
    };
  }

  E.scenes[5] = {
    title: '后面的课，全用它们来写',
    dur: 24.84,
    cues: [
      { at: 0, s: '后面全用**向量**和**矩阵**来写' },
      { at: 3.6, s: '位置、速度、力 → **向量**' },
      { at: 8.42, s: '朝向 → 3×3 旋转矩阵；位姿 → 4×4 **矩阵**' },
      { at: 17.66, s: '关节速度 → 手的速度：雅可比**矩阵**' }
    ],
    build: function (K) {
      var s = K.sceneSvg('后面的课全用向量和矩阵来写：手在哪、走多快、受多大的力写成向量；身子朝哪是三乘三的旋转矩阵（第五集），朝向加位置是四乘四的矩阵（第七集）；关节转多快到手走多快，中间隔着雅可比矩阵（第十三集）');
      var H = E.HEX;

      /* ── 两栏栏头 + 六张卡 ── */
      var heads = [E.tag(s, LX + CW / 2 - 60, 12, 120, 38, '向量', E.ACC, { size: 24, ink: E.ACC, weight: 900 }),
        E.tag(s, RX + CW / 2 - 60, 12, 120, 38, '矩阵', E.VIOLET, { size: 24, ink: E.VIOLET, weight: 900 })];
      var vcards = [
        card(s, LX, CY[0], E.ACC, '位置', E.ACC, '手在哪', E.col(['x', 'y', 'z']), E.ACC),
        card(s, LX, CY[1], E.ACC, '速度', E.GOOD, '往哪走、多快', E.col(['v_x', 'v_y', 'v_z']), E.GOOD),
        card(s, LX, CY[2], E.ACC, '力', E.WARN, '往哪推、多大', E.col(['f_x', 'f_y', 'f_z']), E.WARN)
      ];
      var B = '\\bullet', dots3 = '\\begin{bmatrix}' + [B, B, B].join('&') + '\\\\' + [B, B, B].join('&') + '\\\\' + [B, B, B].join('&') + '\\end{bmatrix}';
      var mcards = [
        card(s, RX, CY[0], E.VIOLET, '朝向', E.VIOLET, '3×3 旋转矩阵', dots3, E.VIOLET, 20, '第 5 集'),
        card(s, RX, CY[1], E.VIOLET, '位姿', E.VIOLET, '4×4 矩阵',
          '\\begin{bmatrix}\\textcolor{' + H.violet + '}{R}&\\textcolor{' + H.hi + '}{p}\\\\0&1\\end{bmatrix}', E.INK, 22, '第 7 集'),
        card(s, RX, CY[2], E.VIOLET, '雅可比', E.VIOLET, '关节速度 → 手速度', null, null, 0, '第 13 集')
      ];
      var jg = E.el('g', {}, mcards[2].g);
      E.tex(jg, RX + 214, CY[2] + 38, 'J', 36, E.VIOLET);

      /* ── 正面 G1 ── */
      var connG = E.el('g', {}, s);                           // 连线压在机器人下面
      var hum = E.front(s);
      hum.place(HX, HS, GY);
      var ov = E.el('g', {}, s);                              // 机器人身上的标注
      var handDot = K.paint(E.el('circle', { r: 7, 'stroke-width': 2.5 }, ov), E.ACC, E.S1);
      var handRing = K.paint(E.el('circle', { r: 7, fill: 'none', 'stroke-width': 2.5 }, ov), null, E.ACC);
      var vel = E.vec(ov, E.GOOD, 4.5, { head: 14 });
      var force = E.vec(ov, E.WARN, 4.5, { head: 14 });
      var triad = [E.vec(ov, E.BAD, 3.5, { head: 10 }), E.vec(ov, E.GOOD, 3.5, { head: 10 }), E.vec(ov, E.ACC, 3.5, { head: 10 })];
      var triO = K.paint(E.el('circle', { r: 4 }, ov), E.INK);
      var pvec = E.vec(ov, E.HI, 3.5, { head: 12, dash: '7 5' });          // 位置 p 用亮黄：和蓝色的 z 轴分开
      var pLabel = E.txt(ov, 0, 0, 'p', 22, E.HI, 'start', 700);
      pLabel.style.fontStyle = 'italic';
      var arc = arcArrow(ov, E.HI);
      var jointDot = K.paint(E.el('circle', { r: 6, 'stroke-width': 2.5 }, ov), E.HI, E.S1);
      var handV = E.vec(ov, E.GOOD, 4.5, { head: 14 });
      // 雅可比卡的连线从手上连过去：手在动，这条每帧重算（drawOn 只认固定的线），按 p 从手往卡片伸
      var jconn = K.paint(E.el('path', { fill: 'none', 'stroke-width': 2.2, 'stroke-dasharray': '6 5', 'stroke-linecap': 'round' }, connG), null, E.VIOLET);

      var conns = null, A = null;
      function at(name, local) { return hum.world(hum.view.at(name, local)); }
      function connector(p, q, color) {
        var l = E.el('path', { d: 'M ' + p[0].toFixed(1) + ' ' + p[1].toFixed(1) + ' L ' + q[0].toFixed(1) + ' ' + q[1].toFixed(1),
          fill: 'none', 'stroke-width': 2.2, 'stroke-dasharray': '6 5', 'stroke-linecap': 'round' }, connG);
        K.paint(l, null, color);
        return l;
      }

      return {
        el: s,
        draw: function (st, t) {
          var fade = E.fade;
          var w = Math.sin(TAU * t / 3.2);
          // 画面右臂（G1 的左臂）：④ 里从外张 12° 匀速抬到 40°；平时两臂随站姿轻晃
          var ab = AB[0] + (AB[1] - AB[0]) * K.seg(st, LIFT[0], LIFT[1]);
          hum.pose({ shift: 2 * w, arm: [[8 + 2 * w, 4 + 3 * w], [ab, ab]] });

          /* 第一次画（挂上页面后）时按站姿量一次各部位，连线定死在这些点上 */
          if (!A && s.isConnected) {
            A = {
              hand: at('right_rubber_hand'), pelvis: at('pelvis'), foot: at('right_ankle_roll_link'),
              chest: at('torso_link', [0, 0, 0.2]), ground: [HX, GY]
            };
            var vt = [A.pelvis[0] - 92, A.pelvis[1] + 6];
            conns = [
              connector(A.hand, [LX + CW + 4, CY[0] + CH / 2], E.ACC),
              connector(vt, [LX + CW + 4, CY[1] + CH / 2], E.GOOD),
              connector([A.foot[0] - 8, GY + 30], [LX + CW + 4, CY[2] + CH / 2], E.WARN),
              connector([A.chest[0] + 36, A.chest[1]], [RX - 4, CY[0] + CH / 2], E.VIOLET),
              connector([A.ground[0] + 10, (A.ground[1] + A.chest[1]) / 2 + 30], [RX - 4, CY[1] + CH / 2], E.VIOLET)
            ];
            conns.forEach(function (c) { E.drawOn(c, 0); });
          }

          /* ① 0.2 G1 出现；1.88「两样」两栏栏头依次弹出 */
          E.op(hum.g, fade(st, 0.1, 0.5));
          heads.forEach(function (h, i) { E.pop(h.g, i ? RX + CW / 2 : LX + CW / 2, 31, fade(st, HEAD_AT + 0.15 * i, 0.3), 0.7); });

          /* ② 念到「手 / 走 / 受」：身上的标注先出来，连线画过去，卡片弹出；6.47「写成向量」三张卡一起亮一下 */
          var hp = at('right_rubber_hand'), pv = at('pelvis');
          handDot.setAttribute('cx', hp[0].toFixed(1)); handDot.setAttribute('cy', hp[1].toFixed(1));
          E.op(handDot, fade(st, VEC_AT[0], 0.2));
          var u = K.seg(st, VEC_AT[0], VEC_AT[0] + 0.7);
          handRing.setAttribute('cx', hp[0].toFixed(1)); handRing.setAttribute('cy', hp[1].toFixed(1));
          handRing.setAttribute('r', (7 + 16 * u).toFixed(2));
          E.op(handRing, u > 0 && u < 1 ? (1 - u).toFixed(3) : 0);
          var pvl = fade(st, VEC_AT[1], 0.35);
          vel.set(pv[0] - 14, pv[1] + 6, pv[0] - 14 - 78 * pvl, pv[1] + 6);
          E.op(vel.g, pvl > 0 ? 1 : 0);
          var pf = fade(st, VEC_AT[2], 0.35), fx = A ? A.foot[0] - 8 : hp[0];
          force.set(fx, GY + 44, fx, GY + 44 - 50 * pf);
          E.op(force.g, pf > 0 ? 1 : 0);
          var glowV = bump(st, AS_VEC, 0.8);
          vcards.forEach(function (c, i) {
            E.pop(c.g, c.cx, c.cy, fade(st, VEC_AT[i] + 0.25, 0.3), 0.85);
            E.scaleAt(c.og, c.cx, c.cy, 1 + 0.05 * glowV);
            c.rect.style.strokeWidth = (2.5 + 2 * glowV).toFixed(2);
            c.rect.style.filter = glowV > 0.05 ? 'drop-shadow(0 0 ' + (8 * glowV).toFixed(1) + 'px ' + E.ACC + ')' : '';
          });
          E.scaleAt(heads[0].g, LX + CW / 2, 31, 1 + 0.1 * glowV);
          if (conns) [0, 1, 2].forEach(function (i) { E.drawOn(conns[i], fade(st, VEC_AT[i] + 0.1, 0.35)); });

          /* ③ 8.51「身子朝哪」胸口三轴坐标架 → 朝向卡；11.84「第五集」；12.81「朝向加上位置」位姿卡，13.46「位置」地面点 → 坐标架的位置箭头 p；15.8「第七集」 */
          var ch = at('torso_link', [0, 0, 0.2]), ptri = fade(st, ORIENT, 0.4);
          triad[0].set(ch[0], ch[1], ch[0] - 16 * ptri, ch[1] + 14 * ptri);     // x：朝着镜头，画短、斜着
          triad[1].set(ch[0], ch[1], ch[0] + 34 * ptri, ch[1]);                  // y：G1 的左边 = 画面右边
          triad[2].set(ch[0], ch[1], ch[0], ch[1] - 34 * ptri);                  // z：朝上
          triad.forEach(function (a) { E.op(a.g, ptri > 0 ? 1 : 0); });
          triO.setAttribute('cx', ch[0].toFixed(1)); triO.setAttribute('cy', ch[1].toFixed(1));
          E.op(triO, ptri);
          var pp = fade(st, P_AT, 0.45);
          pvec.set(HX, GY, HX + (ch[0] - HX) * pp, GY + (ch[1] + 4 - GY) * pp);
          E.op(pvec.g, pp > 0 ? 1 : 0);
          pLabel.setAttribute('x', (HX + 8).toFixed(1)); pLabel.setAttribute('y', ((GY + ch[1]) / 2 + 52).toFixed(1));
          E.op(pLabel, fade(st, P_AT + 0.3, 0.3));
          E.pop(mcards[0].g, mcards[0].cx, mcards[0].cy, fade(st, ORIENT + 0.3, 0.3), 0.85);
          E.pop(mcards[1].g, mcards[1].cx, mcards[1].cy, fade(st, POSE, 0.3), 0.85);
          E.pop(mcards[0].pill.g, RX + 54, CY[0] + 85, fade(st, EP5, 0.3), 0.6);
          E.pop(mcards[1].pill.g, RX + 54, CY[1] + 85, fade(st, EP7, 0.3), 0.6);
          if (conns) { E.drawOn(conns[3], fade(st, ORIENT + 0.2, 0.4)); E.drawOn(conns[4], fade(st, P_AT + 0.35, 0.4)); }

          /* ④ 17.76「关节」右肩旋转小弧；18.0 起胳膊匀速往外抬，18.98「手」切向速度箭头跟着手走；
             20.25「中间」手 → 右栏连线；21.06「一个矩阵」雅可比卡；22.14「雅可比」J 鼓一下；23.0「第十三集」 */
          var sh = at('left_shoulder_roll_link'), hd = at('left_rubber_hand');
          jointDot.setAttribute('cx', sh[0].toFixed(1)); jointDot.setAttribute('cy', sh[1].toFixed(1));
          E.op(jointDot, fade(st, JOINT, 0.25));
          arc.set(sh[0], sh[1], 30, 110, 10, fade(st, JOINT, 0.5));
          var rx = hd[0] - sh[0], ry = hd[1] - sh[1], rl = Math.hypot(rx, ry) || 1;
          var tx = ry / rl, ty = -rx / rl;                        // 垂直于「肩 → 手」、朝上（外张角变大时手往上走）
          if (ty > 0) { tx = -tx; ty = -ty; }
          var ph = fade(st, HANDV, 0.35) * (st < LIFT[1] ? 1 : 1 - fade(st, LIFT[1], 0.3));
          handV.set(hd[0], hd[1], hd[0] + 46 * tx * ph, hd[1] + 46 * ty * ph);
          E.op(handV.g, ph > 0 ? 1 : 0);
          var pj = fade(st, MID, 0.45), jq = [RX - 4, CY[2] + CH / 2];
          jconn.setAttribute('d', pj > 0 ? 'M ' + hd[0].toFixed(1) + ' ' + (hd[1] + 8).toFixed(1) + ' L ' + (hd[0] + (jq[0] - hd[0]) * pj).toFixed(1) + ' ' + (hd[1] + 8 + (jq[1] - hd[1] - 8) * pj).toFixed(1) : '');
          E.pop(mcards[2].g, mcards[2].cx, mcards[2].cy, fade(st, JCARD, 0.3), 0.85);
          E.scaleAt(jg, RX + 214, CY[2] + 30, 1 + 0.25 * bump(st, JNAME, 0.5));
          E.pop(mcards[2].pill.g, RX + 59, CY[2] + 85, fade(st, EP13, 0.3), 0.6);
        }
      };
    }
  };
})();
