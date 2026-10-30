// 007 第 3 幕：齐次变换——把 R 和 p 拼成一个矩阵，点后面补一个 1。整幕不用地图，上面一排是 T 乘 (2, 1, 1)，下面一排按列展开。
// 句 1 顶上写 Rv + p：「先乘」Rv 下面写「① 乘」，「再加」+ p 下面写「② 加」，「两步」两个小标签鼓一下，「一个矩阵」「一次乘完」右边写「一个矩阵，一次乘完？」；
// 句 2 顶上那行收起；「位置」「第三列」旋转矩阵 R（两列红、绿，自带括号）和位置 p（紫色一列，自带括号）并排出现，p 离得远一点，
//    「拼在」「右边」p 挪过来贴在 R 右边，两对小括号换成一对大括号，「补一排」最下面一排的位置框亮黄，「零、零、一」三个数依次弹出（灰）；
// 句 3「袜子的两个数」右边出「×」和列向量 (2, 1)，「补上一个一」最下面弹出一个亮黄的 1（框亮黄），「二、一、一」三个数依次鼓一下；
// 句 4「按列」下面一排：「第一列拿两份」T 第一列和向量的 2 框红，出 2 (0, 1, 0)；「第二列拿一份」第二列和 1 框绿，出 + 1 (−1, 0, 0)；
//    「第三列拿一份」第三列和补上的 1 框亮黄，出 + 1 (3, 1, 1)；「加起来」出 =；「二、三、一」结果三个数依次写出，上面一排也写出 = (2, 3, 1)；
// 句 5「前两个数」结果的前两个数框亮黄，「答案」右边写「袜子在地图上 (2, 3)」并打勾；「补上的那个一」向量里的 1 亮一下，「第三列拿一份」第三列和 1 (3, 1, 1) 框紫，
//    「位置」「加了进来」(3, 1, 1) 里的 3、1 鼓一下；
// 句 6「最下面那一排」T 的最下面一排框灰，「零、零、一」三个数依次鼓一下，「最后一个数还是一」结果最后的 1 框亮黄，右边写「最后还是 1：还是一个点」，
//    「接着再乘」右边再写「还能接着乘下一个矩阵」；
// 句 7「拼起来的矩阵」大括号亮一下，「齐次变换矩阵」T 下面写名字，下面一排换成「Rv + p ⟹ T (v, 1)」，「先乘再加」「一次矩阵乘法」两边依次亮起。
// 时刻取自配音的逐词时间（分镜时间 = 念到这个词的时刻），见 ../ep007.py。
(function () {
  var E = window.EP007;
  var TX = 220, TY = 150, SZ = 34;                           // 上面一排：T 的中心、字号
  var BY = 336, BS = 28;                                     // 下面一排：中线、字号
  var T = {
    mul: 0.36, add: 1.1, two: 2.0, one: [3.86, 5.34], top1Out: 7.4,
    pos: 8.03, col3: 8.75, join: [9.56, 10.98], row: [11.96, 12.45], zeros: [13.01, 13.54, 14.04],
    vec: 15.74, one1: 17.29, nums: [18.37, 18.83, 19.34],
    byCol: 20.85, k1: 21.92, k2: 23.03, k3: 24.23, plus: 25.02, res: [25.93, 26.47, 27.0],
    first2: 28.49, answer: 30.27, key1: 33.02, k3b: 33.52, posIn: [35.41, 36.07],
    bottom: 37.58, bz: [38.32, 38.71, 39.17], last1: 40.72, again: 42.02, kOut: 37.0,
    named: 45.54, name: 47.07, sumIn: 48.6, before: 49.04, after: 50.12
  };

  E.scenes[2] = {
    title: '齐次变换：拼成一个矩阵',
    dur: 52.7,
    cues: [
      { at: 0, s: '先乘 $R$、再加 $p$，要分 2 步：能不能一个矩阵一次乘完？' },
      { at: 6.79, s: '把**位置**当成第 3 列拼在 $R$ 右边，下面补一排 0、0、1' },
      { at: 15.09, s: '袜子的 (2, 1) 后面也补一个 1：(2, 1, 1)' },
      { at: 20.42, s: '按列算：第 1 列拿 2 份 + 第 2 列拿 1 份 + 第 3 列拿 1 份 = (2, 3, 1)' },
      { at: 28.05, s: '前 2 个数 (2, 3) 就是答案：补上的 1 让**位置**这一列加进来 1 份' },
      { at: 37.19, s: '最下面一排 0、0、1：结果最后还是 1，还能接着乘下一个矩阵' },
      { at: 45.09, s: '这就是**齐次变换矩阵**：先乘再加，变成一次矩阵乘法' }
    ],
    build: function (K) {
      var s = K.sceneSvg('把位置 p 当成第三列，拼在旋转矩阵 R 的右边，下面补一排 0、0、1，就是齐次变换矩阵 T；袜子的 (2, 1) 后面补一个 1，变成 (2, 1, 1)；按列算：第一列拿 2 份、第二列拿 1 份、第三列拿 1 份，加起来是 (2, 3, 1)，前两个数就是袜子在地图上的位置；补上的 1 让位置加进来一份，最下面一排让结果最后还是 1，还能接着乘；先乘再加变成了一次矩阵乘法');
      function col(v) { return v.map(function (a) { return [a]; }); }
      function vecO(color, size) { return { size: size || BS, color: function (i) { return Array.isArray(color) ? color[i] : color; } }; }

      /* ── 句 1：顶上的 Rv + p ── */
      var top1 = E.el('g', {}, s);
      E.tex(top1, 250, 92, 'R\\,v\\;+\\;\\textcolor{' + E.HEX.violet + '}{p}', 46, E.INK, 'middle', { w: 220, h: 70 });
      var tagMul = E.tag(top1, 160, 132, 92, 36, '① 乘', E.INK2, { size: 20, weight: 900, ink: E.INK });
      var tagAdd = E.tag(top1, 274, 132, 92, 36, '② 加', E.VIOLET, { size: 20, weight: 900, ink: E.INK });
      var ask = E.txt(top1, 470, 102, '一个矩阵，一次乘完？', 28, E.HI, 'start', 900);

      /* ── 上面一排：T、×、向量、=、结果 ── */
      var Tm = E.matN(s, TX, TY, [[0, -1, 3], [1, 0, 1], [0, 0, 1]], { size: SZ, dx: 64, dy: 46,
        color: function (i, j) { return i === 2 ? E.MUT : j === 0 ? E.X : j === 1 ? E.Y : E.VIOLET; } });
      var bR = Tm.brk(0, 0, 1, 1), bP = Tm.brk(0, 2, 1, 2, E.VIOLET);
      var labR = E.txt(s, (Tm.pos(0, 0)[0] + Tm.pos(0, 1)[0]) / 2, 58, '旋转矩阵 R', 20, E.INK2, 'middle', 900);
      var labP = E.txt(s, Tm.pos(0, 2)[0] + 6, 58, '位置 p', 20, E.VIOLET, 'middle', 900);
      var labRow = E.txt(s, Tm.box[0] - 12, Tm.pos(2, 0)[1] + 7, '补一排', 20, E.HI, 'end', 900);
      var name = E.txt(s, TX, Tm.box[3] + 34, '齐次变换矩阵 T', 24, E.HI, 'middle', 900);
      // T 右边：× (2, 1, 1) = (2, 3, 1)，按括号的真实宽度排（E.flow）
      var fT = E.flow(Tm.box[2] + 1.5, TY, 10);
      var times = fT.sym(s, '×', 34);
      var V = fT.mat(s, col([2, 1, 1]), vecO([E.X, E.Y, E.HI], SZ));
      var eqTop = fT.sym(s, '=', 34);
      var RES = fT.mat(s, col([2, 3, 1]), vecO(E.HI, SZ));
      // 右边的说明
      var noteA = E.el('g', {}, s);
      E.txt(noteA, 552, 108, '袜子在地图上 (2, 3)', 22, E.HI, 'start', 900);
      var okA = E.check(noteA, 756, 88, 22, E.GOOD, 4.5);
      var noteB = E.el('g', {}, s);
      E.txt(noteB, 556, 203, '最后还是 1：还是一个点', 21, E.INK, 'start', 900);
      var noteC = E.txt(s, 556, 238, '还能接着乘下一个矩阵', 21, E.INK2, 'start', 700);
      var link = E.arrow(s, [[550, RES.pos(2, 0)[1]], [RES.box[2] + 6, RES.pos(2, 0)[1]]], E.INK2, 2.5, { head: 10 });

      /* ── 下面一排：按列展开 ── */
      var band = E.el('g', {}, s), fBand = E.flow(196, BY, 10);
      var terms = [
        { k: '2', v: [0, 1, 0], c: E.X },
        { k: '1', v: [-1, 0, 0], c: E.Y },
        { k: '1', v: [3, 1, 1], c: E.VIOLET }
      ].map(function (d, i) {
        var g = E.el('g', {}, band);
        if (i) fBand.sym(g, '+', 28);
        fBand.sym(g, d.k, 28, i === 0 ? E.X : i === 1 ? E.Y : E.HI, i ? 8 : 0);
        var m = fBand.mat(g, col(d.v), vecO(d.c), 5);
        if (i === 2) m.cells[2][0].style.fill = E.MUT;
        return { g: g, m: m };
      });
      var bEq = E.el('g', {}, band);
      fBand.sym(bEq, '=', 28);
      var bRes = fBand.mat(bEq, col([2, 3, 1]), vecO(E.HI));
      // 句 7：先乘再加 ⟹ 一次乘法
      var sumG = E.el('g', {}, s);
      var sL = E.tex(sumG, 250, BY + 12, 'R\\,v+\\textcolor{' + E.HEX.violet + '}{p}', 36, E.INK, 'middle', { w: 180, h: 60 });
      var sArrow = E.tex(sumG, 390, BY + 12, '\\Longrightarrow', 36, E.INK2, 'middle', { w: 80, h: 60 });
      var sR = E.tex(sumG, 530, BY + 12, 'T\\begin{bmatrix}v\\\\\\textcolor{' + E.HEX.hi + '}{1}\\end{bmatrix}', 36, E.INK, 'middle', { w: 160, h: 110 });
      var sTagL = E.txt(sumG, 250, BY + 70, '先乘再加', 22, E.INK2, 'middle', 900);
      var sTagR = E.txt(sumG, 530, BY + 76, '一次矩阵乘法', 22, E.HI, 'middle', 900);

      return {
        el: s,
        draw: function (st, t) {
          var f = E.fade, bump = E.bump;

          /* 句 1 */
          E.op(top1, f(st, 0, 0.4) * (1 - f(st, T.top1Out, 0.4)));
          var bm = Math.max(bump(st, T.mul, 0.7), bump(st, T.two, 0.7)), ba = Math.max(bump(st, T.add, 0.7), bump(st, T.two + 0.15, 0.7));
          var pm = f(st, T.mul, 0.3), pa = f(st, T.add, 0.3);
          E.scaleAt(tagMul.g, 206, 150, (0.8 + 0.2 * pm) * (1 + 0.12 * bm)); E.op(tagMul.g, pm);
          E.scaleAt(tagAdd.g, 320, 150, (0.8 + 0.2 * pa) * (1 + 0.12 * ba)); E.op(tagAdd.g, pa);
          E.op(ask, f(st, T.one[0], 0.4));

          /* 句 2：R、p 并排 → p 挪过来贴上 → 一对大括号 → 补一排 */
          var on2 = f(st, T.pos, 0.4), jn = f(st, T.join[0], 0.6);   // 贴过来只用 0.6 s：两对小括号收起到大括号出来之间不留长空档
          E.op(Tm.g, on2);
          var shift = 46 * (1 - jn);
          [0, 1, 2].forEach(function (i) {
            var c = Tm.cells[i][2], q = Tm.pos(i, 2);
            c.setAttribute('x', (q[0] + shift).toFixed(1));
          });
          // 两对小括号先收起，位置那一列再贴过来，最后出一对大括号
          bP.setAttribute('transform', 'translate(' + shift.toFixed(1) + ' 0)');
          E.op(bP, f(st, T.col3, 0.3) * (1 - f(st, T.join[0] - 0.25, 0.25)));
          E.op(bR, 1 - f(st, T.join[0] - 0.25, 0.25));
          E.op(Tm.frame, f(st, T.join[0] + 0.5, 0.3));
          [0, 1].forEach(function (i) { Tm.cells[i][2].style.opacity = f(st, T.col3, 0.3); });
          Tm.cells[0][0].style.opacity = Tm.cells[0][1].style.opacity = Tm.cells[1][0].style.opacity = Tm.cells[1][1].style.opacity = 1;
          E.op(labR, on2); E.op(labP, f(st, T.col3, 0.3));
          labP.setAttribute('x', (Tm.pos(0, 2)[0] + 6 + shift).toFixed(1));
          var rowOn = f(st, T.row[0], 0.3);
          [0, 1, 2].forEach(function (j) {
            var p = f(st, T.zeros[j], 0.35), q = Tm.pos(2, j), c = Tm.cells[2][j];
            E.op(c, p); E.scaleAt(c, q[0], q[1], 1 + 0.35 * (1 - p) + 0.12 * bump(st, T.bz[j], 0.6));
          });
          Tm.hl('row', 2, 0, 2, 2, E.HI, rowOn * (1 - f(st, T.zeros[2] + 0.6, 0.4)));
          E.op(labRow, rowOn * (1 - f(st, T.zeros[2] + 0.6, 0.4)));

          /* 句 3：× (2, 1) → 补一个 1 */
          var vOn = f(st, T.vec, 0.4);
          E.op(times, vOn); E.op(V.g, vOn);
          var o1 = f(st, T.one1, 0.35), q1 = V.pos(2, 0);
          E.op(V.cells[2][0], o1); E.scaleAt(V.cells[2][0], q1[0], q1[1], 1 + 0.6 * (1 - o1) + 0.25 * bump(st, T.key1, 0.8));
          V.hl('one', 2, 0, 2, 0, E.HI, o1 * (1 - f(st, T.byCol, 0.3)) + bump(st, T.key1, 1.0));
          [0, 1, 2].forEach(function (i) { var q = V.pos(i, 0); if (i < 2) E.scaleAt(V.cells[i][0], q[0], q[1], 1 + 0.25 * bump(st, T.nums[i], 0.6)); });

          /* 句 4：按列展开；上面一排写出结果 */
          var kOut = 1 - f(st, T.kOut, 0.3);
          Tm.hl('c0', 0, 0, 2, 0, E.X, f(st, T.k1, 0.3) * (1 - f(st, T.k2, 0.3)));
          Tm.hl('c1', 0, 1, 2, 1, E.Y, f(st, T.k2, 0.3) * (1 - f(st, T.k3, 0.3)));
          Tm.hl('c2', 0, 2, 2, 2, E.VIOLET, Math.max(f(st, T.k3, 0.3) * (1 - f(st, T.plus, 0.3)), f(st, T.k3b, 0.3) * kOut));
          V.hl('a', 0, 0, 0, 0, E.X, f(st, T.k1, 0.3) * (1 - f(st, T.k2, 0.3)));
          V.hl('b', 1, 0, 1, 0, E.Y, f(st, T.k2, 0.3) * (1 - f(st, T.k3, 0.3)));
          V.hl('c', 2, 0, 2, 0, E.HI, Math.max(f(st, T.k3, 0.3) * (1 - f(st, T.plus, 0.3)), f(st, T.k3b, 0.3) * kOut));
          E.op(band, 1 - f(st, T.sumIn - 0.4, 0.4));
          [T.k1, T.k2, T.k3].forEach(function (a, i) { E.op(terms[i].g, f(st, a, 0.35)); });
          terms[2].m.hl('p', 0, 0, 1, 0, E.VIOLET, f(st, T.posIn[0], 0.3) * kOut);
          [0, 1].forEach(function (i) { var q = terms[2].m.pos(i, 0); E.scaleAt(terms[2].m.cells[i][0], q[0], q[1], 1 + 0.15 * bump(st, T.posIn[i], 0.7)); });
          E.op(bEq, f(st, T.plus, 0.3));
          [0, 1, 2].forEach(function (i) {
            var p = f(st, T.res[i], 0.3);
            E.op(bRes.cells[i][0], p); E.op(RES.cells[i][0], p);
          });
          E.op(eqTop, f(st, T.res[0], 0.3)); E.op(RES.frame, f(st, T.res[0], 0.3));

          /* 句 5、6：前两个数是答案；最下面一排让最后还是 1 */
          RES.hl('ab', 0, 0, 1, 0, E.HI, f(st, T.first2, 0.3) * (1 - f(st, T.kOut, 0.3)));
          E.op(noteA, f(st, T.answer, 0.4)); E.drawOn(okA, f(st, T.answer + 0.3, 0.4));
          Tm.hl('bot', 2, 0, 2, 2, E.INK2, f(st, T.bottom, 0.3) * (1 - f(st, T.named, 0.3)));
          RES.hl('last', 2, 0, 2, 0, E.HI, f(st, T.last1, 0.3) * (1 - f(st, T.named, 0.3)));
          E.op(noteB, f(st, T.last1, 0.4)); E.op(noteC, f(st, T.again, 0.4));
          E.drawOn(link.line, f(st, T.last1, 0.5)); E.op(link.head, f(st, T.last1 + 0.4, 0.2));

          /* 句 7：名字，下面一排换成「先乘再加 ⟹ 一次乘法」 */
          var nb = bump(st, T.named, 0.9);
          Tm.frame.style.filter = nb > 0.05 ? 'drop-shadow(0 0 ' + (8 * nb).toFixed(1) + 'px var(--hi))' : '';
          E.pop(name, TX, Tm.box[3] + 26, f(st, T.name, 0.35), 0.8);
          E.op(sumG, f(st, T.sumIn, 0.4));
          E.op(sTagL, f(st, T.before, 0.3)); E.op(sTagR, f(st, T.after, 0.3));
          E.op(sL, 0.55 + 0.45 * f(st, T.before, 0.3)); E.op(sArrow, f(st, T.after - 0.3, 0.3)); E.op(sR, f(st, T.after, 0.3));
        }
      };
    }
  };
})();
