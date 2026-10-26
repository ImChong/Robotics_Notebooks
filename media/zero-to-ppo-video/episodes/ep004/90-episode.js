// 组装 004：五幕按 EP004.scenes[i] 的下标排好（缺的幕先放占位，某一幕 build 出错时画出报错，不拖累别的幕），
// 再交给 stage.html 的 window.EpisodeVideo。旁白与 [from, to] 对时在 ../ep004.py。
(function () {
  var E = window.EP004, K = E.K, N = 5;
  function placeholder(title, msg) {
    return {
      title: title, dur: 4, cues: [{ at: 0, s: msg }],
      build: function () {
        var s = K.sceneSvg(title);
        E.txt(s, 400, 210, msg, 22, E.BAD);
        return { el: s, draw: function () {} };
      }
    };
  }
  var scenes = [];
  for (var i = 0; i < N; i++) {
    var sc = E.scenes[i];
    if (!sc) { scenes.push(placeholder('第 ' + (i + 1) + ' 幕（待实现）', '这一幕还没写')); continue; }
    scenes.push({
      title: sc.title, dur: sc.dur, cues: sc.cues,
      build: (function (sc) {
        return function (K) {
          try { return sc.build(K); } catch (e) {
            console.error('scene build failed: ' + sc.title + ': ' + e.message);
            return placeholder(sc.title, '出错：' + e.message).build();
          }
        };
      })(sc)
    });
  }

  window.EpisodeVideo = {
    ep: '004',                      // 用于文件名和封面顶栏正中的集数（「No.004」）；正片画面里不出现集数
    badge: '第一季',
    robots: ['ur5'],                // robot3d.js 预先加载的 URDF（robots.json；先跑 python3 fetch_robots.py）
    title: '矩阵代码',
    name: '用 NumPy 跑通矩阵乘法',
    pitch: '纸上算的变换，交给电脑一行算完',
    meta: '<b>数学打底：编程基础</b><br>Robotics Notebooks · 懂向量与矩阵 · 要写代码',
    introAt: { title: 0.1, name: 0.4, hero: 0 },
    // 划重点 / 片尾里的加粗词：向量 accent、矩阵 violet（同 003），顺序 / 先做 warn，列表的错用法 bad
    boldColors: { '向量': 'var(--demo-accent)', '矩阵': 'var(--violet)', '横排': 'var(--violet)', '列表': 'var(--demo-bad)', '接起来': 'var(--demo-bad)',
      '对应位置': 'var(--demo-bad)', '先做': 'var(--demo-warn)', '顺序': 'var(--demo-warn)', '向量、矩阵都是数组': 'var(--demo-accent)' },
    introDur: 24.0,
    hero: function (K) { return E.hero ? E.hero(K) : placeholder('主图', '主图还没写').build(); },
    scenes: scenes,
    // 片尾总结：四张卡片，结论一行（at 是弹出的分镜时间，和 ../ep004.py 的 recap 句对齐）
    recapDur: 27.93,
    recap: {
      act: '本集总结',
      title: '向量和矩阵怎么写成代码',
      items: [
        { c: 'var(--demo-bad)', n: '① 列表不是向量', s: '`[2, 1] + [1, 2]` 是接起来\n算向量要用 `np.array`', at: 1.87 },
        { c: 'var(--violet)', n: '② 矩阵按横排写', s: '外层方括号里，一个横排一个列表', at: 7.03 },
        { c: 'var(--demo-accent)', n: '③ 乘法用 @', s: '`R @ p`、`S @ R`：矩阵乘向量、乘矩阵\n`*` 只是对应位置相乘', at: 12.31 },
        { c: 'var(--demo-warn)', n: '④ 顺序', s: '`S @ R @ p`：离向量最近的先做，不能换', at: 18.55 }
      ],
      foot: { s: '**向量、矩阵都是数组**，相乘用 `@`', at: 22.65 }
    },
    outroDur: 11.39,
    // 片尾不出题：下一集预告、完整内容的网址卡片和衔接语
    outro: {
      next: { ep: '005', q: '两个朝向取个中间值，\n为什么不能直接平均？', at: 0 },
      link: { s: '<b>完整内容</b><br>imchong.github.io/Robotics_Notebooks', at: 6.38 },
      end: { s: '零基础篇 · 一集只讲一个问题　下集见', at: 9.52 }
    }
  };
})();
