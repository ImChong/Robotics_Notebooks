// 组装 003：六幕按 EP003.scenes[i] 的下标排好（缺的幕先放占位，某一幕 build 出错时画出报错，不拖累别的幕），
// 再交给 stage.html 的 window.EpisodeVideo。旁白与 [from, to] 对时在 ../ep003.py。
(function () {
  var E = window.EP003, K = E.K, N = 6;
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
    ep: '003',                      // 用于文件名和封面顶栏正中的集数（「No.003」）；正片画面里不出现集数
    badge: '第一季',
    robots: ['ur5', 'create2', 'g1'],   // robot3d.js 预先加载的 URDF（robots.json；先跑 python3 fetch_robots.py）
    title: '向量与矩阵',
    name: '位置是向量，变换是矩阵',
    pitch: '位置、朝向、速度、力，后面全用它们来写',
    meta: '<b>数学打底：机器人里的线性代数</b><br>Robotics Notebooks · 高中数学 · 不用代码',
    introAt: { title: 0.1, name: 0.4, hero: 0 },
    // 划重点 / 片尾里的加粗词：向量 accent、矩阵 violet（和分镜里的箭头、变形网格同色）
    boldColors: { '向量': 'var(--demo-accent)', '位置是向量': 'var(--demo-accent)', '矩阵': 'var(--violet)', '变换是矩阵': 'var(--violet)',
      '线性变换': 'var(--violet)', '顺序': 'var(--demo-warn)' },
    introDur: 21.34,
    hero: function (K) { return E.hero ? E.hero(K) : placeholder('主图', '主图还没写').build(); },
    scenes: scenes,
    // 片尾总结：四张卡片，结论一行（at 是弹出的分镜时间，和 ../ep003.py 的 recap 句对齐）
    recapDur: 28.2,
    recap: {
      act: '本集总结',
      title: '线性代数在机器人里怎么用',
      items: [
        { c: 'var(--demo-accent)', n: '① 位置是向量', s: '一列数，也是从原点出发的一支箭头', at: 1.87 },
        { c: 'var(--demo-good)', n: '② 向量能相加、能乘数', s: '位移、速度、力，都这么算', at: 7.08 },
        { c: 'var(--violet)', n: '③ 矩阵是变换', s: '每一列 = 一根小箭头最后落在哪', at: 12.5 },
        { c: 'var(--demo-warn)', n: '④ 乘法', s: '矩阵 × 向量 = 新位置\n矩阵 × 矩阵 = 接着做，顺序不能换', at: 17.85 }
      ],
      foot: { s: '**位置是向量**，**变换是矩阵**', at: 24.02 }
    },
    outroDur: 13.22,
    // 片尾不出题：下一集预告、完整内容的网址卡片和衔接语
    outro: {
      next: { ep: '004', q: '今天手算的矩阵乘法，\n怎么用 Python / NumPy 一行算完？', at: 0 },
      link: { s: '<b>完整内容</b><br>imchong.github.io/Robotics_Notebooks', at: 8.21 },
      end: { s: '零基础篇 · 一集只讲一个问题　下集见', at: 11.35 }
    }
  };
})();
