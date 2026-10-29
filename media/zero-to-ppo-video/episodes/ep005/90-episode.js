// 组装 005：五幕按 EP005.scenes[i] 的下标排好（缺的幕先放占位，某一幕 build 出错时画出报错，不拖累别的幕），
// 再交给 stage.html 的 window.EpisodeVideo。旁白与 [from, to] 对时在 ../ep005.py。
(function () {
  var E = window.EP005, K = E.K, N = 5;
  function placeholder(title, msg) {
    return {
      title: title, dur: 4, cues: [{ at: 0, s: msg }],
      build: function () {
        var s = K.sceneSvg(title);
        E.txt(s, 400, 210, msg, 22, E.ERR);
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
    ep: '005',                      // 用于文件名和封面顶栏正中的集数（「No.005」）；正片画面里不出现集数
    badge: '第一季',
    robots: ['g1'],                 // robot3d.js 预先加载的 URDF（robots.json；先跑 python3 fetch_robots.py）
    title: '旋转矩阵',
    name: 'SO(3)：正交、行列式为 1',
    pitch: '两个朝向取个中间值，为什么不能直接平均',
    meta: '<b>数学打底：旋转与 SO(3)</b><br>Robotics Notebooks · 用到矩阵乘法 · 不用代码',
    introAt: { title: 0.1, name: 0.4, hero: 0 },
    // 划重点 / 片尾里的加粗词：矩阵 violet（同 003）；两条规矩 good；不合格用橙（红色留给 x 轴）
    boldColors: { '旋转矩阵': 'var(--violet)', '正交': 'var(--demo-good)', '行列式 = 1': 'var(--demo-good)', '行列式等于 1': 'var(--demo-good)',
      '不合格': E.ERR, '跳出': E.ERR, '不是旋转': E.ERR, '平均角度': 'var(--demo-good)', '特殊正交群': 'var(--violet)' },
    introDur: 25.51,
    hero: function (K) { return E.hero ? E.hero(K) : placeholder('主图', '主图还没写').build(); },
    scenes: scenes,
    // 片尾总结：四张卡片，结论一行（at 是弹出的分镜时间，和 ../ep005.py 的 recap 句对齐）
    recapDur: 32.3,
    recap: {
      act: '本集总结',
      title: '旋转矩阵为什么不能直接平均',
      items: [
        { c: 'var(--violet)', n: '① 旋转矩阵', s: '每一列 = 一根小箭头转到的位置\n只改朝向，不改长短', at: 1.87 },
        { c: 'var(--demo-good)', n: '② 规矩一：正交', s: '每一列长度为 1，两两垂直', at: 9.36 },
        { c: 'var(--demo-accent)', n: '③ 规矩二：行列式 = 1', s: '面积不变，不照镜子（镜子是 −1）', at: 14.98 },
        { c: E.ERR, n: '④ 不能直接平均', s: '对应位置平均会跳出旋转矩阵\n中间的朝向：平均转过的角度', at: 19.92 }
      ],
      foot: { s: '**旋转矩阵**：正交、行列式 = 1，不能直接相加平均', at: 25.99 }
    },
    outroDur: 12.67,
    // 片尾不出题：下一集预告、完整内容的网址卡片和衔接语
    outro: {
      next: { ep: '006', q: '欧拉角、四元数、旋转矩阵：\n三种写法各有什么坑，该怎么选？', at: 0 },
      link: { s: '<b>完整内容</b><br>imchong.github.io/Robotics_Notebooks', at: 7.66 },
      end: { s: '零基础篇 · 一集只讲一个问题　下集见', at: 10.8 }
    }
  };
})();
