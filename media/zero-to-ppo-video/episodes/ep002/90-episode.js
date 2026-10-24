// 组装 002：五幕按 EP002.scenes[i] 的下标排好（缺的幕先放占位，某一幕 build 出错时画出报错，不拖累别的幕），
// 再交给 stage.html 的 window.EpisodeVideo。旁白与 [from, to] 对时在 ../ep002.py。
(function () {
  var E = window.EP002, K = E.K, N = 5;
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
    ep: '002',                      // 只用于文件名；画面和封面上都不出现集数
    badge: '第一季',
    robots: ['g1', 'ur5', 'go2', 'jackal'],   // robot3d.js 预先加载的 URDF（robots.json；先跑 python3 fetch_robots.py）
    title: '人形难点',
    name: '多关节、浮动基、换脚、易倒',
    pitch: '把它控好，换到机械臂、四足几乎都更简单',
    meta: '<b>主路线入门：为什么以人形为主载体</b><br>Robotics Notebooks · 不用数学 · 不用代码',
    introAt: { title: 0.1, name: 0.4, hero: 0 },
    // 划重点 / 片尾里的加粗词：四个难点各用本色（关节多 accent、浮动基 violet、接触切换 good、强不稳定 warn）
    boldColors: { '关节多': 'var(--demo-accent)', '浮动基': 'var(--violet)', '接触切换': 'var(--demo-good)', '强不稳定': 'var(--demo-warn)',
      '降难度': 'var(--demo-good)' },
    introDur: 24.9,
    hero: function (K) { return E.hero ? E.hero(K) : placeholder('主图', '主图还没写').build(); },
    scenes: scenes,
    outroDur: 11.1,
    // 片尾不出题：只有下一集预告和衔接语
    outro: {
      next: { ep: '003', q: '线性代数在机器人里到底怎么用？\n为什么说位置是向量、变换是矩阵？', at: 0 },
      end: { s: '零基础篇 · 一集只讲一个问题　下集见', at: 9.2 }
    }
  };
})();
