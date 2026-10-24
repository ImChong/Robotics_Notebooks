// 组装 001：八幕按 EP001.scenes[i] 的下标排好（缺的幕先放占位，某一幕 build 出错时画出报错，不拖累别的幕），
// 再交给 stage.html 的 window.EpisodeVideo。旁白与 [from, to] 对时在 ../ep001.py。
(function () {
  var E = window.EP001, K = E.K, N = 8;
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
    ep: '001',                      // 只用于文件名；画面和封面上都不出现集数
    badge: '第一季',
    robots: ['g1', 'ur5', 'create2', 'prius'],   // robot3d.js 预先加载的 URDF（robots.json；先跑 python3 fetch_robots.py）
    title: '四步循环',
    name: '感知 → 规划 → 控制 → 执行',
    pitch: '从扫地机到人形机器人，拆开全是这一套',
    meta: '<b>主路线入门：机器人技术栈全景</b><br>Robotics Notebooks · 不用数学 · 不用代码',
    introAt: { title: 0.1, name: 0.4, hero: 0 },
    // 划重点 / 片尾里的加粗词：四步各用本色（感知 good、规划 violet、控制 accent、执行 warn）
    boldColors: { '感知': 'var(--demo-good)', '规划': 'var(--violet)', '控制': 'var(--demo-accent)', '执行': 'var(--demo-warn)',
      '反馈': 'var(--text)' },
    introDur: 11.9,
    hero: function (K) { return E.hero ? E.hero(K) : placeholder('主图', '主图还没写').build(); },
    scenes: scenes,
    outroDur: 9.8,
    // 片尾不出题：只有下一集预告和衔接语
    outro: {
      next: { ep: '002', q: '所有机器人都跑这四步，\n为什么偏偏从人形、双足学起？', at: 0 },
      end: { s: '零基础篇 · 一集只讲一个问题　下集见', at: 8.7 }
    }
  };
})();
