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
    ep: '001',
    season: 1,
    badge: 'L−1 · 第 1 季',
    question: '机器人到底在干嘛？',
    pitch: '拆开任何机器人，都在跑同一个四步圈',
    footer: '从零基础讲起，一路接上《从头学人形机器人算法》',
    introAt: { ep: 0.1, question: 0.3, pitch: 0.6, hero: 0 },
    // 划重点 / 片尾里的加粗词：四步各用本色（感知 good、规划 violet、控制 accent、执行 warn）
    boldColors: { '感知': 'var(--demo-good)', '规划': 'var(--violet)', '控制': 'var(--demo-accent)', '执行': 'var(--demo-warn)',
      '反馈': 'var(--text)', 'C': 'var(--demo-accent)' },
    introDur: 11.9,
    hero: function (K) { return E.hero ? E.hero(K) : placeholder('主图', '主图还没写').build(); },
    scenes: scenes,
    outroDur: 37.0,
    outro: {
      quiz: {
        q: '客厅那台机器人：哪件事属于**控制**？\nA 认出前面有个纸箱　B 决定从旁边绕过\nC 算出膝盖该出多大力　D 电机带着膝盖转',
        a: '**C**　A 感知 · B 规划 · D 执行\n只有 C 在给每个关节派活',
        thinkFrom: 15.2, revealAt: 20.2
      },
      next: { ep: '002', q: '所有机器人都跑这四步，\n为什么偏偏从人形、双足学起？', at: 27.2 },
      end: { s: '零基础篇 · 一集只讲一个问题　下集见', at: 35.9 }
    }
  };
})();
