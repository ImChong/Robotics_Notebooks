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
    ep: '002',                      // 用于文件名和封面顶栏正中的集数（「No.002」）；正片画面里不出现集数
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
    // 片尾总结：四个难点各一张卡片，结论一行（at 是弹出的分镜时间，和 ../ep002.py 的 recap 句对齐）
    recapDur: 23.4,
    recap: {
      act: '本集总结',
      title: '四个难点，人形全占',
      items: [
        { c: 'var(--demo-accent)', n: '① 关节多', s: '二十多个关节，一起派活还要配合', at: 3.6 },
        { c: 'var(--violet)', n: '② 浮动基', s: '骨盆没钉住，只能靠脚踩地借力', at: 7.6 },
        { c: 'var(--demo-good)', n: '③ 接触切换', s: '脚一抬一落，能借的力就变了', at: 11.6 },
        { c: 'var(--demo-warn)', n: '④ 强不稳定', s: '重心高、脚下小，控制一停就倒', at: 15.6 }
      ],
      foot: { s: '学会人形，换机械臂、四足、轮式底盘，几乎都是**降难度**', at: 19.2 }
    },
    outroDur: 13.5,
    // 片尾不出题：下一集预告、完整内容的网址卡片和衔接语
    outro: {
      next: { ep: '003', q: '线性代数在机器人里到底怎么用？\n为什么说位置是向量、变换是矩阵？', at: 0 },
      link: { s: '<b>完整内容</b><br>imchong.github.io/Robotics_Notebooks', at: 9.2 },
      end: { s: '零基础篇 · 一集只讲一个问题　下集见', at: 12.4 }
    }
  };
})();
