// 组装 008：五幕按 EP008.scenes[i] 的下标排好（缺的幕先放占位，某一幕 build 出错时画出报错，不拖累别的幕），
// 再交给 stage.html 的 window.EpisodeVideo。旁白与 [from, to] 对时在 ../ep008.py。
(function () {
  var E = window.EP008, K = E.K, N = 5;
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
    ep: '008',                      // 用于文件名和封面顶栏正中的集数（「No.008」）；正片画面里不出现集数
    badge: '第一季',
    robots: ['g1'],                 // robot3d.js 预先加载的 URDF（robots.json；先跑 python3 fetch_robots.py）
    title: '优化问题',
    name: '变量、目标、约束',
    pitch: '在限制下找最好的解，机器人里到处都是',
    meta: '<b>数学打底：优化问题的直觉</b><br>Robotics Notebooks · 高中数学 · 不用代码',
    introAt: { title: 0.1, name: 0.4, hero: 0 },
    // 划重点 / 片尾里的加粗词：变量 good、代价 / 目标函数 violet、约束 bad、碗底 / 最好的选择 hi
    boldColors: { '变量': 'var(--demo-good)', '代价': 'var(--violet)', '目标函数': 'var(--violet)', '目标': 'var(--violet)',
      '约束': 'var(--demo-bad)', '碗底': 'var(--hi)', '优化问题': 'var(--hi)', '优化': 'var(--hi)', '出界': 'var(--demo-bad)' },
    introDur: 23.17,
    hero: function (K) { return E.hero ? E.hero(K) : placeholder('主图', '主图还没写').build(); },
    scenes: scenes,
    // 片尾总结：四张卡片，结论一行（at 是弹出的分镜时间，和 ../ep008.py 的 recap 句对齐）
    recapDur: 31.23,
    recap: {
      act: '本集总结',
      title: '优化：在限制下找最好的解',
      items: [
        { c: 'var(--demo-good)', n: '① 变量', s: '由我们来选的数：这一步迈几格 u', at: 1.87 },
        { c: 'var(--violet)', n: '② 目标函数', s: '给每个选择算代价：迈 2 格只扣 8 分', at: 6.65 },
        { c: 'var(--demo-bad)', n: '③ 约束', s: '不能越过的线：最好的解常贴在边上', at: 12.6 },
        { c: 'var(--hi)', n: '④ 到处都是优化', s: '逆运动学、最优控制、全身控制、强化学习', at: 18.55 }
      ],
      foot: { s: '**优化**：在约束里，找让代价最小的变量', at: 25.61 }
    },
    outroDur: 18.43,
    // 片尾不出题：下一集预告、完整内容的网址卡片和衔接语
    outro: {
      next: { ep: '009', q: '第二季开讲：刚体又转又移，\n速度怎么用六个数一起说清楚？', at: 0 },
      link: { s: '<b>完整内容</b><br>imchong.github.io/Robotics_Notebooks', at: 9.58 },
      refs: {
        at: 12.72,
        items: [
          '**《Convex Optimization》** Boyd & Vandenberghe：1.1、4.1.1 节（变量、目标函数即代价函数、约束），9.3 节梯度下降',
          '**《Underactuated Robotics》** Tedrake：第 24 章（优化问题的写法）、第 8 章（LQR 的代价）',
          '**《Modern Robotics》** Lynch & Park：6.2 节（逆运动学用优化）、12.1 节（只能推，不能拉）',
          '**mink 文档**：逆运动学写成二次规划，关节不超限位',
          '**Herzog 等**（arXiv:1410.7284）：全身控制的变量是关节加速度、力矩和接触力',
          '**《Reinforcement Learning》** Sutton & Barto：式 (13.1)，策略参数一小步一小步地改',
          '**Unitree G1 URDF**（unitree_ros）：29 个转动关节'
        ]
      },
      end: { s: '零基础篇 · 一集只讲一个问题　下集见', at: 16.56 }
    }
  };
})();
