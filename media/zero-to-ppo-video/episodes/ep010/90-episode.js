// 组装 010：五幕按 EP010.scenes[i] 的下标排好（缺的幕先放占位，某一幕 build 出错时画出报错，不拖累别的幕），
// 再交给 stage.html 的 window.EpisodeVideo。旁白与 [from, to] 对时在 ../ep010.py。
(function () {
  var E = window.EP010, K = E.K, N = 5;
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
    ep: '010',                      // 用于文件名和封面顶栏正中的集数（「No.010」）；正片画面里不出现集数
    badge: '第二季',
    robots: ['ur5', 'create2'],     // robot3d.js 预先加载的 URDF（robots.json；先跑 python3 fetch_robots.py）
    title: '矩阵指数',
    name: 'exp：螺旋运动 → 位姿',
    pitch: '绕哪根轴、转了多少，一步算出停在哪、朝哪',
    meta: '<b>机器人学骨架：刚体变换</b><br>Robotics Notebooks · 懂运动旋量 · 不用代码',
    introAt: { title: 0.1, name: 0.4, hero: 0 },
    // 划重点 / 片尾里的加粗词：矩阵指数 亮黄；螺旋轴 warn；朝向 / 角度 violet；位置 accent；螺距 good；不能 / 不是 bad
    // 划重点 / 片尾里的加粗词（配色照「画面配色」）：矩阵指数、位姿亮黄；螺旋轴琥珀；朝向、角度蓝（转动）；位置紫（同 007 的 p）；螺距绿；不能 / 不是橙
    boldColors: { '矩阵指数': 'var(--hi)', 'exp': 'var(--hi)', '螺旋轴': 'var(--demo-warn)', '朝向': 'var(--demo-accent)', '角度': 'var(--demo-accent)',
      '位置': 'var(--violet)', '螺距': 'var(--demo-good)', '位姿': 'var(--hi)', '矩阵对数': 'var(--hi)', '不能': '#ff7b3a', '不是': '#ff7b3a' },
    introDur: 26.93,
    hero: function (K) { return E.hero ? E.hero(K) : placeholder('主图', '主图还没写').build(); },
    scenes: scenes,
    // 片尾总结：四张卡片，结论一行（at 是弹出的分镜时间，和 ../ep010.py 的 recap 句对齐）
    recapDur: 36.0,
    recap: {
      act: '本集总结',
      title: '螺旋轴 × 角度 → 位姿',
      items: [
        { c: '#ff7b3a', n: '① 不能直接乘角度', s: '前进的方向一直在变：走的是圆弧，不是直线', at: 1.87 },
        { c: 'var(--hi)', n: '② 无穷多小步：矩阵指数', s: '一步一步乘起来，名字来自自然常数 e', at: 8.62 },
        { c: 'var(--demo-accent)', n: '③ 结果是位姿 T', s: '朝向：绕轴转过这个角度\n位置：轴上的点不动 + 角度 × 螺距', at: 15.43 },
        { c: 'var(--demo-good)', n: '④ 守规矩、能直接相乘', s: '关节的运动都用它来写（第 11 集：指数积公式）', at: 23.5 }
      ],
      foot: { s: '**螺旋轴** × **角度**，交给**矩阵指数**，一步得到**位姿**', at: 29.69 }
    },
    outroDur: 16.58,
    // 片尾不出题：下一集预告、完整内容的网址卡片和衔接语
    outro: {
      next: { ep: '011', q: '正运动学：6 个关节各转一个角度，\n手最后停在什么位置？', at: 0 },
      link: { s: '<b>完整内容</b><br>imchong.github.io/Robotics_Notebooks', at: 7.73 },
      refs: {
        at: 10.87,
        items: [
          '**《Modern Robotics》** Lynch & Park，2017：式 (3.51) 罗德里格斯公式，3.2.3.3 节矩阵对数',
          '**《Modern Robotics》** 3.3.3 节命题 3.25（螺旋运动的矩阵指数），4.1 节指数积公式',
          '**Solà 等**（arXiv:1812.01537）：式 (25)(26)，两个朝向之间先求对数、再取一半',
          '**Modern Robotics 官方代码**（NxRLab）：MatrixExp3、MatrixLog3、MatrixExp6'
        ]
      },
      end: { s: '零基础篇 · 一集只讲一个问题　下集见', at: 14.71 }
    }
  };
})();
