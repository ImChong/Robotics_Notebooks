// 组装 011：七幕按 EP011.scenes[i] 的下标排好（缺的幕先放占位，某一幕 build 出错时画出报错，不拖累别的幕），
// 再交给 stage.html 的 window.EpisodeVideo。旁白与 [from, to] 对时在 ../ep011.py。
(function () {
  var E = window.EP011, K = E.K, N = 7;
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
    ep: '011',                      // 用于文件名和封面顶栏正中的集数（「No.011」）；正片画面里不出现集数
    badge: '第二季',
    robots: ['ur5'],                // robot3d.js 预先加载的 URDF（robots.json；先跑 python3 fetch_robots.py）
    title: '正运动学',
    name: 'PoE：关节角 → 末端位姿',
    pitch: '每个关节一个指数，连乘起来就是末端位姿',
    meta: '<b>机器人学骨架：正逆运动学</b><br>Robotics Notebooks · 懂矩阵指数 · 要写代码',
    introAt: { title: 0.1, name: 0.4, hero: 0 },
    // 划重点 / 片尾里的加粗词（配色照「画面配色」）：指数积公式、矩阵指数、位姿亮黄；螺旋轴琥珀；朝向、角度蓝（转动）；位置紫（同 007 的 p）；零位绿；不能 / 不是 / 扯断橙
    boldColors: { '指数积公式': 'var(--hi)', '矩阵指数': 'var(--hi)', '位姿': 'var(--hi)', '正向运动学': 'var(--hi)', '螺旋轴': 'var(--demo-warn)',
      '朝向': 'var(--demo-accent)', '角度': 'var(--demo-accent)', '位置': 'var(--violet)', '零位': 'var(--demo-good)',
      '不能': '#ff7b3a', '不是': '#ff7b3a', '扯断': '#ff7b3a' },
    introDur: 30.58,
    hero: function (K) { return E.hero ? E.hero(K) : placeholder('主图', '主图还没写').build(); },
    scenes: scenes,
    // 片尾总结：四张卡片，结论一行（at 是弹出的分镜时间，和 ../ep011.py 的 recap 句对齐）
    recapDur: 39.38,
    recap: {
      act: '本集总结',
      title: '零位 + 螺旋轴 → 手的位姿',
      items: [
        { c: 'var(--hi)', n: '① 正向运动学', s: '从各个关节的角度，得到手的位姿', at: 2.77 },
        { c: 'var(--demo-warn)', n: '② 指数积公式 T = exp([S₁]θ₁) ⋯ exp([Sₙ]θₙ) M', s: '每个关节一个矩阵指数，按顺序相乘，再乘 M', at: 8.18 },
        { c: 'var(--demo-good)', n: '③ 从右往左算', s: '最外面的关节先转，螺旋轴都按零位写', at: 16.14 },
        { c: 'var(--demo-accent)', n: '④ 比 D-H 参数', s: '不用每节摆坐标系，和雅可比一套；代价：数多', at: 22.58 }
      ],
      foot: { s: '**零位** + **螺旋轴**：各算一个**矩阵指数**，按顺序乘起来 = 手的**位姿**', at: 31.62 }
    },
    outroDur: 19.53,
    // 片尾不出题：下一集预告、完整内容的网址卡片、本集核对过的一手资料和衔接语
    outro: {
      next: { ep: '012', q: '逆运动学：手要停在这个位置，\n关节该转多少？', at: 0 },
      link: { s: '<b>完整内容</b><br>imchong.github.io/Robotics_Notebooks', at: 9.78 },
      refs: {
        at: 13.22,
        items: [
          '**《Modern Robotics》** 第 4 章开头、式 (4.1)–(4.4)、图 4.1',
          '**《Modern Robotics》** 式 (4.6)–(4.14)、4.1.1 节、例 4.2',
          '**《Modern Robotics》** 4.1.3 节：从最外面的关节往里变换',
          '**《Modern Robotics》** 例 4.5、图 4.6、图 4.7（UR5）',
          '**《Modern Robotics》** 附录 C、C.6 节（D-H 参数）、4.3、4.5 节',
          '**《Modern Robotics》** 3.2.1、3.3.1 节（列 = 坐标轴、原点）',
          '**Modern Robotics 官方代码**：FKinSpace、MatrixExp6、VecTose3',
          '**NumPy 2.5.3**、**modern_robotics 1.1.1**：输出都实跑过',
          '**UR5 URDF**（example-robot-data）：关节轴同例 4.5'
        ]
      },
      end: { s: '零基础篇 · 一集只讲一个问题　下集见', at: 17.36 }
    }
  };
})();
