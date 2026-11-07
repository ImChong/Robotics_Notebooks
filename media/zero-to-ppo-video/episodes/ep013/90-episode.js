// 组装 013：六幕按 EP013.scenes[i] 的下标排好（缺的幕先放占位，某一幕 build 出错时画出报错，不拖累别的幕），
// 再交给 stage.html 的 window.EpisodeVideo。旁白与 [from, to] 对时在 ../ep013.py。
(function () {
  var E = window.EP013, K = E.K, N = 6;
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
    ep: '013',                      // 用于文件名和封面顶栏正中的集数（「No.013」）；正片画面里不出现集数
    badge: '第二季',
    robots: ['ur5'],                // robot3d.js 预先加载的 URDF（robots.json；先跑 python3 fetch_robots.py）
    title: '雅可比矩阵',
    name: '关节速度 → 手的速度',   // 发布清单原写「末端速度」：「末端」前几集没讲过，同 011、012 封面换成「手」
    pitch: '关节转多快，手走多快，乘一下就知道',
    meta: '<b>机器人学骨架：雅可比与速度运动学</b><br>Robotics Notebooks · 懂正运动学 · 要写代码',
    introAt: { title: 0.1, name: 0.4, hero: 0 },
    // 划重点 / 片尾里的加粗词（配色照「画面配色」和 00-common.js）：雅可比亮黄；肩那一列亮黄、肘那一列青；手的速度紫；转速蓝；
    // 螺旋轴琥珀；走不了、奇异位形橙
    boldColors: { '雅可比': 'var(--hi)', '雅可比矩阵': 'var(--hi)', '手的速度': 'var(--violet)', '转速': 'var(--demo-accent)', '按列拿份': 'var(--demo-good)',
      '螺旋轴': 'var(--demo-warn)', '运动旋量': 'var(--demo-warn)', '奇异位形': '#ff7b3a', '走不了': '#ff7b3a', '正向运动学': 'var(--demo-accent)' },
    introDur: 23.96,
    hero: function (K) { return E.hero ? E.hero(K) : placeholder('主图', '主图还没写').build(); },
    scenes: scenes,
    // 片尾总结：四张卡片，结论一行（at 是弹出的分镜时间，和 ../ep013.py 的 recap 句对齐）
    recapDur: 39.65,
    recap: {
      act: '本集总结',
      title: '关节速度 → 手的速度：乘雅可比',
      items: [
        { c: 'var(--hi)', n: '① 第几列 = 只转第几个关节', s: '每秒转 1 弧度时手的速度（从关节指向手，转 90°）', at: 2.77 },
        { c: 'var(--violet)', n: '② 一起转：各自那份加起来', s: '手的速度 = 雅可比 × 转速（第 3 集的按列拿份）', at: 11.95 },
        { c: '#ff7b3a', n: '③ 姿势一变，雅可比就变', s: '胳膊伸直：2 列朝同一个方向，有的方向走不了', at: 19.91 },
        { c: 'var(--demo-warn)', n: '④ 写成运动旋量', s: '每一列 = 关节轴此刻的螺旋轴', at: 27.77 }
      ],
      foot: { s: '关节转多快，乘上**雅可比**，就知道手走多快', at: 33.83 }
    },
    outroDur: 19.15,
    // 片尾不出题：下一集预告、完整内容的网址卡片和衔接语
    outro: {
      next: { ep: '014', q: '两种雅可比：站在底座上看、站在手上看，\n写出来为什么不一样？怎么互相换？', at: 0 },
      link: { s: '<b>完整内容</b><br>imchong.github.io/Robotics_Notebooks', at: 9.4 },
      refs: {
        at: 12.84,
        items: [
          '**《Modern Robotics》** Lynch & Park：第 5 章开头',
          '　ẋ = J(θ) θ̇；两连杆式 (5.1)(5.2)、图 5.1（第 i 列 = 只转第 i 个关节）',
          '　θ₂ = 0° 时两列共线：奇异（singularity）',
          '**《Modern Robotics》** 5.1、5.1.1 节式 (5.11)：每列是此刻的螺旋轴',
          '**《Modern Robotics》** 3.3.2、4.1.1 节，例 4.5（UR5 的 6 根螺旋轴）',
          '**官方代码** JacobianSpace 核对（含 UR5 的 6 × 6）、FKinSpace 验算',
          '**NumPy 文档** numpy.column_stack、@（matmul）、ndarray.round',
          '**UR5 URDF**（ur_description）：六个转动关节',
          '**Python 3.13、NumPy 2.5.3、modern_robotics 1.1.1** 实跑'
        ]
      },
      end: { s: '零基础篇 · 一集只讲一个问题　下集见', at: 16.98 }
    }
  };
})();
