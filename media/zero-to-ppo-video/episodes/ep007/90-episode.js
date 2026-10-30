// 组装 007：五幕按 EP007.scenes[i] 的下标排好（缺的幕先放占位，某一幕 build 出错时画出报错，不拖累别的幕），
// 再交给 stage.html 的 window.EpisodeVideo。旁白与 [from, to] 对时在 ../ep007.py。
(function () {
  var E = window.EP007, K = E.K, N = 5;
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
    ep: '007',                      // 用于文件名和封面顶栏正中的集数（「No.007」）；正片画面里不出现集数
    badge: '第一季',
    robots: ['create2', 'g1'],      // robot3d.js 预先加载的 URDF（robots.json；先跑 python3 fetch_robots.py）
    title: '齐次变换',
    name: '朝向 + 位置 → 4×4 矩阵',
    pitch: '点要平移、自由向量不用：推导不出错的前提',
    meta: '<b>数学打底：位姿与 SE(3)</b><br>Robotics Notebooks · 懂旋转矩阵 · 不用代码',
    introAt: { title: 0.1, name: 0.4, hero: 0 },
    // 划重点 / 片尾里的加粗词：位置 violet（同分镜里的 p）；点亮黄（同袜子、杯子）；自由向量白字（同分镜，绿留给「对」和 y 轴）；不平移 good；算错橙（同 005、006）
    boldColors: { '位置': 'var(--violet)', '位置 p': 'var(--violet)', '自由向量': 'var(--text)', '不平移': 'var(--demo-good)', '算错': E.ERR },
    introDur: 34.04,
    hero: function (K) { return E.hero ? E.hero(K) : placeholder('主图', '主图还没写').build(); },
    scenes: scenes,
    // 片尾总结：四张卡片，结论一行（at 是弹出的分镜时间，和 ../ep007.py 的 recap 句对齐）
    recapDur: 35.12,
    recap: {
      act: '本集总结',
      title: '朝向 + 位置，拼成一个矩阵',
      items: [
        { c: 'var(--violet)', n: '① 位姿 = 朝向 + 位置', s: '旋转矩阵 $R$，加上位置 $p$', at: 1.87 },
        { c: 'var(--hi)', n: '② 点：先转、再平移', s: '$v\' = Rv + p$', at: 7.77 },
        { c: 'var(--demo-good)', n: '③ 自由向量：只转、不平移', s: '$v\' = Rv$：同一个速度、力换坐标系也这样', at: 11.75 },
        { c: 'var(--demo-accent)', n: '④ 齐次变换矩阵', s: '点后面补 1，自由向量后面补 0\n这个矩阵：平面上 3×3，三维里 4×4', at: 18.04 }
      ],
      foot: { s: '**朝向 + 位置**拼成齐次变换矩阵：点补 1，自由向量补 0', at: 28.02 }
    },
    outroDur: 17.01,
    // 片尾不出题：下一集预告、完整内容的网址卡片、本集核对过的一手资料（同 006）和衔接语
    outro: {
      next: { ep: '008', q: '优化的直觉：变量、目标、约束——\n在一堆限制里找最好的做法，怎么写下来？', at: 0 },
      link: { s: '<b>完整内容</b><br>imchong.github.io/Robotics_Notebooks', at: 8.16 },
      refs: {
        at: 11.3,
        items: [
          '**《Modern Robotics》**：第 3 章自由向量、换坐标系、SE(3)，式 (4.4)',
          '**CGAL 手册** 线性几何核：向量 = 两点之差，两点相加没有意义',
          '**Strang《线性代数导论》** 第 5 版 1.3 节：矩阵乘向量 = 各列的组合',
          '**Hauser《Robotic Systems》** 第 3 章：$T_d(v) = Rv$，两位置相加无意义',
          '**ROS REP 103**：身上的轴 x 朝前、y 朝左、z 朝上',
          '**ROS REP 105**：base_link 在机器人身上，map 是世界固定坐标系',
          '**ROS geometry_msgs**：Vector3 只转不平移，Pose = 位置 + 朝向',
          '**Eigen** 几何模块：点 t * p，向量 t.linear() * v',
          '**OpenGL 2.1** glLight：w = 0 表示方向、不管位置',
          '**Unitree G1 URDF**：骨盆、腿的连杆、头上的 d435'
        ]
      },
      end: { s: '零基础篇 · 一集只讲一个问题　下集见', at: 15.14 }
    }
  };
})();
