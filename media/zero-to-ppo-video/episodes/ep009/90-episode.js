// 组装 009：五幕按 EP009.scenes[i] 的下标排好（缺的幕先放占位，某一幕 build 出错时画出报错，不拖累别的幕），
// 再交给 stage.html 的 window.EpisodeVideo。旁白与 [from, to] 对时在 ../ep009.py。
(function () {
  var E = window.EP009, K = E.K, N = 5;
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
    ep: '009',                      // 用于文件名和封面顶栏正中的集数（「No.009」）；正片画面里不出现集数
    badge: '第二季',
    robots: ['ur5'],                // robot3d.js 预先加载的 URDF（robots.json；先跑 python3 fetch_robots.py）
    title: '运动旋量',
    name: 'Twist：角速度 + 线速度',
    pitch: '任何刚体速度，都能看成绕一根轴边转边移',
    meta: '<b>机器人学骨架：刚体变换</b><br>Robotics Notebooks · 懂齐次变换 · 不用代码',
    introAt: { title: 0.1, name: 0.4, hero: 0 },
    // 划重点 / 片尾里的加粗词：角速度 violet；v、点的速度 accent；运动旋量 亮黄；螺旋轴 warn；螺距 good
    boldColors: { '角速度': 'var(--violet)', 'ω': 'var(--violet)', 'v': 'var(--demo-accent)', '线速度': 'var(--demo-accent)',
      '运动旋量': 'var(--hi)', '螺旋轴': 'var(--demo-warn)', '螺距': 'var(--demo-good)', '不是': 'var(--demo-bad)' },
    introDur: 27.79,
    hero: function (K) { return E.hero ? E.hero(K) : placeholder('主图', '主图还没写').build(); },
    scenes: scenes,
    // 片尾总结：四张卡片，结论一行（at 是弹出的分镜时间，和 ../ep009.py 的 recap 句对齐）
    recapDur: 32.58,
    recap: {
      act: '本集总结',
      title: '用六个数说清刚体的速度',
      items: [
        { c: 'var(--violet)', n: '① 各点速度不一样', s: '大小 = 角速度 × 到轴的距离\n方向和连线垂直', at: 1.87 },
        { c: 'var(--hi)', n: '② 运动旋量：六个数', s: '角速度 ω + 此刻在原点那一点的速度 v', at: 8.97 },
        { c: 'var(--demo-warn)', n: '③ 绕螺旋轴，边转边移', s: '轴的方向、轴上一点、螺距', at: 14.51 },
        { c: 'var(--demo-good)', n: '④ 运动旋量 = 螺旋轴 × 转速', s: '转动关节螺距为 0，移动关节只移不转', at: 21.83 }
      ],
      foot: { s: '**运动旋量**：六个数，说清整个刚体的速度', at: 27.13 }
    },
    outroDur: 15.19,
    // 片尾不出题：下一集预告、完整内容的网址卡片和衔接语
    outro: {
      next: { ep: '010', q: '矩阵指数：绕螺旋轴转了一个角度，\n怎么一步算出停在哪、朝哪？', at: 0 },
      link: { s: '<b>完整内容</b><br>imchong.github.io/Robotics_Notebooks', at: 10.18 },
      end: { s: '零基础篇 · 一集只讲一个问题　下集见', at: 13.32 }
    }
  };
})();
