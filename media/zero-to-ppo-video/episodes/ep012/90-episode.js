// 组装 012：六幕按 EP012.scenes[i] 的下标排好（缺的幕先放占位，某一幕 build 出错时画出报错，不拖累别的幕），
// 再交给 stage.html 的 window.EpisodeVideo。旁白与 [from, to] 对时在 ../ep012.py。
(function () {
  var E = window.EP012, K = E.K, N = 6;
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
    ep: '012',                      // 用于文件名和封面顶栏正中的集数（「No.012」）；正片画面里不出现集数
    badge: '第二季',
    robots: ['ur5', 'g1'],          // robot3d.js 预先加载的 URDF（robots.json；先跑 python3 fetch_robots.py）
    title: '逆运动学',
    name: '手的位姿 → 关节角',   // 「末端」前几集没讲过，同 011 封面去掉没讲过的词
    pitch: '够不到就无解，够得到往往不止一解',
    meta: '<b>机器人学骨架：正逆运动学</b><br>Robotics Notebooks · 懂正运动学 · 不用代码',
    introAt: { title: 0.1, name: 0.4, hero: 0 },
    // 划重点 / 片尾里的加粗词（配色照「画面配色」和 00-common.js）：逆向运动学亮黄；肘上亮黄、肘下青；位置 / 手腕目标紫；角度蓝；
    // 没有解 / 用不了橙；圆环（工作空间）绿
    boldColors: { '逆向运动学': 'var(--hi)', '正向运动学': 'var(--demo-accent)', '肘上': 'var(--hi)', '肘下': '#4fc1ff', '手腕': 'var(--violet)',
      '角度': 'var(--demo-accent)', '开平方': 'var(--demo-warn)', '圆环': 'var(--demo-good)', '无解': '#ff7b3a', '没有': '#ff7b3a',
      '用不了': '#ff7b3a', '8 组': 'var(--hi)', '16 组': 'var(--hi)', '无穷多组': 'var(--hi)' },
    introDur: 25.39,
    hero: function (K) { return E.hero ? E.hero(K) : placeholder('主图', '主图还没写').build(); },
    scenes: scenes,
    // 片尾总结：四张卡片，结论一行（at 是弹出的分镜时间，和 ../ep012.py 的 recap 句对齐）
    recapDur: 37.54,
    recap: {
      act: '本集总结',
      title: '位姿 → 关节角：倒过来求',
      items: [
        { c: 'var(--hi)', n: '① 逆向运动学', s: '给出手要到的位姿，求每个关节转多少', at: 1.87 },
        { c: 'var(--demo-warn)', n: '② 两个圆的交点', s: '开平方有正负两个：肘上、肘下', at: 8.23 },
        { c: '#ff7b3a', n: '③ 太远、太近：无解', s: '手能到的地方是一个圆环\n环里 2 组，边上 1 组，环外 0 组', at: 14.66 },
        { c: 'var(--demo-good)', n: '④ 六个关节：几组解', s: '这台（UR5）最多 8 组，一般最多 16 组\n关节再多：无穷多组', at: 22.08 }
      ],
      foot: { s: '**正向运动学**只有一个答案；**逆向运动学**可能好几组，也可能**没有**', at: 30.1 }
    },
    outroDur: 14.59,
    // 片尾不出题：下一集预告、完整内容的网址卡片和衔接语
    outro: {
      next: { ep: '013', q: '雅可比矩阵：每个关节转得多快，\n手就走得多快？', at: 0 },
      link: { s: '<b>完整内容</b><br>imchong.github.io/Robotics_Notebooks', at: 5.74 },
      refs: {
        at: 8.88,
        items: [
          '**《Modern Robotics》** Lynch & Park：第 6 章开头、式 (6.1)、图 6.1',
          '　两连杆：工作空间是圆环，0 / 1 / 2 组解，elbow-up / elbow-down',
          '**《Modern Robotics》** 6.1.1 节、图 6.5；附录 B.1（手腕：β 两组解）',
          '**《Modern Robotics》** 6.5 节；6.7 节：一般 6R 最多 16 组解',
          '　（Lee & Liang；Raghavan & Roth）',
          '**ROS-Industrial ur_kinematics**（Hawkins，2013）ur_kin.h / .cpp：',
          '　UR 机械臂解析逆解 maximum of 8，肩、腕、肘各两个',
          '**Python 文档** math.atan2；**UR5 URDF**（ur_description）',
          '**Unitree G1 URDF**：膝 −5°～165°，一条胳膊 7 个转动关节'
        ]
      },
      end: { s: '零基础篇 · 一集只讲一个问题　下集见', at: 12.72 }
    }
  };
})();
