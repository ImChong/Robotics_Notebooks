// 组装 006：六幕按 EP006.scenes[i] 的下标排好（缺的幕先放占位，某一幕 build 出错时画出报错，不拖累别的幕），
// 再交给 stage.html 的 window.EpisodeVideo。旁白与 [from, to] 对时在 ../ep006.py。
(function () {
  var E = window.EP006, K = E.K, N = 6;
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
    ep: '006',                      // 用于文件名和封面顶栏正中的集数（「No.006」）；正片画面里不出现集数
    badge: '第一季',
    robots: ['g1'],                 // robot3d.js 预先加载的 URDF（robots.json；先跑 python3 fetch_robots.py）
    title: '旋转表示',
    name: '欧拉角、四元数、旋转矩阵',
    pitch: '各有各的坑，看完知道什么时候用哪种',
    meta: '<b>数学打底：旋转表示方法对比</b><br>Robotics Notebooks · 懂旋转矩阵 · 不用代码',
    introAt: { title: 0.1, name: 0.4, hero: 0 },
    // 划重点 / 片尾里的加粗词：三种写法各用本色（欧拉角 warn、四元数 accent、旋转矩阵 violet）；坑 bad；扶正、直观 good
    boldColors: { '欧拉角': 'var(--demo-warn)', '四元数': 'var(--demo-accent)', '旋转矩阵': 'var(--violet)', '矩阵': 'var(--violet)',
      '万向锁': E.ERR, '双重覆盖': E.ERR, '冗余': E.ERR, '重合': E.ERR,
      '直观': 'var(--demo-good)', '扶正': 'var(--demo-good)', '同一个朝向': 'var(--hi)', '万向节': 'var(--hi)' },
    introDur: 29.2,
    hero: function (K) { return E.hero ? E.hero(K) : placeholder('主图', '主图还没写').build(); },
    scenes: scenes,
    // 片尾总结：四张卡片，结论一行（at 是弹出的分镜时间，和 ../ep006.py 的 recap 句对齐）
    recapDur: 31.94,
    recap: {
      act: '本集总结',
      title: '三种旋转表示怎么选',
      items: [
        { c: 'var(--hi)', n: '① 同一个朝向，三种写法', s: '欧拉角 3 个数，四元数 4 个，矩阵 9 个', at: 1.87 },
        { c: 'var(--demo-warn)', n: '② 欧拉角', s: '最直观；俯仰到 90° 会万向锁', at: 9.14 },
        { c: 'var(--demo-accent)', n: '③ 四元数', s: '没有万向锁；q 和 −q 同一个朝向，先对齐', at: 14.01 },
        { c: 'var(--violet)', n: '④ 旋转矩阵', s: '连乘方便；9 个数守 6 个等式，要扶正', at: 19.67 }
      ],
      foot: { s: '**欧拉角**给人看，**四元数**做内部计算，**矩阵**用来连乘', at: 25.53 }
    },
    outroDur: 14.8,
    // 片尾不出题：下一集预告、完整内容的网址卡片、本集核对过的一手资料和衔接语
    outro: {
      next: { ep: '007', q: '位姿 = 朝向 + 位置：\n怎么拼成一个 4×4 的矩阵？', at: 0 },
      link: { s: '<b>完整内容</b><br>imchong.github.io/Robotics_Notebooks', at: 5.95 },
      refs: {
        at: 9.09,
        items: [
          '**《Modern Robotics》** Lynch & Park，2017：第 2、3 章，附录 B',
          '**ROS REP 103**：旋转表示（固定轴 RPY 与欧拉角分开列）',
          '**urdfdom_headers**（URDF 解析库）：rpy 的换算',
          '**ROS tf2** Quaternion::setRPY：绕固定轴的横滚、俯仰、偏航',
          '**Eigen** Quaternion::slerp：点积为负先变号',
          '**MuJoCo 文档** Modeling：朝向在内部一律存成单位四元数',
          '**ROS sensor_msgs/Imu**：IMU 的朝向用四元数'
        ]
      },
      end: { s: '零基础篇 · 一集只讲一个问题　下集见', at: 12.93 }
    }
  };
})();
