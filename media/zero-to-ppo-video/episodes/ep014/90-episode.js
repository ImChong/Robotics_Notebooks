// 组装 014：六幕按 EP013.scenes[i] 的下标排好（缺的幕先放占位，某一幕 build 出错时画出报错，不拖累别的幕），
// 再交给 stage.html 的 window.EpisodeVideo。旁白与 [from, to] 对时在 ../ep014.py。
(function () {
  var E = window.EP014, K = E.K, N = 6;
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
    ep: '014',                      // 用于文件名和封面顶栏正中的集数（「No.014」）；正片画面里不出现集数
    badge: '第二季',
    robots: ['ur5'],                // robot3d.js 预先加载的 URDF（robots.json；先跑 python3 fetch_robots.py）
    title: '两种雅可比',
    name: 'Space ↔ Body：伴随变换',
    pitch: '目标写在哪个坐标系，就用对应的那个',
    meta: '<b>机器人学骨架：雅可比与速度运动学</b><br>Robotics Notebooks · 懂雅可比矩阵 · 不用代码',
    introAt: { title: 0.1, name: 0.4, hero: 0 },
    // 划重点 / 片尾里的加粗词（配色照「画面配色」和 00-common.js）：伴随变换亮黄；换一个点（第一步）紫、换一套轴（第二步）蓝；
    // 两种雅可比、两种坐标系用白；螺旋轴琥珀；运动旋量琥珀（同 009–013）
    boldColors: { '伴随变换': 'var(--hi)', '伴随表示': 'var(--hi)', '空间雅可比': 'var(--text)', '物体雅可比': 'var(--text)', '固定坐标系': 'var(--text)', '物体坐标系': 'var(--text)',
      '站在底座上看': 'var(--text)', '站在手上看': 'var(--text)', '换一个点': 'var(--violet)', '换一套轴': 'var(--demo-accent)', '手的速度': 'var(--violet)',
      '螺旋轴': 'var(--demo-warn)', '运动旋量': 'var(--demo-warn)', '物体运动旋量': 'var(--demo-warn)', '雅可比': 'var(--hi)', '目标': 'var(--demo-good)' },
    introDur: 32.87,
    hero: function (K) { return E.hero ? E.hero(K) : placeholder('主图', '主图还没写').build(); },
    scenes: scenes,
    // 片尾总结：四张卡片，结论一行（at 是弹出的分镜时间，和 ../ep014.py 的 recap 句对齐）
    recapDur: 42.19,
    recap: {
      act: '本集总结',
      title: '同一个速度，两种看法',
      items: [
        { c: 'var(--text)', n: '① 站在底座上看：空间雅可比', s: '后 3 个数 = 原点那一点的速度，按底座的轴算', at: 2.77 },
        { c: 'var(--violet)', n: '② 站在手上看：物体雅可比', s: '后 3 个数 = 手这一点的速度，按手的轴算', at: 11.27 },
        { c: 'var(--hi)', n: '③ 伴随变换：先换点，再换轴', s: '6 × 6，只用手的位姿；物体雅可比 = 它 × 空间雅可比', at: 19.37 },
        { c: 'var(--demo-good)', n: '④ 目标写在哪，就配哪一个', s: '目标和雅可比站在同一个地方，算出的转速一样', at: 29.84 }
      ],
      foot: { s: '同一个速度，两种看法，中间隔着一个**伴随变换**', at: 35.54 }
    },
    outroDur: 19.74,
    // 片尾不出题：下一集预告、完整内容的网址卡片和衔接语
    outro: {
      next: { ep: '015', q: '数值逆运动学：写不出公式时，\n怎么用雅可比一步一步把关节挪过去？', at: 0 },
      link: { s: '<b>完整内容</b><br>imchong.github.io/Robotics_Notebooks', at: 9.49 },
      refs: {
        at: 12.93,
        items: [
          '**《Modern Robotics》** Lynch & Park：3.3.2 节、图 3.17',
          '　空间运动旋量 V_s：v_s 是此刻在 {s} 原点那一点的速度',
          '　物体运动旋量 V_b：{b} 原点的速度，按 {b} 的轴写',
          '**《Modern Robotics》** 定义 3.20（伴随表示 [Ad_T]）、式 (3.78)(3.83)(3.84)',
          '**《Modern Robotics》** 5.1、5.1.2（定义 5.4）、5.1.3（图 5.9）、5.1.4 节式 (5.22)(5.23)',
          '**《Modern Robotics》** 5.4 节（手这一点的线速度用 J_b）、6.2、11.3.3 节',
          '**官方代码** Adjoint、TransInv、JacobianSpace、JacobianBody、FKinSpace 核对',
          '**modern_robotics 1.1.1、NumPy 2.5.3** 实跑'
        ]
      },
      end: { s: '零基础篇 · 一集只讲一个问题　下集见', at: 17.07 }
    }
  };
})();
