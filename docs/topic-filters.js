/*
 * 专题视图（Topic Filters）单一事实源。
 * 由 graph.html（专题筛选）与 detail.html / main.js（详情页"属于 X 专题"徽标）共享，
 * 避免两处各写一份命中规则导致漂移。
 *
 * 命中优先级（与 graph.html nodeMatchesTopic 一致）：
 *   excludeSegments 命中 → 直接排除；ids 显式纳入 → 命中；
 *   communities 命中 → 命中；segments 命中任一 → 命中。
 *
 * 每个专题在 wiki/overview/topic-*.md 有对应「汇总节点」（TOPIC_META.wikiPath），
 * 并写入 TOPIC_FILTERS[key].ids 以保证专题视图下始终可见。
 */
(function (global) {
  'use strict';

  var TOPIC_HUB_IDS = {
    'motion-retargeting': 'wiki/overview/topic-motion-retargeting.md',
    'grasp': 'wiki/overview/topic-grasp.md',
    'tactile': 'wiki/overview/topic-tactile.md',
    'communication': 'wiki/overview/topic-communication.md',
    'wbc': 'wiki/overview/topic-wbc.md',
    'locomotion': 'wiki/overview/topic-locomotion.md',
    'vla': 'wiki/overview/topic-vla.md',
    'learning': 'wiki/overview/topic-learning.md',
    'sim2real': 'wiki/overview/topic-sim2real.md',
    'state-estimation': 'wiki/overview/topic-state-estimation.md',
    'wbt': 'wiki/overview/topic-wbt.md',
    'cross-embodiment': 'wiki/overview/topic-cross-embodiment.md',
    'safe-fine-tuning': 'wiki/overview/topic-safe-fine-tuning.md',
    'vision-backbone': 'wiki/overview/topic-vision-backbone.md',
    'data-pipeline': 'wiki/overview/topic-data-pipeline.md'
  };

  function hubIdSet(key) {
    var hub = TOPIC_HUB_IDS[key];
    return hub ? new Set([hub]) : null;
  }

  function mergeIds(key, extra) {
    var base = hubIdSet(key);
    if (!extra) return base;
    if (!base) return extra;
    var merged = new Set(base);
    extra.forEach(function (id) { merged.add(id); });
    return merged;
  }

  var TOPIC_FILTERS = {
    'motion-retargeting': {
      communities: new Set(['community-3']),
      segments: new Set([
        'retargeting', 'retarget', 'gmr', 'nmr', 'reactor', 'sonic', 'exoactor',
        'spider', 'wilor', 'mocap', 'teleoperation', 'deepmimic', 'amp',
        'character', 'animation', 'keyframe', 'pipeline'
      ]),
      ids: mergeIds('motion-retargeting')
    },
    'grasp': {
      segments: new Set([
        'grasp', 'graspnet', 'anygrasp', 'dexterous', 'manipulation',
        'pick', 'place', 'bimanual', 'curobo'
      ]),
      ids: mergeIds('grasp')
    },
    'tactile': {
      segments: new Set([
        'tactile', 'haptic', 'impedance', 'force', 'contact', 'visuo'
      ]),
      excludeSegments: new Set(['reinforcement']),
      ids: mergeIds('tactile')
    },
    'communication': {
      segments: new Set([
        'ethercat', 'can', 'uart', 'dds', 'foxglove', 'rs485', 'rs232',
        'serial', 'communication', 'protocol', 'bus', 'protocols', 'firmware'
      ]),
      ids: mergeIds('communication')
    },
    'wbc': {
      communities: new Set(['community-0']),
      segments: new Set([
        'wbc', 'tsid', 'hqp', 'cbf', 'clf', 'whole', 'body', 'balance', 'hierarchical'
      ]),
      ids: mergeIds('wbc')
    },
    'locomotion': {
      communities: new Set(['community-11', 'community-9']),
      segments: new Set([
        'locomotion', 'gait', 'mpc', 'zmp', 'lip', 'walking', 'swing', 'stance', 'capture'
      ]),
      ids: mergeIds('locomotion')
    },
    'vla': {
      communities: new Set(['community-5']),
      segments: new Set([
        'vla', 'foundation', 'octo', 'openvla', 'rt', 'pi0', 'gr00t'
      ]),
      ids: mergeIds('vla')
    },
    'learning': {
      communities: new Set(['community-4', 'community-6']),
      segments: new Set([
        'imitation', 'reinforcement', 'ppo', 'sac', 'behavior', 'cloning', 'dreamer'
      ]),
      ids: mergeIds('learning')
    },
    'sim2real': {
      segments: new Set([
        'sim2real', 'randomization', 'domain'
      ]),
      ids: mergeIds('sim2real')
    },
    'state-estimation': {
      segments: new Set([
        'estimation', 'ekf', 'ukf', 'slam', 'vio', 'odometry'
      ]),
      ids: mergeIds('state-estimation')
    },
    'wbt': {
      segments: new Set([
        'wbt', 'tracking', 'beyondmimic', 'sdamp', 'heracles', 'opentrack',
        'maskedmimic', 'sonic', 'any2any', 'twist', 'twist2'
      ]),
      ids: mergeIds('wbt')
    },
    'cross-embodiment': {
      segments: new Set([
        'embodiment', 'any2any', 'transfer'
      ]),
      ids: mergeIds('cross-embodiment')
    },
    'safe-fine-tuning': {
      communities: new Set(['community-13']),
      segments: new Set([
        'safe', 'safety', 'cbf', 'clf', 'barrier', 'lyapunov', 'slowrl', 'lora', 'cmdp'
      ]),
      ids: mergeIds('safe-fine-tuning')
    },
    'vision-backbone': {
      segments: new Set([
        'backbone', 'backbones', 'cnn', 'vit', 'resnet', 'yolo', 'detection'
      ]),
      ids: mergeIds('vision-backbone', new Set([
        'wiki/concepts/visual-representation-for-policy.md',
        'wiki/concepts/generative-vision-pretraining.md'
      ]))
    },
    'data-pipeline': {
      segments: new Set([
        'dataset', 'datasets', 'amass', 'lafan1', 'lafan', 'omomo',
        'phuma', 'everyday', 'retargeting', 'retarget', 'retargeter',
        'omniretarget', 'mocap', 'freemocap'
      ]),
      ids: mergeIds('data-pipeline', new Set([
        'wiki/queries/humanoid-training-data-pipeline.md',
        'wiki/concepts/motion-data-quality.md',
        'wiki/concepts/motion-retargeting.md',
        'wiki/comparisons/humanoid-reference-motion-datasets.md'
      ]))
    }
  };

  /* 专题展示元信息（emoji + 简称 + 汇总节点 + 导读），与 graph.html chips 顺序一致。 */
  var TOPIC_META = {
    'motion-retargeting': {
      emoji: '🤸',
      label: '动作重定向 (Motion Retargeting)',
      wikiPath: TOPIC_HUB_IDS['motion-retargeting'],
      description: '把人体/动物参考动作映射到异构机器人骨架，衔接 MoCap、IK 重定向与 WBT 训练数据。'
    },
    'grasp': {
      emoji: '🤏',
      label: '抓取 (Grasp)',
      wikiPath: TOPIC_HUB_IDS.grasp,
      description: '接触丰富环境下的感知抓取、灵巧操作与 loco-manip 操作子栈。'
    },
    'tactile': {
      emoji: '✋',
      label: '触觉 (Tactile)',
      wikiPath: TOPIC_HUB_IDS.tactile,
      description: '触觉传感、视触觉融合与阻抗/力控闭环，支撑稳定抓取与交互。'
    },
    'communication': {
      emoji: '🔌',
      label: '通信协议 (Communication)',
      wikiPath: TOPIC_HUB_IDS.communication,
      description: '电机驱动、EtherCAT/CAN 现场总线与 ROS 2 / LCM 中间件的底层数据链路。'
    },
    'wbc': {
      emoji: '🦾',
      label: '全身控制 (WBC)',
      wikiPath: TOPIC_HUB_IDS.wbc,
      description: '浮基人形上的全身任务/力分配，TSID、HQP 与 CBF 安全约束。'
    },
    'locomotion': {
      emoji: '🚶',
      label: '步态与移动 (Locomotion)',
      wikiPath: TOPIC_HUB_IDS.locomotion,
      description: '腿式与人形在不同地形上的步态生成、平衡与感知式移动。'
    },
    'vla': {
      emoji: '👀',
      label: '视觉-语言-动作 (VLA)',
      wikiPath: TOPIC_HUB_IDS.vla,
      description: '视觉-语言-动作统一建模与 BFM 身体接口，面向多任务 loco-manip。'
    },
    'learning': {
      emoji: '🎓',
      label: '模仿/强化学习 (IL/RL)',
      wikiPath: TOPIC_HUB_IDS.learning,
      description: '强化学习、模仿学习及 PPO/SAC 等范式的选型与机器人落地要点。'
    },
    'sim2real': {
      emoji: '🔁',
      label: '仿真到现实 (Sim2Real)',
      wikiPath: TOPIC_HUB_IDS.sim2real,
      description: '仿真策略迁移真机：域随机化、系统辨识与残差适配路线。'
    },
    'state-estimation': {
      emoji: '📊',
      label: '状态估计 (State Estimation)',
      wikiPath: TOPIC_HUB_IDS['state-estimation'],
      description: '多传感器融合、SLAM/VIO/LIO 与 Kalman/优化估计框架。'
    },
    'wbt': {
      emoji: '🕺',
      label: '全身运动跟踪 (WBT)',
      wikiPath: TOPIC_HUB_IDS.wbt,
      description: '全身参考动作跟踪：重定向→训练→跨具身→真机部署的端到端流水线。'
    },
    'cross-embodiment': {
      emoji: '🔀',
      label: '跨具身迁移 (Cross-Embodiment)',
      wikiPath: TOPIC_HUB_IDS['cross-embodiment'],
      description: '跨机器人形态/仿真-真机的技能与动作迁移策略。'
    },
    'safe-fine-tuning': {
      emoji: '🛡️',
      label: '安全微调 (Safe Fine-Tuning)',
      wikiPath: TOPIC_HUB_IDS['safe-fine-tuning'],
      description: '真机在线 RL 适配：低秩残差、CBF/CLF 安全壳与 Recovery 兜底。'
    },
    'vision-backbone': {
      emoji: '👁️',
      label: '视觉骨干 (Vision Backbone)',
      wikiPath: TOPIC_HUB_IDS['vision-backbone'],
      description: 'CNN/ViT 骨干→检测头→策略输入的视觉表征与选型。'
    },
    'data-pipeline': {
      emoji: '📦',
      label: '训练数据 (Data Pipeline)',
      wikiPath: TOPIC_HUB_IDS['data-pipeline'],
      description: '原始动作捕捉/视频→质量评估→重定向→RL/IL 策略输入的端到端数据链路。'
    }
  };

  function nodeSegments(node) {
    if (node && node._segs) return node._segs;
    var base = ((node && node.id) || '').toLowerCase().replace(/\.md$/, '');
    var segs = new Set(base.split(/[/._-]/).filter(Boolean));
    if (node) node._segs = segs;
    return segs;
  }

  /* 判定单个节点是否命中某专题（topicKey 为 'all' 时恒真）。 */
  function matches(node, topicKey) {
    if (topicKey === 'all') return true;
    var cfg = TOPIC_FILTERS[topicKey];
    if (!cfg) return true;
    var segs = nodeSegments(node);
    if (cfg.excludeSegments) {
      for (var ex of cfg.excludeSegments) if (segs.has(ex)) return false;
    }
    if (cfg.ids && cfg.ids.has(node.id)) return true;
    if (cfg.communities && node.community && cfg.communities.has(node.community)) return true;
    if (cfg.segments) {
      for (var seg of cfg.segments) if (segs.has(seg)) return true;
    }
    return false;
  }

  /* 返回节点命中的全部专题 key 列表（不含 'all'）。 */
  function topicsForNode(node) {
    var out = [];
    for (var key in TOPIC_FILTERS) {
      if (matches(node, key)) out.push(key);
    }
    return out;
  }

  /* 某专题的汇总节点 wiki 路径；无则 null。 */
  function topicHubPath(topicKey) {
    return TOPIC_HUB_IDS[topicKey] || null;
  }

  /* 节点是否为任一专题（或指定专题）的汇总锚点。 */
  function isTopicHub(node, topicKey) {
    if (!node || !node.id) return false;
    if (topicKey && topicKey !== 'all') {
      return TOPIC_HUB_IDS[topicKey] === node.id;
    }
    for (var k in TOPIC_HUB_IDS) {
      if (TOPIC_HUB_IDS[k] === node.id) return true;
    }
    return false;
  }

  global.RNTopicFilters = {
    TOPIC_FILTERS: TOPIC_FILTERS,
    TOPIC_META: TOPIC_META,
    TOPIC_HUB_IDS: TOPIC_HUB_IDS,
    nodeSegments: nodeSegments,
    matches: matches,
    topicsForNode: topicsForNode,
    topicHubPath: topicHubPath,
    isTopicHub: isTopicHub
  };
})(typeof window !== 'undefined' ? window : this);
