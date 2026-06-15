/*
 * 专题视图（Topic Filters）单一事实源。
 * 由 graph.html（专题筛选）与 detail.html / main.js（详情页"属于 X 专题"徽标）共享，
 * 避免两处各写一份命中规则导致漂移。
 *
 * 命中优先级（与 graph.html nodeMatchesTopic 一致）：
 *   excludeSegments 命中 → 直接排除；ids 显式纳入 → 命中；
 *   communities 命中 → 命中；segments 命中任一 → 命中。
 */
(function (global) {
  'use strict';

  var TOPIC_FILTERS = {
    'motion-retargeting': {
      communities: new Set(['community-3']),
      segments: new Set([
        'retargeting', 'retarget', 'gmr', 'nmr', 'reactor', 'sonic', 'exoactor',
        'spider', 'wilor', 'mocap', 'teleoperation', 'deepmimic', 'amp',
        'character', 'animation', 'keyframe', 'pipeline'
      ])
    },
    'grasp': {
      segments: new Set([
        'grasp', 'graspnet', 'anygrasp', 'dexterous', 'manipulation',
        'pick', 'place', 'bimanual', 'curobo'
      ])
    },
    'tactile': {
      segments: new Set([
        'tactile', 'haptic', 'impedance', 'force', 'contact', 'visuo'
      ]),
      excludeSegments: new Set(['reinforcement'])
    },
    'communication': {
      segments: new Set([
        'ethercat', 'can', 'uart', 'dds', 'foxglove', 'rs485', 'rs232',
        'serial', 'communication', 'protocol', 'bus', 'protocols', 'firmware'
      ])
    },
    'wbc': {
      communities: new Set(['community-0']),
      segments: new Set([
        'wbc', 'tsid', 'hqp', 'cbf', 'clf', 'whole', 'body', 'balance', 'hierarchical'
      ])
    },
    'locomotion': {
      communities: new Set(['community-11', 'community-9']),
      segments: new Set([
        'locomotion', 'gait', 'mpc', 'zmp', 'lip', 'walking', 'swing', 'stance', 'capture'
      ])
    },
    'vla': {
      communities: new Set(['community-5']),
      segments: new Set([
        'vla', 'foundation', 'octo', 'openvla', 'rt', 'pi0', 'gr00t'
      ])
    },
    'learning': {
      communities: new Set(['community-4', 'community-6']),
      segments: new Set([
        'imitation', 'reinforcement', 'ppo', 'sac', 'behavior', 'cloning', 'dreamer'
      ])
    },
    'sim2real': {
      segments: new Set([
        'sim2real', 'randomization', 'domain'
      ])
    },
    'state-estimation': {
      segments: new Set([
        'estimation', 'ekf', 'ukf', 'slam', 'vio', 'odometry'
      ])
    },
    'wbt': {
      segments: new Set([
        'wbt', 'tracking', 'beyondmimic', 'sdamp', 'heracles', 'opentrack',
        'maskedmimic', 'sonic', 'any2any', 'twist', 'twist2'
      ])
    },
    'cross-embodiment': {
      segments: new Set([
        'embodiment', 'any2any', 'transfer'
      ])
    },
    'safe-fine-tuning': {
      communities: new Set(['community-13']),
      segments: new Set([
        'safe', 'safety', 'cbf', 'clf', 'barrier', 'lyapunov', 'slowrl', 'lora', 'cmdp'
      ])
    },
    'vision-backbone': {
      segments: new Set([
        'backbone', 'backbones', 'cnn', 'vit', 'resnet', 'yolo', 'detection'
      ]),
      ids: new Set([
        'wiki/concepts/visual-representation-for-policy.md',
        'wiki/concepts/generative-vision-pretraining.md'
      ])
    }
  };

  /* 专题展示元信息（emoji + 简称），与 graph.html chips 顺序一致。 */
  var TOPIC_META = {
    'motion-retargeting': { emoji: '🤸', label: '动作重定向' },
    'grasp': { emoji: '🤏', label: '抓取' },
    'tactile': { emoji: '✋', label: '触觉' },
    'communication': { emoji: '🔌', label: '通信协议' },
    'wbc': { emoji: '🦾', label: 'WBC' },
    'locomotion': { emoji: '🚶', label: 'Locomotion' },
    'vla': { emoji: '👀', label: 'VLA' },
    'learning': { emoji: '🎓', label: 'IL/RL' },
    'sim2real': { emoji: '🔁', label: 'Sim2Real' },
    'state-estimation': { emoji: '📊', label: '状态估计' },
    'wbt': { emoji: '🕺', label: 'WBT' },
    'cross-embodiment': { emoji: '🔀', label: '跨具身' },
    'safe-fine-tuning': { emoji: '🛡️', label: '安全微调' },
    'vision-backbone': { emoji: '👁️', label: '视觉骨干' }
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

  global.RNTopicFilters = {
    TOPIC_FILTERS: TOPIC_FILTERS,
    TOPIC_META: TOPIC_META,
    nodeSegments: nodeSegments,
    matches: matches,
    topicsForNode: topicsForNode
  };
})(typeof window !== 'undefined' ? window : this);
