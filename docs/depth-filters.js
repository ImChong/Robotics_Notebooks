/*
 * 路线视图（Depth Filters）单一事实源。
 * 包含主路线 roadmap/motion-control.md + 策展的 23 条 roadmap/depth-*.md 纵深路线；
 * 由 graph.html（路线筛选）、detail.html（「所属路线」）与 roadmap.html（「路线视图」徽标）共享。
 *
 * 命中优先级（与 graph.html nodeMatchesDepth 一致）：
 *   excludeSegments 命中 → 直接排除；ids 显式纳入 → 命中；
 *   communities 命中 → 命中；segments 命中任一 → 命中。
 *
 * segments 匹配规则：
 *   - 无分隔符的单 token：与 node id 按 [/._-] 切开后的词元集合做精确命中；
 *   - 含 - / _ 的多 token：在把 id 分隔符归一为 '-' 后的路径串上做子串命中
 *     （因此 loco-manip 可命中 .../loco-manipulation...，sim-to-real 可命中
 *     ...closing_sim_to_real_gap...）。勿把过短词干放进多 token segments。
 *
 * 每条路线的汇总锚点是对应 roadmap/*.md（DEPTH_META.wikiPath），
 * 并写入 DEPTH_FILTERS[key].ids 以保证路线视图下始终可见。
 */
(function (global) {
  'use strict';

  /* 主路线置顶，其后与首页 / roadmap/README 一致的纵深历史顺序（23 条）。 */
  var DEPTH_ORDER = [
    'motion-control',
    'teleoperation',
    'torque-motor-design',
    'classical-control',
    'humanoid-hardware-design',
    'safe-control',
    'contact-manipulation',
    'navigation',
    'imitation-learning',
    'rl-locomotion',
    'loco-manipulation',
    'humanoid-soccer',
    'motion-retargeting',
    'humanoid-swarm-performance',
    'sim2real',
    'humanoid-boxing',
    'icl',
    'bfm',
    'embodied-eval',
    'perceptive-locomotion',
    'motion-generation',
    'vla',
    'real2sim',
    'wam'
  ];

  var DEPTH_HUB_IDS = {
    'motion-control': 'roadmap/motion-control.md',
    'teleoperation': 'roadmap/depth-teleoperation.md',
    'torque-motor-design': 'roadmap/depth-torque-motor-design.md',
    'classical-control': 'roadmap/depth-classical-control.md',
    'humanoid-hardware-design': 'roadmap/depth-humanoid-hardware-design.md',
    'safe-control': 'roadmap/depth-safe-control.md',
    'contact-manipulation': 'roadmap/depth-contact-manipulation.md',
    'navigation': 'roadmap/depth-navigation.md',
    'imitation-learning': 'roadmap/depth-imitation-learning.md',
    'rl-locomotion': 'roadmap/depth-rl-locomotion.md',
    'loco-manipulation': 'roadmap/depth-loco-manipulation.md',
    'humanoid-soccer': 'roadmap/depth-humanoid-soccer.md',
    'motion-retargeting': 'roadmap/depth-motion-retargeting.md',
    'humanoid-swarm-performance': 'roadmap/depth-humanoid-swarm-performance.md',
    'sim2real': 'roadmap/depth-sim2real.md',
    'humanoid-boxing': 'roadmap/depth-humanoid-boxing.md',
    'icl': 'roadmap/depth-icl.md',
    'bfm': 'roadmap/depth-bfm.md',
    'embodied-eval': 'roadmap/depth-embodied-eval.md',
    'perceptive-locomotion': 'roadmap/depth-perceptive-locomotion.md',
    'motion-generation': 'roadmap/depth-motion-generation.md',
    'vla': 'roadmap/depth-vla.md',
    'real2sim': 'roadmap/depth-real2sim.md',
    'wam': 'roadmap/depth-wam.md'
  };

  function hubIdSet(key) {
    var hub = DEPTH_HUB_IDS[key];
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

  var DEPTH_FILTERS = {
    /* 主路线：以 motion-control.md 正文链出的 wiki 节点为命中集（不含 depth-* 纵深页）。 */
    'motion-control': {
      segments: new Set(),
      ids: mergeIds('motion-control', [
        'wiki/comparisons/gmr-vs-nmr-vs-reactor.md',
        'wiki/comparisons/humanoid-reference-motion-datasets.md',
        'wiki/comparisons/mpc-vs-rl.md',
        'wiki/comparisons/ppo-vs-sac.md',
        'wiki/comparisons/wbc-vs-rl.md',
        'wiki/concepts/capture-point-dcm.md',
        'wiki/concepts/centroidal-dynamics.md',
        'wiki/concepts/contact-dynamics.md',
        'wiki/concepts/domain-randomization.md',
        'wiki/concepts/embodied-rl-minimal-closed-loop.md',
        'wiki/concepts/floating-base-dynamics.md',
        'wiki/concepts/hqp.md',
        'wiki/concepts/kinematic-vs-dynamic-feasibility.md',
        'wiki/concepts/lip-zmp.md',
        'wiki/concepts/motion-data-quality.md',
        'wiki/concepts/motion-retargeting-pipeline.md',
        'wiki/concepts/motion-retargeting.md',
        'wiki/concepts/optimal-control.md',
        'wiki/concepts/sim2real.md',
        'wiki/concepts/state-estimation.md',
        'wiki/concepts/system-identification.md',
        'wiki/concepts/tsid.md',
        'wiki/concepts/whole-body-control.md',
        'wiki/entities/amass.md',
        'wiki/entities/crocoddyl.md',
        'wiki/entities/dreamwaq-plus.md',
        'wiki/entities/extreme-parkour.md',
        'wiki/entities/hands-on-rl-book.md',
        'wiki/entities/humanoid-robot.md',
        'wiki/entities/isaac-gym-isaac-lab.md',
        'wiki/entities/lafan1-dataset.md',
        'wiki/entities/learn-robotics-qqfly-guide.md',
        'wiki/entities/legged-gym.md',
        'wiki/entities/linear-algebra-curriculum.md',
        'wiki/entities/modern-robotics-book.md',
        'wiki/entities/numerical-optimization-curriculum.md',
        'wiki/entities/pinocchio.md',
        'wiki/entities/pybullet.md',
        'wiki/formalizations/contact-wrench-cone.md',
        'wiki/formalizations/lqr.md',
        'wiki/formalizations/mdp.md',
        'wiki/formalizations/motion-retargeting-objective.md',
        'wiki/formalizations/pomdp.md',
        'wiki/formalizations/se3-representation.md',
        'wiki/formalizations/tsid-formulation.md',
        'wiki/formalizations/zmp-lip.md',
        'wiki/methods/amp-reward.md',
        'wiki/methods/behavior-cloning.md',
        'wiki/methods/dagger.md',
        'wiki/methods/deepmimic.md',
        'wiki/methods/her.md',
        'wiki/methods/imitation-learning.md',
        'wiki/methods/model-predictive-control.md',
        'wiki/methods/motion-retargeting-gmr.md',
        'wiki/methods/policy-optimization.md',
        'wiki/methods/reinforcement-learning.md',
        'wiki/methods/trajectory-optimization.md',
        'wiki/methods/vla.md',
        'wiki/overview/bfm-category-03-intrinsic-reward-pretraining.md',
        'wiki/overview/shenlan-embodied-ai-fundamentals-series.md',
        'wiki/queries/humanoid-motion-control-know-how.md',
        'wiki/queries/mpc-solver-selection.md',
        'wiki/queries/mpc-tuning-guide.md',
        'wiki/queries/open-source-motion-control-projects.md',
        'wiki/queries/robot-policy-debug-playbook.md',
        'wiki/queries/sim2real-checklist.md',
        'wiki/queries/wbc-implementation-guide.md',
        'wiki/queries/wbc-tuning-guide.md',
        'wiki/roadmaps/humanoid-control-roadmap.md',
        'wiki/tasks/locomotion.md',
        'wiki/tasks/manipulation.md'
      ])
    },
    'teleoperation': {
      segments: new Set([
        'teleoperation', 'teleop', 'teleoperate', 'exoskeleton', 'mocap',
        'visionpro', 'open-television', 'homie', 'textop', 'dexumi', 'osmo',
        'data-gloves', 'vision-teleop'
      ]),
      ids: mergeIds('teleoperation', [
        'wiki/tasks/teleoperation.md',
        'wiki/comparisons/data-gloves-vs-vision-teleop.md'
      ])
    },
    'torque-motor-design': {
      segments: new Set([
        'torque', 'motor', 'actuator', 'foc', 'qdd', 'electromagnetic',
        'winding', 'dynamometer', 'simplefoc', 'pyleecan', 'femm',
        'field-oriented', 'armature', 'friction-compensation', 'roller-screw'
      ]),
      ids: mergeIds('torque-motor-design', [
        'wiki/overview/hub-actuator-drive-chain.md',
        'wiki/queries/actuator-drive-chain-selection-loop.md',
        'wiki/entities/simplefoc.md',
        'wiki/entities/kicad.md',
        'wiki/entities/altium-designer.md',
        'wiki/concepts/field-oriented-control.md',
        'wiki/concepts/armature-modeling.md',
        'wiki/methods/sim2real-joint-sysid-experiment-design.md',
        'wiki/concepts/friction-compensation.md',
        'wiki/concepts/planetary-roller-screw-humanoid-leg-actuation.md'
      ])
    },
    'classical-control': {
      segments: new Set([
        'wbc', 'tsid', 'hqp', 'mpc', 'zmp', 'lip', 'centroidal', 'whole',
        'body', 'balance', 'hierarchical', 'model-predictive',
        'capture-point', 'floating-base', 'optimal-control', 'crocoddyl'
      ]),
      ids: mergeIds('classical-control', [
        'wiki/overview/hub-wbc.md',
        'wiki/concepts/whole-body-control.md',
        'wiki/methods/model-predictive-control.md',
        'wiki/concepts/capture-point-dcm.md',
        'wiki/concepts/floating-base-dynamics.md',
        'wiki/concepts/optimal-control.md',
        'wiki/entities/crocoddyl.md'
      ])
    },
    'humanoid-hardware-design': {
      segments: new Set([
        'hardware', 'mechanical', 'chassis', 'ethercat', 'can', 'uart',
        'dds', 'rs485', 'communication', 'protocol', 'bus', 'firmware',
        'power', 'electronics', 'kicad', 'altium'
      ]),
      ids: mergeIds('humanoid-hardware-design', [
        'wiki/overview/hub-communication.md',
        'wiki/overview/hub-systems-engineering.md',
        'wiki/overview/hub-actuator-drive-chain.md'
      ])
    },
    'safe-control': {
      segments: new Set([
        'cbf', 'clf', 'safe', 'safety', 'barrier', 'lyapunov', 'cmdp',
        'recovery', 'shield'
      ]),
      ids: mergeIds('safe-control', [
        'wiki/overview/hub-safe-fine-tuning.md',
        'wiki/concepts/control-barrier-function.md'
      ])
    },
    'contact-manipulation': {
      segments: new Set([
        'grasp', 'graspnet', 'anygrasp', 'dexterous', 'manipulation',
        'tactile', 'haptic', 'impedance', 'admittance', 'wrench',
        'force', 'compliance', 'contact', 'pick', 'place', 'bimanual',
        'contactmimic', 'dexverse', 'visuo-tactile'
      ]),
      excludeSegments: new Set(['reinforcement']),
      ids: mergeIds('contact-manipulation', [
        'wiki/overview/hub-grasp.md',
        'wiki/overview/hub-tactile.md',
        'wiki/overview/hub-contact-force-control.md',
        'wiki/queries/contact-wrench-closed-loop.md',
        'wiki/entities/paper-contactmimic.md',
        'wiki/entities/paper-dexverse.md',
        'wiki/concepts/visuo-tactile-fusion.md',
        'wiki/concepts/impedance-control.md',
        'wiki/tasks/manipulation.md'
      ])
    },
    'navigation': {
      segments: new Set([
        'navigation', 'slam', 'vio', 'lio', 'nav2', 'navigation2', 'vln',
        'localize', 'odometry', 'mapping', 'path', 'planning', 'cartographer',
        'orb-slam', 'habitat', 'matterport'
      ]),
      ids: mergeIds('navigation', [
        'wiki/overview/hub-state-estimation.md',
        'wiki/concepts/state-estimation.md',
        'wiki/entities/navigation2.md',
        'wiki/entities/cartographer.md',
        'wiki/entities/orb-slam3.md',
        'wiki/entities/habitat-sim.md',
        'wiki/entities/matterport3d-simulator.md'
      ])
    },
    'imitation-learning': {
      segments: new Set([
        'imitation', 'behavior', 'cloning', 'dagger', 'demonstration',
        'il', 'bc', 'act', 'diffusion-policy', 'cross-embodiment',
        'action-chunking', 'behavior-cloning'
      ]),
      ids: mergeIds('imitation-learning', [
        'wiki/overview/hub-learning.md',
        'wiki/overview/hub-cross-embodiment.md',
        'wiki/overview/hub-data-pipeline.md',
        'wiki/methods/imitation-learning.md',
        'wiki/methods/behavior-cloning.md',
        'wiki/methods/dagger.md',
        'wiki/methods/diffusion-policy.md',
        'wiki/methods/action-chunking.md'
      ])
    },
    'rl-locomotion': {
      segments: new Set([
        'locomotion', 'gait', 'walking', 'swing', 'stance', 'ppo', 'sac',
        'reinforcement', 'rl', 'amp', 'deepmimic'
      ]),
      ids: mergeIds('rl-locomotion', [
        'wiki/overview/hub-locomotion.md',
        'wiki/overview/hub-learning.md',
        'wiki/methods/reinforcement-learning.md'
      ])
    },
    'loco-manipulation': {
      segments: new Set([
        'loco-manip', 'loco-manipulation', 'locomanip', 'loco_manip',
        'mobile-manip', 'mobile-manipulation', 'whole-body-manip',
        'whole-body-manipulation', 'locomanipulation', 'ultra-survey',
        'humanoidmimicgen', 'resmimic', 'visualmimic', 'halomi', 'coordex',
        'splitadapter'
      ]),
      ids: mergeIds('loco-manipulation', [
        'wiki/tasks/loco-manipulation.md',
        'wiki/tasks/ultra-survey.md',
        'wiki/concepts/whole-body-coordination.md',
        'wiki/entities/paper-resmimic.md',
        'wiki/entities/paper-humanoidmimicgen.md',
        'wiki/entities/paper-notebook-visualmimic.md',
        'wiki/entities/paper-halomi-humanoid-loco-manipulation.md',
        'wiki/entities/paper-coordex-dexterous-humanoid-loco-manipulation.md',
        'wiki/entities/paper-splitadapter-load-aware-loco-manipulation.md',
        'wiki/entities/paper-pilot-perceptive-loco-manipulation.md',
        'wiki/entities/current-robotics-curr0.md',
        'wiki/entities/flexion-reflect-v1.md'
      ])
    },
    'humanoid-soccer': {
      segments: new Set([
        'soccer', 'football', 'robocup', 'goalkeeper', 'striker', 'ball',
        'soccerdiffusion', 'humanoidarena', 'robonaldo', 'socc'
      ]),
      ids: mergeIds('humanoid-soccer', [
        'wiki/entities/paper-notebook-soccerdiffusion-toward-learning-end-to-end-human.md',
        'wiki/entities/paper-humanoidarena.md',
        'wiki/entities/paper-hrl-stack-26-learning_vision_driven_reactive_socc.md',
        'wiki/entities/smplolympics.md'
      ])
    },
    'motion-retargeting': {
      communities: new Set(['community-3']),
      segments: new Set([
        'retargeting', 'retarget', 'gmr', 'nmr', 'reactor', 'sonic',
        'exoactor', 'spider', 'wilor', 'mocap', 'keyframe', 'animation',
        'freemocap', 'fairmotion', 'omniretarget'
      ]),
      ids: mergeIds('motion-retargeting', [
        'wiki/overview/hub-motion-retargeting.md',
        'wiki/overview/hub-data-pipeline.md',
        'wiki/concepts/motion-retargeting.md',
        'wiki/concepts/motion-retargeting-pipeline.md',
        'wiki/concepts/motion-data-quality.md',
        'wiki/methods/motion-retargeting-gmr.md',
        'wiki/comparisons/gmr-vs-nmr-vs-reactor.md',
        'wiki/comparisons/humanoid-reference-motion-datasets.md',
        'wiki/entities/amass.md',
        'wiki/entities/freemocap.md',
        'wiki/entities/fairmotion.md',
        'wiki/entities/lafan1-dataset.md'
      ])
    },
    'humanoid-swarm-performance': {
      segments: new Set([
        'swarm', 'multi-robot', 'formation', 'choreography', 'performance',
        'coordination', 'crazyswarm', 'teamplay', 'teamhoi', 'marl'
      ]),
      ids: mergeIds('humanoid-swarm-performance', [
        'wiki/entities/crazyswarm2.md',
        'wiki/entities/paper-bfm-23-teamplay.md',
        'wiki/entities/paper-amp-survey-17-teamhoi.md',
        'wiki/concepts/humanoid-multi-robot-coordination.md',
        'wiki/comparisons/ctde-vs-decentralized-marl.md'
      ])
    },
    'sim2real': {
      segments: new Set([
        'sim2real', 'sim-to-real', 'domain', 'randomization', 'system-id',
        'sysid', 'residual', 'adaptation', 'physics-fidelity',
        'domain-randomization', 'privileged-training', 'curriculum-learning'
      ]),
      ids: mergeIds('sim2real', [
        'wiki/overview/hub-sim2real.md',
        'wiki/overview/hub-physics-fidelity.md',
        'wiki/concepts/sim2real.md',
        'wiki/queries/simulation-physics-fidelity.md',
        'wiki/concepts/domain-randomization.md',
        'wiki/concepts/privileged-training.md',
        'wiki/concepts/curriculum-learning.md',
        'wiki/concepts/system-identification.md',
        'wiki/methods/sim2real-joint-sysid-experiment-design.md',
        'wiki/concepts/sim-vs-real-eval-gap.md',
        'wiki/queries/sim2real-checklist.md'
      ])
    },
    'humanoid-boxing': {
      segments: new Set([
        'boxing', 'combat', 'adversarial', 'sparring', 'punch',
        'robostriker', 'smplolympics', 'urkl'
      ]),
      ids: mergeIds('humanoid-boxing', [
        'wiki/entities/paper-notebook-robostriker.md',
        'wiki/entities/rek.md',
        'wiki/entities/urkl.md',
        'wiki/entities/smplolympics.md',
        'wiki/entities/paper-notebook-towards-motion-turing-test.md',
        'wiki/entities/paper-hrl-stack-41-safefall.md'
      ])
    },
    'bfm': {
      segments: new Set([
        'bfm', 'behavior-foundation', 'ase', 'phc', 'wbt', 'motion-tracking',
        'shadowing', 'whole-body-tracking', 'beyondmimic', 'deepmimic'
      ]),
      ids: mergeIds('bfm', [
        'wiki/overview/hub-wbt.md',
        'wiki/concepts/behavior-foundation-model.md',
        'wiki/concepts/whole-body-tracking-pipeline.md',
        'wiki/methods/deepmimic.md',
        'wiki/methods/amp-reward.md',
        'wiki/queries/humanoid-motion-tracking-method-selection.md'
      ])
    },
    'embodied-eval': {
      segments: new Set([
        'eval', 'evaluation', 'benchmark', 'bench', 'leaderboard', 'judge',
        'ewmbench', 'worldscore', 'robodojo', 'dexbench', 'humantracker',
        'trackerlab'
      ]),
      ids: mergeIds('embodied-eval', [
        'wiki/overview/hub-embodied-eval-benchmark.md',
        'wiki/queries/embodied-eval-benchmark-selection-loop.md',
        'wiki/concepts/sim-vs-real-eval-gap.md',
        'wiki/concepts/simulation-evaluation-infrastructure.md',
        'wiki/entities/rlbench.md',
        'wiki/entities/paper-daily-omni.md',
        'wiki/entities/paper-mmhu.md',
        'wiki/entities/paper-harnesseval-w.md',
        'wiki/entities/paper-worldecho-worldsync.md',
        'wiki/entities/paper-softvtbench.md',
        'wiki/entities/paper-imitator-game.md',
        'wiki/entities/paper-bet4sim2real.md',
        'wiki/entities/paper-reflexvla.md',
        'wiki/entities/paper-robosynchallenge.md',
        'wiki/entities/paper-sa-2606-15032-how-should-world-models-be-evaluated-for-embodie.md',
        'wiki/entities/xpolicylab.md',
        'wiki/concepts/motion-control-policy-evaluation-metrics.md',
        'wiki/tasks/locomotion.md',
        'wiki/concepts/whole-body-tracking-pipeline.md',
        'wiki/queries/humanoid-motion-tracking-method-selection.md'
      ])
    },
    'perceptive-locomotion': {
      segments: new Set([
        'perceptive', 'parkour', 'stair', 'terrain', 'elevation', 'obstacle',
        'vision-locomotion', 'extreme-parkour', 'dreamwaq', 'footstep'
      ]),
      ids: mergeIds('perceptive-locomotion', [
        'wiki/overview/hub-vision-backbone.md',
        'wiki/entities/extreme-parkour.md',
        'wiki/entities/dreamwaq-plus.md',
        'wiki/concepts/footstep-planning.md',
        'wiki/concepts/hierarchical-quadruped-navigation-stack.md'
      ])
    },
    'motion-generation': {
      segments: new Set([
        'motion-generation', 'motion-diffusion', 'mdm', 'text-to-motion',
        'generative-motion', 'human-motion', 'humanml', 'humanml3d',
        'motion-x', 'posescript', 'kimodo', 'physdiff', 'phygile',
        'omnicontrol', 'hy-motion', 'genmo', 'dimos', 'motionbricks',
        'in-betweening'
      ]),
      ids: mergeIds('motion-generation', [
        'wiki/methods/diffusion-motion-generation.md',
        'wiki/comparisons/hy-motion-vs-genmo-vs-kimodo.md',
        'wiki/entities/awesome-text-to-motion-zilize.md',
        'wiki/entities/dataset-bfm-humanml3d.md',
        'wiki/entities/dataset-bfm-kit-ml.md',
        'wiki/entities/dataset-bfm-babel.md',
        'wiki/entities/dataset-bfm-motion-x.md',
        'wiki/entities/dataset-bfm-posescript.md',
        'wiki/entities/kimodo.md',
        'wiki/entities/ardy.md',
        'wiki/entities/gen2humanoid.md',
        'wiki/entities/paper-phygile.md',
        'wiki/entities/paper-dimos-human-scene-motion-synthesis.md',
        'wiki/entities/paper-omg-omni-modal-humanoid-control.md',
        'wiki/entities/paper-gpc-generative-pretrained-controllers.md',
        'wiki/entities/paper-muninn-trajectory-diffusion-acceleration.md',
        'wiki/entities/paper-heracles-humanoid-diffusion.md'
      ])
    },
    'vla': {
      communities: new Set(['community-5']),
      segments: new Set([
        'vla', 'vision-language-action', 'openvla', 'rt-2', 'pi0', 'gr00t',
        'foundation-policy'
      ]),
      ids: mergeIds('vla', [
        'wiki/overview/hub-vla.md',
        'wiki/overview/hub-embodied-foundation-model.md',
        'wiki/overview/hub-vision-backbone.md',
        'wiki/methods/vla.md',
        'wiki/queries/embodied-fm-taxonomy-loop.md'
      ])
    },
    'real2sim': {
      segments: new Set([
        'real2sim', 'real-to-sim', 'gaussian', 'splatting', '3dgs',
        'reconstruction', 'digital-twin', 'nerf', 'simfoundry', 'articraft',
        'aholo', 'gs-playground'
      ]),
      ids: mergeIds('real2sim', [
        'wiki/entities/paper-agentic-real2sim.md',
        'wiki/entities/paper-simfoundry-real2sim-scene-generation.md',
        'wiki/entities/articraft.md',
        'wiki/entities/aholo-viewer.md',
        'wiki/entities/gs-playground.md',
        'wiki/entities/spark-3dgs-renderer.md',
        'wiki/comparisons/spark-vs-aholo-web-3dgs-renderers.md',
        'wiki/concepts/video-as-simulation.md'
      ])
    },
    'icl': {
      segments: new Set([
        'icl', 'in-context', 'one-shot', 'test-time'
      ]),
      ids: mergeIds('icl', [
        'wiki/concepts/foundation-policy.md',
        'wiki/concepts/llm-robotics-control-interfaces.md',
        'wiki/methods/imitation-learning.md',
        'wiki/methods/pi07-policy.md',
        'wiki/queries/embodied-fm-taxonomy-loop.md',
        'wiki/entities/anthropic-embody.md',
        'wiki/entities/paper-behavior-prompting-policy.md',
        'wiki/entities/paper-imitator-game.md',
        'wiki/entities/paper-zero-wam.md',
        'wiki/entities/qwen-robot-manip.md',
        'wiki/entities/skild-s1.md'
      ])
    },
    'wam': {
      segments: new Set([
        'wam', 'world-action', 'world-model', 'worldmodel', 'video-prediction',
        'imagination', 'cosmos', 'dreamer', 'physisforcing', 'dynawm',
        'dreamsteer'
      ]),
      ids: mergeIds('wam', [
        'wiki/overview/hub-embodied-foundation-model.md',
        'wiki/overview/hub-embodied-eval-benchmark.md',
        'wiki/concepts/world-action-models.md',
        'wiki/queries/embodied-eval-benchmark-selection-loop.md',
        'wiki/entities/cosmos-3.md',
        'wiki/entities/lumo-2.md',
        'wiki/entities/dexmal-dw05.md',
        'wiki/entities/xiaomi-robotics-u0.md',
        'wiki/entities/paper-dit4dit-video-action-model.md',
        'wiki/entities/paper-dreamsteer-vla-deployment-steering.md',
        'wiki/entities/paper-dynawm-vla-online-correction.md',
        'wiki/entities/paper-gigaworld-1-policy-evaluation.md',
        'wiki/entities/paper-physisforcing.md'
      ])
    }
  };

  /* 路线展示元信息（emoji + 简称 + 路线锚点 + 导读），与 graph.html chips 顺序一致。 */
  var DEPTH_META = {
    'motion-control': {
      emoji: '🧭',
      label: '主路线-运动控制',
      wikiPath: DEPTH_HUB_IDS['motion-control'],
      description: '运动控制算法工程师成长路线：L−1 全景 → L0–L7 主干与全栈出口。'
    },
    'teleoperation': {
      emoji: '🎮',
      label: '遥操作',
      wikiPath: DEPTH_HUB_IDS.teleoperation,
      description: '人形全身与手指遥操作，采集高质量示范数据并支持实时接管。'
    },
    'torque-motor-design': {
      emoji: '⚙️',
      label: '力矩电机设计',
      wikiPath: DEPTH_HUB_IDS['torque-motor-design'],
      description: '从任务指标到电磁热、FOC 力矩闭环与可验收关节模组。'
    },
    'classical-control': {
      emoji: '📐',
      label: '传统控制',
      wikiPath: DEPTH_HUB_IDS['classical-control'],
      description: 'LIP/ZMP → Centroidal → MPC → TSID/WBC 的 model-based 主干。'
    },
    'humanoid-hardware-design': {
      emoji: '🛠️',
      label: '整机硬件',
      wikiPath: DEPTH_HUB_IDS['humanoid-hardware-design'],
      description: '指标预算 → 机械 → 电气 → 通信 → 整机验收的硬件交付链。'
    },
    'safe-control': {
      emoji: '🛡️',
      label: '安全控制',
      wikiPath: DEPTH_HUB_IDS['safe-control'],
      description: 'CLF / CBF / Safe RL：把可证明安全约束接进控制与学习环。'
    },
    'contact-manipulation': {
      emoji: '🤏',
      label: '接触操作',
      wikiPath: DEPTH_HUB_IDS['contact-manipulation'],
      description: '装配、插拔、双臂协同等接触丰富操作与力控闭环。'
    },
    'navigation': {
      emoji: '🗺️',
      label: '导航',
      wikiPath: DEPTH_HUB_IDS.navigation,
      description: 'SLAM → Nav2 → VLN → 导航 VLA：定位、规划与语义导航。'
    },
    'imitation-learning': {
      emoji: '🎓',
      label: '模仿学习',
      wikiPath: DEPTH_HUB_IDS['imitation-learning'],
      description: '从人类演示学习技能：BC / ACT / Diffusion Policy 与数据管线。'
    },
    'rl-locomotion': {
      emoji: '🚶',
      label: 'RL 运动控制',
      wikiPath: DEPTH_HUB_IDS['rl-locomotion'],
      description: '用强化学习驱动人形 locomotion 与多地形步态。'
    },
    'loco-manipulation': {
      emoji: '🤖',
      label: 'Loco-Manip',
      wikiPath: DEPTH_HUB_IDS['loco-manipulation'],
      description: '边走边动手的移动操作：全身协调与接触任务。'
    },
    'humanoid-soccer': {
      emoji: '⚽',
      label: '人形足球',
      wikiPath: DEPTH_HUB_IDS['humanoid-soccer'],
      description: '全向行走 → 感知踢球 → 多机战术的整场比赛能力。'
    },
    'motion-retargeting': {
      emoji: '🤸',
      label: '动作重定向',
      wikiPath: DEPTH_HUB_IDS['motion-retargeting'],
      description: '把人体/动物参考动作映射到异构机器人可执行轨迹。'
    },
    'humanoid-swarm-performance': {
      emoji: '🕺',
      label: '人形群控展演',
      wikiPath: DEPTH_HUB_IDS['humanoid-swarm-performance'],
      description: '群舞同步、编队走位与群体特技的多机协同展演。'
    },
    'sim2real': {
      emoji: '🔁',
      label: 'Sim2Real',
      wikiPath: DEPTH_HUB_IDS.sim2real,
      description: '域差画像 → 执行器对齐 → 鲁棒训练 → 真机部署。'
    },
    'humanoid-boxing': {
      emoji: '🥊',
      label: '人形拳击',
      wikiPath: DEPTH_HUB_IDS['humanoid-boxing'],
      description: '动作跟踪 → 潜空间技能 → 对抗自博弈的擂台对打。'
    },
    'icl': {
      emoji: '🧩',
      label: 'ICL',
      wikiPath: DEPTH_HUB_IDS.icl,
      description: '具身上下文学习：读完一条示范就会做新任务，且不更新权重。'
    },
    'bfm': {
      emoji: '🧠',
      label: 'BFM',
      wikiPath: DEPTH_HUB_IDS.bfm,
      description: '人形行为基础模型：一个 checkpoint 控住全身协调。'
    },
    'embodied-eval': {
      emoji: '📊',
      label: '具身测评',
      wikiPath: DEPTH_HUB_IDS['embodied-eval'],
      description: '认知 → 世界模型 → 策略成功率 → 运控指标 → sim↔real 校准的分层评测验收。'
    },
    'perceptive-locomotion': {
      emoji: '👁️',
      label: '感知越障',
      wikiPath: DEPTH_HUB_IDS['perceptive-locomotion'],
      description: '看着地形上楼梯、跨障碍、跑酷的感知式移动。'
    },
    'motion-generation': {
      emoji: '✨',
      label: '动作生成',
      wikiPath: DEPTH_HUB_IDS['motion-generation'],
      description: '文本/多模态条件的人体与人形动作生成。'
    },
    'vla': {
      emoji: '👀',
      label: 'VLA',
      wikiPath: DEPTH_HUB_IDS.vla,
      description: '视觉-语言-动作统一建模：听懂指令并完成操作任务。'
    },
    'real2sim': {
      emoji: '🌍',
      label: 'Real2Sim',
      wikiPath: DEPTH_HUB_IDS.real2sim,
      description: '把真实世界压成可训练/可评测的仿真资产与场景孪生。'
    },
    'wam': {
      emoji: '🔮',
      label: 'WAM',
      wikiPath: DEPTH_HUB_IDS.wam,
      description: '世界–动作模型：出动作前显式预知世界如何变化。'
    }
  };

  function nodeSegments(node) {
    if (node && node._segs) return node._segs;
    var base = ((node && node.id) || '').toLowerCase().replace(/\.md$/, '');
    var segs = new Set(base.split(/[/._-]/).filter(Boolean));
    if (node) node._segs = segs;
    return segs;
  }

  /* 把 id 中的 / . _ - 归一成 '-'，供多 token segment 做子串命中。 */
  function nodePathKey(node) {
    if (node && node._pathKey) return node._pathKey;
    var key = ((node && node.id) || '').toLowerCase().replace(/\.md$/, '').replace(/[/._]+/g, '-');
    if (node) node._pathKey = key;
    return key;
  }

  /* 单 token → 精确词元；含分隔符 → 归一路径子串（兼容 loco-manip / loco-manipulation）。 */
  function segmentHits(node, segs, segment) {
    if (!segment) return false;
    if (/[/._-]/.test(segment)) {
      var needle = String(segment).toLowerCase().replace(/[/._]+/g, '-');
      return needle.length > 0 && nodePathKey(node).indexOf(needle) !== -1;
    }
    return segs.has(segment);
  }

  /* 判定单个节点是否命中某纵深（depthKey 为 'all' 时恒真）。 */
  function matches(node, depthKey) {
    if (depthKey === 'all') return true;
    var cfg = DEPTH_FILTERS[depthKey];
    if (!cfg) return true;
    var segs = nodeSegments(node);
    if (cfg.excludeSegments) {
      for (var ex of cfg.excludeSegments) {
        if (segmentHits(node, segs, ex)) return false;
      }
    }
    if (cfg.ids && cfg.ids.has(node.id)) return true;
    if (cfg.communities && node.community && cfg.communities.has(node.community)) return true;
    if (cfg.segments) {
      for (var seg of cfg.segments) {
        if (segmentHits(node, segs, seg)) return true;
      }
    }
    return false;
  }

  /* 返回节点命中的全部纵深 key 列表（不含 'all'；按 DEPTH_ORDER）。 */
  function depthsForNode(node) {
    var out = [];
    for (var i = 0; i < DEPTH_ORDER.length; i++) {
      var key = DEPTH_ORDER[i];
      if (DEPTH_FILTERS[key] && matches(node, key)) out.push(key);
    }
    return out;
  }

  /* 某纵深的路线锚点路径；无则 null。 */
  function depthHubPath(depthKey) {
    return DEPTH_HUB_IDS[depthKey] || null;
  }

  /* 节点是否为任一纵深（或指定纵深）的路线锚点。 */
  function isDepthHub(node, depthKey) {
    if (!node || !node.id) return false;
    if (depthKey && depthKey !== 'all') {
      return DEPTH_HUB_IDS[depthKey] === node.id;
    }
    for (var k in DEPTH_HUB_IDS) {
      if (DEPTH_HUB_IDS[k] === node.id) return true;
    }
    return false;
  }

  global.RNDepthFilters = {
    DEPTH_ORDER: DEPTH_ORDER,
    DEPTH_FILTERS: DEPTH_FILTERS,
    DEPTH_META: DEPTH_META,
    DEPTH_HUB_IDS: DEPTH_HUB_IDS,
    nodeSegments: nodeSegments,
    nodePathKey: nodePathKey,
    segmentHits: segmentHits,
    matches: matches,
    depthsForNode: depthsForNode,
    depthHubPath: depthHubPath,
    isDepthHub: isDepthHub,
    // 兼容旧名
    TOPIC_FILTERS: DEPTH_FILTERS,
    TOPIC_META: DEPTH_META,
    TOPIC_HUB_IDS: DEPTH_HUB_IDS,
    topicsForNode: depthsForNode,
    topicHubPath: depthHubPath,
    isTopicHub: isDepthHub
  };
  global.RNTopicFilters = global.RNDepthFilters;
})(typeof window !== 'undefined' ? window : this);
