# Entities 关联出边：100 节点修复试点

日期：2026-11-18。基线提交：`f5eaa2cefba6c2be17ab8923e5375b8be0902151`（`main`）。

## 目标与选样

针对 `scripts/lint_wiki.py` 的 `entity_missing_outgoing_legacy`：实体正文至少应有 2 条指向 `wiki/methods/` 或 `wiki/tasks/` 的 Markdown 内链。

从实际触发该规则且在历史基线中的节点里，按以下 10 个主题各选 10 个，共 100 个唯一节点；包含 33 个 `paper-*` 路径节点（包括 A-LOAM 这类沿用该路径的实现页），以及 67 个工具、模型、机器人、数据集、协议栈或专利实体。这是覆盖不同修复场景的目的性试点，**不是随机样本，也不据此推断全库修复质量**。

- 覆盖 0 出边（67 页）和 1 出边（33 页）两种缺项。
- 覆盖算法实现、物理平台、参考动作、感知/导航前端、数据与硬件接口等不同角色。
- 每条新增链接说明方法背景、具体下游用途或对照差异；潜在用途不表述成项目已实现或验证的能力。
- 优先复用原有「关联页面」或「与其他页面/系统的关系」段落；正文只增加解释性内链，并更新 `updated`。

## 试点结果

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| 样本中缺出边的节点 | 100 | 0 |
| 样本中拥有 2 个不同且存在的 methods/tasks 目标 | 0 | 100 |
| 样本合格出边总数 | 33 | 200 |
| 全库实际历史出边缺项 | 3,379 | 3,279 |
| 历史出边豁免列表条目 | 3,381 | 3,281 |

新增 **167 条**出边，涉及 **48 个不同目标**。豁免列表仅移除这 100 个已修复路径，其余条目及缩写豁免不变。列表比实际缺项多 2 条是基线已有差异，本批不顺带清理。

状态、来源、原文通读声明和健康度评分规则未修改；逐页比较确认 **100 页健康度与修复前相同**（1/3：49 页；2/3：34 页；3/3：17 页）。本次仅修复导航关联，未把存量原文阅读、骨架或开源状态缺项声明为完成。

## 节点清单

数量按正文解析，并额外按目标路径去重；不把 frontmatter `related`、重复链接或不存在的路径充作出边。表中只列本次新增的目标，已有出边保留。

| 主题 | 节点 | 修复前 | 修复后 | 新增 methods/tasks 目标 |
|---|---|---:|---:|---|
| 强化学习与训练 | [stable-baselines3](../../wiki/entities/stable-baselines3.md) | 0 | 2 | [ppo](../../wiki/methods/ppo.md)、[sac](../../wiki/methods/sac.md) |
| 强化学习与训练 | [cleanrl](../../wiki/entities/cleanrl.md) | 0 | 2 | [ppo](../../wiki/methods/ppo.md)、[gae](../../wiki/methods/gae.md) |
| 强化学习与训练 | [rl-games](../../wiki/entities/rl-games.md) | 0 | 2 | [ppo](../../wiki/methods/ppo.md)、[gae](../../wiki/methods/gae.md) |
| 强化学习与训练 | [skrl](../../wiki/entities/skrl.md) | 0 | 2 | [ppo](../../wiki/methods/ppo.md)、[sac](../../wiki/methods/sac.md) |
| 强化学习与训练 | [gymnasium](../../wiki/entities/gymnasium.md) | 1 | 2 | [ppo](../../wiki/methods/ppo.md) |
| 强化学习与训练 | [paper-alphago-zero](../../wiki/entities/paper-alphago-zero.md) | 0 | 2 | [reinforcement-learning](../../wiki/methods/reinforcement-learning.md)、[model-based-rl](../../wiki/methods/model-based-rl.md) |
| 强化学习与训练 | [paper-alphazero](../../wiki/entities/paper-alphazero.md) | 1 | 2 | [model-based-rl](../../wiki/methods/model-based-rl.md) |
| 强化学习与训练 | [paper-alphastar](../../wiki/entities/paper-alphastar.md) | 1 | 2 | [marl](../../wiki/methods/marl.md) |
| 强化学习与训练 | [paper-bfm-30-diayn](../../wiki/entities/paper-bfm-30-diayn.md) | 1 | 2 | [sac](../../wiki/methods/sac.md) |
| 强化学习与训练 | [paper-daydreamer-world-models-real-robots](../../wiki/entities/paper-daydreamer-world-models-real-robots.md) | 1 | 2 | [model-based-rl](../../wiki/methods/model-based-rl.md) |
| 仿真与迁移验证 | [gazebo-sim](../../wiki/entities/gazebo-sim.md) | 0 | 2 | [pid-control](../../wiki/methods/pid-control.md)、[manipulation](../../wiki/tasks/manipulation.md) |
| 仿真与迁移验证 | [genesis-sim](../../wiki/entities/genesis-sim.md) | 0 | 2 | [reinforcement-learning](../../wiki/methods/reinforcement-learning.md)、[manipulation](../../wiki/tasks/manipulation.md) |
| 仿真与迁移验证 | [sapien](../../wiki/entities/sapien.md) | 0 | 2 | [manipulation](../../wiki/tasks/manipulation.md)、[grasp-pose-estimation](../../wiki/methods/grasp-pose-estimation.md) |
| 仿真与迁移验证 | [raisim](../../wiki/entities/raisim.md) | 0 | 2 | [reinforcement-learning](../../wiki/methods/reinforcement-learning.md)、[locomotion](../../wiki/tasks/locomotion.md) |
| 仿真与迁移验证 | [webots](../../wiki/entities/webots.md) | 0 | 2 | [pid-control](../../wiki/methods/pid-control.md)、[a-star](../../wiki/methods/a-star.md) |
| 仿真与迁移验证 | [airsim](../../wiki/entities/airsim.md) | 0 | 2 | [trajectory-optimization](../../wiki/methods/trajectory-optimization.md)、[object-detection](../../wiki/methods/object-detection.md) |
| 仿真与迁移验证 | [carla](../../wiki/entities/carla.md) | 0 | 2 | [object-detection](../../wiki/methods/object-detection.md)、[reinforcement-learning](../../wiki/methods/reinforcement-learning.md) |
| 仿真与迁移验证 | [coppeliasim](../../wiki/entities/coppeliasim.md) | 0 | 2 | [manipulation](../../wiki/tasks/manipulation.md)、[behavior-cloning](../../wiki/methods/behavior-cloning.md) |
| 仿真与迁移验证 | [paper-acosta-validating-simulators-real-world-impacts](../../wiki/entities/paper-acosta-validating-simulators-real-world-impacts.md) | 0 | 2 | [sim2real-joint-sysid-experiment-design](../../wiki/methods/sim2real-joint-sysid-experiment-design.md)、[locomotion](../../wiki/tasks/locomotion.md) |
| 仿真与迁移验证 | [paper-bundled-contact-gradients](../../wiki/entities/paper-bundled-contact-gradients.md) | 0 | 2 | [trajectory-optimization](../../wiki/methods/trajectory-optimization.md)、[add](../../wiki/methods/add.md) |
| 足式运动与感知控制 | [anymal](../../wiki/entities/anymal.md) | 1 | 2 | [inverse-dynamics-control](../../wiki/methods/inverse-dynamics-control.md) |
| 足式运动与感知控制 | [berkeley-humanoid-lite](../../wiki/entities/berkeley-humanoid-lite.md) | 1 | 2 | [humanoid-locomotion](../../wiki/tasks/humanoid-locomotion.md) |
| 足式运动与感知控制 | [champ-quadruped-controller](../../wiki/entities/champ-quadruped-controller.md) | 1 | 2 | [pid-control](../../wiki/methods/pid-control.md) |
| 足式运动与感知控制 | [deeprobotics-rl-training](../../wiki/entities/deeprobotics-rl-training.md) | 1 | 2 | [amp-reward](../../wiki/methods/amp-reward.md) |
| 足式运动与感知控制 | [wheel-legged-genesis](../../wiki/entities/wheel-legged-genesis.md) | 1 | 2 | [ppo](../../wiki/methods/ppo.md) |
| 足式运动与感知控制 | [paper-anymal-walk-minutes-parallel-drl](../../wiki/entities/paper-anymal-walk-minutes-parallel-drl.md) | 0 | 2 | [ppo](../../wiki/methods/ppo.md)、[locomotion](../../wiki/tasks/locomotion.md) |
| 足式运动与感知控制 | [paper-cassie-biped-versatile-locomotion-rl](../../wiki/entities/paper-cassie-biped-versatile-locomotion-rl.md) | 0 | 2 | [reinforcement-learning](../../wiki/methods/reinforcement-learning.md)、[locomotion](../../wiki/tasks/locomotion.md) |
| 足式运动与感知控制 | [paper-concurrent-policy-estimator-locomotion](../../wiki/entities/paper-concurrent-policy-estimator-locomotion.md) | 0 | 2 | [reinforcement-learning](../../wiki/methods/reinforcement-learning.md)、[locomotion](../../wiki/tasks/locomotion.md) |
| 足式运动与感知控制 | [paper-barkour-quadruped-agility-benchmark](../../wiki/entities/paper-barkour-quadruped-agility-benchmark.md) | 1 | 2 | [ppo](../../wiki/methods/ppo.md) |
| 足式运动与感知控制 | [paper-deep-whole-body-parkour](../../wiki/entities/paper-deep-whole-body-parkour.md) | 1 | 2 | [beyondmimic](../../wiki/methods/beyondmimic.md) |
| 动作生成与重定向 | [retargetingkit](../../wiki/entities/retargetingkit.md) | 0 | 2 | [teleoperation](../../wiki/tasks/teleoperation.md)、[motion-retargeting-gmr](../../wiki/methods/motion-retargeting-gmr.md) |
| 动作生成与重定向 | [blinkmimic](../../wiki/entities/blinkmimic.md) | 0 | 2 | [deepmimic](../../wiki/methods/deepmimic.md)、[muon](../../wiki/methods/muon.md) |
| 动作生成与重定向 | [cartwheel-control](../../wiki/entities/cartwheel-control.md) | 1 | 2 | [deepmimic](../../wiki/methods/deepmimic.md) |
| 动作生成与重定向 | [wham-world-human-motion](../../wiki/entities/wham-world-human-motion.md) | 1 | 2 | [deepmimic](../../wiki/methods/deepmimic.md) |
| 动作生成与重定向 | [rokoko](../../wiki/entities/rokoko.md) | 0 | 2 | [motion-retargeting-gmr](../../wiki/methods/motion-retargeting-gmr.md)、[imitation-learning](../../wiki/methods/imitation-learning.md) |
| 动作生成与重定向 | [paper-bfm-17-maskedmimic](../../wiki/entities/paper-bfm-17-maskedmimic.md) | 0 | 2 | [imitation-learning](../../wiki/methods/imitation-learning.md)、[deepmimic](../../wiki/methods/deepmimic.md) |
| 动作生成与重定向 | [paper-bfm-02-metamotivo](../../wiki/entities/paper-bfm-02-metamotivo.md) | 0 | 2 | [reinforcement-learning](../../wiki/methods/reinforcement-learning.md)、[hierarchical-reinforcement-learning](../../wiki/methods/hierarchical-reinforcement-learning.md) |
| 动作生成与重定向 | [paper-bfm-19-calm](../../wiki/entities/paper-bfm-19-calm.md) | 0 | 2 | [ase](../../wiki/methods/ase.md)、[hierarchical-reinforcement-learning](../../wiki/methods/hierarchical-reinforcement-learning.md) |
| 动作生成与重定向 | [paper-bfm-15-intermimic](../../wiki/entities/paper-bfm-15-intermimic.md) | 0 | 2 | [imitation-learning](../../wiki/methods/imitation-learning.md)、[loco-manipulation](../../wiki/tasks/loco-manipulation.md) |
| 动作生成与重定向 | [paper-a-mdm-interactive-character-control](../../wiki/entities/paper-a-mdm-interactive-character-control.md) | 1 | 2 | [hierarchical-reinforcement-learning](../../wiki/methods/hierarchical-reinforcement-learning.md) |
| 优化与模型控制 | [controlit](../../wiki/entities/controlit.md) | 0 | 2 | [inverse-dynamics-control](../../wiki/methods/inverse-dynamics-control.md)、[manipulation](../../wiki/tasks/manipulation.md) |
| 优化与模型控制 | [control-toolbox-ct](../../wiki/entities/control-toolbox-ct.md) | 1 | 2 | [trajectory-optimization](../../wiki/methods/trajectory-optimization.md) |
| 优化与模型控制 | [pink-ik](../../wiki/entities/pink-ik.md) | 0 | 2 | [motion-retargeting-gmr](../../wiki/methods/motion-retargeting-gmr.md)、[manipulation](../../wiki/tasks/manipulation.md) |
| 优化与模型控制 | [proxsuite](../../wiki/entities/proxsuite.md) | 0 | 2 | [model-predictive-control](../../wiki/methods/model-predictive-control.md)、[trajectory-optimization](../../wiki/methods/trajectory-optimization.md) |
| 优化与模型控制 | [ruckig](../../wiki/entities/ruckig.md) | 1 | 2 | [time-optimal-path-parameterization](../../wiki/methods/time-optimal-path-parameterization.md) |
| 优化与模型控制 | [robotics-toolbox-python](../../wiki/entities/robotics-toolbox-python.md) | 0 | 2 | [levenberg-marquardt](../../wiki/methods/levenberg-marquardt.md)、[trajectory-optimization](../../wiki/methods/trajectory-optimization.md) |
| 优化与模型控制 | [ceres-solver](../../wiki/entities/ceres-solver.md) | 0 | 2 | [gauss-newton](../../wiki/methods/gauss-newton.md)、[levenberg-marquardt](../../wiki/methods/levenberg-marquardt.md) |
| 优化与模型控制 | [gtsam](../../wiki/entities/gtsam.md) | 0 | 2 | [gauss-newton](../../wiki/methods/gauss-newton.md)、[levenberg-marquardt](../../wiki/methods/levenberg-marquardt.md) |
| 优化与模型控制 | [paper-cartesian-impedance-controller](../../wiki/entities/paper-cartesian-impedance-controller.md) | 0 | 2 | [manipulation](../../wiki/tasks/manipulation.md)、[direct-force-feedback-control](../../wiki/methods/direct-force-feedback-control.md) |
| 优化与模型控制 | [paper-bledt-rpc-thesis](../../wiki/entities/paper-bledt-rpc-thesis.md) | 0 | 2 | [model-predictive-control](../../wiki/methods/model-predictive-control.md)、[locomotion](../../wiki/tasks/locomotion.md) |
| 定位建图与导航 | [cartographer](../../wiki/entities/cartographer.md) | 0 | 2 | [lidar-odometry-fusion](../../wiki/methods/lidar-odometry-fusion.md)、[a-star](../../wiki/methods/a-star.md) |
| 定位建图与导航 | [rtabmap](../../wiki/entities/rtabmap.md) | 0 | 2 | [a-star](../../wiki/methods/a-star.md)、[dwa](../../wiki/methods/dwa.md) |
| 定位建图与导航 | [slam-toolbox](../../wiki/entities/slam-toolbox.md) | 0 | 2 | [a-star](../../wiki/methods/a-star.md)、[dwa](../../wiki/methods/dwa.md) |
| 定位建图与导航 | [vins-fusion](../../wiki/entities/vins-fusion.md) | 0 | 2 | [gauss-newton](../../wiki/methods/gauss-newton.md)、[levenberg-marquardt](../../wiki/methods/levenberg-marquardt.md) |
| 定位建图与导航 | [fast-lio](../../wiki/entities/fast-lio.md) | 0 | 2 | [lidar-odometry-fusion](../../wiki/methods/lidar-odometry-fusion.md)、[autonomous-exploration](../../wiki/tasks/autonomous-exploration.md) |
| 定位建图与导航 | [autonomy-stack-go2](../../wiki/entities/autonomy-stack-go2.md) | 0 | 2 | [lidar-odometry-fusion](../../wiki/methods/lidar-odometry-fusion.md)、[autonomous-exploration](../../wiki/tasks/autonomous-exploration.md) |
| 定位建图与导航 | [arena-rosnav](../../wiki/entities/arena-rosnav.md) | 0 | 2 | [reinforcement-learning](../../wiki/methods/reinforcement-learning.md)、[dwa](../../wiki/methods/dwa.md) |
| 定位建图与导航 | [paper-a-loam](../../wiki/entities/paper-a-loam.md) | 0 | 2 | [lidar-odometry-fusion](../../wiki/methods/lidar-odometry-fusion.md)、[levenberg-marquardt](../../wiki/methods/levenberg-marquardt.md) |
| 定位建图与导航 | [paper-cuvslam](../../wiki/entities/paper-cuvslam.md) | 0 | 2 | [stereo-matching-foundation-models](../../wiki/methods/stereo-matching-foundation-models.md)、[autonomous-exploration](../../wiki/tasks/autonomous-exploration.md) |
| 定位建图与导航 | [paper-compass-1-navigation](../../wiki/entities/paper-compass-1-navigation.md) | 0 | 2 | [vision-language-navigation](../../wiki/tasks/vision-language-navigation.md)、[zero-shot-object-navigation](../../wiki/tasks/zero-shot-object-navigation.md) |
| 视觉感知与多模态 | [detr](../../wiki/entities/detr.md) | 1 | 2 | [rcnn-family](../../wiki/methods/rcnn-family.md) |
| 视觉感知与多模态 | [retinanet](../../wiki/entities/retinanet.md) | 1 | 2 | [object-detection](../../wiki/methods/object-detection.md) |
| 视觉感知与多模态 | [segformer](../../wiki/entities/segformer.md) | 0 | 2 | [fcn-semantic-segmentation](../../wiki/methods/fcn-semantic-segmentation.md)、[unet](../../wiki/methods/unet.md) |
| 视觉感知与多模态 | [setr](../../wiki/entities/setr.md) | 0 | 2 | [fcn-semantic-segmentation](../../wiki/methods/fcn-semantic-segmentation.md)、[unet](../../wiki/methods/unet.md) |
| 视觉感知与多模态 | [deformable-detr](../../wiki/entities/deformable-detr.md) | 0 | 2 | [object-detection](../../wiki/methods/object-detection.md)、[rcnn-family](../../wiki/methods/rcnn-family.md) |
| 视觉感知与多模态 | [paper-clip](../../wiki/entities/paper-clip.md) | 1 | 2 | [zero-shot-object-navigation](../../wiki/tasks/zero-shot-object-navigation.md) |
| 视觉感知与多模态 | [paper-blip2](../../wiki/entities/paper-blip2.md) | 1 | 2 | [vla](../../wiki/methods/vla.md) |
| 视觉感知与多模态 | [paper-cambrian-1](../../wiki/entities/paper-cambrian-1.md) | 0 | 2 | [vision-language-navigation](../../wiki/tasks/vision-language-navigation.md)、[vla](../../wiki/methods/vla.md) |
| 视觉感知与多模态 | [paper-any3r](../../wiki/entities/paper-any3r.md) | 0 | 2 | [stereo-matching-foundation-models](../../wiki/methods/stereo-matching-foundation-models.md)、[autonomous-exploration](../../wiki/tasks/autonomous-exploration.md) |
| 视觉感知与多模态 | [paper-d4rt](../../wiki/entities/paper-d4rt.md) | 1 | 2 | [stereo-matching-foundation-models](../../wiki/methods/stereo-matching-foundation-models.md) |
| 操作与触觉学习 | [behavior-1k](../../wiki/entities/behavior-1k.md) | 0 | 2 | [manipulation](../../wiki/tasks/manipulation.md)、[hierarchical-reinforcement-learning](../../wiki/methods/hierarchical-reinforcement-learning.md) |
| 操作与触觉学习 | [franka-research-3](../../wiki/entities/franka-research-3.md) | 1 | 2 | [direct-force-feedback-control](../../wiki/methods/direct-force-feedback-control.md) |
| 操作与触觉学习 | [yale-openhand](../../wiki/entities/yale-openhand.md) | 1 | 2 | [grasp-pose-estimation](../../wiki/methods/grasp-pose-estimation.md) |
| 操作与触觉学习 | [unitree-lerobot](../../wiki/entities/unitree-lerobot.md) | 1 | 2 | [behavior-cloning](../../wiki/methods/behavior-cloning.md) |
| 操作与触觉学习 | [xr-teleoperate](../../wiki/entities/xr-teleoperate.md) | 1 | 2 | [imitation-learning](../../wiki/methods/imitation-learning.md) |
| 操作与触觉学习 | [paper-deep-visuomotor-policies](../../wiki/entities/paper-deep-visuomotor-policies.md) | 0 | 2 | [trajectory-optimization](../../wiki/methods/trajectory-optimization.md)、[manipulation](../../wiki/tasks/manipulation.md) |
| 操作与触觉学习 | [paper-cogact](../../wiki/entities/paper-cogact.md) | 1 | 2 | [diffusion-policy](../../wiki/methods/diffusion-policy.md) |
| 操作与触觉学习 | [paper-atomicvla](../../wiki/entities/paper-atomicvla.md) | 1 | 2 | [hierarchical-reinforcement-learning](../../wiki/methods/hierarchical-reinforcement-learning.md) |
| 操作与触觉学习 | [paper-anytouch](../../wiki/entities/paper-anytouch.md) | 0 | 2 | [manipulation](../../wiki/tasks/manipulation.md)、[tactile-impedance-control](../../wiki/methods/tactile-impedance-control.md) |
| 操作与触觉学习 | [paper-anytouch2](../../wiki/entities/paper-anytouch2.md) | 0 | 2 | [manipulation](../../wiki/tasks/manipulation.md)、[tactile-impedance-control](../../wiki/methods/tactile-impedance-control.md) |
| 数据集与评测资产 | [roboset-dataset](../../wiki/entities/roboset-dataset.md) | 1 | 2 | [behavior-cloning](../../wiki/methods/behavior-cloning.md) |
| 数据集与评测资产 | [bones-seed](../../wiki/entities/bones-seed.md) | 0 | 2 | [motion-retargeting-gmr](../../wiki/methods/motion-retargeting-gmr.md)、[imitation-learning](../../wiki/methods/imitation-learning.md) |
| 数据集与评测资产 | [gen-human-whole-body-dataset](../../wiki/entities/gen-human-whole-body-dataset.md) | 0 | 2 | [motion-retargeting-gmr](../../wiki/methods/motion-retargeting-gmr.md)、[loco-manipulation](../../wiki/tasks/loco-manipulation.md) |
| 数据集与评测资产 | [sharpadex](../../wiki/entities/sharpadex.md) | 0 | 2 | [manipulation](../../wiki/tasks/manipulation.md)、[tactile-impedance-control](../../wiki/methods/tactile-impedance-control.md) |
| 数据集与评测资产 | [touchscale](../../wiki/entities/touchscale.md) | 0 | 2 | [imitation-learning](../../wiki/methods/imitation-learning.md)、[tactile-impedance-control](../../wiki/methods/tactile-impedance-control.md) |
| 数据集与评测资产 | [euroc-mav-datasets](../../wiki/entities/euroc-mav-datasets.md) | 0 | 2 | [stereo-matching-foundation-models](../../wiki/methods/stereo-matching-foundation-models.md)、[gauss-newton](../../wiki/methods/gauss-newton.md) |
| 数据集与评测资产 | [eth3d-stereo-benchmark](../../wiki/entities/eth3d-stereo-benchmark.md) | 1 | 2 | [visual-servoing](../../wiki/methods/visual-servoing.md) |
| 数据集与评测资产 | [urdf-files-dataset](../../wiki/entities/urdf-files-dataset.md) | 0 | 2 | [motion-retargeting-gmr](../../wiki/methods/motion-retargeting-gmr.md)、[inverse-dynamics-control](../../wiki/methods/inverse-dynamics-control.md) |
| 数据集与评测资产 | [grail-locomanipulation-dataset](../../wiki/entities/grail-locomanipulation-dataset.md) | 0 | 2 | [loco-manipulation](../../wiki/tasks/loco-manipulation.md)、[imitation-learning](../../wiki/methods/imitation-learning.md) |
| 数据集与评测资产 | [dataset-bfm-humanoid-x](../../wiki/entities/dataset-bfm-humanoid-x.md) | 1 | 2 | [imitation-learning](../../wiki/methods/imitation-learning.md) |
| 执行器与硬件接口 | [simplefoc](../../wiki/entities/simplefoc.md) | 0 | 2 | [pid-control](../../wiki/methods/pid-control.md)、[joint-actuator-parameter-identification](../../wiki/methods/joint-actuator-parameter-identification.md) |
| 执行器与硬件接口 | [vesc](../../wiki/entities/vesc.md) | 0 | 2 | [pid-control](../../wiki/methods/pid-control.md)、[joint-actuator-parameter-identification](../../wiki/methods/joint-actuator-parameter-identification.md) |
| 执行器与硬件接口 | [dynamixel-sdk](../../wiki/entities/dynamixel-sdk.md) | 1 | 2 | [teleoperation](../../wiki/tasks/teleoperation.md) |
| 执行器与硬件接口 | [ros2-control](../../wiki/entities/ros2-control.md) | 0 | 2 | [pid-control](../../wiki/methods/pid-control.md)、[trajectory-optimization](../../wiki/methods/trajectory-optimization.md) |
| 执行器与硬件接口 | [soem](../../wiki/entities/soem.md) | 0 | 2 | [pid-control](../../wiki/methods/pid-control.md)、[trajectory-optimization](../../wiki/methods/trajectory-optimization.md) |
| 执行器与硬件接口 | [beckhoff-ssc](../../wiki/entities/beckhoff-ssc.md) | 0 | 2 | [pid-control](../../wiki/methods/pid-control.md)、[mit-mode-csp-emulation-tuning](../../wiki/methods/mit-mode-csp-emulation-tuning.md) |
| 执行器与硬件接口 | [canfestival](../../wiki/entities/canfestival.md) | 0 | 2 | [pid-control](../../wiki/methods/pid-control.md)、[mit-mode-csp-emulation-tuning](../../wiki/methods/mit-mode-csp-emulation-tuning.md) |
| 执行器与硬件接口 | [tinymovr](../../wiki/entities/tinymovr.md) | 0 | 2 | [pid-control](../../wiki/methods/pid-control.md)、[joint-actuator-parameter-identification](../../wiki/methods/joint-actuator-parameter-identification.md) |
| 执行器与硬件接口 | [patent-xpeng-cn119262120a-ankle-cross-dual-pushrod](../../wiki/entities/patent-xpeng-cn119262120a-ankle-cross-dual-pushrod.md) | 1 | 2 | [inverse-dynamics-control](../../wiki/methods/inverse-dynamics-control.md) |
| 执行器与硬件接口 | [paper-3d-printed-open-source-actuators-legged](../../wiki/entities/paper-3d-printed-open-source-actuators-legged.md) | 0 | 2 | [joint-actuator-parameter-identification](../../wiki/methods/joint-actuator-parameter-identification.md)、[locomotion](../../wiki/tasks/locomotion.md) |

## 验证口径

针对清单逐页验证：当前规则通过；目标均存在且各页恰有 2 个不同目标；移除的基线集合与清单完全相同；元数据除 `updated` 外一致；健康度不变。用基线原文回放 `_check_entity_page`，100 页均能重新触发缺出边规则，作为负对照。

可从本文件直接提取样本，复跑实际出边规则及负对照（仓库根目录执行，先安装 `requirements-ci-lite.txt`）：

```python
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import lint_wiki as lint

base = "f5eaa2cefba6c2be17ab8923e5375b8be0902151"
report = Path("docs/checklists/lint-outgoing-pilot-100.md").read_text()
paths = re.findall(r"\]\(\.\./\.\./(wiki/entities/[^)]+)\)", report)
assert len(paths) == len(set(paths)) == 100
for rel in paths:
    page = (lint.REPO_ROOT / rel).resolve()
    content = page.read_text()
    targets = {
        t for t in lint.extract_internal_links(content, page)
        if t.is_relative_to(lint.REPO_ROOT / "wiki/methods")
        or t.is_relative_to(lint.REPO_ROOT / "wiki/tasks")
    }
    assert len(targets) == 2 and all(t.is_file() for t in targets), rel
    old = subprocess.check_output(["git", "show", f"{base}:{rel}"], text=True)
    for text, should_fail in [(content, False), (old, True)]:
        result = {"entity_missing_outgoing": []}
        lint._check_entity_page(page, Path(rel), text, result)
        assert bool(result["entity_missing_outgoing"]) == should_fail, rel
print("100/100 当前通过，100/100 基线负对照可检出")
```

完整门禁与节点健康度表见本批 PR；派生图谱、导出 JSON 和 lint 报告只在本地/CI 生成，不随 PR 入库。

## 本地检查结果

- `make ci-preflight`：通过。Wiki lint 0 个阻塞错误；搜索回归 40/40；导出一致性 14/14；Mermaid 12.3.0 校验 8,189 个文件中的 7,957 张图，0 错误；图谱孤儿节点为 0。
- 相关 pytest：`test_lint_legacy_baseline.py`、`test_lint_wiki_crosslinks.py`、`test_report_health.py`、`test_generate_link_graph_health.py`，45 passed。定向运行使用 `-o addopts=`，不对局部测试套用全库覆盖率门槛。
- 无头 Chromium 抽查 Stable-Baselines3（0→2）、ANYmal（1→2）、CanFestival（硬件接口）：关联项含预期目标，站内跳转可打开对应方法详情页；截图只存本地忽略目录。
- `git diff --check` 与 100 页健康度表核对通过。
- 首次预检因浅克隆活动数据回退而被拒绝；补齐历史及部分克隆所需对象后重新完整执行，最终活动来源为 `git`（220 天）。未修改检查实现或跳过门禁。

完整 lint 仍保留存量信息型提醒（本地全历史运行汇总 674 条；历史出边明细按单桶汇总），不表示全库无质量待办。

## 后续批次

剩余 **3,279 个实际缺项**继续按主题评审。优先复用已存在的方法/任务页；不能建立有意义关联的节点应保留待办，不能批量塞入相同宽泛链接。该历史缺项数是本批基线快照，后续以最新 `lint_wiki.py --report` 实测为准。
