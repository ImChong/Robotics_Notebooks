# Entities 关联出边：第 2 批 100 节点修复

日期：2026-11-18。基线提交：`6c0701076922bfa18490bde5c877155ed1b6ebc6`（最新 `main`）。上一批 [PR #3873](https://github.com/ImChong/Robotics_Notebooks/pull/3873) 已合并；本批从其后续主分支独立创建，与[首批清单](lint-outgoing-pilot-100.md)无重叠。

## 选样与修复范围

从实际触发 `entity_missing_outgoing_legacy` 且仍在冻结豁免中的节点选出 100 页，10 个主题各 10 页，包含 49 个 `paper-*` 路径节点及 51 个训练框架、部署接口、平台、硬件或数据相关实体。优先选择 LIO-SAM、VINS-Mono、Deform360 等有具体机制或任务说明、可核对关联关系的节点。

这是覆盖不同技术角色的目的性批次，不是随机抽样；不能由本批推断全库语义质量。每条新增出边说明算法职责、任务应用或对照关系。潜在下游用途不写成项目已实现的能力。

正文只在现有「关联/相关/关系」段落增加解释性 Markdown 内链，并更新 `updated`；不新增章节来改变健康度，不修改 status、sources、original、身份或评分规则。本批不承担原文补读、实验复现或源码开放状态的更新。

## 修复前后

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| 样本中缺少 Methods/Tasks 出边的页 | 100 | 0 |
| 样本中有 2 个不同且有效目标的页 | 0 | 100 |
| 样本合格出边总数 | 69 | 200 |
| 全库实际历史出边缺项 | 3,279 | 3,179 |
| 历史出边豁免列表条目 | 3,281 | 3,181 |

本批新增 **131 条**出边，涉及 **45 个不同目标**；冻结清单恰好移除本批 100 个路径，其余豁免及缩写基线不变。列表与实际缺项相差的 2 条为既有差异，本批不顺带清理。

两批累计：**200 个不同节点、298 条新增出边、200 条出边豁免移除**。实际缺项从首批基线 3,379 降至本批 3,179。每页仍有至少 2 个不同的有效 Methods/Tasks 目标，后续退化将由未豁免门禁阻塞。

100 页健康度与修改前完全相同：1/3 为 55 页、2/3 为 31 页、3/3 为 14 页；未完成原文通读或阅读骨架的历史缺项仍按实测保留。

## 节点清单

只统计正文中指向存在的 `wiki/methods/` 或 `wiki/tasks/` 的 Markdown 链接，并额外按目标路径去重。frontmatter `related` 与重复链接不充作本批新增出边。表中列出新增目标，原有出边均保留；关系说明详见各节点正文。

| 主题 | 节点 | 修复前 | 修复后 | 本批新增目标 |
|---|---|---:|---:|---|
| 机器人强化学习栈 | [unitree-rl-lab](../../wiki/entities/unitree-rl-lab.md) | 0 | 2 | [ppo](../../wiki/methods/ppo.md)、[locomotion](../../wiki/tasks/locomotion.md) |
| 机器人强化学习栈 | [unitree-rl-gym](../../wiki/entities/unitree-rl-gym.md) | 1 | 2 | [ppo](../../wiki/methods/ppo.md) |
| 机器人强化学习栈 | [unitree-rl-mjlab](../../wiki/entities/unitree-rl-mjlab.md) | 1 | 2 | [ppo](../../wiki/methods/ppo.md) |
| 机器人强化学习栈 | [engineai-rl-lab](../../wiki/entities/engineai-rl-lab.md) | 1 | 2 | [ppo](../../wiki/methods/ppo.md) |
| 机器人强化学习栈 | [cyclo-lab](../../wiki/entities/cyclo-lab.md) | 0 | 2 | [imitation-learning](../../wiki/methods/imitation-learning.md)、[manipulation](../../wiki/tasks/manipulation.md) |
| 机器人强化学习栈 | [holosoma](../../wiki/entities/holosoma.md) | 0 | 2 | [ppo](../../wiki/methods/ppo.md)、[sac](../../wiki/methods/sac.md) |
| 机器人强化学习栈 | [parkour-mjlab](../../wiki/entities/parkour-mjlab.md) | 0 | 2 | [reinforcement-learning](../../wiki/methods/reinforcement-learning.md)、[stair-obstacle-perceptive-locomotion](../../wiki/tasks/stair-obstacle-perceptive-locomotion.md) |
| 机器人强化学习栈 | [robotis-cyclo-mjlab](../../wiki/entities/robotis-cyclo-mjlab.md) | 0 | 2 | [reinforcement-learning](../../wiki/methods/reinforcement-learning.md)、[imitation-learning](../../wiki/methods/imitation-learning.md) |
| 机器人强化学习栈 | [roboparty-ufo](../../wiki/entities/roboparty-ufo.md) | 1 | 2 | [reinforcement-learning](../../wiki/methods/reinforcement-learning.md) |
| 机器人强化学习栈 | [pollen-microduck-rl](../../wiki/entities/pollen-microduck-rl.md) | 1 | 2 | [ppo](../../wiki/methods/ppo.md) |
| 仿真与任务验证 | [unitree-mujoco](../../wiki/entities/unitree-mujoco.md) | 0 | 2 | [pid-control](../../wiki/methods/pid-control.md)、[locomotion](../../wiki/tasks/locomotion.md) |
| 仿真与任务验证 | [unitree-sim-isaaclab](../../wiki/entities/unitree-sim-isaaclab.md) | 1 | 2 | [manipulation](../../wiki/tasks/manipulation.md) |
| 仿真与任务验证 | [robocasa](../../wiki/entities/robocasa.md) | 1 | 2 | [diffusion-policy](../../wiki/methods/diffusion-policy.md) |
| 仿真与任务验证 | [robolab](../../wiki/entities/robolab.md) | 0 | 2 | [vla](../../wiki/methods/vla.md)、[manipulation](../../wiki/tasks/manipulation.md) |
| 仿真与任务验证 | [robosimstudio](../../wiki/entities/robosimstudio.md) | 0 | 2 | [manipulation](../../wiki/tasks/manipulation.md)、[bimanual-manipulation](../../wiki/tasks/bimanual-manipulation.md) |
| 仿真与任务验证 | [f1tenth-gym](../../wiki/entities/f1tenth-gym.md) | 0 | 2 | [reinforcement-learning](../../wiki/methods/reinforcement-learning.md)、[model-predictive-control](../../wiki/methods/model-predictive-control.md) |
| 仿真与任务验证 | [flightmare](../../wiki/entities/flightmare.md) | 0 | 2 | [reinforcement-learning](../../wiki/methods/reinforcement-learning.md)、[trajectory-optimization](../../wiki/methods/trajectory-optimization.md) |
| 仿真与任务验证 | [gym-pybullet-drones](../../wiki/entities/gym-pybullet-drones.md) | 0 | 2 | [reinforcement-learning](../../wiki/methods/reinforcement-learning.md)、[marl](../../wiki/methods/marl.md) |
| 仿真与任务验证 | [quad-swarm-rl](../../wiki/entities/quad-swarm-rl.md) | 0 | 2 | [marl](../../wiki/methods/marl.md)、[reinforcement-learning](../../wiki/methods/reinforcement-learning.md) |
| 仿真与任务验证 | [xtdrone](../../wiki/entities/xtdrone.md) | 0 | 2 | [object-detection](../../wiki/methods/object-detection.md)、[trajectory-optimization](../../wiki/methods/trajectory-optimization.md) |
| 部署与运行接口 | [unitree-sdk2](../../wiki/entities/unitree-sdk2.md) | 0 | 2 | [locomotion](../../wiki/tasks/locomotion.md)、[teleoperation](../../wiki/tasks/teleoperation.md) |
| 部署与运行接口 | [unitree-legged-sdk](../../wiki/entities/unitree-legged-sdk.md) | 0 | 2 | [locomotion](../../wiki/tasks/locomotion.md)、[pid-control](../../wiki/methods/pid-control.md) |
| 部署与运行接口 | [unitree-ros2](../../wiki/entities/unitree-ros2.md) | 0 | 2 | [manipulation](../../wiki/tasks/manipulation.md)、[locomotion](../../wiki/tasks/locomotion.md) |
| 部署与运行接口 | [engineai-native-sdk](../../wiki/entities/engineai-native-sdk.md) | 0 | 2 | [pid-control](../../wiki/methods/pid-control.md)、[humanoid-locomotion](../../wiki/tasks/humanoid-locomotion.md) |
| 部署与运行接口 | [agibot-aimdk-x2](../../wiki/entities/agibot-aimdk-x2.md) | 0 | 2 | [humanoid-locomotion](../../wiki/tasks/humanoid-locomotion.md)、[manipulation](../../wiki/tasks/manipulation.md) |
| 部署与运行接口 | [roboparty-deploy](../../wiki/entities/roboparty-deploy.md) | 0 | 2 | [beyondmimic](../../wiki/methods/beyondmimic.md)、[amp-reward](../../wiki/methods/amp-reward.md) |
| 部署与运行接口 | [rp1-firmware](../../wiki/entities/rp1-firmware.md) | 0 | 2 | [pid-control](../../wiki/methods/pid-control.md)、[humanoid-locomotion](../../wiki/tasks/humanoid-locomotion.md) |
| 部署与运行接口 | [cn-os-tron1-rl-deploy-ros2](../../wiki/entities/cn-os-tron1-rl-deploy-ros2.md) | 1 | 2 | [hybrid-locomotion](../../wiki/tasks/hybrid-locomotion.md) |
| 部署与运行接口 | [cn-os-xbot-sdk-api](../../wiki/entities/cn-os-xbot-sdk-api.md) | 1 | 2 | [model-predictive-control](../../wiki/methods/model-predictive-control.md) |
| 部署与运行接口 | [cn-os-teleop-client](../../wiki/entities/cn-os-teleop-client.md) | 1 | 2 | [teleoperation](../../wiki/tasks/teleoperation.md) |
| 动作生成与重定向 | [stmr-quadruped-retargeting](../../wiki/entities/stmr-quadruped-retargeting.md) | 1 | 2 | [motion-retargeting-gmr](../../wiki/methods/motion-retargeting-gmr.md) |
| 动作生成与重定向 | [axellwppr-motion-tracking](../../wiki/entities/axellwppr-motion-tracking.md) | 1 | 2 | [ppo](../../wiki/methods/ppo.md) |
| 动作生成与重定向 | [go2-motion-imitation](../../wiki/entities/go2-motion-imitation.md) | 1 | 2 | [imitation-learning](../../wiki/methods/imitation-learning.md) |
| 动作生成与重定向 | [videomimic](../../wiki/entities/videomimic.md) | 1 | 2 | [imitation-learning](../../wiki/methods/imitation-learning.md) |
| 动作生成与重定向 | [ddr-direct-dynamics-retargeting](../../wiki/entities/ddr-direct-dynamics-retargeting.md) | 1 | 2 | [model-predictive-control](../../wiki/methods/model-predictive-control.md) |
| 动作生成与重定向 | [holomotion](../../wiki/entities/holomotion.md) | 1 | 2 | [ppo](../../wiki/methods/ppo.md) |
| 动作生成与重定向 | [supermimic-parkour](../../wiki/entities/supermimic-parkour.md) | 0 | 2 | [model-based-rl](../../wiki/methods/model-based-rl.md)、[stair-obstacle-perceptive-locomotion](../../wiki/tasks/stair-obstacle-perceptive-locomotion.md) |
| 动作生成与重定向 | [paper-bfm-20-moconvq](../../wiki/entities/paper-bfm-20-moconvq.md) | 0 | 2 | [hierarchical-reinforcement-learning](../../wiki/methods/hierarchical-reinforcement-learning.md)、[imitation-learning](../../wiki/methods/imitation-learning.md) |
| 动作生成与重定向 | [paper-bfm-38-tokenhsi](../../wiki/entities/paper-bfm-38-tokenhsi.md) | 0 | 2 | [ppo](../../wiki/methods/ppo.md)、[loco-manipulation](../../wiki/tasks/loco-manipulation.md) |
| 动作生成与重定向 | [paper-bfm-39-closd](../../wiki/entities/paper-bfm-39-closd.md) | 0 | 2 | [diffusion-motion-generation](../../wiki/methods/diffusion-motion-generation.md)、[imitation-learning](../../wiki/methods/imitation-learning.md) |
| 导航与几何规划 | [paper-conceptgraphs-open-vocabulary-3d-scene](../../wiki/entities/paper-conceptgraphs-open-vocabulary-3d-scene.md) | 1 | 2 | [zero-shot-object-navigation](../../wiki/tasks/zero-shot-object-navigation.md) |
| 导航与几何规划 | [paper-dcreg-degenerate-lidar-registration](../../wiki/entities/paper-dcreg-degenerate-lidar-registration.md) | 1 | 2 | [conjugate-gradient-method](../../wiki/methods/conjugate-gradient-method.md) |
| 导航与几何规划 | [point-lio-unilidar](../../wiki/entities/point-lio-unilidar.md) | 1 | 2 | [lidar-odometry-fusion](../../wiki/methods/lidar-odometry-fusion.md) |
| 导航与几何规划 | [ego-planner-swarm](../../wiki/entities/ego-planner-swarm.md) | 0 | 2 | [trajectory-optimization](../../wiki/methods/trajectory-optimization.md)、[autonomous-exploration](../../wiki/tasks/autonomous-exploration.md) |
| 导航与几何规划 | [roamerx-navigation](../../wiki/entities/roamerx-navigation.md) | 1 | 2 | [mppi](../../wiki/methods/mppi.md) |
| 导航与几何规划 | [lio-sam](../../wiki/entities/lio-sam.md) | 0 | 2 | [lidar-odometry-fusion](../../wiki/methods/lidar-odometry-fusion.md)、[gauss-newton](../../wiki/methods/gauss-newton.md) |
| 导航与几何规划 | [paper-vins-mono](../../wiki/entities/paper-vins-mono.md) | 0 | 2 | [gauss-newton](../../wiki/methods/gauss-newton.md)、[levenberg-marquardt](../../wiki/methods/levenberg-marquardt.md) |
| 导航与几何规划 | [paper-cmp-irrt-star-height-adaptive-planner](../../wiki/entities/paper-cmp-irrt-star-height-adaptive-planner.md) | 1 | 2 | [informed-rrt-star](../../wiki/methods/informed-rrt-star.md) |
| 导航与几何规划 | [paper-autonomous-spot-nebula-exploration](../../wiki/entities/paper-autonomous-spot-nebula-exploration.md) | 1 | 2 | [autonomous-exploration](../../wiki/tasks/autonomous-exploration.md) |
| 导航与几何规划 | [paper-aptree](../../wiki/entities/paper-aptree.md) | 1 | 2 | [trajectory-optimization](../../wiki/methods/trajectory-optimization.md) |
| 操作策略与动作表示 | [unifolm-vla](../../wiki/entities/unifolm-vla.md) | 1 | 2 | [vla](../../wiki/methods/vla.md) |
| 操作策略与动作表示 | [unifolm-world-model-action](../../wiki/entities/unifolm-world-model-action.md) | 1 | 2 | [vla](../../wiki/methods/vla.md) |
| 操作策略与动作表示 | [paper-ae-vla-arm-wise-compositional](../../wiki/entities/paper-ae-vla-arm-wise-compositional.md) | 1 | 2 | [vla](../../wiki/methods/vla.md) |
| 操作策略与动作表示 | [paper-casd-chunk-semantic-distillation](../../wiki/entities/paper-casd-chunk-semantic-distillation.md) | 1 | 2 | [action-chunking](../../wiki/methods/action-chunking.md) |
| 操作策略与动作表示 | [paper-ciallo](../../wiki/entities/paper-ciallo.md) | 1 | 2 | [imitation-learning](../../wiki/methods/imitation-learning.md) |
| 操作策略与动作表示 | [paper-decal](../../wiki/entities/paper-decal.md) | 1 | 2 | [manipulation](../../wiki/tasks/manipulation.md) |
| 操作策略与动作表示 | [cn-os-graspvla](../../wiki/entities/cn-os-graspvla.md) | 1 | 2 | [manipulation](../../wiki/tasks/manipulation.md) |
| 操作策略与动作表示 | [paper-ctp](../../wiki/entities/paper-ctp.md) | 0 | 2 | [action-chunking](../../wiki/methods/action-chunking.md)、[manipulation](../../wiki/tasks/manipulation.md) |
| 操作策略与动作表示 | [paper-cotograsp](../../wiki/entities/paper-cotograsp.md) | 1 | 2 | [grasp-pose-estimation](../../wiki/methods/grasp-pose-estimation.md) |
| 操作策略与动作表示 | [paper-bimanual-rma-latent-motor-adaptation](../../wiki/entities/paper-bimanual-rma-latent-motor-adaptation.md) | 1 | 2 | [bimanual-manipulation](../../wiki/tasks/bimanual-manipulation.md) |
| 世界模型与数据闭环 | [paper-1xwm-redwood-world-model](../../wiki/entities/paper-1xwm-redwood-world-model.md) | 1 | 2 | [model-based-rl](../../wiki/methods/model-based-rl.md) |
| 世界模型与数据闭环 | [paper-anyworld](../../wiki/entities/paper-anyworld.md) | 1 | 2 | [vla](../../wiki/methods/vla.md) |
| 世界模型与数据闭环 | [paper-act-lam](../../wiki/entities/paper-act-lam.md) | 1 | 2 | [generative-world-models](../../wiki/methods/generative-world-models.md) |
| 世界模型与数据闭环 | [paper-autodidactwam](../../wiki/entities/paper-autodidactwam.md) | 1 | 2 | [generative-world-models](../../wiki/methods/generative-world-models.md) |
| 世界模型与数据闭环 | [paper-aw-0-agentic-world-action-models](../../wiki/entities/paper-aw-0-agentic-world-action-models.md) | 1 | 2 | [vla](../../wiki/methods/vla.md) |
| 世界模型与数据闭环 | [paper-behaviorworldgen](../../wiki/entities/paper-behaviorworldgen.md) | 1 | 2 | [generative-data-augmentation](../../wiki/methods/generative-data-augmentation.md) |
| 世界模型与数据闭环 | [paper-birddreams](../../wiki/entities/paper-birddreams.md) | 1 | 2 | [trajectory-optimization](../../wiki/methods/trajectory-optimization.md) |
| 世界模型与数据闭环 | [paper-causal-jepa](../../wiki/entities/paper-causal-jepa.md) | 1 | 2 | [model-predictive-control](../../wiki/methods/model-predictive-control.md) |
| 世界模型与数据闭环 | [paper-ctrl-world](../../wiki/entities/paper-ctrl-world.md) | 1 | 2 | [vla](../../wiki/methods/vla.md) |
| 世界模型与数据闭环 | [paper-d3wm](../../wiki/entities/paper-d3wm.md) | 1 | 2 | [model-predictive-control](../../wiki/methods/model-predictive-control.md) |
| 触觉与灵巧操作 | [eflesh](../../wiki/entities/eflesh.md) | 0 | 2 | [manipulation](../../wiki/tasks/manipulation.md)、[tactile-impedance-control](../../wiki/methods/tactile-impedance-control.md) |
| 触觉与灵巧操作 | [sharpa-w02-hand](../../wiki/entities/sharpa-w02-hand.md) | 0 | 2 | [manipulation](../../wiki/tasks/manipulation.md)、[in-hand-reorientation](../../wiki/methods/in-hand-reorientation.md) |
| 触觉与灵巧操作 | [sharpa-ae01-data-glove](../../wiki/entities/sharpa-ae01-data-glove.md) | 0 | 2 | [teleoperation](../../wiki/tasks/teleoperation.md)、[imitation-learning](../../wiki/methods/imitation-learning.md) |
| 触觉与灵巧操作 | [tipsim](../../wiki/entities/tipsim.md) | 1 | 2 | [direct-force-feedback-control](../../wiki/methods/direct-force-feedback-control.md) |
| 触觉与灵巧操作 | [paper-deform360-deformable-visuotactile-dataset](../../wiki/entities/paper-deform360-deformable-visuotactile-dataset.md) | 1 | 2 | [model-predictive-control](../../wiki/methods/model-predictive-control.md) |
| 触觉与灵巧操作 | [rp-hand](../../wiki/entities/rp-hand.md) | 1 | 2 | [teleoperation](../../wiki/tasks/teleoperation.md) |
| 触觉与灵巧操作 | [dexbench](../../wiki/entities/dexbench.md) | 1 | 2 | [tactile-impedance-control](../../wiki/methods/tactile-impedance-control.md) |
| 触觉与灵巧操作 | [paper-aero-hand-open](../../wiki/entities/paper-aero-hand-open.md) | 1 | 2 | [in-hand-reorientation](../../wiki/methods/in-hand-reorientation.md) |
| 触觉与灵巧操作 | [paper-berkeley-quad-hand](../../wiki/entities/paper-berkeley-quad-hand.md) | 1 | 2 | [grasp-pose-estimation](../../wiki/methods/grasp-pose-estimation.md) |
| 触觉与灵巧操作 | [paper-contact-imitation-interface](../../wiki/entities/paper-contact-imitation-interface.md) | 1 | 2 | [imitation-learning](../../wiki/methods/imitation-learning.md) |
| 模型控制与接触优化 | [paper-capturability-footstep-mpc](../../wiki/entities/paper-capturability-footstep-mpc.md) | 1 | 2 | [model-predictive-control](../../wiki/methods/model-predictive-control.md) |
| 模型控制与接触优化 | [paper-compliantwbc-heavy-humanoid](../../wiki/entities/paper-compliantwbc-heavy-humanoid.md) | 1 | 2 | [residual-policy-learning](../../wiki/methods/residual-policy-learning.md) |
| 模型控制与接触优化 | [paper-ceer2-directional-compliance](../../wiki/entities/paper-ceer2-directional-compliance.md) | 1 | 2 | [hierarchical-reinforcement-learning](../../wiki/methods/hierarchical-reinforcement-learning.md) |
| 模型控制与接触优化 | [facet-impedance](../../wiki/entities/facet-impedance.md) | 1 | 2 | [reinforcement-learning](../../wiki/methods/reinforcement-learning.md) |
| 模型控制与接触优化 | [paper-contact-aware-adaptive-impedance](../../wiki/entities/paper-contact-aware-adaptive-impedance.md) | 1 | 2 | [direct-force-feedback-control](../../wiki/methods/direct-force-feedback-control.md) |
| 模型控制与接触优化 | [paper-aerial-manipulator-nonlinear-control](../../wiki/entities/paper-aerial-manipulator-nonlinear-control.md) | 1 | 2 | [sliding-mode-control](../../wiki/methods/sliding-mode-control.md) |
| 模型控制与接触优化 | [paper-anti-gravity-flying-humanoid-wbc-mpc](../../wiki/entities/paper-anti-gravity-flying-humanoid-wbc-mpc.md) | 1 | 2 | [model-predictive-control](../../wiki/methods/model-predictive-control.md) |
| 模型控制与接触优化 | [paper-assmpy-soft-robot-orientation](../../wiki/entities/paper-assmpy-soft-robot-orientation.md) | 1 | 2 | [nonlinear-model-predictive-control](../../wiki/methods/nonlinear-model-predictive-control.md) |
| 模型控制与接触优化 | [paper-contactnets-contact-dynamics](../../wiki/entities/paper-contactnets-contact-dynamics.md) | 1 | 2 | [model-based-rl](../../wiki/methods/model-based-rl.md) |
| 模型控制与接触优化 | [paper-cqdc-warp-differentiable-quasi-dynamic](../../wiki/entities/paper-cqdc-warp-differentiable-quasi-dynamic.md) | 1 | 2 | [model-predictive-control](../../wiki/methods/model-predictive-control.md) |
| 全身技能与人形交互 | [paper-adadextrack](../../wiki/entities/paper-adadextrack.md) | 1 | 2 | [imitation-learning](../../wiki/methods/imitation-learning.md) |
| 全身技能与人形交互 | [paper-amc](../../wiki/entities/paper-amc.md) | 1 | 2 | [residual-policy-learning](../../wiki/methods/residual-policy-learning.md) |
| 全身技能与人形交互 | [paper-amp-survey-12-haml](../../wiki/entities/paper-amp-survey-12-haml.md) | 1 | 2 | [imitation-learning](../../wiki/methods/imitation-learning.md) |
| 全身技能与人形交互 | [paper-amp-survey-13-humanoid-goalkeeper](../../wiki/entities/paper-amp-survey-13-humanoid-goalkeeper.md) | 1 | 2 | [humanoid-soccer](../../wiki/tasks/humanoid-soccer.md) |
| 全身技能与人形交互 | [paper-amp-survey-14-husky](../../wiki/entities/paper-amp-survey-14-husky.md) | 1 | 2 | [locomotion](../../wiki/tasks/locomotion.md) |
| 全身技能与人形交互 | [paper-amp-survey-15-physhsi](../../wiki/entities/paper-amp-survey-15-physhsi.md) | 1 | 2 | [amp-reward](../../wiki/methods/amp-reward.md) |
| 全身技能与人形交互 | [paper-amp-survey-17-teamhoi](../../wiki/entities/paper-amp-survey-17-teamhoi.md) | 1 | 2 | [marl](../../wiki/methods/marl.md) |
| 全身技能与人形交互 | [paper-anchor-1](../../wiki/entities/paper-anchor-1.md) | 1 | 2 | [imitation-learning](../../wiki/methods/imitation-learning.md) |
| 全身技能与人形交互 | [paper-agile-perceptive-traversal-sparse-3d](../../wiki/entities/paper-agile-perceptive-traversal-sparse-3d.md) | 1 | 2 | [multi-expert-distillation](../../wiki/methods/multi-expert-distillation.md) |
| 全身技能与人形交互 | [paper-aheadance](../../wiki/entities/paper-aheadance.md) | 1 | 2 | [diffusion-motion-generation](../../wiki/methods/diffusion-motion-generation.md) |

## 可复验检查

本批已逐页验证当前规则通过、目标存在且不同、豁免差异恰等于 100 页、与首批无交集、元数据仅更新日期以及健康度不变。用基线原文回放实际 `_check_entity_page`，100 页均能检出原始缺项，作为负对照。

仓库根目录执行以下代码可复核本批清单、2 个不同目标及负对照（依赖 `requirements-ci-lite.txt`）：

```python
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import lint_wiki as lint

base = "6c0701076922bfa18490bde5c877155ed1b6ebc6"
report = Path("docs/checklists/lint-outgoing-batch-02-100.md").read_text()
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

完整预检、测试与每页健康度以本批 PR 的实测记录为准。生成的图谱、统计、导出 JSON 和 lint 报告不入库。

## 本地检查结果

- `make ci-preflight`：通过。Wiki Lint 0 个阻塞错误；搜索回归 40/40；导出一致性 14/14；Mermaid 12.3.0 校验 8,189 个文件中的 7,957 张图，0 错误；图谱孤儿节点为 0。活动数据来源为完整 Git 历史。
- 相关 pytest：45 passed（历史基线、crosslinks、健康度汇报与图谱健康度四个测试文件；定向运行使用 `-o addopts=`）。
- 无头 Chromium：抽查 `unitree_rl_lab`（0→2）、DDR（1→2）与 CompliantWBC（残差控制关系）；新增目标出现在详情关联项，点击可打开对应方法详情页，截图存本地忽略目录。
- `git diff --check` 及全部 100 页健康度表比对通过。

完整 lint 仍保留存量信息型提醒（本地汇总 672 条，历史出边明细按单桶汇总）；本批未宣称全库无待办。实际历史出边缺项为 3,179 页。

## 后续

剩余 3,179 个实际历史缺项按主题继续评审。对资料不足或无法建立明确关系的节点保留待办，不为降低计数统一塞入宽泛链接；后续数量以最新主分支的实际 lint 为准。
