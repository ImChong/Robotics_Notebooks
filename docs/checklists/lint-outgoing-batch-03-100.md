# Entities 关联出边：第 3 批 100 节点修复

日期：2026-11-18。基线提交：`a5ef875192fecdbf9b0c9a8c96f2527de37d295e`（PR #3875 已有第 2 批，并同步主分支 `d326c6fc9` 后的提交）。本批按要求继续加入 [PR #3875](https://github.com/ImChong/Robotics_Notebooks/pull/3875)，不另开 PR。与[首批清单](lint-outgoing-pilot-100.md)及[第 2 批清单](lint-outgoing-batch-02-100.md)均无重叠。

## 选样与修复范围

从实际仍触发 `entity_missing_outgoing_legacy` 且在冻结豁免中的节点选出 100 页，10 个主题各 10 页，包含 74 个 `paper-*` 路径节点及 26 个定位建图项目、人形平台和执行器等实体；其中 54 页原有 0 条合格出边，46 页原有 1 条。

这是按技术角色覆盖的目的性选样，不是随机抽样。逐页依据既有正文建立方法背景、任务用途或对照关系，并写明适用边界；不由本批推断全库语义质量，也不将潜在下游用途写成项目已实现能力。

正文仅在现有关系区块增加说明性 Markdown 内链，并更新 `updated`；不新增章节改变健康度，不修改 status、sources、original、身份或评分规则。本次是交叉引用维护，不声称补读原文、复现实验或重新核查开源状态。

## 修复前后

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| 本批缺少 Methods/Tasks 出边的页 | 100 | 0 |
| 本批有 2 个不同且有效目标的页 | 0 | 100 |
| 本批合格出边总数 | 46 | 200 |
| 全库实际历史出边缺项 | 3,179 | 3,079 |
| 历史出边豁免列表条目 | 3,181 | 3,081 |

本批新增 **154 条**出边，涉及 **47 个不同目标**；冻结清单恰好移除本批 100 个路径，其余出边豁免、缩写基线和注释不变。列表与实际缺项相差的 2 条为既有差异，本批不顺带清理。

PR #3875 合计第 2–3 批 **200 个不同节点、285 条新增出边（66 个不同目标）、200 条出边豁免移除**；相对同步后的 main，实际缺项 3,279 → 3,079，出边豁免 3,281 → 3,081。三批累计修复 **300 个不同节点、新增 452 条出边、移除 300 条出边豁免**，实际缺项从首批基线 3,379 降至 3,079。

本批 100 页健康度均保持原值：1/3 为 62 页、2/3 为 21 页、3/3 为 17 页。PR #3875 全部 200 页为 1/3 共 117 页、2/3 共 52 页、3/3 共 31 页。历史阅读及骨架缺项仍如实保留；移除出边豁免后，后续出边退化会触发未豁免门禁。

## 节点清单

仅统计正文中指向存在的 `wiki/methods/` 或 `wiki/tasks/` 的 Markdown 链接，并额外按目标路径去重。frontmatter `related` 与重复链接不充作本批新增出边。表中列出新增目标，原有出边均保留；关系说明见各节点正文。

| 主题 | 节点 | 修复前 | 修复后 | 本批新增目标 |
|---|---|---:|---:|---|
| 定位建图与几何优化 | [colmap](../../wiki/entities/colmap.md) | 0 | 2 | [gauss-newton](../../wiki/methods/gauss-newton.md)、[levenberg-marquardt](../../wiki/methods/levenberg-marquardt.md) |
| 定位建图与几何优化 | [hdl-graph-slam](../../wiki/entities/hdl-graph-slam.md) | 0 | 2 | [lidar-odometry-fusion](../../wiki/methods/lidar-odometry-fusion.md)、[gauss-newton](../../wiki/methods/gauss-newton.md) |
| 定位建图与几何优化 | [lego-loam](../../wiki/entities/lego-loam.md) | 0 | 2 | [lidar-odometry-fusion](../../wiki/methods/lidar-odometry-fusion.md)、[gauss-newton](../../wiki/methods/gauss-newton.md) |
| 定位建图与几何优化 | [open-vins](../../wiki/entities/open-vins.md) | 0 | 2 | [autonomous-exploration](../../wiki/tasks/autonomous-exploration.md)、[visual-servoing](../../wiki/methods/visual-servoing.md) |
| 定位建图与几何优化 | [voxgraph](../../wiki/entities/voxgraph.md) | 0 | 2 | [gauss-newton](../../wiki/methods/gauss-newton.md)、[autonomous-exploration](../../wiki/tasks/autonomous-exploration.md) |
| 定位建图与几何优化 | [paper-loam-lidar-odometry-and-mapping](../../wiki/entities/paper-loam-lidar-odometry-and-mapping.md) | 0 | 2 | [lidar-odometry-fusion](../../wiki/methods/lidar-odometry-fusion.md)、[levenberg-marquardt](../../wiki/methods/levenberg-marquardt.md) |
| 定位建图与几何优化 | [paper-loam-livox](../../wiki/entities/paper-loam-livox.md) | 0 | 2 | [lidar-odometry-fusion](../../wiki/methods/lidar-odometry-fusion.md)、[levenberg-marquardt](../../wiki/methods/levenberg-marquardt.md) |
| 定位建图与几何优化 | [paper-fast-livo](../../wiki/entities/paper-fast-livo.md) | 0 | 2 | [lidar-odometry-fusion](../../wiki/methods/lidar-odometry-fusion.md)、[visual-servoing](../../wiki/methods/visual-servoing.md) |
| 定位建图与几何优化 | [paper-faster-lio](../../wiki/entities/paper-faster-lio.md) | 1 | 2 | [autonomous-exploration](../../wiki/tasks/autonomous-exploration.md) |
| 定位建图与几何优化 | [paper-ame-attention-based-map-encoding](../../wiki/entities/paper-ame-attention-based-map-encoding.md) | 1 | 2 | [ppo](../../wiki/methods/ppo.md) |
| 视觉表征与神经建图 | [paper-gaussian-slam](../../wiki/entities/paper-gaussian-slam.md) | 0 | 2 | [gauss-newton](../../wiki/methods/gauss-newton.md)、[autonomous-exploration](../../wiki/tasks/autonomous-exploration.md) |
| 视觉表征与神经建图 | [paper-gs-slam-yan](../../wiki/entities/paper-gs-slam-yan.md) | 0 | 2 | [gauss-newton](../../wiki/methods/gauss-newton.md)、[autonomous-exploration](../../wiki/tasks/autonomous-exploration.md) |
| 视觉表征与神经建图 | [paper-nice-slam](../../wiki/entities/paper-nice-slam.md) | 0 | 2 | [gauss-newton](../../wiki/methods/gauss-newton.md)、[autonomous-exploration](../../wiki/tasks/autonomous-exploration.md) |
| 视觉表征与神经建图 | [paper-panogs-slam](../../wiki/entities/paper-panogs-slam.md) | 0 | 2 | [stereo-matching-foundation-models](../../wiki/methods/stereo-matching-foundation-models.md)、[autonomous-exploration](../../wiki/tasks/autonomous-exploration.md) |
| 视觉表征与神经建图 | [paper-unisim-slam](../../wiki/entities/paper-unisim-slam.md) | 1 | 2 | [gauss-newton](../../wiki/methods/gauss-newton.md) |
| 视觉表征与神经建图 | [paper-functional-slam](../../wiki/entities/paper-functional-slam.md) | 1 | 2 | [zero-shot-object-navigation](../../wiki/tasks/zero-shot-object-navigation.md) |
| 视觉表征与神经建图 | [paper-fast-feature-field-f3](../../wiki/entities/paper-fast-feature-field-f3.md) | 0 | 2 | [stereo-matching-foundation-models](../../wiki/methods/stereo-matching-foundation-models.md)、[fcn-semantic-segmentation](../../wiki/methods/fcn-semantic-segmentation.md) |
| 视觉表征与神经建图 | [paper-dinov2](../../wiki/entities/paper-dinov2.md) | 1 | 2 | [vla](../../wiki/methods/vla.md) |
| 视觉表征与神经建图 | [paper-pixelumm](../../wiki/entities/paper-pixelumm.md) | 1 | 2 | [unified-multimodal-tokens](../../wiki/methods/unified-multimodal-tokens.md) |
| 视觉表征与神经建图 | [paper-pilotnet-saliency](../../wiki/entities/paper-pilotnet-saliency.md) | 0 | 2 | [behavior-cloning](../../wiki/methods/behavior-cloning.md)、[visual-servoing](../../wiki/methods/visual-servoing.md) |
| 行为表征与适应 | [paper-bfm-03-fb-aw](../../wiki/entities/paper-bfm-03-fb-aw.md) | 0 | 2 | [reinforcement-learning](../../wiki/methods/reinforcement-learning.md)、[awr](../../wiki/methods/awr.md) |
| 行为表征与适应 | [paper-bfm-04-fast-imitation-bfm](../../wiki/entities/paper-bfm-04-fast-imitation-bfm.md) | 1 | 2 | [reinforcement-learning](../../wiki/methods/reinforcement-learning.md) |
| 行为表征与适应 | [paper-bfm-05-learning-one-representation](../../wiki/entities/paper-bfm-05-learning-one-representation.md) | 0 | 2 | [reinforcement-learning](../../wiki/methods/reinforcement-learning.md)、[value-based-reinforcement-learning](../../wiki/methods/value-based-reinforcement-learning.md) |
| 行为表征与适应 | [paper-bfm-06-successor-states](../../wiki/entities/paper-bfm-06-successor-states.md) | 0 | 2 | [reinforcement-learning](../../wiki/methods/reinforcement-learning.md)、[value-based-reinforcement-learning](../../wiki/methods/value-based-reinforcement-learning.md) |
| 行为表征与适应 | [paper-bfm-26-aps](../../wiki/entities/paper-bfm-26-aps.md) | 0 | 2 | [reinforcement-learning](../../wiki/methods/reinforcement-learning.md)、[ase](../../wiki/methods/ase.md) |
| 行为表征与适应 | [paper-bfm-27-proto-rl](../../wiki/entities/paper-bfm-27-proto-rl.md) | 0 | 2 | [reinforcement-learning](../../wiki/methods/reinforcement-learning.md)、[policy-optimization](../../wiki/methods/policy-optimization.md) |
| 行为表征与适应 | [paper-bfm-28-re3](../../wiki/entities/paper-bfm-28-re3.md) | 1 | 2 | [policy-optimization](../../wiki/methods/policy-optimization.md) |
| 行为表征与适应 | [paper-bfm-31-task-tokens](../../wiki/entities/paper-bfm-31-task-tokens.md) | 0 | 2 | [reinforcement-learning](../../wiki/methods/reinforcement-learning.md)、[imitation-learning](../../wiki/methods/imitation-learning.md) |
| 行为表征与适应 | [paper-bfm-32-unseen-dynamics](../../wiki/entities/paper-bfm-32-unseen-dynamics.md) | 0 | 2 | [reinforcement-learning](../../wiki/methods/reinforcement-learning.md)、[joint-actuator-parameter-identification](../../wiki/methods/joint-actuator-parameter-identification.md) |
| 行为表征与适应 | [paper-bfm-33-fast-adaptation-bfm](../../wiki/entities/paper-bfm-33-fast-adaptation-bfm.md) | 0 | 2 | [reinforcement-learning](../../wiki/methods/reinforcement-learning.md)、[imitation-learning](../../wiki/methods/imitation-learning.md) |
| 动作生成与交互 | [paper-mdm-human-motion-diffusion](../../wiki/entities/paper-mdm-human-motion-diffusion.md) | 1 | 2 | [deepmimic](../../wiki/methods/deepmimic.md) |
| 动作生成与交互 | [paper-mobius-motion-language](../../wiki/entities/paper-mobius-motion-language.md) | 0 | 2 | [diffusion-motion-generation](../../wiki/methods/diffusion-motion-generation.md)、[unified-multimodal-tokens](../../wiki/methods/unified-multimodal-tokens.md) |
| 动作生成与交互 | [paper-nap-control](../../wiki/entities/paper-nap-control.md) | 1 | 2 | [ppo](../../wiki/methods/ppo.md) |
| 动作生成与交互 | [paper-expressive-motion-behavior](../../wiki/entities/paper-expressive-motion-behavior.md) | 1 | 2 | [dmp](../../wiki/methods/dmp.md) |
| 动作生成与交互 | [paper-humanoidttt](../../wiki/entities/paper-humanoidttt.md) | 0 | 2 | [value-based-reinforcement-learning](../../wiki/methods/value-based-reinforcement-learning.md)、[deepmimic](../../wiki/methods/deepmimic.md) |
| 动作生成与交互 | [paper-notebook-control-operators-for-interactive-character-anim](../../wiki/entities/paper-notebook-control-operators-for-interactive-character-anim.md) | 0 | 2 | [imitation-learning](../../wiki/methods/imitation-learning.md)、[diffusion-motion-generation](../../wiki/methods/diffusion-motion-generation.md) |
| 动作生成与交互 | [paper-notebook-motionvae](../../wiki/entities/paper-notebook-motionvae.md) | 0 | 2 | [hierarchical-reinforcement-learning](../../wiki/methods/hierarchical-reinforcement-learning.md)、[imitation-learning](../../wiki/methods/imitation-learning.md) |
| 动作生成与交互 | [paper-humanoid-badminton-dynamic-racket-skills](../../wiki/entities/paper-humanoid-badminton-dynamic-racket-skills.md) | 1 | 2 | [hierarchical-reinforcement-learning](../../wiki/methods/hierarchical-reinforcement-learning.md) |
| 动作生成与交互 | [paper-bfm-18-hgap](../../wiki/entities/paper-bfm-18-hgap.md) | 0 | 2 | [hierarchical-reinforcement-learning](../../wiki/methods/hierarchical-reinforcement-learning.md)、[deepmimic](../../wiki/methods/deepmimic.md) |
| 动作生成与交互 | [paper-bfm-41-unihsi](../../wiki/entities/paper-bfm-41-unihsi.md) | 0 | 2 | [loco-manipulation](../../wiki/tasks/loco-manipulation.md)、[imitation-learning](../../wiki/methods/imitation-learning.md) |
| 足式控制与状态估计 | [paper-extracting-legged-locomotion-heuristics-rpc](../../wiki/entities/paper-extracting-legged-locomotion-heuristics-rpc.md) | 0 | 2 | [model-predictive-control](../../wiki/methods/model-predictive-control.md)、[locomotion](../../wiki/tasks/locomotion.md) |
| 足式控制与状态估计 | [paper-momentum-control-hierarchical-id](../../wiki/entities/paper-momentum-control-hierarchical-id.md) | 0 | 2 | [inverse-dynamics-control](../../wiki/methods/inverse-dynamics-control.md)、[balance-recovery](../../wiki/tasks/balance-recovery.md) |
| 足式控制与状态估计 | [paper-wbic-mpc-mini-cheetah](../../wiki/entities/paper-wbic-mpc-mini-cheetah.md) | 0 | 2 | [model-predictive-control](../../wiki/methods/model-predictive-control.md)、[locomotion](../../wiki/tasks/locomotion.md) |
| 足式控制与状态估计 | [paper-rapid-locomotion-rl](../../wiki/entities/paper-rapid-locomotion-rl.md) | 0 | 2 | [reinforcement-learning](../../wiki/methods/reinforcement-learning.md)、[locomotion](../../wiki/tasks/locomotion.md) |
| 足式控制与状态估计 | [paper-rma-rapid-motor-adaptation](../../wiki/entities/paper-rma-rapid-motor-adaptation.md) | 1 | 2 | [imitation-learning](../../wiki/methods/imitation-learning.md) |
| 足式控制与状态估计 | [paper-tan-quadruped-agile-locomotion-sim2real](../../wiki/entities/paper-tan-quadruped-agile-locomotion-sim2real.md) | 0 | 2 | [reinforcement-learning](../../wiki/methods/reinforcement-learning.md)、[joint-actuator-parameter-identification](../../wiki/methods/joint-actuator-parameter-identification.md) |
| 足式控制与状态估计 | [paper-notebook-contact-aided-invariant-ekf-for-legged-robots](../../wiki/entities/paper-notebook-contact-aided-invariant-ekf-for-legged-robots.md) | 0 | 2 | [locomotion](../../wiki/tasks/locomotion.md)、[inverse-dynamics-control](../../wiki/methods/inverse-dynamics-control.md) |
| 足式控制与状态估计 | [paper-differentiable-contact-legged](../../wiki/entities/paper-differentiable-contact-legged.md) | 1 | 2 | [trajectory-optimization](../../wiki/methods/trajectory-optimization.md) |
| 足式控制与状态估计 | [paper-humanoid-multiple-contact-fall-planning](../../wiki/entities/paper-humanoid-multiple-contact-fall-planning.md) | 1 | 2 | [balance-recovery](../../wiki/tasks/balance-recovery.md) |
| 足式控制与状态估计 | [paper-wheel-motion-uncertainty-degeneracy-aware-lio](../../wiki/entities/paper-wheel-motion-uncertainty-degeneracy-aware-lio.md) | 1 | 2 | [lidar-odometry-fusion](../../wiki/methods/lidar-odometry-fusion.md) |
| 世界模型与验证 | [paper-abot-world-0](../../wiki/entities/paper-abot-world-0.md) | 1 | 2 | [model-based-rl](../../wiki/methods/model-based-rl.md) |
| 世界模型与验证 | [paper-demiurge-navigable-4d-worlds](../../wiki/entities/paper-demiurge-navigable-4d-worlds.md) | 1 | 2 | [autonomous-exploration](../../wiki/tasks/autonomous-exploration.md) |
| 世界模型与验证 | [paper-dreamx-phi](../../wiki/entities/paper-dreamx-phi.md) | 1 | 2 | [manipulation](../../wiki/tasks/manipulation.md) |
| 世界模型与验证 | [paper-fast-wam](../../wiki/entities/paper-fast-wam.md) | 0 | 2 | [generative-world-models](../../wiki/methods/generative-world-models.md)、[vla](../../wiki/methods/vla.md) |
| 世界模型与验证 | [paper-holi-world](../../wiki/entities/paper-holi-world.md) | 0 | 2 | [auto-labeling-pipelines](../../wiki/methods/auto-labeling-pipelines.md)、[vision-language-navigation](../../wiki/tasks/vision-language-navigation.md) |
| 世界模型与验证 | [paper-lala-world](../../wiki/entities/paper-lala-world.md) | 1 | 2 | [model-based-rl](../../wiki/methods/model-based-rl.md) |
| 世界模型与验证 | [paper-rollback-world-model](../../wiki/entities/paper-rollback-world-model.md) | 1 | 2 | [model-based-rl](../../wiki/methods/model-based-rl.md) |
| 世界模型与验证 | [paper-veo-world-simulator-policy-testing](../../wiki/entities/paper-veo-world-simulator-policy-testing.md) | 1 | 2 | [safe-rl](../../wiki/methods/safe-rl.md) |
| 世界模型与验证 | [paper-video-world-model-survey](../../wiki/entities/paper-video-world-model-survey.md) | 1 | 2 | [model-based-rl](../../wiki/methods/model-based-rl.md) |
| 世界模型与验证 | [atlas-world-model](../../wiki/entities/atlas-world-model.md) | 1 | 2 | [unified-multimodal-tokens](../../wiki/methods/unified-multimodal-tokens.md) |
| 策略规划与闭环纠偏 | [paper-dreamcan-sparse-visual-planning](../../wiki/entities/paper-dreamcan-sparse-visual-planning.md) | 0 | 2 | [generative-world-models](../../wiki/methods/generative-world-models.md)、[manipulation](../../wiki/tasks/manipulation.md) |
| 策略规划与闭环纠偏 | [paper-dreamsteer-vla-deployment-steering](../../wiki/entities/paper-dreamsteer-vla-deployment-steering.md) | 1 | 2 | [model-predictive-control](../../wiki/methods/model-predictive-control.md) |
| 策略规划与闭环纠偏 | [paper-fast-plans-faithful-actions](../../wiki/entities/paper-fast-plans-faithful-actions.md) | 1 | 2 | [action-chunking](../../wiki/methods/action-chunking.md) |
| 策略规划与闭环纠偏 | [paper-fast-vla-once-per-episode](../../wiki/entities/paper-fast-vla-once-per-episode.md) | 1 | 2 | [action-chunking](../../wiki/methods/action-chunking.md) |
| 策略规划与闭环纠偏 | [paper-notebook-navdp-learning-sim-to-real-navigation-diffusion](../../wiki/entities/paper-notebook-navdp-learning-sim-to-real-navigation-diffusion.md) | 0 | 2 | [a-star](../../wiki/methods/a-star.md)、[model-predictive-control](../../wiki/methods/model-predictive-control.md) |
| 策略规划与闭环纠偏 | [paper-openvla](../../wiki/entities/paper-openvla.md) | 1 | 2 | [behavior-cloning](../../wiki/methods/behavior-cloning.md) |
| 策略规划与闭环纠偏 | [paper-pears-tactile-failure-reasoning-diffusion-steering](../../wiki/entities/paper-pears-tactile-failure-reasoning-diffusion-steering.md) | 1 | 2 | [manipulation](../../wiki/tasks/manipulation.md) |
| 策略规划与闭环纠偏 | [paper-pi-human-to-robot](../../wiki/entities/paper-pi-human-to-robot.md) | 1 | 2 | [imitation-learning](../../wiki/methods/imitation-learning.md) |
| 策略规划与闭环纠偏 | [paper-taco-tactile-wm-vla-posttrain](../../wiki/entities/paper-taco-tactile-wm-vla-posttrain.md) | 1 | 2 | [generative-world-models](../../wiki/methods/generative-world-models.md) |
| 策略规划与闭环纠偏 | [paper-dawn](../../wiki/entities/paper-dawn.md) | 1 | 2 | [reinforcement-learning](../../wiki/methods/reinforcement-learning.md) |
| 开源人形与本体模型 | [agibot-lingxi-x1](../../wiki/entities/agibot-lingxi-x1.md) | 0 | 2 | [humanoid-locomotion](../../wiki/tasks/humanoid-locomotion.md)、[reinforcement-learning](../../wiki/methods/reinforcement-learning.md) |
| 开源人形与本体模型 | [asimov-v1](../../wiki/entities/asimov-v1.md) | 0 | 2 | [amp-reward](../../wiki/methods/amp-reward.md)、[humanoid-locomotion](../../wiki/tasks/humanoid-locomotion.md) |
| 开源人形与本体模型 | [atom01-description](../../wiki/entities/atom01-description.md) | 0 | 2 | [motion-retargeting-gmr](../../wiki/methods/motion-retargeting-gmr.md)、[inverse-dynamics-control](../../wiki/methods/inverse-dynamics-control.md) |
| 开源人形与本体模型 | [atom01-hardware](../../wiki/entities/atom01-hardware.md) | 0 | 2 | [humanoid-locomotion](../../wiki/tasks/humanoid-locomotion.md)、[pid-control](../../wiki/methods/pid-control.md) |
| 开源人形与本体模型 | [rp1-description](../../wiki/entities/rp1-description.md) | 0 | 2 | [motion-retargeting-gmr](../../wiki/methods/motion-retargeting-gmr.md)、[inverse-dynamics-control](../../wiki/methods/inverse-dynamics-control.md) |
| 开源人形与本体模型 | [rp1-hardware](../../wiki/entities/rp1-hardware.md) | 0 | 2 | [joint-actuator-parameter-identification](../../wiki/methods/joint-actuator-parameter-identification.md)、[humanoid-locomotion](../../wiki/tasks/humanoid-locomotion.md) |
| 开源人形与本体模型 | [roboparty-rp1](../../wiki/entities/roboparty-rp1.md) | 1 | 2 | [reinforcement-learning](../../wiki/methods/reinforcement-learning.md) |
| 开源人形与本体模型 | [fourier-grx-n1](../../wiki/entities/fourier-grx-n1.md) | 0 | 2 | [teleoperation](../../wiki/tasks/teleoperation.md)、[humanoid-locomotion](../../wiki/tasks/humanoid-locomotion.md) |
| 开源人形与本体模型 | [tienkung-humanoid-open-source](../../wiki/entities/tienkung-humanoid-open-source.md) | 0 | 2 | [humanoid-locomotion](../../wiki/tasks/humanoid-locomotion.md)、[reinforcement-learning](../../wiki/methods/reinforcement-learning.md) |
| 开源人形与本体模型 | [tienkung-lab](../../wiki/entities/tienkung-lab.md) | 0 | 2 | [amp-reward](../../wiki/methods/amp-reward.md)、[humanoid-locomotion](../../wiki/tasks/humanoid-locomotion.md) |
| 关节驱动与足式硬件 | [internal-cycloidal-actuator](../../wiki/entities/internal-cycloidal-actuator.md) | 0 | 2 | [joint-actuator-parameter-identification](../../wiki/methods/joint-actuator-parameter-identification.md)、[pid-control](../../wiki/methods/pid-control.md) |
| 关节驱动与足式硬件 | [moteus](../../wiki/entities/moteus.md) | 0 | 2 | [pid-control](../../wiki/methods/pid-control.md)、[mit-joint-impedance-mode-tuning](../../wiki/methods/mit-joint-impedance-mode-tuning.md) |
| 关节驱动与足式硬件 | [opentorque-actuator](../../wiki/entities/opentorque-actuator.md) | 0 | 2 | [joint-actuator-parameter-identification](../../wiki/methods/joint-actuator-parameter-identification.md)、[pid-control](../../wiki/methods/pid-control.md) |
| 关节驱动与足式硬件 | [cycloidal-quasi-direct-drive-actuator](../../wiki/entities/cycloidal-quasi-direct-drive-actuator.md) | 0 | 2 | [joint-actuator-parameter-identification](../../wiki/methods/joint-actuator-parameter-identification.md)、[mit-joint-impedance-mode-tuning](../../wiki/methods/mit-joint-impedance-mode-tuning.md) |
| 关节驱动与足式硬件 | [mit-mini-cheetah](../../wiki/entities/mit-mini-cheetah.md) | 1 | 2 | [model-predictive-control](../../wiki/methods/model-predictive-control.md) |
| 关节驱动与足式硬件 | [odri-solo-and-bolt](../../wiki/entities/odri-solo-and-bolt.md) | 1 | 2 | [inverse-dynamics-control](../../wiki/methods/inverse-dynamics-control.md) |
| 关节驱动与足式硬件 | [encos-hunter130](../../wiki/entities/encos-hunter130.md) | 1 | 2 | [amp-reward](../../wiki/methods/amp-reward.md) |
| 关节驱动与足式硬件 | [romomo](../../wiki/entities/romomo.md) | 0 | 2 | [joint-actuator-parameter-identification](../../wiki/methods/joint-actuator-parameter-identification.md)、[mit-joint-impedance-mode-tuning](../../wiki/methods/mit-joint-impedance-mode-tuning.md) |
| 关节驱动与足式硬件 | [sts32xx-firmware](../../wiki/entities/sts32xx-firmware.md) | 1 | 2 | [pid-control](../../wiki/methods/pid-control.md) |
| 关节驱动与足式硬件 | [ecat-servo](../../wiki/entities/ecat-servo.md) | 0 | 2 | [pid-control](../../wiki/methods/pid-control.md)、[mit-mode-csp-emulation-tuning](../../wiki/methods/mit-mode-csp-emulation-tuning.md) |
| 接触操作与辨识 | [paper-contact-anchored-retargeting-residual-dexterous](../../wiki/entities/paper-contact-anchored-retargeting-residual-dexterous.md) | 1 | 2 | [residual-policy-learning](../../wiki/methods/residual-policy-learning.md) |
| 接触操作与辨识 | [paper-contact-adaptive-meshing](../../wiki/entities/paper-contact-adaptive-meshing.md) | 0 | 2 | [manipulation](../../wiki/tasks/manipulation.md)、[trajectory-optimization](../../wiki/methods/trajectory-optimization.md) |
| 接触操作与辨识 | [paper-continuous-cloth-neural-ode](../../wiki/entities/paper-continuous-cloth-neural-ode.md) | 1 | 2 | [model-predictive-control](../../wiki/methods/model-predictive-control.md) |
| 接触操作与辨识 | [paper-core-collaboration-role-experts](../../wiki/entities/paper-core-collaboration-role-experts.md) | 1 | 2 | [action-chunking](../../wiki/methods/action-chunking.md) |
| 接触操作与辨识 | [paper-cosmi](../../wiki/entities/paper-cosmi.md) | 1 | 2 | [motion-retargeting-gmr](../../wiki/methods/motion-retargeting-gmr.md) |
| 接触操作与辨识 | [paper-cuarms](../../wiki/entities/paper-cuarms.md) | 1 | 2 | [bimanual-manipulation](../../wiki/tasks/bimanual-manipulation.md) |
| 接触操作与辨识 | [paper-curaeval](../../wiki/entities/paper-curaeval.md) | 1 | 2 | [behavior-cloning](../../wiki/methods/behavior-cloning.md) |
| 接触操作与辨识 | [paper-catch-it-all](../../wiki/entities/paper-catch-it-all.md) | 1 | 2 | [reinforcement-learning](../../wiki/methods/reinforcement-learning.md) |
| 接触操作与辨识 | [paper-a2i-calib-plus-multi-imu-legged-calibration](../../wiki/entities/paper-a2i-calib-plus-multi-imu-legged-calibration.md) | 1 | 2 | [amp-reward](../../wiki/methods/amp-reward.md) |
| 接触操作与辨识 | [paper-actuator-dynamics-curricula-legged-rl](../../wiki/entities/paper-actuator-dynamics-curricula-legged-rl.md) | 1 | 2 | [sim2real-joint-sysid-experiment-design](../../wiki/methods/sim2real-joint-sysid-experiment-design.md) |

## 可复验检查

已逐页验证本批 100 页当前规则通过、目标存在且不同；基线原文回放实际 `_check_entity_page`，100 页均检出原始缺项，作为负对照。另对 PR #3875 全部 200 页相对同步后的 main 重做同样验证，确认规则 200/200 通过、负对照 200/200 可检出、健康度不变、元数据仅更新日期，以及豁免移除集合恰等于两批清单。三份清单各为 100 个唯一节点且两两无交集。

仓库根目录执行以下代码可复核本批清单、2 个不同目标及负对照（依赖 `requirements-ci-lite.txt`）：

```python
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import lint_wiki as lint

base = "a5ef875192fecdbf9b0c9a8c96f2527de37d295e"
report = Path("docs/checklists/lint-outgoing-batch-03-100.md").read_text()
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

本地实测结果：

- `make ci-preflight` 完整通过：Wiki Lint 0 个阻塞错误、搜索 40/40、导出 14/14；Mermaid 7,959 图 / 8,190 文件 / 0 错误；图谱 0 孤儿节点，活动数据来源为完整 Git 历史。
- 定向 pytest 45 passed，覆盖冻结基线、交叉引用、健康度计算及 PR 正文健康度校验。
- 本页复验代码实跑：100/100 当前通过，100/100 基线负对照可检出。
- 无头 Chromium：COLMAP（0→2）、AME（1→2）、moteus（0→2）关联项展示及 5 个新增目标详情跳转通过；本地截图位于 `.cursor-artifacts/screenshots/batch3-*-related.png`。
- PR #3875 正文覆盖全部 200 页的实测健康度，并通过同一正文校验器；`git diff --check` 通过。

全库仍有 672 条信息型预警（明细按桶汇总），实际历史出边缺项 3,079 页。生成的图谱、统计、导出 JSON、截图和 lint 报告不入库。

## 后续

剩余 3,079 个实际历史缺项继续按主题评审。资料不足或无法建立明确关系的节点保留待办；后续数量以最新主分支实际 lint 为准。
