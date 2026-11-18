# Entities 关联出边：第 4 批 100 节点修复

日期：2026-11-18。基线提交：`6ceeed5cd17019b9836c4712cf56f071ea465bb6`（最新主分支）。[PR #3873](https://github.com/ImChong/Robotics_Notebooks/pull/3873) 与 [PR #3875](https://github.com/ImChong/Robotics_Notebooks/pull/3875) 已合并；本批另建分支与 PR，与[首批](lint-outgoing-pilot-100.md)、[第 2 批](lint-outgoing-batch-02-100.md)、[第 3 批](lint-outgoing-batch-03-100.md)各 100 节点均无重叠。

## 选样与修复范围

从实际仍触发 `entity_missing_outgoing_legacy` 且在冻结豁免中的节点选出 100 页，10 个主题各 10 页，包括 46 个 `paper-*` 路径、视觉与触觉数据、仿真基础设施、固件/总线及传感接口。其中 45 页原有 0 条合格出边，55 页原有 1 条。

这是按技术角色覆盖的目的性选样，不是随机抽样。逐页依据既有正文建立方法背景、任务用途或对照关系，并明确适用边界；不能由本批推断全库语义质量。选样中替换了仅有待深读占位内容的 HumanoidPano，以及原有任务关联不适合本批读法的 HandMimicX，采用 StairMaster 和 InterMimicGen。二手索引节点只补关系读法，原有待核实状态保留。

只在现有关系区块增加说明性 Markdown 内链并更新 `updated`；不新增章节改变健康度，不修改 status、sources、original、身份或评分规则。这是交叉引用维护，不声称补读原文、复现实验或重核开源状态。

## 修复前后

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| 本批缺少 Methods/Tasks 出边的页 | 100 | 0 |
| 本批有 2 个不同且有效目标的页 | 0 | 100 |
| 本批合格出边总数 | 55 | 200 |
| 全库实际历史出边缺项 | 3,079 | 2,979 |
| 历史出边豁免列表条目 | 3,081 | 2,981 |

本批新增 **145 条**出边，涉及 **51 个不同目标**；冻结清单恰好移除本批 100 个路径，其他出边豁免、缩写基线和注释不变。豁免列表与实际缺项相差的 2 条为既有差异，本批不顺带清理。

四批累计：**400 个不同节点、597 条新增出边、400 条出边豁免移除**。实际历史出边缺项从首批基线 3,379 降至 2,979。移除豁免后，节点出边退化会触发未豁免门禁。

本批 100 页健康度全部与基线相同：1/3 为 46 页、2/3 为 36 页、3/3 为 18 页。原有阅读、骨架和状态缺项仍如实保留。

## 节点清单

仅统计正文中指向存在的 `wiki/methods/` 或 `wiki/tasks/` 的 Markdown 链接，并额外按目标路径去重。frontmatter `related` 与重复链接不充作新增出边。表中列出新增目标，原有出边均保留；关系说明见各节点正文。

| 主题 | 节点 | 修复前 | 修复后 | 本批新增目标 |
|---|---|---:|---:|---|
| 几何重建与多传感定位 | [openvslam](../../wiki/entities/openvslam.md) | 0 | 2 | [gauss-newton](../../wiki/methods/gauss-newton.md)、[autonomous-exploration](../../wiki/tasks/autonomous-exploration.md) |
| 几何重建与多传感定位 | [orb-slam3](../../wiki/entities/orb-slam3.md) | 0 | 2 | [gauss-newton](../../wiki/methods/gauss-newton.md)、[autonomous-exploration](../../wiki/tasks/autonomous-exploration.md) |
| 几何重建与多传感定位 | [paper-r2live](../../wiki/entities/paper-r2live.md) | 0 | 2 | [lidar-odometry-fusion](../../wiki/methods/lidar-odometry-fusion.md)、[gauss-newton](../../wiki/methods/gauss-newton.md) |
| 几何重建与多传感定位 | [paper-r3live](../../wiki/entities/paper-r3live.md) | 0 | 2 | [lidar-odometry-fusion](../../wiki/methods/lidar-odometry-fusion.md)、[autonomous-exploration](../../wiki/tasks/autonomous-exploration.md) |
| 几何重建与多传感定位 | [paper-vggt-visual-geometry-grounded-transformer](../../wiki/entities/paper-vggt-visual-geometry-grounded-transformer.md) | 0 | 2 | [lingbot-map](../../wiki/methods/lingbot-map.md)、[stereo-matching-foundation-models](../../wiki/methods/stereo-matching-foundation-models.md) |
| 几何重建与多传感定位 | [paper-avggt](../../wiki/entities/paper-avggt.md) | 0 | 2 | [lingbot-map](../../wiki/methods/lingbot-map.md)、[stereo-matching-foundation-models](../../wiki/methods/stereo-matching-foundation-models.md) |
| 几何重建与多传感定位 | [paper-puma-lidar-mesh-odometry](../../wiki/entities/paper-puma-lidar-mesh-odometry.md) | 1 | 2 | [gauss-newton](../../wiki/methods/gauss-newton.md) |
| 几何重建与多传感定位 | [paper-ultra-fusion-multi-sensor-slam](../../wiki/entities/paper-ultra-fusion-multi-sensor-slam.md) | 0 | 2 | [lidar-odometry-fusion](../../wiki/methods/lidar-odometry-fusion.md)、[gauss-newton](../../wiki/methods/gauss-newton.md) |
| 几何重建与多传感定位 | [paper-super-odometry-2](../../wiki/entities/paper-super-odometry-2.md) | 1 | 2 | [autonomous-exploration](../../wiki/tasks/autonomous-exploration.md) |
| 几何重建与多传感定位 | [paper-raven-active-slam](../../wiki/entities/paper-raven-active-slam.md) | 0 | 2 | [autonomous-exploration](../../wiki/tasks/autonomous-exploration.md)、[a-star](../../wiki/methods/a-star.md) |
| 视觉数据与几何评测 | [dataset-ade20k](../../wiki/entities/dataset-ade20k.md) | 1 | 2 | [fcn-semantic-segmentation](../../wiki/methods/fcn-semantic-segmentation.md) |
| 视觉数据与几何评测 | [dataset-cityscapes](../../wiki/entities/dataset-cityscapes.md) | 1 | 2 | [segnet](../../wiki/methods/segnet.md) |
| 视觉数据与几何评测 | [dataset-coco](../../wiki/entities/dataset-coco.md) | 1 | 2 | [mask-rcnn](../../wiki/methods/mask-rcnn.md) |
| 视觉数据与几何评测 | [dataset-flickr30k-entities](../../wiki/entities/dataset-flickr30k-entities.md) | 1 | 2 | [zero-shot-object-navigation](../../wiki/tasks/zero-shot-object-navigation.md) |
| 视觉数据与几何评测 | [dataset-imagenet](../../wiki/entities/dataset-imagenet.md) | 1 | 2 | [fcn-semantic-segmentation](../../wiki/methods/fcn-semantic-segmentation.md) |
| 视觉数据与几何评测 | [dataset-mapillary](../../wiki/entities/dataset-mapillary.md) | 1 | 2 | [pspnet](../../wiki/methods/pspnet.md) |
| 视觉数据与几何评测 | [dataset-objects365](../../wiki/entities/dataset-objects365.md) | 1 | 2 | [rcnn-family](../../wiki/methods/rcnn-family.md) |
| 视觉数据与几何评测 | [dataset-pascal-voc](../../wiki/entities/dataset-pascal-voc.md) | 1 | 2 | [fcn-semantic-segmentation](../../wiki/methods/fcn-semantic-segmentation.md) |
| 视觉数据与几何评测 | [kitti-stereo-benchmark](../../wiki/entities/kitti-stereo-benchmark.md) | 1 | 2 | [autonomous-exploration](../../wiki/tasks/autonomous-exploration.md) |
| 视觉数据与几何评测 | [middlebury-stereo-benchmark](../../wiki/entities/middlebury-stereo-benchmark.md) | 1 | 2 | [visual-servoing](../../wiki/methods/visual-servoing.md) |
| 场景数据与导航验证 | [dataset-gibson-environment](../../wiki/entities/dataset-gibson-environment.md) | 1 | 2 | [autonomous-exploration](../../wiki/tasks/autonomous-exploration.md) |
| 场景数据与导航验证 | [dataset-hm3d](../../wiki/entities/dataset-hm3d.md) | 1 | 2 | [zero-shot-object-navigation](../../wiki/tasks/zero-shot-object-navigation.md) |
| 场景数据与导航验证 | [dataset-hssd](../../wiki/entities/dataset-hssd.md) | 1 | 2 | [zero-shot-object-navigation](../../wiki/tasks/zero-shot-object-navigation.md) |
| 场景数据与导航验证 | [dataset-procthor-10k](../../wiki/entities/dataset-procthor-10k.md) | 1 | 2 | [reinforcement-learning](../../wiki/methods/reinforcement-learning.md) |
| 场景数据与导航验证 | [dataset-replicacad](../../wiki/entities/dataset-replicacad.md) | 1 | 2 | [manipulation](../../wiki/tasks/manipulation.md) |
| 场景数据与导航验证 | [dataset-scannet](../../wiki/entities/dataset-scannet.md) | 1 | 2 | [fcn-semantic-segmentation](../../wiki/methods/fcn-semantic-segmentation.md) |
| 场景数据与导航验证 | [dataset-embodiedscan](../../wiki/entities/dataset-embodiedscan.md) | 1 | 2 | [object-detection](../../wiki/methods/object-detection.md) |
| 场景数据与导航验证 | [matterport3d-simulator](../../wiki/entities/matterport3d-simulator.md) | 1 | 2 | [zero-shot-object-navigation](../../wiki/tasks/zero-shot-object-navigation.md) |
| 场景数据与导航验证 | [habitatgs](../../wiki/entities/habitatgs.md) | 0 | 2 | [vision-language-navigation](../../wiki/tasks/vision-language-navigation.md)、[auto-labeling-pipelines](../../wiki/methods/auto-labeling-pipelines.md) |
| 场景数据与导航验证 | [isaac-ros-nvblox](../../wiki/entities/isaac-ros-nvblox.md) | 0 | 2 | [autonomous-exploration](../../wiki/tasks/autonomous-exploration.md)、[trajectory-optimization](../../wiki/methods/trajectory-optimization.md) |
| 接触感知数据 | [dataset-objectfolder-2-0](../../wiki/entities/dataset-objectfolder-2-0.md) | 1 | 2 | [tactile-impedance-control](../../wiki/methods/tactile-impedance-control.md) |
| 接触感知数据 | [dataset-objectfolder-real](../../wiki/entities/dataset-objectfolder-real.md) | 1 | 2 | [direct-force-feedback-control](../../wiki/methods/direct-force-feedback-control.md) |
| 接触感知数据 | [dataset-opentouch](../../wiki/entities/dataset-opentouch.md) | 1 | 2 | [imitation-learning](../../wiki/methods/imitation-learning.md) |
| 接触感知数据 | [dataset-touch-and-go](../../wiki/entities/dataset-touch-and-go.md) | 1 | 2 | [unified-multimodal-tokens](../../wiki/methods/unified-multimodal-tokens.md) |
| 接触感知数据 | [dataset-touch-in-the-wild](../../wiki/entities/dataset-touch-in-the-wild.md) | 1 | 2 | [imitation-learning](../../wiki/methods/imitation-learning.md) |
| 接触感知数据 | [dataset-touch100k](../../wiki/entities/dataset-touch100k.md) | 1 | 2 | [auto-labeling-pipelines](../../wiki/methods/auto-labeling-pipelines.md) |
| 接触感知数据 | [dataset-visgel](../../wiki/entities/dataset-visgel.md) | 0 | 2 | [unified-multimodal-tokens](../../wiki/methods/unified-multimodal-tokens.md)、[manipulation](../../wiki/tasks/manipulation.md) |
| 接触感知数据 | [dataset-ycb-sight](../../wiki/entities/dataset-ycb-sight.md) | 0 | 2 | [fcn-semantic-segmentation](../../wiki/methods/fcn-semantic-segmentation.md)、[manipulation](../../wiki/tasks/manipulation.md) |
| 接触感知数据 | [dataset-ycb-slide](../../wiki/entities/dataset-ycb-slide.md) | 1 | 2 | [tactile-impedance-control](../../wiki/methods/tactile-impedance-control.md) |
| 接触感知数据 | [dataset-rct-robotic-contact-tactile](../../wiki/entities/dataset-rct-robotic-contact-tactile.md) | 0 | 2 | [tactile-impedance-control](../../wiki/methods/tactile-impedance-control.md)、[manipulation](../../wiki/tasks/manipulation.md) |
| 仿真与训练基础设施 | [mujoco-menagerie](../../wiki/entities/mujoco-menagerie.md) | 0 | 2 | [joint-actuator-parameter-identification](../../wiki/methods/joint-actuator-parameter-identification.md)、[inverse-dynamics-control](../../wiki/methods/inverse-dynamics-control.md) |
| 仿真与训练基础设施 | [mujoco-mjx](../../wiki/entities/mujoco-mjx.md) | 1 | 2 | [trajectory-optimization](../../wiki/methods/trajectory-optimization.md) |
| 仿真与训练基础设施 | [mujoco-playground](../../wiki/entities/mujoco-playground.md) | 0 | 2 | [reinforcement-learning](../../wiki/methods/reinforcement-learning.md)、[teleoperation](../../wiki/tasks/teleoperation.md) |
| 仿真与训练基础设施 | [mujoco-warp](../../wiki/entities/mujoco-warp.md) | 1 | 2 | [mppi](../../wiki/methods/mppi.md) |
| 仿真与训练基础设施 | [mujoco-wasm](../../wiki/entities/mujoco-wasm.md) | 0 | 2 | [pid-control](../../wiki/methods/pid-control.md)、[locomotion](../../wiki/tasks/locomotion.md) |
| 仿真与训练基础设施 | [isaac-sim](../../wiki/entities/isaac-sim.md) | 0 | 2 | [reinforcement-learning](../../wiki/methods/reinforcement-learning.md)、[auto-labeling-pipelines](../../wiki/methods/auto-labeling-pipelines.md) |
| 仿真与训练基础设施 | [isaac-launchable](../../wiki/entities/isaac-launchable.md) | 0 | 2 | [reinforcement-learning](../../wiki/methods/reinforcement-learning.md)、[locomotion](../../wiki/tasks/locomotion.md) |
| 仿真与训练基础设施 | [genesis-world-10](../../wiki/entities/genesis-world-10.md) | 0 | 2 | [joint-actuator-parameter-identification](../../wiki/methods/joint-actuator-parameter-identification.md)、[reinforcement-learning](../../wiki/methods/reinforcement-learning.md) |
| 仿真与训练基础设施 | [loco-mujoco](../../wiki/entities/loco-mujoco.md) | 0 | 2 | [amp-reward](../../wiki/methods/amp-reward.md)、[deepmimic](../../wiki/methods/deepmimic.md) |
| 仿真与训练基础设施 | [maniskill2](../../wiki/entities/maniskill2.md) | 1 | 2 | [manipulation](../../wiki/tasks/manipulation.md) |
| 固件总线与传感接口 | [atom01-firmware](../../wiki/entities/atom01-firmware.md) | 0 | 2 | [pid-control](../../wiki/methods/pid-control.md)、[mit-joint-impedance-mode-tuning](../../wiki/methods/mit-joint-impedance-mode-tuning.md) |
| 固件总线与传感接口 | [cherryecat](../../wiki/entities/cherryecat.md) | 0 | 2 | [pid-control](../../wiki/methods/pid-control.md)、[mit-mode-csp-emulation-tuning](../../wiki/methods/mit-mode-csp-emulation-tuning.md) |
| 固件总线与传感接口 | [opensml](../../wiki/entities/opensml.md) | 0 | 2 | [time-optimal-path-parameterization](../../wiki/methods/time-optimal-path-parameterization.md)、[pid-control](../../wiki/methods/pid-control.md) |
| 固件总线与传感接口 | [soes](../../wiki/entities/soes.md) | 0 | 2 | [mit-mode-csp-emulation-tuning](../../wiki/methods/mit-mode-csp-emulation-tuning.md)、[direct-force-feedback-control](../../wiki/methods/direct-force-feedback-control.md) |
| 固件总线与传感接口 | [igh-ethercat-master](../../wiki/entities/igh-ethercat-master.md) | 0 | 2 | [mit-mode-csp-emulation-tuning](../../wiki/methods/mit-mode-csp-emulation-tuning.md)、[pid-control](../../wiki/methods/pid-control.md) |
| 固件总线与传感接口 | [ros2-ethercat](../../wiki/entities/ros2-ethercat.md) | 0 | 2 | [mit-mode-csp-emulation-tuning](../../wiki/methods/mit-mode-csp-emulation-tuning.md)、[inverse-dynamics-control](../../wiki/methods/inverse-dynamics-control.md) |
| 固件总线与传感接口 | [ardupilot](../../wiki/entities/ardupilot.md) | 0 | 2 | [pid-control](../../wiki/methods/pid-control.md)、[autonomous-exploration](../../wiki/tasks/autonomous-exploration.md) |
| 固件总线与传感接口 | [px4-autopilot](../../wiki/entities/px4-autopilot.md) | 0 | 2 | [pid-control](../../wiki/methods/pid-control.md)、[autonomous-exploration](../../wiki/tasks/autonomous-exploration.md) |
| 固件总线与传感接口 | [betaflight](../../wiki/entities/betaflight.md) | 0 | 2 | [pid-control](../../wiki/methods/pid-control.md)、[lqr-ilqr](../../wiki/methods/lqr-ilqr.md) |
| 固件总线与传感接口 | [holoscan-sensor-bridge](../../wiki/entities/holoscan-sensor-bridge.md) | 0 | 2 | [lidar-odometry-fusion](../../wiki/methods/lidar-odometry-fusion.md)、[visual-servoing](../../wiki/methods/visual-servoing.md) |
| 人形跟踪与形态迁移 | [paper-dense-temporal-motion-retargeting](../../wiki/entities/paper-dense-temporal-motion-retargeting.md) | 0 | 2 | [trajectory-optimization](../../wiki/methods/trajectory-optimization.md)、[deepmimic](../../wiki/methods/deepmimic.md) |
| 人形跟踪与形态迁移 | [paper-dirtymocap-robust-motion-capture](../../wiki/entities/paper-dirtymocap-robust-motion-capture.md) | 0 | 2 | [gauss-newton](../../wiki/methods/gauss-newton.md)、[motion-retargeting-gmr](../../wiki/methods/motion-retargeting-gmr.md) |
| 人形跟踪与形态迁移 | [paper-mimicall](../../wiki/entities/paper-mimicall.md) | 0 | 2 | [deepmimic](../../wiki/methods/deepmimic.md)、[humanoid-locomotion](../../wiki/tasks/humanoid-locomotion.md) |
| 人形跟踪与形态迁移 | [paper-morphmimic](../../wiki/entities/paper-morphmimic.md) | 1 | 2 | [model-based-rl](../../wiki/methods/model-based-rl.md) |
| 人形跟踪与形态迁移 | [paper-motionskill-cross-embodiment](../../wiki/entities/paper-motionskill-cross-embodiment.md) | 0 | 2 | [motion-retargeting-gmr](../../wiki/methods/motion-retargeting-gmr.md)、[residual-policy-learning](../../wiki/methods/residual-policy-learning.md) |
| 人形跟踪与形态迁移 | [paper-intermimicgen](../../wiki/entities/paper-intermimicgen.md) | 1 | 2 | [ppo](../../wiki/methods/ppo.md) |
| 人形跟踪与形态迁移 | [isaac-asimov](../../wiki/entities/isaac-asimov.md) | 1 | 2 | [amp-reward](../../wiki/methods/amp-reward.md) |
| 人形跟踪与形态迁移 | [lerobot-humanoid](../../wiki/entities/lerobot-humanoid.md) | 1 | 2 | [joint-actuator-parameter-identification](../../wiki/methods/joint-actuator-parameter-identification.md) |
| 人形跟踪与形态迁移 | [paper-digit-humanoid-locomotion-rl](../../wiki/entities/paper-digit-humanoid-locomotion-rl.md) | 1 | 2 | [teacher-student-dagger-training](../../wiki/methods/teacher-student-dagger-training.md) |
| 人形跟踪与形态迁移 | [paper-cassie-iterative-locomotion-sim2real](../../wiki/entities/paper-cassie-iterative-locomotion-sim2real.md) | 0 | 2 | [reinforcement-learning](../../wiki/methods/reinforcement-learning.md)、[locomotion](../../wiki/tasks/locomotion.md) |
| 人类示范与灵巧操作 | [paper-dexmv](../../wiki/entities/paper-dexmv.md) | 1 | 2 | [imitation-learning](../../wiki/methods/imitation-learning.md) |
| 人类示范与灵巧操作 | [paper-dexseed](../../wiki/entities/paper-dexseed.md) | 1 | 2 | [reinforcement-learning](../../wiki/methods/reinforcement-learning.md) |
| 人类示范与灵巧操作 | [paper-dexstitch-skill-stitching-human-videos](../../wiki/entities/paper-dexstitch-skill-stitching-human-videos.md) | 1 | 2 | [manipulation](../../wiki/tasks/manipulation.md) |
| 人类示范与灵巧操作 | [paper-demomimic](../../wiki/entities/paper-demomimic.md) | 1 | 2 | [imitation-learning](../../wiki/methods/imitation-learning.md) |
| 人类示范与灵巧操作 | [paper-dexninja-real2sim2real-cutting](../../wiki/entities/paper-dexninja-real2sim2real-cutting.md) | 0 | 2 | [generative-data-augmentation](../../wiki/methods/generative-data-augmentation.md)、[manipulation](../../wiki/tasks/manipulation.md) |
| 人类示范与灵巧操作 | [paper-dexremo-2611-15981](../../wiki/entities/paper-dexremo-2611-15981.md) | 0 | 2 | [action-chunking](../../wiki/methods/action-chunking.md)、[teleoperation](../../wiki/tasks/teleoperation.md) |
| 人类示范与灵巧操作 | [paper-goaldex](../../wiki/entities/paper-goaldex.md) | 1 | 2 | [imitation-learning](../../wiki/methods/imitation-learning.md) |
| 人类示范与灵巧操作 | [paper-flashdexretarget](../../wiki/entities/paper-flashdexretarget.md) | 1 | 2 | [flashsac](../../wiki/methods/flashsac.md) |
| 人类示范与灵巧操作 | [paper-modex](../../wiki/entities/paper-modex.md) | 1 | 2 | [residual-policy-learning](../../wiki/methods/residual-policy-learning.md) |
| 人类示范与灵巧操作 | [paper-dexverse](../../wiki/entities/paper-dexverse.md) | 1 | 2 | [teleoperation](../../wiki/tasks/teleoperation.md) |
| 导航与移动操作策略 | [paper-notebook-nomad-goal-masked-diffusion-policies-for-navigat](../../wiki/entities/paper-notebook-nomad-goal-masked-diffusion-policies-for-navigat.md) | 0 | 2 | [diffusion-policy](../../wiki/methods/diffusion-policy.md)、[autonomous-exploration](../../wiki/tasks/autonomous-exploration.md) |
| 导航与移动操作策略 | [paper-notebook-anymal-parkour-robust-perceptive-locomotion](../../wiki/entities/paper-notebook-anymal-parkour-robust-perceptive-locomotion.md) | 1 | 2 | [hierarchical-reinforcement-learning](../../wiki/methods/hierarchical-reinforcement-learning.md) |
| 导航与移动操作策略 | [paper-footquery-perceptive-humanoid-locomotion](../../wiki/entities/paper-footquery-perceptive-humanoid-locomotion.md) | 1 | 2 | [ppo](../../wiki/methods/ppo.md) |
| 导航与移动操作策略 | [paper-mtc-scene-aware-humanoid-locomotion](../../wiki/entities/paper-mtc-scene-aware-humanoid-locomotion.md) | 1 | 2 | [motion-retargeting-gmr](../../wiki/methods/motion-retargeting-gmr.md) |
| 导航与移动操作策略 | [paper-stairmaster](../../wiki/entities/paper-stairmaster.md) | 0 | 2 | [stair-obstacle-perceptive-locomotion](../../wiki/tasks/stair-obstacle-perceptive-locomotion.md)、[ppo](../../wiki/methods/ppo.md) |
| 导航与移动操作策略 | [paper-locomanilab](../../wiki/entities/paper-locomanilab.md) | 1 | 2 | [teleoperation](../../wiki/tasks/teleoperation.md) |
| 导航与移动操作策略 | [paper-locomotion-grounded-humanoid-soccer](../../wiki/entities/paper-locomotion-grounded-humanoid-soccer.md) | 1 | 2 | [hierarchical-reinforcement-learning](../../wiki/methods/hierarchical-reinforcement-learning.md) |
| 导航与移动操作策略 | [paper-polymorphwbc-quadruped](../../wiki/entities/paper-polymorphwbc-quadruped.md) | 0 | 2 | [loco-manipulation](../../wiki/tasks/loco-manipulation.md)、[locomotion](../../wiki/tasks/locomotion.md) |
| 导航与移动操作策略 | [paper-walk-these-ways-quadruped-mob](../../wiki/entities/paper-walk-these-ways-quadruped-mob.md) | 1 | 2 | [reinforcement-learning](../../wiki/methods/reinforcement-learning.md) |
| 导航与移动操作策略 | [paper-walking-on-roofs-quadruped-go2](../../wiki/entities/paper-walking-on-roofs-quadruped-go2.md) | 1 | 2 | [balance-recovery](../../wiki/tasks/balance-recovery.md) |
| 生成策略与运动条件 | [paper-action-representation-pretraining](../../wiki/entities/paper-action-representation-pretraining.md) | 1 | 2 | [imitation-learning](../../wiki/methods/imitation-learning.md) |
| 生成策略与运动条件 | [paper-actioncodec2-streaming-action-tokenizer](../../wiki/entities/paper-actioncodec2-streaming-action-tokenizer.md) | 0 | 2 | [unified-multimodal-tokens](../../wiki/methods/unified-multimodal-tokens.md)、[action-chunking](../../wiki/methods/action-chunking.md) |
| 生成策略与运动条件 | [paper-diffusion-vla](../../wiki/entities/paper-diffusion-vla.md) | 0 | 2 | [vla](../../wiki/methods/vla.md)、[diffusion-policy](../../wiki/methods/diffusion-policy.md) |
| 生成策略与运动条件 | [paper-aamdm-accelerated-motion-diffusion](../../wiki/entities/paper-aamdm-accelerated-motion-diffusion.md) | 1 | 2 | [deepmimic](../../wiki/methods/deepmimic.md) |
| 生成策略与运动条件 | [paper-notebook-guided-motion-diffusion-for-controllable-human-m](../../wiki/entities/paper-notebook-guided-motion-diffusion-for-controllable-human-m.md) | 1 | 2 | [trajectory-optimization](../../wiki/methods/trajectory-optimization.md) |
| 生成策略与运动条件 | [paper-notebook-flexible-motion-in-betweening-with-diffusion-mod](../../wiki/entities/paper-notebook-flexible-motion-in-betweening-with-diffusion-mod.md) | 1 | 2 | [deepmimic](../../wiki/methods/deepmimic.md) |
| 生成策略与运动条件 | [paper-notebook-physdiff-physics-guided-human-motion-diffusion-m](../../wiki/entities/paper-notebook-physdiff-physics-guided-human-motion-diffusion-m.md) | 1 | 2 | [deepmimic](../../wiki/methods/deepmimic.md) |
| 生成策略与运动条件 | [paper-notebook-taming-diffusion-probabilistic-models-for-charac](../../wiki/entities/paper-notebook-taming-diffusion-probabilistic-models-for-charac.md) | 0 | 2 | [diffusion-motion-generation](../../wiki/methods/diffusion-motion-generation.md)、[unified-multimodal-tokens](../../wiki/methods/unified-multimodal-tokens.md) |
| 生成策略与运动条件 | [paper-notebook-tedi-temporally-entangled-diffusion-for-long-ter](../../wiki/entities/paper-notebook-tedi-temporally-entangled-diffusion-for-long-ter.md) | 0 | 2 | [diffusion-motion-generation](../../wiki/methods/diffusion-motion-generation.md)、[deepmimic](../../wiki/methods/deepmimic.md) |
| 生成策略与运动条件 | [paper-pami-part-anchored-motion](../../wiki/entities/paper-pami-part-anchored-motion.md) | 0 | 2 | [motion-retargeting-gmr](../../wiki/methods/motion-retargeting-gmr.md)、[deepmimic](../../wiki/methods/deepmimic.md) |

## 可复验检查

已逐页验证 100 页通过当前真实出边规则，目标存在且不同；基线原文回放同一 `_check_entity_page`，100 页均检出原始缺项，作为负对照。四份清单各有 100 个唯一节点且两两无交集；豁免移除集合恰等于本批清单，元数据除日期外一致，健康度全部不变。

仓库根目录运行以下代码可复核本批清单、2 个不同目标和负对照（依赖 `requirements-ci-lite.txt`）：

```python
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import lint_wiki as lint

base = "6ceeed5cd17019b9836c4712cf56f071ea465bb6"
report = Path("docs/checklists/lint-outgoing-batch-04-100.md").read_text()
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

- `make ci-preflight` 完整通过：Wiki Lint 0 个阻塞错误、搜索 40/40、导出 14/14；Mermaid 7,974 图 / 8,199 文件 / 0 错误；图谱 0 孤儿节点，活动数据来源为完整 Git 历史。
- 定向 pytest 45 passed，覆盖冻结基线、交叉引用、健康度计算及 PR 正文健康度校验。
- 本页复验代码实跑：100/100 当前通过，100/100 基线负对照可检出。
- 无头 Chromium：OpenVSLAM（0→2）、COCO（1→2）、Betaflight（0→2）新增关联项展示及 5 个目标详情跳转通过；本地截图位于 `.cursor-artifacts/screenshots/batch4-*-related.png`。
- PR 正文覆盖全部 100 页实测健康度，并通过同一正文校验器；`git diff --check` 通过。

全库仍有 675 条信息型预警（明细按桶汇总），实际历史出边缺项 2,979 页。生成的图谱、统计、导出 JSON、截图和 lint 报告不入库。

## 后续

剩余 2,979 个实际历史缺项继续按主题评审。资料不足或关系无法核实的节点保留待办；后续数量以最新主分支实际 lint 为准。
