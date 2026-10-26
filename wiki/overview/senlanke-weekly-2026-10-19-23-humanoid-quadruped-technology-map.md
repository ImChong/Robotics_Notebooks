---
type: overview
tags: [survey, humanoid, quadruped, legged-robot, senlanke]
status: complete
updated: 2026-10-26
related:
  - ../tasks/locomotion.md
  - ../tasks/humanoid-locomotion.md
  - ../methods/reinforcement-learning.md
  - senlanke-weekly-2026-10-19-23-manipulation-technology-map.md
sources:
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-19_23.md
summary: "senlanke 具身运控lab 10-19–10-23 人形/四足周报：39 条论文条目，每条对应唯一 arXiv 详情节点（新建 22 / 复用 17）。"
---

# senlanke 周报索引：10-19–10-23 人形/四足（39 篇）

> **本页定位**：为 [【10.19-10.23前沿论文动态】人形／四足39篇](https://mp.weixin.qq.com/s/DBARqsgZl-RdP1LYvAS2rA) 提供按周报分栏组织的 **39 个条目 → 唯一 arXiv 详情节点**索引。★ 为周报标注的重点论文。

## 一句话观点

本页只按周报原有分栏（人形机器人；四足机器人；轮足与通用腿式机器人）列出条目；论文间的横向优劣不在此判断，请按 arXiv 进各详情节点，条目说明来自周报「问题/创新点」与 arXiv 摘要页。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| WBC | Whole-Body Control | 全身控制 |
| MPC | Model Predictive Control | 模型预测控制 |
| LiDAR | Light Detection and Ranging | 激光雷达 |
| IMU | Inertial Measurement Unit | 惯性测量单元 |

## 论文索引

### 人形机器人（30）

| # | 论文节点 | arXiv | 状态 |
|---|----------|-------|------|
| 1 | [SWAP：测试时可控的人形全身避障](../entities/paper-swap-humanoid-collision-avoidance.md) ★ | [2610.25711](https://arxiv.org/abs/2610.25711) | 复用 |
| 2 | [MotorDev-RL：利用人形机器人验证人类运动发育量表](../entities/paper-motor-development-scales-humanoid-curriculum.md) | [2610.25701](https://arxiv.org/abs/2610.25701) | 新建 |
| 3 | [OmniInteract：双人形全身交互控制](../entities/paper-omniinteract-dual-humanoids.md) | [2610.26043](https://arxiv.org/abs/2610.26043) | 复用 |
| 4 | [G1Exo 组合操作：基于同构遥操作与独立技能路由的组合式人形操作](../entities/paper-isomorphic-teleop-skill-routing-humanoid.md) | [2610.26169](https://arxiv.org/abs/2610.26169) | 新建 |
| 5 | [特征级示范学习：基于特征级感知与提示式目标检测的人形机器人数据高效示范学习](../entities/paper-feature-level-lfd-prompt-detection-nao.md) | [2610.26170](https://arxiv.org/abs/2610.26170) | 新建 |
| 6 | [HAF（Humanoid Adaptation Framework）](../entities/paper-haf-humanoid-vla-adaptation.md) | [2610.26584](https://arxiv.org/abs/2610.26584) | 复用 |
| 7 | [Ropedia W0：跨本体运动-操作世界模型](../entities/paper-ropedia-w0.md) | [2610.26855](https://arxiv.org/abs/2610.26855) | 复用 |
| 8 | [面向孤独症教育人形机器人的视觉任务建模与自动规划](../entities/paper-autism-education-humanoid-visual-task-planning.md) | [2610.27162](https://arxiv.org/abs/2610.27162) | 新建 |
| 9 | [EmbodyHOI（面向人形灵巧操作的具身感知 HOI 生成）](../entities/paper-embodyhoi.md) | [2610.27242](https://arxiv.org/abs/2610.27242) | 复用 |
| 10 | [Traversal Tubes：面向双足人形姿态感知导航的自适应通行管道](../entities/paper-adaptive-traversal-tubes-bipedal-navigation.md) | [2610.27688](https://arxiv.org/abs/2610.27688) | 新建 |
| 11 | [Fast Retargeting：面向规模化人形学习的快速动作重定向流程](../entities/paper-fast-motion-retargeting-pipeline-humanoid.md) | [2610.28470](https://arxiv.org/abs/2610.28470) | 新建 |
| 12 | [SuperMimic：用一个可微世界模型扩展人形全身控制](../entities/paper-supermimic.md) ★ | [2610.28480](https://arxiv.org/abs/2610.28480) | 复用 |
| 13 | [面向人形—人交互的 VLM 引导接触重定向与生成式伙伴建模](../entities/paper-vlm-contact-retargeting-partner-modeling.md) ★ | [2610.28523](https://arxiv.org/abs/2610.28523) | 新建 |
| 14 | [Warping the Workspace：工作空间扭曲：通过可调触达映射扩展人形遥操作视觉范围](../entities/paper-warping-workspace-adjustable-reach-teleop.md) | [2610.28712](https://arxiv.org/abs/2610.28712) | 新建 |
| 15 | [One Policy Many Styles：一个策略、多种风格：可引导人形控制应在何处聚合偏好？](../entities/paper-one-policy-many-styles-preference-aggregation.md) | [2610.28925](https://arxiv.org/abs/2610.28925) | 新建 |
| 16 | [SMR：从人体动作语义学习流式人形动作重定向](../entities/paper-smr-streaming-motion-retargeting.md) | [2610.29169](https://arxiv.org/abs/2610.29169) | 新建 |
| 17 | [Watch Your Step（arXiv:2610.29612）](../entities/paper-watch-your-step.md) ★ | [2610.29612](https://arxiv.org/abs/2610.29612) | 复用 |
| 18 | [HuMiT：基于紧凑参考跟踪的低延迟人形全身遥操作](../entities/paper-humit-low-latency-whole-body-teleop.md) | [2610.29903](https://arxiv.org/abs/2610.29903) | 新建 |
| 19 | [SP-HLM：稀疏姿态驱动的人形全身控制](../entities/paper-sp-hlm.md) ★ | [2610.30044](https://arxiv.org/abs/2610.30044) | 复用 |
| 20 | [RISE：感知式人形起身的测试时运动引导](../entities/paper-rise-humanoid-fall-recovery.md) ★ | [2610.30061](https://arxiv.org/abs/2610.30061) | 复用 |
| 21 | [Moon Walk Transfer：在地球训练的人形控制器能在月球行走吗？重力、地形与土壤迁移研究](../entities/paper-earth-trained-humanoid-controllers-moon-transfer.md) ★ | [2610.30080](https://arxiv.org/abs/2610.30080) | 新建 |
| 22 | [MPBRS：九种人形机器人共享奖励配置](../entities/paper-mpbrs-humanoid-reward-shaping.md) | [2610.30334](https://arxiv.org/abs/2610.30334) | 复用 |
| 23 | [PhysibleMDM：推理时优化动作扩散噪声以提升物理可行性](../entities/paper-physiblemdm.md) | [2610.30500](https://arxiv.org/abs/2610.30500) | 复用 |
| 24 | [STR：面向人形移动操作的时空偏好动作重定向](../entities/paper-str-spatiotemporal-retargeting.md) | [2610.30540](https://arxiv.org/abs/2610.30540) | 复用 |
| 25 | [STAMP：Robot Motion Transition with a Human Motion Prior](../entities/paper-stamp.md) | [2610.30817](https://arxiv.org/abs/2610.30817) | 复用 |
| 26 | [LocoManiLab：人形移动操作全栈测试床](../entities/paper-locomanilab.md) | [2610.30876](https://arxiv.org/abs/2610.30876) | 复用 |
| 27 | [WholeBodyMem：人形移动操作的目标前历史记忆](../entities/paper-wholebodymem.md) | [2610.31166](https://arxiv.org/abs/2610.31166) | 复用 |
| 28 | [ACP：注意力式接触规划人形移动（arXiv:2610.31608）](../entities/paper-acp-humanoid-mobility.md) ★ | [2610.31608](https://arxiv.org/abs/2610.31608) | 复用 |
| 29 | [Humanoid SRL](../entities/paper-humanoid-srl.md) | [2610.25628](https://arxiv.org/abs/2610.25628) | 复用 |
| 30 | [PHD-Hand：采用三驱动自由度连杆手指的可打印人形灵巧手](../entities/paper-phd-hand-printable-humanoid-dexterous-hand.md) | [2610.29030](https://arxiv.org/abs/2610.29030) | 新建 |
### 四足机器人（4）

| # | 论文节点 | arXiv | 状态 |
|---|----------|-------|------|
| 1 | [Legged-Aerial SCCM：基于分离控制收缩度量的多模态四足空地机器人设计与控制](../entities/paper-multimodal-legged-aerial-robot-sccm.md) | [2610.27315](https://arxiv.org/abs/2610.27315) | 新建 |
| 2 | [使用 Boston Dynamics Spot 实现自主砌砖](../entities/paper-autonomous-bricklaying-boston-dynamics-spot.md) | [2610.27472](https://arxiv.org/abs/2610.27472) | 新建 |
| 3 | [LocoPilot：面向移动操作策略学习的单人双向遥操作](../entities/paper-locopilot-bilateral-teleop-loco-manipulation.md) | [2610.27998](https://arxiv.org/abs/2610.27998) | 新建 |
| 4 | [月球重力跳跃：面向月球重力四足跳跃的端到端强化学习](../entities/paper-quadruped-jumping-lunar-gravity-rl.md) ★ | [2610.29586](https://arxiv.org/abs/2610.29586) | 新建 |
### 轮足与通用腿式机器人（5）

| # | 论文节点 | arXiv | 状态 |
|---|----------|-------|------|
| 1 | [SkateNav：通过自适应滑行学习被动轮四足机器人的高能效导航](../entities/paper-skatenav-passive-wheel-quadruped-skating.md) ★ | [2610.25962](https://arxiv.org/abs/2610.25962) | 新建 |
| 2 | [Wheel Uncertainty LIO：面向四足轮腿机器人的轮运动不确定性学习与退化感知激光—惯性里程计](../entities/paper-wheel-motion-uncertainty-degeneracy-aware-lio.md) ★ | [2610.26550](https://arxiv.org/abs/2610.26550) | 新建 |
| 3 | [RoboMMT：调制式表达与交互机器人动作的稳定跟踪学习](../entities/paper-robommt-modulated-expressive-motion-tracking.md) | [2610.26414](https://arxiv.org/abs/2610.26414) | 新建 |
| 4 | [Thrust-Assisted Trotting：基于实时 QP-MPC 的仿生推力辅助空地双足小跑](../entities/paper-thrust-assisted-trotting-legged-aerial-qp-mpc.md) | [2610.26810](https://arxiv.org/abs/2610.26810) | 新建 |
| 5 | [A2I-Calib++：面向腿式机器人全身多 IMU 的运动学驱动抗噪主动标定](../entities/paper-a2i-calib-plus-multi-imu-legged-calibration.md) | [2610.29438](https://arxiv.org/abs/2610.29438) | 新建 |

## 关联页面

- [locomotion](../tasks/locomotion.md)
- [humanoid-locomotion](../tasks/humanoid-locomotion.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)
- [Manipulation周报索引（同期另一篇）](senlanke-weekly-2026-10-19-23-manipulation-technology-map.md)

## 参考来源

- [wechat_senlanke_weekly_humanoid_quadruped_2026-10-19_23.md](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-19_23.md)
- [原文链接](https://mp.weixin.qq.com/s/DBARqsgZl-RdP1LYvAS2rA)
