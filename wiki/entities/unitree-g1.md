---
type: entity
tags: [hardware, humanoid, platform, unitree]
status: complete
updated: 2026-04-21
related:
  - ./humanoid-robot.md
  - ./unitree.md
  - ../roadmaps/humanoid-control-roadmap.md
sources:
  - ../../sources/papers/humanoid_hardware.md
summary: "Unitree G1 是一款由宇树科技推出的入门级教育科研用人形机器人，以其极高的性价比、高集成度以及对仿真学习框架的良好支持而备受关注。"
---

# Unitree G1 (人形机器人)

**Unitree G1** 是宇树科技 (Unitree) 在 H1 之后推出的一款量产型、高性价比的人形机器人平台。其设计初衷是降低人形机器人研究的门槛，使其能够大规模进入实验室、高校和家庭场景。

## 核心特性

1. **高集成度与便携性**：G1 的体型较 H1 更小，支持折叠收纳，单人即可搬运和部署。
2. **丰富的感知方案**：集成了 3D 激光雷达 (LiDAR) 和深度相机，原生支持 [地形自适应](../concepts/terrain-adaptation.md)。
3. **力控能力**：全关节支持高带宽力控，极其适配 [WBC](../concepts/whole-body-control.md) 与强化学习。
4. **生态支持**：完美适配 [Isaac Lab](../entities/nvidia-omniverse.md)、[robot_lab](../entities/robot-lab.md) 和 `legged_gym`；同时作为 NVIDIA [MotionBricks](../methods/motionbricks.md) 生成式运动框架的首批验证硬件。
5. **高效数据生成**：支持 [CLAW](../methods/claw.md) 等合成数据管线，通过网页交互快速生成带语言标签的全身动作数据。
6. **足球技能研究**：作为 [PAiD](../methods/paid-framework.md) 框架的主要实验平台，证明了其在执行类人化踢球动作方面的卓越物理特性。
7. **敏捷技能切换**：[Switch](../methods/switch-framework.md) 框架在 G1 上实现了 100% 的跨技能切换成功率与极强的抗扰动能力。
8. **视频生成驱动的零样本控制**：作为 [ExoActor](../methods/exoactor.md)（BAAI, 2026）的端到端验证平台，把第三人称视频生成 + 通用动作跟踪整合为零真实数据的交互行为生成系统。

## 在具身智能中的作用

G1 的出现极大地加速了大规模数据的采集。由于其成本低廉，研究者可以构建“机器人机房（Robot Farms）”，利用海量实体机器人通过 [自动化标注](../methods/auto-labeling-pipelines.md) 或利用 [CLAW](../methods/claw.md) 等仿真合成手段快速生成训练基础策略（Foundation Policies）所需的真实数据。

## 关联页面
- [smp](../methods/smp.md) (基于得分匹配的运动先验，已在 G1 完成验证)
- [人形机器人 (Humanoid Robot)](./humanoid-robot.md)
- [Unitree 品牌主页](./unitree.md)
- [robot_lab (IsaacLab 扩展框架)](./robot-lab.md)
- [PAiD Framework (足球技能学习)](../methods/paid-framework.md)
- [Humanoid Soccer (足球任务)](../tasks/humanoid-soccer.md)
- [CLAW (宇树 G1 全身动作数据生成管线)](../methods/claw.md)
- [Humanoid Control Roadmap](../roadmaps/humanoid-control-roadmap.md)
- [ExoActor](../methods/exoactor.md) — G1 上的视频生成驱动的零样本交互控制系统。

## 参考来源
- Unitree G1 官方规格书。
- [sources/papers/exoactor.md](../../sources/papers/exoactor.md) — ExoActor 在 G1 上的端到端实现。
