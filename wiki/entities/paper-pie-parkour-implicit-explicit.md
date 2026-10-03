---
type: entity
tags: [paper, quadruped, locomotion, perception, depth, implicit-explicit-learning, parkour]
status: complete
updated: 2026-10-03
arxiv: "2408.13740"
related:
  - ../methods/pie-perceptive-locomotion.md
  - ../methods/dreamwaq.md
  - ./dreamwaq-plus.md
  - ./paper-hiking-in-the-wild.md
sources:
  - ../../sources/papers/pie_arxiv_2408_13740.md
  - ../../sources/blogs/humanoid_motion_intelligence_day2_locomotion_motion_priors_2026_10_03.md
summary: "PIE：将深度历史与本体历史用于隐式—显式状态估计和策略学习，面向腿式机器人的感知跑酷。"
---

# PIE：腿式机器人的隐式—显式感知跑酷

**PIE**（*Parkour with Implicit-Explicit Learning Framework for Legged Robots*，[arXiv:2408.13740](https://arxiv.org/abs/2408.13740)）由浙江大学团队提出，研究深度感知与本体反馈如何共同支持腿式机器人穿越复杂地形。

## 一句话理解

让策略同时利用可解释的地形和运动估计，以及难以逐项标注的隐式环境线索。

## 方法要点

- 输入包括深度观测与本体历史。
- 显式估计分支学习与运动和地形有关的物理量；隐式分支压缩其他环境信息。
- 文章将 PIE 放在「提前看地形」路线中，与盲走 DreamWaQ 及点云历史路线 DreamWaQ++ 对照。

## 与方法页的关系

[PIE 感知行走方法页](../methods/pie-perceptive-locomotion.md)介绍机制；本页保留论文级标题、来源和独立阅读入口。

## 来源

- [arXiv:2408.13740](https://arxiv.org/abs/2408.13740)
- [Day 2 文章来源索引](../../sources/blogs/humanoid_motion_intelligence_day2_locomotion_motion_priors_2026_10_03.md)
