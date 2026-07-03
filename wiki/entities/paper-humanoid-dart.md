---
type: entity
tags: [paper, loco-manip-contact-survey, humanoid, loco-manipulation, diffusion, reinforcement-learning, goal-conditioned, unitree-g1]
status: complete
updated: 2026-07-03
arxiv: "2606.26855"
venue: "2026 · arXiv"
summary: "Humanoid-DART 以扩散轨迹生成 + RL 跟踪交替自举，从稀疏示范扩展目标条件 loco-manip 行为，G1 真机推/踢/搬运验证。"
related:
  - ../overview/loco-manip-contact-technology-map.md
  - ../overview/loco-manip-contact-category-03-generative-data.md
  - ../tasks/loco-manipulation.md
  - ../methods/reinforcement-learning.md
sources:
  - ../../sources/papers/humanoid_dart_arxiv_2606_26855.md
  - ../../sources/blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md
---

# Humanoid-DART

**Humanoid-DART**（arXiv:2606.26855，MPI-IS 等）收录于 [具身智能研究室 · Loco-Manip 接触专题](https://mp.weixin.qq.com/s/UjShbwl8p1h9ukymfiRNaw) **03 生成式补数** 组：用 **扩散 + RL 跟踪** 自举扩展 loco-manip **目标空间**。

## 一句话定义

从稀疏专家示范启动，交替训练 **目标条件扩散运动生成器** 与 **轨迹跟踪 RL 策略**，在轨迹空间结构化探索并逐步扩大可解 loco-manip 目标分布；**Unitree G1** 真机部署推、踢、pick-and-place 等技能。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DART | Diffusion-guided Augmentation through Relabeling and Tracking | 论文方法缩写 |
| RL | Reinforcement Learning | 低层跟踪扩散生成的目标轨迹 |
| Loco-Manip | Loco-Manipulation | 全身行走与物体交互耦合任务 |
| IL | Imitation Learning | 与纯模仿对比，本文强调自举扩目标空间 |
| OCP | Optimal Control Problem | 非本文主路线，对照 WOLF-VLA |

## 为什么重要

- **稀疏示范可扩展：** 接触丰富 loco-manip 目标空间连续且组合爆炸，纯堆演示难以覆盖。
- **生成-跟踪解耦：** 扩散负责 **扩轨迹分布**，RL 跟踪提供 **稠密任务无关监督**。
- **策展定位：** 扩散模型角色是 **扩大可训练目标空间**，须与物理跟踪验证并用。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 分组 | Loco-Manip 接触专题 · 03 生成式补数 |
| 原文题目 | Humanoid-DART: Humanoid Loco-Manipulation using Diffusion-guided Augmentation through Relabeling and Tracking |
| 机构 | Max Planck Institute for Intelligent Systems 等 |
| 论文/项目 | <https://arxiv.org/abs/2606.26855> |

## 评测速览（索引级）

- **真机验证：** Unitree G1 上完成 **推、踢、pick-and-place** 等目标条件 loco-manip 技能。
- **方法定位：** 扩散生成器与 RL 跟踪交替自举，从稀疏示范逐步扩大可解目标分布。
- 定量指标（目标覆盖率 / 成功率）以 arXiv:2606.26855 为准，索引级页暂不展开。

## 与其他页面的关系

- 分类 hub：[loco-manip-contact-category-03-generative-data.md](../overview/loco-manip-contact-category-03-generative-data.md)
- 姊妹：[GRAIL](paper-grail.md)、[HumanoidMimicGen](paper-humanoidmimicgen.md)

## 参考来源

- [humanoid_dart_arxiv_2606_26855.md](../../sources/papers/humanoid_dart_arxiv_2606_26855.md)
- [wechat_embodied_ai_lab_loco_manip_contact_survey.md](../../sources/blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md)

## 推荐继续阅读

- [接触五段链路技术地图](../overview/loco-manip-contact-technology-map.md)
- [扩散运动生成](../methods/diffusion-motion-generation.md)
