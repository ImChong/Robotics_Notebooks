---
type: entity
tags: [paper, humanoid, soccer, locomotion, multi-skill, reinforcement-learning]
status: complete
updated: 2026-10-03
arxiv: "2609.38852"
related:
  - ../tasks/humanoid-soccer.md
  - ../tasks/humanoid-locomotion.md
  - ./paper-skillx-humanoid-soccer.md
  - ./paper-banana-kick-humanoid-soccer.md
sources:
  - ../../sources/blogs/humanoid_motion_intelligence_day2_locomotion_motion_priors_2026_10_03.md
summary: "Locomotion-Grounded Humanoid Soccer（arXiv:2609.38852）：以通用行走策略为基础，逐步加入多方向踢球技能并训练技能切换。"
---

# Locomotion-Grounded Humanoid Soccer：以行走为基础的多方向踢球

**Locomotion-Grounded Humanoid Soccer**（*Task-Gated Reinforcement Learning of a Multi-Directional Kicking Library*，[arXiv:2609.38852](https://arxiv.org/abs/2609.38852)）研究如何让人形在行走与踢球间切换，并保留不同方向的踢球能力。

## 一句话理解

先训练可响应速度命令的行走策略，再把多个踢球技能接入同一可控运动状态。

## 方法要点

- 以通用命令条件行走策略作为运动底座。
- 用任务门控加入多方向踢球技能，并让技能共享可控的行走状态。
- 文章报告 Unitree G1 上七种重定向踢球动作与真机验证；详细指标应以论文为准。

## 与相邻工作

- [SkillX](./paper-skillx-humanoid-soccer.md)研究停球、盘带和射门的统一策略及技能交接。
- [Banana Kick](./paper-banana-kick-humanoid-soccer.md)针对脚球接触与弧线球奖励优化。

## 来源

- [arXiv:2609.38852](https://arxiv.org/abs/2609.38852)
- [Day 2 文章来源索引](../../sources/blogs/humanoid_motion_intelligence_day2_locomotion_motion_priors_2026_10_03.md)
