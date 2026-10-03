---
type: entity
tags: [paper, humanoid, badminton, motion-prior, skill-learning, sim2real]
status: complete
updated: 2026-10-03
arxiv: "2609.31840"
related:
  - ./paper-notebook-humanoid-whole-body-badminton-via-multi-stage-re.md
  - ./paper-notebook-learning-human-like-badminton-skills-for-humanoi.md
  - ../tasks/humanoid-locomotion.md
sources:
  - ../../sources/blogs/humanoid_motion_intelligence_day2_locomotion_motion_priors_2026_10_03.md
summary: "Humanoid Badminton（arXiv:2609.31840）：用有限的人类击球动作学习动态球拍技能，并按来球状态在线选择和组合技能。"
---

# Humanoid Badminton：从有限人类动作学习动态球拍技能

**Humanoid Badminton**（*Learning Dynamic Racket Skills from Limited Human Motion Data*，[arXiv:2609.31840](https://arxiv.org/abs/2609.31840)，[项目页](https://sunlight02.github.io/humanoid-badminton/)）研究人形机器人如何利用有限的人类击球动作，在来球运动中选择并执行多种回球技能。

## 一句话理解

动作示范提供身体协调经验，球的位置和速度决定何时调用哪项技能。

## 方法要点

- 通过击球事件扩增有限的动作示范。
- 低层学习连续技能表示，高层按来球状态提供技能提示。
- 文章报告 Unitree G1 实机展示正手、反手、跳起回球及人机对打。

## 与其他羽毛球工作区分

本页对应 *Learning Dynamic Racket Skills from Limited Human Motion Data*（arXiv:2609.31840）。已有 [Humanoid Whole-Body Badminton via an Annealed Reinforcement Learning Curriculum](./paper-notebook-humanoid-whole-body-badminton-via-multi-stage-re.md) 和 [Learning Human-Like Badminton Skills for Humanoid Robots](./paper-notebook-learning-human-like-badminton-skills-for-humanoi.md) 是不同论文，分别保留独立页面。

## 来源

- [arXiv:2609.31840](https://arxiv.org/abs/2609.31840)
- [项目页](https://sunlight02.github.io/humanoid-badminton/)
- [Day 2 文章来源索引](../../sources/blogs/humanoid_motion_intelligence_day2_locomotion_motion_priors_2026_10_03.md)
