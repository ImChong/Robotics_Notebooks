---
type: entity
tags: [paper, humanoid, navigation, dynamic-obstacles, safety, reinforcement-learning]
status: complete
updated: 2026-10-03
arxiv: "2609.38873"
related:
  - ../tasks/locomotion.md
  - ../tasks/navigation.md
  - ../concepts/safe-reinforcement-learning.md
sources:
  - ../../sources/blogs/humanoid_motion_intelligence_day2_locomotion_motion_priors_2026_10_03.md
summary: "DODGER（arXiv:2609.38873）：通过安全引导的强化学习在人群等动态障碍中生成导航速度命令，再交由低层行走控制执行。"
---

# DODGER：动态障碍中的安全引导导航

**DODGER**（*Safety-Guided Reinforcement Learning for Robot Navigation Among Dynamic Obstacles*，[arXiv:2609.38873](https://arxiv.org/abs/2609.38873)，[项目页](https://psh0823.github.io/dodger-homepage/)）研究机器人如何在人群等动态障碍中选择安全导航动作。

## 一句话理解

高层根据目标和移动障碍决定速度指令，低层行走控制器负责把指令变成稳定身体运动。

## 方法要点

- 以关系图组织机器人、目标和动态行人信息。
- 训练阶段使用控制屏障函数和安全引导信号，部署时由学习策略输出导航命令。
- 该接口把高层路线选择与低层平衡控制连接起来。

## 来源

- [arXiv:2609.38873](https://arxiv.org/abs/2609.38873)
- [项目页](https://psh0823.github.io/dodger-homepage/)
- [Day 2 文章来源索引](../../sources/blogs/humanoid_motion_intelligence_day2_locomotion_motion_priors_2026_10_03.md)
