---
type: method
tags: [imitation-learning, unsupervised, latent-plan, manipulation, google-robotics]
status: complete
updated: 2026-05-10
related:
  - ./imitation-learning.md
  - ./behavior-cloning.md
  - ./her.md
sources:
  - ../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md
summary: "Learning from Play（Play-LMP）从非任务化的遥操作『玩耍』轨迹中学习潜在规划结构，再用目标条件策略完成指定任务。"
---

# Learning from Play（Play-LMP）

## 一句话定义

**Learning from Play**：利用大量未标注任务边界的机器人交互片段（play），学习潜在「计划」表征与条件策略，从而在少量任务标注下完成复杂操控序列。

## 主要技术路线

- **非任务化数据**：从大量「玩耍」遥操作片段中学习，不要求逐步语言标注（与 [Reward Design](../concepts/reward-design.md) 稀疏奖励设定对比）。
- **潜在规划 + 目标条件策略**：从回放窗口抽取片段，学习潜在技能表征后再合成任务行为；思想上接近 [HER](./her.md) 的事后重标记，但面向操控语义。

## 关联页面

- [Imitation Learning](./imitation-learning.md)
- [HER](./her.md)

## 参考来源

- Lynch et al., *Learning Latent Plans from Play*, https://arxiv.org/abs/1903.01973
- 项目页：https://learning-from-play.github.io/
- [ted_xiao_embodied_three_eras_primary_refs.md](../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md)
