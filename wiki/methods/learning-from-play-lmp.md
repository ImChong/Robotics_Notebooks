---
type: method
tags: [imitation-learning, unsupervised, latent-plan, manipulation, google-robotics]
status: complete
updated: 2026-11-07
related:
  - ./imitation-learning.md
  - ./behavior-cloning.md
  - ./her.md
sources:
  - ../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md
summary: "Learning from Play（Play-LMP）从非任务化的遥操作『玩耍』轨迹中自监督学习潜在规划空间，再用目标条件策略完成指定目标；在 18 项仿真桌面视觉操控任务上优于逐任务训练的专家策略。"
---

# Learning from Play（Play-LMP）

## 一句话定义

**Learning from Play**：利用大量未标注任务边界的机器人交互片段（play），学习潜在「计划」表征与条件策略，从而在**无需任务标签**的情况下，给定目标（如目标图像）即可完成操控任务。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Reward | Reward Function | 塑造强化学习策略行为的标量反馈 |

## 主要技术路线

- **非任务化数据**：从大量「玩耍」遥操作片段中学习，不要求逐步语言标注（与 [Reward Design](../concepts/reward-design.md) 稀疏奖励设定对比）。
- **潜在规划 + 目标条件策略**：从回放窗口抽取片段，学习潜在技能表征后再合成任务行为；思想上接近 [HER](./her.md) 的事后重标记，但面向操控语义。
- **论文结论**：在仿真桌面环境的 18 项视觉操控任务上，单一 play 训练模型优于逐任务训练的专家策略，对扰动更鲁棒；潜在空间在从未给出任务标签的情况下自发按功能任务组织。

## 关联页面

- [Imitation Learning](./imitation-learning.md)
- [HER](./her.md)
- [Behavior Cloning](./behavior-cloning.md)
- [Play2Perfect](../entities/paper-play2perfect.md)（灵巧「玩耍」预训练，RL 路线）
- [Affordance Learning from Play](../entities/paper-aff-2203-00352-affordance-learning-from-play-for-sample-efficient-polic.md)

## 参考来源

- Lynch et al., *Learning Latent Plans from Play*, https://arxiv.org/abs/1903.01973
- 项目页：https://learning-from-play.github.io/
- [ted_xiao_embodied_three_eras_primary_refs.md](../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md)
