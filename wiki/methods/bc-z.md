---
type: method
tags: [imitation-learning, language-conditioned, zero-shot, google-robotics, manipulation]
status: complete
updated: 2026-11-07
related:
  - ./imitation-learning.md
  - ./dial-instruction-augmentation.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md
summary: "BC-Z 通过大规模真实世界模仿学习（示范 + 人工干预）与灵活任务条件（预训练语言嵌入或人类视频），在 100 多项真机任务上训练后，对 24 项未见任务零样本平均成功率 44%。"
---

# BC-Z

## 一句话定义

**BC-Z**：面向机器人操控的规模化模仿学习系统，结合交互式数据收集（遥操作示范 + 策略执行中的人工干预）与多种任务条件（预训练语言嵌入、人类执行任务的视频），在大量真实轨迹上训练单一策略并评测未见任务的迁移。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BC | Behavior Cloning | 将状态映射到动作的监督式模仿，易受分布偏移影响 |

## 主要技术路线

- **规模化模仿**：海量遥操作轨迹上的语言 / 视频等多形态任务条件行为克隆，面向零样本迁移到新指令。论文把数据收集扩展到 100 多项真机操控任务后，策略在 24 项训练中未见的任务上取得 44% 平均成功率，且这些任务没有任何机器人示范。
- **与通用策略衔接**：与 [Foundation Policy](../concepts/foundation-policy.md) 叙事一致，并为后续 [DIAL](./dial-instruction-augmentation.md)、[Octo](./octo-model.md) 等方法提供数据范式参照。

## 关联页面

- [Imitation Learning](./imitation-learning.md)
- [DIAL](./dial-instruction-augmentation.md)
- [Behavior Cloning](./behavior-cloning.md)
- [SayCan](../entities/paper-saycan.md)
- [具身三个时代叙事](../queries/robot-learning-three-eras-narrative.md)

## 参考来源

- Jang et al., *BC-Z: Zero-Shot Task Generalization with Robotic Imitation Learning*, https://arxiv.org/abs/2202.02005
- [ted_xiao_embodied_three_eras_primary_refs.md](../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md)
