---
type: method
tags: [imitation-learning, language-conditioned, zero-shot, google-robotics, manipulation]
status: complete
updated: 2026-05-10
related:
  - ./imitation-learning.md
  - ./dial-instruction-augmentation.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md
summary: "BC-Z 通过大规模真实世界模仿学习与灵活任务条件（语言等），展示桌面操控策略对新任务的零样本泛化潜力。"
---

# BC-Z

## 一句话定义

**BC-Z**：面向机器人操控的规模化模仿学习系统，结合共享遥操作收集与多种任务条件（自然语言、示范视频等），在大量真实轨迹上训练单一策略并评测未见任务的迁移。

## 主要技术路线

- **规模化模仿**：海量遥操作轨迹上的语言 / 视频等多形态任务条件行为克隆，面向零样本迁移到新指令。
- **与通用策略衔接**：与 [Foundation Policy](../concepts/foundation-policy.md) 叙事一致，并为后续 [DIAL](./dial-instruction-augmentation.md)、[Octo](./octo-model.md) 等方法提供数据范式参照。

## 关联页面

- [Imitation Learning](./imitation-learning.md)
- [DIAL](./dial-instruction-augmentation.md)

## 参考来源

- Jang et al., *BC-Z: Zero-Shot Task Generalization with Robotic Imitation Learning*, https://arxiv.org/abs/2202.02005
- [ted_xiao_embodied_three_eras_primary_refs.md](../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md)
