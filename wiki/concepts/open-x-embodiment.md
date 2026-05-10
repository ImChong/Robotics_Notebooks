---
type: concept
tags: [dataset, scaling, cross-embodiment, manipulation, community]
status: complete
updated: 2026-05-10
related:
  - ./foundation-policy.md
  - ./embodied-scaling-laws.md
  - ../methods/octo-model.md
  - ../methods/vla.md
sources:
  - ../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md
summary: "Open X-Embodiment（OXE）联合多机构把异构机器人演示数据规范化并开源，支撑跨本体规模化学习与通用策略预训练。"
---

# Open X-Embodiment（OXE）

## 一句话定义

**Open X-Embodiment**：面向机器人模仿学习的大规模跨机构、跨硬件形态数据集与基准管线，把多种机器人的演示统一到可比格式上，用于训练与评测「通用操作策略」。

## 为什么重要

它为 [Embodied Scaling Laws](./embodied-scaling-laws.md) 与 [Foundation Policy](./foundation-policy.md) 叙事提供了可公开核验的数据轴：在同一大混合上预训练的策略（如 [Octo](../methods/octo-model.md)）成为后续微调与对比实验的默认起点。

## 关联页面

- [Octo Model](../methods/octo-model.md)
- [Foundation Policy](./foundation-policy.md)

## 参考来源

- Padalkar et al., *Open X-Embodiment: Robotic Learning at Scale*, https://arxiv.org/abs/2310.08864
- [ted_xiao_embodied_three_eras_primary_refs.md](../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md)
