---
type: concept
tags: [dataset, scaling, cross-embodiment, manipulation, community]
status: complete
updated: 2026-07-31
related:
  - ../entities/paper-open-x-embodiment.md
  - ../queries/contact-wrench-closed-loop.md
  - ./foundation-policy.md
  - ./embodied-scaling-laws.md
  - ../methods/octo-model.md
  - ../methods/vla.md
  - ../entities/paper-topreward.md
  - ../entities/paper-data-pyramid-embodied-manipulation.md
sources:
  - ../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md
  - ../../sources/papers/topreward_arxiv_2602_19313.md
  - ../../sources/papers/data_pyramid_embodied_manipulation_arxiv_2607_24744.md
summary: "Open X-Embodiment（OXE）联合多机构把异构机器人演示数据规范化并开源，支撑跨本体规模化学习与通用策略预训练。"
---

# Open X-Embodiment（OXE）

## 一句话定义

**Open X-Embodiment**：面向机器人模仿学习的大规模跨机构、跨硬件形态数据集与基准管线，把多种机器人的演示统一到可比格式上，用于训练与评测「通用操作策略」。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| OXE | Open X-Embodiment | 跨形态机器人的大规模操作数据集 |
| RT | Robotics Transformer | 早期通用操作策略系列，常与 OXE 数据叙事并列 |
| VLA | Vision-Language-Action | 在 OXE 等混合数据上预训练的通用策略形态 |

## 为什么重要

它为 [Embodied Scaling Laws](./embodied-scaling-laws.md) 与 [Foundation Policy](./foundation-policy.md) 叙事提供了可公开核验的数据轴：在同一大混合上预训练的策略（如 [Octo](../methods/octo-model.md)）成为后续微调与对比实验的默认起点。

## 关联页面

- [Query：接触力旋量闭环知识链](../queries/contact-wrench-closed-loop.md) — 跨本体操作数据训练的策略，其接触执行环节由本链托底
- [Octo Model](../methods/octo-model.md)
- [Foundation Policy](./foundation-policy.md)
- [TOPReward](../entities/paper-topreward.md) — 在 OXE 39 数据集上评测零样本进度奖励（Mean VOC）
- [具身数据金字塔综述](../entities/paper-data-pyramid-embodied-manipulation.md) — 把 OXE 定位为五层金字塔真机层的聚合代表，并给出跨层数据选型坐标系

## 参考来源

- Padalkar et al., *Open X-Embodiment: Robotic Learning at Scale*, https://arxiv.org/abs/2310.08864
- [ted_xiao_embodied_three_eras_primary_refs.md](../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md)
- [sources/papers/topreward_arxiv_2602_19313.md](../../sources/papers/topreward_arxiv_2602_19313.md) — OXE 进度估计评测轴
