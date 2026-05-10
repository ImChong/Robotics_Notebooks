---
type: method
tags: [benchmark, evaluation, generalist-policy, distributed]
status: complete
updated: 2026-05-10
related:
  - ./octo-model.md
  - ../queries/robot-learning-three-eras-narrative.md
sources:
  - ../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md
summary: "RoboArena 通过多机构分布式真实机器人评测与盲测配对比较，为通用操作策略提供更难『刷榜』的外部验证框架。"
---

# RoboArena（评测基准）

## 一句话定义

**RoboArena**：面向通用机器人策略的分布式真实世界评测框架，聚合多个实验室在不同环境与任务上的配对对比实验，用排名反映策略泛化而非单一固定场景得分。

## 主要技术路线

- **分布式配对评测**：多实验室各自选题与环境，对两套策略做盲测配对，聚合排名而非单一榜单得分。
- **约束 overly broad 声称**：当 [Foundation Policy](../concepts/foundation-policy.md) / [VLA](./vla.md) 能力叙事扩张时，用真实多样性对抗中心化静态基准的过拟合。

## 关联页面

- [Octo Model](./octo-model.md)
- [Foundation Policy](../concepts/foundation-policy.md)

## 参考来源

- Atreya et al., *RoboArena: Distributed Real-World Evaluation of Generalist Robot Policies*, https://arxiv.org/abs/2506.18123
- 代码：https://github.com/robo-arena/roboarena
- [ted_xiao_embodied_three_eras_primary_refs.md](../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md)
