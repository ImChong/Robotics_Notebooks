---
type: method
tags: [benchmark, evaluation, generalist-policy, distributed]
status: complete
updated: 2026-06-20
related:
  - ./octo-model.md
  - ../entities/paper-oscar.md
  - ../entities/paper-shenlan-wm-15-worldgym.md
  - ../queries/robot-learning-three-eras-narrative.md
sources:
  - ../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md
summary: "RoboArena 通过多机构分布式真实机器人评测与盲测配对比较，为通用操作策略提供更难『刷榜』的外部验证框架。"
---

# RoboArena（评测基准）

## 一句话定义

**RoboArena**：面向通用机器人策略的分布式真实世界评测框架，聚合多个实验室在不同环境与任务上的配对对比实验，用排名反映策略泛化而非单一固定场景得分。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作多模态基础策略方向 |

## 主要技术路线

- **分布式配对评测**：多实验室各自选题与环境，对两套策略做盲测配对，聚合排名而非单一榜单得分。
- **约束 overly broad 声称**：当 [Foundation Policy](../concepts/foundation-policy.md) / [VLA](./vla.md) 能力叙事扩张时，用真实多样性对抗中心化静态基准的过拟合。
- **WM 虚拟评测代理**：动作条件视频世界模型可在 rollout 上估计策略成功率并与真机排名对照——[OSCAR](../entities/paper-oscar.md) 在 RoboArena 七策略池报告 Pearson **ρ +0.750**；同类路线见 [WorldGym](../entities/paper-shenlan-wm-15-worldgym.md)。

## 关联页面

- [Octo Model](./octo-model.md)
- [OSCAR](../entities/paper-oscar.md) — 跨具身骨架条件 WM 虚拟策略评估
- [Foundation Policy](../concepts/foundation-policy.md)

## 参考来源

- Atreya et al., *RoboArena: Distributed Real-World Evaluation of Generalist Robot Policies*, https://arxiv.org/abs/2506.18123
- 代码：https://github.com/robo-arena/roboarena
- [ted_xiao_embodied_three_eras_primary_refs.md](../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md)
