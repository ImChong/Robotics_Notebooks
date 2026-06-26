---
type: entity
tags: [agibot, vla, manipulation, open-source]
status: complete
updated: 2026-06-26
related:
  - ../overview/agibot-june-2026-release-technology-map.md
  - ../overview/agibot-release-category-04-execution-vla.md
  - ./genie-sim-3.md
  - ../methods/vla.md
  - ../concepts/behavior-foundation-model.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_agibot_june_2026_release.md
summary: "GO-2 是智元开源的执行基座 VLA：以动作思维链与异步双系统跨越语义–运动鸿沟，在 LIBERO、VLABench 与 Genie Sim 3.0 等基准上报告领先结果。"
---

# GO-2（智元执行基座）

**GO-2**（arXiv:[2601.11404](https://arxiv.org/abs/2601.11404)，项目页：<https://libra-vla.github.io/>）是智元在 [2026-06 发布地图](../overview/agibot-june-2026-release-technology-map.md) 中的 **语义到动作执行基座**。文内将其定位为跨越 **语义–运动鸿沟** 的 VLA：理解任务后须把规划落实为可执行、可修正的底层动作。

## 一句话定义

**动作思维链 + 异步双系统 VLA**——把推理放进动作空间生成结构化序列，并由快慢系统分工完成规划与高频视觉对齐执行。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| CoT | Chain of Thought | 思维链，逐步推理再作答 |

## 为什么重要

- **接口问题：** 真机失败常来自视觉偏差、轨迹漂移、接触不稳——GO-2 把 **可执行性** 写进模型设计。
- **动作思维链：** 在动作空间生成结构化动作序列，使规划更接近可执行形式。
- **异步双系统：** 低频慢系统产出高层动作意图；高频快系统据视觉观测持续对齐、修正与执行。
- **评测：** 文内称在 LIBERO、LIBERO-Plus、VLABench、Genie Sim 3.0 Benchmark 领先；Genie Sim Sim2Real 评测中仅用仿真数据训练后真机平均成功率 **82.9%**（对照 π0.5 **77.5%**，以论文/材料为准）。

## 与 GO-1 / 其他 VLA

- 论文与生态中另有 **GO-1** 等前代/并行产品线（如 [GreenVLA](../../sources/papers/greenvla_arxiv_2602_00919.md) 对照表）；本页仅覆盖 **GO-2** 在 2026-06 发布语境下的定位。

## 关联页面

- [语义执行分类 hub](../overview/agibot-release-category-04-execution-vla.md)
- [VLA](../methods/vla.md)
- [Genie Sim 3.0](./genie-sim-3.md)

## 参考来源

- [wechat_embodied_ai_lab_agibot_june_2026_release.md](../../sources/blogs/wechat_embodied_ai_lab_agibot_june_2026_release.md)

## 推荐继续阅读

- [GO-2 论文](https://arxiv.org/abs/2601.11404)
- [LIBRA-VLA 项目页](https://libra-vla.github.io/)
