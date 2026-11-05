---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2505.13709"
venue: "arXiv 2025"
summary: "Policy-driven world model adaptation for robust offline MBRL."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2505_13709_rombrl-policy-driven-world-model-adaptat.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# ROMBRL

**ROMBRL: Policy-Driven World Model Adaptation for Robust Offline Model-based RL** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 265/571** 篇，分组 **73 Offline & Generalization**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Policy-driven world model adaptation for robust offline MBRL.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Policy-driven world model adaptation for robust offline MBRL.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
  DS["离线数据集 D_µ"] --> WM0["世界模型 P_ϕ（以数据拟合的 P_ϕ̄ 作约束中心：KL(P_ϕ̄ ‖ P_ϕ) ≤ ε）"]
  WM0 --> ROLL["在 P_ϕ 中 rollout 得到轨迹与折扣回报 J(θ,ϕ)"]
  ROLL --> LEAD["策略 π_θ（领导者）：梯度上升，含对最优响应 ϕ*(θ) 的隐式影响"]
  ROLL --> FOL["世界模型 P_ϕ（跟随者）：最小化 J，拉格朗日乘子 λ 的原始-对偶更新"]
  LEAD -->|"二阶项：Fisher 信息矩阵近似 + Woodbury 恒等式求 A⁻¹"| FOL
  FOL --> LEAD
  LEAD --> OUT["鲁棒策略（maximin：最大化最坏情形性能）"]
```

图依据：论文 §3 / §4 / §5（式 (5)(9)(12)(14)；伪代码在附录 J，未读）

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 265/571 |
| 分组 | 73 Offline & Generalization |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2505.13709> |

## 核心机制（归纳）

### 策展导读要点

Policy-driven world model adaptation for robust offline MBRL.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2505.13709) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **73 Offline & Generalization** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「ROMBRL」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **73 Offline & Generalization**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Highlights 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome World Models](../entities/awesome-world-models.md)
- 技术地图：[Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md)
- 方法/任务：[generative-world-models.md](../methods/generative-world-models.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/sun_awesome_wm_2505_13709_rombrl-policy-driven-world-model-adaptat.md`](../../sources/papers/sun_awesome_wm_2505_13709_rombrl-policy-driven-world-model-adaptat.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2505.13709>
- [arXiv PDF](https://arxiv.org/pdf/2505.13709)：已通读方法部分（实验与附录以略读为准）：§1 Introduction、§2 Background（略读）、§3 Theoretical Results（略读）、§4 Policy-guided World Model Adaptation、§5 Practical Algorithm

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2505.13709)
