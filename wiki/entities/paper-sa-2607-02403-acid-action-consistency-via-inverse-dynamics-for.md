---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2607.02403"
venue: "arXiv 2026"
summary: "Action consistency via inverse dynamics for planning with world models."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2607_02403_acid-action-consistency-via-inverse-dyna.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# ACID

**ACID: Action Consistency via Inverse Dynamics for Planning with World Models** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 238/571** 篇，分组 **72 Continuous Control**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Action consistency via inverse dynamics for planning with world models.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Action consistency via inverse dynamics for planning with world models.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TB
  OBS["当前观测 o_0 与目标观测 o_g"] --> ENC["编码器 E_θ<br/>z_0 = E_θ(o_0), z_g = E_θ(o_g)"]
  ENC --> CEM["CEM 采样 N 条候选动作序列 a_0:H-1<br/>(高斯分布，初始 μ=0, Σ=I)"]
  CEM --> WM["动作条件世界模型 F_θ 自回归 rollout<br/>ẑ_t+1 = F_θ(ẑ_t, a_t)"]
  WM --> CG["目标代价 c_g = ||ẑ_H − z_g||²"]
  WM --> IDM["逆动力学 IDM G_ϕ（flow-matching 动作解码器）<br/>â_t = G_ϕ(ẑ_t, ẑ_t+1)"]
  IDM --> CA["动作一致性代价（cycle action consistency）<br/>c_a = 1/H · Σ ||a_t − â_t||²"]
  CG --> COST["增广代价 c = c_g + w_a · c_a"]
  CA --> COST
  STD["候选间标准差 σ_g, σ_a"] --> W["尺度不变自适应权重 w_a = λ · σ_g / σ_a<br/>每个 CEM 迭代重算"]
  CG -.-> STD
  CA -.-> STD
  W --> COST
  COST --> ELITE["取代价最低的 K 个 elite，更新高斯分布<br/>迭代 J 轮"]
  ELITE -->|"下一轮采样"| CEM
  ELITE --> EXEC["MPC：执行优化出的 H 步动作，随后用新观测重规划"]
```

图依据：论文 §3.1–3.3、Algorithm 1 / Fig.1

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 238/571 |
| 分组 | 72 Continuous Control |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2607.02403> |

## 核心机制（归纳）

### 策展导读要点

Action consistency via inverse dynamics for planning with world models.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2607.02403) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **72 Continuous Control** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「ACID」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **72 Continuous Control**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2607_02403_acid-action-consistency-via-inverse-dyna.md`](../../sources/papers/sun_awesome_wm_2607_02403_acid-action-consistency-via-inverse-dyna.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2607.02403>
- [arXiv PDF](https://arxiv.org/pdf/2607.02403)：已通读方法部分（实验与附录以略读为准）：§3 Method（3.1–3.3）、Algorithm 1；IDM 实现细节仅读正文概述

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2607.02403)
