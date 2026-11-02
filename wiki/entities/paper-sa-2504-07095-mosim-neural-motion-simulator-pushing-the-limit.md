---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2504.07095"
venue: "arXiv 2025"
summary: "A world model that predicts future physical states of embodied systems from current observations and actions, significantly outperforming prior work."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2504_07095_mosim-neural-motion-simulator-pushing-th.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# MoSim

**MoSim: Neural Motion Simulator — Pushing the Limit of World Models in Reinforcement Learning** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 026/571** 篇，分组 **112 Differentiable Physics Engines & Neural Simulators**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

A world model that predicts future physical states of embodied systems from current observations and actions, significantly outperforming prior work.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- A world model that predicts future physical states of embodied systems from current observations and actions, significantly outperforming prior work.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    S["物理状态 s = (q, q̇)"] --> PE["Position Encoder (ResNet)<br/>输出下三角 L，M = L Lᵀ"]
    S --> SE["State Encoder (ResNet)<br/>输出 b(s)"]
    AC["动作 a（关节力矩等）"] --> AE["Action Encoder (MLP)<br/>输出 τ(a)"]
    PE --> PR["Predictor<br/>q̈ = M·(b + τ)，与 q̇ 组成 ṡ_pred"]
    SE --> PR
    AE --> PR
    S --> CO["Corrector (并联 ResNet)<br/>ε(s, a) 建模接触/摩擦等未建模项"]
    AC --> CO
    PR --> SUM["ṡ = Predictor + Corrector"]
    CO --> SUM
    SUM --> ODE["Neural ODE（DOPRI5 积分多个子步）"]
    ODE --> NS["预测下一状态；滚动得到长时域预测"]
    TR1["多阶段训练 阶段1：只训 Predictor"] --> TR2["阶段2：冻结 Predictor，训 Corrector"]
    NS --> RL["可作为 dm_control 环境替身<br/>接入 TQC / SAC 做零样本、少样本 RL"]
```

图依据：论文 §2.1 Architecture / Fig.2、§2.3 Multi-Stage Training、§3.4（多阶段训练两节点以独立链路画出，未连入推理主干）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 026/571 |
| 分组 | 112 Differentiable Physics Engines & Neural Simulators |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2504.07095> |

## 核心机制（归纳）

### 策展导读要点

A world model that predicts future physical states of embodied systems from current observations and actions, significantly outperforming prior work.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2504.07095) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **112 Differentiable Physics Engines & Neural Simulators** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「MoSim」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **112 Differentiable Physics Engines & Neural Simulators**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2504_07095_mosim-neural-motion-simulator-pushing-th.md`](../../sources/papers/sun_awesome_wm_2504_07095_mosim-neural-motion-simulator-pushing-th.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2504.07095>
- [arXiv PDF](https://arxiv.org/pdf/2504.07095)：已通读方法部分（实验与附录以略读为准）：§2.1–2.3、§3.4、§3.6–3.7

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2504.07095)
