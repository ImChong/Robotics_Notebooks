---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2602.23152"
venue: "arXiv 2026"
summary: "Proposes that world models must be grounded on the trinity of consistency: modal, spatial, and temporal."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2602_23152_the-trinity-of-consistency-as-a-defining.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# The Trinity of Consistency as a Defining Principle for General World Models

**The Trinity of Consistency as a Defining Principle for General World Models** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 564/571** 篇，分组 **Theory Interpretability & Positions**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Proposes that world models must be grounded on the trinity of consistency: modal, spatial, and temporal.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Proposes that world models must be grounded on the trinity of consistency: modal, spatial, and temporal.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    T["Trinity of Consistency<br/>世界模型须同时具备三种一致性（Fig. 1）"]
    T --> M["Modal Consistency<br/>语义接口"]
    T --> S["Spatial Consistency<br/>几何基础"]
    T --> TC["Temporal Consistency<br/>因果引擎"]
    subgraph S2["§2 Foundational Exploration of Consistencies"]
        M --> M1["2.2 Modal：理论基础 / 架构演进 / RL 意图对齐 / 测试时计算"]
        S --> S1["2.3 Spatial：几何分解 / 隐式连续场 / 显式拉格朗日基元 / 生成式统计先验"]
        TC --> T1["2.4 Temporal：从频率稳定到物理合规 / 自回归 / DiT / 逻辑与因果推理"]
    end
    S2 --> I["§3 Initial Integration of Multiple Consistencies<br/>3.2 Modal+Spatial · 3.3 Modal+Temporal · 3.4 Spatial+Temporal<br/>3.5 Preliminary Emergence of World Models"]
    I --> C4["§4 Challenges, Benchmarks, and Outlook<br/>通向 General World Simulator"]
    C4 --> B["§5 CoW-Bench"]
    subgraph BN["CoW-Bench"]
        B --> B1["任务蓝图：Modal / Spatial / Temporal 及 M×S、M×T、S×T<br/>共 18 个子任务"]
        B1 --> B2["Reasoning-Driven Seed Construction<br/>结构化种子实例"]
        B2 --> B3["多源审核 / 人机协同校验"]
        B3 --> B4["原子检查分解 → 评测视频生成模型与 UMM"]
    end
```

图依据：综述分类体系图（兼基准构建流程），论文 Contents、Fig. 1、§2–§5（§5.1.1 Dataset Construction、§5.2 Evaluation metrics）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 564/571 |
| 分组 | Theory Interpretability & Positions |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2602.23152> |

## 核心机制（归纳）

### 策展导读要点

Proposes that world models must be grounded on the trinity of consistency: modal, spatial, and temporal.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2602.23152) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **Theory Interpretability & Positions** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「The Trinity of Consistency as a Defining Principle for General World Models」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Theory Interpretability & Positions**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2602_23152_the-trinity-of-consistency-as-a-defining.md`](../../sources/papers/sun_awesome_wm_2602_23152_the-trinity-of-consistency-as-a-defining.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2602.23152>
- [arXiv PDF](https://arxiv.org/pdf/2602.23152)：已通读摘要、目录、Introduction、§4.1–§4.3 的概述段落、§5.1 Dataset 与 §5.2 Evaluation metrics 的开头；§2、§3 各小节只据目录与章节标题，正文细节与 §5.4 之后的实验结果未逐项核对。

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2602.23152)
