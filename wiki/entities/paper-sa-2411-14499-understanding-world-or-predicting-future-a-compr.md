---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2411.14499"
venue: "arXiv 2024"
summary: "Broad survey of world models with a focus on prediction and understanding."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2411_14499_understanding-world-or-predicting-future.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Understanding World or Predicting Future? A Comprehensive Survey of World Models

**Understanding World or Predicting Future? A Comprehensive Survey of World Models** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 541/571** 篇，分组 **Surveys & Overviews**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Broad survey of world models with a focus on prediction and understanding.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Broad survey of world models with a focus on prediction and understanding.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    CORE["世界模型的核心目的（§2.3）：理解外部世界的动力学并预测未来场景（Fig.2）"] --> C3["§3 Implicit Representation of the External World"]
    CORE --> C4["§4 Future Prediction of the Physical World"]
    CORE --> C5["§5 Application Domains"]
    C3 --> C31["§3.1 World Model in Decision-Making：model-based RL、language backbone"]
    C3 --> C32["§3.2 World Knowledge Learned by Models：Global Physical World、Local Physical World、Human Society"]
    C4 --> C41["§4.1 World Model as Video Generation：Towards Video World Models、Capabilities of Video World Models"]
    C4 --> C42["§4.2 World Model as Embodied Environment：Indoor、Outdoor、Dynamic Environments"]
    C5 --> C51["§5.1 Game Intelligence"]
    C5 --> C52["§5.2 Embodied Intelligence：Learning Implicit Representation、Predicting Future States of the Environment、From Simulation to Real World"]
    C5 --> C53["§5.3 Urban Intelligence：Autonomous Driving、Autonomous Logistics、Urban Analytics"]
    C5 --> C54["§5.4 Societal Intelligence：Building Social Simulacra Mirroring Real-world Society、Agent's Understanding of External World in Social Simulacra"]
    C5 --> C55["§5.5 Functions of World Models：cloud-based environments / edge-side agent brains"]
    CORE --> C6["§6 Open Problems and Future Directions：Physical Rules and Counterfactual Simulation、Enriching the Social Dimension、Benchmarks、Bridging Simulation and Reality with Embodied Intelligence、Simulation Efficiency、Ethical and Safety Concerns"]
```

图依据：综述分类体系图，依据论文 §1 末尾的组织说明、§2.3 Categorization 与 Fig.2，以及 §3–§6 的章节标题。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 541/571 |
| 分组 | Surveys & Overviews |
| 出处 | arXiv 2024 |
| 论文 | <https://arxiv.org/abs/2411.14499> |

## 核心机制（归纳）

### 策展导读要点

Broad survey of world models with a focus on prediction and understanding.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2411.14499) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **Surveys & Overviews** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Understanding World or Predicting Future? A Comprehensive Survey of World Models」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Surveys & Overviews**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2411_14499_understanding-world-or-predicting-future.md`](../../sources/papers/sun_awesome_wm_2411_14499_understanding-world-or-predicting-future.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2411.14499>
- [arXiv PDF](https://arxiv.org/pdf/2411.14499)：已读摘要、§1 引言与组织说明、§2.3 分类（含 Fig.2）、§5.5，以及 §3–§6 的章节标题与小节结构；各章正文细节、§7 结论略读

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2411.14499)
