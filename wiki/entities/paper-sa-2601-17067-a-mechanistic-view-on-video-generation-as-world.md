---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2601.17067"
venue: "arXiv 2026"
summary: "Taxonomy centered on State Construction and Dynamics Modeling."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2601_17067_a-mechanistic-view-on-video-generation-a.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# A Mechanistic View on Video Generation as World Models

**A Mechanistic View on Video Generation as World Models: State and Dynamics** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 069/571** 篇，分组 **42 Visual / Video World Models**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Taxonomy centered on State Construction and Dynamics Modeling.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Taxonomy centered on State Construction and Dynamics Modeling.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["Preliminaries §3<br/>世界模型三要素：Observation / State / Dynamics"] --> B["提出以 State Construction 与 Dynamics Modeling 为两大支柱的分类体系"]
    B --> S["§4 State"]
    B --> D["§5 Dynamics"]
    S --> S1["4.1 Implicit State：Memory Mechanism<br/>在 context 中管理历史"]
    S1 --> S11["Compression：Static / Dynamic"]
    S1 --> S12["Retrieval：Internal / External"]
    S1 --> S13["Consolidation"]
    S --> S2["4.2 Explicit State<br/>把历史压缩为紧凑变量"]
    S2 --> S21["Coupled States：Hidden-variable / Parametric"]
    S2 --> S22["Decoupled States：Semantics-oriented / Geometry-oriented"]
    D --> D1["5.1 Causal Architecture Reformulation<br/>Causal Video Model"]
    D --> D2["5.2 Causal Knowledge Integration<br/>Video Model + VLM"]
    S --> E["§6 Evaluation：三条轴"]
    D --> E
    E --> E1["6.1 Quality"]
    E --> E2["6.2 Persistence：Long-horizon Coherence / Memory Capacity Tasks"]
    E --> E3["6.3 Causality：Temporal Reasoning / Interventions / Planning"]
    E --> F["§7 Future Work：Persistence 与 Causality"]
```

图依据：综述分类体系图，论文摘要与 §1 贡献条目、Fig.（Sec.1–6 分类总览图）、§3 Preliminaries、§4 Categorization - State、§5 Categorization - Dynamics、§6 Evaluation、§7 Future Work

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 069/571 |
| 分组 | 42 Visual / Video World Models |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2601.17067> |

## 核心机制（归纳）

### 策展导读要点

Taxonomy centered on State Construction and Dynamics Modeling.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2601.17067) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **42 Visual / Video World Models** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「A Mechanistic View on Video Generation as World Models」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **42 Visual / Video World Models**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2601_17067_a-mechanistic-view-on-video-generation-a.md`](../../sources/papers/sun_awesome_wm_2601_17067_a-mechanistic-view-on-video-generation-a.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2601.17067>
- [arXiv PDF](https://arxiv.org/pdf/2601.17067)：已读摘要、§1 Introduction、§3 Preliminaries、§4 State（4.1–4.3 的开头、定义与小结，逐篇方法略读）、§5 Dynamics 开头、§6 Evaluation（6.2–6.3）、§7–§8；总览分类图的文字由 pdftotext 抽出，部分子类名按正文小节核对（参考文献与各方法细节未读）

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2601.17067)
