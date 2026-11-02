---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2510.20668"
venue: "arXiv 2025"
summary: "A tutorial-style guide to world models."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2510_20668_from-masks-to-worlds-a-hitchhiker-s-guid.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# From Masks to Worlds

**From Masks to Worlds: A Hitchhiker's Guide to World Models** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 535/571** 篇，分组 **Surveys & Overviews**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

A tutorial-style guide to world models.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- A tutorial-style guide to world models.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    ANAT["§2.2 The Anatomy of a True World Model：Generative Heart (G)、Interactive Loop (F, C)、Memory System (M) 三个子系统"]
    S1["Stage I: Mask-based Models（§3）：Lays the foundation with masks"]
    S2["Stage II: Unified Models（§4）：Unifies the architectures"]
    S3["Stage III: Interactive Generative Models（§5）：Opens the loop with interaction"]
    S4["Stage IV: Memory and Consistency（§6）：Makes the loop persistent"]
    S5["Stage V: True World Models（§7）：Gives the world a life of its own"]
    S1 --> S2 --> S3 --> S4 --> S5
    ANAT --> S5
    S5 --> P["§7.1 三项性质：Persistence（对应 M）、Agency（对应 F, C）、Emergence（G、F, C、M 共同作用）"]
    S5 --> P2["§7.2 三个定义性挑战 · §7.3 From Simulator to Scientific Instrument"]
```

图依据：综述路线图（论文自定的五阶段演进）与子系统定义，论文 Fig. 1（五阶段演进）、Fig. 2（True World Model 架构）、§1、§2.2、§3–§7；Stage V 与三项性质同三个子系统的对应取自 §7.1 原文；§3–§6 各阶段与子系统之间的对应论文未逐一写成图，故未画。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 535/571 |
| 分组 | Surveys & Overviews |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2510.20668> |

## 核心机制（归纳）

### 策展导读要点

A tutorial-style guide to world models.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2510.20668) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **Surveys & Overviews** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「From Masks to Worlds」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_wm_2510_20668_from-masks-to-worlds-a-hitchhiker-s-guid.md`](../../sources/papers/sun_awesome_wm_2510_20668_from-masks-to-worlds-a-hitchhiker-s-guid.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2510.20668>
- [arXiv PDF](https://arxiv.org/pdf/2510.20668)：已通读摘要、§1、§2（含 §2.2 Anatomy）、§3 Stage I 及 §4 开头；§4.2–§8 仅按章节标题与各节开头略读，用于确认路线图结构；附录未读

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2510.20668)
