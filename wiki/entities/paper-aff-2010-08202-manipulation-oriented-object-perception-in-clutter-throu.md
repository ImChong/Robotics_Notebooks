---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-manipulation-oriented-object-perception-in-clutter-throu
arxiv: "2010.08202"
venue: "Humanoids 2022"
summary: "Humanoids 2022 的 affordance 论文，清单分组：Perception（Scope: Object；Perception Subcategory: Spatial）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/affordance_awesome_manipulation-oriented-object-perception-in-clutter-throu.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# ACF

**Manipulation-Oriented Object Perception in Clutter through Affordance Coordinate Frames**（Humanoids 2022）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Manipulation-Oriented Object Perception in Clutter through Affordance Coordinate Frames——在清单中归入 **Perception（Scope: Object；Perception Subcategory: Spatial）**（清单未附方法摘要，此处不替原文概括贡献）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Affordance | Action possibility (Gibson) | 环境/物体对 agent 提供的可交互机会 |
| HOI | Human-Object Interaction | 人–物交互，常作 affordance 监督来源 |
| VLM | Vision-Language Model | 视觉–语言模型，用于 affordance 推理与接地 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |

## 为什么重要

- 综述把 affordance 学习拆成感知、推理、引导动作三段（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），本条目是其中一段的代表文献之一。
- 可顺着 [技术地图](../overview/affordance-awesome-technology-map.md) 的同分组条目横向阅读同方向工作。

## 流程总览

```mermaid
flowchart TD
  IN["RGB-D 观测\n深度作为第 4 通道输入"] --> BB["ResNet50 + FPN 主干\n(类 Mask R-CNN)"]
  BB --> DET["Faster R-CNN 模块\n检测框与 ROI 特征"]
  DET --> CTX["ROI 特征与全局特征融合\nGlobal Context Block (自注意力)"]
  CTX --> K["3D 关键点头\n种子点偏移 + deep Hough voting"]
  CTX --> AX["有向轴头\n两个 3D 端点, 同样投票"]
  CTX --> PAF["Part Affinity Field 头\n2D 单位向量指向关联部件"]
  K --> CL["Mean Shift 聚类 voter\n得到关键点与轴端点"]
  AX --> CL
  CL --> ACF["每个部件的 ACF\n3D 关键点 + 有向轴"]
  PAF --> ASSOC["判定哪些部件属于同一物体"]
  ACF --> ASSOC
  ASSOC --> ACT["机器人用估计的 ACF 执行任务\n(如抓取、倒、搅拌)"]
```

图依据：论文 §III-B（Estimation Pipeline，Fig.3）；末节「执行任务」来自 §I/§IV 对机器人使用 ACF 的描述

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | Humanoids 2022 |
| 方法名（清单标注） | ACF |
| 分组 1 | Perception · Scope: Object；Perception Subcategory: Spatial |
| 论文 | <https://arxiv.org/abs/2010.08202> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，没有方法摘要；论文自述的方法流程见上方「流程总览」，实验数字仍以原文为准；需要机制与实验请读 [原文](https://arxiv.org/abs/2010.08202)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2010.08202) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「ACF」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的只有标题、出处与分类标签；方法与结果未经本页核实。
- 开源状态清单未标注，入库时未打开项目页核查，复现前请自行确认。
- 要深读：从原文入手，再回综述实体页对照 perception → reasoning → action 的位置。

## 常见误区

1. 不要把分类标签当作论文结论；标签来自清单维护者的归类。
2. 不要把本页当作方法证明；它只是清单入口。

## 关联页面

- [Affordance 学习综述](../entities/paper-affordance-learning-embodied-ai-survey.md)
- [Awesome-Affordance-Learning 技术地图](../overview/affordance-awesome-technology-map.md)
- [manipulation.md](../tasks/manipulation.md)、[vla.md](../methods/vla.md)

## 参考来源

- [`sources/papers/affordance_awesome_manipulation-oriented-object-perception-in-clutter-throu.md`](../../sources/papers/affordance_awesome_manipulation-oriented-object-perception-in-clutter-throu.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2010.08202>
- [arXiv PDF](https://arxiv.org/pdf/2010.08202)：已通读方法部分（实验与附录以略读为准）：§III 表示与估计流程、§IV 数据集与实现细节

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2010.08202)
