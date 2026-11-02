---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-panoramic-affordance-prediction
arxiv: "2603.15558"
venue: "ECCV 2026"
summary: "ECCV 2026 的 affordance 论文，清单分组：Perception（Scope: Scene；Perception Subcategory: Visual） ｜ Datasets / Benchmarks（Data Modality: panoramic image/text）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/affordance_awesome_panoramic-affordance-prediction.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# PAP

**Panoramic Affordance Prediction**（ECCV 2026）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Panoramic Affordance Prediction——在清单中归入 **Perception（Scope: Scene；Perception Subcategory: Visual） ｜ Datasets / Benchmarks（Data Modality: panoramic image/text）**（清单未附方法摘要；方法流程见「流程总览」，贡献概括以原文为准）。

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
  IN[ERP 全景图 I_ERP + 任务描述 T] --> GRID
  subgraph RVR["4.2 递归视觉路由 Recursive Visual Routing via Grid Prompting"]
    GRID[叠加 4x3 数字网格（编号 1 至 12）<br/>全景图先被大幅降采样]
    VLM[VLM 输出目标物体文本描述 T_obj<br/>并选择含目标的网格编号]
    JUDGE{目标跨多个网格？}
    CROP[目标仅在单个网格内：裁剪该网格<br/>重叠新网格再递归，降采样比逐步减小]
    GRID --> VLM --> JUDGE
    JUDGE -->|否，子尺度目标| CROP --> GRID
  end
  JUDGE -->|是，尺度足够，终止递归| GAZE
  subgraph AG["4.3 自适应注视 Adaptive Gaze（免训练域适配）"]
    GAZE[视线：切点对准网格中心经纬度<br/>焦距：按网格位置与大小自适应缩放 FoV]
    PROJ[球面到切平面投影 → 局部透视图 I_persp]
    GAZE --> PROJ
  end
  subgraph CAG["4.4 级联 affordance 接地"]
    OVD[开放词汇检测器 OVD<br/>用 T_obj 在 I_persp 中得框 B 与关键点 P]
    SAM[SAM：以 B 与 P 为提示 → 透视 mask M_persp]
    OVD --> SAM
  end
  VLM -.->|T_obj| OVD
  PROJ --> OVD
  SAM --> INV[逆透视到球面投影映射回 ERP]
  INV --> OUT[全景 affordance mask M_ERP]
```

图依据：论文 §4.1–§4.4 / Fig.5（PAP 框架，免训练 coarse-to-fine 流程）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | ECCV 2026 |
| 方法名（清单标注） | PAP |
| 分组 1 | Perception · Scope: Scene；Perception Subcategory: Visual |
| 分组 2 | Datasets / Benchmarks · Data Modality: panoramic image/text |
| 论文 | <https://arxiv.org/abs/2603.15558> |

## 核心机制（归纳）

清单本身没有方法摘要；方法流程见上方「流程总览」（依据论文方法部分），实验与细节请读 [原文](https://arxiv.org/abs/2603.15558)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2603.15558) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「PAP」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的是标题、出处、分类标签与「流程总览」所述方法流程；实验结果未经本页核实。
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

- [`sources/papers/affordance_awesome_panoramic-affordance-prediction.md`](../../sources/papers/affordance_awesome_panoramic-affordance-prediction.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2603.15558>
- [arXiv PDF](https://arxiv.org/pdf/2603.15558)：已通读方法部分（实验与附录以略读为准）：摘要、§1 引言、§4 PAP（4.1 概览、4.2 递归视觉路由、4.3 自适应注视、4.4 级联 affordance 接地）；§3 任务与数据集 PAP-12K 及 §5 实验仅略读。

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2603.15558)
