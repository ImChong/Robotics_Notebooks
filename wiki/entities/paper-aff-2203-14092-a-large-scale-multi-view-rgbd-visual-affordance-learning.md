---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-a-large-scale-multi-view-rgbd-visual-affordance-learning
arxiv: "2203.14092"
venue: "ICIP 2023"
summary: "ICIP 2023 的 affordance 论文，清单分组：Perception（Scope: Object；Perception Subcategory: Visual）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/affordance_awesome_a-large-scale-multi-view-rgbd-visual-affordance-learning.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# A Large Scale Multi-View RGBD Visual Affordance Learning Dataset

**A Large Scale Multi-View RGBD Visual Affordance Learning Dataset**（ICIP 2023）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

A Large Scale Multi-View RGBD Visual Affordance Learning Dataset——在清单中归入 **Perception（Scope: Object；Perception Subcategory: Visual）**（清单未附方法摘要；方法流程见「流程总览」，贡献概括以原文为准）。

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
    A["Washington RGB-D 多视角物体数据集<br/>51 类，转台采集，30/45/60° 三个机位"] --> B["每 5 帧取 1 帧，避免帧间重叠"]
    B --> C["筛选 37 类适合 affordance 的物体<br/>剔除水果、蔬菜等"]
    C --> D["定义 15 类 affordance<br/>沿用综述 12 类 + 新增 absorb / grip / stapling"]
    D --> E["按物体功能为每个类别指定 affordance<br/>每类至多 4 个标签"]
    E --> F["LabelMe 在 RGB 图上画多边形标注<br/>30 名标注员，每类两人标注"]
    F --> G["两位专家核验"]
    F --> H["标注映射到对应深度图"]
    G --> I["47210 张 RGBD 图 / 51304 条 affordance 标注"]
    H --> I
    I --> J["Affordance 分割基准<br/>U-Net / PSPNet / Segmenter / HRViT 等，80/10/10 划分"]
    I --> K["多标签 Affordance 识别基准<br/>VGG-16 / ResNet-50 / InceptionV3，5 折 70/30"]
```

图依据：论文 §3（Dataset Curation）、§3.1、§4（Benchmarking）

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | ICIP 2023 |
| 方法名（清单标注） | - |
| 分组 1 | Perception · Scope: Object；Perception Subcategory: Visual |
| 论文 | <https://arxiv.org/abs/2203.14092> |

## 核心机制（归纳）

清单本身只给出标题、出处与分类标签；数据集构建与基准流程见上文「流程总览」，实验细节请读 [原文](https://arxiv.org/abs/2203.14092)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2203.14092) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「A Large Scale Multi-View RGBD Visual Affordance Learning Dataset」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的有标题、出处与分类标签，以及「流程总览」所述方法流程（依据论文方法部分）；实验结果未经本页核实。
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

- [`sources/papers/affordance_awesome_a-large-scale-multi-view-rgbd-visual-affordance-learning.md`](../../sources/papers/affordance_awesome_a-large-scale-multi-view-rgbd-visual-affordance-learning.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2203.14092>
- [arXiv PDF](https://arxiv.org/pdf/2203.14092)：已通读方法部分（实验与附录以略读为准）：§1 Introduction、§3 Dataset Curation（含 3.1/3.2）、§4 Benchmarking（4.1/4.2）

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2203.14092)
