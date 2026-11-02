---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-3d-affordancenet-a-benchmark-for-visual-object-affordanc
arxiv: "2103.16397"
venue: "CVPR 2021"
summary: "CVPR 2021 的 affordance 论文，清单分组：Perception（Scope: Object；Perception Subcategory: Spatial） ｜ Datasets / Benchmarks（Data Modality: 3D）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/affordance_awesome_3d-affordancenet-a-benchmark-for-visual-object-affordanc.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# 3D AffordanceNet

**3D AffordanceNet: A Benchmark for Visual Object Affordance Understanding**（CVPR 2021）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

3D AffordanceNet: A Benchmark for Visual Object Affordance Understanding——在清单中归入 **Perception（Scope: Object；Perception Subcategory: Spatial） ｜ Datasets / Benchmarks（Data Modality: 3D）**（清单未附方法摘要；方法流程见「流程总览」，贡献概括以原文为准）。

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
  P["PartNet 3D 形状\n室内常见物体类别"] --> T["定义 18 类 affordance\n并按物体类别关联可用 affordance"]
  T --> G["3D 问答式标注 GUI\n42 名标注员, 每个形状 3 人标注"]
  G --> Q1["选择该形状支持的 affordance"]
  Q1 --> Q2["选取支撑该 affordance 的关键点\n每个 affordance 至少 3 个"]
  Q2 --> Q3["决定是否传播到其它 part"]
  Q3 --> LP["标签传播\nkNN 图 + 闭式解 S = (I - αW)^-1 Y"]
  LP --> GT["逐点概率 affordance 真值\n归一化到 0-1"]
  GT --> B["基准任务\n全形状 / 部分视角 / 旋转不变 / 半监督估计\n基线: PointNet++, DGCNN, U-Net"]
```

图依据：论文 §3（Dataset Construction，§3.1–3.3，Fig.2）与 §4（基准任务设定）；本图为数据集构建与评测流程，非提出的模型

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | CVPR 2021 |
| 方法名（清单标注） | 3D AffordanceNet |
| 分组 1 | Perception · Scope: Object；Perception Subcategory: Spatial |
| 分组 2 | Datasets / Benchmarks · Data Modality: 3D |
| 论文 | <https://arxiv.org/abs/2103.16397> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，没有方法摘要；论文自述的数据集构建与评测流程见上方「流程总览」，实验数字仍以原文为准；需要机制与实验请读 [原文](https://arxiv.org/abs/2103.16397)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2103.16397) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「3D AffordanceNet」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

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

- [`sources/papers/affordance_awesome_3d-affordancenet-a-benchmark-for-visual-object-affordanc.md`](../../sources/papers/affordance_awesome_3d-affordancenet-a-benchmark-for-visual-object-affordanc.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2103.16397>
- [arXiv PDF](https://arxiv.org/pdf/2103.16397)：已通读方法部分（实验与附录以略读为准）：§3 数据集构建（3.1–3.4）、§4.1 基准任务与基线设定

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2103.16397)
