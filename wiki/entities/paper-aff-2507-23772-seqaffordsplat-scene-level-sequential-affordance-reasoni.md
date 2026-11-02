---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-seqaffordsplat-scene-level-sequential-affordance-reasoni
arxiv: "2507.23772"
venue: "arXiv 2025"
summary: "arXiv 2025 的 affordance 论文，清单分组：Reasoning（Reasoning Subcategory: Language-Centric Affordance Reasoning） ｜ Datasets / Benchmarks（Data Modality: language/3D Gaussian）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/affordance_awesome_seqaffordsplat-scene-level-sequential-affordance-reasoni.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# SeqAffordSplat

**SeqAffordSplat: Scene-level Sequential Affordance Reasoning on 3D Gaussian Splatting**（arXiv 2025）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

SeqAffordSplat: Scene-level Sequential Affordance Reasoning on 3D Gaussian Splatting——在清单中归入 **Reasoning（Reasoning Subcategory: Language-Centric Affordance Reasoning） ｜ Datasets / Benchmarks（Data Modality: language/3D Gaussian）**（清单未附方法摘要；方法流程见「流程总览」，贡献概括以原文为准）。

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
flowchart LR
  Q["高层语言指令 Qinstr"] --> LLM["LLM<br/>自回归生成文本与 SEG token"]
  G["3DGS 场景"] --> Enc["3DGS Encoder<br/>PointNet 系，输入位置/旋转/尺度"]
  Enc -->|"几何特征 Fgeo"| Dec
  G -->|"多视角渲染 RGB"| VFM["冻结 2D VFM<br/>DINOv2 / CLIP"]
  VFM --> Lift["免学习聚合<br/>按渲染权重提升到各 Gaussian"]
  Lift -->|"语义特征 Fsem，多尺度加性融合"| Dec
  LLM -->|"每个 SEG 的隐状态 hseg 作为 query"| Dec["Conditional Affordance Decoder<br/>query 式分割"]
  Dec --> Masks["按顺序的 3D affordance mask 序列"]
  subgraph PRE["Conditional Geometric Reconstruction 预训练"]
    GT["GT mask"] --> ME["mask encoder → 条件嵌入 e_mask"]
    ME -->|"作为 query 注意 Fgeo"| RC["decoder 重建 mask"]
  end
  RC -.->|"初始化"| Enc
```

图依据：论文 SeqSplatNet 架构 / Conditional Geometric Reconstruction Pre-train / VFM Semantic Feature Injection 各节 / Fig.2

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | arXiv 2025 |
| 方法名（清单标注） | SeqAffordSplat |
| 分组 1 | Reasoning · Reasoning Subcategory: Language-Centric Affordance Reasoning |
| 分组 2 | Datasets / Benchmarks · Data Modality: language/3D Gaussian |
| 论文 | <https://arxiv.org/abs/2507.23772> |

## 核心机制（归纳）

清单本身只给出标题、出处与分类标签，**没有**方法摘要；方法流程见上方「流程总览」（依据论文方法部分，只画论文明确写出的模块与数据流）。完整机制与实验请读 [原文](https://arxiv.org/abs/2507.23772)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2507.23772) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「SeqAffordSplat」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的有标题、出处与分类标签，以及「流程总览」所述方法流程（依据论文方法部分）；实验结果未经本页核实。
- 开源状态清单未标注，入库时未打开项目页核查，复现前请自行确认。
- 要深读：从原文入手，再回综述实体页对照 perception → reasoning → action 的位置。

## 常见误区

1. 不要把分类标签当作论文结论；标签来自清单维护者的归类。
2. 不要把本页当作方法证明；它只是清单入口。

## 关联页面

- [Affordance 学习综述](../entities/paper-affordance-learning-embodied-ai-survey.md)
- [Awesome-Affordance-Learning 技术地图](../overview/affordance-awesome-technology-map.md)
- [vla.md](../methods/vla.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/affordance_awesome_seqaffordsplat-scene-level-sequential-affordance-reasoni.md`](../../sources/papers/affordance_awesome_seqaffordsplat-scene-level-sequential-affordance-reasoni.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2507.23772>
- [arXiv PDF](https://arxiv.org/pdf/2507.23772)：已通读方法部分（实验与附录以略读为准）：SeqSplatNet 架构、Conditional Geometric Reconstruction Pre-train、VFM Semantic Feature Injection 与实验设定两节

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2507.23772)
