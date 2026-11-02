---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-affogato-learning-open-vocabulary-affordance-grounding-w
arxiv: "2506.12009"
venue: "ECCV 2026"
summary: "ECCV 2026 的 affordance 论文，清单分组：Perception（Scope: Object；Perception Subcategory: Visual · Spatial · Generalizable） ｜ Datasets / Benchmarks（Data Modality: language-3D）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/affordance_awesome_affogato-learning-open-vocabulary-affordance-grounding-w.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# Affogato

**Affogato: Learning Open-Vocabulary Affordance Grounding with Automated Data Generation at Scale**（ECCV 2026）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Affogato: Learning Open-Vocabulary Affordance Grounding with Automated Data Generation at Scale——在清单中归入 **Perception（Scope: Object；Perception Subcategory: Visual · Spatial · Generalizable） ｜ Datasets / Benchmarks（Data Modality: language-3D）**（清单未附方法摘要，此处不替原文概括贡献）。

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
flowchart TB
  subgraph PIPE["Affogato-750K 自动标注流程（§3）"]
    O["Objaverse 3D 物体"] --> R["G-Objaverse 多视角渲染"]
    R --> G1["Stage 1：Gemma3<br/>先判物体类别（CoT），再生成每物体 5 条 affordance query"]
    G1 --> M1["Stage 2：Molmo<br/>每个 query 在每个视角预测交互点"]
    M1 --> S1["Stage 3：MobileSAM<br/>点提示取最小 mask，sigmoid 得 2D heatmap"]
    S1 --> V["利用相机参数与深度投影到 3D 表面<br/>多视角投票聚合"]
    V --> H["3D affordance heatmap"]
    H -->|"投影到 25 个视角，选最可见视角"| H2["2D heatmap 划分"]
  end
  subgraph ESP["Espresso 基线（§4）"]
    TXT["affordance 文本"] --> TE["文本编码器<br/>3D：Recap-CLIP / 2D：CLIP"]
    VIS["3D：PartField 点云编码器<br/>2D：DINOv2"] --> DEC["text-conditioned heatmap decoder<br/>文本嵌入作 query"]
    TE --> DEC
    DEC --> OUT["affordance heatmap<br/>视觉与文本的跨模态相似度"]
  end
  H -.->|"3D 训练/评测数据"| VIS
  H2 -.->|"2D 训练/评测数据"| VIS
```

图依据：论文 §3（数据标注流程）/ §4（Espresso）/ Fig.2、Fig.5

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | ECCV 2026 |
| 方法名（清单标注） | Affogato |
| 分组 1 | Perception · Scope: Object；Perception Subcategory: Visual · Spatial · Generalizable |
| 分组 2 | Datasets / Benchmarks · Data Modality: language-3D |
| 论文 | <https://arxiv.org/abs/2506.12009> |

## 核心机制（归纳）

清单本身只给出标题、出处与分类标签，**没有**方法摘要；方法流程见上方「流程总览」（依据论文方法部分，只画论文明确写出的模块与数据流）。完整机制与实验请读 [原文](https://arxiv.org/abs/2506.12009)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2506.12009) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「Affogato」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

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

- [`sources/papers/affordance_awesome_affogato-learning-open-vocabulary-affordance-grounding-w.md`](../../sources/papers/affordance_awesome_affogato-learning-open-vocabulary-affordance-grounding-w.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2506.12009>
- [arXiv PDF](https://arxiv.org/pdf/2506.12009)：已通读方法部分（实验与附录以略读为准）：§3 Affogato-750K 标注流程（含 3.1、3.2 三个阶段）、§4 Espresso 模型

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2506.12009)
