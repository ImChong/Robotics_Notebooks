---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-glover-unleashing-the-potential-of-affordance-learning-f
arxiv: "2505.11865"
venue: "CoRL 2025"
summary: "CoRL 2025 的 affordance 论文，清单分组：Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action） ｜ Datasets / Benchmarks（Data Modality: image）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/papers/affordance_awesome_glover-unleashing-the-potential-of-affordance-learning-f.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# GLOVER++

**GLOVER++: Unleashing the Potential of Affordance Learning from Human Behaviors for Robotic Manipulation**（CoRL 2025）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

GLOVER++: Unleashing the Potential of Affordance Learning from Human Behaviors for Robotic Manipulation——在清单中归入 **Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action） ｜ Datasets / Benchmarks（Data Modality: image）**（清单未附方法摘要；论文方法流程见下方「流程总览」）。

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
  A["输入：RGB 图像 I_R + 语言指令 T"] --> B["LLaVA-1.5 多模态编码"]
  B --> C["取出 affordance token &lt;AFF&gt;"]
  C --> D["全局解码 F_dec^glo：&lt;AFF&gt; + 视觉特征 v"]
  D --> E["全局语义图 M_sem"]
  E -->|"作 mask prompt"| F["局部解码 F_dec^loc"]
  F --> G["affordance 热图 M_A"]
  G --> H["取最大概率像素 P_2D"]
  H -->|"相机内参投影"| I["3D affordable 点 A_3D"]
  I --> J["零样本操作：位姿估计 / 几何约束 + IK 运动规划"]
  I --> K["模仿学习：affordance 表示调制注意力"]
  GT["Gaussian 热图 M_gt"] -.->|"Focal + KLD 损失监督"| G
```

图依据：论文 §4.1–4.3 / Fig.1, Fig.5

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | CoRL 2025 |
| 方法名（清单标注） | GLOVER++ |
| 分组 1 | Action · Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action |
| 分组 2 | Datasets / Benchmarks · Data Modality: image |
| 论文 | <https://arxiv.org/abs/2505.11865> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，**没有**方法摘要；方法流程见上方「流程总览」，更多机制与实验请读 [原文](https://arxiv.org/abs/2505.11865)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2505.11865) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「GLOVER++」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的有标题、出处与分类标签，以及「流程总览」所述方法流程（依据论文方法部分）；实验结果未经本页核实。
- 开源状态清单未标注，入库时未打开项目页核查，复现前请自行确认。
- 要深读：从原文入手，再回综述实体页对照 perception → reasoning → action 的位置。

## 常见误区

1. 不要把分类标签当作论文结论；标签来自清单维护者的归类。
2. 不要把本页当作方法证明；它只是清单入口。

## 关联页面

- [Affordance 学习综述](../entities/paper-affordance-learning-embodied-ai-survey.md)
- [Awesome-Affordance-Learning 技术地图](../overview/affordance-awesome-technology-map.md)
- [manipulation.md](../tasks/manipulation.md)、[imitation-learning.md](../methods/imitation-learning.md)

## 参考来源

- [`sources/papers/affordance_awesome_glover-unleashing-the-potential-of-affordance-learning-f.md`](../../sources/papers/affordance_awesome_glover-unleashing-the-potential-of-affordance-learning-f.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2505.11865>
- [arXiv PDF](https://arxiv.org/pdf/2505.11865)：已通读方法部分（实验与附录以略读为准）：§1 引言、§3 数据集、§4 方法（4.1–4.3）、§5.2–5.3 实验设定

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2505.11865)
