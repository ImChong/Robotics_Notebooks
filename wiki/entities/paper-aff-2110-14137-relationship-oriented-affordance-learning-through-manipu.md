---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-relationship-oriented-affordance-learning-through-manipu
arxiv: "2110.14137"
venue: "arXiv 2021"
summary: "arXiv 2021 的 affordance 论文，清单分组：Reasoning（Reasoning Subcategory: Relation-Based Affordance Reasoning）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/affordance_awesome_relationship-oriented-affordance-learning-through-manipu.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# AR-Net

**Relationship Oriented Affordance Learning through Manipulation Graph Construction**（arXiv 2021）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Relationship Oriented Affordance Learning through Manipulation Graph Construction——在清单中归入 **Reasoning（Reasoning Subcategory: Relation-Based Affordance Reasoning）**（清单未附方法摘要；方法流程见「流程总览」，贡献概括以原文为准）。

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
    I["RGB 图像：包含多个物体的场景"] --> RPN["RPN 生成类别无关的物体候选<br/>（objectness 二分类，不预测类别）"]
    RPN --> FC["两两配对，构建有向全连接图<br/>每条边是一个候选 (subject, relationship, object)"]
    FC --> SG["取 subject 与 object 的 union box 作为 subgraph<br/>subgraph clustering 合并相同三元组的子图"]
    SG --> FT["全连接层 / 2D 卷积<br/>得到物体特征 o 与 subgraph 特征图 S"]
    FT --> CTX["Context 模块（注意力）<br/>用物体特征加权聚合到 subgraph：Ŝ = S + α·Ô"]
    FT --> ATT["Attribute 模块：多标签预测 R=6 类属性<br/>仅训练时作辅助任务，推理时丢弃"]
    ATT -.-> FT
    CTX --> PRED["p_ij = FC(ReLU(o_i + Ŝ + o_j))<br/>R 类操作关系 + 背景"]
    PRED --> NMS["Top-1 关系 + 三元组得分 + triplet NMS"]
    NMS --> MRG["操作关系图 MRG<br/>节点=物体，边=关系，三元组得分为边权"]
    MRG --> ROB["机器人执行：AffordanceNet 分割 subject 与 object<br/>取 grasp point 与 effect point，运动基元执行"]
```

图依据：论文 §IV（AR-Net 流程概述与 A–D 小节）/ Fig.3，§V-C（物理操作实验中的 grasp / effect point 计算，Fig.6）

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | arXiv 2021 |
| 方法名（清单标注） | AR-Net |
| 分组 1 | Reasoning · Reasoning Subcategory: Relation-Based Affordance Reasoning |
| 论文 | <https://arxiv.org/abs/2110.14137> |

## 核心机制（归纳）

清单本身只给出标题、出处与分类标签；方法流程见上文「流程总览」，实验细节请读 [原文](https://arxiv.org/abs/2110.14137)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2110.14137) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「AR-Net」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

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

- [`sources/papers/affordance_awesome_relationship-oriented-affordance-learning-through-manipu.md`](../../sources/papers/affordance_awesome_relationship-oriented-affordance-learning-through-manipu.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2110.14137>
- [arXiv PDF](https://arxiv.org/pdf/2110.14137)：已通读方法部分（实验与附录以略读为准）：§I Introduction、§III MRG 定义、§IV AR-Net（A–D）、§V 实验设置（含 SMRD 数据集与物理操作实验设置）

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2110.14137)
