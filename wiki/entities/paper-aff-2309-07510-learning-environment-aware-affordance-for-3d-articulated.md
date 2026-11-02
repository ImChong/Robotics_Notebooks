---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-learning-environment-aware-affordance-for-3d-articulated
arxiv: "2309.07510"
venue: "NeurIPS 2023"
summary: "NeurIPS 2023 的 affordance 论文，清单分组：Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/papers/affordance_awesome_learning-environment-aware-affordance-for-3d-articulated.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# Learning Environment-aware Affordance for 3D Articulated Object Manipulation ...

**Learning Environment-aware Affordance for 3D Articulated Object Manipulation under Occlusions**（NeurIPS 2023）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Learning Environment-aware Affordance for 3D Articulated Object Manipulation under Occlusions——在清单中归入 **Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action）**（清单未附方法摘要，此处不替原文概括贡献）。

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
    S["场景点云 S（目标铰接物体 + 遮挡物）"] --> TE["Target Encoder：分割版 PointNet++ 提取逐点特征 f_Tp|S"]
    R["机器人位置 R"] --> RE["Robot Encoder（MLP）：f_R"]
    S --> OF["机器人-目标条件遮挡场 F_R,Tp：到 R 与 Tp 的向量叉积；保留场值小的点"]
    R --> OF
    TE -.目标点 Tp.-> OF
    OF --> SE["Scene Encoder（PointNet）：f_S|Tp,R"]
    TE --> AP["Affordance Predictor（MLP）：逐点 affordance 分数"]
    RE --> AP
    SE --> AP
    AP --> OUT["目标点的 pushing / pulling affordance，用于引导铰接物体操作"]
    SE -.训练.-> CL["对比学习：原任务 / 正样本（加无关遮挡物）/ 负样本（换目标点），triplet loss"]
    AP -.训练.-> L1["L1 监督损失，总损失 = L1 + λ_CL · L_CL"]
```

图依据：论文 §4.1–4.4 / Fig.2、Fig.3

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | NeurIPS 2023 |
| 方法名（清单标注） | - |
| 分组 1 | Action · Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action |
| 论文 | <https://arxiv.org/abs/2309.07510> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，**没有**方法摘要。方法流程见上方「流程总览」（依据论文方法部分）；需要机制与实验请读 [原文](https://arxiv.org/abs/2309.07510)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2309.07510) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「Learning Environment-aware Affordance for 3D Articulated Object Manipulation ...」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的只有标题、出处与分类标签；方法与结果未经本页核实。
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

- [`sources/papers/affordance_awesome_learning-environment-aware-affordance-for-3d-articulated.md`](../../sources/papers/affordance_awesome_learning-environment-aware-affordance-for-3d-articulated.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2309.07510>
- [arXiv PDF](https://arxiv.org/pdf/2309.07510)：已通读方法部分（实验与附录以略读为准）：§1 引言、§3 问题定义、§4 方法（4.1–4.4）、§5.1–5.2 实验设定与基线

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2309.07510)
