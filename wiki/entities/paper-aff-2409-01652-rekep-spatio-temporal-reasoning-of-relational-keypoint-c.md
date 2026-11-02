---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-rekep-spatio-temporal-reasoning-of-relational-keypoint-c
arxiv: "2409.01652"
venue: "CoRL 2024"
summary: "CoRL 2024 的 affordance 论文，清单分组：Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/papers/affordance_awesome_rekep-spatio-temporal-reasoning-of-relational-keypoint-c.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# ReKep

**ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation**（CoRL 2024）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation——在清单中归入 **Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action）**（清单未附方法摘要；方法流程见「流程总览」，贡献概括以原文为准）。

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
  IN["输入：RGB-D 观测 + 自由形式语言指令"]
  KP["关键点提议：DINOv2 逐 patch 特征上采样 + SAM 掩码，逐掩码 k-means（k=5）取聚类中心，经标定 RGB-D 投影到 3D 并滤除 8cm 内重复点"]
  VLM["GPT-4o 视觉提示：关键点编号叠加到图像，生成阶段数及每阶段的 ReKep（Python 函数）"]
  C["每阶段 N 个：子目标约束 C_sub-goal 与路径约束 C_path"]
  TRK["3D 关键点跟踪（20 Hz）提供当前关键点位置"]
  SUB["子目标问题：求当前阶段末端位姿 e_gi"]
  PATH["路径问题：求从当前位姿到子目标的轨迹（含碰撞、可达性等辅助代价）"]
  EXE["执行末端动作（约 10 Hz 重新求解）"]
  NEXT["距子目标小于容差 ε 则进入下一阶段"]
  BT["若 C_path 被违反：回溯到满足约束的先前阶段"]
  IN --> KP
  KP --> VLM
  IN --> VLM
  VLM --> C
  C --> SUB
  TRK --> SUB
  SUB --> PATH
  TRK --> PATH
  PATH --> EXE
  EXE --> NEXT
  NEXT --> SUB
  EXE --> BT
  BT --> SUB
```

图依据：论文 §3.1–3.4 与 Fig. 2（「进入下一阶段」「回溯到先前阶段」两条回边来自 §3.3 正文；未读 Algorithm 1 伪代码）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | CoRL 2024 |
| 方法名（清单标注） | ReKep |
| 分组 1 | Action · Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action |
| 论文 | <https://arxiv.org/abs/2409.01652> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，没有方法摘要；方法流程见上方「流程总览」（依据论文方法部分），其余机制与实验请读 [原文](https://arxiv.org/abs/2409.01652)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2409.01652) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「ReKep」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

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

- [`sources/papers/affordance_awesome_rekep-spatio-temporal-reasoning-of-relational-keypoint-c.md`](../../sources/papers/affordance_awesome_rekep-spatio-temporal-reasoning-of-relational-keypoint-c.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2409.01652>
- [arXiv PDF](https://arxiv.org/pdf/2409.01652)：已通读方法部分（实验与附录以略读为准）：§3.1–3.4 方法

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2409.01652)
