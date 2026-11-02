---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-dualafford-learning-collaborative-visual-affordance-for
arxiv: "2207.01971"
venue: "ICLR 2023"
summary: "ICLR 2023 的 affordance 论文，清单分组：Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/papers/affordance_awesome_dualafford-learning-collaborative-visual-affordance-for.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# DualAfford

**DualAfford: Learning Collaborative Visual Affordance for Dual-gripper Object Manipulation**（ICLR 2023）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

DualAfford: Learning Collaborative Visual Affordance for Dual-gripper Object Manipulation——在清单中归入 **Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action）**（清单未附方法摘要，此处不替原文概括贡献）。

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
    DATA["SAPIEN 仿真离线采集交互数据（随机采样，或可选 RL 增强采样）"] --> TRAIN
    subgraph TRAIN["训练（红色箭头方向：先 M2，后 M1）"]
        T2["先训练第二夹爪模块 M2：以 u1 为条件，学习给出协作的 u2"] --> T1["再训练第一夹爪模块 M1：用已训练 M2 提供监督的 Critic C1 评估 u1 是否便于协作"]
    end
    TRAIN --> ADAPT["Collaborative Adaptation：两夹爪动作在仿真中同时执行，用在线结果同步更新 M1 与 M2"]
    IN["推理输入：部分点云 O + 任务 l"] --> M1
    subgraph M1["第一夹爪模块 M1"]
        A1["Affordance Network A1：逐点 affordance 图，选接触点 p1"] --> P1["Proposal Network P1（cVAE）：采样抓取朝向 R1"]
        P1 --> C1["Critic Network C1：评估动作成功可能性"]
    end
    M1 -->|"u1 = (p1, R1)"| M2
    subgraph M2["第二夹爪模块 M2（输入额外包含 u1）"]
        A2["Affordance Network A2：协作 affordance 图，选 p2"] --> P2["Proposal Network P2（cVAE）：候选朝向 R2"]
        P2 --> C2["Critic Network C2：评估并选择 R2"]
    end
    M2 --> OUT["输出 u1、u2：双夹爪协同操作"]
    ADAPT -.-> M1
    ADAPT -.-> M2
```

图依据：论文 §4.1–§4.5 / Fig.2、Fig.3、Fig.4（虚线表示 Collaborative Adaptation 作用于两模块，为按 §4.4 文字的归纳）

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | ICLR 2023 |
| 方法名（清单标注） | DualAfford |
| 分组 1 | Action · Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action |
| 论文 | <https://arxiv.org/abs/2207.01971> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，没有方法摘要；方法流程见上方「流程总览」（依据论文方法部分），实验细节请读 [原文](https://arxiv.org/abs/2207.01971)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2207.01971) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「DualAfford」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

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

- [`sources/papers/affordance_awesome_dualafford-learning-collaborative-visual-affordance-for.md`](../../sources/papers/affordance_awesome_dualafford-learning-collaborative-visual-affordance-for.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2207.01971>
- [arXiv PDF](https://arxiv.org/pdf/2207.01971)：已通读方法部分（实验与附录以略读为准）：§4.1 框架概览、§4.2 感知模块（第一/第二夹爪模块）、§4.3 损失、§4.4 Collaborative Adaptation、§4.5 数据采集；§5 实验略读

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2207.01971)
