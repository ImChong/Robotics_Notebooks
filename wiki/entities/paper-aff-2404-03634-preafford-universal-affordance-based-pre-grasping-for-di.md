---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-preafford-universal-affordance-based-pre-grasping-for-di
arxiv: "2404.03634"
venue: "IROS 2024"
summary: "IROS 2024 的 affordance 论文，清单分组：Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/papers/affordance_awesome_preafford-universal-affordance-based-pre-grasping-for-di.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# PreAfford

**PreAfford: Universal Affordance-Based Pre-Grasping for Diverse Objects and Environments**（IROS 2024）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

PreAfford: Universal Affordance-Based Pre-Grasping for Diverse Objects and Environments——在清单中归入 **Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action）**（方法流程见下方「流程总览」）。

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
    OBS["输入：物体 + 环境点云 O"] --> CHK["必要性检查：grasping 模块的 A2/P2 生成抓取提案<br/>C2 打分取平均得成功率 ĉ2"]
    CHK --> D{"ĉ2 超过阈值?"}
    D -- 是 --> GRASP["grasping 模块：A2 选接触点，P2 (cVAE) 生成夹爪朝向<br/>C2 评分，执行得分最高的抓取"]
    D -- 否 --> A1["pre-grasping 模块：A1 对每个接触点算 affordance<br/>取 top n1 点"]
    A1 --> P1["P1 (cVAE) 为每点生成 m1 个推动位移 Δx1"]
    P1 --> C1["C1 评估 (p1, Δx1)，argmax 选出最优推动并执行"]
    C1 --> NEW["物体姿态改变，得到新点云 O2"]
    NEW --> CHK
    GRASP --> DONE["输出：抓取执行"]
    subgraph TRAIN["训练（relay，与推理方向相反）"]
        T1["先在仿真数据上训练 grasping 模块"] --> T2["grasping 模块给出 ĉ2 前后差作为奖励信号"]
        T2 --> T3["再训练 pre-grasping 模块"]
    end
```

图依据：论文 §III-B / Fig.3（框架与 relay）、§III-D（推理四阶段与闭环控制）、§III-E（训练顺序）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | IROS 2024 |
| 方法名（清单标注） | PreAfford |
| 分组 1 | Action · Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action |
| 论文 | <https://arxiv.org/abs/2404.03634> |

## 核心机制（归纳）

清单本身只给出标题、出处与分类标签；方法流程见上方「流程总览」，实验细节请读 [原文](https://arxiv.org/abs/2404.03634)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2404.03634) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「PreAfford」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

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

- [`sources/papers/affordance_awesome_preafford-universal-affordance-based-pre-grasping-for-di.md`](../../sources/papers/affordance_awesome_preafford-universal-affordance-based-pre-grasping-for-di.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2404.03634>
- [arXiv PDF](https://arxiv.org/pdf/2404.03634)：已通读方法部分（实验与附录以略读为准）：§I 引言、§III 方法全部（预备概念、框架、网络结构、推理、损失、数据采集）、§IV-A/B 仿真设定与基线

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2404.03634)
