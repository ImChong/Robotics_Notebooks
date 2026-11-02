---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-a3d-adaptive-affordance-assembly-with-dual-arm-manipulat
arxiv: "2601.11076"
venue: "AAAI 2026"
summary: "AAAI 2026 的 affordance 论文，清单分组：Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/papers/affordance_awesome_a3d-adaptive-affordance-assembly-with-dual-arm-manipulat.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# A3D

**A3D: Adaptive Affordance Assembly with Dual-Arm Manipulation**（AAAI 2026）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

A3D: Adaptive Affordance Assembly with Dual-Arm Manipulation——在清单中归入 **Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action）**（清单未附方法摘要；方法流程见「流程总览」）。

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
    OBS["输入：furniture 部件局部点云 O（含法向）+ 操作点 p_op"] --> VFE["视觉特征提取：PointNet++ 逐点特征"]
    VFE --> AFF["Affordance 模块：逐点预测 support affordance 分数<br/>取 Top-K 个支撑候选点"]
    AFF --> PROP["Action Proposal 模块（cVAE）<br/>为每个候选点生成多个支撑方向 d"]
    PROP --> SCORE["Action Scoring 模块：为每个 点-方向 对打成功分数<br/>选出最优支撑动作 u = (p_sp, d)"]
    SCORE --> EXEC["支撑臂移动夹爪到推荐位姿，辅助操作臂完成装配"]
    EXEC --> CHK{"部件是否发生位移<br/>（支撑不足）？"}
    CHK -- 否 --> DONE["继续装配 / 完成"]
    CHK -- 是 --> LOG["记录 交互上下文 I：支撑前点云、已执行动作、位移"]
    LOG --> CTX["Interaction Context 提取器<br/>MLP + 注意力加权聚合得到 f_I"]
    CTX -.-> AFF
    CTX -.-> PROP
    CTX -.-> SCORE
    LOG --> REPRED["用更新后的点云与累积上下文重新预测支撑动作"]
    REPRED --> AFF
```

图依据：论文 Method（Problem Formulation、Support Affordance Module、Interaction Context Adaptation Module）/ Fig.2、Fig.3。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | AAAI 2026 |
| 方法名（清单标注） | A3D |
| 分组 1 | Action · Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action |
| 论文 | <https://arxiv.org/abs/2601.11076> |

## 核心机制（归纳）

清单本身只给出标题、出处与分类标签；机制见上方「流程总览」（依据论文方法部分），实验细节请读 [原文](https://arxiv.org/abs/2601.11076)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2601.11076) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「A3D」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

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

- [`sources/papers/affordance_awesome_a3d-adaptive-affordance-assembly-with-dual-arm-manipulat.md`](../../sources/papers/affordance_awesome_a3d-adaptive-affordance-assembly-with-dual-arm-manipulat.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2601.11076>
- [arXiv PDF](https://arxiv.org/pdf/2601.11076)：已通读方法部分（实验与附录以略读为准）：Method 全部小节（问题形式化、Support Affordance Module、Interaction Context Adaptation Module、训练损失）

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2601.11076)
