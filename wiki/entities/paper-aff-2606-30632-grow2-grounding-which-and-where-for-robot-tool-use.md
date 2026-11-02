---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-grow2-grounding-which-and-where-for-robot-tool-use
arxiv: "2606.30632"
venue: "arXiv 2026"
summary: "arXiv 2026 的 affordance 论文，清单分组：Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/papers/affordance_awesome_grow2-grounding-which-and-where-for-robot-tool-use.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# GROW2

**GROW2: Grounding Which and Where for Robot Tool Use**（arXiv 2026）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

GROW2: Grounding Which and Where for Robot Tool Use——在清单中归入 **Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action）**（清单未附方法摘要；方法流程见下方「流程总览」）。

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
    IN["输入：单视角 RGB-D (I, D) + 语言指令 l（含目标物体 o_B）"] --> EX["对象与部件提取<br/>VLM 枚举物体名 → SAM3 分割并裁剪 → VLM 为每个物体生成部件候选"]
    EX --> SEL["工具与部件选择（语义空间）<br/>VLM 选出工具 o_A、目标 o_B，以及抓取部件 p_G、工具功能部件 p_A、目标部件 p_B"]
    SEL --> REC["3D 重建与配准<br/>SAM3 分割 → 深度裁剪转点图 → SAM3D 重建网格 → ICP 刚体配准到点图"]
    REC --> MV["多视角渲染（K=8）→ SAM3 按部件描述分割<br/>复杂描述时用 VLM 转包围框作回退"]
    MV --> FU["掩码反投影到网格并取并集<br/>DBSCAN 聚类，保留最大簇 = 3D 亲和区域"]
    FU --> ACT["仿真实验中的执行：约束定义的关键位姿<br/>Contact-GraspNet 在抓取区域采样抓取 + RRT-Connect 运动规划"]
```

图依据：论文 §3、§4.1、§4.2、Fig. 1–2；末节执行链路来自 §5.2 仿真实验设置（Skill Library）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | arXiv 2026 |
| 方法名（清单标注） | GROW2 |
| 分组 1 | Action · Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action |
| 论文 | <https://arxiv.org/abs/2606.30632> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，**没有**方法摘要；方法模块与数据流见上方「流程总览」（依据论文方法部分）。实验请读 [原文](https://arxiv.org/abs/2606.30632)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2606.30632) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「GROW2」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 清单可确证的只有标题、出处与分类标签；方法流程见「流程总览」，实验结果未经本页核实。
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

- [`sources/papers/affordance_awesome_grow2-grounding-which-and-where-for-robot-tool-use.md`](../../sources/papers/affordance_awesome_grow2-grounding-which-and-where-for-robot-tool-use.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2606.30632>
- [arXiv PDF](https://arxiv.org/pdf/2606.30632)：已通读方法部分（实验与附录以略读为准）：§3 问题设定、§4 Method（4.1–4.2）、§5.2 仿真实验设置（Skill Library）

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2606.30632)
