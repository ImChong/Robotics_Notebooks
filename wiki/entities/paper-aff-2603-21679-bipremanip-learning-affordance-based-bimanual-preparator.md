---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-bipremanip-learning-affordance-based-bimanual-preparator
arxiv: "2603.21679"
venue: "CVPR 2026"
summary: "CVPR 2026 的 affordance 论文，清单分组：Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/papers/affordance_awesome_bipremanip-learning-affordance-based-bimanual-preparator.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# BiPreManip

**BiPreManip: Learning Affordance-Based Bimanual Preparatory Manipulation through Anticipatory Collaboration**（CVPR 2026）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

BiPreManip: Learning Affordance-Based Bimanual Preparatory Manipulation through Anticipatory Collaboration——在清单中归入 **Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action）**（清单未附方法摘要，此处不替原文概括贡献）。

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
  IN[输入：物体点云 O + 语言指令 l] --> GA1
  subgraph S1["3.2 预期阶段：Goal Affordance Network（主臂）"]
    GA1[PointNet++ 逐点特征 + CLIP 文本特征<br/>MLP 融合 → 逐点接触得分 s]
    GA1 --> GA2[取高分点 p_goal<br/>cVAE 预测夹爪朝向 d_goal → 预期目标动作 a_goal]
  end
  subgraph S2["3.3 Pre-Affordance Network（辅臂）"]
    PA1[融合 f_p, f_l 与预期目标特征<br/>MLP → 逐点预备 affordance]
    PA1 --> PA2[cVAE 采样朝向 d_pre → 预备动作 a_pre]
  end
  GA2 --> PA1
  PA2 --> PG[辅臂预抓取物体]
  subgraph S3["3.4 物体位姿预测与重定向"]
    OP[Anticipatory Object Pose Predictor<br/>输入 goal/pre 动作特征 + 全局物体特征 → 预期物体变换 T_obj]
    RA[Reorient Actor<br/>输入 O' 与预抓取观测 O_grasped → 6D 重定向运动]
    OP --> RA
  end
  GA2 --> OP
  PA2 --> OP
  PG --> RA
  RA --> UPD[更新后的场景：物体已重定向，辅臂在位]
  UPD --> GA3[3.5 重新调用 Goal Affordance Network<br/>参数共享，同一语言指令]
  GA3 --> EXE[主臂执行目标导向操作]
```

图依据：论文 §3.1 Overview、§3.2–§3.5 / Fig.3（The BiPreManip pipeline）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | CVPR 2026 |
| 方法名（清单标注） | BiPreManip |
| 分组 1 | Action · Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action |
| 论文 | <https://arxiv.org/abs/2603.21679> |

## 核心机制（归纳）

清单本身没有方法摘要；方法流程见上方「流程总览」（依据论文方法部分），实验与细节请读 [原文](https://arxiv.org/abs/2603.21679)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2603.21679) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「BiPreManip」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的是标题、出处、分类标签与「流程总览」所述方法流程；实验结果未经本页核实。
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

- [`sources/papers/affordance_awesome_bipremanip-learning-affordance-based-bimanual-preparator.md`](../../sources/papers/affordance_awesome_bipremanip-learning-affordance-based-bimanual-preparator.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2603.21679>
- [arXiv PDF](https://arxiv.org/pdf/2603.21679)：已通读方法部分（实验与附录以略读为准）：摘要、§1 引言、§3 Method 全部（3.1 概览至 3.6 训练与损失）。

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2603.21679)
