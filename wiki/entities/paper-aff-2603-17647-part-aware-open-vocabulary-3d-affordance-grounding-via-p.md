---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-part-aware-open-vocabulary-3d-affordance-grounding-via-p
arxiv: "2603.17647"
venue: "arXiv 2026"
summary: "arXiv 2026 的 affordance 论文，清单分组：Reasoning（Reasoning Subcategory: Language-Centric Affordance Reasoning）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/affordance_awesome_part-aware-open-vocabulary-3d-affordance-grounding-via-p.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# Part-Aware Open-Vocabulary 3D Affordance Grounding via Prototypical Semantic ...

**Part-Aware Open-Vocabulary 3D Affordance Grounding via Prototypical Semantic and Geometric Alignment**（arXiv 2026）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Part-Aware Open-Vocabulary 3D Affordance Grounding via Prototypical Semantic and Geometric Alignment——在清单中归入 **Reasoning（Reasoning Subcategory: Language-Centric Affordance Reasoning）**（清单未附方法摘要；方法流程见「流程总览」，贡献概括以原文为准）。

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
  Q[自然语言问题 x_lang] --> LLM[大语言模型生成部件感知结构化指令 x_struct<br/>例：A bag consists of body, strap, zipper. For grasp, I should focus on the strap.]
  LLM --> TE[RoBERTa 文本编码 F_T<br/>并取出部件 token 特征 T_i]
  P[物体点云 P] --> BB[PointNet++ 层级编码器<br/>大尺度 / 小尺度区域特征]
  BB --> IORM[IORM 物体内关系建模<br/>区域间点积相似度，取 top-k 区域聚合 value]
  subgraph CMFM["跨模态融合 CMFM（大 / 小两个尺度）"]
    F1[阶段 1 文本到点：点特征为 Query，文本为 Key/Value → M_P]
    F2[阶段 2 点到文本：文本为 Query，M_P 为 Key/Value → M_T]
    F1 --> F2
  end
  TE --> F1
  IORM --> F1
  F1 --> DEC
  F2 --> DEC[解码：增强点特征 ⊙ 逐 patch 池化的 M_P ⊙ 逐通道池化的 M_T<br/>再经 PointNet++ Feature Propagation 上采样]
  DEC --> MSSM[MSSM 多尺度选择模块<br/>软权重 α_l, α_s 门控融合]
  MSSM --> HEAD[MLP mask 头 + sigmoid → 逐点 affordance 概率 M̂]
  subgraph TRAIN["仅训练期正则"]
    APA[APA 原型聚合：以 M̂ 加权池化得区域嵌入 z<br/>与可学习原型集做余弦相似度关联]
    AL[对齐损失 L_align：部件 token T_i 与<br/>真值 mask 加权池化的区域嵌入做余弦损失]
  end
  MSSM -.-> APA
  TE -.-> AL
  MSSM -.-> AL
```

图依据：论文 §3.1 Overview、§3.2–§3.6、§4.1 损失、§4.2 / Fig.2。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | arXiv 2026 |
| 方法名（清单标注） | - |
| 分组 1 | Reasoning · Reasoning Subcategory: Language-Centric Affordance Reasoning |
| 论文 | <https://arxiv.org/abs/2603.17647> |

## 核心机制（归纳）

清单本身没有方法摘要；方法流程见上方「流程总览」（依据论文方法部分），实验与细节请读 [原文](https://arxiv.org/abs/2603.17647)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2603.17647) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「Part-Aware Open-Vocabulary 3D Affordance Grounding via Prototypical Semantic ...」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的是标题、出处、分类标签与「流程总览」所述方法流程；实验结果未经本页核实。
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

- [`sources/papers/affordance_awesome_part-aware-open-vocabulary-3d-affordance-grounding-via-p.md`](../../sources/papers/affordance_awesome_part-aware-open-vocabulary-3d-affordance-grounding-via-p.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2603.17647>
- [arXiv PDF](https://arxiv.org/pdf/2603.17647)：已通读方法部分（实验与附录以略读为准）：摘要、§1 引言、§3 Method（3.1–3.6）、§4 训练损失与推理；§5 实验与附录仅略读。

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2603.17647)
