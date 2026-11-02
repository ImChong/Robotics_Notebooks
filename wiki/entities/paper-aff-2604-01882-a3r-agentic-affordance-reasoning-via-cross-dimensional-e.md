---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-a3r-agentic-affordance-reasoning-via-cross-dimensional-e
arxiv: "2604.01882"
venue: "arXiv 2026"
summary: "arXiv 2026 的 affordance 论文，清单分组：Reasoning（Reasoning Subcategory: Agentic Affordance Reasoning）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/affordance_awesome_a3r-agentic-affordance-reasoning-via-cross-dimensional-e.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# A3R

**A3R: Agentic Affordance Reasoning via Cross-Dimensional Evidence in 3D Gaussian Scenes**（arXiv 2026）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

A3R: Agentic Affordance Reasoning via Cross-Dimensional Evidence in 3D Gaussian Scenes——在清单中归入 **Reasoning（Reasoning Subcategory: Agentic Affordance Reasoning）**（清单未附方法摘要；方法流程见「流程总览」，贡献概括以原文为准）。

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
  I[输入：3DGS 场景 S + affordance 查询 q] --> S0[初始化决策状态 s_t<br/>工作区 S_t + affordance 置信图 h_t + 动作-证据历史 m_t]
  S0 --> CTX[渲染多视角视觉上下文 V_t<br/>叠加 h_t 与历史结果，并拼文本历史 ξ_t、任务 prompt、算子 schema]
  CTX --> POL[MLLM 智能体策略 π_θ<br/>选择 u_t = 算子类型 + 参数]
  POL -->|Term| OUT[对 h 取阈值 δ<br/>输出逐 Gaussian 预测 ŷ]
  POL -->|Inst / Part / SAM2D / Zoom| OP[执行证据获取算子]
  subgraph OPS["跨维度证据算子"]
    O1[Inst：3D 实例分割]
    O2[Part：3D 部件分割]
    O3[SAM2D：渲染视图 2D 分割并提升到 3D]
    O4[Zoom：收缩到局部子场景]
  end
  OP --> OPS
  OPS --> PRJ[投影回 Gaussian 支持 e_t]
  PRJ -->|Part / SAM2D| UB[替换更新置信图 h]
  PRJ -->|Inst / Zoom| UW[细化工作区，置信图不变]
  UB --> UH[更新历史 m 与工作区 S]
  UW --> UH
  UH --> CTX
  UH -.->|达到步数预算 T| OUT
  GRPO[GRPO 训练策略<br/>奖励 = IoU 与成功项 - 步数惩罚，LoRA，算子冻结] -.-> POL
```

图依据：论文 §3.1–§3.5 / Fig.2、Algorithm 1。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | arXiv 2026 |
| 方法名（清单标注） | A3R |
| 分组 1 | Reasoning · Reasoning Subcategory: Agentic Affordance Reasoning |
| 论文 | <https://arxiv.org/abs/2604.01882> |

## 核心机制（归纳）

清单本身没有方法摘要；方法流程见上方「流程总览」（依据论文方法部分），实验与细节请读 [原文](https://arxiv.org/abs/2604.01882)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2604.01882) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「A3R」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

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

- [`sources/papers/affordance_awesome_a3r-agentic-affordance-reasoning-via-cross-dimensional-e.md`](../../sources/papers/affordance_awesome_a3r-agentic-affordance-reasoning-via-cross-dimensional-e.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2604.01882>
- [arXiv PDF](https://arxiv.org/pdf/2604.01882)：已通读方法部分（实验与附录以略读为准）：摘要、§1 引言、§3.1–§3.5 方法（含 Algorithm 1）、§4.1 实验设定。

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2604.01882)
