---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-tracer-texture-robust-affordance-chain-of-thought-for-de
arxiv: "2601.20208"
venue: "arXiv 2026"
summary: "arXiv 2026 的 affordance 论文，清单分组：Reasoning（Reasoning Subcategory: Language-Centric Affordance Reasoning）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/affordance_awesome_tracer-texture-robust-affordance-chain-of-thought-for-de.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# TRACER

**TRACER: Texture-Robust Affordance Chain-of-Thought for Deformable-Object Refinement**（arXiv 2026）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

TRACER: Texture-Robust Affordance Chain-of-Thought for Deformable-Object Refinement——在清单中归入 **Reasoning（Reasoning Subcategory: Language-Centric Affordance Reasoning）**（清单未附方法摘要；方法流程见「流程总览」）。

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
    IN["输入：高层任务指令 + 第一视角图像"] --> TACOT["TA-CoT 树状 Affordance 思维链<br/>（推理阶段，高层语义推理与感知定位解耦）"]
    TACOT --> L1["类型层：单步物体 / 多步物体"]
    L1 --> L2["结构层：是否有兜帽（accept / reject / dormant 门控）"]
    L2 --> L3["属性层：袖长 / 裤长，视觉状态反馈决定是否触发指令"]
    L3 --> L4["收尾层：最终折叠指令"]
    L4 --> SUB["子任务级 affordance 语义指令序列"]
    SUB --> GR["OS-AGDO 骨干 affordance 接地"]
    subgraph TRAIN["两阶段训练"]
        S1["阶段1：SAM 处理后的掩码图 + 文本<br/>SCBR 损失约束边界"] --> S2["阶段2：加载阶段1权重，接入 ICRF<br/>加速度场细化分散响应"]
    end
    S2 -.-> GR
    GR --> REG["连贯的 affordance 区域 → 操作点（Pick / Place）"]
    REG --> ARM["双臂机器人执行"]
```

图依据：论文 §III（Methodology）/ Fig.3、Fig.4、Fig.5；SCBR 见 Fig.6，ICRF 见 Fig.7 与 Algorithm 1。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | arXiv 2026 |
| 方法名（清单标注） | TRACER |
| 分组 1 | Reasoning · Reasoning Subcategory: Language-Centric Affordance Reasoning |
| 论文 | <https://arxiv.org/abs/2601.20208> |

## 核心机制（归纳）

清单本身只给出标题、出处与分类标签；机制见上方「流程总览」（依据论文方法部分），实验细节请读 [原文](https://arxiv.org/abs/2601.20208)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2601.20208) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「TRACER」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的只有标题、出处与分类标签；方法与结果未经本页核实。
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

- [`sources/papers/affordance_awesome_tracer-texture-robust-affordance-chain-of-thought-for-de.md`](../../sources/papers/affordance_awesome_tracer-texture-robust-affordance-chain-of-thought-for-de.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2601.20208>
- [arXiv PDF](https://arxiv.org/pdf/2601.20208)：已通读方法部分（实验与附录以略读为准）：§III-A TA-CoT、§III-B SCBR、§III-C ICRF 及整体框架概述

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2601.20208)
