---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-affordance-agent-harness-verification-gated-skill-orches
arxiv: "2605.00663"
venue: "arXiv 2026"
summary: "arXiv 2026 的 affordance 论文，清单分组：Reasoning（Reasoning Subcategory: Agentic Affordance Reasoning）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/affordance_awesome_affordance-agent-harness-verification-gated-skill-orches.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# A-Harness

**Affordance Agent Harness: Verification-Gated Skill Orchestration**（arXiv 2026）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Affordance Agent Harness: Verification-Gated Skill Orchestration——在清单中归入 **Reasoning（Reasoning Subcategory: Agentic Affordance Reasoning）**（清单未附方法摘要；方法流程见「流程总览」，贡献概括以原文为准）。

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
    IN["观测 x + 可选指令 q + 预算 B（以 skill 调用次数计）"] --> RET["用 DINO-v3 + SigLIP 联合嵌入检索<br/>两层记忆 M_cs 与 M_tt 的 top-N 条目"]
    RET --> R["Router π_θ<br/>按收益/成本比选择 skill 与参数"]
    R --> SK["调用 skill：检测 / 分割 / 放大 / 网络搜索 / 交互想象"]
    SK --> ES["Evidence Store<br/>带来源（producer id）的证据记录"]
    ES --> V["Verifier：一致性 ω、跨尺度稳定性 ζ、证据充分度 μ<br/>聚合成 v_t"]
    V -->|"v_t ≥ δ 且 ω_t ≥ 下限"| FU["最终融合 Φ：<br/>检测 skill 定位关键点，SAM 2 以关键点为 prompt 出 mask"]
    V -->|未通过：诊断最大短板，给出补救动作| R
    V -->|预算耗尽| FU
    FU --> OUT["affordance 图 A"]
    OUT -->|仅当 v_t ≥ δ| WB["写回 M_tt（verifier 接受的 episode）"]
```

图依据：论文 §3.2 Overview / §3.3 Details of A-Harness（Eq.1–12）/ Algorithm 1 / Fig.3。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | arXiv 2026 |
| 方法名（清单标注） | A-Harness |
| 分组 1 | Reasoning · Reasoning Subcategory: Agentic Affordance Reasoning |
| 论文 | <https://arxiv.org/abs/2605.00663> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，没有方法摘要；网络结构与流程见上方「流程总览」（依据论文方法部分），训练细节与实验请读 [原文](https://arxiv.org/abs/2605.00663)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2605.00663) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「A-Harness」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的有标题、出处与分类标签，以及「流程总览」所述方法流程（依据论文方法部分）；实验结果未经本页核实。
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

- [`sources/papers/affordance_awesome_affordance-agent-harness-verification-gated-skill-orches.md`](../../sources/papers/affordance_awesome_affordance-agent-harness-verification-gated-skill-orches.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2605.00663>
- [arXiv PDF](https://arxiv.org/pdf/2605.00663)：已通读方法部分（实验与附录以略读为准）：§3.1–§3.4、Algorithm 1

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2605.00663)
