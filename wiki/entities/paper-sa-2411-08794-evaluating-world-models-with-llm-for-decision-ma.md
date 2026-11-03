---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2411.08794"
venue: "arXiv 2024"
summary: "Evaluates world models using LLMs for decision-making tasks."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2411_08794_evaluating-world-models-with-llm-for-dec.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Evaluating World Models with LLM for Decision Making

**Evaluating World Models with LLM for Decision Making** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 060/571** 篇，分组 **41 Language-Based World Models**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Evaluates world models using LLMs for decision-making tasks.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Evaluates world models using LLMs for decision-making tasks.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    CLAIM["论点：LLM-based world models 可以单独做决策，但需要严格评测"] --> S3["§3 可单独决策的两点观察"]
    S3 --> O31["Obs 3.1：选出潜在动作应是世界模型的重要能力"]
    S3 --> O32["Obs 3.2：结合下一状态预测与动作提议，规划即可找到策略（Fig.1：不一定需要 critic）"]
    CLAIM --> S4["§4 当前评测不充分的三点观察"]
    S4 --> O41["Obs 4.1：预测重要，但不那么重要（Fig.2：更准的预测不一定带来正确决策）"]
    S4 --> O42["Obs 4.2：并非所有状态对决策同等重要（Fig.3）"]
    S4 --> O43["Obs 4.3：表现通常与其他模块耦合（Fig.4）"]
    O31 --> AP["Task II 动作提议 Action Proposal"]
    O41 --> PV["Task I 策略验证 Policy Verification"]
    O42 --> PV
    O42 --> AP
    O32 --> PP["Task III 策略规划 Policy Planning"]
    O43 --> PP
    PV --> PP
    AP --> PP
    S5["§5 建议的评测框架（Fig.5）：下一状态预测 + 奖励/终止预测"] --> PV
    PV --> ENV["评测设置：31 个环境（Wang et al., 2023），每个环境整理规则型策略（rule-based policy）；GPT-4o / GPT-4o-mini 作为世界模型"]
    AP --> ENV
    PP --> ENV
    ENV --> PROTO["协议：策略验证与规划取 ρ ∈ {0.25, 0.5, 0.75, 1.0}（世界模型负责动作序列末尾 ρ 比例）；动作提议取 top-K，K ∈ {1, 2, 3, 5, 10}；规划仅 K=1"]
```

图依据：立场论文的论证结构与评测框架图，依据论文 §3 Obs 3.1–3.2、§4 Obs 4.1–4.3（Fig.1–4）、§5 开头与 Fig.5、§5.1–5.3 的 Task Description 与 Evaluation Protocol。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 060/571 |
| 分组 | 41 Language-Based World Models |
| 出处 | arXiv 2024 |
| 论文 | <https://arxiv.org/abs/2411.08794> |

## 核心机制（归纳）

### 策展导读要点

Evaluates world models using LLMs for decision-making tasks.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2411.08794) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **41 Language-Based World Models** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Evaluating World Models with LLM for Decision Making」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **41 Language-Based World Models**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Highlights 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome World Models](../entities/awesome-world-models.md)
- 技术地图：[Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md)
- 方法/任务：[generative-world-models.md](../methods/generative-world-models.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/sun_awesome_wm_2411_08794_evaluating-world-models-with-llm-for-dec.md`](../../sources/papers/sun_awesome_wm_2411_08794_evaluating-world-models-with-llm-for-dec.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2411.08794>
- [arXiv PDF](https://arxiv.org/pdf/2411.08794)：已读摘要、§1 引言、§2 预备知识、§3–§4 的观察、§5 评测框架与 §5.1–5.3 的任务描述和评测协议；实验结果图表、§6 相关工作与附录略读

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2411.08794)
