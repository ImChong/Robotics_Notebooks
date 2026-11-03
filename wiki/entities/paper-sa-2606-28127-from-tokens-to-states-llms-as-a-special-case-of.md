---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2606.28127"
venue: "arXiv 2026"
summary: "Formalizes LLMs as a special case of world models and proposes a continuous path beyond."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2606_28127_from-tokens-to-states-llms-as-a-special.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# From Tokens to States

**From Tokens to States: LLMs as a Special Case of World Models and the Continuous Path Beyond** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 503/571** 篇，分组 **83 GUI / Computer-Use Agents**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Formalizes LLMs as a special case of world models and proposes a continuous path beyond.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Formalizes LLMs as a special case of world models and proposes a continuous path beyond.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    C0["立场：LLM 与世界模型并非二分<br/>（LeCun 2022：token 预测 vs 潜空间 JEPA）"] --> C1["主张一：LLM 是世界模型的退化特例（§2）"]
    C0 --> C2["主张二：从 NTP 到 JEPA 存在连续谱（§3）"]
    C1 --> F["世界模型形式化 (S, A, T, ρ0)（§2.1）"]
    F --> E["LLM 嵌入：S = V*，A = V，<br/>T(s,a) = 追加 token（§2.2）"]
    E --> K["LLM 相对一般世界模型的五项约束（Table 1）<br/>状态表示 / 动作粒度 / 状态持久 / 训练监督 / 层架构"]
    E --> M["经验支持：OthelloGPT 等隐层编码世界状态（§2.4）"]
    C2 --> S1["NTP / LLM → MTP → Future Summary → Next-Latent → JEPA（Fig. 5）<br/>每一步放松一项约束"]
    K --> S1
    S1 --> Q1["数据问题：自监督文本 → 带动作标注的环境（§3.2）"]
    S1 --> Q2["架构问题：Transformer 能否推广到连续状态预测（§3.3）"]
    M --> R["重审 LeCun 批评（§3.4）"]
    S1 --> R
    Q1 --> Z["讨论与结论（§4）"]
    Q2 --> Z
    R --> Z
```

图依据：论证结构图，论文 §2 / §3 / §4 与 Fig. 3、Fig. 5 的陈述（立场论文，无方法流程）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 503/571 |
| 分组 | 83 GUI / Computer-Use Agents |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2606.28127> |

## 核心机制（归纳）

### 策展导读要点

Formalizes LLMs as a special case of world models and proposes a continuous path beyond.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2606.28127) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **83 GUI / Computer-Use Agents** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「From Tokens to States」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **83 GUI / Computer-Use Agents**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2606_28127_from-tokens-to-states-llms-as-a-special.md`](../../sources/papers/sun_awesome_wm_2606_28127_from-tokens-to-states-llms-as-a-special.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2606.28127>
- [arXiv PDF](https://arxiv.org/pdf/2606.28127)：已通读全文正文（§1 引言、§2 LLM 作为世界模型特例、§3 连续谱、§4 讨论与结论；参考文献略读）；论证结构图依据论文 §2–§4。

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2606.28127)
