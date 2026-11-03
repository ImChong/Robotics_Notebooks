---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2409.12278"
venue: "arXiv 2024"
summary: "Transforms LLMs into world models using precondition and effect knowledge."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2409_12278_making-large-language-models-into-world.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Making Large Language Models into World Models with Precondition and Effect K...

**Making Large Language Models into World Models with Precondition and Effect Knowledge** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 064/571** 篇，分组 **41 Language-Based World Models**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Transforms LLMs into world models using precondition and effect knowledge.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Transforms LLMs into world models using precondition and effect knowledge.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph DATA["语料构建：global-local prompting（GPT-4，五步）"]
        D1["1 生成完整行动计划"] --> D2["2 局部：丢弃孤立步骤、可补充步骤"]
        D2 --> D3["3 为每步同时生成前置条件与效果"]
        D3 --> D4["4 局部：找出链接弱的步骤，重生成一次并选择性丢弃"]
        D4 --> D5["5 全局：按未被覆盖比例过滤样本（丢弃最高 5%）"]
    end
    D5 --> M1["微调 LLM：前置条件推理（动作→前置条件）"]
    D5 --> M2["微调 LLM：效果推理（动作→效果）"]
    subgraph WM["世界模型（自然语言状态）"]
        subgraph VA["有效动作预测"]
            V1["动作 → 前置条件推理模块"] --> V2["语义匹配模块（GPT-4）：前置条件是否都被当前状态覆盖"]
            V2 --> V3["判定该动作是否有效"]
        end
        subgraph ST["状态转移预测"]
            S1["动作 → 效果推理模块"] --> S2["语义匹配模块（GPT-4）：找出与效果矛盾的状态部分"]
            S2 --> S3["旧状态删除矛盾部分并加入效果 → 新状态"]
        end
    end
    M1 --> V1
    M2 --> S1
```

图依据：论文 §3.1–3.3（全局-局部提示五步、语义状态匹配、世界模型应用）

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 064/571 |
| 分组 | 41 Language-Based World Models |
| 出处 | arXiv 2024 |
| 论文 | <https://arxiv.org/abs/2409.12278> |

## 核心机制（归纳）

### 策展导读要点

Transforms LLMs into world models using precondition and effect knowledge.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2409.12278) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **41 Language-Based World Models** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Making Large Language Models into World Models with Precondition and Effect K...」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_wm_2409_12278_making-large-language-models-into-world.md`](../../sources/papers/sun_awesome_wm_2409_12278_making-large-language-models-into-world.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2409.12278>
- [arXiv PDF](https://arxiv.org/pdf/2409.12278)：已通读方法部分（实验与附录以略读为准）：§2 相关工作略读、§3.1 前置条件/效果推理模块、§3.2 语义状态匹配、§3.3 世界模型应用

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2409.12278)
