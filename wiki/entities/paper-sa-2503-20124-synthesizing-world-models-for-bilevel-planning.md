---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2503.20124"
venue: "arXiv 2025"
summary: "Synthesizes world models for bilevel planning."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2503_20124_synthesizing-world-models-for-bilevel-pl.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Synthesizing world models for bilevel planning

**Synthesizing world models for bilevel planning** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 140/571** 篇，分组 **46 Symbolic & Neuro-Symbolic World Models**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Synthesizes world models for bilevel planning.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Synthesizes world models for bilevel planning.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
  A["最多 10 步随机动作 → 回放缓冲 R_random"] --> B["LLM 初始化 Python 低层转移模型 T̂：输入初始状态、动作空间、R_random、领域自然语言描述"]
  C["硬编码 PDDL domain / problem 文件"] --> D["Fast-Downward 生成高层计划：grounded operator 序列"]
  D --> E["低层 BFS 规划器：逐个 operator 用 T̂ 模拟转移，checker 判断 effect 是否满足，再拼接低层动作"]
  B --> E
  E --> F["环境中执行低层计划：预测转移 R_p 与实际转移 R_a 存入回放缓冲"]
  F --> G{"最终目标满足？"}
  G -- 是 --> H["返回低层计划 π"]
  G -- 否 --> I{"R_p 与 R_a 存在不一致？"}
  I -- 是 --> J["LLM 依据预测与实际转移的差异修正 T̂"]
  I -- 否 --> K["枚举前提满足的探索性高层计划并执行，收集转移"]
  K --> J
  J --> E
```

图依据：论文 §3.1–§3.5 / Fig.1–Fig.2、Algorithm 1。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 140/571 |
| 分组 | 46 Symbolic & Neuro-Symbolic World Models |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2503.20124> |

## 核心机制（归纳）

### 策展导读要点

Synthesizes world models for bilevel planning.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2503.20124) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **46 Symbolic & Neuro-Symbolic World Models** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Synthesizing world models for bilevel planning」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **46 Symbolic & Neuro-Symbolic World Models**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2503_20124_synthesizing-world-models-for-bilevel-pl.md`](../../sources/papers/sun_awesome_wm_2503_20124_synthesizing-world-models-for-bilevel-pl.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2503.20124>
- [arXiv PDF](https://arxiv.org/pdf/2503.20124)：已通读方法部分（实验与附录以略读为准）：§2（相关工作，略读）、§3（3.1–3.5、Algorithm 1）

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2503.20124)
