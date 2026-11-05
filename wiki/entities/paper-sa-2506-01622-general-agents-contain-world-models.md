---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2506.01622"
venue: "arXiv 2025"
summary: "General agents contain world models."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2506_01622_general-agents-contain-world-models.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# General agents Contain World Models

**General agents Contain World Models** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 551/571** 篇，分组 **Theory Interpretability & Positions**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

General agents contain world models.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- General agents contain world models.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["给定目标条件策略 π(a | h; ψ)（有界 agent，失败率 δ、最大目标深度 n）"] --> B["选定状态 s、动作 a、结果 s'，以及备选动作 b ≠ a"]
    B --> C["构造 LTL 子目标：φ1 = 最终到达 s 并执行 a；φ2 = 下一步到 s'；φ2' = 下一步到 s' 以外"]
    C --> D["对 k = 1…n 构造复合目标 ψ_a,b(k) = ψ_a(k) ∨ ψ_b(k)：先执行 a 或 b，再接 n 轮 (成功 / 失败) 序列，两者成功次数门槛不同"]
    D --> E["查询策略 π 的第一个动作 a0"]
    E --> F{"a0 = a ？"}
    F -- "否，k 加 1" --> D
    F -- "是，记 k*" --> G["估计 P̂_ss'(a) = (k* - 1/2) / n"]
    G --> H["遍历所有 (s, a, s') 得到完整转移函数，即从策略中提取的世界模型"]
```

图依据：论文 §2 Setup / §3 Theorem 1 / §3.1 与附录 C Algorithm 1（本文为理论工作，图为从策略恢复世界模型的算法流程，Algorithm 2 为其简化版）

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 551/571 |
| 分组 | Theory Interpretability & Positions |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2506.01622> |

## 核心机制（归纳）

### 策展导读要点

General agents contain world models.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2506.01622) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **Theory Interpretability & Positions** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「General agents Contain World Models」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Theory Interpretability & Positions**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2506_01622_general-agents-contain-world-models.md`](../../sources/papers/sun_awesome_wm_2506_01622_general-agents-contain-world-models.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2506.01622>
- [arXiv PDF](https://arxiv.org/pdf/2506.01622)：已通读方法部分（实验与附录以略读为准）：§2 Setup（目标、agent、世界模型定义）、§3 主要结果 Theorem 1、附录 C Algorithm 1/2；证明细节与实验略读

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2506.01622)
