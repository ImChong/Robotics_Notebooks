---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2408.11816"
venue: "arXiv 2024"
summary: "Efficient exploration and discriminative world model learning with object-centric abstraction."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2408_11816_efficient-exploration-and-discriminative.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Efficient Exploration and Discriminative World Model Learning with an Object-...

**Efficient Exploration and Discriminative World Model Learning with an Object-Centric Abstraction** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 127/571** 篇，分组 **45 Object-Centric World Models**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Efficient exploration and discriminative world model learning with object-centric abstraction.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Efficient exploration and discriminative world model learning with object-centric abstraction.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    OBS["低层观测 S"] --> MAP["对象中心映射 M（给定或学得）"]
    MAP --> ABS["抽象状态 X：(item identity, item attribute) 集合"]
    BEH["行为 b = (item, 新 attribute)：对象扰动策略 πb"] --> FWD
    ABS --> FWD["前向模型 fθ(X, b)：预测行为成功概率 q"]
    subgraph TRAIN["训练阶段：无外部奖励的探索"]
        RINTR["计数型内在奖励 r_intr(X, b)"] --> MCTS["在 fθ 想象中做 MCTS，取首个 behaviour"]
        FWD --> MCTS
        MCTS --> EXEC["在 Ab-MDP 中执行行为，收集 (X, b, X')"]
        EXEC --> FIT["统计成功频率，BCE 损失更新 fθ"]
        FIT --> FWD
    end
    subgraph INFER["推理阶段：到达目标抽象状态"]
        RGOAL["r_goal：目标状态为 1，其余为 0"] --> DIJ["Dijkstra：边权 -log fθ(X, b)"]
        FWD --> DIJ
        DIJ --> SEQ["最大成功概率的 behaviour 序列"]
    end
```

图依据：论文 §2 Problem Setting（Fig.1）与 §3 Methods（§3.1 前向模型、§3.2 探索规划、§3.3 目标规划，Fig.2、Fig.3）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 127/571 |
| 分组 | 45 Object-Centric World Models |
| 出处 | arXiv 2024 |
| 论文 | <https://arxiv.org/abs/2408.11816> |

## 核心机制（归纳）

### 策展导读要点

Efficient exploration and discriminative world model learning with object-centric abstraction.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2408.11816) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **45 Object-Centric World Models** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Efficient Exploration and Discriminative World Model Learning with an Object-...」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **45 Object-Centric World Models**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2408_11816_efficient-exploration-and-discriminative.md`](../../sources/papers/sun_awesome_wm_2408_11816_efficient-exploration-and-discriminative.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2408.11816>
- [arXiv PDF](https://arxiv.org/pdf/2408.11816)：已通读方法部分（实验与附录以略读为准）：§2 问题设定、§3 方法（前向模型、探索规划、目标规划）；§4 实验设定略读

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2408.11816)
