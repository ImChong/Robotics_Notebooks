---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2404.10775"
venue: "arXiv 2024"
code: https://github.com/UMass-Foundation-Model/COMBO
summary: "Compositional world models for embodied multi-agent cooperation."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2404_10775_combo-compositional-world-models-for-emb.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# COMBO

**COMBO: Compositional World Models for Embodied Multi-Agent Cooperation** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 361/571** 篇，分组 **814 Locomotion & Loco-Manipulation**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Compositional world models for embodied multi-agent cooperation.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Compositional world models for embodied multi-agent cooperation.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    O["部分第一视角 RGBD 观测 o_i"] --> PC["重建局部点云，转为俯视正交图"]
    PC --> INP["扩散模型补全含噪不完整的世界状态"]
    INP --> S["估计的世界状态 s"]
    S --> AP["Action Proposer：VLM 提出 P 个自身候选动作"]
    S --> IT["Intent Tracker：VLM 推断其他智能体动作"]
    AP --> JA["联合动作 a"]
    IT --> JA
    JA --> CWM["Compositional World Model：按智能体分解动作，各自条件下的分数合成视频扩散预测"]
    CWM --> SN["下一状态 s_next"]
    SN --> OE["Outcome Evaluator：VLM 对状态打启发式分"]
    OE --> TS["树搜索：保留 B 个最优 plan beam，展开至深度 D"]
    TS -->|"未到深度，继续展开"| AP
    TS --> BP["选出得分最优的 plan，执行并在每步重新规划"]
```

图依据：论文 §4（式 4-7）、§5.1-§5.3 / Fig.2、Fig.3、Algorithm 1。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 361/571 |
| 分组 | 814 Locomotion & Loco-Manipulation |
| 出处 | arXiv 2024 |
| 论文 | <https://arxiv.org/abs/2404.10775> |
| 代码/项目 | <https://github.com/UMass-Foundation-Model/COMBO> |

## 核心机制（归纳）

### 策展导读要点

Compositional world models for embodied multi-agent cooperation.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2404.10775) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **814 Locomotion & Loco-Manipulation** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「COMBO」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **814 Locomotion & Loco-Manipulation**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2404_10775_combo-compositional-world-models-for-emb.md`](../../sources/papers/sun_awesome_wm_2404_10775_combo-compositional-world-models-for-emb.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2404.10775>
- [arXiv PDF](https://arxiv.org/pdf/2404.10775)：已通读方法部分（§4 Compositional World Model、§4.2 Agent-Dependent Loss Scaling、§5.1 世界状态估计、§5.2 规划子模块、§5.3 树搜索与 Algorithm 1）与 Fig.2、Fig.3；实验与附录以略读为准

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2404.10775)
