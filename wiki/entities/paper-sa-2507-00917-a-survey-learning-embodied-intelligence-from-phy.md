---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2507.00917"
venue: "arXiv 2025"
summary: "A systematic survey of recent advances in learning embodied AI through the integration of physical simulators and world models."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2507_00917_a-survey-learning-embodied-intelligence.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# A Survey

**A Survey: Learning Embodied Intelligence from Physical Simulators and World Models** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 017/571** 篇，分组 **111 General Simulation Platforms**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

A systematic survey of recent advances in learning embodied AI through the integration of physical simulators and world models.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- A systematic survey of recent advances in learning embodied AI through the integration of physical simulators and world models.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["综述总览：物理模拟器与世界模型如何促进具身智能"] --> L["§2 Levels of Intelligent Robot：IR-L0 至 IR-L4 五级分级"]
    A --> M["§3 Robotic Mobility, Dexterity and Interaction"]
    A --> S["§4 General Physical Simulators"]
    A --> W["§5 World Models"]
    A --> R["§6 World Models for Intelligent Robots"]
    L --> L1["IR-L0 Basic Execution → IR-L1 Programmatic Response → IR-L2 Basic Perception and Adaptation → IR-L3 Humanoid Cognition and Collaboration → IR-L4 Fully Autonomous"]
    M --> M1["§3.1 Technical Concepts · §3.2 Robot Locomotion · §3.3 Robot Manipulation · §3.4 Human-Robot Interaction"]
    S --> S1["§4.1 Mainstream Simulators · §4.2 Physical Simulation · §4.3 Realistic rendering · §4.4 Sensor and Joint Component Types · §4.5 Discussions and future Perspectives"]
    W --> W1["§5.1 Representative Architectures of WM · §5.2 Core Roles of World Models"]
    R --> R1["§6.1 World Models for Autonomous Driving · §6.2 World Models for Articulated Robots · §6.3 Challenges and Future Perspectives"]
```

图依据：综述组织结构图，论文 Fig. 2（Survey outline）与 §1.3 Structure；图中只画章节框架，未画论文内容的评价。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 017/571 |
| 分组 | 111 General Simulation Platforms |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2507.00917> |

## 核心机制（归纳）

### 策展导读要点

A systematic survey of recent advances in learning embodied AI through the integration of physical simulators and world models.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2507.00917) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **111 General Simulation Platforms** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「A Survey」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **111 General Simulation Platforms**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2507_00917_a-survey-learning-embodied-intelligence.md`](../../sources/papers/sun_awesome_wm_2507_00917_a-survey-learning-embodied-intelligence.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2507.00917>
- [arXiv PDF](https://arxiv.org/pdf/2507.00917)：已通读摘要、§1（含 §1.2 贡献、§1.3 Structure）与 Fig. 2 综述大纲；§3–§6 正文仅按章节标题略读（仅用于确认章节框架）

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2507.00917)
