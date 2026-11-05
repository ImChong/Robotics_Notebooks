---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2605.00080"
venue: "arXiv 2025"
summary: "Comprehensive survey of world models for robot learning, with a focus on policy learning."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2605_00080_world-model-for-robot-learning-a-compreh.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# World Model for Robot Learning

**World Model for Robot Learning: A Comprehensive Survey** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 543/571** 篇，分组 **Surveys & Overviews**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Comprehensive survey of world models for robot learning, with a focus on policy learning.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Comprehensive survey of world models for robot learning, with a focus on policy learning.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
  I["§1 Introduction：以机器人学习为中心的世界模型视角<br/>三项能力：foresight、imagination-driven planning、data amplification"] --> B["§2 Background<br/>2.1 World Model and Video Generation Model<br/>2.2 Robot Policy"]
  B --> S3["§3 World Model for Policy<br/>按模型架构分类"]
  B --> S4["§4 World Model as Simulator<br/>按不同应用分类"]
  B --> S5["§5 World Model for Robotic Video Generation<br/>按 world-modeling capability 分类"]
  S3 --> S3a["3.1 Why World Models Help Robot Policy Learning<br/>3.2 Inverse Dynamics Policies with World Models<br/>3.3 Unified Policies with a Single World Model Backbone"]
  S3 --> S3b["3.4 MoE/MoT-Style Policies with World-Model Backbone<br/>3.5 Unified Vision-Language-Action Models<br/>3.6 Policies with Latent-Space World Modeling"]
  S4 --> S4a["4.1 World Model for Reinforcement Learning<br/>4.2 World Model for Evaluation"]
  S5 --> S5a["5.1 Problem Setting and Scope<br/>5.2 Video Generation as Imagination for Policy Learning<br/>5.3 Action-Controllable World Model"]
  S5 --> S5b["5.4 Structure-Aware World Model<br/>5.5 From Video Backbones to Foundation World Models<br/>5.6 Technical Progression and Open Challenges"]
  S3 --> S6["§6 Other Applications：6.1 Navigation、6.2 Autonomous Driving"]
  S4 --> S6
  S5 --> S6
  S6 --> S7["§7 Benchmarks, Datasets, and Results<br/>7.1 Benchmarks、7.2 Datasets、7.3 Representative Results"]
  S7 --> S8["§8 Challenges and Future Directions<br/>8.1 至 8.6"]
```

图依据：综述分类体系图，论文 Fig. 1（survey 组织结构）、§1 末尾的章节安排与各节标题。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 543/571 |
| 分组 | Surveys & Overviews |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2605.00080> |

## 核心机制（归纳）

### 策展导读要点

Comprehensive survey of world models for robot learning, with a focus on policy learning.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2605.00080) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **Surveys & Overviews** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「World Model for Robot Learning」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Surveys & Overviews**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2605_00080_world-model-for-robot-learning-a-compreh.md`](../../sources/papers/sun_awesome_wm_2605_00080_world-model-for-robot-learning-a-compreh.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2605.00080>
- [arXiv PDF](https://arxiv.org/pdf/2605.00080)：已通读摘要、§1（含 Fig. 1–2）、各章节标题；§3–§8 正文以略读为准。

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2605.00080)
