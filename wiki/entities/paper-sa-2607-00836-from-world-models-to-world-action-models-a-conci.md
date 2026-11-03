---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2607.00836"
venue: "arXiv 2026"
summary: "Defines world models as action-conditioned predictive models, summarizing four representative paradigms: imagine-then-execute, video-feature-conditioned action prediction, joint video-action modeling, and auxiliary video"
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2607_00836_from-world-models-to-world-action-models.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# From World Models to World Action Models

**From World Models to World Action Models: A Concise Tutorial for Robotics** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 536/571** 篇，分组 **Surveys & Overviews**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Defines world models as action-conditioned predictive models, summarizing four representative paradigms: imagine-then-execute, video-feature-conditioned action prediction, joint video-action modeling, and auxiliary video prediction.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Defines world models as action-conditioned predictive models, summarizing four representative paradigms: imagine-then-execute, video-feature-conditioned action prediction, joint video-action modeling, and auxiliary video prediction.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    F["§1 基础：世界 World → 物理 AI 任务与策略 Policy<br/>→ 世界模型 WM 与世界动作模型 WAM（及其与 VLA 的区别）"]
    F --> W["§2 世界模型的设计空间：按预测空间划分"]
    F --> A["§3 世界动作模型：把视觉未来预测与动作生成耦合"]
    W --> O["§2.1 观测空间世界模型<br/>观测类型：RGB / 多视角 RGB / RGB-D / 点云<br/>动作类型：机器人动作 / 界面动作 / 潜在动作 / 语言"]
    W --> S["§2.2 状态空间世界模型<br/>潜状态 / 点轨迹 / 神经符号谓词 / 物理状态"]
    A --> P1["Imagine-then-execute<br/>先生成视觉子目标，再由逆动力学或目标条件策略出动作"]
    A --> P2["Video-feature-conditioned action prediction<br/>取视频模型中间特征条件化动作预测"]
    A --> P3["Joint video-action modeling<br/>单一生成模型联合建模未来观测与动作"]
    A --> P4["Auxiliary video prediction<br/>视频预测仅作训练辅助目标，推理时可去掉"]
```

图依据：教程组织结构 / 分类体系图，论文 §1–§3 及 Fig. 4–Fig. 8（Fig. 6 观测空间设计空间、Fig. 7 状态空间设计空间、Fig. 8 WAM 分类）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 536/571 |
| 分组 | Surveys & Overviews |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2607.00836> |

## 核心机制（归纳）

### 策展导读要点

Defines world models as action-conditioned predictive models, summarizing four representative paradigms: imagine-then-execute, video-feature-conditioned action prediction, joint video-action modeling, and auxiliary video prediction.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2607.00836) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **Surveys & Overviews** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「From World Models to World Action Models」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_wm_2607_00836_from-world-models-to-world-action-models.md`](../../sources/papers/sun_awesome_wm_2607_00836_from-world-models-to-world-action-models.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2607.00836>
- [arXiv PDF](https://arxiv.org/pdf/2607.00836)：已通读全文正文（§1 基础、§2 世界模型设计空间、§3 世界动作模型；参考文献略读）；分类体系图依据论文 §1–§3 与 Fig. 4–8。

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2607.00836)
