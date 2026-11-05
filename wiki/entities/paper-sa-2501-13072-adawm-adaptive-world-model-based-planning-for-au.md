---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2501.13072"
venue: "arXiv 2025"
summary: "Adaptive world model-based planning for autonomous driving."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2501_13072_adawm-adaptive-world-model-based-plannin.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# AdaWM

**AdaWM: Adaptive World Model based Planning for Autonomous Driving** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 396/571** 篇，分组 **82 Autonomous Driving**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Adaptive world model-based planning for autonomous driving.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Adaptive world model-based planning for autonomous driving.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    PRE["预训练（离线）<br/>学习动力学模型 WM_φ（RNN 潜空间世界模型）<br/>与规划策略 π_ω；保留预训练回放缓冲区 B"] --> FT
    subgraph FT["在线微调：每个 finetuning step t"]
        COL["按当前策略 π_t 与动力学模型 WM_t 规划并交互<br/>收集新任务样本 W = (x, a, r)"]
        COL --> MI["Mismatch Identification<br/>用 B 与 W 估计 TV 距离<br/>动力学失配 D_TV(P‖P̂)，策略失配 D_TV(π_t‖π_ω)"]
        MI --> CMP{"D_TV(P‖P̂) > C · D_TV(π_t‖π_ω) ?"}
        CMP -->|"是：动力学失配占主导"| UM["更新动力学模型<br/>LoRA 式低秩适配，仅更新基向量权重 B"]
        CMP -->|"否：策略失配占主导"| UP["更新规划策略<br/>策略分解为子单元凸组合，仅更新权重向量 Δ"]
    end
    UM --> NEXT["进入下一微调步"]
    UP --> NEXT
    NEXT -.-> COL
```

图依据：论文 §2.2（Mismatch Identification and Alignment-driven Finetuning）、Algorithm 1 与 Fig. 2；预训练部分依据 §2 Basic Setting。判据中的阈值 C 由 §2.1 的理论界（Theorem 1）导出。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 396/571 |
| 分组 | 82 Autonomous Driving |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2501.13072> |

## 核心机制（归纳）

### 策展导读要点

Adaptive world model-based planning for autonomous driving.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2501.13072) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **82 Autonomous Driving** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「AdaWM」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **82 Autonomous Driving**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2501_13072_adawm-adaptive-world-model-based-plannin.md`](../../sources/papers/sun_awesome_wm_2501_13072_adawm-adaptive-world-model-based-plannin.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2501.13072>
- [arXiv PDF](https://arxiv.org/pdf/2501.13072)：已通读方法部分（实验与附录以略读为准）：§1 引言末段、§2 AdaWM（基本设定、2.1 失配对性能退化的影响、2.2 失配识别与对齐驱动微调、Algorithm 1）

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2501.13072)
