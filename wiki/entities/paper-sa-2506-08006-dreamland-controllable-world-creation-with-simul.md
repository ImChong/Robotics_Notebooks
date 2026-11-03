---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2506.08006"
venue: "arXiv 2025"
summary: "Controllable world creation with simulator and generative models."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2506_08006_dreamland-controllable-world-creation-wi.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Dreamland

**Dreamland: Controllable World Creation with Simulator and Generative Models** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 413/571** 篇，分组 **82 Autonomous Driving**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Controllable world creation with simulator and generative models.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Controllable world creation with simulator and generative models.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph ST1["Stage-1 仿真：物理仿真器构造场景"]
        REC["场景记录 scene record<br/>智能体运动来自轨迹回放或训练好的策略"] --> SIM["仿真器渲染驾驶帧"]
        SIM --> SL["按像素语义分解为 Sim-LWA（分层世界抽象）"]
    end
    subgraph LWA["LWA：W = ∪ L_i ⊙ V_i（各层含 K 种条件模态）"]
        LD["交通参与者层 L_d（仿真器，保留）"]
        LL["地图布局层 L_l（仿真器，保留）"]
        LB["背景层 L_b（可编辑）"]
    end
    SL --> LWA
    subgraph ST2["Stage-2 LWA-Sim2Real"]
        ED["指令式编辑模型 ε_e（基于 ACE++）：<br/>在保留 L_d、L_l 的前提下按文本指令 c 改写背景层 L_b"] --> RL["Real-LWA（贴近真实分布，同时保留仿真器接地信息）"]
    end
    LD --> ED
    LL --> ED
    LB --> ED
    subgraph ST3["Stage-3 混合条件生成"]
        ENC["编码器 E 编码 K 种条件 → 通道拼接 → 线性层 ξ 投影"] --> ADD["与噪声 latent 相加 x' = x + ξ(⊕E(c_j))"]
        ADD --> GEN["预训练条件生成模型（Flux Depth）<br/>仅微调投影层 ξ → 真实感图像/视频"]
    end
    RL --> ENC
    GEN --> OUT["可控、逼真的驾驶场景"]
```

图依据：论文 §3 概述、§3.1–3.4 / Fig.2（三阶段）与式(1)–(6)；两步训练见 §3.4。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 413/571 |
| 分组 | 82 Autonomous Driving |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2506.08006> |
| 项目页 | <https://metadriverse.github.io/dreamland/> |

## 核心机制（归纳）

### 策展导读要点

Controllable world creation with simulator and generative models.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2506.08006) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **82 Autonomous Driving** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Dreamland」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_wm_2506_08006_dreamland-controllable-world-creation-wi.md`](../../sources/papers/sun_awesome_wm_2506_08006_dreamland-controllable-world-creation-wi.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2506.08006>
- [arXiv PDF](https://arxiv.org/pdf/2506.08006)：已通读方法部分（实验与附录以略读为准）：§3.1–3.4 方法

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2506.08006)
