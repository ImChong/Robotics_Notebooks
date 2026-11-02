---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2604.10517"
venue: "ICML 2026"
summary: "A curriculum-based framework for learning task-oriented spatiotemporal reasoning. Constructs EgoTSR-Data comprising 46 million samples organized into three stages. Achieves 92.4% accuracy on long-horizon logical reasonin"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2604_10517_egotsr-from-perception-to-planning-evolv.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# EgoTSR

**EgoTSR: From Perception to Planning: Evolving Ego-Centric Task-Oriented Spatiotemporal Reasoning via Curriculum Learning** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 085/249** 篇，分组 **27 Embodied AI & Vision-Language-Action Ego-VLA**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

A curriculum-based framework for learning task-oriented spatiotemporal reasoning. Constructs EgoTSR-Data comprising 46 million samples organized into three stages. Achieves 92.4% accuracy on long-horizon logical reasoning tasks

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- A curriculum-based framework for learning task-oriented spatiotemporal reasoning. Constructs EgoTSR-Data comprising 46 million samples organized into three stages. Achieves 92.4% accuracy on long-horizon logical reasoning tasks
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    DATA["EgoTSR-Data：CoT 子集 + Tag 子集 + LongTag 子集（每个样本含两帧 Ia, Ib、任务指令、完成度更接近的标签 y）"]
    subgraph CUR["三阶段 Curriculum Learning（同一 VLM 依次训练）"]
        S1["阶段 1 知识引导：CoT 数据，先生成空间推理路径再给结论"]
        S2["阶段 2 能力内化：Tag 数据，抑制显式推理，直接输出判断"]
        S3["阶段 3 泛化巩固：LongTag 数据，以子任务序列 S 作为条件"]
        S1 --> S2 --> S3
    end
    DATA --> S1
    DATA --> S2
    DATA --> S3
    subgraph DEC["Reasoning-Enhanced Task Decomposition"]
        PLAN["Subtask Planner：由初始场景 I0 与任务指令生成逻辑骨架 S（正交原子子任务序列）"]
    end
    PLAN -->|"训练时给出 S 的标注，推理时自行生成"| S3
    subgraph EVAL["Dual-Level Evaluation Framework"]
        L1["Level 1 原子感知 T_short：不同帧间隔的短程两帧比较"]
        L2["Level 2 规划一致性 T_long：依据子任务骨架做同任务内 / 相邻 / 跨多子任务比较"]
        BI["双向评测：前向 (Ia, Ib) 与反向 (Ib, Ia)，衡量时间顺序偏置"]
    end
    S3 --> L1
    S3 --> L2
    L1 --> BI
    L2 --> BI
```

图依据：论文 Fig.1 / Fig.2、§3.1（数据与课程学习）、§3.2（任务分解）、§3.3（双层评测）

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 085/249 |
| 分组 | 27 Embodied AI & Vision-Language-Action Ego-VLA |
| 出处 | ICML 2026 |
| 论文 | <https://arxiv.org/abs/2604.10517> |

## 核心机制（归纳）

### 策展导读要点

A curriculum-based framework for learning task-oriented spatiotemporal reasoning. Constructs EgoTSR-Data comprising 46 million samples organized into three stages. Achieves 92.4% accuracy on long-horizon logical reasoning tasks

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2604.10517) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **27 Embodied AI & Vision-Language-Action Ego-VLA** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「EgoTSR」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **27 Embodied AI & Vision-Language-Action Ego-VLA**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Highlights 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md)
- 技术地图：[Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md)
- 方法/任务：[vla.md](../methods/vla.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/sun_awesome_ego_2604_10517_egotsr-from-perception-to-planning-evolv.md`](../../sources/papers/sun_awesome_ego_2604_10517_egotsr-from-perception-to-planning-evolv.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2604.10517>
- [arXiv PDF](https://arxiv.org/pdf/2604.10517)：已通读方法部分（实验与附录以略读为准）：§3 Methods（3.1 数据集与课程学习、3.2 任务分解、3.3 双层评测框架）

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2604.10517)
