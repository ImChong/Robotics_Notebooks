---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2502.07869"
venue: "arXiv 2025"
summary: "End-to-end pipeline for fisheye event-based full-body motion capture on head-worn devices"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2502_07869_eventego3d-3d-human-motion-capture-from.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# EventEgo3D++

**EventEgo3D++: 3D Human Motion Capture from a Head-Mounted Event Camera** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 204/249** 篇，分组 **Event Camera-based Perception**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

End-to-end pipeline for fisheye event-based full-body motion capture on head-worn devices

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- End-to-end pipeline for fisheye event-based full-body motion capture on head-worn devices
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["头戴事件相机事件流（鱼眼镜头）"] --> B["构造 LNES 帧 Lq（时间窗 T）"]
    B --> C["与帧缓冲中的上一帧 L̂q-1 加权相加：L̂q = L̂q-1 ⊙ Cq-1 ⊕ Lq"]
    C --> D["共享编码器 Encoder"]
    D --> E["Heatmap Decoder：16 个关节 2D 热图"]
    E --> F["HM-to-3D Lifting：回归 16×3 个 3D 关节 Ĵq"]
    D --> G["Segmentation Decoder：人体掩码 Ŝq"]
    D --> H["Confidence Decoder：特征图 F̂q"]
    G --> I["置信图 Cq = sigmoid(Ŝq ⊙ F̂q)"]
    H --> I
    I -->|"写入帧缓冲"| J["帧缓冲：Cq 与 L̂q"]
    J -->|"下一帧"| C
```

图依据：论文 §3.1 事件相机与 LNES、§3.2 / Fig. 3 EventEgo3D++ 架构（EPM 与 REPM）、§3.2.2 残差事件传播（Eq. 9 及帧缓冲公式）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 204/249 |
| 分组 | Event Camera-based Perception |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2502.07869> |

## 核心机制（归纳）

### 策展导读要点

End-to-end pipeline for fisheye event-based full-body motion capture on head-worn devices

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2502.07869) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **Event Camera-based Perception** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「EventEgo3D++」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Event Camera-based Perception**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_ego_2502_07869_eventego3d-3d-human-motion-capture-from.md`](../../sources/papers/sun_awesome_ego_2502_07869_eventego3d-3d-human-motion-capture-from.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2502.07869>
- [arXiv PDF](https://arxiv.org/pdf/2502.07869)：已通读方法部分（实验与附录以略读为准）：§3.1、§3.2（3.2.1 EPM、3.2.2 REPM、3.2.3 损失）、§5.1 实现细节

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2502.07869)
