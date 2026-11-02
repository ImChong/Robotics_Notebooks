---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2602.09076"
venue: "arXiv 2026"
summary: "Finds that robots can forecast human movement efficiently by watching their legs, providing actionable insights for designing sensing capabilities for social robot navigation"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2602_09076_legs-over-arms-on-the-predictive-value-o.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# Legs Over Arms

**Legs Over Arms: On the Predictive Value of Lower-Body Pose for Human Trajectory Prediction from Egocentric Robot Perception** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 094/249** 篇，分组 **27 Embodied AI & Vision-Language-Action Ego-VLA**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Finds that robots can forecast human movement efficiently by watching their legs, providing actionable insights for designing sensing capabilities for social robot navigation

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- Finds that robots can forecast human movement efficiently by watching their legs, providing actionable insights for designing sensing capabilities for social robot navigation
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart LR
  A["机器人第一视角感知（360° 相机）"] --> B["多智能体历史轨迹 X（过去 6 步 / 2 s，3 Hz）"]
  A --> C["骨架关键点 K（3D 33 点 或 2D COCO 17 点）"]
  C --> D["按配置 s 选子集：全身 / 下半身 / 上半身"]
  D --> E["派生生物力学线索 C：腿/臂关节角、步长、头朝向"]
  D --> F["关键点 MLP 编码"]
  E --> G["线索 MLP 编码"]
  B --> H["与轨迹特征融合"]
  F --> H
  G --> H
  H --> I["HST 全自注意力（跨智能体与时间步）"]
  I --> J["预测未来 12 步 / 4 s，6 个模式"]
  J --> K["指标：minADE / minFDE / MLADE / NLLpos"]
```

图依据：论文 §III-A 问题设定、§III-C 实现细节（HST 骨干 + 附加特征流）；本图为对比研究的输入—模型流程，非新提出的网络结构。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 094/249 |
| 分组 | 27 Embodied AI & Vision-Language-Action Ego-VLA |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2602.09076> |

## 核心机制（归纳）

### 策展导读要点

Finds that robots can forecast human movement efficiently by watching their legs, providing actionable insights for designing sensing capabilities for social robot navigation

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2602.09076) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **27 Embodied AI & Vision-Language-Action Ego-VLA** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Legs Over Arms」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_ego_2602_09076_legs-over-arms-on-the-predictive-value-o.md`](../../sources/papers/sun_awesome_ego_2602_09076_legs-over-arms-on-the-predictive-value-o.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2602.09076>
- [arXiv PDF](https://arxiv.org/pdf/2602.09076)：已通读方法部分（实验与附录以略读为准）：摘要、§I 引言、§III 方法论（问题设定、数据集、实现细节）、§IV 评测指标部分

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2602.09076)
