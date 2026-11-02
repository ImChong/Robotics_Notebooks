---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2512.04085"
venue: "arXiv 2025"
summary: "Studies the learning problem of ultra-long life-cycle videos from a single individual"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2512_04085_unique-lives-shared-world-learning-from.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# Unique Lives, Shared World

**Unique Lives, Shared World: Learning from Single-Life Egocentric Videos** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 051/249** 篇，分组 **24 Long-form Video Understanding & Structured Reasoning**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Studies the learning problem of ultra-long life-cycle videos from a single individual

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- Studies the learning problem of ultra-long life-cycle videos from a single individual
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TB
  L["每个参与者 i 的一段 life：第一人称视频 Di"] --> PAIR
  subgraph PAIR["配对策略（Single-life CroCo）"]
    TP["时间配对：时间相近帧"]
    SP["空间配对：相机位姿 + 3D 点云，Jaccard 重叠度（有位姿时）"]
  end
  PAIR --> SX["源图 xs / 目标图 xt"]
  SX --> MK["切 patch；目标图高比例随机 mask，仅留可见 patch"]
  MK --> ENC["权重共享 ViT 编码器：zs、zt"]
  ENC --> DEC["Transformer 解码器（zt + mask token 作 Q，zs 作 K/V 的交叉注意力）重建 mask 的目标 patch"]
  DEC --> LOSS["MSE 像素重建损失"]
  LOSS --> M["从随机初始化为每个 life 单独训练一个模型 θi*"]
  M --> ALIGN["对齐：各解码器块交叉注意力平均，top-k 互近邻得 CAS 分数"]
  M --> GEN["泛化：冻结编码器 + 轻量读出头做深度估计、零样本对应等"]
```

图依据：论文 §3.1–§3.4 / Fig.1、Fig.2（CAS）；下游任务设定来自 §5 实验设置开头

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 051/249 |
| 分组 | 24 Long-form Video Understanding & Structured Reasoning |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2512.04085> |

## 核心机制（归纳）

### 策展导读要点

Studies the learning problem of ultra-long life-cycle videos from a single individual

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2512.04085) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **24 Long-form Video Understanding & Structured Reasoning** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Unique Lives, Shared World」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **24 Long-form Video Understanding & Structured Reasoning**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_ego_2512_04085_unique-lives-shared-world-learning-from.md`](../../sources/papers/sun_awesome_ego_2512_04085_unique-lives-shared-world-learning-from.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2512.04085>
- [arXiv PDF](https://arxiv.org/pdf/2512.04085)：已通读方法部分（实验与附录以略读为准）：§3.1 Formulation、§3.2 Model Architecture、§3.3 Pairing Strategies、§3.4 Patch-Level Alignment Metric、§4 数据集概述

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2512.04085)
