---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2305.12953"
venue: "arXiv 2023"
summary: "Aims at the next active object prediction task, proposing an object-guided attention mechanism between object embeddings and spatiotemporal video features (the paper does not use gaze; per arXiv:2305.12953v2 abstract and §3)"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2305_12953_gano-gaze-augmented-next-active-object-b.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# GANO

**GANO（Guided Attention for Next active Objects）**（清单标题写作 "Gaze-Augmented Next Active Object-Based Egocentric Action Anticipation"；论文 v1/v2 的 arXiv 标题均为 "Enhancing Next Active Object-based Egocentric Action Anticipation with Guided Attention"，全文无 gaze 相关内容） 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 154/249** 篇，分组 **Action Anticipation**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Aims at the next active object prediction task, proposing an object-guided attention mechanism between object embeddings and spatiotemporal video features (the paper does not use gaze; per arXiv:2305.12953v2 abstract and §3)

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- Aims at the next active object prediction task, proposing an object-guided attention mechanism between object embeddings and spatiotemporal video features (the paper does not use gaze; per arXiv:2305.12953v2 abstract and §3)
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
  V["RGB 视频帧（等间隔采样）"] --> C3["Conv3D\n帧 patch 特征"]
  V --> DET["目标检测器\n（Ego4D 预训练）→ 边界框"]
  DET --> MLP["MLP → object embeddings"]
  C3 -->|"Q"| GA["Object-Guided Attention\n(多头注意力)"]
  MLP -->|"K, V"| GA
  GA --> ENC["Transformer 编码器\nMulti-Scale Attention + 时空位置编码\n→ Fe"]
  DET -->|"末帧检测框 ROI；不足则补可学习 query"| Q["Object queries"]
  ENC --> DEC["解码器\nMulti-Scale Attention"]
  Q --> DEC
  DEC --> OUT["每个 query 的输出\n名词类别 n / 边界框 b / 动词 v / 接触时间 TTC δ"]
  OUT --> LOSS["损失 L = λ1·box + λ2·noun + λ3·verb + λ4·ttc"]
```

图依据：论文 §3 方法 / Fig.1

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 154/249 |
| 分组 | Action Anticipation |
| 出处 | arXiv 2023 |
| 论文 | <https://arxiv.org/abs/2305.12953> |

## 核心机制（归纳）

### 策展导读要点

Aims at the next active object prediction task, proposing an object-guided attention mechanism between object embeddings and spatiotemporal video features (the paper does not use gaze; per arXiv:2305.12953v2 abstract and §3)

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2305.12953) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **Action Anticipation** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「GANO」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Action Anticipation**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_ego_2305_12953_gano-gaze-augmented-next-active-object-b.md`](../../sources/papers/sun_awesome_ego_2305_12953_gano-gaze-augmented-next-active-object-b.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2305.12953>
- [arXiv PDF](https://arxiv.org/pdf/2305.12953)：已通读方法部分（实验与附录以略读为准）：§3 Proposed Method（特征提取、Object Guided Attention、Transformer 编码器/解码器、Learning）、§4 实验设定
- 核对（2026-11-02）：对照 arXiv:2305.12953v2 摘要与 §3（Object Guided Attention：Q=视频 patch 特征，K/V=检测框经 MLP 的 object embeddings），该注意力是 object-guided 而非 gaze-guided，论文无凝视信息；已改一句话定义/summary/导读及名称说明，流程总览图与论文一致未改。

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2305.12953)
