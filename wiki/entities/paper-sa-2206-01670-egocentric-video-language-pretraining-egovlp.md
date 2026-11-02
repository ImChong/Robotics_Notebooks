---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2206.01670"
venue: "CVPR 2022"
summary: "The first video-language pre-training framework specifically designed for egocentric scenarios"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2206_01670_egocentric-video-language-pretraining-eg.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# Egocentric Video-Language Pretraining (EgoVLP)

**Egocentric Video-Language Pretraining (EgoVLP)** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 236/249** 篇，分组 **Pre-trained Foundation Models**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

The first video-language pre-training framework specifically designed for egocentric scenarios

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- The first video-language pre-training framework specifically designed for egocentric scenarios
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["Ego4D 带时间戳的叙述 narrations"] --> B["EgoClip：过滤噪声并为每条叙述配对可变长度 clip"]
    B --> C["双编码器 Frozen：TimeSformer 视频编码器 + DistilBERT 文本编码器"]
    C --> D["视频嵌入 v 与文本嵌入 t"]
    D --> E["EgoNCE 预训练目标"]
    P["动作感知正样本：同批中共享名词与动词"] --> E
    N["场景感知负样本：同视频内相邻 clip 作为难负例"] --> E
    E --> F["EgoMCQ 开发集：inter-video 与 intra-video 五选一验证"]
    F -->|"反馈迭代 EgoClip 与模型设计"| B
    E --> G["迁移到下游egocentric任务"]
```

图依据：论文 §3 / §4 / §5 与 Fig.1（EgoClip、预训练模型、EgoMCQ 三部分及迭代反馈）；流程总览中「迁移到下游任务」仅取自 Fig.1 与摘要的概括。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 236/249 |
| 分组 | Pre-trained Foundation Models |
| 出处 | CVPR 2022 |
| 论文 | <https://arxiv.org/abs/2206.01670> |

## 核心机制（归纳）

### 策展导读要点

The first video-language pre-training framework specifically designed for egocentric scenarios

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2206.01670) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **Pre-trained Foundation Models** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Egocentric Video-Language Pretraining (EgoVLP)」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Pre-trained Foundation Models**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_ego_2206_01670_egocentric-video-language-pretraining-eg.md`](../../sources/papers/sun_awesome_ego_2206_01670_egocentric-video-language-pretraining-eg.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2206.01670>
- [arXiv PDF](https://arxiv.org/pdf/2206.01670)：已通读方法部分（实验与附录以略读为准）：§3 EgoClip、§4 视频-语言预训练模型与 EgoNCE、§5 EgoMCQ

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2206.01670)
