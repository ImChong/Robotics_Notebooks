---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2308.07123"
venue: "IJCV 2024"
summary: "Co-authored by authoritative scholars in the field, combined with future scenarios of wearable computing, sorting out research shortcomings and directions to be broken through"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2308_07123_an-outlook-into-the-future-of-egocentric.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# An Outlook into the Future of Egocentric Vision

**An Outlook into the Future of Egocentric Vision** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 245/249** 篇，分组 **Surveys & Overviews**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Co-authored by authoritative scholars in the field, combined with future scenarios of wearable computing, sorting out research shortcomings and directions to be broken through

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- Co-authored by authoritative scholars in the field, combined with future scenarios of wearable computing, sorting out research shortcomings and directions to be broken through
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
  S["未来到当下（future-to-present）视角的综述<br/>设想可穿戴设备 EgoAI"]
  S --> N["Sec. 2 Imagining the Future：五个角色化用例"]
  N --> N1["EGO-Home"]
  N --> N2["EGO-Worker"]
  N --> N3["EGO-Tourist"]
  N --> N4["EGO-Police"]
  N --> N5["EGO-Designer"]
  N --> T["Sec. 3 From Narratives to Research Tasks<br/>把故事片段对应到研究任务（Fig. 6）"]
  T --> RT["Sec. 4 Research Tasks and Capabilities"]
  RT --> G1["定位与场景：4.1 Localisation · 4.2 3D Scene Understanding"]
  RT --> G2["动作层任务：4.3 Recognition · 4.4 Anticipation<br/>4.5 Gaze Understanding and Prediction"]
  RT --> G3["理解人：4.6 Social Behaviour Understanding · 4.7 Full-body Pose Estimation<br/>4.8 Hand and Hand-Object Interactions · 4.9 Person Identification"]
  RT --> G4["用户交互：4.10 Summarisation · 4.11 Dialogue"]
  RT --> G5["4.12 Privacy"]
  RT --> G6["4.13 Beyond individual tasks"]
  RT --> P["每个任务的小节结构：seminal works → 当前 SOTA → 数据集 → 局限"]
  S --> D["Sec. 5 General Datasets"]
  S --> C["Sec. 6 Conclusion：近期值得探索的方向"]
```

图依据：综述分类体系图，依据论文 §1 末段对全文组织的说明、目录、§3 对任务分组的叙述与 §4 开头对各任务小节内容的说明；EGO-* 用例名与 4.x 任务名取自原文。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 245/249 |
| 分组 | Surveys & Overviews |
| 出处 | IJCV 2024 |
| 论文 | <https://arxiv.org/abs/2308.07123> |

## 核心机制（归纳）

### 策展导读要点

Co-authored by authoritative scholars in the field, combined with future scenarios of wearable computing, sorting out research shortcomings and directions to be broken through

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2308.07123) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **Surveys & Overviews** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「An Outlook into the Future of Egocentric Vision」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Surveys & Overviews**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_ego_2308_07123_an-outlook-into-the-future-of-egocentric.md`](../../sources/papers/sun_awesome_ego_2308_07123_an-outlook-into-the-future-of-egocentric.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2308.07123>
- [arXiv PDF](https://arxiv.org/pdf/2308.07123)：已读与流程图相关的章节（其余部分以略读为准）：摘要、目录、§1、§2 的用例概述、§3 全文、§4 开头与各小节标题、§5 开头与 Tab. 3、§6；§4 各任务的具体方法综述与参考文献略读

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2308.07123)
