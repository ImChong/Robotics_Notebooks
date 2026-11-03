---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2506.06253"
venue: "arXiv 2025"
summary: "Systematically sorts out the direction of egocentric-exocentric collaborative learning, covering three paradigms: exocentric-assisted egocentric, egocentric-assisted exocentric, and joint learning"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2506_06253_bridging-perspectives-a-survey-on-cross.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# Bridging Perspectives

**Bridging Perspectives: A Survey on Cross-view Collaborative Intelligence with Egocentric-Exocentric Vision** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 246/249** 篇，分组 **Surveys & Overviews**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Systematically sorts out the direction of egocentric-exocentric collaborative learning, covering three paradigms: exocentric-assisted egocentric, egocentric-assisted exocentric, and joint learning

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- Systematically sorts out the direction of egocentric-exocentric collaborative learning, covering three paradigms: exocentric-assisted egocentric, egocentric-assisted exocentric, and joint learning
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    ROOT["Ego-Exo 综述的整体结构（论文 Fig. 2）"]
    ROOT --> APP["§II Applications：Cooking / Sports / Healthcare / Education / Traffic / Embodied Intelligence / Public Service / Industry（Fig. 3）"]
    APP --> TASKS["§III From Applications to Research Tasks：按三个方向归类（Fig. 4 映射应用与任务）"]
    TASKS --> RT["§IV Research Tasks（Fig. 5）"]
    RT --> A["A. Egocentric for Exocentric：用第一人称提升第三人称理解（ego-to-exo 视频生成 / Action Understanding / View Birdification / View Selection）"]
    RT --> B["B. Exocentric for Egocentric：用第三人称提升第一人称分析（Video Generation / Video Captioning / Action Understanding / Affordance Grounding）"]
    RT --> C["C. Joint Learning：训练与推理均需两种视角（Remote Drone Teleoperation / Video Captioning / Cross-View Retrieval / 3D Camera Localization / Action Understanding / Egocentric Wearer Identification / Cross-View Human Identification / Robotic Manipulation / View Selection）"]
    ROOT --> DS["§V Datasets：同时覆盖两种视角的数据集（Table I），按任务分类"]
    ROOT --> DISC["§VI Discussion：限制与未来方向，从 data / model / application 三个角度"]
    ROOT --> CON["§VII Conclusion"]
```

图依据：综述分类体系图（论文 Fig. 2 全文结构与 Fig. 5 研究任务三方向）：论文 Fig. 2、§III、§IV（Fig. 5）、§V、§VI。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 246/249 |
| 分组 | Surveys & Overviews |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2506.06253> |

## 核心机制（归纳）

### 策展导读要点

Systematically sorts out the direction of egocentric-exocentric collaborative learning, covering three paradigms: exocentric-assisted egocentric, egocentric-assisted exocentric, and joint learning

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2506.06253) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **Surveys & Overviews** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Bridging Perspectives」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_ego_2506_06253_bridging-perspectives-a-survey-on-cross.md`](../../sources/papers/sun_awesome_ego_2506_06253_bridging-perspectives-a-survey-on-cross.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2506.06253>
- [arXiv PDF](https://arxiv.org/pdf/2506.06253)：已通读方法部分（实验与附录以略读为准）：摘要、§I–§III、§IV 的三个方向定义与各任务小标题、§V 数据集分类与 §VI 讨论开头；各任务具体方法与 Table I 以略读为准

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2506.06253)
