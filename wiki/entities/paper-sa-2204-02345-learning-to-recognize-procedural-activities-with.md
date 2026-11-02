---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2204.02345"
venue: "CVPR 2022"
summary: "Distant supervision based procedural activity recognition framework"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2204_02345_learning-to-recognize-procedural-activit.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# Learning To Recognize Procedural Activities with Distant Supervision

**Learning To Recognize Procedural Activities with Distant Supervision** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 128/249** 篇，分组 **292 Procedural Learning**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Distant supervision based procedural activity recognition framework

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- Distant supervision based procedural activity recognition framework
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["wikiHow 知识库 B：T 个任务的文本步骤描述"] --> S
    V["HowTo100M 无标注教学视频：片段 x_l + ASR 叙述 a_l"] --> E["MPNet 语言嵌入 e(a_l) 与 e(y)"]
    A --> E
    E --> S["点积相似度 S + softmax：得到 P(y|x_l) 远程监督"]
    S --> L["TimeSformer 视频模型训练"]
    L --> L1["Step Classification：argmax 步骤的交叉熵"]
    L --> L2["Distribution Matching：对 top-K 步骤做 KL 散度"]
    L --> L3["Step Regression：预测步骤语言嵌入的 NCE 损失"]
    L1 --> F["步骤嵌入 f(x_l)"]
    L2 --> F
    L3 --> F
    F --> T["单层 Transformer 时序模型"]
    F -.->|"检索最匹配的 KB 步骤 e(ŷ)"| T
    T --> O1["程序性活动分类"]
    T --> O2["下一步预测 step forecasting"]
```

图依据：论文 §3.1–§3.5 与 Fig.1；注意页面 arXiv 号 2204.02345 与本论文不符，本论文实际为 arXiv 2201.10990（图与参考来源均据此编号）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 128/249 |
| 分组 | 292 Procedural Learning |
| 出处 | CVPR 2022 |
| 论文 | <https://arxiv.org/abs/2204.02345> |

## 核心机制（归纳）

### 策展导读要点

Distant supervision based procedural activity recognition framework

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2204.02345) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **292 Procedural Learning** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Learning To Recognize Procedural Activities with Distant Supervision」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **292 Procedural Learning**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_ego_2204_02345_learning-to-recognize-procedural-activit.md`](../../sources/papers/sun_awesome_ego_2204_02345_learning-to-recognize-procedural-activit.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2204.02345>
- [arXiv PDF](https://arxiv.org/pdf/2201.10990)：已通读方法部分（实验与附录以略读为准）：§3.1 从 wikiHow 提取远程监督、§3.2 步骤嵌入学习、§3.3 活动分类、§3.4 步骤预测、§3.5 模型设计。注意：本页 frontmatter 的 arXiv 号 2204.02345 对应的是另一篇无关论文（关于 tilted fireball 椭圆流分裂的物理论文），本论文实际编号为 2201.10990

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2204.02345)
