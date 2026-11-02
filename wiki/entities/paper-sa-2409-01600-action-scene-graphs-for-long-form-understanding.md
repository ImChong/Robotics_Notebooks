---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2409.01600"
venue: "arXiv 2024"
summary: "Proposes action scene graphs for long-form understanding of egocentric videos, enabling structured analysis of human activities from wearable device captures"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2409_01600_action-scene-graphs-for-long-form-unders.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# Action Scene Graphs for Long-Form Understanding of Egocentric Videos

**Action Scene Graphs for Long-Form Understanding of Egocentric Videos** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 030/249** 篇，分组 **24 Long-form Video Understanding & Structured Reasoning**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Proposes action scene graphs for long-form understanding of egocentric videos, enabling structured analysis of human activities from wearable device captures

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- Proposes action scene graphs for long-form understanding of egocentric videos, enabling structured analysis of human activities from wearable device captures
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["Ego4D 动作片段 C_i 的 PRE / PNR / POST 帧标注<br/>(名词 n、物体框 b_o、左右手框、叙述 r)"] --> B["图初始化：相机佩戴者节点 + 动词节点<br/>动词取自叙述，物体节点取自标注，加 direct object 边"]
    B --> C["图精修：3 位独立 AMT 标注员<br/>核对动词-名词；补充间接物体、介词边及 PRE/PNR/POST 框"]
    C --> D["得到 3 份候选图 G1, G2, G3"]
    D --> E["图验证：最多 4 类问题<br/>动词-名词 / 边介词 / 使用哪只手 / 空间关系"]
    E --> F["每个片段每个时刻一张静态图 G_i(t)"]
    F --> G["时间重组 Temporal Recollection<br/>静态图串成时间动态图 EASG"]
    G --> H["Ego4D-EASG 数据集"]
    H --> I["EASG 生成基线：Edge Cls / SG Cls / EASG Cls<br/>(SlowFast 片段特征 + ROIAlign + 全连接层)"]
    H --> J["下游：以 LLM 做动作预期与长活动摘要<br/>输入图三元组序列"]
```

图依据：论文 §3、§4.1–4.3 / Fig.3（Ego4D-EASG 标注流程）、§5–6（基线与下游实验）。注意：页面 frontmatter 的 arXiv 号 2409.01600 对应的是另一篇无关论文（MCBA 组合服务推荐），本图按正确论文 arXiv:2312.03391（Action Scene Graphs for Long-Form Understanding of Egocentric Videos）绘制。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 030/249 |
| 分组 | 24 Long-form Video Understanding & Structured Reasoning |
| 出处 | arXiv 2024 |
| 论文 | <https://arxiv.org/abs/2409.01600> |

## 核心机制（归纳）

### 策展导读要点

Proposes action scene graphs for long-form understanding of egocentric videos, enabling structured analysis of human activities from wearable device captures

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2409.01600) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **24 Long-form Video Understanding & Structured Reasoning** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Action Scene Graphs for Long-Form Understanding of Egocentric Videos」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_ego_2409_01600_action-scene-graphs-for-long-form-unders.md`](../../sources/papers/sun_awesome_ego_2409_01600_action-scene-graphs-for-long-form-unders.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2409.01600>
- [arXiv PDF](https://arxiv.org/pdf/2312.03391)：已通读方法部分（实验与附录以略读为准）：§1 引言、§3 EASG 定义、§4 Ego4D-EASG 数据集（4.1 标注、4.2 验证、4.3 时间重组）、§5 EASG 生成基线、§6 下游任务设定；注：页面所列编号 2409.01600 与本论文不符，正确编号为 2312.03391

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2409.01600)
