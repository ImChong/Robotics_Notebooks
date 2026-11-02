---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2605.24496"
venue: "CVPR 2026 (EPIC-KITCHENS Challenge Champion Solution)"
summary: "Proposes a decoupled verb-noun detection + dynamic weighted fusion strategy, achieving SOTA accuracy on long untrimmed videos based on VideoMAE-L features"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2605_24496_egoaction-egocentric-action-composition.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# EgoAction

**EgoAction: Egocentric Action Composition with Reliability-Aware Temporal Fusion** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 166/249** 篇，分组 **Action Recognition & Detection**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Proposes a decoupled verb-noun detection + dynamic weighted fusion strategy, achieving SOTA accuracy on long untrimmed videos based on VideoMAE-L features

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- Proposes a decoupled verb-noun detection + dynamic weighted fusion strategy, achieving SOTA accuracy on long untrimmed videos based on VideoMAE-L features
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    V["未裁剪第一人称视频<br/>(EPIC-KITCHENS, 30 FPS)"] --> F["VideoMAE-L 特征<br/>(OpenTAD 预提取, clip 16 / stride 8)"]
    F --> SW["滑窗推理: 窗长最多 4608 特征步, 50% 重叠"]
    SW --> NS["名词检测流 (CausalTAD 风格)<br/>因果投影 + 7 层特征金字塔 + 无锚点头"]
    SW --> VS["动词检测流 (CausalTAD 风格)<br/>因果投影 + 7 层特征金字塔 + 无锚点头"]
    NS --> NP["名词边界 b_n 与类别分数 P_n"]
    VS --> VP["动词边界 b_v 与类别分数 P_v"]
    NP --> COMP["组合动作打分<br/>各取 top-10 名词 / 动词<br/>S = sqrt(P_n * P_v)"]
    VP --> COMP
    NP --> DWF["Dynamic Weighted Fusion<br/>C = 各流最大类别置信度<br/>W_n = C_n / (C_n + C_v), W_v 同理<br/>b = W_n * b_n + W_v * b_v"]
    VP --> DWF
    COMP --> POST["按置信度过滤并取 top 候选"]
    DWF --> POST
    POST --> NMS["跨滑窗输出的类别内 Soft-NMS"]
    NMS --> OUT["转换为 JSON 提交检测结果"]
```

图依据：论文 §2.2–§2.6 / Fig.1（§2.4 跨流可靠性引导仅为对照的双流变体，最终提交保持解耦，故未入图）

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 166/249 |
| 分组 | Action Recognition & Detection |
| 出处 | CVPR 2026 (EPIC-KITCHENS Challenge Champion Solution) |
| 论文 | <https://arxiv.org/abs/2605.24496> |

## 核心机制（归纳）

### 策展导读要点

Proposes a decoupled verb-noun detection + dynamic weighted fusion strategy, achieving SOTA accuracy on long untrimmed videos based on VideoMAE-L features

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2605.24496) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **Action Recognition & Detection** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「EgoAction」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Action Recognition & Detection**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_ego_2605_24496_egoaction-egocentric-action-composition.md`](../../sources/papers/sun_awesome_ego_2605_24496_egoaction-egocentric-action-composition.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2605.24496>
- [arXiv PDF](https://arxiv.org/pdf/2605.24496)：已通读方法部分（实验与附录以略读为准）：Abstract、§1 Introduction、§2 Methodology（2.1–2.6）；§3 实验略读

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2605.24496)
