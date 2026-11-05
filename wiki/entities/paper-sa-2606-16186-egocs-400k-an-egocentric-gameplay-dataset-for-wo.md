---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2606.18180"
venue: "arXiv 2026"
summary: "A large-scale replay-grounded egocentric Counter-Strike dataset for world models, built from public professional CS match demos preserving human gameplay trajectories"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2606_16186_egocs-400k-an-egocentric-gameplay-datase.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# EgoCS-400K

**EgoCS-400K: An Egocentric Gameplay Dataset for World Models** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 105/249** 篇，分组 **28 Egocentric World Models**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

A large-scale replay-grounded egocentric Counter-Strike dataset for world models, built from public professional CS match demos preserving human gameplay trajectories

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- A large-scale replay-grounded egocentric Counter-Strike dataset for world models, built from public professional CS match demos preserving human gameplay trajectories
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["采集 HLTV 公开职业 CS:GO / CS2 demo"] --> B["按 demo+round+player+tick 区间<br/>用 CS Demo Manager 渲染第一人称视频"]
    B --> C["渲染视频过滤：去除观战视角、录屏失败、<br/>视角缺失或时长与 tick 区间不符"]
    A --> D["demoparser2 解析 demo<br/>逐 tick 玩家状态 ticks.csv"]
    D --> E["键鼠重建 + 规则检测原子动作 events.csv"]
    E --> F["动作时间线 → 合并 cut-protected 保护链"]
    F --> G["动态规划切分训练段 dp_segments.json"]
    C --> H["先验引导 VLM 字幕<br/>动作 / 移动 / 相机先验 + 片段视频"]
    G --> H
    H --> I["结构化 JSON：scene_draft、long_prompt、<br/>confidence、flags"]
    I --> J["发布：>400K 视频 / >10K 小时 / 多粒度标注"]
```

图依据：数据采集与标注流程图，论文 §3（Overview、§3.1 Demo Collection/Rendering/Filtering、§3.2 Parsing and Segmentation、§3.3 Prior-Guided VLM Captioning）与 Fig. 2；规模数字见 Table 2。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 105/249 |
| 分组 | 28 Egocentric World Models |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2606.18180> |

## 核心机制（归纳）

### 策展导读要点

A large-scale replay-grounded egocentric Counter-Strike dataset for world models, built from public professional CS match demos preserving human gameplay trajectories

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2606.18180) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **28 Egocentric World Models** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「EgoCS-400K」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **28 Egocentric World Models**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_ego_2606_16186_egocs-400k-an-egocentric-gameplay-datase.md`](../../sources/papers/sun_awesome_ego_2606_16186_egocs-400k-an-egocentric-gameplay-datase.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2606.18180>
- 编号核对（2026-11-02）：上游清单登记的 2606.16186 经 arXiv 官方记录核对为《MOSAIC: Mobile Object Segmentation under Adverse Imaging Conditions for Rapid L-PBF Keyhole Behavior Characterization》，与本页论文不符；已更正为 2606.18180《EgoCS-400K: An Egocentric Gameplay Dataset for World Models》。
- [arXiv PDF](https://arxiv.org/pdf/2606.18180)：已通读 §1 引言、§2 相关工作、§3 方法（§3.1–§3.3）与 §4 分析与局限（实验与附录以略读为准）；流程图依据论文 §3 与 Fig. 2。

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2606.18180)
