---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2607.02096"
venue: "arXiv 2026"
summary: "A benchmark constructed from long-form Ego4D videos containing 1,498 referring expressions with average video duration of 45 minutes, exhibiting extreme target sparsity and complex human-object interactions"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2607_02096_longegorefer-a-benchmark-for-long-form-e.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# LongEgoRefer

**LongEgoRefer: A Benchmark for Long-Form Egocentric Video Referring Expression Comprehension** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 044/249** 篇，分组 **24 Long-form Video Understanding & Structured Reasoning**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

A benchmark constructed from long-form Ego4D videos containing 1,498 referring expressions with average video duration of 45 minutes, exhibiting extreme target sparsity and complex human-object interactions

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- A benchmark constructed from long-form Ego4D videos containing 1,498 referring expressions with average video duration of 45 minutes, exhibiting extreme target sparsity and complex human-object interactions
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph BUILD["基准构建（半自动标注）"]
        A["EgoTracks 标注<br/>Ego4D 完整长视频"] --> B["数据过滤<br/>视频至少 20 分钟<br/>is_active 轨迹至少 10 秒"]
        B --> C["Gemini 2.5 Flash 阶段一<br/>生成 10 条目标描述"]
        C --> D["阶段二<br/>聚合精炼为单条 referring expression"]
        D --> E["人工四步审校<br/>验证过滤 / 修正 / 消歧 / 二审质检"]
        E --> F["LongEgoRefer<br/>1,498 条样本"]
    end
    subgraph BASE["训练无关两阶段基线"]
        Q["长视频 + 文本查询"] --> V["VLM 时间定位<br/>when：找出匹配片段"]
        V --> M["取片段中间帧"]
        M --> G["Grounding DINO<br/>检测目标得初始框"]
        G --> S["SAM2 跟踪<br/>得到时空轨迹"]
    end
    F -.评测.-> Q
```

图依据：论文 §3.2 Benchmark Creation / Fig.2、§4.2 Training-free Two-stage Pipeline

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 044/249 |
| 分组 | 24 Long-form Video Understanding & Structured Reasoning |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2607.02096> |

## 核心机制（归纳）

### 策展导读要点

A benchmark constructed from long-form Ego4D videos containing 1,498 referring expressions with average video duration of 45 minutes, exhibiting extreme target sparsity and complex human-object interactions

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2607.02096) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **24 Long-form Video Understanding & Structured Reasoning** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「LongEgoRefer」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_ego_2607_02096_longegorefer-a-benchmark-for-long-form-e.md`](../../sources/papers/sun_awesome_ego_2607_02096_longegorefer-a-benchmark-for-long-form-e.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2607.02096>
- [arXiv PDF](https://arxiv.org/pdf/2607.02096)：已通读方法部分（实验与附录以略读为准）：§1、§3.1–3.2、§4.1–4.2、§5.1

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2607.02096)
