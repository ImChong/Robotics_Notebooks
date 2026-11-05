---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2506.05287"
venue: "arXiv 2025"
summary: "A benchmark designed to systematically evaluate object-centric embodied cognition in dynamic egocentric scenarios"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2506_05287_eoc-bench-can-mllms-identify-recall-and.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# EOC-Bench

**EOC-Bench: Can MLLMs Identify, Recall, and Forecast Objects in an Egocentric World?** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 136/249** 篇，分组 **293 Procedural Question Answering**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

A benchmark designed to systematically evaluate object-centric embodied cognition in dynamic egocentric scenarios

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- A benchmark designed to systematically evaluate object-centric embodied cognition in dynamic egocentric scenarios
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    VID["视频收集 §3.2.1：EPIC-KITCHENS（239）/ Ego4D（201）/ Charades-ego（294）/ MECCANO（12）+ 自录 110 个，共 656 个视频，统一裁成 3 到 10 分钟"]
    VID --> TAXO["能力分类 §3.2.2：Past / Present / Future 三个时间维度，共 11 个任务"]
    TAXO --> PAST["Past：OSR / OLR / ORE / ATP"]
    TAXO --> PRES["Present：ISR / OR / PFI / AP"]
    TAXO --> FUT["Future：TMP / SCP / DRP"]
    PAST --> QA["QA 构建 §3.2.3：10 名标注者，人在回路的混合格式标注：True/False、单选、多选、开放式（主要用于绝对时间戳）"]
    PRES --> QA
    FUT --> QA
    REF["视觉对象指代：point / box / mask 提示，标注在视频最后一帧"] --> QA
    QA --> CHECK["标注后筛查：标注者交叉检查与验证（视觉提示偏移、遗漏、选项歧义）"]
    CHECK --> BENCH["EOC-Bench：3,277 条 QA，11 个维度，3 类对象指代"]
    BENCH --> METRIC["指标 §3.2.4：T/F、SCA、MCA 用精确匹配 Accuracy；开放式用 MSTA（Multi-Scale Temporal Accuracy，容差阈值 1% / 10% / 20% / 30% 取平均）"]
    METRIC --> EVAL["评测对象 §4.1：专有 MLLM / 开源 MLLM / 对象级 MLLM，零样本推理，另有随机基线与 3 名志愿者的人类表现"]
```

图依据：基准构建与评测协议图（视频收集 → 能力分类 → 人在回路 QA 标注 → 指标 → 评测对象）：论文 Fig. 1、Fig. 2、§3.1–§3.3、§4.1–§4.2。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 136/249 |
| 分组 | 293 Procedural Question Answering |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2506.05287> |

## 核心机制（归纳）

### 策展导读要点

A benchmark designed to systematically evaluate object-centric embodied cognition in dynamic egocentric scenarios

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2506.05287) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **293 Procedural Question Answering** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「EOC-Bench」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **293 Procedural Question Answering**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_ego_2506_05287_eoc-bench-can-mllms-identify-recall-and.md`](../../sources/papers/sun_awesome_ego_2506_05287_eoc-bench-can-mllms-identify-recall-and.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2506.05287>
- [arXiv PDF](https://arxiv.org/pdf/2506.05287)：已通读方法部分（实验与附录以略读为准）：摘要、§1–§3（含 Fig. 1–3）、§4.1–§4.2 的评测设置与基线说明；Table 2–4 数字、§4.3 起的分析与附录以略读为准

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2506.05287)
