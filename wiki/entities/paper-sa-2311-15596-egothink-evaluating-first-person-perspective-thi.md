---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2311.15596"
venue: "CVPR 2024"
summary: "Introduces a novel visual question-answering benchmark that encompasses six core capabilities with twelve detailed dimensions, constructed using selected clips from egocentric videos with manually annotated QA pairs"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2311_15596_egothink-evaluating-first-person-perspec.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# EgoThink

**EgoThink: Evaluating First-Person Perspective Thinking Capability of Vision-Language Models** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 184/249** 篇，分组 **Advanced Reasoning & Cutting-edge Advances**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Introduces a novel visual question-answering benchmark that encompasses six core capabilities with twelve detailed dimensions, constructed using selected clips from egocentric videos with manually annotated QA pairs

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- Introduces a novel visual question-answering benchmark that encompasses six core capabilities with twelve detailed dimensions, constructed using selected clips from egocentric videos with manually annotated QA pairs
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    EGO4D["Ego4D 第一人称视频子集：逐帧抽取，约每数十帧采样一张图"] --> FILTER["图像筛选：剔除不清晰或无第一人称特征的图；每个视频至多保留 2 张"]
    FILTER --> CAND["第一人称候选图像"]
    CAND --> ANNO["6 名标注员手工标注问答对：每人负责 2 个维度，选中的图像从候选集移除以免重复"]
    ANNO --> REVIEW["另 3 名标注员复核：三人一致认为符合维度定义才保留"]
    REVIEW --> BENCH["EgoThink 基准：6 类能力 / 12 个细分维度（Object、Activity、Localization、Reasoning、Forecasting、Planning）"]
    BENCH --> ZS["零样本评测 VLM：输入第一人称图像 + 问题，生成开放式回答"]
    ZS --> JUDGE["GPT-4 作为自动评判（single-answer grading）：问题、模型输出、参考答案组成提示"]
    JUDGE --> SCORE["每条回答打分 0 / 0.5 / 1（错 / 部分正确 / 正确）"]
    SCORE -.-> AGREE["§5.2 一致性检验：GPT-3.5-Turbo、Claude-2、人工作为对照评判者"]
```

图依据：数据采集与评测协议流程图，依据论文 §3.1、§3.2（Fig.2）、§4.1 single-answer grading、§5.2。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 184/249 |
| 分组 | Advanced Reasoning & Cutting-edge Advances |
| 出处 | CVPR 2024 |
| 论文 | <https://arxiv.org/abs/2311.15596> |

## 核心机制（归纳）

### 策展导读要点

Introduces a novel visual question-answering benchmark that encompasses six core capabilities with twelve detailed dimensions, constructed using selected clips from egocentric videos with manually annotated QA pairs

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2311.15596) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **Advanced Reasoning & Cutting-edge Advances** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「EgoThink」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Advanced Reasoning & Cutting-edge Advances**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_ego_2311_15596_egothink-evaluating-first-person-perspec.md`](../../sources/papers/sun_awesome_ego_2311_15596_egothink-evaluating-first-person-perspec.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2311.15596>
- [arXiv PDF](https://arxiv.org/pdf/2311.15596)：已读摘要、§1、§3（3.1 核心能力、3.2 数据采集与统计）、§4.1 实验设定（含 single-answer grading）、§5.2 评判者一致性、§6；§4.2 结果与附录略读

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2311.15596)
