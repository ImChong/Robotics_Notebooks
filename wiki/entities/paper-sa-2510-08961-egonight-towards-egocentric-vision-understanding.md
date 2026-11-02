---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2510.06218"
venue: "arXiv 2025"
summary: "The first comprehensive benchmark for nighttime egocentric vision with VQA as the core task"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2510_08961_egonight-towards-egocentric-vision-under.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# EgoNight

**EgoNight: Towards Egocentric Vision Understanding at Night with a Challenging Benchmark** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 139/249** 篇，分组 **293 Procedural Question Answering**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

The first comprehensive benchmark for nighttime egocentric vision with VQA as the core task

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- The first comprehensive benchmark for nighttime egocentric vision with VQA as the core task
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    S1["EgoNight-Synthetic：Infinigen 生成室内场景，Blender 沿同一轨迹渲染日/夜视频对（50 对）"] --> TAX
    S2["EgoNight-Sofia：真人佩戴相机录制日/夜配对视频（20 对），夜间版回放日间视频作引导"] --> TAX
    S3["EgoNight-Oxford：采用 Oxford Day-Night 的日夜视频（20 个）"] --> TAX
    TAX["按 12 类 QA 任务分类组织：成对类型 8 类，非成对类型 4 类"] --> C1["阶段 1 夜间字幕：GPT-4.1 按 QA 类型提示生成字幕"]
    C1 --> C2["阶段 2 夜间问题生成：字幕加夜间片段，生成候选问题"]
    C2 --> J{"成对类型且有对齐的日间片段 ?"}
    J -->|"是"| A1["阶段 3 日间增强伪答案：参考对齐的日间片段合成答案"]
    J -->|"否"| A2["阶段 3 直接由夜间片段生成答案"]
    A1 --> H["人工精修：删除、修改、新增，每条 QA 至少人工核验一次"]
    A2 --> H
    H --> V["EgoNight-VQA：3658 条 QA"]
```

图依据：论文 §3.1 / §3.2 / Fig.2。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 139/249 |
| 分组 | 293 Procedural Question Answering |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2510.06218> |

## 核心机制（归纳）

### 策展导读要点

The first comprehensive benchmark for nighttime egocentric vision with VQA as the core task

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2510.06218) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **293 Procedural Question Answering** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「EgoNight」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_ego_2510_08961_egonight-towards-egocentric-vision-under.md`](../../sources/papers/sun_awesome_ego_2510_08961_egonight-towards-egocentric-vision-under.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2510.06218>
- [arXiv PDF](https://arxiv.org/pdf/2510.06218)：已通读 §3.1 视频来源与 §3.2 VQA 构建流程（§3.3 与实验、附录未读）
- 编号核对（2026-11-02）：上游清单登记的 2510.08961 经 arXiv 官方记录核对为《Stability Conditions and Algebraic Hearts for Acyclic Quivers》，与本页论文不符；已更正为 2510.06218《EgoNight: Towards Egocentric Vision Understanding at Night with a Challenging Benchmark》。

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2510.06218)
