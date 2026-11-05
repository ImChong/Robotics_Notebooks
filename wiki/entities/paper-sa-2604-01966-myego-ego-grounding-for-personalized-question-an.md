---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2604.01966"
venue: "CVPR 2026"
summary: "Large-scale personalized first-person video dataset for long-form QA"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2604_01966_myego-ego-grounding-for-personalized-que.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# MyEgo

**MyEgo: Ego-Grounding for Personalized Question-Answering in Egocentric Videos** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 046/249** 篇，分组 **24 Long-form Video Understanding & Structured Reasoning**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Large-scale personalized first-person video dataset for long-form QA

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- Large-scale personalized first-person video dataset for long-form QA
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["视频收集：Ego4D（去掉单人视频）/ EgoLife / CASTEL2024<br/>拼接、去水印、裁剪 → 541 段，平均 9.2 分钟"] --> B["人工 QA 标注：10 名学生<br/>原则 Egocentric / Personalized / Visual Answer<br/>标注 question moment 与 answer moment"]
    B --> C["5,012 条开放式 QA"]
    C --> D["多选扩展：Gemini-2.5 Pro 生成 4 个干扰项<br/>yes/no 题为 2 选项"]
    D --> E["偏差过滤：仅给视频帧与选项（不含问题）<br/>Gemini-2.5 Pro 与 GPT-5 都能答对的题，人工改写干扰项"]
    E --> F["再次人工检查：2 名学生 + 作者"]
    F --> G["MyEgo：5,012 题，每题含开放式与多选两种形式"]
    G --> H["评测 §4.2：开放式由 GPT-5 mini 判对错并打 0-5 分<br/>多选看准确率；模型 + 2 名学生人类基线"]
    H --> I["§4.3 受控分析：Q&A 时刻感知采样 · 随时间的准确率 · 帧数变化 · prompt 修改"]
```

图依据：数据集构建与评测协议图，论文 §3.1–§3.2（Task Definition、Dataset Construction）、§4.1–§4.3。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 046/249 |
| 分组 | 24 Long-form Video Understanding & Structured Reasoning |
| 出处 | CVPR 2026 |
| 论文 | <https://arxiv.org/abs/2604.01966> |

## 核心机制（归纳）

### 策展导读要点

Large-scale personalized first-person video dataset for long-form QA

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2604.01966) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **24 Long-form Video Understanding & Structured Reasoning** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「MyEgo」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_ego_2604_01966_myego-ego-grounding-for-personalized-que.md`](../../sources/papers/sun_awesome_ego_2604_01966_myego-ego-grounding-for-personalized-que.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2604.01966>
- [arXiv PDF](https://arxiv.org/pdf/2604.01966)：已通读摘要、Introduction、§3 MyEgo Dataset（任务定义、构建、统计）、§4.1–§4.2 评测设置与 §4.3 受控分析的文字部分；结果表与附录未逐项核对。

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2604.01966)
