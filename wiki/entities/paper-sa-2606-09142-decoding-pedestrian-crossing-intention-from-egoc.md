---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2606.09142"
venue: "arXiv 2026"
summary: "Uses VLM to decode pedestrian crossing intention for the first time, with accuracy improved by 14.5% after fusing eye movement and ego-motion"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2606_09142_decoding-pedestrian-crossing-intention-f.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# Decoding Pedestrian Crossing Intention from Egocentric Vision via Vision Lang...

**Decoding Pedestrian Crossing Intention from Egocentric Vision via Vision Language Models** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 176/249** 篇，分组 **Advanced Reasoning & Cutting-edge Advances**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Uses VLM to decode pedestrian crossing intention for the first time, with accuracy improved by 14.5% after fusing eye movement and ego-motion

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- Uses VLM to decode pedestrian crossing intention for the first time, with accuracy improved by 14.5% after fusing eye movement and ego-motion
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["2 秒第一人称视频片段 V"] --> B["GroundingDINO 检测车辆与白色穿越圈"]
    B --> C["几何过滤 / 跨标签抑制 / SORT 时序跟踪"]
    C --> D["Set-of-Mark 叠加带持久 ID 的视觉提示"]
    D --> G["VLM（Qwen3-VL / InternVL3 / VLP）"]
    X["可选上下文 C：ego motion、vehicle motion、eye gaze、个人属性"] --> P["拼入提示：附在末尾或与对应输入并置"]
    Q["封闭式 VQA 问题：(A) cross / (B) yield，标准或 CoT 提示"] --> G
    P --> G
    G --> Y["预测未来 1 秒意图 cross / yield"]
    S["LoRA 监督微调：语言 / cross-modal 模块，视觉编码器冻结"] -.-> G
```

图依据：论文 §III（A 问题定义至 D 上下文表示及微调部分）/ Fig.1、Fig.2

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 176/249 |
| 分组 | Advanced Reasoning & Cutting-edge Advances |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2606.09142> |

## 核心机制（归纳）

### 策展导读要点

Uses VLM to decode pedestrian crossing intention for the first time, with accuracy improved by 14.5% after fusing eye movement and ego-motion

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2606.09142) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **Advanced Reasoning & Cutting-edge Advances** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Decoding Pedestrian Crossing Intention from Egocentric Vision via Vision Lang...」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_ego_2606_09142_decoding-pedestrian-crossing-intention-f.md`](../../sources/papers/sun_awesome_ego_2606_09142_decoding-pedestrian-crossing-intention-f.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2606.09142>
- [arXiv PDF](https://arxiv.org/pdf/2606.09142)：已通读方法部分（实验与附录以略读为准）：摘要、§III 方法（问题定义、VQA 构造与提示、视觉提示、上下文表示、LoRA 微调）、§IV 实验设定；§V 结果略读

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2606.09142)
