---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2602.04600"
venue: "arXiv 2026"
summary: "Introduces CoMe-VLA, a cognitive and memory-aware VLA framework leveraging large-scale human egocentric data to learn versatile exploration and manipulation priors"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2602_04600_act-sense-act-learning-non-markovian-act.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# Act, Sense, Act

**Act, Sense, Act: Learning Non-Markovian Active Perception Strategies from Large-Scale Egocentric Human Data** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 068/249** 篇，分组 **27 Embodied AI & Vision-Language-Action Ego-VLA**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Introduces CoMe-VLA, a cognitive and memory-aware VLA framework leveraging large-scale human egocentric data to learn versatile exploration and manipulation priors

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- Introduces CoMe-VLA, a cognitive and memory-aware VLA framework leveraging large-scale human egocentric data to learn versatile exploration and manipulation priors
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
  subgraph IN["输入"]
    V["第一视角视觉：当前帧 + 过去 5 秒内 5 帧历史"]
    TX["子任务文本，如 find the croissant"]
    CT["可学习 [COG] 认知 token"]
    PR["时序本体状态（proprioception）"]
  end
  V --> VLM["预训练 VLM：Qwen3-VL-2B（GR00T 风格，无 LM head）"]
  TX --> VLM
  CT --> VLM
  PR --> PM["Proprioceptive Memory Encoder（Transformer）"]
  VLM -->|"[COG] 隐状态"| CH["Cognitive Auxiliary Head（MLP）→ 二值标签 c_t：信息是否已足够"]
  CH -->|"c_t = 1"| SW["切换子任务文本（探索 → 利用）"]
  SW -.->|"重新条件化"| TX
  VLM -->|"视觉-语义上下文"| AD["流匹配 Action Decoder：每层对两路记忆做 Cross-Attention"]
  PM -->|"本体记忆上下文"| AD
  AD --> OUT["29 维动作块 A_t：视点 pos+rot6d、双臂末端 pos+rot6d、双夹爪"]
```

图依据：论文 §5 / Fig. 3（CoMe-VLA 概览）、§5.1 模型结构、§5.2 三阶段训练（阶段 1：认知预训练；阶段 2：认知-动作联合预训练；阶段 3：机器人数据微调，三阶段未画入图中）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 068/249 |
| 分组 | 27 Embodied AI & Vision-Language-Action Ego-VLA |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2602.04600> |

## 核心机制（归纳）

### 策展导读要点

Introduces CoMe-VLA, a cognitive and memory-aware VLA framework leveraging large-scale human egocentric data to learn versatile exploration and manipulation priors

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2602.04600) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **27 Embodied AI & Vision-Language-Action Ego-VLA** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Act, Sense, Act」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **27 Embodied AI & Vision-Language-Action Ego-VLA**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_ego_2602_04600_act-sense-act-learning-non-markovian-act.md`](../../sources/papers/sun_awesome_ego_2602_04600_act-sense-act-learning-non-markovian-act.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2602.04600>
- [arXiv PDF](https://arxiv.org/pdf/2602.04600)：已通读方法部分（实验与附录以略读为准）：摘要、§1 引言、§5 模型设计（5.1 架构、5.2 训练策略，含 Fig. 3）、§6.1 实现细节

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2602.04600)
