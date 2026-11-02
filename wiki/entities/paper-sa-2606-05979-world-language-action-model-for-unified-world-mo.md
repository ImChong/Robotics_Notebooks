---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2606.05979"
venue: "arXiv 2026"
summary: "Proposes world-language-action (WLA) models as a new class of embodied foundation models. WLA takes textual instructions, images, and robot states as inputs to jointly predict textual subtasks, subgoal images, and robot"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2606_05979_world-language-action-model-for-unified.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# World-Language-Action Model for Unified World Modeling, Language Reasoning, a...

**World-Language-Action Model for Unified World Modeling, Language Reasoning, and Action Synthesis** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 117/249** 篇，分组 **28 Egocentric World Models**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Proposes world-language-action (WLA) models as a new class of embodied foundation models. WLA takes textual instructions, images, and robot states as inputs to jointly predict textual subtasks, subgoal images, and robot actions. Leverages extensive egocentric videos for world modeling and language reasoning capaciti...

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- Proposes world-language-action (WLA) models as a new class of embodied foundation models. WLA takes textual instructions, images, and robot states as inputs to jointly predict textual subtasks, subgoal images, and robot actions. Leverages extensive egocentric videos for world modeling and language reasoning capaciti...
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    IN["输入：当前观测 o_t、历史观测 o_t-h、指令 ℓ、本体状态 q_t、记忆 M"] --> AR["AR Transformer 骨干（VLM 初始化）"]
    AR --> TXT["文本意图：预测未来子任务窗口 S_t（并更新记忆 M）"]
    TXT -.-> M2["M ← M ⊕ 已预测子任务"]
    Q["meta-queries Q（因果注意力聚合上下文）"] --> AR
    AR --> H["物理动态表征 h_t（隐式 latent action）"]
    H --> ACT["Action Expert：a_t:t+n = f_act(h_t, q_t)"]
    H --> WM["World Expert：预测未来帧 o_t+n 的 VAE 特征（推理时可关闭）"]
    ACT --> EXEC["执行 n 步 action chunk，滚动时域重复"]
    WM -.->|"TTS 模式"| VAL["Value Model 对 K 个候选 chunk 的想象未来帧打分"]
    VAL -.-> EXEC
```

图依据：论文 §3 方法（3.1 World-Language-Action Models）/ Fig.2、Inference Optimization（TTS 模式）

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 117/249 |
| 分组 | 28 Egocentric World Models |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2606.05979> |

## 核心机制（归纳）

### 策展导读要点

Proposes world-language-action (WLA) models as a new class of embodied foundation models. WLA takes textual instructions, images, and robot states as inputs to jointly predict textual subtasks, subgoal images, and robot actions. Leverages extensive egocentric videos for world modeling and language reasoning capaciti...

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2606.05979) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **28 Egocentric World Models** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「World-Language-Action Model for Unified World Modeling, Language Reasoning, a...」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_ego_2606_05979_world-language-action-model-for-unified.md`](../../sources/papers/sun_awesome_ego_2606_05979_world-language-action-model-for-unified.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2606.05979>
- [arXiv PDF](https://arxiv.org/pdf/2606.05979)：已通读方法部分（实验与附录以略读为准）：摘要、§3 方法（3.1 含文本意图、物理动态建模、训练目标）、推理优化（Efficient / TTS 模式）；实验与附录略读

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2606.05979)
