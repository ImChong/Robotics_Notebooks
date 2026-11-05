---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2506.23919"
venue: "arXiv 2025"
summary: "Image-generative VLMs as object-centric world models empowering zero-shot manipulation."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2506_23919_goal-vla-image-generative-vlms-as-object.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Goal-VLA

**Goal-VLA: Image-Generative VLMs as Object-Centric World Models Empowering Zero-shot Robot Manipulation** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 196/571** 篇，分组 **62 Joint WAM & Unified VLA-World Models**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Image-generative VLMs as object-centric world models empowering zero-shot manipulation.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Image-generative VLMs as object-centric world models empowering zero-shot manipulation.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    IN["输入：单视角 RGB-D 观测 O=(I, D) + 语言指令 L"] --> EP["Prompt 增强<br/>文本输出 VLM (Gemini 2.5 Pro) 结合 I 丰富指令 → L_e"]
    subgraph GSR["(a) Goal State Reasoning：Reflection-through-Synthesis 循环"]
        direction TB
        GEN["图像生成 VLM (Gemini 2.5 Flash-image)<br/>生成候选目标图"] --> SYN["Grounded SAM 分割目标物体<br/>半透明叠加到初始场景，得合成图"]
        SYN --> REF["Reflector VLM (Gemini 2.5 Pro)<br/>判断是否满足任务语义"]
        REF -->|"失败：修订 prompt"| GEN
    end
    EP --> GEN
    REF -->|"通过（或达最大次数）"| GOAL["目标图 I′"]
    GOAL --> DA["Depth Anything V2 → 目标深度 D′"]
    GOAL --> MK["Grounded SAM → 初始/目标物体掩码 M, M′"]
    subgraph SG["(b) Spatial Grounding"]
        direction TB
        SM["语义匹配（Geo-Aware 语义特征，余弦相似度最高像素配对）"] --> RS["在背景像素上回归 D′_rescaled = s·D′ + b 恢复尺度"]
        RS --> UM["点云配准（Umeyama）→ 物体变换 (R, t)"]
    end
    GOAL --> SM
    DA --> RS
    MK --> RS
    UM --> GP
    subgraph LLP["(c) Low-level Policy"]
        direction TB
        CM["Contact Module：表面法向采样候选 → 过滤/碰撞检查 → 打分选接触位姿"] --> GP["把 (R, t) 作用于接触位姿得抓取器目标位姿"]
        GP --> MP["Motion Planning（采样式规划器）→ 动作序列"]
    end
    IN --> CM
```

图依据：论文 §III-A 至 §III-D / Fig.2（三阶段框架）、Algorithm 1。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 196/571 |
| 分组 | 62 Joint WAM & Unified VLA-World Models |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2506.23919> |
| 项目页 | <https://nus-lins-lab.github.io/goalvlaweb/> |

## 核心机制（归纳）

### 策展导读要点

Image-generative VLMs as object-centric world models empowering zero-shot manipulation.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2506.23919) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **62 Joint WAM & Unified VLA-World Models** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Goal-VLA」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **62 Joint WAM & Unified VLA-World Models**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Highlights 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome World Models](../entities/awesome-world-models.md)
- 技术地图：[Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md)
- 方法/任务：[generative-world-models.md](../methods/generative-world-models.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/sun_awesome_wm_2506_23919_goal-vla-image-generative-vlms-as-object.md`](../../sources/papers/sun_awesome_wm_2506_23919_goal-vla-image-generative-vlms-as-object.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2506.23919>
- [arXiv PDF](https://arxiv.org/pdf/2506.23919)：已通读方法部分（实验与附录以略读为准）：§III Method（A–D）、Alg.1

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2506.23919)
