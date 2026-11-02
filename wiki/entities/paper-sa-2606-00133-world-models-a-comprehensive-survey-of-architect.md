---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2606.00133"
venue: "arXiv 2026"
summary: "Comprehensive survey covering architectures, methodologies, reasoning paradigms, and applications of world models; proposes a multi-axis taxonomy organized along four dimensions."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2606_00133_world-models-a-comprehensive-survey-of-a.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# World Models

**World Models: A Comprehensive Survey of Architectures, Methodologies, Reasoning Paradigms, and Applications** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 546/571** 篇，分组 **Surveys & Overviews**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Comprehensive survey covering architectures, methodologies, reasoning paradigms, and applications of world models; proposes a multi-axis taxonomy organized along four dimensions.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Comprehensive survey covering architectures, methodologies, reasoning paradigms, and applications of world models; proposes a multi-axis taxonomy organized along four dimensions.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
  B["§2 Background and Conceptual Foundations<br/>2.1 Definition and Basic Concepts；2.2 Key components；2.3 与 model-free RL 的区别；2.4 Role of latent spaces"] --> X["多轴分类（摘要：四个互补维度）"]
  X --> A1["§3 按 Architecture 分类"]
  X --> A2["§4 按 Methodological Families 分类"]
  X --> A3["§5 按 Reasoning Strategy 分类"]
  X --> A4["§6 按 Application Domains 分类"]
  A1 --> A1a["3.1 representation：pixel、continuous latent、discrete token、joint-embedding、structured/object-centric<br/>3.2 dynamics：deterministic、stochastic、implicit generative、representation-space predictive、memory-augmented<br/>3.3 modality：visual-only、language-only、3D geometric、proprioceptive and tactile、multimodal fusion<br/>3.4 learning paradigms；3.5 downstream use"]
  A2 --> A2a["4.1 State-space and recurrent<br/>4.2 Transformer-based<br/>4.3 Diffusion-Based<br/>4.4 Physics-informed and structured<br/>4.5 Language-augmented and multi-modal"]
  A3 --> A3a["5.1 Imagination-based planning<br/>5.2 Policy learning with a world model<br/>5.3 Counterfactual reasoning<br/>5.4 Planning under uncertainty"]
  A4 --> A4a["6.1 Robotics；6.2 Autonomous driving；6.3 Video prediction；6.4 Multimodal agents；6.5 RL and Games；6.6 Scientific modeling；6.7 Medical images；6.8 Educational measurement；6.9 Business and Finance"]
  A1a --> E["§7 Evaluation Protocols and Benchmarks"]
  A2a --> E
  A3a --> E
  A4a --> E
  E --> C["§8 Major Challenges and Limitations"]
  C --> F["§9 Discussion and Future Directions"]
```

图依据：综述分类体系图，论文摘要（四维分类）、§1 末尾的轴向说明与 Contents 目录（§2–§9 的章节层次）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 546/571 |
| 分组 | Surveys & Overviews |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2606.00133> |

## 核心机制（归纳）

### 策展导读要点

Comprehensive survey covering architectures, methodologies, reasoning paradigms, and applications of world models; proposes a multi-axis taxonomy organized along four dimensions.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2606.00133) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **Surveys & Overviews** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「World Models」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Surveys & Overviews**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2606_00133_world-models-a-comprehensive-survey-of-a.md`](../../sources/papers/sun_awesome_wm_2606_00133_world-models-a-comprehensive-survey-of-a.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2606.00133>
- [arXiv PDF](https://arxiv.org/pdf/2606.00133)：已通读摘要、Contents 目录、§1 引言；§2–§9 正文以略读为准（图中小节名取自目录）。

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2606.00133)
