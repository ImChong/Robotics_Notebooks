---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2406.09455"
venue: "arXiv 2024"
code: https://github.com/maitrix-org/Pandora
summary: "General world model with natural language actions and video states."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2406_09455_pandora-towards-general-world-model-with.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Pandora

**Pandora: Towards General World Model with Natural Language Actions and Video States** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 150/571** 篇，分组 **47 General Foundation World Models**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

General world model with natural language actions and video states.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- General world model with natural language actions and video states.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph STEP["单步生成 §2.1"]
        A["前一状态 S(t-1) 图像或视频"] --> B["Vision encoder + Q-Former 适配器"]
        C["自然语言动作 A(t-1)"] --> D["文本 token embedding"]
        Q["可学习 query embedding"] --> E
        B --> E["自回归骨干 LLM<br/>Chat-UniVi Vicuna-7B"]
        D --> E
        E --> F["输出嵌入经 Q-Former 适配器"]
        F --> G["视频生成器<br/>DynamiCrafter 每次生成 16 帧"]
        H["S(t-1) 最后 4 帧 + FPS 数"] --> G
        G --> I["下一状态 St 视频片段"]
        I -->|"自回归续写 可随时换新动作"| A
    end
    subgraph TRAIN["两阶段训练 §2.2"]
        T1["预训练对齐<br/>WebVid-10M 视频-文本对<br/>冻结 LLM 与视频模型<br/>只训适配器与 query embedding<br/>L2 对齐 LLM 输出与视频模型文本嵌入"] --> T2["指令微调<br/>含动作描述的高质量视频数据<br/>扩散损失 微调视频生成器与 query embedding"]
    end
    TRAIN -.-> STEP
```

图依据：论文 §2.1–§2.2 / Fig.2、Table 1（Fig.2 原图仅据正文描述核对）

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 150/571 |
| 分组 | 47 General Foundation World Models |
| 出处 | arXiv 2024 |
| 论文 | <https://arxiv.org/abs/2406.09455> |
| 代码/项目 | <https://github.com/maitrix-org/Pandora> |

## 核心机制（归纳）

### 策展导读要点

General world model with natural language actions and video states.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2406.09455) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **47 General Foundation World Models** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Pandora」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **47 General Foundation World Models**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2406_09455_pandora-towards-general-world-model-with.md`](../../sources/papers/sun_awesome_wm_2406_09455_pandora-towards-general-world-model-with.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2406.09455>
- [arXiv PDF](https://arxiv.org/pdf/2406.09455)：已通读方法部分（实验与附录以略读为准）：§1 贡献概述、§2.1 模型架构、§2.2 分阶段训练（预训练与指令微调及数据构成）；§3 实验与附录以略读/未读为准


## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2406.09455)
