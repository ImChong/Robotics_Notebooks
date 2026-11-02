---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2402.15391"
venue: "arXiv 2024"
summary: "Google DeepMind's generative interactive environments."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2402_15391_genie-generative-interactive-environment.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Genie

**Genie: Generative Interactive Environments** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 148/571** 篇，分组 **47 General Foundation World Models**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Google DeepMind's generative interactive environments.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Google DeepMind's generative interactive environments.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart LR
  subgraph TRAIN["训练 (论文 §2，Fig. 3)"]
    V["无标注互联网视频 x_1:T"] --> TOK["视频分词器 Video Tokenizer\nVQ-VAE + ST-transformer，得到离散 token z"]
    V --> LAM["潜在动作模型 LAM\n输入历史帧与下一帧，输出离散潜在动作 a~ (|A| = 8)"]
    TOK --> DYN["动力学模型 Dynamics Model\n解码器式 MaskGIT transformer"]
    LAM -->|"潜在动作作为加性嵌入"| DYN
    DYN --> LOSS["交叉熵：预测 token z^_2:T 对真实 token"]
  end
  subgraph INF["推理：动作可控的视频生成 (Fig. 8)"]
    P["提示图像 x_1"] --> T1["分词器编码为 z_1"]
    U["玩家选择离散潜在动作 a_1"] --> CB["查 VQ 码本得到 a~_1"]
    T1 --> D2["动力学模型预测下一帧 token"]
    CB --> D2
    D2 --> DEC["分词器解码器还原帧 x^_2"]
    DEC -->|"自回归继续"| D2
  end
```

图依据：论文 §2 Methodology（LAM、Video Tokenizer、Dynamics Model）、§2.2 Inference 与 Fig. 3 / Fig. 8；训练分两阶段：先训分词器，再联合训 LAM 与动力学模型，推理时丢弃 LAM。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 148/571 |
| 分组 | 47 General Foundation World Models |
| 出处 | arXiv 2024 |
| 论文 | <https://arxiv.org/abs/2402.15391> |
| 项目页 | <https://sites.google.com/view/genie-2024/home> |

## 核心机制（归纳）

### 策展导读要点

Google DeepMind's generative interactive environments.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2402.15391) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **47 General Foundation World Models** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Genie」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_wm_2402_15391_genie-generative-interactive-environment.md`](../../sources/papers/sun_awesome_wm_2402_15391_genie-generative-interactive-environment.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2402.15391>
- [arXiv PDF](https://arxiv.org/pdf/2402.15391)：已通读方法部分（实验与附录以略读为准）：§2 Methodology 与 §2.2 Inference（训练设定与实验仅略读）

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2402.15391)
