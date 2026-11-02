---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2512.15692"
venue: "arXiv 2025"
summary: "Video-action models for generalizable robot control beyond VLAs."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2512_15692_mimic-video-video-action-models-for-gene.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# mimic-video

**mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 218/571** 篇，分组 **62 Joint WAM & Unified VLA-World Models**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Video-action models for generalizable robot control beyond VLAs.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Video-action models for generalizable robot control beyond VLAs.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
  IN["输入: 过去帧上下文 (5 帧) + 语言指令 l + 本体状态 q"] --> VM["视频模型 Cosmos-Predict2 (2B latent DiT)\n3D tokenizer 编码, T5 语言 cross-attention"]
  VM --> PART["部分去噪: 从噪声积分到中间流时间 τv\n(默认 τv = 1, 仅需一次前向)"]
  PART --> HID["取前 k 层隐藏状态 h(τv)\n作为潜在视频计划"]
  HID --> AD["动作解码器 (DiT, 充当逆动力学模型 IDM)\ncross-attend 视频表示, 并编码本体状态 + 带噪动作"]
  AD --> DEN["以独立动作流时间 τa 完整去噪"]
  DEN --> ACT["动作块 A_t"]
  S1["训练阶段 1: LoRA 微调视频骨干\n(机器人视频数据)"] -.-> VM
  S2["训练阶段 2: 冻结骨干, 从零训练动作解码器\n(flow matching, 独立采样 τv 与 τa)"] -.-> AD
```

图依据：论文 §IV（架构：视频模型 / 动作解码器 / 动作采样 / 训练）、Fig.3、Algorithm 1–2

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 218/571 |
| 分组 | 62 Joint WAM & Unified VLA-World Models |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2512.15692> |
| 项目页 | <https://mimic-video.github.io> |

## 核心机制（归纳）

### 策展导读要点

Video-action models for generalizable robot control beyond VLAs.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2512.15692) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **62 Joint WAM & Unified VLA-World Models** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「mimic-video」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_wm_2512_15692_mimic-video-video-action-models-for-gene.md`](../../sources/papers/sun_awesome_wm_2512_15692_mimic-video-video-action-models-for-gene.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2512.15692>
- [arXiv PDF](https://arxiv.org/pdf/2512.15692)：已通读方法部分（实验与附录以略读为准）：摘要、§I–§II 略读、§IV（Preliminaries、视频模型、动作解码器、动作采样、训练）、§V 实验设定

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2512.15692)
