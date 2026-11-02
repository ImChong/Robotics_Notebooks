---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2603.23376"
venue: "arXiv 2026"
summary: "14B Diffusion Transformer generating visually realistic, physically plausible, and action-controllable videos."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2603_23376_abot-physworld-interactive-world-foundat.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
  - ../../sources/repos/abot-physworld.md
---

# ABot-PhysWorld

**ABot-PhysWorld: Interactive World Foundation Model for Robotic Manipulation with Physics Alignment** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 070/571** 篇，分组 **42 Visual / Video World Models**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

14B Diffusion Transformer generating visually realistic, physically plausible, and action-controllable videos.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- 14B Diffusion Transformer generating visually realistic, physically plausible, and action-controllable videos.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart LR
  RAW["五个开源数据集，约 300 万片段\nAgiBot / RoboCoin / RoboMind / Galaxea / OXE"] --> FIL["具身数据过滤\n质量门 → 光流运动 → CLIP 时间连贯 → 视觉-动作对齐"]
  FIL --> BAL["分层分布均衡\n头部任务 8-15% / 中部 40-50% / 长尾全保留"]
  BAL --> CAP["物理感知四阶段描述\nQwen3-VL 感知 → Qwen3 写作"]
  CAP --> SFT["Stage 1：全量微调 Wan2.1-I2V-14B\n预测未来帧"]
  SFT --> CAND["同一提示生成 N 个候选视频"]
  CAND --> PROP["Proposer：Qwen3-VL 32B Thinking\n生成任务专属物理 checklist"]
  PROP --> SCORE["Scorer：Gemini 3 Pro\n锦标赛采样选出最优 y_w / 最差 y_l"]
  SCORE --> DPO["Stage 2：Diffusion-DPO\nLoRA rank 64，冻结 DiT"]
  DPO --> MODEL["物理对齐的世界模型"]
  ACT["动作向量 a ∈ R^7\n投影为 2D action map"] --> CTX["并行 context blocks\n每隔 5 个 DiT 块复制一个"]
  CTX -->|"零初始化卷积，残差相加"| MAIN["主 DiT：动作条件视频生成"]
```

图依据：论文 §2、§3.1–3.3 / Fig.1、Fig.2、Fig.4

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 070/571 |
| 分组 | 42 Visual / Video World Models |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2603.23376> |

## 核心机制（归纳）

### 策展导读要点

14B Diffusion Transformer generating visually realistic, physically plausible, and action-controllable videos.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2603.23376) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **42 Visual / Video World Models** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「ABot-PhysWorld」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **42 Visual / Video World Models**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [ABot-PhysWorld 源码归档](../../sources/repos/abot-physworld.md)（<https://github.com/amap-cvlab/ABot-PhysWorld>）

- [`sources/papers/sun_awesome_wm_2603_23376_abot-physworld-interactive-world-foundat.md`](../../sources/papers/sun_awesome_wm_2603_23376_abot-physworld-interactive-world-foundat.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2603.23376>
- [arXiv PDF](https://arxiv.org/pdf/2603.23376)：已通读方法部分（实验与附录以略读为准）：Fig.1–2 图注、§2 数据整理（2.1–2.2）、§3 方法（3.1–3.3）

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2603.23376)
