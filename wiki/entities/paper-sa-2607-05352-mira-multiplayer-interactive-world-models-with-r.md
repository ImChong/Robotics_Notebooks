---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2607.05352"
venue: "arXiv 2026"
code: https://github.com/mira-wm/mira
summary: "Multiplayer interactive world models with representation autoencoders."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2607_05352_mira-multiplayer-interactive-world-model.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# MIRA

**MIRA: Multiplayer Interactive World Models with Representation Autoencoders** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 175/571** 篇，分组 **52 Game & Open-World Exploration Application**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Multiplayer interactive world models with representation autoencoders.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Multiplayer interactive world models with representation autoencoders.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TB
  V["4 名玩家各自的视频流<br/>(2v2 Rocket League)"] --> ENC["冻结 Codec 编码器<br/>DINOv3-L/16 多层特征聚合<br/>+ 线性 Space-Time Bottleneck (2x2x2, 通道 1024→32)"]
  ENC --> Z["每玩家 latent z (10 Hz)<br/>按高度拼成一个联合 latent 网格 (Tiled views)"]
  A["4 名玩家的键盘动作 a<br/>每玩家 embedding 后按固定顺序 concat<br/>(训练时 Action dropout)"] --> C["条件向量 c = emb(a) + emb(flow time τ)"]
  Z --> DF["Diffusion Forcing<br/>逐帧独立 τ，加噪得 zτ"]
  DF --> WM["Flow-matching 世界模型 vθ<br/>AdaLN 空间-时间 Transformer x N<br/>空间注意力(帧内双向) + 时间注意力(因果)"]
  C -->|"AdaLN 缩放/偏移"| WM
  WM --> LOSS["Flow-matching 损失<br/>||vθ − (z1 − z0)||²"]
  WM -.->|"PSD 少步蒸馏"| FS["少步采样器 (1~2 次函数评估)"]
  FS --> ROLL["自回归 rollout + KV cache<br/>窗口 T=20 latent，每步追加噪声 latent 去噪"]
  ROLL --> DEC["Codec 解码器<br/>Spatial/Temporal Upsample + 因果 ST-ViT<br/>→ 20 fps 视频"]
  DEC --> OUT["各玩家视角视频，浏览器实时展示"]
```

图依据：论文 §4.1–4.5、§5 / Fig.3、Fig.4

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 175/571 |
| 分组 | 52 Game & Open-World Exploration Application |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2607.05352> |
| 代码/项目 | <https://github.com/mira-wm/mira> |

## 核心机制（归纳）

### 策展导读要点

Multiplayer interactive world models with representation autoencoders.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2607.05352) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **52 Game & Open-World Exploration Application** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「MIRA」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **52 Game & Open-World Exploration Application**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2607_05352_mira-multiplayer-interactive-world-model.md`](../../sources/papers/sun_awesome_wm_2607_05352_mira-multiplayer-interactive-world-model.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2607.05352>
- [arXiv PDF](https://arxiv.org/pdf/2607.05352)：已通读方法部分（实验与附录以略读为准）：§4 Method（4.1–4.5）、§5 Streaming inference

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2607.05352)
