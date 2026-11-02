---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "2607.28243"
summary: "We present \\method, an egocentric world-action simulator that synthesizes controllable, high-quality manipulation videos to expand scarce real-world training data. \\method{} builds on a pretrained video generation prior"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_2607_28243_egogenesis-egocentric-world-action-model.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# EgoGenesis

**EgoGenesis: Egocentric World-Action Modeling with Online Anchored Projective Memory and Action-3D RoPE** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 017/564** 篇，分组 **Benchmarks & simulators**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

We present \method, an egocentric world-action simulator that synthesizes controllable, high-quality manipulation videos to expand scarce real-world training data. \method{} builds on a pretrained video generation prior and introduces two geometry-aware conditioning mechanisms.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- We present \method, an egocentric world-action simulator that synthesizes controllable, high-quality manipulation videos to expand scarce real-world training data. \method{} builds on a pretrained video generation prior and introduces two geometry-aware conditioning mechanisms.
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    IN["输入：首帧 I0、语言指令 y、相机内外参、动作几何（MANO / 灵巧手骨架 / 机器人末端轨迹）"] --> GEN["分块（chunkwise）因果 DiT：逐块 flow matching 去噪，首帧固定"]
    IN --> OAPM["OAPM：不可变首帧锚点 Ma + 可替换的近期快照 Mr，经 VGGT-Ω 编码得到 3D 场景记忆 Mb"]
    OAPM --> GCA["选定 DiT 层的门控 cross-attention：H ← H + GatedCrossAttn(Q, K, V)"]
    IN --> A3D["A3D-RoPE：骨架 / 末端 latent 带度量 3D 坐标，作为旋转相位用于骨架到视频的 cross-attention"]
    GCA --> GEN
    A3D --> GEN
    GEN --> COMMIT["Z_b 去噪完成 → 提交到因果 KV cache（供下一块使用）"]
    COMMIT --> REFRESH["每隔 s_r 块：解码最近帧并编码成新快照，仅替换近期槽 Mr，锚点不覆盖"]
    REFRESH --> OAPM
    COMMIT --> PAIR["生成的视频–动作配对轨迹"]
    PAIR --> AUG["与真实示教混合（400 条真实 + 400 条生成）"]
    AUG --> FT["微调下游 WAM（LingBot-VA 官方 checkpoint），评测 OOD 泛化"]
```

图依据：论文 §3 Preliminaries、§4 Method（Model Overview / OAPM / A3D-RoPE）与 Fig. 3、附录 Downstream WAM Training 与 §5 Downstream WAM Generalization。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 017/564 |
| 分组 | Benchmarks & simulators |
| 出处 | 见清单 / 原文 |
| 论文 | <https://arxiv.org/abs/2607.28243> |
| 项目页 | <https://egogenesis.github.io/> |
| 子类 / 象限 | 神经世界模拟器 · 不适用 |

## 核心机制（归纳）

### 策展导读要点

We present \method, an egocentric world-action simulator that synthesizes controllable, high-quality manipulation videos to expand scarce real-world training data. \method{} builds on a pretrained video generation prior and introduces two geometry-aware conditioning mechanisms.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2607.28243) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **Benchmarks & simulators** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「EgoGenesis」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Benchmarks & simulators**，以及 Contribution 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Contribution 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome World-Action Models（RCL）](paper-rcl-wam-robot-learning-control-survey.md)
- 技术地图：[RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md)
- 方法/任务：[generative-world-models.md](../methods/generative-world-models.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/rcl_awesome_wam_2607_28243_egogenesis-egocentric-world-action-model.md`](../../sources/papers/rcl_awesome_wam_2607_28243_egogenesis-egocentric-world-action-model.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://arxiv.org/abs/2607.28243>
- [arXiv PDF](https://arxiv.org/pdf/2607.28243)：已通读方法部分（实验与附录以略读为准）：§1 Introduction、§3 Preliminaries、§4 Method（Model Overview、OAPM、A3D-RoPE）；§5 下游 WAM 泛化与附录下游训练细节为略读

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://arxiv.org/abs/2607.28243)
