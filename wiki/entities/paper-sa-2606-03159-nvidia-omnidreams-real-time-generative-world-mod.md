---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2606.03159"
venue: "arXiv 2026"
summary: "NVIDIA's real‑time generative world model for closed‑loop autonomous driving simulation, with an action‑conditioned generative world model at its core, fine‑tuned on Cosmos for mid‑ and post‑training."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2606_03159_nvidia-omnidreams-real-time-generative-w.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# NVIDIA OmniDreams

**NVIDIA OmniDreams: Real-Time Generative World Model for Closed-Loop Autonomous Vehicle Simulation** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 021/571** 篇，分组 **111 General Simulation Platforms**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

NVIDIA's real‑time generative world model for closed‑loop autonomous driving simulation, with an action‑conditioned generative world model at its core, fine‑tuned on Cosmos for mid‑ and post‑training.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- NVIDIA's real‑time generative world model for closed‑loop autonomous driving simulation, with an action‑conditioned generative world model at its core, fine‑tuned on Cosmos for mid‑ and post‑training.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph LOOP["闭环仿真（推理）"]
        POL["驾驶策略模型或用户<br/>输出动作：转向 / 油门 / 刹车"] --> SIM["AlpaSim 仿真运行时<br/>更新抽象状态：地图 + 动态体 bounding box"]
        SIM --> OD["OmniDreams<br/>因果扩散生成下一步多帧相机画面"]
        TXT["文本提示：光照 / 天气 / 时段"] --> OD
        CACHE["流式 KV cache<br/>历史帧记忆"] --> OD
        FIRST["首步额外条件：单张 RGB 首帧"] -.-> OD
        OD --> FRAME["合成相机画面<br/>SV：8 帧/步；MV：4 视角共 16 帧/步"]
        FRAME -->|"返回给策略"| POL
        FRAME -->|"更新"| CACHE
    end
    subgraph TRAIN["多阶段训练"]
        B0["Cosmos-Predict 2.5 双向视频模型"] --> B1["RDS 数据 mid-training<br/>提升 AV 能力"]
        B1 --> B2["多视角适配 + cross-view attention<br/>（仅 MV）"]
        B2 --> B3["World-scenario 地图控制分支<br/>得到双向教师模型"]
        B3 --> B4["Diffusion Forcing 因果化<br/>用于自回归生成"]
        B4 --> B5["Self Forcing + DMD 蒸馏<br/>少步生成，减小累积误差"]
    end
    B5 -.->|"部署为"| OD
```

图依据：论文 §3 / §3.1、§4（训练阶段概述）/ Fig.1、Fig.3

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 021/571 |
| 分组 | 111 General Simulation Platforms |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2606.03159> |

## 核心机制（归纳）

### 策展导读要点

NVIDIA's real‑time generative world model for closed‑loop autonomous driving simulation, with an action‑conditioned generative world model at its core, fine‑tuned on Cosmos for mid‑ and post‑training.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2606.03159) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **111 General Simulation Platforms** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「NVIDIA OmniDreams」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **111 General Simulation Platforms**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2606_03159_nvidia-omnidreams-real-time-generative-w.md`](../../sources/papers/sun_awesome_wm_2606_03159_nvidia-omnidreams-real-time-generative-w.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2606.03159>
- [arXiv PDF](https://arxiv.org/pdf/2606.03159)：已通读方法部分（实验与附录以略读为准）：§1 引言、§3 模型架构（3.1–3.2）、§4 训练（4.1–4.3）；§2 数据、§5 以后略读

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2606.03159)
