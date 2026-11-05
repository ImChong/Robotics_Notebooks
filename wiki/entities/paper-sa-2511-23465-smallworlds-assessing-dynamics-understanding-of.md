---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2511.23465"
venue: "arXiv 2025"
summary: "Evaluates world models’ understanding of fundamental physics principles (e.g., gravity, elastic collisions) under isolated and precisely controlled dynamics conditions."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2511_23465_smallworlds-assessing-dynamics-understan.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# SmallWorlds

**SmallWorlds: Assessing Dynamics Understanding of World Models in Isolated Environments** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 047/571** 篇，分组 **115 Benchmarks for World Model Simulation**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Evaluates world models’ understanding of fundamental physics principles (e.g., gravity, elastic collisions) under isolated and precisely controlled dynamics conditions.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Evaluates world models’ understanding of fundamental physics principles (e.g., gravity, elastic collisions) under isolated and precisely controlled dynamics conditions.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["设计要点 §3.1<br/>孤立可控动力学 / 无人工奖励 / 参数化数据分布<br/>完全可观测状态 / 状态空间与像素空间可互转"] --> B["任务套件"]
    B --> B1["物理理解：10 个 MuJoCo 任务<br/>线速度主导 / 旋转主导 / 能量守恒"]
    B --> B2["几何理解：keypoint reprojection<br/>预测关键点像素坐标"]
    B --> B3["联合其他基准：Go / Panda Hand / DMControl / Atari / Memory-Maze"]
    B1 --> C["随机采集 episode<br/>不依赖特定 agent 的 replay buffer"]
    B2 --> C
    B3 --> C
    C --> D["孤立训练 4 类 world model<br/>RSSM / Transformer / Diffusion Forcing / Neural ODE (MoSim)"]
    D --> E["评测协议：10 步真值预热后<br/>想象 90 步 rollout，状态空间 MSE"]
    E --> F["多领域能力对比 §4.2"]
    E --> G["长时域想象：误差随预测步长增长 §4.3"]
```

图依据：基准构建与评测协议流程图，论文 §3 The SmallWorld Benchmark（3.1–3.4）、§4.1 Baseline Architectures、§4.2 Multi-Domain Capability Evaluation（Table 1 协议）、§4.3 Long-horizon Imagination

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 047/571 |
| 分组 | 115 Benchmarks for World Model Simulation |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2511.23465> |

## 核心机制（归纳）

### 策展导读要点

Evaluates world models’ understanding of fundamental physics principles (e.g., gravity, elastic collisions) under isolated and precisely controlled dynamics conditions.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2511.23465) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **115 Benchmarks for World Model Simulation** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「SmallWorlds」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **115 Benchmarks for World Model Simulation**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2511_23465_smallworlds-assessing-dynamics-understan.md`](../../sources/papers/sun_awesome_wm_2511_23465_smallworlds-assessing-dynamics-understan.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2511.23465>
- [arXiv PDF](https://arxiv.org/pdf/2511.23465)：已通读正文：§1 Introduction、§2 Related Work、§3 The SmallWorld Benchmark（3.1–3.4）、§4.1 Baseline Architectures、§4.2 Multi-Domain Capability Evaluation、§4.3 Long-horizon Imagination 开头（§4.3 后半及之后章节、附录未读）

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2511.23465)
