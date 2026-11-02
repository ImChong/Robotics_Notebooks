---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2509.19080"
venue: "arXiv 2025"
summary: "Diffusion world models for policy refinement with RL."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2509_19080_world4rl-diffusion-world-models-for-poli.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# World4RL

**World4RL: Diffusion World Models for Policy Refinement with RL for Robotic Manipulation** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 343/571** 篇，分组 **812 Manipulation**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Diffusion world models for policy refinement with RL.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Diffusion world models for policy refinement with RL.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
  subgraph S1["阶段一：预训练"]
    D1["专家示范 D_exp"] --> P1["模仿学习预训练策略 π_ξ（BC，最大化专家动作对数似然）"]
    D2["D_exp + 预训练策略 rollout D_rollout"] --> R1["训练奖励分类器 C_ψ（ResNet18，二元交叉熵）"]
    D3["D_exp + D_rollout + 随机 rollout D_rand"] --> W1["训练扩散转移模型 D_θ（EDM 形式，U-Net 2D）"]
  end
  P1 --> INIT["以 BC 策略初始化 RL 策略"]
  W1 --> FZ["冻结世界模型，作为模拟器"]
  R1 --> RC["奖励分类器给出稀疏二元奖励"]
  subgraph S2["阶段二：想象环境中的策略优化"]
    OBS["历史观测 x_{t-T:t}"] --> ACT["策略采样动作 a_t（σ 截断以减少 OOD 动作）"]
    ACT --> TH["two-hot 编码 z_t"]
    TH --> PRED["扩散转移模型预测下一观测 x_{t+1}"]
    OBS --> PRED
    PRED --> RW["奖励分类器评估 x_{t+1}，得 r_t"]
    RW --> PPO["PPO 更新策略与价值网络"]
    PPO --> ACT
  end
  INIT --> ACT
  FZ --> PRED
  RC --> RW
```

图依据：论文 §III-A / §III-B / §III-C、Fig. 1 与 Algorithm 1。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 343/571 |
| 分组 | 812 Manipulation |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2509.19080> |
| 项目页 | <https://world4rl.github.io> |

## 核心机制（归纳）

### 策展导读要点

Diffusion world models for policy refinement with RL.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2509.19080) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **812 Manipulation** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「World4RL」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **812 Manipulation**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2509_19080_world4rl-diffusion-world-models-for-poli.md`](../../sources/papers/sun_awesome_wm_2509_19080_world4rl-diffusion-world-models-for-poli.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2509.19080>
- [arXiv PDF](https://arxiv.org/pdf/2509.19080)：已通读方法部分（§III）；实验与附录以略读为准：§I 引言、§III 方法、§IV 实验设定。

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2509.19080)
