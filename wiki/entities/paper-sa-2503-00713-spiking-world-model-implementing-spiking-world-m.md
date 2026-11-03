---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2503.00713"
venue: "arXiv 2025"
summary: "Implements spiking world models with multi-compartment neurons for MBRL."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2503_00713_spiking-world-model-implementing-spiking.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Spiking World Model

**Spiking World Model: Implementing Spiking World Model with Multi-Compartment Neurons for MBRL** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 269/571** 篇，分组 **73 Offline & Generalization**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Implements spiking world models with multi-compartment neurons for MBRL.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Implements spiking world models with multi-compartment neurons for MBRL.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    O["观测 O[t]"] --> SC["LIF 脉冲卷积网络 SCNN，得到 S_ob[t]"]
    ZA["上一步 Z[t-1] 与动作 A[t-1]"] --> FC["全连接 LIF-SNN，得到序列模型输入 S_in[t]"]
    FC --> MCN["MCN 序列模型：基底树突 V_b、顶端树突 V_a、胞体 U"]
    MCN --> NOTE["V_a 经 Sigmoid 门控 z，调制 V_b 与胞体输入对 U 的贡献"]
    MCN --> SH["脉冲状态 S_h[t]（含递归反馈）"]
    SC --> POST["后验 Z[t] ~ q(S_ob, S_h)"]
    SH --> POST
    SH --> PRIOR["先验 Ẑ[t] ~ p(S_h)"]
    SH --> HEADS["预测头：奖励 R̂、continue Ĉ、脉冲反卷积解码器 Ô"]
    POST --> HEADS
    SH --> AC["SNN Actor-Critic：输入 S_h 与 Z"]
    POST --> AC
    HEADS --> LOSS["损失：预测损失 + KL(后验, 先验)；Critic 用 λ-return"]
    AC --> LOSS
    LOSS --> BP["STBP 训练，脉冲发放处用代理梯度"]
```

图依据：论文 §III-C（Sequential Nonlinear Dendritic Integration for MCN）、§III-D（Spiking World Model，Fig. 1(b)，式 4–26）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 269/571 |
| 分组 | 73 Offline & Generalization |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2503.00713> |

## 核心机制（归纳）

### 策展导读要点

Implements spiking world models with multi-compartment neurons for MBRL.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2503.00713) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **73 Offline & Generalization** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Spiking World Model」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **73 Offline & Generalization**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2503_00713_spiking-world-model-implementing-spiking.md`](../../sources/papers/sun_awesome_wm_2503_00713_spiking-world-model-implementing-spiking.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2503.00713>
- [arXiv PDF](https://arxiv.org/pdf/2503.00713)：已通读方法部分（实验与附录以略读为准）：§III-B 至 §III-D（含 Fig. 1 与式 1–26）；§IV 实验以略读为准

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2503.00713)
