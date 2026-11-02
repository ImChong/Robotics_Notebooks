---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2402.05290"
venue: "arXiv 2024"
summary: "Investigates whether transformer world models give better policy gradients."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2402_05290_awm-do-transformer-world-models-give-bet.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# AWM

**AWM: Do Transformer World Models Give Better Policy Gradients?** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 240/571** 篇，分组 **72 Continuous Control**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Investigates whether transformer world models give better policy gradients.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Investigates whether transformer world models give better policy gradients.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
  ENVD["与环境交互\n用当前策略 π_θ 采集一个 episode"] --> BUF["数据缓冲区 B"]
  BUF --> WML["世界模型学习\n最小化下一状态预测误差"]
  WML --> WM["世界模型\n(Markov 模型 / HWM / AWM)"]
  WM --> UNR["展开世界模型\n计算回报 J(θ; ψ)"]
  UNR --> BP["反向传播得到策略梯度\n更新策略参数 θ"]
  BP --> ENVD
  subgraph CMP["三类世界模型的梯度路径"]
    MK["Markov 模型 f\ns_t+1 = f(s_t, a_t)"] --> CIRC["梯度经过自回归状态预测\n形成迂回梯度路径"]
    HWM["历史世界模型 HWM\n条件于完整状态-动作历史"] --> CIRC
    AWM["动作世界模型 AWM\n仅以初始状态与动作序列 a_1:t 预测状态"] --> DIRECT["梯度从奖励直接回到动作"]
  end
  WM -.-> CMP
```

图依据：论文 Background（BPO，Algorithm 1）、§3 History World Models（Fig. 1、Theorem 1）、§4 Actions World Models（Proposition 1、Theorem 2）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 240/571 |
| 分组 | 72 Continuous Control |
| 出处 | arXiv 2024 |
| 论文 | <https://arxiv.org/abs/2402.05290> |

## 核心机制（归纳）

### 策展导读要点

Investigates whether transformer world models give better policy gradients.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2402.05290) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **72 Continuous Control** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「AWM」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **72 Continuous Control**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2402_05290_awm-do-transformer-world-models-give-bet.md`](../../sources/papers/sun_awesome_wm_2402_05290_awm-do-transformer-world-models-give-bet.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2402.05290>
- [arXiv PDF](https://arxiv.org/pdf/2402.05290)：已通读方法部分（实验与附录以略读为准）：Background、History World Models 与 Actions World Models 各节（实验部分略读）

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2402.05290)
