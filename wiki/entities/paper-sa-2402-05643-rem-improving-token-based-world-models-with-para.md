---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2402.05643"
venue: "arXiv 2024"
code: https://github.com/leor-c/REM
summary: "Improves token-based world models with parallel observation prediction."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2402_05643_rem-improving-token-based-world-models-w.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# REM

**REM: Improving Token-Based World Models with Parallel Observation Prediction** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 248/571** 篇，分组 **72 Continuous Control**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Improves token-based world models with parallel observation prediction.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Improves token-based world models with parallel observation prediction.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
  ENV["真实环境交互"] -->|"1 经验收集"| BUF["回放缓冲区\n观测 / 动作 / 奖励 / 终止"]
  BUF -->|"2 训练"| V["V 视觉感知：VQ-VAE 分词器\n观测 o_t 变为 K 个离散 token z_t"]
  BUF -->|"3 训练"| M["M 世界模型：POP 增强的 RetNet\n预测下一观测 token / 奖励 / 终止"]
  V -.->|"码本 E 用于 token 嵌入"| M
  subgraph POP["Parallel Observation Prediction"]
    U["K 个专用预测 token u"] --> ONE["一次前向并行生成下一观测的全部 K 个 token"]
    CH["分块 (chunkwise) 前向 + 循环状态 S\n并行计算每个观测-动作块之后的状态"] --> ONE
  end
  M --- POP
  BUF -->|"采样短轨迹作为起点"| IMG["想象 H 步\nC 与世界模型交互"]
  M --> IMG
  IMG -->|"4 训练"| C["C 控制器：Actor-Critic\n目标沿用 DreamerV2"]
  C -->|"策略 π"| ENV
```

图依据：论文 §2.1 Overview（V-M-C 结构、Fig. 2 训练循环）、§2.3 World Model Imagination、§2.4 World Model Training（POP，Fig. 4 / Fig. 6）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 248/571 |
| 分组 | 72 Continuous Control |
| 出处 | arXiv 2024 |
| 论文 | <https://arxiv.org/abs/2402.05643> |
| 代码/项目 | <https://github.com/leor-c/REM> |

## 核心机制（归纳）

### 策展导读要点

Improves token-based world models with parallel observation prediction.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2402.05643) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **72 Continuous Control** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「REM」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_wm_2402_05643_rem-improving-token-based-world-models-w.md`](../../sources/papers/sun_awesome_wm_2402_05643_rem-improving-token-based-world-models-w.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2402.05643>
- [arXiv PDF](https://arxiv.org/pdf/2402.05643)：已通读方法部分（实验与附录以略读为准）：§2 Method（2.1–2.4，含 Fig. 2 训练循环与 POP 算法）

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2402.05643)
