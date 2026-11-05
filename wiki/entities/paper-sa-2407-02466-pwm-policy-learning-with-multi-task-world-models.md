---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2407.02466"
venue: "arXiv 2024"
code: https://www.imgeorgiev.com/pwm/
summary: "Policy learning with large world models."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2407_02466_pwm-policy-learning-with-multi-task-worl.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# PWM

**PWM: Policy Learning with Multi-Task World Models** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 263/571** 篇，分组 **73 Offline & Generalization**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Policy learning with large world models.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Policy learning with large world models.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    DATA["多任务离线数据集 B：(s, a, r, 任务嵌入 e)"] --> PRE
    subgraph PRE["预训练多任务世界模型（一次）"]
        WMN["TD-MPC2 式模型：编码器 Eϕ(s,e)、动力学 Fϕ(z,a,e)、奖励 Rϕ(z,a,e)，SimNorm 激活，H=16"] --> LWM["自回归损失 Lwm：隐状态一致性 + 奖励交叉熵"]
    end
    LWM --> WM["预训练好的可微世界模型"]
    subgraph POLI["针对单个任务训练策略（约 10k 步，分钟级）"]
        S1["从 B 采样 s1，编码得到 z1 = Eϕ(s1,e)"] --> ROLL["批量想象 rollout：a_h ~ πθ(·|z_h)，r_h = Rϕ，z_h+1 = Fϕ"]
        ROLL --> ACTOR["Actor：一阶梯度 FoG 穿过世界模型反传，H 步回报 + 终端价值 Vψ(z_H)"]
        ROLL --> CRITIC["Critic：TD(λ) 训练，3 个 critic 集成"]
    end
    WM --> ROLL
    ACTOR --> DEPLOY["部署策略解决该任务，可选微调世界模型与策略"]
```

图依据：论文 §3.3 PWM（Algorithm 1、Eq.6 至 Eq.10）；§4.2 提供 MT30 / MT80 设定下的训练步数。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 263/571 |
| 分组 | 73 Offline & Generalization |
| 出处 | arXiv 2024 |
| 论文 | <https://arxiv.org/abs/2407.02466> |
| 代码/项目 | <https://www.imgeorgiev.com/pwm/> |

## 核心机制（归纳）

### 策展导读要点

Policy learning with large world models.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2407.02466) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **73 Offline & Generalization** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「PWM」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_wm_2407_02466_pwm-policy-learning-with-multi-task-worl.md`](../../sources/papers/sun_awesome_wm_2407_02466_pwm-policy-learning-with-multi-task-worl.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2407.02466>
- [arXiv PDF](https://arxiv.org/pdf/2407.02466)：已通读方法部分（实验与附录以略读为准）：§3 策略优化（§3.3 PWM 与 Algorithm 1）、§4.2 多任务训练设定；其余实验与附录略读

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2407.02466)
