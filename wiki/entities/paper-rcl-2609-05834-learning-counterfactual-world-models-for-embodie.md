---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "2609.05834"
venue: "2026"
summary: "Counterfactual Latent World Models (CLWM) train action-conditioned imagined futures to preserve differences in intervention outcomes even when observations look alike. A recurrent world model supplies latent rollouts to"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_2609_05834_learning-counterfactual-world-models-for.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# Learning Counterfactual World Models for Embodied Reasoning under Partial Obs...

**Learning Counterfactual World Models for Embodied Reasoning under Partial Observability** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 408/564** 篇，分组 **WAMs**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Counterfactual Latent World Models (CLWM) train action-conditioned imagined futures to preserve differences in intervention outcomes even when observations look alike. A recurrent world model supplies latent rollouts to model-predictive control; privileged outcome labels supervise an additional contrastive objective...

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- Counterfactual Latent World Models (CLWM) train action-conditioned imagined futures to preserve differences in intervention outcomes even when observations look alike. A recurrent world model supplies latent rollouts to model-predictive control; privileged outcome labels supervise an additional contrastive objective...
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph TRAIN["训练"]
        A["多模态观测 o_t = (图像 x, 本体 q, 可选触觉/力 u)"] --> B["信念状态编码器：确定性循环信念 b_t<br/>+ 随机潜变量 z_t（后验）"]
        B --> C["从锚点历史 h_t 采样 M 条候选干预序列 α_i<br/>来源：随机 shooting / 策略提议 / MPC 候选 / 回放缓冲区"]
        C --> D["动作条件先验在潜空间开环 rollout<br/>信念用采样的先验潜变量推进，无真实未来观测"]
        D --> E["投影头 g_θ → 反事实嵌入 c_i<br/>（仅训练期使用，部署不用）"]
        E --> F["反事实对比损失 L_cf<br/>按干预结果划分正负例，难负例：同信念动作 / 感知混淆 / 价值分歧"]
        B --> G["常规世界模型损失：预测 / KL / 奖励 / 价值"]
        F --> H["总目标 L_CLWM = L_pred + β_KL·L_KL + β_r·L_r + β_v·L_v + λ_cf·L_cf + λ_inv·L_inv"]
        G --> H
    end
    subgraph DEPLOY["部署：潜空间 MPC 规划（无需对比计算）"]
        P1["N 条候选动作序列，视野 H<br/>用预测奖励 + 终端价值打分 J(α)"] --> P2["执行最优序列的第一个动作并重新规划"]
        P2 -.-> P1
    end
    H --> DEPLOY
```

图依据：论文 §3–§4.5（式 4–13；论文无架构图，流程由 §4.1–4.5 文字整理）

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 408/564 |
| 分组 | WAMs |
| 出处 | 2026 |
| 论文 | <https://arxiv.org/abs/2609.05834> |
| 子类 / 象限 | 潜空间预测与JEPA · 策略后训练与WM-RL · 四象限外 |

## 核心机制（归纳）

### 策展导读要点

Counterfactual Latent World Models (CLWM) train action-conditioned imagined futures to preserve differences in intervention outcomes even when observations look alike. A recurrent world model supplies latent rollouts to model-predictive control; privileged outcome labels supervise an additional contrastive objective...

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2609.05834) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **WAMs** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Learning Counterfactual World Models for Embodied Reasoning under Partial Obs...」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **WAMs**，以及 Contribution 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/rcl_awesome_wam_2609_05834_learning-counterfactual-world-models-for.md`](../../sources/papers/rcl_awesome_wam_2609_05834_learning-counterfactual-world-models-for.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://arxiv.org/abs/2609.05834>
- [arXiv PDF](https://arxiv.org/pdf/2609.05834)：已通读方法部分（实验与附录以略读为准）：摘要、§1、§3 问题设定、§4.1–4.5；§5–9 略读

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://arxiv.org/abs/2609.05834)
