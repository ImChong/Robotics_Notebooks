---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2503.10480"
venue: "arXiv 2025"
summary: "Dual preference optimization for embodied task planning."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2503_10480_d-2po-world-modeling-makes-a-better-plan.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# D^2PO

**D^2PO: World Modeling Makes a Better Planner – Dual Preference Optimization for Embodied Task Planning** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 363/571** 篇，分组 **814 Locomotion & Loco-Manipulation**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Dual preference optimization for embodied task planning.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Dual preference optimization for embodied task planning.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TB
  subgraph ts["Step-wise 树搜索数据探索（Fig. 2a）"]
    A["当前状态节点 s_t"] --> B["基础策略采样 K 个候选动作"]
    B --> C["混合打分：GPT-4o 过程奖励 r_proc 与环境可行性 r_env 等权组合"]
    C --> D["得分 ≥ 阈值 τ 的动作按广度优先展开<br/>执行后得到下一层状态"]
    D --> A
    D --> E["到达目标状态后回溯成功轨迹"]
    E --> F["构造偏好对：动作选择（与兄弟节点对比）<br/>与状态预测（与替代动作的后继状态对比）"]
  end
  subgraph dpo["D2PO 双偏好优化（Fig. 2b）"]
    G["动作选择损失 L_action"] 
    H["状态预测损失 L_state（世界建模）"]
    G --> I["L_total = L_action + λ·L_state"]
    H --> I
  end
  F --> G
  F --> H
  I --> J["更新 LVLM 规划器"]
```

图依据：论文 §3.2 Step-wise Tree Search、§3.3 D2PO Framework（Fig. 2a/2b）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 363/571 |
| 分组 | 814 Locomotion & Loco-Manipulation |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2503.10480> |

## 核心机制（归纳）

### 策展导读要点

Dual preference optimization for embodied task planning.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2503.10480) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **814 Locomotion & Loco-Manipulation** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「D^2PO」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **814 Locomotion & Loco-Manipulation**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2503_10480_d-2po-world-modeling-makes-a-better-plan.md`](../../sources/papers/sun_awesome_wm_2503_10480_d-2po-world-modeling-makes-a-better-plan.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2503.10480>
- [arXiv PDF](https://arxiv.org/pdf/2503.10480)：已通读方法部分（实验与附录以略读为准）：§1 引言、§3 Method、§4.1.4 Implementation Details，其余实验与分析略读

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2503.10480)
