---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2606.16274"
venue: "arXiv 2026"
summary: "Long-horizon planning with graph world models."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2606_16274_graphworld-long-horizon-planning-with-wo.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# GraphWorld

**GraphWorld: Long-Horizon Planning with World Models for End-to-End Autonomous Driving** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 442/571** 篇，分组 **82 Autonomous Driving**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Long-horizon planning with graph world models.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Long-horizon planning with graph world models.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    OBS["多视角观测 → 感知得到实例级特征<br/>ego 与周围 agent (f_i, f_ego)"]
    SEL["Ego-centric 邻居选择<br/>距离 < τ，PadOrTrim 成 K 个邻居"]
    STAR["星形交互图<br/>邻居 → ego 的有向边"]
    CA1["Cross-Attention<br/>ego 规划 query Q_ego 关注邻居特征 → Q~_ego"]
    GRU["GRU 世界编码器<br/>历史状态 H_t + 池化地图嵌入 M → s_t"]
    CA2["Cross-Attention<br/>s_t 对 Q~_ego ∪ F_nbr → c_int"]
    WCUR["残差注入得到<br/>当前潜在世界状态 W_cur"]
    WTGT["目标世界状态 W_tgt<br/>由运动 / 规划 / 地图线索投影，仅作监督"]
    FM["Flow-Matching 速度场 v_θ<br/>W(t) 在 W_cur 与 W_tgt 间插值；推理 Euler 更新(2 步)"]
    WT["精炼世界状态 W_t"]
    MOT["Importance Reweighting<br/>以 W_t 调制 agent 运动 query，sigmoid 权重 α"]
    PLAN["World-to-Ego Feedback<br/>Q'_plan = Q_plan + W_ego"]
    HEAD["任务头联合解码<br/>agent 多模态轨迹 + ego 多模态轨迹与置信度"]
    OBS --> SEL --> STAR --> CA1
    CA1 --> CA2
    GRU --> CA2
    CA2 --> WCUR --> FM
    WTGT -.训练监督.-> FM
    FM --> WT
    WT --> MOT --> HEAD
    WT --> PLAN --> HEAD
    CA1 -.-> WTGT
    HEAD --> LOSS["训练：两阶段<br/>I 单步端到端；II 加 W_t 对 stop-grad(W_t+1) 的 ℓ2 一致性损失"]
```

图依据：论文 §4 / Fig.3（§4.1 ECIG、§4.2 WSCP、§4.3 两阶段监督）

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 442/571 |
| 分组 | 82 Autonomous Driving |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2606.16274> |

## 核心机制（归纳）

### 策展导读要点

Long-horizon planning with graph world models.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2606.16274) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **82 Autonomous Driving** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「GraphWorld」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **82 Autonomous Driving**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2606_16274_graphworld-long-horizon-planning-with-wo.md`](../../sources/papers/sun_awesome_wm_2606_16274_graphworld-long-horizon-planning-with-wo.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2606.16274>
- [arXiv PDF](https://arxiv.org/pdf/2606.16274)：已通读方法部分（实验与附录以略读为准）：§4 方法（ECIG、WSCP、两阶段时序世界状态监督）；§5 实验略读

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2606.16274)
