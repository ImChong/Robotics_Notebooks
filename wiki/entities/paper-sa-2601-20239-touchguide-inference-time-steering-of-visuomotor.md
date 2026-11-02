---
type: entity
tags: [paper, curated-index, awesome-touch, sun254667-touch]
status: complete
updated: 2026-10-06
arxiv: "2601.20239"
venue: "RSS 2026"
summary: "Cross-policy visuo-tactile fusion paradigm fusing modalities within low-dimensional action space."
related:
  - ../entities/awesome-touch.md
  - ../overview/sun-awesome-touch-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/bimanual-manipulation.md
sources:
  - ../../sources/papers/sun_awesome_touch_2601_20239_touchguide-inference-time-steering-of-vi.md
  - ../../sources/papers/sun_awesome_touch_catalog.md
  - ../../sources/repos/awesome-touch.md
---

# TouchGuide

**TouchGuide: Inference-Time Steering of Visuomotor Policies via Touch Guidance** 收录于 [Awesome Touch](https://github.com/sun254667/awesome-touch) **第 029/065** 篇，分组 **51 Imitation Learning with Tactile Feedback**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Cross-policy visuo-tactile fusion paradigm fusing modalities within low-dimensional action space.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VTLA | Vision-Tactile-Language-Action | 视–触–语言–动作策略 |
| WM | World Model | 视触觉前向预测 |
| WAM | World Action Model | 世界–动作联合建模 |
| Sim2Real | Simulation to Real | 仿真到真机迁移 |

## 为什么重要

- Cross-policy visuo-tactile fusion paradigm fusing modalities within low-dimensional action space.
- 想横向对照同一分组的其他工作，可以从 [Awesome Touch 技术地图](../overview/sun-awesome-touch-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Touch](../entities/awesome-touch.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    V["视觉观测 V_t"] --> BP["预训练 Base Policy π_θ<br/>(diffusion / flow matching，如 π0.5 action expert)"]
    N["初始噪声 A_t^K ~ N(0, I)"] --> BP
    BP --> E["去噪早期：仅凭视觉生成粗动作 A_t"]
    E --> L["去噪后期 (k ≤ K_TouchGuide)：调用 CPM 打分"]
    T["触觉观测 T_t"] --> CPM
    V --> CPM
    L --> CPM["Contact Physical Model (CPM)<br/>DINOv2 编码 + Transformer 融合 → O_t<br/>动作编码器 (1D CNN + MLP) → a_t<br/>可行性分数 s = O_t^T a_t"]
    CPM --> G["对 s 求动作梯度 ∇ s<br/>按引导系数 η 修正 ε / 速度场"]
    G --> U["更新 A_t^{k-1}，迭代至 k=0"]
    U --> X["输出受触觉引导的动作 A_t^0 并执行"]
    D["少量专家示范"] --> C["对比学习训练 CPM<br/>同一时刻 (O_t, a_t) 为正样本，不同时刻为负样本"]
    C -.-> CPM
```

图依据：论文 §IV-B/C（Fig. 3、Algorithm 1）

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 029/065 |
| 分组 | 51 Imitation Learning with Tactile Feedback |
| 出处 | RSS 2026 |
| 论文 | <https://arxiv.org/abs/2601.20239> |

## 核心机制（归纳）

### 策展导读要点

Cross-policy visuo-tactile fusion paradigm fusing modalities within low-dimensional action space.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2601.20239) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-touch-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-touch-technology-map.md) 的 **51 Imitation Learning with Tactile Feedback** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「TouchGuide」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **51 Imitation Learning with Tactile Feedback**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Highlights 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome Touch](../entities/awesome-touch.md)
- 技术地图：[Awesome Touch 技术地图](../overview/sun-awesome-touch-technology-map.md)
- 方法/任务：[vla.md](../methods/vla.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/sun_awesome_touch_2601_20239_touchguide-inference-time-steering-of-vi.md`](../../sources/papers/sun_awesome_touch_2601_20239_touchguide-inference-time-steering-of-vi.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_touch_catalog.md`](../../sources/papers/sun_awesome_touch_catalog.md) — 列表总表
- [`sources/repos/awesome-touch.md`](../../sources/repos/awesome-touch.md)
- 论文：<https://arxiv.org/abs/2601.20239>
- [arXiv PDF](https://arxiv.org/pdf/2601.20239)：已通读方法部分（实验与附录以略读为准）：§I 引言、§IV 方法（Classifier Guidance、Contact Physical Model、Steering）；§III TacUMI 与 §V 实验仅略读

## 推荐继续阅读

- [Awesome Touch 仓库](https://github.com/sun254667/awesome-touch)
- [原文](https://arxiv.org/abs/2601.20239)
