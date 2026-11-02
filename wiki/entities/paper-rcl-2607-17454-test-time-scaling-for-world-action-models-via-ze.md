---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "2607.17454"
summary: "We propose \\methodgated, a training-free selective test-time scaling framework for WAMs."
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_2607_17454_test-time-scaling-for-world-action-model.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# Test-Time Scaling for World Action Models via Zero-Shot Geometric Evaluation

**Test-Time Scaling for World Action Models via Zero-Shot Geometric Evaluation** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 492/564** 篇，分组 **WAMs**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

We propose \methodgated, a training-free selective test-time scaling framework for WAMs.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- We propose \methodgated, a training-free selective test-time scaling framework for WAMs.
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    IN["多视角观测 + 本体状态 + 语言指令"] --> R1["WAM 生成初始 rollout τ1：预测主视角/腕部视角未来帧 + 动作块 a_1:H"]
    R1 --> FLOW["对当前主视角帧与预测未来帧计算稠密光流（Farneback）"]
    R1 --> PROJ["末端执行器位姿经前向运动学 / 位移累积，投影到主相机得到动作运动 Δu"]
    FLOW --> COS["在末端轨迹胶囊区域内平均光流 f̄，与 Δu 计算余弦一致度 c"]
    PROJ --> COS
    COS --> GATE{"任一运动臂 c < τ_gate ？（空闲臂忽略）"}
    GATE -->|"否：一致"| EXEC1["直接执行初始动作块 a_1"]
    GATE -->|"是：不一致"| MORE["额外采样 N_max-1 个 rollout，与初始 rollout 组成候选集"]
    MORE --> GEO["跨视角几何评估器：冻结 VGGT-Ω 预测主视角深度与腕部 3D 点图，将腕部点图投影到主视角，计算深度重投影不一致度 e_depth"]
    GEO --> PICK["选 e_depth 最低的候选，执行其动作块"]
```

图依据：论文 §3 Method（§3.1 Action–Future Gate、§3.2 Cross-View Geometric Evaluator、§3.3 Inference Procedure）与 Fig. 1；τ_gate 取值见 §4.1。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 492/564 |
| 分组 | WAMs |
| 出处 | 见清单 / 原文 |
| 论文 | <https://arxiv.org/abs/2607.17454> |
| 子类 / 象限 | 三维多视角建模 · 高效推理与实时控制 · 策略后训练与WM-RL · 四象限外 |

## 核心机制（归纳）

### 策展导读要点

We propose \methodgated, a training-free selective test-time scaling framework for WAMs.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2607.17454) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **WAMs** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Test-Time Scaling for World Action Models via Zero-Shot Geometric Evaluation」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/rcl_awesome_wam_2607_17454_test-time-scaling-for-world-action-model.md`](../../sources/papers/rcl_awesome_wam_2607_17454_test-time-scaling-for-world-action-model.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://arxiv.org/abs/2607.17454>
- [arXiv PDF](https://arxiv.org/pdf/2607.17454)：已通读方法部分（实验与附录以略读为准）：§3 Method（§3.1–§3.3）、§4.1 实验设置

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://arxiv.org/abs/2607.17454)
