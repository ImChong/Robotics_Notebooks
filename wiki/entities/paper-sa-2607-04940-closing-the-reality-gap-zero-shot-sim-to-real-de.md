---
type: entity
tags: [paper, curated-index, awesome-touch, sun254667-touch]
status: complete
updated: 2026-10-06
arxiv: "2607.04940"
venue: "arXiv 2026"
summary: "Holistic sim-to-real recipe combining scalable tactile simulation, current-to-torque calibration, and actuator modelling for dexterous manipulation."
related:
  - ../entities/awesome-touch.md
  - ../overview/sun-awesome-touch-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/bimanual-manipulation.md
sources:
  - ../../sources/papers/sun_awesome_touch_2607_04940_closing-the-reality-gap-zero-shot-sim-to.md
  - ../../sources/papers/sun_awesome_touch_catalog.md
  - ../../sources/repos/awesome-touch.md
---

# Closing the Reality Gap

**Closing the Reality Gap: Zero-Shot Sim-to-Real Deployment for Dexterous Force-Based Grasping and Manipulation** 收录于 [Awesome Touch](https://github.com/sun254667/awesome-touch) **第 046/065** 篇，分组 **Tactile Simulation & Sim2Real**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Holistic sim-to-real recipe combining scalable tactile simulation, current-to-torque calibration, and actuator modelling for dexterous manipulation.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VTLA | Vision-Tactile-Language-Action | 视–触–语言–动作策略 |
| WM | World Model | 视触觉前向预测 |
| WAM | World Action Model | 世界–动作联合建模 |
| Sim2Real | Simulation to Real | 仿真到真机迁移 |

## 为什么重要

- Holistic sim-to-real recipe combining scalable tactile simulation, current-to-torque calibration, and actuator modelling for dexterous manipulation.
- 想横向对照同一分组的其他工作，可以从 [Awesome Touch 技术地图](../overview/sun-awesome-touch-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Touch](../entities/awesome-touch.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart LR
  subgraph SIM["IsaacLab 仿真训练 (XHand 12-DoF)"]
    TS["快速触觉仿真<br/>并行 FK 计算虚拟 tactile unit 到物体距离<br/>每指 120 个，共 600 个"] --> TF["触觉特征 Ti = (总接触力 Fi, 力加权接触中心 μi)<br/>力由 Mooney-Rivlin 应力-应变近似"]
    AM["随机化执行器模型<br/>PD + backlash 死区 + 速度相关力矩饱和 + 系数 η"] --> TQ["关节力矩"]
    TF --> OBS["观测<br/>关节角 / 关节力矩 / 接触力 / 接触中心 / 指尖位置 / 任务指令"]
    TQ --> OBS
    OBS --> AC["非对称 Actor-Critic PPO<br/>Actor 只用真机可测观测，Critic 额外用特权信息"]
    AC --> TASK["任务<br/>力自适应抓取 (倒置 catch-the-object) / 绕单轴手内旋转"]
  end
  subgraph CAL["电流-力矩标定"]
    CUR["真机电机电流 vs 触觉力"] --> MAP["准静态拟合电流→力矩映射<br/>归一化后与仿真力矩对齐"]
  end
  AC --> POL["训练好的策略"]
  MAP --> POL
  POL -->|"零样本部署，不微调"| REAL["真机 XHand<br/>电流估计力矩 + 指尖触觉反馈"]
```

图依据：论文 §III（C 任务设定、D 仿真策略学习、E Sim-to-Real Transfer）/ Fig.2–4、Tab.I–III；摘要

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 046/065 |
| 分组 | Tactile Simulation & Sim2Real |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2607.04940> |

## 核心机制（归纳）

### 策展导读要点

Holistic sim-to-real recipe combining scalable tactile simulation, current-to-torque calibration, and actuator modelling for dexterous manipulation.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2607.04940) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-touch-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-touch-technology-map.md) 的 **Tactile Simulation & Sim2Real** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Closing the Reality Gap」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Tactile Simulation & Sim2Real**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_touch_2607_04940_closing-the-reality-gap-zero-shot-sim-to.md`](../../sources/papers/sun_awesome_touch_2607_04940_closing-the-reality-gap-zero-shot-sim-to.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_touch_catalog.md`](../../sources/papers/sun_awesome_touch_catalog.md) — 列表总表
- [`sources/repos/awesome-touch.md`](../../sources/repos/awesome-touch.md)
- 论文：<https://arxiv.org/abs/2607.04940>
- [arXiv PDF](https://arxiv.org/pdf/2607.04940)：已通读方法部分（实验与附录以略读为准）：摘要、§III Methodology（A–E）

## 推荐继续阅读

- [Awesome Touch 仓库](https://github.com/sun254667/awesome-touch)
- [原文](https://arxiv.org/abs/2607.04940)
