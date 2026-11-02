---
type: entity
tags: [paper, curated-index, awesome-real2sim2real, sun254667-r2s2r]
status: complete
updated: 2026-10-06
arxiv: "2607.18135"
venue: "arXiv 2026"
summary: "Utilizes NVIDIA's Isaac Sim and Isaac Lab for training, achieving zero-shot sim-to-real policy transfer to the Unitree Go1 quadruped robot"
related:
  - ../entities/awesome-real2sim2real.md
  - ../overview/sun-awesome-r2s2r-technology-map.md
  - ../methods/reinforcement-learning.md
  - ../methods/crisp-real2sim.md
  - ../tasks/locomotion.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/sun_awesome_r2s2r_2607_18135_isaac-sim-to-real-reinforcement-learning.md
  - ../../sources/papers/sun_awesome_r2s2r_catalog.md
  - ../../sources/repos/awesome-real2sim2real.md
---

# Isaac Sim-to-Real

**Isaac Sim-to-Real: Reinforcement Learning based Locomotion for Quadrupeds** 收录于 [Awesome-Real2Sim2Real](https://github.com/sun254667/Awesome-Real2Sim2Real) **第 011/063** 篇，分组 **22 Zero-Shot Sim2Real**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Utilizes NVIDIA's Isaac Sim and Isaac Lab for training, achieving zero-shot sim-to-real policy transfer to the Unitree Go1 quadruped robot

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 仿真策略迁移到真机 |
| Real2Sim | Real to Simulation | 真机数据重建/校准仿真 |
| R2S2R | Real2Sim2Real | 真机→仿真→真机闭环 |
| DR | Domain Randomization | 域随机化 |

## 为什么重要

- Utilizes NVIDIA's Isaac Sim and Isaac Lab for training, achieving zero-shot sim-to-real policy transfer to the Unitree Go1 quadruped robot
- 想横向对照同一分组的其他工作，可以从 [Awesome-Real2Sim2Real 技术地图](../overview/sun-awesome-r2s2r-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome-Real2Sim2Real](../entities/awesome-real2sim2real.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
  REC["真机记录关节位置 / 速度与力矩数据"] --> ACTNET["Actuator-net<br/>MLP，3 层 × 64，soft-sign"]
  subgraph TRAIN["Isaac Sim / Isaac Lab 训练（4096 并行环境）"]
    DR["域随机化 + 地形课程 + 随机推力"] --> ENV["Go1 仿真环境"]
    OBS["48 维观测<br/>线速度 / 角速度 / 投影重力 / 速度指令<br/>关节位置 / 关节速度 / 上一动作"] --> PI["Actor-Critic MLP<br/>512-256-128，ELU，PPO"]
    PI --> ACT["12 维关节位置动作<br/>缩放 0.2 + 名义姿态偏置"]
    ACT --> ACTNET
    ACTNET --> TAU["关节力矩"]
    TAU --> ENV
    ENV --> OBS
    ENV --> RW["奖励与惩罚项<br/>速度跟踪 / 步态 / 足端离地 / 平滑性 / 关节限位等"]
    RW --> PI
  end
  PI -->|"零样本部署"| ONNX["ONNX 导出<br/>Jetson Nano 推理"]
  KF["卡尔曼滤波估计线速度<br/>C++ 实现"] --> ONNX
  ONNX --> GO1["Unitree Go1 真机"]
```

图依据：论文 §II.B–§II.E、§III.A。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 011/063 |
| 分组 | 22 Zero-Shot Sim2Real |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2607.18135> |

## 核心机制（归纳）

### 策展导读要点

Utilizes NVIDIA's Isaac Sim and Isaac Lab for training, achieving zero-shot sim-to-real policy transfer to the Unitree Go1 quadruped robot

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2607.18135) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-r2s2r-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-r2s2r-technology-map.md) 的 **22 Zero-Shot Sim2Real** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Isaac Sim-to-Real」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **22 Zero-Shot Sim2Real**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Highlights 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome-Real2Sim2Real](../entities/awesome-real2sim2real.md)
- 技术地图：[Awesome-Real2Sim2Real 技术地图](../overview/sun-awesome-r2s2r-technology-map.md)
- 方法/任务：[reinforcement-learning.md](../methods/reinforcement-learning.md)、[locomotion.md](../tasks/locomotion.md)

## 参考来源

- [`sources/papers/sun_awesome_r2s2r_2607_18135_isaac-sim-to-real-reinforcement-learning.md`](../../sources/papers/sun_awesome_r2s2r_2607_18135_isaac-sim-to-real-reinforcement-learning.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_r2s2r_catalog.md`](../../sources/papers/sun_awesome_r2s2r_catalog.md) — 列表总表
- [`sources/repos/awesome-real2sim2real.md`](../../sources/repos/awesome-real2sim2real.md)
- 论文：<https://arxiv.org/abs/2607.18135>
- [arXiv PDF](https://arxiv.org/pdf/2607.18135)：已通读方法部分（实验与附录以略读为准）：§II 强化学习框架（问题定义、Sim-to-Real 技术、动作空间、观测、奖励）、§III.A 网络架构与训练、§III.B–C 实验与结果

## 推荐继续阅读

- [Awesome-Real2Sim2Real 仓库](https://github.com/sun254667/Awesome-Real2Sim2Real)
- [原文](https://arxiv.org/abs/2607.18135)
