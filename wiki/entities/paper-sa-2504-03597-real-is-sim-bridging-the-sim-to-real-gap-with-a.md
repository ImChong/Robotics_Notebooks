---
type: entity
tags: [paper, curated-index, awesome-real2sim2real, sun254667-r2s2r]
status: complete
updated: 2026-10-06
arxiv: "2504.03597"
venue: "arXiv 2025"
summary: "A novel behavior cloning framework that incorporates a dynamic digital twin (based on Embodied Gaussians) throughout the entire policy development pipeline: data collection, training, and deployment"
related:
  - ../entities/awesome-real2sim2real.md
  - ../overview/sun-awesome-r2s2r-technology-map.md
  - ../methods/reinforcement-learning.md
  - ../methods/crisp-real2sim.md
  - ../tasks/locomotion.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/sun_awesome_r2s2r_2504_03597_real-is-sim-bridging-the-sim-to-real-gap.md
  - ../../sources/papers/sun_awesome_r2s2r_catalog.md
  - ../../sources/repos/awesome-real2sim2real.md
---

# Real-is-Sim

**Real-is-Sim: Bridging the Sim-to-Real Gap with a Dynamic Digital Twin** 收录于 [Awesome-Real2Sim2Real](https://github.com/sun254667/Awesome-Real2Sim2Real) **第 057/063** 篇，分组 **Benchmarks Simulators & Datasets**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

A novel behavior cloning framework that incorporates a dynamic digital twin (based on Embodied Gaussians) throughout the entire policy development pipeline: data collection, training, and deployment

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 仿真策略迁移到真机 |
| Real2Sim | Real to Simulation | 真机数据重建/校准仿真 |
| R2S2R | Real2Sim2Real | 真机→仿真→真机闭环 |
| DR | Domain Randomization | 域随机化 |

## 为什么重要

- A novel behavior cloning framework that incorporates a dynamic digital twin (based on Embodied Gaussians) throughout the entire policy development pipeline: data collection, training, and deployment
- 想横向对照同一分组的其他工作，可以从 [Awesome-Real2Sim2Real 技术地图](../overview/sun-awesome-r2s2r-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome-Real2Sim2Real](../entities/awesome-real2sim2real.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    CAM["真实世界相机 RGB 图像 o_rgb"] --> COR["修正机制 C(o_rgb, s_t)<br/>Embodied Gaussians：虚拟力/力矩 u_t"]
    COR --> SIM["可修正物理仿真器<br/>s_t+1, r_t+1 = F(s_t, u_t, r_t)"]
    SIM --> REP["表征 r_t = g(s_t)<br/>物体位姿 / 虚拟相机渲染图（ResNet 编码）"]
    REP --> POL["策略 π(r_t, q_t)<br/>输出 H=1s、m=32 个路点的轨迹，含 q' 与进度 p"]
    POL --> SR["仿真机器人按轨迹更新期望关节 q'"]
    SR --> SIM
    SIM -->|"在线模式：真实机器人跟随仿真关节位置 q"| ROB["真实机器人（follower）"]
    ROB --> CAM
    SIM -.->|"离线模式：关闭真实耦合与 u_t，可并行"| EVAL["虚拟环境策略评测（5–8× 加速）"]
    DEMO["演示采集：仿真器在环，60Hz 记录仿真状态"] --> TRAIN["策略训练：行为克隆，进度值 p 由时间标注"]
    TRAIN --> POL
```

图依据：论文 §3.1 Framework / Fig.3–5、§3.2 Pipeline（Scene Setup / Demonstration Collection / Policy Training / Deployment / Evaluation）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 057/063 |
| 分组 | Benchmarks Simulators & Datasets |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2504.03597> |

## 核心机制（归纳）

### 策展导读要点

A novel behavior cloning framework that incorporates a dynamic digital twin (based on Embodied Gaussians) throughout the entire policy development pipeline: data collection, training, and deployment

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2504.03597) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-r2s2r-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-r2s2r-technology-map.md) 的 **Benchmarks Simulators & Datasets** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Real-is-Sim」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Benchmarks Simulators & Datasets**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_r2s2r_2504_03597_real-is-sim-bridging-the-sim-to-real-gap.md`](../../sources/papers/sun_awesome_r2s2r_2504_03597_real-is-sim-bridging-the-sim-to-real-gap.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_r2s2r_catalog.md`](../../sources/papers/sun_awesome_r2s2r_catalog.md) — 列表总表
- [`sources/repos/awesome-real2sim2real.md`](../../sources/repos/awesome-real2sim2real.md)
- 论文：<https://arxiv.org/abs/2504.03597>
- [arXiv PDF](https://arxiv.org/pdf/2504.03597)：已通读方法部分（实验与附录以略读为准）：§2 Preliminaries（略读）、§3.1–3.2

## 推荐继续阅读

- [Awesome-Real2Sim2Real 仓库](https://github.com/sun254667/Awesome-Real2Sim2Real)
- [原文](https://arxiv.org/abs/2504.03597)
