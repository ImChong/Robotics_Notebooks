---
type: entity
tags: [paper, curated-index, awesome-real2sim2real, sun254667-r2s2r]
status: complete
updated: 2026-10-06
arxiv: "2602.11575"
venue: "arXiv 2026"
summary: "Real-to-sim dynamic 3D Gaussian splatting simulation for environment-specific visual navigation with moving obstacles"
related:
  - ../entities/awesome-real2sim2real.md
  - ../overview/sun-awesome-r2s2r-technology-map.md
  - ../methods/reinforcement-learning.md
  - ../methods/crisp-real2sim.md
  - ../tasks/locomotion.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/sun_awesome_r2s2r_2602_11575_ready-go-real-to-sim-dynamic-3d-gaussian.md
  - ../../sources/papers/sun_awesome_r2s2r_catalog.md
  - ../../sources/repos/awesome-real2sim2real.md
---

# ReaDy-Go

**ReaDy-Go: Real-to-Sim Dynamic 3D Gaussian Splatting Simulation for Environment-Specific Visual Navigation With Moving Obstacles** 收录于 [Awesome-Real2Sim2Real](https://github.com/sun254667/Awesome-Real2Sim2Real) **第 038/063** 篇，分组 **41 Gaussian Splatting-based Frameworks**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Real-to-sim dynamic 3D Gaussian splatting simulation for environment-specific visual navigation with moving obstacles

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 仿真策略迁移到真机 |
| Real2Sim | Real to Simulation | 真机数据重建/校准仿真 |
| R2S2R | Real2Sim2Real | 真机→仿真→真机闭环 |
| DR | Domain Randomization | 域随机化 |

## 为什么重要

- Real-to-sim dynamic 3D Gaussian splatting simulation for environment-specific visual navigation with moving obstacles
- 想横向对照同一分组的其他工作，可以从 [Awesome-Real2Sim2Real 技术地图](../overview/sun-awesome-r2s2r-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome-Real2Sim2Real](../entities/awesome-real2sim2real.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    V["目标部署环境的单目静态视频<br/>COLMAP + ArUco 估计公制尺度位姿"] --> S["静态 GS 场景重建 PGSR"]
    S --> VX["体素化 + 不透明度滤波<br/>得到机器人可通行图与行人可行走图"]
    H["HUGS 人体 GS 化身（NeuMan 数据）"] --> HA["人体动画模块<br/>PriorMDM 生成 SMPL 运动，SMPLify 拟合"]
    VX --> HP["人类规划器：A* 种子路径 + 样条优化"]
    HP --> HA
    VX --> RP["机器人专家规划器：Hybrid A* + 运动基元<br/>动态障碍物更新地图并反应式重规划"]
    HA --> SIM["动态 GS 仿真器渲染 RGB"]
    S --> SIM
    RP --> DATA["数据生成：RGB 观测、专家动作 (v, w)、相对目标位置"]
    SIM --> DATA
    DATA --> POL["模仿学习训练端到端视觉导航策略<br/>连续 3 帧 RGB 编码 + 上一动作 + 相对目标，MSE 监督"]
```

图依据：论文 §III A–C / Fig.2、Fig.3

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 038/063 |
| 分组 | 41 Gaussian Splatting-based Frameworks |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2602.11575> |

## 核心机制（归纳）

### 策展导读要点

Real-to-sim dynamic 3D Gaussian splatting simulation for environment-specific visual navigation with moving obstacles

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2602.11575) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-r2s2r-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-r2s2r-technology-map.md) 的 **41 Gaussian Splatting-based Frameworks** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「ReaDy-Go」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **41 Gaussian Splatting-based Frameworks**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Highlights 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- [StreamSplat：未标定视频流的在线动态 3D 重建](paper-streamsplat-iclr-2026.md) — 通用动态场景重建方法，可作为动态 3DGS 方法参照

- 列表实体：[Awesome-Real2Sim2Real](../entities/awesome-real2sim2real.md)
- 技术地图：[Awesome-Real2Sim2Real 技术地图](../overview/sun-awesome-r2s2r-technology-map.md)
- 方法/任务：[reinforcement-learning.md](../methods/reinforcement-learning.md)、[locomotion.md](../tasks/locomotion.md)

## 参考来源

- [`sources/papers/sun_awesome_r2s2r_2602_11575_ready-go-real-to-sim-dynamic-3d-gaussian.md`](../../sources/papers/sun_awesome_r2s2r_2602_11575_ready-go-real-to-sim-dynamic-3d-gaussian.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_r2s2r_catalog.md`](../../sources/papers/sun_awesome_r2s2r_catalog.md) — 列表总表
- [`sources/repos/awesome-real2sim2real.md`](../../sources/repos/awesome-real2sim2real.md)
- 论文：<https://arxiv.org/abs/2602.11575>
- [arXiv PDF](https://arxiv.org/pdf/2602.11575)：已通读方法部分（实验与附录以略读为准）：Abstract、§I–II、§III A–C（动态 GS 仿真器、导航数据生成、策略训练）、§IV A 实验设定

## 推荐继续阅读

- [Awesome-Real2Sim2Real 仓库](https://github.com/sun254667/Awesome-Real2Sim2Real)
- [原文](https://arxiv.org/abs/2602.11575)
