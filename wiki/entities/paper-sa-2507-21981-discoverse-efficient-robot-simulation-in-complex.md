---
type: entity
tags: [paper, curated-index, awesome-real2sim2real, sun254667-r2s2r, awesome-sim2real]
status: complete
updated: 2026-10-06
arxiv: "2507.21981"
venue: "IROS 2025"
summary: "The first unified, modular, open-source 3DGS-based simulation framework for Real2Sim2Real robot learning. Features a holistic Real2Sim pipeline that synthesizes hyper-realistic geometry and appearance of complex real-wor"
related:
  - ../entities/awesome-real2sim2real.md
  - ../overview/sun-awesome-r2s2r-technology-map.md
  - ../methods/reinforcement-learning.md
  - ../methods/crisp-real2sim.md
  - ../tasks/locomotion.md
  - ../tasks/manipulation.md
  - ../entities/awesome-sim2real.md
  - ../overview/lc-awesome-sim2real-technology-map.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/papers/sun_awesome_r2s2r_2507_21981_discoverse-efficient-robot-simulation-in.md
  - ../../sources/papers/sun_awesome_r2s2r_catalog.md
  - ../../sources/repos/awesome-real2sim2real.md
  - ../../sources/repos/discoverse.md
  - ../../sources/papers/lc_awesome_sim2real_noarxiv_084_discoverse-efficient-robot-simulation-in.md
  - ../../sources/papers/lc_awesome_sim2real_catalog.md
  - ../../sources/repos/awesome-sim2real.md
---

# DISCOVERSE

**DISCOVERSE: Efficient Robot Simulation in Complex High-Fidelity Environments** 收录于 [Awesome-Real2Sim2Real](https://github.com/sun254667/Awesome-Real2Sim2Real) **第 035/063** 篇，分组 **41 Gaussian Splatting-based Frameworks**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

The first unified, modular, open-source 3DGS-based simulation framework for Real2Sim2Real robot learning. Features a holistic Real2Sim pipeline that synthesizes hyper-realistic geometry and appearance of complex real-world scenarios, paving the way for analyzing and bridging the Sim2Real gap

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 仿真策略迁移到真机 |
| Real2Sim | Real to Simulation | 真机数据重建/校准仿真 |
| R2S2R | Real2Sim2Real | 真机→仿真→真机闭环 |
| DR | Domain Randomization | 域随机化 |

## 为什么重要

- The first unified, modular, open-source 3DGS-based simulation framework for Real2Sim2Real robot learning. Features a holistic Real2Sim pipeline that synthesizes hyper-realistic geometry and appearance of complex real-world scenarios, paving the way for analyzing and bridging the Sim2Real gap
- 想横向对照同一分组的其他工作，可以从 [Awesome-Real2Sim2Real 技术地图](../overview/sun-awesome-r2s2r-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome-Real2Sim2Real](../entities/awesome-real2sim2real.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TB
    subgraph BG["背景节点 Real2Sim（§IV-A）"]
        direction TB
        CAP["真实场景多视图采集<br/>+ LixelKity K1 激光扫描几何"] --> REC["3DGS 重建<br/>用扫描几何正则化辐射场"]
        IMG1["场景单图"] --> DL["DiffusionLight<br/>生成 HDR 环境光图"]
    end
    subgraph OBJ["交互节点 Real2Sim（§IV-B）"]
        direction TB
        SCAN["Artec Leo 激光扫描<br/>转台多次扫描并配准"] --> MESH["带纹理 mesh"]
        GEN["CLAY 3D 生成<br/>（单张真实图像条件）"] --> MESH
        MESH --> RELIGHT["Blender 预处理 PBR 重打光<br/>（使用 HDR 环境光图）"]
        RELIGHT --> M2G["Mesh-Gaussian transfer<br/>每个三角面初始化一个 Gaussian"]
    end
    DL --> RELIGHT
    M2G --> DUAL["交互物体 / 机器人：<br/>3DGS（.ply）视觉 + mesh（MJCF）物理 双表示"]
    REC --> SCENE["背景：仅 3DGS 渲染表示"]
    DUAL --> ENG["DISCOVERSE 引擎"]
    SCENE --> ENG
    subgraph ENGB["引擎（§III）"]
        direction LR
        R["3DGS tile-based 渲染<br/>RGB / 深度 / LiDAR"]
        P["MuJoCo 物理<br/>接触 / 力控 / 传感器"]
        ROS["ROS2 接口<br/>关节空间与笛卡尔空间 API"]
    end
    ENG --> ENGB
    ENGB --> DR["域随机化：视频叠加 / HSV 增强 / gamma"]
    DR --> IL["模仿学习数据（如 ACT）<br/>零样本 Sim2Real 部署"]
```

图依据：论文 §III 系统架构 / Fig.2、§IV Real2Sim 流水线 / Fig.3、§IV-C。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 035/063 |
| 分组 | 41 Gaussian Splatting-based Frameworks |
| 出处 | IROS 2025 |
| 论文 | <https://arxiv.org/abs/2507.21981> |

## 核心机制（归纳）

### 策展导读要点

The first unified, modular, open-source 3DGS-based simulation framework for Real2Sim2Real robot learning. Features a holistic Real2Sim pipeline that synthesizes hyper-realistic geometry and appearance of complex real-world scenarios, paving the way for analyzing and bridging the Sim2Real gap

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2507.21981) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-r2s2r-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-r2s2r-technology-map.md) 的 **41 Gaussian Splatting-based Frameworks** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「DISCOVERSE」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **41 Gaussian Splatting-based Frameworks**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [DISCOVERSE 源码归档](../../sources/repos/discoverse.md)（<https://github.com/discoverse-dev/DISCOVERSE>）

- [`sources/papers/sun_awesome_r2s2r_2507_21981_discoverse-efficient-robot-simulation-in.md`](../../sources/papers/sun_awesome_r2s2r_2507_21981_discoverse-efficient-robot-simulation-in.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_r2s2r_catalog.md`](../../sources/papers/sun_awesome_r2s2r_catalog.md) — 列表总表
- [`sources/repos/awesome-real2sim2real.md`](../../sources/repos/awesome-real2sim2real.md)
- 论文：<https://arxiv.org/abs/2507.21981>
- [`sources/papers/lc_awesome_sim2real_noarxiv_084_discoverse-efficient-robot-simulation-in.md`](../../sources/papers/lc_awesome_sim2real_noarxiv_084_discoverse-efficient-robot-simulation-in.md) — 本条目策展摘录
- [`sources/papers/lc_awesome_sim2real_catalog.md`](../../sources/papers/lc_awesome_sim2real_catalog.md) — 列表总表
- [`sources/repos/awesome-sim2real.md`](../../sources/repos/awesome-sim2real.md)
- [arXiv PDF](https://arxiv.org/pdf/2507.21981)：已通读方法部分（实验与附录以略读为准）：§III 系统架构（引擎、资产、传感器）、§IV Real2Sim 流水线（含域随机化），§V 实验略读

## 推荐继续阅读

- [Awesome-Real2Sim2Real 仓库](https://github.com/sun254667/Awesome-Real2Sim2Real)
- [原文](https://arxiv.org/abs/2507.21981)
