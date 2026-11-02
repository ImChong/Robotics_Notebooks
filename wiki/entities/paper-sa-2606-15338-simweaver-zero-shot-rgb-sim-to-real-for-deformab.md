---
type: entity
tags: [paper, curated-index, awesome-real2sim2real, sun254667-r2s2r]
status: complete
updated: 2026-10-06
arxiv: "2606.15338"
venue: "arXiv 2026"
summary: "Zero-shot RGB sim-to-real framework for deformable manipulation"
related:
  - ../entities/awesome-real2sim2real.md
  - ../overview/sun-awesome-r2s2r-technology-map.md
  - ../methods/reinforcement-learning.md
  - ../methods/crisp-real2sim.md
  - ../tasks/locomotion.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/sun_awesome_r2s2r_2606_15338_simweaver-zero-shot-rgb-sim-to-real-for.md
  - ../../sources/papers/sun_awesome_r2s2r_catalog.md
  - ../../sources/repos/awesome-real2sim2real.md
---

# SimWeaver

**SimWeaver: Zero-Shot RGB Sim-to-Real for Deformable Manipulation** 收录于 [Awesome-Real2Sim2Real](https://github.com/sun254667/Awesome-Real2Sim2Real) **第 014/063** 篇，分组 **22 Zero-Shot Sim2Real**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Zero-shot RGB sim-to-real framework for deformable manipulation

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 仿真策略迁移到真机 |
| Real2Sim | Real to Simulation | 真机数据重建/校准仿真 |
| R2S2R | Real2Sim2Real | 真机→仿真→真机闭环 |
| DR | Domain Randomization | 域随机化 |

## 为什么重要

- Zero-shot RGB sim-to-real framework for deformable manipulation
- 想横向对照同一分组的其他工作，可以从 [Awesome-Real2Sim2Real 技术地图](../overview/sun-awesome-r2s2r-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome-Real2Sim2Real](../entities/awesome-real2sim2real.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    AST["SimWeaver-Asset<br/>CLOTH3D 服装网格 + 塑料袋等资产<br/>实测物理参数（RGBench 织物库）"]
    SIM["SimWeaver-Sim<br/>RGBench 求解器 + 主动碰撞区域机制<br/>抑制穿透与抓取抖动"]
    AST --> SIM
    subgraph SYN["SimWeaver-Syn：拓扑感知轨迹合成"]
        TAS["TAS 拓扑邻接选择<br/>规范网格上的语义地标图 G，边为可行双臂抓取对"]
        PRED["拓扑感知状态谓词<br/>遮挡、分层、可达、防交叉臂等"]
        LOOP["闭环合成<br/>重新观测 → 选取抓取对 → 执行并验证 → 失败则重选（至多 T 次）"]
        PLAN["约束规划器生成可执行轨迹"]
        TAS --> LOOP
        PRED --> TAS
        PRED --> LOOP
        LOOP --> PLAN
    end
    SIM --> TAS
    SIM --> RAND["SimWeaver-Real 域随机化<br/>物理驱动布料状态初始化、传感器感知光度增强、机器人/场景 DR"]
    PLAN --> DEMO["每任务 200 条仿真示范"]
    RAND --> DEMO
    DEMO --> POL["全量微调 π0.5 RGB 策略"]
    POL --> REAL["真机零样本部署<br/>双 Piper 臂 + 3 台 RealSense D435i"]
```

图依据：论文 Fig.1 与 §3（SimWeaver-Sim）、§4（SimWeaver-Syn）、§5（SimWeaver-Real）、§6（SimWeaver-Asset）、§7 实验设置

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 014/063 |
| 分组 | 22 Zero-Shot Sim2Real |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2606.15338> |

## 核心机制（归纳）

### 策展导读要点

Zero-shot RGB sim-to-real framework for deformable manipulation

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2606.15338) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-r2s2r-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-r2s2r-technology-map.md) 的 **22 Zero-Shot Sim2Real** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「SimWeaver」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_r2s2r_2606_15338_simweaver-zero-shot-rgb-sim-to-real-for.md`](../../sources/papers/sun_awesome_r2s2r_2606_15338_simweaver-zero-shot-rgb-sim-to-real-for.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_r2s2r_catalog.md`](../../sources/papers/sun_awesome_r2s2r_catalog.md) — 列表总表
- [`sources/repos/awesome-real2sim2real.md`](../../sources/repos/awesome-real2sim2real.md)
- 论文：<https://arxiv.org/abs/2606.15338>
- [arXiv PDF](https://arxiv.org/pdf/2606.15338)：已通读方法部分（实验与附录以略读为准）：§3–§6（SimWeaver-Sim / Syn / Real / Asset 四个方法节）、§7 实验设置与主结果；附录略读

## 推荐继续阅读

- [Awesome-Real2Sim2Real 仓库](https://github.com/sun254667/Awesome-Real2Sim2Real)
- [原文](https://arxiv.org/abs/2606.15338)
