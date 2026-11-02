---
type: entity
tags: [paper, curated-index, awesome-real2sim2real, sun254667-r2s2r]
status: complete
updated: 2026-10-06
arxiv: "2602.02402"
venue: "arXiv 2026"
summary: "Improves resimulation accuracy and generalization on real-world robot manipulation by 20%, enabling stable simulation of complex tasks such as long-horizon cloth folding"
related:
  - ../entities/awesome-real2sim2real.md
  - ../overview/sun-awesome-r2s2r-technology-map.md
  - ../methods/reinforcement-learning.md
  - ../methods/crisp-real2sim.md
  - ../tasks/locomotion.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/sun_awesome_r2s2r_2602_02402_soma-a-real-to-sim-neural-simulator-for.md
  - ../../sources/papers/sun_awesome_r2s2r_catalog.md
  - ../../sources/repos/awesome-real2sim2real.md
---

# SoMA

**SoMA: A Real-to-Sim Neural Simulator for Robotic Soft-body Manipulation** 收录于 [Awesome-Real2Sim2Real](https://github.com/sun254667/Awesome-Real2Sim2Real) **第 048/063** 篇，分组 **42 Simulation Frameworks & Pipelines**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Improves resimulation accuracy and generalization on real-world robot manipulation by 20%, enabling stable simulation of complex tasks such as long-horizon cloth folding

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 仿真策略迁移到真机 |
| Real2Sim | Real to Simulation | 真机数据重建/校准仿真 |
| R2S2R | Real2Sim2Real | 真机→仿真→真机闭环 |
| DR | Domain Randomization | 域随机化 |

## 为什么重要

- Improves resimulation accuracy and generalization on real-world robot manipulation by 20%, enabling stable simulation of complex tasks such as long-horizon cloth folding
- 想横向对照同一分组的其他工作，可以从 [Awesome-Real2Sim2Real 技术地图](../overview/sun-awesome-r2s2r-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome-Real2Sim2Real](../entities/awesome-real2sim2real.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    IN["输入：三相机同步多视角 RGB 视频<br/>+ 机器人关节状态 R_t"] --> REC["重建：多视角几何估计相机位姿<br/>→ Gaussian Splatting 得到初始 G_0"]
    REC --> MAP["R2S 映射（§4.2.1）<br/>尺度 s 与刚体变换 (R, t) 对齐机器人/相机/世界坐标<br/>桌面平面拟合 → 重力方向"]
    IN --> FK["关节 q_t → 正运动学得末端位姿<br/>+ 夹爪开合 c_t → 机器人动作"]
    MAP --> FK
    REC --> HG["层次图：splat 逐级聚类成 cluster 节点"]
    FK --> RF["机器人力：控制点与 GS 节点的交互图<br/>图神经交互模块 Φ_θ"]
    HG --> EF["环境力：重力 + 靠近桌面的支撑力"]
    RF --> SUM["总作用力 f = f_env + f_rob"]
    EF --> SUM
    SUM --> DYN["力驱动层次动力学 ψ_θ<br/>由 G_t-1, G_t-2 与 f 预测速度/旋转<br/>自顶向下传播到 splat"]
    DYN --> GT["下一状态 G_t"]
    GT --> RND["渲染 Î_t,i"]
    RND --> LOSS["混合监督：掩码图像损失 L_img<br/>+ 层级质量守恒正则 L_mom"]
    OBS["真实图像 I_t,i + 物体掩码"] --> LOSS
    LOSS -.-> DYN
    TRAIN["两阶段多分辨率训练<br/>阶段1 大帧间隔 k 学全局运动<br/>阶段2 间隔 1 学细节（子序列长 n·k）"] -.-> DYN
    GT -.->|"自回归：作为下一步输入"| DYN
```

图依据：论文 §3（层次图模拟器）、§4.1-4.2（问题定义与 SoMA 框架，式 4-13）、Fig.2。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 048/063 |
| 分组 | 42 Simulation Frameworks & Pipelines |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2602.02402> |

## 核心机制（归纳）

### 策展导读要点

Improves resimulation accuracy and generalization on real-world robot manipulation by 20%, enabling stable simulation of complex tasks such as long-horizon cloth folding

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2602.02402) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-r2s2r-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-r2s2r-technology-map.md) 的 **42 Simulation Frameworks & Pipelines** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「SoMA」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **42 Simulation Frameworks & Pipelines**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_r2s2r_2602_02402_soma-a-real-to-sim-neural-simulator-for.md`](../../sources/papers/sun_awesome_r2s2r_2602_02402_soma-a-real-to-sim-neural-simulator-for.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_r2s2r_catalog.md`](../../sources/papers/sun_awesome_r2s2r_catalog.md) — 列表总表
- [`sources/repos/awesome-real2sim2real.md`](../../sources/repos/awesome-real2sim2real.md)
- 论文：<https://arxiv.org/abs/2602.02402>
- [arXiv PDF](https://arxiv.org/pdf/2602.02402)：已通读方法部分（实验与附录以略读为准）：§1 引言、§3 Preliminaries、§4 Method（4.1-4.2.4）。

## 推荐继续阅读

- [Awesome-Real2Sim2Real 仓库](https://github.com/sun254667/Awesome-Real2Sim2Real)
- [原文](https://arxiv.org/abs/2602.02402)
