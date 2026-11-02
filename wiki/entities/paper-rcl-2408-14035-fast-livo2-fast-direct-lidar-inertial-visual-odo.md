---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "2408.14035"
venue: "IEEE Transactions on Robotics"
code: https://github.com/hku-mars/FAST-LIVO2
summary: "RCL Awesome WAM 清单收录（Related resources）；细节以原文 PDF / 项目页为准。"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_2408_14035_fast-livo2-fast-direct-lidar-inertial-vi.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# FAST-LIVO2

**FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 205/564** 篇，分组 **Related resources**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

RCL Awesome WAM 清单收录（Related resources）；细节以原文 PDF / 项目页为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- RCL Awesome WAM 清单收录（Related resources）；细节以原文 PDF / 项目页为准。
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["IMU 测量"] --> P["ESIKF 传播 Propagation<br/>先验状态与协方差，后向传播补偿运动畸变"]
    B["异步 LiDAR 原始点"] --> C["Scan Recombination<br/>按相机采样时刻切分为 LiDAR 扫描"]
    C --> P
    P --> U1["顺序更新 ①：LiDAR 更新<br/>帧到地图 point-to-plane 残差"]
    M["统一体素地图<br/>(平面几何 + 视觉地图点与参考 patch)"] --> U1
    U1 --> V1["视觉地图点选择：可见体素查询 + 按需 raycasting<br/>剔除遮挡 / 深度不连续离群点"]
    M --> V1
    V1 --> U2["顺序更新 ②：视觉更新<br/>帧到地图图像光度误差（稀疏直接法）"]
    U2 --> X["输出位姿状态（含曝光时间估计）"]
    X --> L["局部建图更新：LiDAR 更新地图几何<br/>生成并更新视觉地图点与 patch"]
    L --> R["参考 patch 更新"]
    R --> N["独立线程：法向量精化"]
    L --> M
    N --> M
```

图依据：论文 §III 系统概述 / Fig.2、§IV ESIKF 顺序更新、§V 局部建图、§VI LiDAR 测量模型、§VII 视觉测量模型。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 205/564 |
| 分组 | Related resources |
| 出处 | IEEE Transactions on Robotics |
| 论文 | <https://ieeexplore.ieee.org/document/10757429/> |
| 代码/项目 | <https://github.com/hku-mars/FAST-LIVO2> |
| 子类 / 象限 | 三维表示与状态估计 · 不适用 |

## 核心机制（归纳）

### 策展导读要点

RCL Awesome WAM 清单收录（Related resources）；细节以原文 PDF / 项目页为准。

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://ieeexplore.ieee.org/document/10757429/) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **Related resources** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「FAST-LIVO2」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Related resources**，以及 Contribution 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/rcl_awesome_wam_2408_14035_fast-livo2-fast-direct-lidar-inertial-vi.md`](../../sources/papers/rcl_awesome_wam_2408_14035_fast-livo2-fast-direct-lidar-inertial-vi.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://ieeexplore.ieee.org/document/10757429/>
- [arXiv PDF](https://arxiv.org/pdf/2408.14035)：已通读方法部分（实验与附录以略读为准）：§III 系统概述、§IV ESIKF（Scan Recombination、传播、顺序更新）、§V 局部建图（视觉地图点生成与参考 patch 更新）、§VII-A 视觉地图点选择；§VI、实验与附录略读

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://ieeexplore.ieee.org/document/10757429/)
