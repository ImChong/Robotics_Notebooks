---
type: overview
tags: [topic, topic-state-estimation, slam, odometry, ekf, perception]
status: complete
updated: 2026-07-01
summary: "状态估计专题汇总：本体感知融合、SLAM/VIO/LIO 选型与 Kalman/优化估计框架，服务 locomotion 与导航中的位姿与速度估计。"
---

# 状态估计（专题汇总）

> **图谱专题视图**：本页是知识图谱「📊 状态估计 (State Estimation)」专题的统一入口；在 [图谱专题视图](../../docs/graph.html?topic=state-estimation) 筛选时，本节点为汇总锚点。

## 一句话定义

**状态估计** 从 **IMU、关节编码器、相机、LiDAR 等传感器** 融合出机器人位姿、速度与接触/地形状态，是感知式 locomotion 与导航的控制输入基础。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| EKF | Extended Kalman Filter | 非线性系统常用滤波框架 |
| UKF | Unscented Kalman Filter | 无迹卡尔曼变体 |
| VIO | Visual-Inertial Odometry | 视觉-惯性里程计 |
| LIO | LiDAR-Inertial Odometry | 激光-惯性里程计 |
| SLAM | Simultaneous Localization and Mapping | 同步定位与建图 |

## 为什么重要

- **盲走 vs 感知走**：估计质量决定能否在复杂地形稳定行走。
- **多传感器时间对齐**：与 [通信/时钟同步](./topic-communication.md) 强相关。
- **Sim2Real 感知 gap**：仿真传感器噪声模型与真机不一致会拖垮策略。

## 本专题覆盖什么

| 层次 | 典型问题 | 站内入口 |
|------|----------|----------|
| 概念 | 状态估计总览 | [State Estimation](../concepts/state-estimation.md) |
| 融合 | 多传感器融合 | [Sensor Fusion](../concepts/sensor-fusion.md) |
| 对比 | KF vs 优化估计 | [Kalman vs Optimization Estimation](../comparisons/kalman-filter-vs-optimization-based-estimation.md) |
| 对比 | LiDAR SLAM 选型 | [LiDAR SLAM / LIO / VIO Selection](../comparisons/lidar-slam-lio-vio-selection.md) |
| 导航栈 | SLAM 与自主导航 | [Navigation SLAM Autonomy Stack](./navigation-slam-autonomy-stack.md) |

## 与其他专题的关系

- **[Locomotion](./topic-locomotion.md)**：感知式越障依赖状态估计。
- **[Sim2Real](./topic-sim2real.md)**：感知域随机与噪声建模。
- **[视觉骨干](./topic-vision-backbone.md)**：VIO 依赖视觉特征质量。

## 关联页面

- [Ultra-Fusion（多传感器 SLAM）](../entities/paper-ultra-fusion-multi-sensor-slam.md) — 统一滑窗 LVIO/LVWIO、退化调度与在线时空标定（arXiv:2606.21223）
- [X-IONet（跨平台惯性里程计）](../entities/paper-x-ionet-cross-platform-inertial-odometry.md) — 单 IMU 行人/四足 IO + EKF（IEEE RA-L 2026）
- [Contact Estimation](../concepts/contact-estimation.md)
- [Terrain Latent Representation](../concepts/terrain-latent-representation.md)
- [3D Spatial VQA](../concepts/3d-spatial-vqa.md)

## 参考来源

- 本库归纳自 [State Estimation](../concepts/state-estimation.md) 及 SLAM/VIO 对比页
- 图谱专题定义：[docs/topic-filters.js](../../docs/topic-filters.js)（`state-estimation` 命中规则）
