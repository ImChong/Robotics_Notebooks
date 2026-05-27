---
type: entity
tags: [repo, lidar, lio, slam, ros, hku]
status: complete
updated: 2026-05-27
related:
  - ../entities/lio-sam.md
  - ../comparisons/lidar-slam-lio-vio-selection.md
  - ../entities/navigation2.md
sources:
  - ../../sources/repos/fast_lio.md
summary: "FAST-LIO 是高效鲁棒的 LiDAR-惯性里程计：ikd-Tree + 迭代 ESKF，适合 3D 旋转激光高频状态估计。"
---

# FAST-LIO

**FAST-LIO** 以 **紧耦合迭代卡尔曼滤波** 实现低延迟 3D LiDAR-惯性里程计。

## 为什么重要

- **速度与鲁棒性**：在 Livox/Velodyne 等平台广泛使用。
- **HKU Mars 生态**：与 FAST-LIO2、R2LIVE 等形成系列。

## 核心结构/机制

| 组件 | 说明 |
|------|------|
| **ikd-Tree** | 增量点云地图 |
| **IESKF** | 迭代误差状态卡尔曼 |
| **输出** | 高频 odometry / TF |

## 常见误区或局限

- **里程计 vs SLAM**：默认偏 odometry；全局一致需外部位姿图或回环模块。
- **对比**：[LIO-SAM](./lio-sam.md) 更重因子图与 GPS。

## 参考来源

- [sources/repos/fast_lio.md](../../sources/repos/fast_lio.md)
- [hku-mars/FAST_LIO](https://github.com/hku-mars/FAST_LIO)

## 关联页面

- [LIO-SAM](../entities/lio-sam.md)
- [LiDAR / LIO / VIO 选型](../comparisons/lidar-slam-lio-vio-selection.md)
- [Navigation2](../entities/navigation2.md)

## 推荐继续阅读

- https://arxiv.org/abs/2010.08196
