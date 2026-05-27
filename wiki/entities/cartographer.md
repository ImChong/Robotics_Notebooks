---
type: entity
tags: [repo, slam, google, 2d, 3d, ros]
status: complete
updated: 2026-05-27
related:
  - ../entities/navigation2.md
  - ../overview/navigation-slam-autonomy-stack.md
  - ../comparisons/lidar-slam-lio-vio-selection.md
sources:
  - ../../sources/repos/cartographer.md
summary: "Google Cartographer 提供实时 2D/3D SLAM：子图 scan matching + 位姿图优化，cartographer_ros 广泛用于工业与科研。"
---

# Cartographer

**Cartographer** 是 Google 开源的 **子图 SLAM** 系统，支持 2D/3D 激光与多传感器配置。

## 为什么重要

- **工业成熟度**：大量仓储/服务机器人历史部署案例。
- **子图架构**：局部匹配 + 全局优化，利于大规模环境。

## 核心结构/机制

| 模块 | 说明 |
|------|------|
| **Local SLAM** | 扫描匹配、子图插入 |
| **Global SLAM** | 闭环检测、位姿图优化 |
| **cartographer_ros** | ROS/ROS2 桥接与传感器配置 |

## 常见误区或局限

- **误区：配置即开箱即用** — URDF、激光话题、帧率与时间同步需仔细标定。
- **对比**：与 slam_toolbox 相比 ROS2 迁移与社区重心不同，选型看团队栈。

## 参考来源

- [sources/repos/cartographer.md](../../sources/repos/cartographer.md)
- [cartographer-project/cartographer](https://github.com/cartographer-project/cartographer)

## 关联页面

- [Navigation2](../entities/navigation2.md)
- [导航·SLAM·自动驾驶栈总览](../overview/navigation-slam-autonomy-stack.md)
- [LiDAR / LIO / VIO 选型](../comparisons/lidar-slam-lio-vio-selection.md)

## 推荐继续阅读

- https://google-cartographer-ros.readthedocs.io/
