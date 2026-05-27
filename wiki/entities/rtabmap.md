---
type: entity
tags: [repo, slam, rgbd, ros, mapping, loop-closure]
status: complete
updated: 2026-05-27
related:
  - ../entities/navigation2.md
  - ../comparisons/lidar-slam-lio-vio-selection.md
sources:
  - ../../sources/repos/rtabmap.md
summary: "RTAB-Map 支持 RGB-D/立体/激光的多模态 SLAM 与记忆管理，提供独立 GUI 与 ROS/ROS2 集成。"
---

# RTAB-Map

**RTAB-Map** 以 **记忆管理（WM）** 处理长期建图与闭环，一套工具链覆盖采集到导航。

## 为什么重要

- **RGB-D 机器人常用**：TurtleBot 等历史生态。
- **GUI 友好**：适合快速原型与数据集回放。

## 核心结构/机制

| 能力 | 说明 |
|------|------|
| **Multi-session** | 多会话建图 |
| **Loop closure** | 词袋 + 图优化 |
| **Navigation** | 可选与 move_base/Nav2 衔接 |

## 常见误区或局限

- **高动态场景**：需运动补偿或过滤动态点。
- **与纯 LIO 对比**：激光-only 大场景可能不如 FAST-LIO 简洁。

## 参考来源

- [sources/repos/rtabmap.md](../../sources/repos/rtabmap.md)
- [introlab/rtabmap](https://github.com/introlab/rtabmap)

## 关联页面

- [Navigation2](../entities/navigation2.md)
- [LiDAR / LIO / VIO 选型](../comparisons/lidar-slam-lio-vio-selection.md)

## 推荐继续阅读

- http://introlab.github.io/rtabmap/
