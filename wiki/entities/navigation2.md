---
type: entity
tags: [repo, ros2, navigation, nav2, mobile-robot, planning]
status: complete
updated: 2026-05-27
related:
  - ../overview/navigation-slam-autonomy-stack.md
  - ../concepts/ros2-basics.md
  - ../comparisons/lidar-slam-lio-vio-selection.md
  - ./autoware.md
sources:
  - ../../sources/repos/navigation2.md
summary: "Navigation2（Nav2）是 ROS 2 标准导航框架：行为树、全局/局部规划插件、代价地图与恢复行为，承接 SLAM 输出的 map/odom 并输出 cmd_vel。"
---

# Navigation2（Nav2）

**Navigation2**（[ros-navigation/navigation2](https://github.com/ros-navigation/navigation2)）是 ROS 2 生态的 **移动机器人导航参考实现**，将定位结果、静态/动态代价地图与路径跟踪解耦为可替换插件。

## 为什么重要

- **事实标准**：AMR、服务机器人、科研平台普遍以 Nav2 为 **规划—控制中枢**。
- **插件化**：全局规划（NavFn、Smac Planner 等）、局部规划（DWB、RPP、MPPI 等）、控制器与 **行为树（BT）** 可独立升级。
- 与 [ROS 2 基础](../concepts/ros2-basics.md)、[slam_toolbox](https://github.com/SteveMacenski/slam_toolbox) 形成常见 **「建图 + 导航」** 闭环。

## 核心结构/机制

| 组件 | 说明 |
|------|------|
| **bt_navigator** | 行为树编排：导航、充电、恢复（spin、backup、wait） |
| **planner_server** | 全局路径；支持 2D grid、Hybrid-A* 等 |
| **controller_server** | 局部跟踪；输出 `geometry_msgs/Twist` |
| **smoother_server** | 路径平滑 |
| **costmap_2d** | 静态层 + 膨胀层 + 可选 3D 源（如 nvblox） |
| **lifecycle** | 节点生命周期管理，便于量产启动顺序 |

典型数据流：**SLAM/AMCL** 提供 `map`↔`odom`↔`base_link` → **global costmap** → **planner** → **local costmap** → **controller** → 底盘驱动。

## 常见误区或局限

- **误区：装好 Nav2 即可自动驾驶** — 仍需可靠定位、地图质量、传感器标定与安全监控。
- **误区：默认 DWB 适用所有底盘** — 全向、阿克曼、差速动力学差异大，需换 **controller / footprint**。
- **局限**：Nav2 面向 **平面移动底盘**；人形/腿式 **步态规划** 不在其范围（见 [OpenLoong](./openloong.md)）。

## 参考来源

- [sources/repos/navigation2.md](../../sources/repos/navigation2.md)
- [sources/repos/navigation_slam_autonomy_stack_catalog.md](../../sources/repos/navigation_slam_autonomy_stack_catalog.md)
- [ros-navigation/navigation2](https://github.com/ros-navigation/navigation2)

## 关联页面

- [导航·SLAM·自动驾驶栈总览](../overview/navigation-slam-autonomy-stack.md)
- [LiDAR / LIO / VIO 选型](../comparisons/lidar-slam-lio-vio-selection.md)
- [Autoware](./autoware.md)

## 推荐继续阅读

- [Nav2 官方文档](https://docs.nav2.org/)
