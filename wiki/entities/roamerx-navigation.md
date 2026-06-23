---
type: entity
tags: [navigation, slam, ros2, quadruped, genisom, path-planning]
status: complete
updated: 2026-06-23
related:
  - ./matrix-simulation-platform.md
  - ./quadruped-control-curriculum.md
  - ../concepts/hierarchical-quadruped-navigation-stack.md
  - ../tasks/vision-language-navigation.md
  - ../overview/navigation-slam-autonomy-stack.md
  - ../comparisons/lidar-slam-lio-vio-selection.md
  - ./navigation2.md
sources:
  - ../../sources/courses/quadruped_control_simulation_rl_curriculum.md
summary: "RoamerX 是智身科技开源的四足导航栈（GENISOM-AI）：ROS2 Nav2 增强、LiDAR SLAM、MPPI 局部规划与行为树，与 MATRiX/RL 运动策略分层集成。"
---

# RoamerX（智身四足导航栈）

**RoamerX**（社区开源版 **RoamerX Lite**）是智身科技（GENISOM AI）面向四足机器人的 **ROS 2 导航栈**：在 **Nav2** 基础上增强 **SLAM、全局/局部规划、行为树与代价地图**，支持与 **RL 运动策略** 及 [MATRiX](./matrix-simulation-platform.md) 仿真联合调试。课程第 7 章与 Final Project 以其为 **中层导航模块**。

## 一句话定义

> 四足机上的 **「感知建图 → 全局路径 → 局部跟踪 → 速度指令」** 开源栈，输出平面速度/航向给底层 RL 或 PD 运动控制器。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SLAM | Simultaneous Localization and Mapping | 同步定位与建图 |
| Nav2 | Navigation 2 | ROS 2 标准导航框架 |
| MPPI | Model Predictive Path Integral | 基于采样的模型预测路径积分局部控制器 |
| BT | Behavior Tree | 行为树，任务编排与状态机 |
| LiDAR | Light Detection and Ranging | 激光雷达，2D/3D 建图主传感器 |
| ROS2 | Robot Operating System 2 | 机器人中间件与消息生态 |
| LCM | Lightweight Communications and Marshalling | 部分机型可选的低延迟总线协议 |
| costmap | 2D Costmap | 障碍物栅格代价地图 |

## 为什么重要

1. **四足导航 ≠ 轮式 Nav2 直搬**：足式平台需要把 Nav2 输出的 **cmd_vel** 交给 **姿态自适应的 RL/PD 跟踪层**（见 [分层导航栈](../concepts/hierarchical-quadruped-navigation-stack.md)）。
2. **与 MATRiX 同厂商闭环**：仿真中验证 SLAM + 规划 + RL loco，再迁移到 **ZSL-1 / 钢镚** 等实机。
3. **开源可复现**：[`zsibot/genisom_roamerx_open`](https://github.com/zsibot/genisom_roamerx_open) 提供 Gazebo/UE 仿真与 NX/3588 硬件支持。

## 核心模块（开源仓库）

| 模块 | 功能 |
|------|------|
| `robot_slam` | LiDAR SLAM；2D 栅格图 + 3D 点云图 |
| `navigo_bt_navigator` | 行为树导航编排 |
| `navigo_mppi_controller` | MPPI 局部路径跟踪 |
| `navigo_costmap_2d` | 实时障碍代价地图 |
| `navigo_navfn_planner` | 全局路径规划 |

典型流程：`ros2 launch robot_slam slam.launch.py` → 建图/定位服务 → Nav2 跟踪目标点 → 底层 RL 执行步态。

## 与其他页面的关系

- 上游仿真：[MATRiX](./matrix-simulation-platform.md)
- 任务：[Vision-Language Navigation](../tasks/vision-language-navigation.md) — 自然语言目标可接在导航栈之上
- 对比：[LiDAR SLAM 选型](../comparisons/lidar-slam-lio-vio-selection.md)
- 通用人形/移动栈：[Navigation2](./navigation2.md)、[Navigation SLAM Overview](../overview/navigation-slam-autonomy-stack.md)

## 常见误区

- **误区：「有 RoamerX 就不需要 RL loco」。** 导航栈解决 **去哪里**；崎岖地形上的 **怎么站稳走** 仍靠 RL/PD 运动层。
- **误区：「3D 点云图够就不用 2D costmap」。** 局部规划与 Nav2 生态仍以 **2D 投影代价图** 为主力接口。

## 推荐继续阅读

- GitHub：[genisom_roamerx_open](https://github.com/zsibot/genisom_roamerx_open)
- [Quadruped Control Curriculum](./quadruped-control-curriculum.md) — Project 4 集成说明

## 参考来源

- [sources/courses/quadruped_control_simulation_rl_curriculum.md](../../sources/courses/quadruped_control_simulation_rl_curriculum.md) — 课程第 7 章与 Final Project
