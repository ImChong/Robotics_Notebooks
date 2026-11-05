---
type: entity
project_id: robotics-library-rl
tags:
  - kinematics
  - dynamics
  - motion-planning
  - cpp
  - library
  - independent
status: complete
updated: 2026-11-03
code: https://github.com/roboticslibrary/rl
related:
  - ../entities/pinocchio.md
  - ../entities/ompl.md
  - ../methods/trajectory-optimization.md
  - ../overview/cpp-motion-control-open-source-libraries-map.md
sources:
  - ../../sources/repos/robotics_library_rl.md
  - ../../sources/blogs/wechat_qt_adventure_cpp_motion_control_libs_2026-11-02.md
summary: "Robotics Library（RL）是自包含的 C++ 库，覆盖刚体运动学与动力学、运动规划与控制：空间向量代数、多体系统、硬件抽象、路径规划、碰撞检测与可视化，支持 Linux / macOS / Windows，BSD 许可、商用免费。"
---

# Robotics Library（RL）

**Robotics Library（RL）**（[https://github.com/roboticslibrary/rl](https://github.com/roboticslibrary/rl)）是 [Qt历险记《C/C++ 运动控制开源库分类整理》](https://mp.weixin.qq.com/s/RIXXIltQ0ABoBO_OVLZTZw) 中「机器人动力学 / 整机控制库」一类的项目。本页以**一手资料**为准，文章说法放在「文章说法核对」里对照。

## 一句话定义

**Robotics Library（RL）是自包含的 C++ 库，覆盖刚体运动学与动力学、运动规划与控制：空间向量代数、多体系统、硬件抽象、路径规划、碰撞检测与可视化，支持 Linux / macOS / Windows，BSD 许可、商用免费。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| OTG | Online Trajectory Generation | 在线轨迹生成 |
| PTP | Point-To-Point | 点到点运动 |

## 为什么重要

- 在文章的选型建议里属于「机器人动力学 / 整机控制库」；横向对比见 [C/C++ 运动控制开源库地图：轨迹生成、动力学、工业总线与 ROS 栈（一手资料核对版）](../overview/cpp-motion-control-open-source-libraries-map.md)。

## 一手资料

| 项 | 内容 |
|----|------|
| **仓库 / 官方来源** | [https://github.com/roboticslibrary/rl](https://github.com/roboticslibrary/rl) |
| **许可（核查）** | BSD-2-Clause（LICENSE.md；README「All source code files of RL are licensed under the permissive BSD 2-clause license」） |
| **最近提交** | 2025-04-15 |
| **论文 / 文档** | 官方站点 roboticslibrary.org；本次未核对论文 |

**一手描述：** README：self-contained C++ library for rigid body kinematics and dynamics, motion planning, and control；spatial vector algebra、multibody systems、hardware abstraction、path planning、collision detection、visualization；用 CMake，可用 Clang / GCC / Visual Studio 编译。

## 文章说法核对

- **文章：** C++；BSD（商用免费）；运动学 + 动力学 + 碰撞检测 + 路径规划 + 可视化，独立库不强制 ROS；机械臂离线 / 在线规划，Windows + Linux 实时内核都支持。
- 许可：README 写明为 **BSD 2-clause**（文章只写「BSD」）。
- 「Windows + Linux 实时内核都支持」：README 只说明可在 Linux / macOS / Windows 上运行，**未提及实时内核**；该点来自文章，未核实。
- 仓库最近提交 2025-04-15，相对其他项目更新较慢。

## 关联页面

- [pinocchio](../entities/pinocchio.md)
- [ompl](../entities/ompl.md)
- [trajectory-optimization](../methods/trajectory-optimization.md)
- [cpp-motion-control-open-source-libraries-map](../overview/cpp-motion-control-open-source-libraries-map.md)

## 参考来源

- [robotics_library_rl.md](../../sources/repos/robotics_library_rl.md)
- [wechat_qt_adventure_cpp_motion_control_libs_2026-11-02.md](../../sources/blogs/wechat_qt_adventure_cpp_motion_control_libs_2026-11-02.md)
- [原文（微信公众号）](https://mp.weixin.qq.com/s/RIXXIltQ0ABoBO_OVLZTZw)
