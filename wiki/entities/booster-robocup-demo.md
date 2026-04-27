---
type: entity
tags: [humanoid, robocup, ros2, booster-robotics, perception, rl]
status: drafting
updated: 2026-04-27
related:
  - ./robot-lab.md
  - ../concepts/ros2-basics.md
  - ../methods/auto-labeling-pipelines.md
sources:
  - ../../sources/repos/booster-robocup-demo.md
summary: "Booster Robotics RoboCup Demo 是专为 Booster 系列人形机器人设计的足球比赛自主决策框架，集成了 YOLOv8 感知与基于强化学习的视觉踢球算法。"
---

# Booster Robotics RoboCup Demo

**Booster Robotics RoboCup Demo** 是由 [Booster Robotics](https://github.com/BoosterRobotics) 官方维护的开源项目，旨在为其人形机器人（尤其是 **Booster K1**）提供参加 RoboCup 足球比赛的完整自主软件方案。

## 系统架构

该系统基于 **ROS 2 Humble** 构建，采用了典型的感知-决策-执行分层架构：

### 1. 感知层 (`vision`)
- **目标检测**: 使用 **YOLOv8** 模型实时识别足球、机器人对手/队友及场地边线。
- **坐标变换**: 将 2D 图像坐标通过相机外参及地面约束转换为机器人坐标系下的 3D 位姿。
- **推理优化**: 在真机端（NVIDIA Orin）使用 **TensorRT** 加速，仿真端支持 ONNX Runtime。

### 2. 决策层 (`brain`)
- **逻辑控制**: 采用分层状态机，管理机器人的比赛状态（Search, Chase, Align, Kick）。
- **通信对接**: 集成了 RoboCup 官方的 GameController 协议，监听裁判机的比赛指令（如 Start, Ready, Set, Finish）。
- **踢球控制**: 包含基础踢球逻辑和高级的 **`RLVisionKick`**（基于强化学习的视觉引导踢球），后者能够根据实时视觉反馈动态调整步态，实现更精准的击球。

### 3. 通信层 (`game_controller`)
- 负责将裁判机的网络广播包转换为 ROS 2 话题，实现比赛状态的自动化流转。

## 硬件与软件需求

- **硬件**: 
    - 官方支持：Booster K1 (固件 v1.5.2+), Booster T1。
    - 计算平台：支持 NVIDIA Jetpack 6.2。
- **软件**: 
    - 操作系统：Ubuntu 22.04 + ROS 2 Humble。
    - 核心依赖：Booster Robotics SDK。

## 技术特色：RLVisionKick

该框架的一个亮点是集成了强化学习（RL）踢球模块。相比传统的基于预设轨迹的踢球（Keyframe-based），RL 踢球能够：
1. **闭环调节**: 在接近足球的过程中，根据视觉偏差实时修正脚部位置。
2. **高成功率**: 在不平整草地或足球微小移动时表现出更好的鲁棒性。

## 开发与仿真

该项目支持在 **Booster Studio** 仿真器中运行，允许开发者在没有真机的情况下：
- 验证视觉算法的有效性。
- 调试状态机逻辑与协作策略。
- 模拟不同的比赛地形与光照条件。

## 关联页面

- [robot_lab (IsaacLab 扩展框架)](./robot-lab.md) — 了解 Booster T1 在仿真环境中的适配。
- [ROS 2 基础](../concepts/ros2-basics.md) — 该项目的通信底座。
- [自动化标注流水线](../methods/auto-labeling-pipelines.md) — 针对比赛环境定制 YOLO 模型的关键。

## 参考来源

- [BoosterRobotics/robocup_demo 源码仓库](../../sources/repos/booster-robocup-demo.md)
- Booster Robotics 官方文档
