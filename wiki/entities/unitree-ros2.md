---
type: entity
tags: [repo, unitree, unitreerobotics, ros2, dds, sdk, humanoid, quadruped]
status: complete
updated: 2026-07-24
related:
  - ./unitree.md
  - ./unitree-sdk2.md
  - ./unitree-ros.md
  - ./unitree-mujoco.md
  - ./unitree-g1-software-stack.md
  - ../concepts/ros2-basics.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/repos/unitree_ros2.md
  - ../../sources/repos/unitree_ros2_to_real.md
  - ../../sources/repos/unitree.md
summary: "unitree_ros2 让 ROS 2 直接使用 Unitree DDS 消息控制 Go2/B2/H1 等机型，无需再包一层 SDK 调用；推荐 Ubuntu 22.04 + Humble。Go1 专用的 unitree_ros2_to_real 为代际偏旧示例，不单独建页。"
---

# unitree_ros2

**unitree_ros2** 是宇树官方 ROS 2 功能包：底层与 SDK2 一样走 CycloneDDS，因此 **ROS 2 msg 可直接用于通信与控制**，而不必把每个调用再 wrap 一层 C++ SDK。

## 一句话定义

在 ROS 2 工作空间里编译 Unitree 的 `unitree_go` / `unitree_api` 等包，使 Nav、可视化与自研节点能以标准 ROS 2 话题/服务对接真机 DDS。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ROS 2 | Robot Operating System 2 | 机器人中间件；推荐 Humble |
| DDS | Data Distribution Service | ROS 2 与 SDK2 共用的通信层 |
| RMW | ROS Middleware | 需切换到 `rmw_cyclonedds_cpp` |
| IDL | Interface Definition Language | Unitree 自定义 msg 生成来源 |
| SDK2 | Unitree SDK version 2 | 并行的非 ROS 控制入口 |
| G1 | Unitree G1 Humanoid | 人形平台；与 H1/Go2 等同属新栈 |

## 为什么重要

- 实验室大量工具（RViz2、rosbag2、Nav2）默认 ROS 2；本仓是「不丢弃 ROS 生态」时的官方入口。
- 与 [`unitree_sdk2`](./unitree-sdk2.md) **同语义**：选型是语言/生态偏好，不是两套互斥协议。
- 对照 [`unitree_ros`](./unitree-ros.md)（ROS1 + Gazebo）可清晰划分 **遗产仿真栈** vs **现行真机 ROS 2 栈**。

## 核心原理

| 目录/包 | 作用 |
|---------|------|
| `cyclonedds_ws` | 工作空间；内含 Unitree msg（`unitree_go`、`unitree_api` 等） |
| `example` | 示例工作空间 |
| RMW 切换 | `ros-$DISTRO-rmw-cyclonedds-cpp`；Foxy 常需自编译匹配 0.10.2 的 cyclonedds |

**已测组合（上游）**：Ubuntu 20.04 + Foxy；Ubuntu 22.04 + **Humble（推荐）**。可用 `.devcontainer` / Dockerfile。

**历史仓说明（不单独成 wiki 节点）**：[unitree_ros2_to_real](https://github.com/unitreerobotics/unitree_ros2_to_real) 面向 **Go1** 的 ROS 2 真机示例（最近推送 2023），归档见 [`sources/repos/unitree_ros2_to_real.md`](../../sources/repos/unitree_ros2_to_real.md)；新机型请用本仓而非该遗产示例。

## 工程实践

1. 安装 `ros-$DISTRO-rmw-cyclonedds-cpp`、`rosidl-generator-dds-idl`、`libyaml-cpp-dev`。
2. **Foxy**：在未 source ROS 的终端中先编译与机器人一致的 CycloneDDS 0.10.x，再 source ROS 编译 Unitree 包；**Humble** 可跳过自编译 DDS 步骤（以上游 README 为准）。
3. `colcon build` 后 source 工作空间，按文档配置网口与 RMW 环境变量。
4. 同网段若同时跑 [`unitree_mujoco`](./unitree-mujoco.md) 仿真，注意 DDS 域与真机冲突。

## 局限与风险

- **发行版绑定**：Foxy 的 DDS 自举步骤繁琐，优先 Humble。
- **不是 Gazebo 高层行走包**：仿真 URDF/Gazebo 仍看 `unitree_ros`；本仓主攻真机 ROS 2 通信。
- **与 ROS1 桥不可混搭消息定义**。

## 关联页面

- [unitree_sdk2](./unitree-sdk2.md)
- [unitree_ros（ROS1）](./unitree-ros.md)
- [unitree_mujoco](./unitree-mujoco.md)
- [ROS 2 基础](../concepts/ros2-basics.md)
- [Unitree](./unitree.md)

## 参考来源

- [sources/repos/unitree_ros2.md](../../sources/repos/unitree_ros2.md)
- [sources/repos/unitree_ros2_to_real.md](../../sources/repos/unitree_ros2_to_real.md)
- 上游：<https://github.com/unitreerobotics/unitree_ros2>

## 推荐继续阅读

- SDK2 文档：<https://support.unitree.com/home/zh/developer>

