---
type: entity
tags: [repo, unitree, unitreerobotics, ros]
status: draft
updated: 2026-07-24
related:
  - ./unitree.md
  - ./unitree-ros.md
  - ./unitree-ros2.md
  - ../concepts/ros2-basics.md
  - ./unitree-rl-mjlab.md
sources:
  - ../../sources/repos/unitree_ros_to_real.md
  - ../../sources/repos/unitree.md
summary: "ROS1 ↔ 真机桥接仓，提供 unitree_legged_msgs 与 high/low level 示例。"
---

# unitree_ros_to_real

**unitree_ros_to_real** 是 [unitreerobotics](https://github.com/unitreerobotics) 组织下的官方仓库，归属 **ROS 集成** 主线。

## 一句话定义

ROS1 ↔ 真机桥接仓，提供 unitree_legged_msgs 与 high/low level 示例。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ROS | Robot Operating System | 机器人操作系统（经典一代） |
| ROS 2 | Robot Operating System 2 | 机器人系统集成常用中间件 |
| DDS | Data Distribution Service | 分布式实时通信中间件 |
| URDF | Unified Robot Description Format | 统一机器人描述格式 |
| SDK | Software Development Kit | 软件开发工具包 |

## 为什么重要

ROS/ROS2 仍是实验室系统集成的默认胶水层；官方包决定消息、launch 与真机桥的可用边界。

在宇树官方开源地图中，本仓是 **ROS 集成** 的独立节点；与其它仓的选型关系见 [Unitree](./unitree.md)。

## 核心信息

| 字段 | 内容 |
|------|------|
| 仓库 | [`unitreerobotics/unitree_ros_to_real`](https://github.com/unitreerobotics/unitree_ros_to_real) |
| 组织分类 | ROS 集成 |
| 星标（2026-07-24） | ~170 |
| 最近推送 | 2023-06-27 |
| 主要语言 | C++ |

## 工程实践

- unitreeleggedsdk

- 组织级导航与五条研发主线见 [Unitree 软件生态](./unitree.md)；原始归档见 [sources/repos/unitree_ros_to_real.md](../../sources/repos/unitree_ros_to_real.md)。

## 局限与风险

- **不要脱离代际混用**：SDK1/ROS1 遗产仓与 SDK2/ROS2/DDS 新栈的消息与启动方式不兼容。
- **README 边界优先**：功能承诺以官方 README 为准；星标高不等于适合你的机型/仿真器。
- **外设/第三方手部仓**：驱动与固件版本需与具体灵巧手/雷达硬件对齐，否则 Serial↔DDS 桥会静默失败。

## 关联页面

- [Unitree 组织枢纽](./unitree.md)
- [unitree_ros](./unitree-ros.md)
- [unitree_ros2](./unitree-ros2.md)
- [ROS 2 基础](../concepts/ros2-basics.md)
- [unitree_rl_mjlab](./unitree-rl-mjlab.md)

## 参考来源

- [sources/repos/unitree_ros_to_real.md](../../sources/repos/unitree_ros_to_real.md)
- [sources/repos/unitree.md](../../sources/repos/unitree.md) — 组织级仓库地图
- 上游仓库：<https://github.com/unitreerobotics/unitree_ros_to_real>

## 推荐继续阅读

- 官方开发者文档：<https://support.unitree.com/home/zh/developer>
- 组织总览：<https://github.com/unitreerobotics>
