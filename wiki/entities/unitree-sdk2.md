---
type: entity
tags: [repo, unitree, unitreerobotics, sdk]
status: draft
updated: 2026-07-24
related:
  - ./unitree.md
  - ./unitree-sdk2-python.md
  - ./unitree-ros2.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/repos/unitree_sdk2.md
  - ../../sources/repos/unitree.md
summary: "Unitree 当前主推的 C++ SDK v2（CycloneDDS），新机型真机控制与自定义部署的默认底层入口。"
---

# unitree_sdk2

**unitree_sdk2** 是 [unitreerobotics](https://github.com/unitreerobotics) 组织下的官方仓库，归属 **底层 SDK / 通信** 主线。

## 一句话定义

Unitree 当前主推的 C++ SDK v2（CycloneDDS），新机型真机控制与自定义部署的默认底层入口。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 软件开发工具包 |
| DDS | Data Distribution Service | 分布式实时通信中间件 |
| ROS 2 | Robot Operating System 2 | 机器人系统集成常用中间件 |
| API | Application Programming Interface | 应用程序编程接口 |
| G1 | Unitree G1 Humanoid | 宇树入门级人形平台 |

## 为什么重要

真机通信与低层控制是所有 RL/IL/遥操作部署的共同底座；弄错 SDK 代际会导致整条链路无法联调。

在宇树官方开源地图中，本仓是 **底层 SDK / 通信** 的独立节点；与其它仓的选型关系见 [Unitree](./unitree.md)。

## 核心信息

| 字段 | 内容 |
|------|------|
| 仓库 | [`unitreerobotics/unitree_sdk2`](https://github.com/unitreerobotics/unitree_sdk2) |
| 组织分类 | 底层 SDK / 通信 |
| 星标（2026-07-24） | ~1243 |
| 最近推送 | 2026-07-09 |
| 主要语言 | C++ |

## 工程实践

- OS  (Ubuntu 20.04 LTS)
- CPU  (aarch64 and x8664)
- Compiler  (gcc version 9.4.0)
- CMake (version 3.10 or higher)

- 组织级导航与五条研发主线见 [Unitree 软件生态](./unitree.md)；原始归档见 [sources/repos/unitree_sdk2.md](../../sources/repos/unitree_sdk2.md)。

## 局限与风险

- **不要脱离代际混用**：SDK1/ROS1 遗产仓与 SDK2/ROS2/DDS 新栈的消息与启动方式不兼容。
- **README 边界优先**：功能承诺以官方 README 为准；星标高不等于适合你的机型/仿真器。
- **外设/第三方手部仓**：驱动与固件版本需与具体灵巧手/雷达硬件对齐，否则 Serial↔DDS 桥会静默失败。

## 关联页面

- [Unitree 组织枢纽](./unitree.md)
- [unitree_sdk2_python](./unitree-sdk2-python.md)
- [unitree_ros2](./unitree-ros2.md)
- [Sim2Real](../concepts/sim2real.md)

## 参考来源

- [sources/repos/unitree_sdk2.md](../../sources/repos/unitree_sdk2.md)
- [sources/repos/unitree.md](../../sources/repos/unitree.md) — 组织级仓库地图
- 上游仓库：<https://github.com/unitreerobotics/unitree_sdk2>

## 推荐继续阅读

- 官方开发者文档：<https://support.unitree.com/home/zh/developer>
- 组织总览：<https://github.com/unitreerobotics>
