---
type: entity
tags: [repo, unitree, unitreerobotics, lidar]
status: draft
updated: 2026-07-24
related:
  - ./unitree.md
  - ./unilidar-sdk.md
  - ./point-lio-unilidar.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/repos/unilidar_sdk2.md
  - ../../sources/repos/unitree.md
summary: "Unitree L2 LiDAR 官方 SDK。"
---

# unilidar_sdk2

**unilidar_sdk2** 是 [unitreerobotics](https://github.com/unitreerobotics) 组织下的官方仓库，归属 **LiDAR 感知** 主线。

## 一句话定义

Unitree L2 LiDAR 官方 SDK。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| LiDAR | Light Detection and Ranging | 激光雷达 |
| SLAM | Simultaneous Localization and Mapping | 同步定位与建图 |
| API | Application Programming Interface | 应用程序编程接口 |
| SDK | Software Development Kit | 软件开发工具包 |
| ROS 2 | Robot Operating System 2 | 机器人系统集成常用中间件 |

## 为什么重要

地形感知与建图依赖官方 LiDAR SDK/算法对齐；独立节点便于外设选型。

在宇树官方开源地图中，本仓是 **LiDAR 感知** 的独立节点；与其它仓的选型关系见 [Unitree](./unitree.md)。

## 核心信息

| 字段 | 内容 |
|------|------|
| 仓库 | [`unitreerobotics/unilidar_sdk2`](https://github.com/unitreerobotics/unilidar_sdk2) |
| 组织分类 | LiDAR 感知 |
| 星标（2026-07-24） | ~93 |
| 最近推送 | 2025-03-10 |
| 主要语言 | C++ |

## 工程实践

- The original C++ based SDK: unitreelidarsdk
- The package for parsing and publishing LiDAR data in the ROS environment: unitreelidarros
- The package for parsing and publishing LiDAR data in the ROS2 environment: unitreelidarros2

- 组织级导航与五条研发主线见 [Unitree 软件生态](./unitree.md)；原始归档见 [sources/repos/unilidar_sdk2.md](../../sources/repos/unilidar_sdk2.md)。

## 局限与风险

- **不要脱离代际混用**：SDK1/ROS1 遗产仓与 SDK2/ROS2/DDS 新栈的消息与启动方式不兼容。
- **README 边界优先**：功能承诺以官方 README 为准；星标高不等于适合你的机型/仿真器。
- **外设/第三方手部仓**：驱动与固件版本需与具体灵巧手/雷达硬件对齐，否则 Serial↔DDS 桥会静默失败。

## 关联页面

- [Unitree 组织枢纽](./unitree.md)
- [unilidar_sdk](./unilidar-sdk.md)
- [point_lio_unilidar](./point-lio-unilidar.md)
- [Locomotion](../tasks/locomotion.md)

## 参考来源

- [sources/repos/unilidar_sdk2.md](../../sources/repos/unilidar_sdk2.md)
- [sources/repos/unitree.md](../../sources/repos/unitree.md) — 组织级仓库地图
- 上游仓库：<https://github.com/unitreerobotics/unilidar_sdk2>

## 推荐继续阅读

- 官方开发者文档：<https://support.unitree.com/home/zh/developer>
- 组织总览：<https://github.com/unitreerobotics>
