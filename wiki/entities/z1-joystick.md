---
type: entity
tags: [repo, unitree, unitreerobotics, arm]
status: draft
updated: 2026-07-24
related:
  - ./unitree.md
  - ./z1-sdk.md
  - ./z1-ros.md
  - ./z1-controller.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/repos/z1_joystick.md
  - ../../sources/repos/unitree.md
summary: "用宇树手柄控制 Z1 的 ROS 包。"
---

# z1_joystick

**z1_joystick** 是 [unitreerobotics](https://github.com/unitreerobotics) 组织下的官方仓库，归属 **Z1 机械臂** 主线。

## 一句话定义

用宇树手柄控制 Z1 的 ROS 包。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 软件开发工具包 |
| ROS | Robot Operating System | 机器人操作系统（经典一代） |
| API | Application Programming Interface | 应用程序编程接口 |
| DDS | Data Distribution Service | 分布式实时通信中间件 |
| URDF | Unified Robot Description Format | 统一机器人描述格式 |

## 为什么重要

Z1 是宇树机械臂产品线；SDK/ROS/手柄包构成独立于腿式主线的控制栈。

在宇树官方开源地图中，本仓是 **Z1 机械臂** 的独立节点；与其它仓的选型关系见 [Unitree](./unitree.md)。

## 核心信息

| 字段 | 内容 |
|------|------|
| 仓库 | [`unitreerobotics/z1_joystick`](https://github.com/unitreerobotics/z1_joystick) |
| 组织分类 | Z1 机械臂 |
| 星标（2026-07-24） | ~8 |
| 最近推送 | 2024-02-26 |
| 主要语言 | C++ |

## 工程实践

- 从组织枢纽 [Unitree](./unitree.md) 确认本仓所属主线后再克隆。
- 对照上游 README 安装依赖，并与 SDK2 / ROS2 代际对齐。

- 组织级导航与五条研发主线见 [Unitree 软件生态](./unitree.md)；原始归档见 [sources/repos/z1_joystick.md](../../sources/repos/z1_joystick.md)。

## 局限与风险

- **不要脱离代际混用**：SDK1/ROS1 遗产仓与 SDK2/ROS2/DDS 新栈的消息与启动方式不兼容。
- **README 边界优先**：功能承诺以官方 README 为准；星标高不等于适合你的机型/仿真器。
- **外设/第三方手部仓**：驱动与固件版本需与具体灵巧手/雷达硬件对齐，否则 Serial↔DDS 桥会静默失败。

## 关联页面

- [Unitree 组织枢纽](./unitree.md)
- [z1_sdk](./z1-sdk.md)
- [z1_ros](./z1-ros.md)
- [z1_controller](./z1-controller.md)
- [Manipulation](../tasks/manipulation.md)

## 参考来源

- [sources/repos/z1_joystick.md](../../sources/repos/z1_joystick.md)
- [sources/repos/unitree.md](../../sources/repos/unitree.md) — 组织级仓库地图
- 上游仓库：<https://github.com/unitreerobotics/z1_joystick>

## 推荐继续阅读

- 官方开发者文档：<https://support.unitree.com/home/zh/developer>
- 组织总览：<https://github.com/unitreerobotics>
