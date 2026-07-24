---
type: entity
tags: [repo, unitree, unitreerobotics, sensor]
status: draft
updated: 2026-07-24
related:
  - ./unitree.md
  - ./teleimager.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/repos/UnitreecameraSDK.md
  - ../../sources/repos/unitree.md
summary: "Go1 相机 SDK（偏旧机型外设栈）。"
---

# UnitreecameraSDK

**UnitreecameraSDK** 是 [unitreerobotics](https://github.com/unitreerobotics) 组织下的官方仓库，归属 **相机外设** 主线。

## 一句话定义

Go1 相机 SDK（偏旧机型外设栈）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 软件开发工具包 |
| API | Application Programming Interface | 应用程序编程接口 |
| ROS | Robot Operating System | 机器人操作系统（经典一代） |
| G1 | Unitree G1 Humanoid | 宇树入门级人形平台 |
| DDS | Data Distribution Service | 分布式实时通信中间件 |

## 为什么重要

相机外设 SDK 影响视觉遥操作与感知实验的驱动层选择。

在宇树官方开源地图中，本仓是 **相机外设** 的独立节点；与其它仓的选型关系见 [Unitree](./unitree.md)。

## 核心信息

| 字段 | 内容 |
|------|------|
| 仓库 | [`unitreerobotics/UnitreecameraSDK`](https://github.com/unitreerobotics/UnitreecameraSDK) |
| 组织分类 | 相机外设 |
| 星标（2026-07-24） | ~120 |
| 最近推送 | 2022-04-28 |
| 主要语言 | C++ |

## 工程实践

- 从组织枢纽 [Unitree](./unitree.md) 确认本仓所属主线后再克隆。
- 对照上游 README 安装依赖，并与 SDK2 / ROS2 代际对齐。

- 组织级导航与五条研发主线见 [Unitree 软件生态](./unitree.md)；原始归档见 [sources/repos/UnitreecameraSDK.md](../../sources/repos/UnitreecameraSDK.md)。

## 局限与风险

- **不要脱离代际混用**：SDK1/ROS1 遗产仓与 SDK2/ROS2/DDS 新栈的消息与启动方式不兼容。
- **README 边界优先**：功能承诺以官方 README 为准；星标高不等于适合你的机型/仿真器。
- **外设/第三方手部仓**：驱动与固件版本需与具体灵巧手/雷达硬件对齐，否则 Serial↔DDS 桥会静默失败。

## 关联页面

- [Unitree 组织枢纽](./unitree.md)
- [teleimager](./teleimager.md)
- [Teleoperation](../tasks/teleoperation.md)

## 参考来源

- [sources/repos/UnitreecameraSDK.md](../../sources/repos/UnitreecameraSDK.md)
- [sources/repos/unitree.md](../../sources/repos/unitree.md) — 组织级仓库地图
- 上游仓库：<https://github.com/unitreerobotics/UnitreecameraSDK>

## 推荐继续阅读

- 官方开发者文档：<https://support.unitree.com/home/zh/developer>
- 组织总览：<https://github.com/unitreerobotics>
