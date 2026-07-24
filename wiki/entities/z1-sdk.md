---
type: entity
tags: [repo, unitree, unitreerobotics, arm, manipulation, z1]
status: complete
updated: 2026-07-24
related:
  - ./unitree.md
  - ./unifolm-world-model-action.md
  - ./unitree-lerobot.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/repos/z1_sdk.md
  - ../../sources/repos/z1_ros.md
  - ../../sources/repos/z1_controller.md
  - ../../sources/repos/z1_joystick.md
  - ../../sources/repos/unitree.md
summary: "Unitree Z1 机械臂软件栈总页：以 z1_sdk 为入口，合并 z1_ros / z1_controller / z1_joystick 等配套仓说明，避免四个重复 stub 节点；详细 API 以 Z1 开发者文档为准。"
---

# Unitree Z1 软件栈（z1_sdk 等）

**Z1** 是宇树六轴协作机械臂产品线。组织下拆有 `z1_sdk`、`z1_ros`、`z1_controller`、`z1_joystick` 等多个仓库；本页作为**唯一 wiki 节点**归纳，避免图谱重复。

## 一句话定义

Z1 的控制与仿真入口集合——SDK 对接开发者文档，ROS/手柄仓提供仿真与遥操作胶水。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | `z1_sdk` 主入口 |
| ROS | Robot Operating System | `z1_ros` 仿真包 |
| API | Application Programming Interface | 以开发者站点为准 |
| IL | Imitation Learning | UnifoLM-WMA 等使用 Z1 数据 |
| DDS | Data Distribution Service | 部分手/臂组合会经 DDS |
| WMA | World-Model-Action | 官方 Z1 数据集应用方之一 |

## 为什么重要

- UnifoLM-WMA 等多份官方开源数据以 **Z1 / 双臂 Z1** 采集，需要可复现的臂侧软件入口。
- 与腿式 SDK2 主线分离：做桌面操作时不应误装 `unitree_sdk2` 当 Z1 驱动。
- 多仓并存但职责碎——合并叙述减少「四个几乎空的节点」。

## 核心原理

| 仓库 | 角色 |
|------|------|
| [`z1_sdk`](https://github.com/unitreerobotics/z1_sdk) | SDK 工具；README 指向中英文开发者文档 |
| [`z1_ros`](https://github.com/unitreerobotics/z1_ros) | Z1 仿真 ROS 包 |
| [`z1_controller`](https://github.com/unitreerobotics/z1_controller) | 控制器相关 |
| [`z1_joystick`](https://github.com/unitreerobotics/z1_joystick) | 宇树手柄控制 Z1 |

详细消息、运动学与安全限制：**以 [Z1 开发者文档](https://support.unitree.com/home/zh/Z1_developer) 为准**（本页不复制过时 API 表）。

## 工程实践

1. 从开发者文档确认固件与 SDK 版本矩阵。
2. 仿真优先 `z1_ros`；真机跟 `z1_sdk` 示例。
3. 需要 IL 数据时对照 HF 上 Z1_StackBox 等数据集与 [`unifolm-world-model-action`](./unifolm-world-model-action.md)。
4. 与 G1 灵巧手组合时，另见 [灵巧手服务](./unitree-dexterous-hand-services.md)。

## 局限与风险

- 上游 `z1_sdk` README 极短，**文档站点才是真源**；勿只依赖 GitHub 根 README。
- 力矩/速度限制与碰撞检测配置错误会导致危险运动。
- 不要与 Go/H1/G1 的 SDK2 示例混用网段与进程。

## 关联页面

- [UnifoLM-WMA](./unifolm-world-model-action.md)
- [Manipulation](../tasks/manipulation.md)
- [Teleoperation](../tasks/teleoperation.md)
- [Unitree](./unitree.md)

## 参考来源

- [sources/repos/z1_sdk.md](../../sources/repos/z1_sdk.md)
- [sources/repos/z1_ros.md](../../sources/repos/z1_ros.md)
- [sources/repos/z1_controller.md](../../sources/repos/z1_controller.md)
- [sources/repos/z1_joystick.md](../../sources/repos/z1_joystick.md)
- 文档：<https://support.unitree.com/home/zh/Z1_developer>

## 推荐继续阅读

- 英文文档：<https://support.unitree.com/home/en/Z1_developer>

