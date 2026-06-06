---
type: entity
tags: [manipulator, cobot, force-control, open-source, franka, research]
status: complete
updated: 2026-05-18
related:
  - ../tasks/manipulation.md
  - ../overview/robot-open-source-wechat-issue02-curator.md
  - ../concepts/ros2-basics.md
  - ./kinova-gen3.md
sources:
  - ../../sources/blogs/wechat_jixie_robot_open_source_treasury_issue02_10_robots.md
summary: "Franka Research 3：Franka Robotics 七轴科研协作臂；官方 docs 站点与 frankarobotics GitHub 组织提供 libfranka、ROS 与示例。"
---

# Franka Research 3

## 一句话定义

**Franka Research 3** 是 **Franka Robotics** 面向 **科研与教育** 的 **七轴力控协作臂**：技术文档在 **[frankarobotics.github.io/docs](https://frankarobotics.github.io/docs/)**，开源 SDK 与驱动在 GitHub 组织 **[frankarobotics](https://github.com/frankarobotics)**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 软件开发工具包 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| API | Application Programming Interface | 应用程序编程接口 |
| Manipulation | Robot Manipulation | 抓取、移动、操作物体的任务总称 |
| ROS 2 | Robot Operating System 2 | 机器人系统集成与通信的常用中间件 |

## 为什么重要

- **力控与接触丰富操作基线**：大量 **操作学习 / 遥操作 / Sim2Real** 论文以 Franka 为硬件参照（本仓库多处仿真与 GS 演示亦常出现 Franka 模型）。
- **与 Kinova 对照**：两者都强调 **协作安全**，但 **力矩传感器布局与控制 API** 不同，迁移算法时需分别建模。

## 开源入口（策展摘录）

| 类型 | 链接 |
|------|------|
| 文档中心 | [frankarobotics.github.io/docs](https://frankarobotics.github.io/docs/) |
| GitHub 组织 | [github.com/frankarobotics](https://github.com/frankarobotics) |

## 关联页面

- [Manipulation](../tasks/manipulation.md)
- [Kinova Gen3](./kinova-gen3.md)
- [ROS 2 基础](../concepts/ros2-basics.md)
- [机器人开源宝库（微信策展第02期）索引](../overview/robot-open-source-wechat-issue02-curator.md)

## 推荐继续阅读

- `libfranka` 与 **FCI** 文档（以官方 docs 当前章节为准）

## 参考来源

- [wechat_jixie_robot_open_source_treasury_issue02_10_robots.md](../../sources/blogs/wechat_jixie_robot_open_source_treasury_issue02_10_robots.md)
