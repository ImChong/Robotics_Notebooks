---
type: entity
tags: [manipulator, cobot, open-source, kinova, medical-robotics, ros]
status: complete
updated: 2026-11-15
related:
  - ../tasks/manipulation.md
  - ../overview/robot-open-source-wechat-issue02-curator.md
  - ../concepts/ros2-basics.md
sources:
  - ../../sources/repos/open-robot-hardware-entry-review-2026-11-15.md
  - ../../sources/blogs/wechat_jixie_robot_open_source_treasury_issue02_10_robots.md
summary: "Kinova Gen3：加拿大 Kinova 轻型协作臂系列；具体本体、自由度与夹爪配置须对照官方型号，ROS 接口按仓库和发行版选择。"
---

# Kinova Gen3

## 一句话定义

**Kinova Gen3** 是 **Kinova** 公司的 **轻型协作机械臂系列**：面向 **医疗、服务与科研**；产品介绍在 **[Gen3 产品页](https://www.kinovarobotics.com/product/gen3-robots)**，开源 SDK / ROS 接口等在 GitHub 组织 **[Kinovarobotics](https://github.com/Kinovarobotics)**。具体自由度与末端配置须核对型号，不能把 Gen3 与 Gen3 lite 混成单一六轴描述。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 软件开发工具包 |
| ROS 2 | Robot Operating System 2 | 机器人系统集成与通信的常用中间件 |
| Manipulation | Robot Manipulation | 抓取、移动、操作物体的任务总称 |

## 为什么重要

- **协作安全与力控**：作为 **实验室常见工业臂对照组**，与 **Franka**、**UR** 等并列出现在仿真与真机管线讨论中。
- **ROS 生态成熟**：组织内多仓拆分 **驱动 / 示例 / Gen3 专用接口**，集成前需核对 **ROS1 vs ROS2** 支持矩阵。

## 开源入口（策展摘录）

| 类型 | 链接 |
|------|------|
| 产品页 | [kinovarobotics.com/product/gen3-robots](https://www.kinovarobotics.com/product/gen3-robots) |
| GitHub 组织 | [github.com/Kinovarobotics](https://github.com/Kinovarobotics) |

## 本体、夹爪与 ROS 分支

[官方 `ros_kortex` 文档复核](../../sources/repos/open-robot-hardware-entry-review-2026-11-15.md)区分 `gen3` / `gen3_lite` 与对应夹爪 xacro。所读 `noetic-devel` 是 ROS1 分支；应选与 ROS 发行版、本体及夹爪匹配的配置，不能由该分支的 Gazebo/MoveIt 示例推断 ROS2 或实机控制器均已支持。

## 关联页面

- [Manipulation](../tasks/manipulation.md)
- [ROS 2 基础](../concepts/ros2-basics.md)
- [机器人开源宝库（微信策展第02期）索引](../overview/robot-open-source-wechat-issue02-curator.md)

## 推荐继续阅读

- Kinova 官方文档门户（以产品页导航为准）

## 参考来源

- [官方硬件文档与入口复核](../../sources/repos/open-robot-hardware-entry-review-2026-11-15.md)

- [wechat_jixie_robot_open_source_treasury_issue02_10_robots.md](../../sources/blogs/wechat_jixie_robot_open_source_treasury_issue02_10_robots.md)
