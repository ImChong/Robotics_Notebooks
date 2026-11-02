---
type: entity
project_id: mangdang-quadruped-robot
tags:
  - repo
  - independent
  - quadruped
  - ros2
  - mini-pupper
  - sim2real
status: complete
updated: 2026-11-01
code: https://github.com/mangdangroboticsclub/QuadrupedRobot
related:
  - ../entities/quadruped-robot.md
  - ../entities/mini-pupper.md
  - ../concepts/sim2real.md
  - ../overview/quadruped-open-source-projects-technology-map.md
sources:
  - ../../sources/repos/mangdang_quadruped_robot.md
  - ../../sources/blogs/wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md
summary: "mangdangroboticsclub/QuadrupedRobot 是 MangDang 的 Mini Pupper 系列软件仓库，README 当前主打 Mini Pupper 4：3-DoF coreless 智能舵机、ROS 2 Jazzy + micro-ROS、SLAM 与 Nav2。"
---

# MangDang QuadrupedRobot

**MangDang QuadrupedRobot**（[GitHub](https://github.com/mangdangroboticsclub/QuadrupedRobot)）来自 [古月居转载的四足机器人开源项目综述](https://mp.weixin.qq.com/s/pRGoPm41qYCNyHPWR6Dl6A)，属「ROS 生态」组。

## 一句话定义

**mangdangroboticsclub/QuadrupedRobot 是 MangDang 的 Mini Pupper 系列软件仓库，README 当前主打 Mini Pupper 4：3-DoF coreless 智能舵机、ROS 2 Jazzy + micro-ROS、SLAM 与 Nav2。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BAM | Better Actuator Models | 更好的执行器模型 |
| ROS 2 | Robot Operating System 2 | 第二代机器人中间件 |

## 为什么重要

- 综述把它放在「ROS 生态」这条技术路线下，作为可读源码的参照项目；横向对比见 [四足机器人开源项目技术地图：从极简运动学到研究级硬件（36 个 GitHub 项目）](../overview/quadruped-open-source-projects-technology-map.md)。
- 文中星标：1088（文章表述，入库日未复核）。

## 核心信息

| 项 | 内容 |
|----|------|
| **仓库** | [mangdangroboticsclub/QuadrupedRobot](https://github.com/mangdangroboticsclub/QuadrupedRobot) |
| **最近提交** | 2026-10-29 |
| **可达性** | `git ls-remote` 可达（2026-11-01） |
| **文章归类** | ROS 生态 |

## 文章说法核对

- **文章：** 文章只在附录清单列出（1088 星），未展开描述。
- **README / 核查：** README 称支持在仿真中训练神经网络策略并部署到实机（Sim2Real）、与 BAM（Better Actuator Models）兼容以刻画关节摩擦，并提供 ROS 2 Jazzy 与 micro-ROS 模板。
- **差异与保留：** 与库内 `mini-pupper`（mini_pupper 仓库）是同一厂商的不同仓库，故分开建节点。

## 关联页面

- [quadruped-robot](../entities/quadruped-robot.md)
- [mini-pupper](../entities/mini-pupper.md)
- [sim2real](../concepts/sim2real.md)
- [quadruped-open-source-projects-technology-map](../overview/quadruped-open-source-projects-technology-map.md)

## 参考来源

- [mangdang_quadruped_robot.md](../../sources/repos/mangdang_quadruped_robot.md)
- [wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md](../../sources/blogs/wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md)
- [原文（微信公众号）](https://mp.weixin.qq.com/s/pRGoPm41qYCNyHPWR6Dl6A)
