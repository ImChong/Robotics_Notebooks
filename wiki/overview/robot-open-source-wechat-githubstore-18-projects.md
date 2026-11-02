---
type: overview
tags: [open-source, hardware, humanoid, quadruped, manipulator, mobile-robot, wechat-curator, education]
status: complete
updated: 2026-10-27
related:
  - ./open-source-buildable-humanoids-19-technology-map.md
  - ./robot-open-source-wechat-issue01-curator.md
  - ./robot-open-source-wechat-issue02-curator.md
  - ../entities/open-source-humanoid-hardware.md
  - ../entities/humanoid-robot.md
  - ../entities/quadruped-robot.md
  - ../entities/microban-humanoid.md
  - ../entities/miniloong-humanoid.md
  - ../entities/poppy-project-robots.md
  - ../entities/otto-diy-biped.md
  - ../entities/pib-printable-intelligent-bot.md
  - ../entities/mini-pupper.md
  - ../entities/petoi-bittle-opencat.md
  - ../entities/stanford-doggo-and-pupper.md
  - ../entities/spotmicro.md
  - ../entities/open-quadruped.md
  - ../entities/puguette-quadruped.md
  - ../entities/painode-170-soarm100.md
  - ../entities/waveshare-roarm-m2.md
  - ../entities/robotis-open-manipulator-line.md
  - ../entities/lekiwi-mobile-manipulator.md
  - ../entities/turtlebot3.md
  - ../entities/paper-openbot-smartphone-mobile-robot.md
  - ../entities/duckietown-duckiebot.md
sources:
  - ../../sources/blogs/wechat_githubstore_18_open_source_robot_projects.md
summary: "微信公众号 GitHubStore《想进入机器人领域吗？学习这18个开源项目就够了》：人形、四足、机械臂、移动平台 18 个开源项目的独立实体索引（12 个新节点 + 6 个既有节点复用）。"
---

# 机器人开源项目 18 例（微信 GitHubStore）— 索引

## 一句话定义

按原文编号 1–18 列出入门向开源机器人项目，**每个项目只对应一个实体节点**；已有节点（Poppy、Stanford Doggo、SO-ARM100、OpenMANIPULATOR、TurtleBot3、OpenBot）复用，不重复建页。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ROS 2 | Robot Operating System 2 | 机器人系统集成与通信的常用中间件 |
| DoF | Degrees of Freedom | 自由度 |
| BOM | Bill of Materials | 物料清单 |
| LeRobot | Hugging Face LeRobot | 开源机器人学习框架 |

## 18 项一览

| # | 实体页 | 项目 | 一句话 |
|---|--------|------|--------|
| 1 | [Microban](../entities/microban-humanoid.md) | 30 cm 人形，<$600，MuJoCo RL |
| 2 | [MiniLoong](../entities/miniloong-humanoid.md) | 55 cm / 22 DoF，OpenLoong 生态 |
| 3 | [Poppy Humanoid](../entities/poppy-project-robots.md) | Inria 教育人形（既有节点） |
| 4 | [Otto DIY](../entities/otto-diy-biped.md) | Arduino 低成本双足 |
| 5 | [pib](../entities/pib-printable-intelligent-bot.md) | 全 3D 打印人形，pib#5 |
| 6 | [Mini Pupper](../entities/mini-pupper.md) | 小型四足，ROS2 + RL |
| 7 | [Petoi Bittle / OpenCat](../entities/petoi-bittle-opencat.md) | 开源软件 + 商业硬件 |
| 8 | [Stanford Doggo](../entities/stanford-doggo-and-pupper.md) | 准直驱高动态四足（既有节点） |
| 9 | [SpotMicro](../entities/spotmicro.md) | 社区四足，主仓库迁 GitLab |
| 10 | [OpenQuadruped](../entities/open-quadruped.md) | 12 DoF，Bezier 步态 + IK |
| 11 | [Puguette](../entities/puguette-quadruped.md) | BLDC 55 Nm 可打印执行器 |
| 12 | [SO-ARM100 / SO-101](../entities/painode-170-soarm100.md) | LeRobot 低成本机械臂（既有节点） |
| 13 | [RoArm-M2](../entities/waveshare-roarm-m2.md) | 桌面 4 DoF 臂 |
| 14 | [OpenMANIPULATOR-X](../entities/robotis-open-manipulator-line.md) | ROBOTIS 5 DoF 臂（既有节点） |
| 15 | [LeKiwi](../entities/lekiwi-mobile-manipulator.md) | Kiwi 底盘 + SO-ARM |
| 16 | [TurtleBot3 Burger](../entities/turtlebot3.md) | ROS 教育移动平台（既有节点） |
| 17 | [OpenBot](../entities/paper-openbot-smartphone-mobile-robot.md) | 手机作大脑的 $50 小车（既有节点） |
| 18 | [Duckiebot](../entities/duckietown-duckiebot.md) | 自动驾驶教学小车 |

## 选型提示（据原文，未核对官方）

- 人形入门：Microban（<$600、有 RL）→ MiniLoong（VLA/IL）→ pib（教育）。
- 四足：Mini Pupper（ROS/RL）、OpenQuadruped（经典控制）、Puguette（高扭矩执行器）；Bittle 硬件非开源。
- 机械臂/移动操作：SO-ARM100 → LeKiwi；OpenMANIPULATOR-X 偏 ROS/MoveIt 教学。

## 关联页面
- [能自己造的开源人形机器人 19 款（同账号后续一篇）](./open-source-buildable-humanoids-19-technology-map.md)

- [机器人开源宝库（微信策展第01期）](./robot-open-source-wechat-issue01-curator.md)
- [机器人开源宝库（微信策展第02期）](./robot-open-source-wechat-issue02-curator.md)
- [开源人形硬件方案对比](../entities/open-source-humanoid-hardware.md)

## 推荐继续阅读

- [原文（微信）](https://mp.weixin.qq.com/s/DVGSnatQ5hw2UxAkPeS1Wg)

## 参考来源

- [wechat_githubstore_18_open_source_robot_projects.md](../../sources/blogs/wechat_githubstore_18_open_source_robot_projects.md)
