---
type: entity
tags: [quadruped, open-source, education, ros2, sim2real, raspberry-pi]
status: complete
updated: 2026-10-27
project_id: mini-pupper
code: https://github.com/mangdangroboticsclub/mini_pupper
summary: "Mini Pupper：MangDang 的小型开源四足，自研 3-DoF coreless 智能舵机 + Raspberry Pi，支持 ROS/ROS2、OpenCV、OAK-D 与 MuJoCo/Isaac RL，Apache-2.0。"
related:
  - ./quadruped-robot.md
  - ./stanford-doggo-and-pupper.md
  - ./sim2real.md
  - ../concepts/ros2-basics.md
  - ../overview/robot-open-source-wechat-githubstore-18-projects.md
sources:
  - ../../sources/blogs/wechat_githubstore_18_open_source_robot_projects.md
---

# Mini Pupper

## 一句话定义

**Mini Pupper** 是 MangDang 推出的小型开源四足机器人，仓库 [mangdangroboticsclub/mini_pupper](https://github.com/mangdangroboticsclub/mini_pupper)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BSP | Board Support Package | 板级支持包 |
| ROS | Robot Operating System | 机器人中间件 |
| RL | Reinforcement Learning | 强化学习 |
| DIY | Do It Yourself | 自己动手 |

## 为什么重要

- 价格门槛低于大型四足，又覆盖 ROS 与 RL 两条学习线。
- 在微信文章《想进入机器人领域吗？学习这18个开源项目就够了》中排第 **6** 位。

## 核心要点（据文章）

- 自研 3-DoF coreless 智能舵机；基于 Raspberry Pi，支持 ROS/ROS2、OpenCV、OAK-D 相机和强化学习（MuJoCo / NVIDIA Isaac 环境）。
- 强调 sim-to-real，有完整 BSP、文档和社区（Discord/Facebook）；许可证 Apache-2.0。
- 有成品套件出售，也允许 DIY。

## 工程实践

- 适合学习 ROS 与物理 AI：先跑 BSP 与 ROS2 示例，再接 MuJoCo/Isaac 训练 RL 步态。
- 以上规格与许可证均出自文章叙述，**未逐项核对官方仓库**；动手前请以官方 README/BOM 为准。

## 局限与风险

- 本页仅基于一篇第三方公众号文章，规格数字可能随版本变化。
- 仓库迁移、分叉和许可证以官方当前状态为准。

## 关联页面

- [quadruped-robot](./quadruped-robot.md)
- [stanford-doggo-and-pupper](./stanford-doggo-and-pupper.md)
- [sim2real](./sim2real.md)
- [ros2-basics](../concepts/ros2-basics.md)
- [机器人开源项目 18 例（微信 GitHubStore）索引](../overview/robot-open-source-wechat-githubstore-18-projects.md)

## 推荐继续阅读

- [mini_pupper 仓库](https://github.com/mangdangroboticsclub/mini_pupper)

## 参考来源

- [wechat_githubstore_18_open_source_robot_projects.md](../../sources/blogs/wechat_githubstore_18_open_source_robot_projects.md)
