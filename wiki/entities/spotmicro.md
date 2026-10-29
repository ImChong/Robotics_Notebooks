---
type: entity
tags: [quadruped, open-source, community, 3d-printing, raspberry-pi]
status: complete
updated: 2026-10-27
project_id: spotmicro
code: https://gitlab.com/custom_robots/spotmicroai
summary: "SpotMicro：社区开源四足，外观参考 Boston Dynamics Spot Mini，12 舵机 + 3D 打印 + Raspberry Pi；主仓库已迁至 GitLab，GitHub 有旧仓库与大量个人实现。"
related:
  - ./quadruped-robot.md
  - ./open-quadruped.md
  - ./mini-pupper.md
  - ../overview/robot-open-source-wechat-githubstore-18-projects.md
sources:
  - ../../sources/blogs/wechat_githubstore_18_open_source_robot_projects.md
---

# SpotMicro

## 一句话定义

**SpotMicro** 是社区开源四足项目，设计基于 Boston Dynamics Spot Mini 的外观，原始 STL 由 KDY0523 在 Thingiverse 发布；主仓库 [GitLab custom_robots/spotmicroai](https://gitlab.com/custom_robots/spotmicroai)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| STL | Stereolithography file | 3D 打印模型格式 |
| ROS | Robot Operating System | 机器人中间件 |
| RL | Reinforcement Learning | 强化学习 |
| ESP32 | Espressif ESP32 | 低成本 MCU |

## 为什么重要

- 社区驱动的低成本四足入门项目，同时是「分叉多、需甄别」的典型案例。
- 在微信文章《想进入机器人领域吗？学习这18个开源项目就够了》中排第 **9** 位。

## 核心要点（据文章）

- 通常采用 12 个舵机，3D 打印结构，Raspberry Pi 控制。
- 主开发仓库已迁移至 GitLab；GitHub 保留旧仓库 [FlorianWilk/SpotMicroAI](https://github.com/FlorianWilk/SpotMicroAI) 和大量个人实现（如 [SpotMicroESP32-Leika](https://github.com/runeharlyk/SpotMicroESP32-Leika)，ROS/ESP32/Teensy 等版本）。
- 有仿真、Bezier 步态和强化学习相关工作。

## 工程实践

- 选实现时先确认维护状态：版本分散，建议从 GitLab 主仓库入手再挑个人分支。
- 以上规格与许可证均出自文章叙述，**未逐项核对官方仓库**；动手前请以官方 README/BOM 为准。

## 局限与风险

- 本页仅基于一篇第三方公众号文章，规格数字可能随版本变化。
- 仓库迁移、分叉和许可证以官方当前状态为准。

## 关联页面

- [quadruped-robot](./quadruped-robot.md)
- [open-quadruped](./open-quadruped.md)
- [mini-pupper](./mini-pupper.md)
- [机器人开源项目 18 例（微信 GitHubStore）索引](../overview/robot-open-source-wechat-githubstore-18-projects.md)

## 推荐继续阅读

- [SpotMicroAI GitLab](https://gitlab.com/custom_robots/spotmicroai)

## 参考来源

- [wechat_githubstore_18_open_source_robot_projects.md](../../sources/blogs/wechat_githubstore_18_open_source_robot_projects.md)
