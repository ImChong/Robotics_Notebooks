---
type: entity
tags: [humanoid, deployment, ros2, middleware, sim2real, roboparty]
status: complete
updated: 2026-04-25
related:
  - ./roboto-origin.md
  - ./atom01-train.md
  - ./atom01-firmware.md
sources:
  - ../../sources/repos/atom01_deploy.md
summary: "atom01_deploy 是 Atom01 的上机部署仓库，覆盖 ROS2 驱动、中间件接入与真机执行流程。"
---

# Atom01 Deploy

**atom01_deploy** 负责 Atom01 的真机部署链路，连接训练策略与机器人执行系统，是 Sim2Real 落地关键环节。

## 为什么重要

- 打通策略输出到硬件执行器的最后一公里。
- 管理 ROS2 通信、设备接入与运行配置。
- 决定真机稳定性、可调试性与上线效率。

## 核心结构/机制

- **驱动与中间件**：ROS2 节点、消息流与设备桥接。
- **系统集成**：IMU/电机等关键设备配置。
- **运行部署**：启动流程、参数管理与环境依赖。

## 常见误区或局限

- 误区：部署问题只是“脚本没写好”。很多问题来自时序、同步、硬件状态与网络抖动。
- 局限：部署仓库不直接改进策略能力，但会显著影响策略实用性。

## 参考来源

- [sources/repos/atom01_deploy.md](../../sources/repos/atom01_deploy.md)
- [Roboparty/atom01_deploy](https://github.com/Roboparty/atom01_deploy)

## 关联页面

- [Roboto Origin（开源人形机器人基线）](./roboto-origin.md)
- [Atom01 Train](./atom01-train.md)
- [Atom01 Firmware](./atom01-firmware.md)

## 推荐继续阅读

- [Sim2Real](../concepts/sim2real.md)
- [sources/sim2real.md](../../sources/sim2real.md)
