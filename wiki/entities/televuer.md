---
type: entity
tags: [repo, unitree, unitreerobotics, teleop]
status: draft
updated: 2026-07-24
related:
  - ./unitree.md
  - ./xr-teleoperate.md
  - ./unitree-sim-isaaclab.md
  - ./unitree-lerobot.md
  - ../tasks/teleoperation.md
  - ./unitree-g1.md
sources:
  - ../../sources/repos/televuer.md
  - ../../sources/repos/unitree.md
summary: "XR 视觉与手部/手柄遥操作接口层，服务官方遥操作管线。"
---

# televuer

**televuer** 是 [unitreerobotics](https://github.com/unitreerobotics) 组织下的官方仓库，归属 **遥操作与采数** 主线。

## 一句话定义

XR 视觉与手部/手柄遥操作接口层，服务官方遥操作管线。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| XR | Extended Reality | 扩展现实（VR/AR） |
| DDS | Data Distribution Service | 分布式实时通信中间件 |
| G1 | Unitree G1 Humanoid | 宇树入门级人形平台 |
| IL | Imitation Learning | 模仿学习 |
| Sim2Real | Simulation to Real | 仿真到真机迁移 |

## 为什么重要

人形数据闭环的入口；遥操作质量直接决定后续 IL/VLA 数据上限。

在宇树官方开源地图中，本仓是 **遥操作与采数** 的独立节点；与其它仓的选型关系见 [Unitree](./unitree.md)。

## 核心信息

| 字段 | 内容 |
|------|------|
| 仓库 | [`unitreerobotics/televuer`](https://github.com/unitreerobotics/televuer) |
| 组织分类 | 遥操作与采数 |
| 星标（2026-07-24） | ~46 |
| 最近推送 | 2026-05-27 |
| 主要语言 | Python |

## 工程实践

- immersive: fully immersive mode; VR shows the robot's first-person view (zmq or webrtc must be enabled).
- pass-through: VR shows the real world through the VR headset cameras; no image from zmq or webrtc is displayed (even if enabled).
- ego: a small window in the center shows the robot's first-person view, while the surrounding area shows the real world.

- 组织级导航与五条研发主线见 [Unitree 软件生态](./unitree.md)；原始归档见 [sources/repos/televuer.md](../../sources/repos/televuer.md)。

## 局限与风险

- **不要脱离代际混用**：SDK1/ROS1 遗产仓与 SDK2/ROS2/DDS 新栈的消息与启动方式不兼容。
- **README 边界优先**：功能承诺以官方 README 为准；星标高不等于适合你的机型/仿真器。
- **外设/第三方手部仓**：驱动与固件版本需与具体灵巧手/雷达硬件对齐，否则 Serial↔DDS 桥会静默失败。

## 关联页面

- [Unitree 组织枢纽](./unitree.md)
- [xr_teleoperate](./xr-teleoperate.md)
- [unitree_sim_isaaclab](./unitree-sim-isaaclab.md)
- [unitree_lerobot](./unitree-lerobot.md)
- [Teleoperation](../tasks/teleoperation.md)
- [Unitree G1](./unitree-g1.md)

## 参考来源

- [sources/repos/televuer.md](../../sources/repos/televuer.md)
- [sources/repos/unitree.md](../../sources/repos/unitree.md) — 组织级仓库地图
- 上游仓库：<https://github.com/unitreerobotics/televuer>

## 推荐继续阅读

- 官方开发者文档：<https://support.unitree.com/home/zh/developer>
- 组织总览：<https://github.com/unitreerobotics>
