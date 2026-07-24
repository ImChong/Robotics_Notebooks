---
type: entity
tags: [repo, unitree, unitreerobotics, platform]
status: draft
updated: 2026-07-24
related:
  - ./unitree.md
  - ./unitree-rl-gym.md
  - ./quadruped-robot.md
sources:
  - ../../sources/repos/Qmini.md
  - ../../sources/repos/unitree.md
summary: "宇树相关小型四足开源项目（社区热度高）。"
---

# Qmini

**Qmini** 是 [unitreerobotics](https://github.com/unitreerobotics) 组织下的官方仓库，归属 **开源平台/整机** 主线。

## 一句话定义

宇树相关小型四足开源项目（社区热度高）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| Sim2Real | Simulation to Real | 仿真到真机迁移 |
| SDK | Software Development Kit | 软件开发工具包 |
| URDF | Unified Robot Description Format | 统一机器人描述格式 |
| MuJoCo | Multi-Joint dynamics with Contact | 接触丰富的刚体物理引擎 |

## 为什么重要

整机/小型开源平台提供可复现的硬件+软件对照样本。

在宇树官方开源地图中，本仓是 **开源平台/整机** 的独立节点；与其它仓的选型关系见 [Unitree](./unitree.md)。

## 核心信息

| 字段 | 内容 |
|------|------|
| 仓库 | [`unitreerobotics/Qmini`](https://github.com/unitreerobotics/Qmini) |
| 组织分类 | 开源平台/整机 |
| 星标（2026-07-24） | ~722 |
| 最近推送 | 2025-09-17 |
| 主要语言 | N/A |

## 工程实践

- Complete Bill of Materials (BOM)
- Electrical system block diagram
- DIY instructions
- STEP files for all mechanical components

- 组织级导航与五条研发主线见 [Unitree 软件生态](./unitree.md)；原始归档见 [sources/repos/Qmini.md](../../sources/repos/Qmini.md)。

## 局限与风险

- **不要脱离代际混用**：SDK1/ROS1 遗产仓与 SDK2/ROS2/DDS 新栈的消息与启动方式不兼容。
- **README 边界优先**：功能承诺以官方 README 为准；星标高不等于适合你的机型/仿真器。
- **外设/第三方手部仓**：驱动与固件版本需与具体灵巧手/雷达硬件对齐，否则 Serial↔DDS 桥会静默失败。

## 关联页面

- [Unitree 组织枢纽](./unitree.md)
- [unitree_rl_gym](./unitree-rl-gym.md)
- [四足机器人](./quadruped-robot.md)

## 参考来源

- [sources/repos/Qmini.md](../../sources/repos/Qmini.md)
- [sources/repos/unitree.md](../../sources/repos/unitree.md) — 组织级仓库地图
- 上游仓库：<https://github.com/unitreerobotics/Qmini>

## 推荐继续阅读

- 官方开发者文档：<https://support.unitree.com/home/zh/developer>
- 组织总览：<https://github.com/unitreerobotics>
