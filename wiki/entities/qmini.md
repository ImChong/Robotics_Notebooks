---
type: entity
tags: [repo, unitree, unitreerobotics, quadruped, open-hardware, education]
status: complete
updated: 2026-07-24
related:
  - ./unitree.md
  - ./quadruped-robot.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/repos/Qmini.md
  - ../../sources/repos/unitree.md
summary: "Qmini 是宇树相关的小型开源四足平台：BOM、STEP、URDF 与 DIY 文档齐全，默认参考树莓派 4B，执行器使用 Unitree 8010 电机；软件核心另见 RoboTamer4Qmini。"
---

# Qmini

**Qmini** 面向爱好者、教育与科研的小型四足开源项目：强调可 3D 打印结构、一站式零件清单与模块化扩展。

## 一句话定义

「乐高式」可组装开源小四足——硬件 BOM + 机械 STEP + URDF，配合社区/配套软件快速上手动态步态实验。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BOM | Bill of Materials | 物料清单 |
| STEP | STEP CAD format | 机械零件格式 |
| URDF | Unified Robot Description Format | 机器人描述 |
| DIY | Do It Yourself | 自行组装 |
| STEM | Science Technology Engineering Math | 教育场景 |
| RL | Reinforcement Learning | 可选算法层 |

## 为什么重要

- 降低「买整机才能做四足实验」的门槛；适合课程与原型。
- 使用与商业平台同源的 **Unitree 8010** 电机体验，利于迁移到更大机型。
- 在组织仓中社区热度高（星标显著），是硬件开源叙事的代表节点。

## 核心原理

| 开源面 | 内容 |
|--------|------|
| 硬件 | 完整 BOM、电气框图、DIY PDF |
| 机械 | 全零件 STEP、装配 SOP；结构可 3D 打印 |
| 软件 | `urdf/Qmini.urdf`；核心栈指向 [RoboTamer4Qmini](https://github.com/vsislab/RoboTamer4Qmini) |

**驱动**：11 个 8010 电机——10 个主运动，1 个颈部预留扩展。控制板默认参考 **Raspberry Pi 4 Model B**，可替换。

## 工程实践

1. 按 BOM 采购/打印零件；组装约 3–5 小时量级（上游表述）。
2. 加载 URDF 做仿真或可视化。
3. 软件跟 `RoboTamer4Qmini` 文档，而不是假设本仓自含完整 RL 训练。

> 注：`unitree_actuator_sdk` 等电机调试工具见 sources 归档；不另建重复 wiki 页，需要时从 [unitree_sdk2](./unitree-sdk2.md) 周边说明进入。

## 局限与风险

- 本仓偏硬件与入门软件入口，**不等于**官方 Go2 RL 部署栈。
- 3D 打印公差与电机标定会影响落地步态，需独立做系统辨识。
- 配套软件在外部仓库，版本需自行钉扎。

## 关联页面

- [四足机器人](./quadruped-robot.md)
- [Locomotion](../tasks/locomotion.md)
- [Unitree](./unitree.md)
- [unitree_sdk2](./unitree-sdk2.md)

## 参考来源

- [sources/repos/Qmini.md](../../sources/repos/Qmini.md)
- 上游：<https://github.com/unitreerobotics/Qmini>

## 推荐继续阅读

- RoboTamer4Qmini：<https://github.com/vsislab/RoboTamer4Qmini>

