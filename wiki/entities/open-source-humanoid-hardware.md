---
type: entity
tags: [humanoid, hardware, open-source, robotics, research]
status: complete
updated: 2026-04-21
related:
  - ./humanoid-robot.md
  - ../queries/humanoid-hardware-selection.md
  - ../roadmaps/humanoid-control-roadmap.md
sources:
  - ../../sources/papers/humanoid_hardware.md
summary: "主流开源人形机器人硬件方案对比：详细梳理了 Roboto Origin、Berkeley Humanoid (ODRI) 等平台的机械结构、执行器选型及开源生态，为研究者提供低成本入门指南。"
---

# 开源人形机器人硬件方案对比

随着具身智能的爆发，人形机器人的硬件门槛正在迅速降低。对于预算有限的实验室或个人研究者，**开源硬件方案 (Open-source Humanoid Hardware)** 是验证算法的首选。

## 核心方案对比

| 方案名称 | 主导机构 | 自由度 (DOF) | 核心执行器 | 成本估算 | 软件生态 |
|------|------|-----|-----------|---------|---------|
| **Berkeley Humanoid** | UC Berkeley | 12-14 | 准直接驱动 (QDD) | < 5,000 USD | 基于 Python/C++ 的简易控制框架 |
| **Roboto Origin** | Roboparty | 20+ | QDD + 舵机混合 | < 3,000 USD | ROS2 支持，兼容简单平衡算法 |
| **ODRI (Bolt/Solo)** | Max Planck | 6-12 (双腿/四足) | 基于 T-Motor 改造 | 中等 | OCS2 / Pinocchio 深度支持 |
| **Unitree H1 (SDK版)** | Unitree | 19 | 商业级 QDD | > 50,000 USD | Isaac Gym (legged_gym) 生态极强 |

## 1. Berkeley Humanoid (准直接驱动派)
- **特点**：极其强调低成本和维修便捷性。它证明了使用廉价的无刷电机和 3D 打印结构，也能完成稳定的动态行走。
- **优点**：动力学透明度高，非常适合做强化学习的 Sim2Real 验证。

## 2. Roboto Origin (科普与原型派)
- **特点**：由国内开源社区驱动，旨在打造人人都能拥有的“第一台人形机器人”。
- **优点**：文档详尽，组装门槛低。

## 3. ODRI 架构 (学术严谨派)
- **特点**：Open Dynamic Robot Initiative。虽然其人形版本较少，但其开源的执行器模块（Actuator Modules）被广泛借鉴。
- **优点**：力控精度极高，代码质量达工业级。

## 选型建议

- **如果你想验证 RL 算法**：首选 **Berkeley Humanoid** 类方案，因为其 QDD 电机的动力学建模最为简单。
- **如果你想研究全身协调 (WBC)**：建议寻找支持更高自由度的平台，或者在仿真中使用 **ODRI** 模型进行先行验证。

## 关联页面
- [人形机器人 (Humanoid Robot)](./humanoid-robot.md)
- [人形机器人硬件怎么选](../queries/humanoid-hardware-selection.md)
- [Humanoid Control Roadmap](../roadmaps/humanoid-control-roadmap.md)

## 参考来源
- [humanoid_hardware.md](../../sources/papers/humanoid_hardware.md)
- 各开源项目 GitHub Readme 与 Wiki。
