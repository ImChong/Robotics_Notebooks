---
type: entity
title: Motrix (MotrixSim / MotrixLab)
tags: [simulation, physics-engine, robot-learning, rust, mjcf]
summary: "Motrix 是高性能机器人物理仿真与训练平台，采用 Rust 开发，深度兼容 MJCF 格式，专注于高频控制与强化学习训练。"
updated: 2026-05-01
---

# Motrix (Motphys 机器人仿真与训练平台)

**Motrix** 是由 Motphys 开发的高性能机器人物理仿真与强化学习训练平台。它由核心仿真引擎 **MotrixSim** 和上层学习框架 **MotrixLab** 组成，旨在为机器人研究与工业应用提供精度高、吞吐量大的动力学环境。

## 核心组件

### 1. MotrixSim (仿真引擎)
- **定位**：工业级高性能多体动力学引擎。
- **技术底座**：使用 **Rust** 语言开发（CPU 版），兼顾内存安全与执行效率。
- **建模方式**：采用 **广义坐标 (Generalized Coordinates)** 系建模，支持关节空间的精确动力学解算，与 MuJoCo 的底层逻辑一致。
- **兼容性**：深度支持 **MJCF** 格式，允许用户无缝迁移原有的 MuJoCo 模型资产。

### 2. MotrixLab (训练平台)
- **功能**：将仿真环境与 AI 训练流程打通的“一站式”平台。
- **集成环境**：内置了针对足式机器人的 `legged_gym` 环境，支持四足与双足人形。
- **算法适配**：支持 SKRL, RSLRL 等主流强化学习框架，并支持 **JAX** 和 **PyTorch** 双后端。

## 为什么选择 Motrix？

- **高性能 CPU 后端**：相比过度依赖 GPU 的 [[isaac-gym-isaac-lab]]，Motrix 的 Rust CPU 后端在 CPU 资源环境下提供了卓越的并行仿真能力，适合需要高确定性或 GPU 资源受限的工业场景。
- **极致的 RL 吞吐量**：针对大规模并行采样进行了优化，缩短了从算法定义到策略收敛的迭代周期。
- **现代化的生态**：使用 `uv` 管理依赖，支持 TensorBoard 实时监控，提供 Pythonic 的 API 交互。

## 与其他系统的关系

- **对比 [[mujoco]]**：MotrixSim 是 MuJoCo 的现代化、高性能替代方案，保持了 MJCF 兼容性，但在并行化和系统稳定性上做了更多工作。
- **对比 [[isaac-gym-isaac-lab]]**：Motrix 提供了更轻量、更灵活的 CPU 并行方案，而非强制绑定特定的 NVIDIA 驱动与硬件。
- **对比 [[genesis-sim]]**：Genesis 更强调多物理场（流体、柔性体），而 Motrix 更专注于刚体关节型机器人的高频控制与 RL 训练。

## 关联页面

- [[references/repos/simulation]] (仿真平台导航)
- [[references/repos/rl-frameworks]] (RL 框架导航)
- [[mujoco]] (底层物理引擎参考)

## 参考来源
- [Motrix 原始资料](../../sources/repos/motphys-motrix.md)
- [MotrixSim 官方文档](https://motphys.github.io/motrixsim-docs/)
- [MotrixLab GitHub](https://github.com/Motphys/MotrixLab)
