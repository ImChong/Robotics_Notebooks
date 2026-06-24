---

type: entity
title: Motrix (MotrixSim / MotrixLab)
tags: [simulation, physics-engine, robot-learning, rust, mjcf, web-viewer, motphys]
summary: "Motrix 是高性能机器人物理仿真与训练平台，采用 Rust 开发，深度兼容 MJCF 格式，并提供浏览器 Web Viewer 零安装验模。"
updated: 2026-06-18
related:
  - ./botworld.md
---

# Motrix (Motphys 机器人仿真与训练平台)

**Motrix** 是由 Motphys 开发的高性能机器人物理仿真与强化学习训练平台。它由核心仿真引擎 **MotrixSim** 和上层学习框架 **MotrixLab** 组成，旨在为机器人研究与工业应用提供精度高、吞吐量大的动力学环境；并通过 **MotrixSim Web Viewer** 在浏览器中零安装加载与交互仿真。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MJCF | MuJoCo XML Format | MuJoCo 的模型与场景描述格式 |
| URDF | Unified Robot Description Format | 机器人连杆与关节的 XML 描述格式 |
| WASM | WebAssembly | 浏览器内运行原生级代码的虚拟机格式 |
| CPU | Central Processing Unit | 中央处理器 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| GPU | Graphics Processing Unit | 图形处理器，大规模并行仿真训练的算力基础 |

## 核心组件

### 1. MotrixSim (仿真引擎)
- **定位**：工业级高性能多体动力学引擎。
- **技术底座**：使用 **Rust** 语言开发（CPU 版），兼顾内存安全与执行效率。
- **建模方式**：采用 **广义坐标 (Generalized Coordinates)** 系建模，支持关节空间的精确动力学解算，与 MuJoCo 的底层逻辑一致。
- **兼容性**：深度支持 **MJCF** 格式，并可在 Web Viewer 中加载 **URDF**、**JSON** 等场景描述；允许用户无缝迁移原有的 MuJoCo 模型资产。

### 2. MotrixLab (训练平台)
- **功能**：将仿真环境与 AI 训练流程打通的“一站式”平台。
- **集成环境**：内置了针对足式机器人的 `legged_gym` 环境，支持四足与双足人形。
- **算法适配**：支持 SKRL, RSLRL 等主流强化学习框架，并支持 **JAX** 和 **PyTorch** 双后端。

### 3. MotrixSim Web Viewer (浏览器端)
- **入口**：[Motrix Viewer](https://motrix.motphys.com/) — 现代浏览器 + **WebAssembly** 即可运行，无需本地安装；亦作为 [BotWorld](./botworld.md) 插件中心推荐入口，便于从资产广场跳转验模。
- **加载方式**：Web 端无法像桌面版直接读本地盘；应**将整个模型文件夹**（含 `scene.xml`、meshes、textures 等相对路径依赖）拖入页面，再在左侧 `Customize` 文件树中点击场景文件。
- **文件来源**：
  - **Online**：站点内置只读示例，加载 manifest 后自动列出。
  - **Customize**：当前会话中由用户拖入的本地资产，刷新后消失。
- **交互**：顶部栏 Play / Pause / Next、Reset、Reload；支持轨道相机、单步（F10）、重置（Ctrl+E）、重载（Ctrl+R）及可选的 **Ctrl+左键物理拖拽**。
- **定位**：快速验模、教学演示与对外分享；[UniLab](unilab.md) 等叙事中的浏览器 MotrixSim 策略试玩与此同属 Web 侧能力。大规模 RL 批量训练仍应使用 MotrixLab / 桌面 MotrixSim。

## 为什么选择 Motrix？

- **高性能 CPU 后端**：相比过度依赖 GPU 的 [isaac-gym-isaac-lab](isaac-gym-isaac-lab.md)，Motrix 的 Rust CPU 后端在 CPU 资源环境下提供了卓越的并行仿真能力，适合需要高确定性或 GPU 资源受限的工业场景。
- **零安装 Web 验模**：Web Viewer 降低协作与展示门槛，适合在分享 MJCF/URDF 资产前做快速动力学检视。
- **极致的 RL 吞吐量**：针对大规模并行采样进行了优化，缩短了从算法定义到策略收敛的迭代周期。
- **现代化的生态**：使用 `uv` 管理依赖，支持 TensorBoard 实时监控，提供 Pythonic 的 API 交互。

## 与其他系统的关系

- **对比 [mujoco](mujoco.md)**：MotrixSim 是 MuJoCo 的现代化、高性能替代方案，保持了 MJCF 兼容性，但在并行化和系统稳定性上做了更多工作。
- **对比 [isaac-gym-isaac-lab](isaac-gym-isaac-lab.md)**：Motrix 提供了更轻量、更灵活的 CPU 并行方案，而非强制绑定特定的 NVIDIA 驱动与硬件。
- **与 [UniLab](unilab.md)**：UniLab 将 **MotrixSim** 与 MuJoCoUni 作为可选 CPU 批量物理后端，经统一 runtime 对接 GPU learner（见论文 arXiv:2605.30313）；项目页亦链到浏览器 MotrixSim demo。
- **对比 [genesis-sim](genesis-sim.md)**：Genesis 更强调多物理场（流体、柔性体），而 Motrix 更专注于刚体关节型机器人的高频控制与 RL 训练。

## 关联页面

- [UniLab](unilab.md) — 异构 CPU-sim / GPU-learn 训练系统（MotrixSim 后端）
- [simulation](../../references/repos/simulation.md) (仿真平台导航)
- [rl-frameworks](../../references/repos/rl-frameworks.md) (RL 框架导航)
- [mujoco](mujoco.md) (底层物理引擎参考)
- [BotWorld](./botworld.md) — 机器人资产广场；插件中心挂载 Motrix Viewer

## 推荐继续阅读

- [MotrixSim Web Viewer 用户指南](https://motrixsim.readthedocs.io/en/latest/user_guide/getting_started/motrixsim_web.html) — 拖文件夹加载、快捷键与工具栏说明
- [MotrixSim 官方文档](https://motphys.github.io/motrixsim-docs/) — 安装与 API
- [MotrixLab GitHub](https://github.com/Motphys/MotrixLab) — RL 训练栈

## 参考来源
- [Motrix 原始资料](../../sources/repos/motphys-motrix.md)
- [MotrixSim Web Viewer 用户指南](../../sources/sites/motrixsim-web-viewer.md)
- [MotrixSim 官方文档](https://motphys.github.io/motrixsim-docs/)
- [MotrixLab GitHub](https://github.com/Motphys/MotrixLab)
