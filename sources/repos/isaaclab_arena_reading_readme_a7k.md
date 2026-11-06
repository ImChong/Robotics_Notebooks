# Isaac Lab-Arena 上游 README 通读

> 来源归档（原文通读 · 支撑 `wiki/entities/isaac-lab-arena.md`）

- **标题：** Isaac Lab-Arena: Composable Environment Creation and Policy Evaluation for Robotics Simulation（上游 README）
- **作者：** NVIDIA Isaac Lab-Arena Contributors
- **类型：** repo / benchmark-framework
- **原文：** <https://github.com/isaac-sim/IsaacLab-Arena>（2026-11 经 raw.githubusercontent.com 获取 `main` 分支 README.md 全文，约 337 行）
- **一句话说明：** Isaac Lab 的开源扩展，以 scene / embodiment / task 三原语经 `ArenaEnvBuilder` 编译为标准 `ManagerBasedRLEnvCfg`，覆盖 Author（编写基准）、Execute（大规模执行）、Analyze（失败分析）三段评测工作流。
- **核对范围：** 通读 README 全文（概述、Key Features、Quick Start、Usage Example、项目结构、版本兼容表、Project Status、Ecosystem、贡献/许可/引用/致谢）。未打开 README 链出的 Sphinx 文档各章节与源码，相关细节仅依赖 wiki 页面既有内容。
- **通读：** 已完成

## 核心摘录（均出自 README）

- **定位：** 面向可扩展基准编写与策略仿真评测；工作流分 Author / Execute / Analyze 三段。
- **三原语：** scene（布局、物体、家具）、embodiment（机器人、观测、动作、传感器、控制器）、task（目标）；`ArenaEnvBuilder` 合成标准 `ManagerBasedRLEnvCfg`。
- **特性：** 关系式摆放（on / next to）、顺序任务链、agentic 环境生成、受控环境扰动（光照、背景、相机、物体质量）、与 Isaac Lab RL/数据生成互通、Newton 后端的可变形与接触丰富场景（如齿轮插入）。
- **执行：** GPU 上对数千个异构环境并行评测；多任务多策略实验可经 OSMO 跨节点；策略 client-server 架构（GR00T、π0.5、VLM、自定义）。
- **分析：** 子任务谓词（抓取、抬升、搬运、放置）与敏感性分析。
- **版本（README 当前）：** v0.3；`main` = Lab 3.0.0 + Sim 6.2.0 + Python ≥3.12；`release/0.3.1` = Sim 6.1.0；`release/0.2.1` = Lab 3.0.0 beta + Sim 6.0.0；`release/0.1.1` = Lab 2.3.0 + Sim 5.0.0。
- **状态：** Alpha，API 会变且无弃用告警，勿用于生产；稳定性推荐 `release/0.3.1`。
- **生态：** Published 列表含 RoboFinals、RoboCasa/LIBERO 任务（LW-BenchHub）、RoboTwin 2.0、LeRobot EnvHub、RHEO、RoboDojo、VABAR、DexBench；Coming Soon 为完整 RoboTwin 与 X Square、Sharpa、NVIDIA GEAR 等伙伴基准。
- **许可：** Apache 2.0，但需要含专有条款的 Isaac Sim。

## 对 wiki 的映射

- [`wiki/entities/isaac-lab-arena.md`](../../wiki/entities/isaac-lab-arena.md)

## 参考来源

- 仓库：<https://github.com/isaac-sim/IsaacLab-Arena>
