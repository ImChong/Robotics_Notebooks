# newton-physics

> 来源归档

- **标题：** Newton Physics Engine
- **类型：** repo
- **来源：** Disney Research、Google DeepMind、NVIDIA（Linux Foundation 社区维护）
- **链接：** https://github.com/newton-physics/newton
- **Stars：** ~4.9k（2026-05，入库时）
- **入库日期：** 2026-05-19
- **许可证：** Apache-2.0（代码）；文档 CC-BY-4.0
- **一句话说明：** 基于 NVIDIA Warp 的 GPU 加速、可扩展、可微物理仿真引擎，以 MuJoCo Warp 为主要后端，面向机器人学与仿真研究。
- **沉淀到 wiki：** 是 → [`wiki/entities/newton-physics.md`](../../wiki/entities/newton-physics.md)

---

## 核心定位（README 摘录）

- 在 [NVIDIA Warp](https://github.com/NVIDIA/warp) 之上构建；扩展并泛化 Warp 已弃用的 `warp.sim` 模块。
- 集成 [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) 作为**主要物理后端**。
- 强调：GPU 计算、OpenUSD 场景描述、**可微仿真**、用户可插拔求解器与组件扩展。
- [Linux Foundation](https://www.linuxfoundation.org/) 托管的社区项目。

## 环境要求（摘要）

| 项 | 要求 |
|----|------|
| Python | 3.10+ |
| OS | Linux (x86-64/aarch64)、Windows (x86-64)、macOS（仅 CPU） |
| GPU | NVIDIA Maxwell+，驱动 545+（CUDA 12）；无需本地安装 CUDA Toolkit |
| 快速安装 | `pip install "newton[examples]"` → `python -m newton.examples` |

## 对 wiki 的映射

| 主题 | 目标 wiki |
|------|-----------|
| 引擎定位、求解器谱系、与 MuJoCo Warp / Isaac Lab 关系 | [`wiki/entities/newton-physics.md`](../../wiki/entities/newton-physics.md) |
| MuJoCo 生态对照 | [`wiki/entities/mujoco.md`](../../wiki/entities/mujoco.md)、[`wiki/entities/mjlab.md`](../../wiki/entities/mjlab.md) |
| GPU 并行 RL 训练栈 | [`wiki/entities/isaac-gym-isaac-lab.md`](../../wiki/entities/isaac-gym-isaac-lab.md) |
| 仿真器选型 | [`wiki/queries/simulator-selection-guide.md`](../../wiki/queries/simulator-selection-guide.md) |
