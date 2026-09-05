# newton-physics

> 来源归档

- **标题：** Newton Physics Engine
- **类型：** repo
- **来源：** Disney Research、Google DeepMind、NVIDIA（Linux Foundation 社区维护）
- **链接：** https://github.com/newton-physics/newton
- **Stars：** ~5.6k（2026-09-05 再核）
- **入库日期：** 2026-05-19
- **再核日期：** 2026-09-05
- **许可证：** Apache-2.0（代码）；文档 CC-BY-4.0
- **一句话说明：** 基于 NVIDIA Warp 的 GPU 加速、可扩展、可微物理仿真引擎，以 MuJoCo Warp 为主要后端，面向机器人学与仿真研究。
- **沉淀到 wiki：** 是 → [`wiki/entities/newton-physics.md`](../../wiki/entities/newton-physics.md)

---

## 开源边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **状态** | **已开源**（Apache-2.0） |
| **代码** | <https://github.com/newton-physics/newton> |
| **文档** | <https://newton-physics.github.io/newton/stable/guide/overview.html>（stable）；latest 见同站 `/latest/` |
| **产品页** | <https://developer.nvidia.com/newton-physics> |
| **Isaac Lab 集成** | <https://github.com/isaac-sim/IsaacLab/tree/feature/newton> |

官方 README 与文档站均可跑：`pip install "newton[examples]"` → `python -m newton.examples`。无需本地 CUDA Toolkit；macOS 仅 CPU。

---

## 核心定位（README 摘录，2026-09-05）

- 在 [NVIDIA Warp](https://github.com/NVIDIA/warp) 之上构建；扩展并泛化 Warp 已弃用的 `warp.sim` 模块。
- 集成 [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) 作为**主要物理后端**。
- 强调：GPU 计算、OpenUSD 场景描述、**可微仿真**、用户可插拔求解器与组件扩展。
- [Linux Foundation](https://www.linuxfoundation.org/) 托管的社区项目。
- 发起方：Disney Research、Google DeepMind、NVIDIA。

## 环境要求（摘要）

| 项 | 要求 |
|----|------|
| Python | 3.10+ |
| OS | Linux (x86-64/aarch64)、Windows (x86-64)、macOS（仅 CPU） |
| GPU | NVIDIA Maxwell+，驱动 545+（CUDA 12）；无需本地安装 CUDA Toolkit |
| 快速安装 | `pip install "newton[examples]"` → `python -m newton.examples` |

源码 + [uv](https://docs.astral.sh/uv/) 安装见 [installation guide](https://newton-physics.github.io/newton/latest/guide/installation.html)。版本与弃用策略见 [compatibility guide](https://newton-physics.github.io/newton/latest/guide/compatibility.html)。

## 示例族（README 目录，2026-09-05）

官方把示例按任务族列出，说明引擎已超出「刚体关节步进」：

| 族 | 代表入口 | 说明 |
|----|----------|------|
| Basic | `basic_pendulum` / `basic_urdf` / `basic_viewer` | 入门与 Viewer |
| Robot | `robot_g1` / `robot_h1` / `robot_anymal_d` / `robot_panda_hydro` / `robot_allegro_hand` / `robot_policy` | 人形 / 四足 / 机械臂 / 策略回放 |
| Controller | `controller_joint_impedance_heterogeneous` / `controller_operational_space_hybrid_force_motion` / `controller_differential_ik` | 阻抗 / 操作空间 / 差分 IK |
| Cloth / Cable | `cloth_style3d` / `cloth_franka` / `cable_twist` | Style3D 布料与缆索 |
| MPM | `mpm_granular` / `mpm_anymal` / `mpm_snow_ball` / `mpm_twoway_coupling` | ImplicitMPM 颗粒 / 雪 / 双向耦合 |
| DiffSim | `diffsim_ball` / `diffsim_cloth` / `diffsim_drone` | 可微仿真 |
| Kamino | `kamino_basic_fourbar` / `kamino_robot_anymal_d` | Kamino 求解器 |
| IK / Contacts | `ik_franka` / `nut_bolt_sdf` / `contacts_rj45_plug` | 接触丰富装配 |

共用 CLI：`--viewer {gl,usd,rtx,rerun,viser,null}`、`--device`、`--num-frames`、`--output-path`。

## 对 wiki 的映射

| 主题 | 目标 wiki |
|------|-----------|
| 引擎定位、求解器谱系、CollisionPipeline、与 MuJoCo Warp / Isaac Lab 关系 | [`wiki/entities/newton-physics.md`](../../wiki/entities/newton-physics.md) |
| 与 Cosmos / Omniverse 的 Physical AI 分工 | [`wiki/entities/nvidia-cosmos.md`](../../wiki/entities/nvidia-cosmos.md)、[`wiki/entities/nvidia-omniverse.md`](../../wiki/entities/nvidia-omniverse.md) |
| MuJoCo 生态对照 | [`wiki/entities/mujoco.md`](../../wiki/entities/mujoco.md)、[`wiki/entities/mjlab.md`](../../wiki/entities/mjlab.md) |
| GPU 并行 RL 训练栈 | [`wiki/entities/isaac-gym-isaac-lab.md`](../../wiki/entities/isaac-gym-isaac-lab.md) |
| 仿真器选型 | [`wiki/queries/simulator-selection-guide.md`](../../wiki/queries/simulator-selection-guide.md) |
