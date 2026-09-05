---
type: entity
tags: [software, simulation, physics-engine, reinforcement-learning, deepmind]
status: complete
updated: 2026-09-05
related:
  - ../overview/sim-platforms-decade-technology-map.md
  - ./mujoco-wasm.md
  - ./mujoco-mjx.md
  - ./mujoco-warp.md
  - ./nvidia-warp.md
  - ./mujoco-playground.md
  - ../overview/robot-training-stack-layers-technology-map.md
  - ./brax.md
  - ../comparisons/mujoco-vs-isaac-sim.md
  - ./paper-barkour-quadruped-agility-benchmark.md
  - ./robot-motion-keyframe-editors.md
  - ./dm-control.md
  - ./gymnasium.md
  - ./wheel-legged-genesis.md
  - ./jackhan-walke3-e3-ecosystem.md
  - ./nvidia-omniverse.md
  - ./newton-physics.md
  - ../methods/reinforcement-learning.md
  - ../concepts/sim2real.md
  - ./adams.md
  - ./robot-descriptions-py.md
  - ../comparisons/robot-description-catalogs.md
sources:
  - ../../sources/repos/mujoco-menagerie.md
  - ../../sources/papers/simulation.md
  - ../../sources/repos/mujoco.md
  - ../../sources/blogs/wechat_embodied_ai_lab_robot_training_stack_layers_2026.md
  - ../../sources/blogs/wechat_shenlan_sim_platforms_top8_decade.md
summary: "MuJoCo 是专为生物力学、机器人学开发的高精度物理引擎。开源后成为机器人强化学习的基石，以极佳的接触稳定性和解析优化支持著称。"
---

# MuJoCo (物理引擎)

**MuJoCo (Multi-Joint dynamics with Contact)** 是一款专为机器人、生物力学和控制研究开发的高性能物理引擎。自被 DeepMind 收购并完全开源（Apache 2.0）以来，它已成为机器人强化学习（RL）和控制社区无可争议的基石工具。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MuJoCo | Multi-Joint dynamics with Contact | 接触丰富刚体仿真引擎 |
| MJCF | MuJoCo XML Format | 模型与场景描述格式 |
| MJX | MuJoCo JAX | JAX/XLA 后端，便于可微与批量 |
| RL | Reinforcement Learning | 腿足/人形 loco 常用训练后端 |
| PD | Proportional–Derivative | 仿真中常见的低层关节控制接口 |
| WASM | WebAssembly | 浏览器内运行 MuJoCo 的 `@mujoco/mujoco` 绑定载体 |

## 核心设计理念

不同于面向游戏或视觉特效的引擎（如 PhysX, Havok, Bullet），MuJoCo 是为**严格的控制理论**而生的：

1. **连续时间动力学的精确离散化**：
   MuJoCo 并不使用传统的基于“惩罚冲量”的方法解决碰撞。相反，它将接触和摩擦建模为一个平滑的凸优化问题。这使得即使在极大的时间步长下，系统的能量守恒和接触稳定性依然极佳。
2. **极易求导**：
   其内部状态可以极其方便地进行有限差分求导甚至解析求导，这对于基于梯度的轨迹优化（Trajectory Optimization）和 iLQR 等算法极度友好。

## 对机器人研究的统治力

- **RL 领域的基准测试**：Gymnasium（原 OpenAI Gym）注册表中的连续控制任务（如 HalfCheetah, Ant, Humanoid）几乎全部由 MuJoCo 驱动，是评价 PPO、SAC 等算法的常用标准；接口层见 [Gymnasium](./gymnasium.md)。DeepMind 的 [dm-control](./dm-control.md) 则在 MuJoCo 上提供另一套广泛使用、约定更统一的连续控制基准（Control Suite）与 Python 工具链。四足敏捷方向另有官方资产 **MuJoCo Menagerie** 中的 [`google_barkour_v0` / `google_barkour_vb`](./paper-barkour-quadruped-agility-benchmark.md)（与 [Barkour](./paper-barkour-quadruped-agility-benchmark.md) 论文及开源机体 README 交叉索引）。
- **Sim2Real 的证明**：诸多成功的 Sim2Real 论文（尤其是四足机器人和灵巧手操作领域）都证明了：只要系统辨识和域随机化做得好，在 MuJoCo 中训练的策略可以直接无缝迁移到物理硬件上。

## 优势与局限

- **优势**：
  - 单线程计算性能极高（千赫兹级的闭环仿真毫无压力）。
  - 接触模型非常稳定，很少发生“穿模”或无理的反弹（Explosion）。
  - `mjcf` (XML) 模型描述文件格式严谨且专为机器人设计。
- **局限**：
  - 原生 CPU MuJoCo 在单机多 GPU **环境复制数** 上，仍常逊色于 Isaac Gym 类专并行栈。JAX 批量 / 可微走 [**MuJoCo MJX**](./mujoco-mjx.md)；NVIDIA GPU 高吞吐、对齐 MJCF 的 drop-in 走 [**MuJoCo Warp**](./mujoco-warp.md)（PGS / PLUGIN 等有缺口，AD 未通）。
  - 对流体、软体（Soft body）和极其复杂的传感器渲染（如高保真相机）支持较弱；大规模 **壳/体 FEM + 亿级接触** 的离线路径见 [ppf-contact-solver](./ppf-contact-solver.md)。
  - **浏览器 WASM**（[`@mujoco/mujoco`](./mujoco-wasm.md)）适合 demo、教学与轻量 Sim2Sim，吞吐与 API 完备度仍弱于原生绑定；多线程版另需 COOP/COEP 隔离头。

## HMI 开源主表入口

[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) 收录于具身智能研究室 [开源项目主表](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md)。

主表定位：为常见机器人维护可直接运行的 MJCF 资产（网格、执行器、传感器、默认姿态）。算法对照应复用同一资产以减少模型适配差异。本库在本页索引 Menagerie，不另建重复实体；Barkour 等机体见 [Barkour](./paper-barkour-quadruped-agility-benchmark.md)。

覆盖核对见 [HMI 开源项目主表覆盖索引](../queries/hmi-opensource-projects-coverage.md)。Python 侧按名加载 Menagerie 镜像模块，见 [robot_descriptions.py](./robot-descriptions-py.md) 的 `*_mj_description`；与 Awesome / ROS 2 仓的选型见 [机器人描述目录选型](../comparisons/robot-description-catalogs.md)。

## 关联页面
- [ADAMS（Automatic Dynamic Analysis of Mechanical Systems）](./adams.md) — 工业 MBD 程序谱系原点（Orlandea 1973/1977）；与开源 RL 引擎分工不同
- [MuJoCo WASM（浏览器绑定）](./mujoco-wasm.md) — 官方 `@mujoco/mujoco` 与社区 demo（zalo/mujoco_wasm）生态
- [MuJoCo Playground](./mujoco-playground.md) — MJX 上的任务入口层，强调 time-to-robot 与 sim2real 短链路
- [训练栈分层地图](../overview/robot-training-stack-layers-technology-map.md) — MuJoCo 在物理/sim2sim 层的定位
- [MuJoCo MJX（JAX / XLA 后端）](./mujoco-mjx.md) — 与 MJCF 对齐的 JAX 重实现，用于高吞吐与可微 rollout
- [MuJoCo Warp](./mujoco-warp.md) — Warp/CUDA 上的 GPU MuJoCo；Newton 主要刚体后端
- [NVIDIA Warp](./nvidia-warp.md) — MJWarp / Newton 的 JIT 计算层
- [Brax](./brax.md) — JAX 侧 RL 训练与 README 中的 Playground / MJX 迁移指引
- [机器人关键帧与运动编辑工具](./robot-motion-keyframe-editors.md) — MJCF 场景上的关键帧编排与 LZ4 轨迹包（Stanford `robot_keyframe_kit` 等）
- [Gymnasium](./gymnasium.md) — 单智能体 RL 环境 API；MuJoCo 域经典控制基准的注册入口
- [dm_control / Control Suite](./dm-control.md) — MuJoCo 上的连续控制基准与 Python 栈
- [对比：MuJoCo vs Isaac Sim](../comparisons/mujoco-vs-isaac-sim.md)
- [Motrix](./motrix.md) — 现代化 Rust 高性能仿真引擎
- [SAPIEN (仿真引擎)](./sapien.md) — 侧重关节体交互
- [Robot Viewer](./robot-viewer.md) — 支持 MJCF 格式的 Web 查看器
- [URDD（Beyond URDF）](./paper-urdd-universal-robot-description-directory.md) — 以 URDF 为起点的派生模块目录（与 MJCF 这类仿真专用描述对照理解「预处理资产层」）
- [NVIDIA Omniverse 具身仿真底座](./nvidia-omniverse.md)
- [Newton Physics](./newton-physics.md) — Warp + MJWarp 的 GPU 多求解器引擎（LF 托管；MJWarp 路径 AD 未通）
- [Reinforcement Learning](../methods/reinforcement-learning.md)
- [Sim2Real 概念](../concepts/sim2real.md)
- [wheel_legged_genesis](./wheel-legged-genesis.md) — Genesis 策略迁 MuJoCo 的双轮足 sim2sim
- [Barkour（Menagerie MJCF + 敏捷课）](./paper-barkour-quadruped-agility-benchmark.md)
- [robot_descriptions.py](./robot-descriptions-py.md) — `loaders.mujoco` 与 `*_mj_description`
- [机器人描述目录选型](../comparisons/robot-description-catalogs.md)

## 参考来源

- [lqr_ilqr_primary_refs.md](../../sources/papers/lqr_ilqr_primary_refs.md) — MuJoCo + iLQR 在线轨迹优化（Tassa et al. 2012/14）
- [MuJoCo 物理引擎（仓库归档）](../../sources/repos/mujoco.md)
- [mujoco-mjx（MJX 子树归档）](../../sources/repos/mujoco-mjx.md)
- Todorov, E., Erez, T., & Tassa, Y. (2012). *MuJoCo: A physics engine for model-based control*.
