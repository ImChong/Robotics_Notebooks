# Argus（General Robotics Lab / Duke）

> 来源归档

- **标题：** Argus — Extreme Dynamic Symmetry Enables Omnidirectional and Multifunctional Robots
- **类型：** repo
- **维护方：** General Robotics Lab，杜克大学（Duke University）
- **链接：** <https://github.com/generalroboticslab/Argus>
- **Stars / Forks：** ~27 / 0（2026-07-01 检索）
- **入库日期：** 2026-07-01
- **一句话说明：** Science Robotics 2026 Argus 论文官方代码：Isaac Gym 定制 fork 上 PPO 训练球形腿式机器人族，含预训练 checkpoint、多任务评测脚本与 Blender 渲染管线。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-argus-dynamic-symmetry.md`](../../wiki/entities/paper-argus-dynamic-symmetry.md)

---

## 核心定位

**Argus** 仓库发布论文 *Extreme dynamic symmetry enables omnidirectional and multifunctional robots*（Science Robotics, eaec1725）的 **仿真训练、评测与可视化** 代码。机器人族为 **径向线性腿球形平台**，通过改变腿数与腿向分布探索 **动态各向同性 η** 对 locomotion / 鲁棒性 / loco-manipulation 的影响。

---

## 环境依赖

| 组件 | 要求 |
|------|------|
| OS | Linux（Ubuntu 20.04/22.04 测试） |
| GPU | NVIDIA CUDA 12，≥16 GB VRAM 推荐 |
| Python | 3.8（conda 环境 `argus`） |
| 仿真 | [boxiXia/isaacgym](https://github.com/boxiXia/isaacgym) 定制 fork（**非**官方 NVIDIA Isaac Gym） |
| 任务框架 | IsaacGymEnvs（需 `patch_isaacgymenvs.py` 去除 urdfpy 依赖） |
| 配置/日志 | Hydra + WandB（`envs/exp.sh` 设 `wandb_entity`） |

一键安装：`bash install.sh`（幂等，可设 `ENV_NAME` / `PY_VERSION` / `ISAACGYM_DIR` 等）。

---

## 形态变体（Morphology）

| 代号 | 腿数 / DoF | 动态各向同性 | 说明 |
|------|------------|--------------|------|
| `dof_12` | 12 | 较低 | 仿真对比 |
| `dof_20` | 20 | 近极端（实物原型） | 正十二面体顶点布局，η≈0.91 |
| `dof_32` | 32 | 仿真最高 | 大规模对比 |

---

## 任务与预训练 Play 命令（`envs/`）

| 任务类别 | `run.sh` 配置示例 | 说明 |
|----------|-------------------|------|
| Locomotion | `argus_base` | 平地滚动速度跟踪 |
| Terrain | `argus_terrain` | 离散障碍穿越 |
| Robustness | `argus_disable_leg_dof_20_const_vel` | 20-DoF 腿失效容错 |
| Carry | `argus_carry_object_dof_20_const_vel` | 恒速负载搬运 |
| Push rejection | `argus_push` | 外推扰动恢复 |
| Loco-manip (IL) | `argus_object_pushing_IL` / `argus_object_tracking_IL` | 点云感知物体推/跟（两阶段 IL） |

Play 加 `-p`；键盘控制加 `-k`（`i/k/j/l` 调速度）。

**感知任务两阶段：**

1. `argus_object_pushing_base` — 无感知 locomotion 基策略（RL）
2. `argus_object_pushing_IL` — PointNet 点云编码器 + IL 微调

---

## 仓库结构

```
Argus/
├── assets/           # checkpoint、URDF
├── envs/             # 训练入口 train.py、PPO、Hydra cfg、run.sh
├── visualization/    # 演示 GIF
├── blender_rendering/
├── install.sh
└── requirements.txt
```

---

## 对 wiki 的映射

- [paper-argus-dynamic-symmetry.md](../../wiki/entities/paper-argus-dynamic-symmetry.md) — 论文知识归纳与工程栈
- [argus_dynamic_symmetry_scirobotics_2026.md](../papers/argus_dynamic_symmetry_scirobotics_2026.md) — 论文原文归档
