# HumanoidVerse（LeCAR-Lab 多仿真器人形框架）

- **标题：** HumanoidVerse — A Multi-Simulator Framework for Humanoid Robot Sim-to-Real Learning
- **类型：** repo
- **仓库：** <https://github.com/LeCAR-Lab/HumanoidVerse>
- **许可：** MIT
- **机构：** 卡内基梅隆大学（CMU）LECAR Lab
- **收录日期：** 2026-09-22

## 一句话摘要

LeCAR-Lab 开源的 **人形多仿真器 RL 框架**：将 simulator、task、algorithm **模块化分离**，通过 Hydra 配置 **`+simulator=isaacgym|isaacsim|genesis`** 一行切换后端；支持 H1/G1 多 DoF 变体与 locomotion 等任务，是 [ASAP](./asap.md)、[FALCON](https://github.com/LeCAR-Lab/FALCON) 等工作的训练底座。

## 为何值得保留

- **Multi-sim + Sim2Real 工程语言：** README 对比表强调相对 Mujoco Playground、ProtoMotions、Humanoid Gym 等同系框架，HumanoidVerse 同时覆盖 **多仿真器** 与 **sim2sim/sim2real 管线**（后者在 ASAP 仓库中落地）。
- **最小切换成本：** 训练/评测统一入口 `humanoidverse/train_agent.py` / `eval_agent.py`，换 simulator 仅改 `+simulator=<name>`。
- **被广泛复用：** ASAP、FALCON、PBHC/KungfuBot 等多篇人形工作 README 均指向 HumanoidVerse 目录结构；社区 470+ stars（2026-09）。

## 支持范围（截至 README）

| 维度 | 内容 |
|------|------|
| 仿真器 | IsaacGym Preview4、IsaacSim 4.2 + IsaacLab、Genesis 0.2.1 |
| 机器人 | Unitree H1（10/19 DoF）、G1（12/23 DoF）等 |
| 任务 | locomotion（已发布）；motion tracking / sim2sim 管线在 ASAP 等下游仓库扩展 |
| Python | IsaacGym 环境 3.8；IsaacSim/Genesis 环境 3.10（分环境安装） |

## 典型命令

```bash
python humanoidverse/train_agent.py \
  +simulator=isaacgym \
  +exp=locomotion \
  +robot=h1/h1_10dof \
  num_envs=4096 \
  headless=True
```

评测：`python humanoidverse/eval_agent.py +checkpoint=<path>`

## 命名消歧

- 本仓库是 **训练框架**，与 Paper Notebooks 中待深读的 VLN 论文 *HumanoidVerse: A Versatile Humanoid for Vision-Language Guided Multi-Object Rearrangement*（arXiv:2508.16943）**同名不同物**；后者见 [paper-notebook-humanoidverse](../../wiki/entities/paper-notebook-humanoidverse.md)。

## 对 Wiki 的映射

- [HumanoidVerse 框架实体](../../wiki/entities/humanoidverse.md)
- [ASAP 论文+代码](../../wiki/entities/paper-notebook-asap-aligning-simulation-and-real-world-physics.md)
- [FALCON](../../wiki/entities/paper-loco-manip-161-109-falcon.md)
- [human2humanoid](../../wiki/entities/human2humanoid.md)

## 参考来源（原始）

- 代码仓库：<https://github.com/LeCAR-Lab/HumanoidVerse>
- 初始发布说明：2025-02-04 public release（locomotion pipeline）
