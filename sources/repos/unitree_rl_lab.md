# unitree_rl_lab

> 来源归档

- **标题：** unitree_rl_lab
- **类型：** repo
- **来源：** unitreerobotics（Unitree 官方 GitHub 组织）
- **链接：** https://github.com/unitreerobotics/unitree_rl_lab
- **星标（截至 2026-08-07）：** ~1260
- **最近推送：** 2026-05-25
- **主要语言：** Python
- **许可证：** Apache-2.0
- **分类：** 强化学习训练
- **入库日期：** 2026-07-24
- **最近复核：** 2026-08-07
- **一句话说明：** 官方 Isaac Lab 2.x RL 环境，面向 Go2/H1/G1-29dof 的并行训练入口，并含 C++ deploy（Sim2Sim / Sim2Real）。
- **沉淀到 wiki：** 是 → [`wiki/entities/unitree-rl-lab.md`](../../wiki/entities/unitree-rl-lab.md)
- **组织地图：** [`sources/repos/unitree.md`](unitree.md)

---

## README 要点（编译自上游）

- 依赖徽章：**Isaac Sim 5.1.0**、**Isaac Lab 2.3.0**、Apache-2.0。
- 支持机型：Unitree **Go2**、**H1**、**G1-29dof**；官方对照展示 Isaac Lab / MuJoCo / Physical。
- 安装：Lab 目录外 clone → `./unitree_rl_lab.sh -i`；资产可选 HF [`unitree_model`](https://huggingface.co/datasets/unitreerobotics/unitree_model) USD（`UNITREE_MODEL_DIR`）或 [`unitree_ros`](https://github.com/unitreerobotics/unitree_ros) URDF（`UNITREE_ROS_DIR`，推荐 Isaac Sim ≥ 5.0）。
- 训练/列表：`./unitree_rl_lab.sh -l` / `-t --task Unitree-G1-29dof-Velocity` / `-p`；等价 `scripts/rsl_rl/train.py`、`play.py`。
- 部署：`deploy/` 下 C++ `robot_controller`，依赖 `unitree_sdk2`；Sim2Sim 走 [`unitree_mujoco`](https://github.com/unitreerobotics/unitree_mujoco)。

## 开源状态

- **已开源**：公开 GitHub 仓库（unitreerobotics/unitree_rl_lab），Apache-2.0。

## 对 wiki 的映射

- 实体页：[`wiki/entities/unitree-rl-lab.md`](../../wiki/entities/unitree-rl-lab.md)
- 组织枢纽：[`wiki/entities/unitree.md`](../../wiki/entities/unitree.md)
- 厂商 Lab 对照：[`wiki/entities/deeprobotics-rl-training.md`](../../wiki/entities/deeprobotics-rl-training.md)、[`wiki/entities/ddt-lab.md`](../../wiki/entities/ddt-lab.md)
