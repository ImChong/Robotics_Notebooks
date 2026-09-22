# ASAP（LeCAR-Lab 官方实现）

- **标题：** ASAP — Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills
- **类型：** repo
- **仓库：** <https://github.com/LeCAR-Lab/ASAP>
- **项目页：** <https://agile.human2humanoid.com/>
- **论文：** RSS 2025，arXiv:[2502.01143](https://arxiv.org/abs/2502.01143)
- **许可：** MIT
- **机构：** 卡内基梅隆大学（CMU）；英伟达（NVIDIA）
- **收录日期：** 2026-09-22

## 一句话摘要

LeCAR-Lab 官方 **敏捷人形全身 Sim2Real** 代码库：基于 [HumanoidVerse](./humanoidverse.md) 与 [human2humanoid](https://github.com/LeCAR-Lab/human2humanoid) 工程栈，提供 phase-based motion tracking、delta action 模型训练与回灌微调、PHC 风格 SMPL→G1 重定向、MuJoCo sim2sim 与 Unitree G1 sim2real 部署；仓库内附带 ASAP 论文用 motion 数据与预置 CR7 等示例。

## 为何值得保留

- **残差动力学 Sim2Real 标杆：** 把 sim–real gap 显式建成可学习的 delta action，冻结后嵌入仿真微调，部署时摘掉——与纯 SysID/DR/不回灌 delta 形成清晰对照轴。
- **端到端可复现：** README 覆盖 IsaacGym / IsaacSim(IsaacLab) / Genesis 多后端、delta 开环/闭环训练 CLI、AMASS 重定向五步、sim2real 键盘操控与真机数据采集 demo（`listener_deltaa.py`）。
- **数据已发布：** `humanoidverse/data/motions/` 含 raw SMPL 与 retargeted G1 motions（TairanTestbed singles）。
- **社区活跃：** GitHub 2100+ stars（2026-09）。

## 仓库模块（编译自 README）

| 模块 / 入口 | 作用 |
|-------------|------|
| `humanoidverse/train_agent.py` | Hydra 统一训练：`+exp=motion_tracking` / `train_delta_a_open_loop` / `train_delta_a_closed_loop` |
| `humanoidverse/eval_agent.py` | 策略可视化与 rollout |
| `scripts/data_process/fit_smpl_shape.py` | 人形–SMPL 形体拟合 |
| `scripts/data_process/fit_smpl_motion.py` | AMASS/SMPL → 机器人 motion 重定向 |
| `sim2real/rl_policy/deepmimic_dec_loco_height.py` | decoupled locomotion + mimic ONNX 真机/sim2sim 部署 |
| `sim2real/rl_policy/listener_deltaa.py` | 真机 delta 数据采集 demo |

## 典型命令摘要

**Motion tracking（示例 CR7）：** `+exp=motion_tracking +simulator=isaacgym num_envs=4096 ...`

**Delta action 开环训练：** `+exp=train_delta_a_open_loop`，motion 文件需含 `"action"` 键。

**Delta 回灌微调：** `+exp=train_delta_a_closed_loop algo.config.policy_checkpoint=<delta_ckpt> env.config.add_extra_action=True +checkpoint=<tracking_ckpt>`

**Sim2Sim / Sim2Real：** `sim_env/base_sim.py` + `rl_policy/deepmimic_dec_loco_height.py`（ONNX loco + mimic checkpoints）

## 对 Wiki 的映射

- [paper-notebook-asap-aligning-simulation-and-real-world-physics](../../wiki/entities/paper-notebook-asap-aligning-simulation-and-real-world-physics.md)
- [paper-hrl-stack-25-asap](../../wiki/entities/paper-hrl-stack-25-asap.md)
- [HumanoidVerse 框架](../../wiki/entities/humanoidverse.md)
- [residual-policy-learning](../../wiki/methods/residual-policy-learning.md)

## 参考来源（原始）

- 代码仓库：<https://github.com/LeCAR-Lab/ASAP>
- 项目页：<https://agile.human2humanoid.com/>
- 论文 PDF：<https://arxiv.org/pdf/2502.01143>
