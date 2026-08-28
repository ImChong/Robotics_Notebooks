# RSL-RL

> 来源归档

- **标题：** RSL-RL
- **类型：** repo
- **链接：** https://github.com/leggedrobotics/rsl_rl
- **PyPI：** `pip install rsl-rl-lib`
- **文档：** https://leggedrobotics.github.io/rsl_rl/
- **论文：** [arXiv:2509.10771](https://arxiv.org/abs/2509.10771)（Schwarke / Mittal / Rudin / Hoeller / Hutter）
- **维护者：** ETH Zürich Robotic Systems Lab + NVIDIA
- **License：** BSD-3-Clause（源码 SPDX；ETH Zurich and NVIDIA CORPORATION）
- **Stars：** ~2.9k（2026-08-27）
- **入库日期：** 2026-08-28
- **一句话说明：** GPU 加速的轻量机器人 RL 库：PPO 与 Student–Teacher Distillation；`update()` 可选 bfloat16 混合精度；对接 Isaac Lab / Legged Gym / mjlab / MuJoCo Playground。
- **沉淀到 wiki：** 是 → [`wiki/entities/rsl-rl.md`](../../wiki/entities/rsl-rl.md)

## 开源核查（2026-08-28）

**已开源** — 可运行训练循环，不是占位 README。

| 项 | 内容 |
|----|------|
| 算法 | `rsl_rl/algorithms/ppo.py`、`distillation.py` |
| Runner | `OnPolicyRunner`、`DistillationRunner` |
| 模型 | MLP / RNN / CNN |
| 扩展 | RND、Symmetry（仅 PPO；蒸馏显式拒绝） |
| 混合精度 | `use_mixed_precision: bool = False`；为 True 时前向+损失走 `torch.amp.autocast(..., dtype=torch.bfloat16)`，反向 / clip / all-reduce / step 仍 fp32（[PR #219](https://github.com/leggedrobotics/rsl_rl/pull/219)） |
| 安装 | `pip install rsl-rl-lib` 或 `pip install -e .`（Python 3.9+） |

## 生态位置

被 [Isaac Lab](https://github.com/isaac-sim/IsaacLab)、[Legged Gym](https://github.com/leggedrobotics/legged_gym)、[mjlab](https://github.com/mujocolab/mjlab)、[MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) 用作默认 PPO 后端之一。下游 AMP 扩展见 [amp_rsl_rl.md](./amp_rsl_rl.md)。

## 对 wiki 的映射

- [RSL-RL 实体](../../wiki/entities/rsl-rl.md)
- [PPO](../../wiki/methods/ppo.md) — clip / KL 自适应 lr / rsl_rl 代码对照
- [RL Runner](../../wiki/concepts/rl-runner.md) — `OnPolicyRunner` vs Distillation Runner
- [特权训练](../../wiki/concepts/privileged-training.md) — student/teacher 观测分组
