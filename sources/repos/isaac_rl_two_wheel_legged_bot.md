# Isaac-RL-Two-wheel-Legged-Bot

> 来源归档

- **标题：** Isaac-RL-Two-wheel-Legged-Bot（lab.flamingo）
- **类型：** repo
- **来源：** jaykorea（Jaehyung Cho，POSTECH 邮箱）
- **链接：** https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot
- **星标（截至 2026-08-28）：** 335
- **最近推送：** 2026-03-26
- **主要语言：** Python
- **许可证：** GitHub SPDX 为 MIT（文件名 `LICENCE`）；`setup.py` 另写 BSD-3-Clause，使用前核对
- **分类：** 强化学习训练 / Isaac Lab / 轮腿双足 / 约束 RL
- **入库日期：** 2026-08-28
- **一句话说明：** Isaac Lab 扩展 `lab.flamingo`：为 Flamingo 双轮足（及 Edu / Light / 4w4l / 人形变体）注册速度跟踪、TrackZ/RP/YK/JUMP、Backflip 等任务；CoRL 跑 PPO/SAC/TQC，并实现 Constraints as Termination（CaT）约束管理器。
- **沉淀到 wiki：** 是 → [`wiki/entities/isaac-rl-two-wheel-legged-bot.md`](../../wiki/entities/isaac-rl-two-wheel-legged-bot.md)
- **机构：** 浦项工科大学（POSTECH）→ `postech`
- **项目页：** 无独立项目页；README 徽章钉 **Isaac Sim 4.5 / Isaac Lab 2.0.0 / Python 3.10**

---

## README 要点（编译自上游）

- 标题叙事：**Isaac LAB for Flamingo**；包名 `lab.flamingo`（`pip install -e .` 在仓根执行）。
- 新特性（README 列表）：Flamingo rev.0.1.4、Flamingo Edu v1、观测 stack、**Constraint Manager**（[CaT, arXiv:2403.18765](https://arxiv.org/abs/2403.18765)）、**CoRL**（基于 rsl_rl，另实现 off-policy runner）。
- 宣称 **Sim2Real 零样本**（README 动图）；Lab→MuJoCo sim2sim 在 `sim2sim_onnx` 分支，**正在迁移**。
- 资产：USD 以 zip 入库（git 对 `.usd` 不友好），需解压到 `lab/flamingo/assets/data/Robots/Flamingo/...`。
- 训练 / 回放（仓根、已激活 Isaac Lab conda）：
  ```text
  python scripts/co_rl/train.py --task Isaac-Velocity-Flat-Flamingo-v1-ppo --algo ppo --num_envs 4096 --headless --num_policy_stacks 2 --num_critic_stacks 2
  python scripts/co_rl/play.py --task Isaac-Velocity-Flat-Flamingo-Play-v1-ppo --algo ppo --num_envs 64 --num_policy_stacks 2 --num_critic_stacks 2 --load_run <run>
  ```
- 已核对的 Gym ID（不完全）：
  - 速度：`Isaac-Velocity-Flat-Flamingo-v1-ppo`、`Isaac-Velocity-Rough-Flamingo-v1-ppo`、Light / Edu 变体
  - 辅助技能：`Isaac-TrackZ/RP/YK/JUMP-Flat-Flamingo-v1-ppo`
  - 约束：`Isaac-Velocity-Flat-Flamingo-v1-ppo-constraint`、`Isaac-Backflip-Flat-Flamingo-v1-ppo-constraint`
  - off-policy：`Isaac-Velocity-Flat-Flamingo-v3-sac` / `-tqc`；SRM-PPO 变体
- 约束路径入口点是仓内拷贝的 `lab.flamingo.isaaclab.isaaclab.envs:ManagerBasedConstraintRLEnv`，不是上游 Isaac Lab 原版 env 类。
- `config/extension.toml` 的 `repository` 仍指向已 404 的 `jaykorea/lab.flamingo.git`；clone 以本仓 URL 为准。

## 开源状态

- **已开源**：公开 GitHub 仓库；含可运行 `scripts/co_rl/train.py` / `play.py`、任务注册与 USD zip。
- **部分边界**：sim2sim ONNX 分支「Currently on migration update」；真机栈不在本仓 README 逐步展开（宣称零样本，部署细节需自行对照硬件仓）。
- **无独立项目页**；源码开放以 GitHub 为准。

## 对 wiki 的映射

- 实体页：[`wiki/entities/isaac-rl-two-wheel-legged-bot.md`](../../wiki/entities/isaac-rl-two-wheel-legged-bot.md)
- 形态概念：[`wiki/concepts/wheel-legged-biped.md`](../../wiki/concepts/wheel-legged-biped.md)
- 框架：[`wiki/entities/isaac-lab.md`](../../wiki/entities/isaac-lab.md)、[`wiki/entities/rsl-rl.md`](../../wiki/entities/rsl-rl.md)
- 对照厂商 Lab：[DDT_Lab](../../wiki/entities/ddt-lab.md)、[tita_rl](../../wiki/entities/tita-rl.md)
