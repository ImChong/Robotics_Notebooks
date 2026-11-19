# CoorDex 官方源码（Skevinci/coordex）

> 官方 GitHub 仓库阅读归档（2026-11-19）

- **类型：** repo
- **标题：** CoorDex: Coordinating Body and Hand Priors for Continuous Dexterous Humanoid Loco-Manipulation
- **仓库：** <https://github.com/Skevinci/coordex>
- **README：** <https://github.com/Skevinci/coordex/blob/main/README.md>
- **论文：** <https://arxiv.org/abs/2606.23680>
- **项目主页：** <https://skevinci.github.io/coordex/>
- **项目页归档：** [coordex.md](../sites/coordex.md)
- **论文归档：** [coordex_arxiv_2606_23680.md](../papers/coordex_arxiv_2606_23680.md)
- **归档日期：** 2026-11-19
- **开放结论：** **部分开源（支持 checkpoint 仿真回放）**。仓库有环境、VAE/残差策略定义及权重；README 只介绍 rollout，当前文件树未见可从原始示范重训 tracking teacher、VAE 蒸馏及任务 PPO 的公开训练脚本。
- **核对范围：** 实读 README、`scripts/rsl_rl/play_locomanip.py`、`source/coordex/coordex/tasks/locomanip/__init__.py`、`tasks/locomanip/mdp/actions.py` 关键运行逻辑、`policies/actor_critic_coord_residual.py`、`distillation/vae.py` 结构，以及三个任务配置与 PPO config 的路径/入口；未运行 Isaac Sim 复现。

## 复现依赖与入口（README）

- Python **3.11**、Isaac Sim **5.0.0**、Isaac Lab **2.2.0**、PyTorch **2.7.0**（cu128）。
- 注册的 Gym ID：`CoorDex-WalkGrab-Wuji-v0`、`CoorDex-Fridge-Wuji-v0`、`CoorDex-WalkPickTurn-Wuji-v0`。
- 最小回放命令：

```bash
python scripts/rsl_rl/play_locomanip.py --task CoorDex-WalkGrab-Wuji-v0 --headless --num_envs 1
```

- 支持覆盖 `--checkpoint`、`--body-prior-checkpoint`、`--hand-prior-checkpoint`；不同 task 在入口文件的 `DEFAULTS` 对应不同 policy、body prior、hand prior checkpoint。尤其 WalkPickTurn 绑定 `active_wrist_20k.pt`，其他任务绑定 `kinematic_wrist_16k.pt`，不要混用。

## 阅读确认的运行链路

1. `play_locomanip.py` 启动 `AppLauncher`；`hydra_task_config` 读取任务与 RSL-RL agent config，并解析三类 checkpoint 的磁盘路径，注入 `env_cfg.actions.joint_pos`。
2. `coordex.tasks.locomanip.__init__` 用 `gym.register` 将三项 task 接到 Isaac Lab `ManagerBasedRLEnv`；`gym.make` 初始化场景及 `ResidualLatentAction`。
3. `ResidualLatentAction.__init__` 加载冻结的 body/hand `StudentTeacherVAE` prior + decoder；提供 `get_body_prior_mean` / `get_hand_prior_mean` 作为策略观测项。
4. `OnPolicyRunner` 加载已训练的 `ActorCriticCoordResidual` checkpoint；`get_inference_policy` 在每步由 `_CoordResidualActor` 读取 prior means、任务/手–物几何与接触状态，协调 trunk 后分成 body/hand 两个 `tanh` 残差头，共输出 28-D 潜残差。
5. `env.step(actions)` 进入 `ResidualLatentAction.process_actions`，通过 `_PriorPolicy.infer_with_delta` 做 `prior mean + residual` 后分别解码为关节动作；手部目标经 EMA/关节限位，合并到 `JointPositionAction`；`apply_actions` 驱动物理仿真，再反馈下一帧观测。

## 复现边界与注意

- 这是**已训练策略的 Isaac Lab 仿真回放**，不是 README 承诺的训练教程；不能凭 `rsl_rl_ppo_cfg.py` 存在就认为训练脚本已发布。
- checkpoint 加载依赖特定动作维、关节顺序、观测 term 和 prior 变体；换模型/hand 时需要重新核对配置和权重。
- GitHub 源码提供仿真任务与权重，但此处**未执行** GPU 回放或成功率复测；论文数值仍应按 arXiv 原文引用。
- 项目主页直连读取失败；代码开放状态已通过上述官方 GitHub 的实际文件核查。

## 对 wiki 的映射

- [CoorDex](../../wiki/entities/paper-coordex-dexterous-humanoid-loco-manipulation.md) — 源码时序图、入口和复现边界
