# legged_gym

> 来源归档（官方 README + 关键源码通读）

- **标题：** legged_gym（Isaac Gym Environments for Legged Robots）
- **类型：** repo
- **来源：** ETH Zurich RSL（Robotic Systems Lab），维护者 Nikita Rudin
- **链接：** https://github.com/leggedrobotics/legged_gym
- **项目页：** https://leggedrobotics.github.io/legged_gym/
- **配套论文：** Rudin et al., *Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning*（<https://arxiv.org/abs/2109.11978>），论文归档见 [`../papers/learning_to_walk_in_minutes_reading_arxiv_2109_11978.md`](../papers/learning_to_walk_in_minutes_reading_arxiv_2109_11978.md)
- **配套 PPO 实现：** https://github.com/leggedrobotics/rsl_rl（README 安装步骤指定 `v1.0.2`）
- **入库日期：** 2026-04-11（2026-11-07 补通读核对）
- **一句话说明：** ETH 的足式机器人 RL 训练框架，建立在 Isaac Gym 之上，RL locomotion 训练框架的工程参考范本。
- **核对范围：** 2026-11-07 通过 raw.githubusercontent.com 取得 `master` 分支并读完：`README.md` 全文；`legged_gym/envs/base/legged_robot_config.py` 全文（环境与 PPO 默认配置）；`legged_gym/envs/base/legged_robot.py` 的 `step`、`compute_reward`、`compute_observations`、`_process_rigid_shape_props`、`_process_rigid_body_props`、`_post_physics_step_callback`、`_push_robots`、课程函数与各 `_reward_*` 函数；`legged_gym/utils/terrain.py` 前半（地形生成与课程）；`legged_gym/envs/__init__.py`（任务注册）；`legged_gym/envs/anymal_c/anymal.py` 中执行器网络相关片段；rsl_rl `v1.0.2` README。**没有读**：`legged_robot.py` 其余工具函数细节、`terrain.py` 各具体地形生成函数、Cassie/A1 的具体配置文件、`play.py`/`train.py`/`task_registry.py` 的实现，没有安装或运行 Isaac Gym 训练。
- **通读：** 已完成
- **沉淀到 wiki：** 是 → [`wiki/entities/legged-gym.md`](../../wiki/entities/legged-gym.md)

## 核心摘录（均出自 README 与源码）

- **定位（README）：** 提供在 Isaac Gym 里训练 ANYmal（及其他机器人）在粗糙地形行走的环境，包含 sim2real 所需组件：执行器网络、摩擦与质量随机化、观测噪声、训练中随机推力。
- **维护状态（README 公告，2024-01-09）：** 随着 NVIDIA 从 Isaac Gym 转向 Isaac Sim，本工作的环境已迁移到 Isaac Lab；此仓库此后仅有限更新与支持，官方鼓励迁移。
- **安装要求（README）：** Python 3.6/3.7/3.8（推荐 3.8）；PyTorch 1.10 + CUDA 11.3；Isaac Gym **Preview 3**（Preview 2 不可用）；rsl_rl `v1.0.2`（PPO 实现）。
- **代码结构（README）：** 每个环境由 env 文件与 config 文件组成；config 内含 `LeggedRobotCfg`（环境参数）与 `LeggedRobotCfgPPO`（训练参数），均用继承；`cfg.rewards.scales` 中每个非零奖励尺度会自动加入名称对应的 `_reward_<name>` 函数并求和；任务通过 `task_registry.register(name, EnvClass, EnvConfig, TrainConfig)` 注册。已注册任务：`anymal_c_rough`、`anymal_c_flat`、`anymal_b`、`a1`、`cassie`。
- **默认环境配置（`LeggedRobotCfg`）：** `num_envs=4096`；观测 235 维（含 187 个地形高度采样点，17×11 的 1 m×1.6 m 矩形）；`num_actions=12`；`num_privileged_obs=None`（不为 None 时 `step()` 额外返回 critic 用的特权观测，即非对称 actor-critic 接口）；回合 20 s，并向算法传递超时信息 `send_timeouts=True`。
- **控制：** `control_type='P'` 位置目标 + PD；`action_scale=0.5`；`decimation=4`；`sim.dt=0.005`（故策略频率 50 Hz）；PD 增益在各机器人 cfg 里按关节给定。
- **观测内容（`compute_observations`）：** 基座线速度、角速度、投影重力、前 3 维指令、关节位置（减默认位姿）、关节速度、上一步动作，非盲模式再接高度测量，最后加均匀噪声；基座线速度在观测里（真机需额外估计，推测：这是该基线的已知差异，源码本身未注释）。
- **指令：** `lin_vel_x/y ∈ [−1,1]` m/s，`ang_vel_yaw ∈ [−1,1]` rad/s，heading 指令模式下由航向误差算角速度；每 10 s 重采样；指令课程默认关闭。
- **奖励（默认尺度，基类）：** tracking_lin_vel +1.0、tracking_ang_vel +0.5、lin_vel_z −2.0、ang_vel_xy −0.05、torques −1e-5、dof_acc −2.5e-7、feet_air_time +1.0、collision −1.0、action_rate −0.01；orientation、dof_vel、base_height、stand_still、feet_stumble、termination 默认为 0；`only_positive_rewards=True` 时总奖励被截断到非负，termination 奖励在截断之后再加；跟踪奖励为 exp(−误差²/σ)，σ=0.25。
- **域随机化（`domain_rand`）：** 摩擦系数 [0.5,1.25]（64 个桶）；`randomize_base_mass` 默认 **False**（范围 ±1 kg）；`push_robots=True`，每 15 s 把基座 xy 速度设为 ±1 m/s 的随机值；观测噪声 `add_noise=True`（dof_pos 0.01、dof_vel 1.5、lin_vel 0.1、ang_vel 0.2、gravity 0.05、height 0.1）。基类**没有**控制延迟随机化。
- **地形（`terrain`）：** 默认 `trimesh`，水平尺度 0.1 m；10 行（难度）× 20 列（类型）子地形，每块 8 m×8 m；类型比例 [平滑坡 0.1, 粗糙坡 0.1, 上楼梯 0.35, 下楼梯 0.25, 离散障碍 0.2]；课程模式下难度 = 行号/行数。
- **地形课程（`_update_terrain_curriculum`）：** 走出距离超过地块长度的一半则升级；走的距离不到「指令速度 × 回合时长 × 0.5」则降级；通关最高一级的被随机分到某一级。
- **执行器网络（`anymal.py`）：** `use_actuator_network` 时加载 TorchScript 网络，带隐状态（`sea_hidden_state`）；仅 ANYmal 系列使用。
- **PPO 默认（`LeggedRobotCfgPPO`）：** actor/critic 均为 [512,256,128] MLP、ELU；`clip_param=0.2`、`entropy_coef=0.01`、`num_learning_epochs=5`、`num_mini_batches=4`、`learning_rate=1e-3` 配 `schedule='adaptive'`（`desired_kl=0.01`）、`gamma=0.99`、`lam=0.95`、`max_grad_norm=1.0`；`num_steps_per_env=24`、`max_iterations=1500`。rsl_rl README：只实现了 PPO，是 NVIDIA `rl-pytorch` 的演进。
- **已知问题（README）：** GPU + 三角网格地形下 `net_contact_force_tensor` 报告的接触力不可靠；变通是在脚端加力传感器并关闭前向动力学力。
- **用法（README）：** `python legged_gym/scripts/train.py --task=anymal_c_flat`，可加 `--headless`；训练开始后按 `v` 关闭渲染以提速；`play.py` 回放最近一次模型。

## 对 wiki 的映射

- [`wiki/entities/legged-gym.md`](../../wiki/entities/legged-gym.md)
- [`wiki/entities/paper-anymal-walk-minutes-parallel-drl.md`](../../wiki/entities/paper-anymal-walk-minutes-parallel-drl.md)
