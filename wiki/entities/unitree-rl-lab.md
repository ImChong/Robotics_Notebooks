---
type: entity
tags: [repo, unitree, unitreerobotics, reinforcement-learning, isaac-lab, locomotion, sim2real]
status: complete
updated: 2026-09-15
related:
  - ./unitree.md
  - ./unitree-rl-gym.md
  - ./unitree-rl-mjlab.md
  - ./unitree-mujoco.md
  - ./unitree-ros.md
  - ./unitree-model.md
  - ./robot-lab.md
  - ./deeprobotics-rl-training.md
  - ./ddt-lab.md
  - ./cyclo-lab.md
  - ./paper-humoslope-physics-guided-slope-locomotion.md
  - ../tasks/locomotion.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/repos/unitree_rl_lab.md
  - ../../sources/repos/unitree.md
  - ../../sources/papers/humoslope_arxiv_2607_07830.md
summary: "unitree_rl_lab 是基于 Isaac Lab 的官方 Unitree RL 环境扩展，支持 Go2/H1/G1-29dof；资产可从 Hugging Face USD 或 unitree_ros URDF 引入，含 C++ deploy，与 Isaac Gym 版 unitree_rl_gym 并行。"
---

# unitree_rl_lab

**unitree_rl_lab** 在 [Isaac Lab](https://github.com/isaac-sim/IsaacLab) 之上提供 Unitree 机器人的强化学习环境；徽章显示面向 **Isaac Sim 5.1 / Isaac Lab 2.3** 一代（以仓库当时 README 为准）。

## 一句话定义

把 Go2 / H1 / G1-29dof 的官方任务环境装进 Isaac Lab 工作流，用 Lab 的 manager-based API 做并行训练，再走 MuJoCo/真机验证。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Isaac Lab | NVIDIA Isaac Lab | 基于 Omniverse 的机器人学习框架 |
| Isaac Sim | NVIDIA Isaac Sim | Lab 依赖的仿真应用 |
| USD | Universal Scene Description | 资产格式；可从 HF `unitree_model` 获取 |
| URDF | Unified Robot Description Format | 可从 `unitree_ros` 引入（Isaac Sim ≥ 5.0） |
| RL | Reinforcement Learning | 强化学习 |
| Sim2Real | Simulation to Real | 仿真到真机 |

## 为什么重要

- 团队若已标准化 **Isaac Lab 2.x**，这是官方对齐入口，避免继续锁死在 Isaac Gym。
- 资产路径提供 **USD（HF）** 与 **URDF（unitree_ros）** 两套，减少「找不到机器人描述」的卡点。
- 与 gym / mjlab 形成清晰三选一，便于在 [`unitree`](./unitree.md) 枢纽做选型。

## 核心原理

| 步骤 | 说明 |
|------|------|
| 独立克隆 | 仓库放在 Isaac Lab 目录**之外** |
| 可编辑安装 | `./unitree_rl_lab.sh -i`（需已激活含 Lab 的 Python） |
| 资产 | `UNITREE_MODEL_DIR`（USD）或 `UNITREE_ROS_DIR`（URDF，推荐 Sim≥5.0） |
| 训练/推演 | 遵循仓库脚本与 Lab 惯例；官方展示 Isaac Lab / MuJoCo / Physical 对照 |

## 工程实践

1. 按 Isaac Lab 官方指南装好仿真与 Lab（徽章：**Isaac Sim 5.1 / Isaac Lab 2.3**）。
2. `git clone https://github.com/unitreerobotics/unitree_rl_lab.git` 后执行 `./unitree_rl_lab.sh -i`。
3. 配置 `source/unitree_rl_lab/.../assets/robots/unitree.py` 中的 `UNITREE_MODEL_DIR`（USD）或 `UNITREE_ROS_DIR`（URDF）。
4. 注意：GitHub [`unitree_model`](./unitree-model.md) 已 deprecated，USD 请用 Hugging Face 数据集。
5. 部署：编译 `deploy/robots/<robot>/` 的 `robot_controller`（依赖 `unitree_sdk2`）；Sim2Sim 先跑 [`unitree_mujoco`](./unitree-mujoco.md)。
6. 厂商 Lab 对照：云深处 [rl_training](./deeprobotics-rl-training.md)、直驱 [DDT_Lab](./ddt-lab.md)、ROBOTIS [cyclo_lab](./cyclo-lab.md)（偏臂/AI Worker 操作）；多厂商社区扩展见 [robot_lab](./robot-lab.md)。

## 局限与风险

- **硬件与驱动门槛高**（GPU / 驱动 / Isaac 版本矩阵）；RTX 50 系需核对 Isaac Sim 版本说明。
- 与 `unitree_rl_gym` 的任务名、观测空间**不互通**。
- 首次加载资源受网络与磁盘影响，勿在未就绪时判断「安装失败」。
- **开源状态：已开源**（Apache-2.0）。

## 源码运行时序图

官方实现 [unitreerobotics/unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab)（短 SHA `4960b84`）：基于 Isaac Lab 的强化学习任务库，训练入口 `scripts/rsl_rl/train.py`。任务通过 `source/unitree_rl_lab/.../robots/*/__init__.py` 中的 `gym.register` 注册，环境与 PPO 配置分别来自 `velocity_env_cfg.py` 与 `agents/rsl_rl_ppo_cfg.py`；仿真环境、包装器与 PPO 训练循环由外部包 Isaac Lab（`isaaclab`、`isaaclab_rl`）与 `rsl_rl` 提供。`deploy/` 下的 C++ 真机部署（`deploy/robots/*/main.cpp`）未纳入本图。

入口：`python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity`

```mermaid
sequenceDiagram
    autonumber
    actor U as 用户
    participant TR as train.py<br/>scripts/rsl_rl/train.py
    participant AL as AppLauncher<br/>isaaclab（外部）
    participant REG as gym.register<br/>tasks/locomotion/robots/g1/29dof/__init__.py
    participant CFG as velocity_env_cfg / rsl_rl_ppo_cfg<br/>BasePPORunnerCfg
    participant GYM as gym.make<br/>ManagerBasedRLEnv（isaaclab 外部）
    participant WRAP as RslRlVecEnvWrapper<br/>isaaclab_rl（外部）
    participant RUN as OnPolicyRunner<br/>rsl_rl（外部）
    participant EXP as 日志与导出<br/>logs/rsl_rl/<experiment>/
    U->>TR: python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity
    TR->>AL: AppLauncher(args_cli)（先启动 Isaac Sim，再导入任务与 RL 包）
    TR->>REG: import unitree_rl_lab.tasks（执行 gym.register，写入 env 与 rsl_rl 入口）
    TR->>CFG: hydra_task_config 按任务名解析 env_cfg 与 agent_cfg
    TR->>TR: update_rsl_rl_cfg；覆盖 num_envs、seed、device、max_iterations
    TR->>EXP: 生成 {时间戳}_{run_name} 日志目录
    TR->>GYM: gym.make(task, cfg=env_cfg)
    TR->>WRAP: RslRlVecEnvWrapper(env, clip_actions)
    TR->>RUN: OnPolicyRunner(env, agent_cfg.to_dict(), log_dir, device)
    TR->>EXP: add_git_repo_to_log；dump env.yaml / agent.yaml；export_deploy_cfg；复制环境配置文件
    TR->>RUN: learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)
    Note over RUN,GYM: rollout 与 PPO 更新在 rsl_rl 内部完成（未读源码）；每步经 WRAP 调用环境 step（isaaclab 内部未读）
    RUN-->>EXP: 周期保存 checkpoint（save_interval）
    TR->>GYM: env.close()；simulation_app.close()
```

图依据：unitree_rl_lab@4960b84，读过 `README.md`（训练命令与 `unitree_rl_lab.sh` 用法）、`scripts/rsl_rl/train.py`（主流程，`AppLauncher` 与 `learn` 等调用位置）、`source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/__init__.py`、`source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/__init__.py`（`gym.register`）、`source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py`（`BasePPORunnerCfg` 片段）。`velocity_env_cfg.py`、`export_deploy_cfg`、`hydra_task_config`、Isaac Lab 与 rsl_rl 内部实现未读。未运行代码。

## 关联页面

- [unitree_rl_gym](./unitree-rl-gym.md)
- [unitree_rl_mjlab](./unitree-rl-mjlab.md)
- [unitree_model](./unitree-model.md)
- [unitree_ros](./unitree-ros.md)
- [Unitree](./unitree.md)
- [robot_lab](./robot-lab.md)
- [Deep Robotics rl_training](./deeprobotics-rl-training.md)
- [DDT_Lab](./ddt-lab.md)
- [HumoSlope](./paper-humoslope-physics-guided-slope-locomotion.md) — 以本仓库为本体感知基线，对比陡坡专用 ZMP/BSGA 管线

## 参考来源

- [sources/repos/unitree_rl_lab.md](../../sources/repos/unitree_rl_lab.md)
- 上游：<https://github.com/unitreerobotics/unitree_rl_lab>
- [HumoSlope（arXiv:2607.07830）](../../sources/papers/humoslope_arxiv_2607_07830.md) — 论文将 Unitree RL Lab 作 proprioception-only 基线
- 源码核对（2026-11-02）：unitreerobotics/unitree_rl_lab@4960b84，读过 README.md、scripts/rsl_rl/train.py、source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/__init__.py、tasks/locomotion/robots/g1/29dof/__init__.py、tasks/locomotion/agents/rsl_rl_ppo_cfg.py（部分）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- Isaac Lab 安装指南：<https://isaac-sim.github.io/IsaacLab/>

