---
type: entity
tags: [repo, unitree, unitreerobotics, reinforcement-learning, isaac-gym, locomotion, sim2real]
status: complete
updated: 2026-10-06
related:
  - ./unitree.md
  - ./unitree-rl-lab.md
  - ./unitree-rl-mjlab.md
  - ./unitree-mujoco.md
  - ./legged-gym.md
  - ../tasks/locomotion.md
  - ../concepts/sim2real.md
  - ../methods/reinforcement-learning.md
sources:
  - ../../sources/repos/unitree_rl_gym.md
  - ../../sources/repos/unitree.md
summary: "unitree_rl_gym 是宇树官方基于 Isaac Gym + legged_gym 风格的 RL 训练仓（组织内星标最高），支持 Go2/H1/H1_2/G1；标准流程 Train→Play→Sim2Sim→Sim2Real，与 rl_lab / rl_mjlab 并行不可混拼观测。"
---

# unitree_rl_gym

**unitree_rl_gym** 是官方 **Isaac Gym** 路线的强化学习实现，任务与脚本组织接近社区 [legged_gym](./legged-gym.md)，覆盖 Unitree **Go2、H1、H1_2、G1**。

## 一句话定义

在 GPU 并行 Isaac Gym 环境里训练速度跟踪等 locomotion 策略，再经 Play / 外仿真 / 真机完成官方四段式落地。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| Isaac Gym | NVIDIA Isaac Gym | GPU 并行刚体仿真（本仓训练后端） |
| legged_gym | Legged Gym | ETH RSL 风格足式 RL 框架 |
| Sim2Sim | Simulation to Simulation | 换仿真器验证 |
| Sim2Real | Simulation to Real | 真机部署 |
| PPO | Proximal Policy Optimization | 常见策略优化算法（具体实现以上游为准） |

## 项目时间边界

官方仓库在 **2023-10-11** 的代码提交已包含实现，本路线将其标为「代码历史起点」；这不证明正式首发或首次公开日，也不表示当日已具备本页介绍的全部现有功能。提交证据见 [原始仓库时间核对](../../sources/repos/unitree_rl_gym.md)。

## 为什么重要

- **组织内星标最高**的官方 RL 入口，社区教程与复现材料最多。
- 明确写出 **Train → Play → Sim2Sim → Sim2Real**，便于对照其它两条官方线。
- 适合「先跑通一条完整链」；若团队已全面迁到 Isaac Lab 2.x / mjlab，应改选对应仓而非在本仓硬改。

## 核心原理

```text
Train (Isaac Gym) → Play (同环境可视化)
        → Sim2Sim (如 MuJoCo / 其它仿真)
        → Sim2Real (SDK2 / 真机)
```

| 阶段 | 含义 |
|------|------|
| Train | 多环境交互最大化回报；建议 headless |
| Play | 加载 checkpoint 目视/指标检查 |
| Sim2Sim | 导出网络到其它仿真，避免过拟合 Gym |
| Sim2Real | 物理机器人部署 |

训练产物默认：`logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`。

## 工程实践

```bash
python legged_gym/scripts/train.py --task=go2   # 或 g1 / h1 / h1_2
python legged_gym/scripts/play.py --task=go2
```

常用参数：`--headless`、`--resume`、`--num_envs`、`--max_iterations`、`--sim_device` / `--rl_device`。安装步骤见仓库 `doc/setup_en.md` / 中文文档。

**与另外两条官方线对照**：

| 仓 | 后端 | 何时选 |
|----|------|--------|
| **本仓** | Isaac Gym | 快速复现、社区资料多 |
| [unitree_rl_lab](./unitree-rl-lab.md) | Isaac Lab 2.x | 已在 Lab/Sim 5.x 生态 |
| [unitree_rl_mjlab](./unitree-rl-mjlab.md) | mjlab + MuJoCo Warp | 要官方 ONNX→C++ 闭环 |

## 局限与风险

- **Isaac Gym 已进入维护后期**，长期项目需评估迁 Lab / mjlab 的成本。
- **观测与动作定义与其它官方仓不通用**，禁止混拷配置。
- Play 默认加载「最近一次 run」——多人共享 logs 目录时易拿错模型。

## 源码运行时序图

官方实现 [unitreerobotics/unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)（短 SHA `276801e`）：基于 Isaac Gym 的 `legged_gym` 训练框架，`legged_gym/scripts/train.py` 为训练入口，策略优化由外部包 `rsl_rl` 的 `OnPolicyRunner` 完成。环境类在 `envs/base/legged_robot.py` 中实现：每个 RL 步对同一动作执行 `control.decimation` 次物理子步，子步内按 `control_type` 计算关节力矩（位置 PD / 速度 / 直接力矩）后交给 Isaac Gym 仿真。

入口：`python legged_gym/scripts/train.py --task=g1`（`--task` 可取 go2 / g1 / h1 / h1_2；回放验证用 `legged_gym/scripts/play.py`）

```mermaid
sequenceDiagram
    autonumber
    actor U as 用户
    participant TR as train.py<br/>train()
    participant REG as task_registry<br/>utils/task_registry.py
    participant ENV as LeggedRobot 及子类<br/>envs/base/legged_robot.py
    participant BT as BaseTask.__init__<br/>envs/base/base_task.py
    participant GS as Isaac Gym<br/>gym.simulate / 张量接口
    participant RUN as OnPolicyRunner<br/>rsl_rl（外部包）
    U->>TR: python legged_gym/scripts/train.py --task=g1
    TR->>TR: get_args() 解析命令行
    TR->>REG: make_env(name=task, args)
    Note over REG: 导入 envs 时已 register(name, 环境类, 环境配置, PPO 配置)
    REG->>REG: update_cfg_from_args / set_seed / parse_sim_params
    REG->>ENV: task_class(cfg, sim_params, physics_engine, sim_device, headless)
    ENV->>ENV: _parse_cfg(cfg)
    ENV->>BT: super().__init__（gym.acquire_gym）
    BT->>ENV: create_sim()
    ENV->>GS: gym.create_sim → _create_ground_plane → _create_envs
    ENV->>ENV: _init_buffers() 与 _prepare_reward_function()
    TR->>REG: make_alg_runner(env, name, args)
    REG->>RUN: OnPolicyRunner(env, train_cfg, log_dir, device)
    TR->>RUN: learn(max_iterations, init_at_random_ep_len=True)
    Note over RUN,ENV: rollout 与 PPO 更新在 rsl_rl 内部完成（未读源码）；每个 RL 步调用 env.step(actions)
    loop 每个 RL 步
        RUN->>ENV: step(actions)
        ENV->>ENV: clip(actions)；render()
        loop control.decimation 次物理子步
            ENV->>ENV: _compute_torques：P 为 kp·(a·scale + default − q) − kd·dq，V 与 T 分支另算，再按 torque_limits 裁剪
            ENV->>GS: set_dof_actuation_force_tensor(torques)
            ENV->>GS: simulate(sim)；refresh_dof_state_tensor
        end
        ENV->>ENV: post_physics_step()：刷新根状态，计算基座线速度、角速度与重力投影
        ENV->>ENV: check_termination()；compute_reward()（各 _reward_* 加权求和）
        ENV->>ENV: reset_idx(已终止环境)；compute_observations()
        ENV-->>RUN: obs, privileged_obs, rew, reset, extras
    end
```

图依据：unitree_rl_gym@276801e，读过 `README.md`（训练与回放命令、参数）、`legged_gym/scripts/train.py`、`legged_gym/scripts/play.py`（部分）、`legged_gym/envs/__init__.py`（注册）、`legged_gym/utils/task_registry.py`（`make_env` 与 `make_alg_runner`，部分）、`legged_gym/envs/base/base_task.py`（`__init__` 片段）、`legged_gym/envs/base/legged_robot.py`（`__init__`、`step`、`post_physics_step`、`_compute_torques` 等，部分）。`rsl_rl` 为外部依赖，未读；`V` 与 `T` 控制分支的具体公式未逐行核对。未运行代码。

## 关联页面

- [unitree_rl_lab](./unitree-rl-lab.md)
- [unitree_rl_mjlab](./unitree-rl-mjlab.md)
- [unitree_mujoco](./unitree-mujoco.md)
- [legged_gym](./legged-gym.md)
- [Locomotion](../tasks/locomotion.md)
- [Unitree](./unitree.md)

## 参考来源

- [sources/repos/unitree_rl_gym.md](../../sources/repos/unitree_rl_gym.md)
- 上游：<https://github.com/unitreerobotics/unitree_rl_gym>
- 源码核对（2026-11-02）：unitree_rl_gym@276801e，读过 README.md、legged_gym/scripts/train.py、legged_gym/scripts/play.py（部分）、legged_gym/envs/__init__.py、legged_gym/utils/task_registry.py（部分）、legged_gym/envs/base/base_task.py（部分）、legged_gym/envs/base/legged_robot.py（部分）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- 仓库内 `doc/setup_en.md` 与中文 README

