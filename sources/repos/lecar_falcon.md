# LeCAR-Lab/FALCON（官方源码、训练配置与 Sim2Real 部署核查归档）

> 来源归档（ingest · 仓库核查）

- **仓库：** <https://github.com/LeCAR-Lab/FALCON>（MIT 协议）
- **类型：** repo
- **依赖栈：** Python 3.8（训练：HumanoidVerse + IsaacGym Preview4 / IsaacSim / Genesis）/ Python 3.10（部署：Pinocchio 3.2.0、ONNX Runtime、`unitree_sdk2`、`booster_robotics_sdk`）
- **入库日期：** 2026-11-14
- **核对范围：** 核查官方仓库 `LeCAR-Lab/FALCON` 的 `README.md`（全文）、`sim2real/README.md`（全文）、`humanoidverse/train_agent.py`（全文 118 行）、`humanoidverse/eval_agent.py`（全文 147 行）、`humanoidverse/agents/decouple/ppo_decoupled_wbc_ma.py`（前 90 行）及 `humanoidverse/config/`、`humanoidverse/envs/decoupled_locomotion/` 目录清单。未在 IsaacGym 或真机环境运行训练与部署脚本。
- **通读：** 已完成
- **一句话说明：** `LeCAR-Lab/FALCON` 基于 `HumanoidVerse` 实现双代理多 Critic 算法 `PPOMultiActorCritic` 与差分力课程环境，并在 `sim2real/` 目录提供以 MuJoCo 作为仿真发布器替代真机的无缝 ONNX Sim2Sim/Sim2Real 部署链路（支持 Unitree G1 与 Booster T1）。

## 核心源码结构与入口核查

### 1) 双代理训练与环境实现（`humanoidverse/`）
- **训练主入口：** `python humanoidverse/train_agent.py +exp=decoupled_locomotion_stand_height_waist_wbc_diff_force_ma_ppo_ma_env +simulator=isaacgym ... num_envs=4096`，同时提供 `g1/g1_29dof_waist_fakehand` 与 `t1/t1_29dof_waist_wrist` 两组机器人配置（均约 `6k` iterations 收敛出步态与力补偿）。
- **核心算法类：** `humanoidverse/agents/decouple/ppo_decoupled_wbc_ma.py` 定义 `PPOMultiActorCritic`，按 `body_keys = ['lower_body', 'upper_body']` 实例化两套独立的 `PPOActor` 与 `PPOCritic`，共享 `RolloutStorage.MultiTransitions` 存储转移并在联合仿真步进中分别按 `reward_groups` 优化。
- **核心环境：** `humanoidverse/envs/decoupled_locomotion/decoupled_locomotion_stand_height_waist_wbc_ma_diff_force.py` 实现末端 Jacobian 力矩边界计算、Dirichlet 3D 力采样课程与随机施力点偏置。

### 2) 评估与模型导出（`humanoidverse/eval_agent.py`）
- 通过 `python humanoidverse/eval_agent.py +checkpoint=<path>` 加载训练 `config.yaml` 与权重执行 `algo.evaluate_policy()`；源码内集成了 `export_policy_as_jit` 与 `export_multi_agent_decouple_policy_as_onnx`（默认 `EXPORT_ONNX = False`、`EXPORT_POLICY = False`，导出前需在脚本中显式开启开关）。

### 3) Sim2Sim 与 Sim2Real 统一部署架构（`sim2real/`）
- **统一通信解耦：** 将 MuJoCo 环境（`python sim_env/loco_manip.py --config=config/g1/g1_29dof_falcon.yaml`）封装为与真机完全一致的状态发布器与指令接收器；Sim2Sim 时需先启动 `sim_env` 再运行策略端 `python rl_policy/loco_manip/loco_manip.py --model_path=models/falcon/g1_29dof.onnx`，真机时跳过 `sim_env` 直接启动 `rl_policy` 并指定网卡 `INTERFACE`。
- **实时性提示（`sim2real/README.md`）：** 文档特别提示纯 Python 版 `unitree_sdk2_python` 在 Unitree G1 机载 Jetson Orin 上无法保证实时推理，建议使用带 C++ 后端与 pybinding 封装的 `unitree_sdk2`；Booster T1 则使用补充状态发布器的 `hang0610/booster_robotics_sdk`。部署端依赖 `pinocchio=3.2.0` 求解上肢 IK。

## 对 wiki 的映射

- [paper-loco-manip-161-109-falcon](../../wiki/entities/paper-loco-manip-161-109-falcon.md)
