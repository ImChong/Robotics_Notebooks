---
type: entity
tags: [repo, framework, unitree, mjlab, mujoco, reinforcement-learning, sim2real, humanoid, locomotion]
status: complete
updated: 2026-11-21
original:
  - ../../sources/repos/unitree_rl_mjlab.md
related:
  - ./unitree-ros.md
  - ./unitree-rl-gym.md
  - ./unitree-rl-lab.md
  - ./unitree-mujoco.md
  - ./unitree-sdk2.md
  - ./mjlab.md
  - ./mjlab-playground.md
  - ./unitree.md
  - ./unitree-g1.md
  - ./amp-mjlab.md
  - ./legged-gym.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/repos/unitree_rl_mjlab.md
summary: "unitree_rl_mjlab 是 Unitree 官方基于 mjlab 的 RL 训练框架，支持 Go2/G1/H1_2 等 7 款机型，覆盖速度跟踪与动作模仿，内建 ONNX 导出到 C++ 真机部署的完整链路。"
---

# unitree_rl_mjlab (Unitree 官方 RL 框架)

**unitree_rl_mjlab** 是由 Unitree Robotics 官方维护的强化学习训练框架，以 **mjlab**（Isaac Lab API + MuJoCo Warp）为底层，覆盖旗下 7 款机器人型号，提供从仿真训练到真机部署的完整 pipeline。

## 一句话定义

在 mjlab（MuJoCo Warp GPU 并行仿真 + Isaac Lab 风格任务接口）上训练宇树四足与人形的速度跟踪和动作模仿策略，在每次保存 checkpoint 时自动导出带元数据的 `policy.onnx`，并直接衔接官方 C++ `unitree_sdk2` 部署到 `unitree_mujoco` 与真机。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| G1 | Unitree G1 Humanoid | 宇树入门级教育科研人形平台 |
| ONNX | Open Neural Network Exchange | 跨框架神经网络模型交换格式 |
| Isaac Lab | NVIDIA Isaac Lab | 基于 Omniverse 的机器人学习训练框架 |
| API | Application Programming Interface | 应用程序编程接口 |
| MuJoCo | Multi-Joint dynamics with Contact | 接触丰富的刚体物理仿真引擎 |
| legged_gym | Legged Gym | 足式机器人 RL 训练的常用开源框架 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |
| SDK | Software Development Kit | 软件开发工具包 |
| GPU | Graphics Processing Unit | 图形处理器，大规模并行仿真训练的算力基础 |
| URDF | Unified Robot Description Format | 统一机器人描述格式 |
| Locomotion | Robot Locomotion | 足式/人形等无轮移动能力的总称 |

## 为什么重要？

这是 Unitree 的**官方推荐**训练路线之一，与社区框架（legged_gym、AMP_mjlab）的关键区别：

- **官方维护**：与 Unitree SDK、ONNX 部署链路深度整合
- **多机型覆盖**：单框架支持四足（Go2）和人形（G1、H1_2、H2 等）
- **完整 Sim2Real 链路**：训练 → ONNX 导出 → C++ 编译 → 真机部署，有明确的操作步骤

## 核心原理

`unitree_rl_mjlab` 将宇树各机型环境注册到 `mjlab` 任务表（`register_mjlab_task`），并结合 `rsl_rl` 与自定义 `VelocityOnPolicyRunner` 实现「并行训练 → 自动导出 ONNX → C++ 真机部署」闭环：

```mermaid
flowchart LR
  CFG["任务与配置注册<br/>env_cfgs.py / rl_cfg.py"] --> TR["scripts/train.py<br/>mjlab ManagerBasedRlEnv (4096 envs)"]
  MOT["参考动作 CSV → NPZ<br/>scripts/csv_to_npz.py"] --> TR
  TR --> RUN["VelocityOnPolicyRunner<br/>rsl_rl PPO 训练 + 导出 policy.onnx"]
  RUN --> PLAY["scripts/play.py<br/>MuJoCo 回放验证"]
  RUN --> SIM["unitree_mujoco + C++ 控制器<br/>./g1_ctrl --network=lo"]
  SIM --> REAL["真机部署 (192.168.123.222)<br/>./g1_ctrl --network=enp5s0"]
```

- **支持机器人（7 款）**：Go2、A2、As2、**G1**（含 23DoF）、R1、**H1_2**、**H2**。
- **训练任务**：
  - **速度跟踪（Velocity Tracking）**：如 `Unitree-Go2-Flat`、`Unitree-G1-Flat`、`Unitree-G1-23Dof-Flat`、`Unitree-H1_2-Flat`、`Unitree-A2-Flat`、`Unitree-R1-Flat`（支持 4096 并行环境与 `--gpu-ids` 多卡）。
  - **动作模仿（Motion Imitation）**：`Unitree-G1-Tracking-No-State-Estimation`、`Unitree-G1-23Dof-Tracking-No-State-Estimation`，先由 `scripts/csv_to_npz.py` 将 30 fps 动作 CSV 转为 50 fps NPZ 再训练。
- **产物输出**：checkpoint 默认写入 `logs/rsl_rl/<robot>_(velocity|tracking)/<date_time>/model_<iteration>.pt`，并在保存时同步导出 `policy.onnx` 与 `policy.onnx.data`（附带环境观测/动作元数据）。

## 工程实践

1. **训练与回放**：按 `doc/setup_en.md` 配好 `mjlab` 环境后执行：
   ```bash
   python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096
   python scripts/play.py Unitree-G1-Flat --checkpoint_file=logs/rsl_rl/g1_velocity/2026-xx-xx_xx-xx-xx/model_xx.pt
   ```
2. **ONNX 导出与 C++ 编译**：将生成的 `policy.onnx` 与 `policy.onnx.data` 放入 `deploy/robots/g1/config/policy/velocity/v0/exported`，在 `deploy/robots/g1/` 下使用 `cyclonedds` 与 `unitree_sdk2` 执行 `mkdir build && cd build && cmake .. && make`。
3. **Sim2Sim 与真机部署**：先编译并启动 `unitree_mujoco`（手柄需连接），运行 `./g1_ctrl --network=lo` 验证；真机部署需在零力矩模式下按 `L2 + R2` 切入 `debug mode`、本机网口配置 `192.168.123.222/24`，再执行 `./g1_ctrl --network=enp5s0`。

### 与相关框架的对比

| 维度 | unitree_rl_mjlab | AMP_mjlab | legged_gym | robot_lab |
|------|-----------------|-----------|------------|-----------|
| 维护方 | Unitree 官方 | 社区 | ETH RSL | 个人 |
| 底层仿真 | mjlab（MuJoCo Warp） | mjlab | IsaacGym | IsaacLab |
| 支持机型 | Unitree 7 款 | G1 | 通用 | 26+ |
| 核心任务 | 速度跟踪 + 动作模仿 | AMP + recovery | 速度跟踪 | 速度跟踪 + AMP Dance |
| 部署链路 | ONNX → C++（官方） | ONNX | 各异 | 各异 |

**与官方其它 RL 仓的关系**：同属 [unitreerobotics](./unitree.md) 组织时，`unitree_rl_gym`（Isaac Gym）与 `unitree_rl_lab`（Isaac Lab）是并行训练入口；本仓特长是 **mjlab + 官方 ONNX→C++** 闭环。组织级仓库地图见 [sources/repos/unitree.md](../../sources/repos/unitree.md)。

**与 `unitree_ros` 的分工**：`unitree_ros` 提供 ROS1 + Gazebo8 的 URDF 与关节级仿真/控制示例（README 明确 Gazebo 不做高层行走）；`unitree_rl_mjlab` 面向当前研究常用的 **MuJoCo 并行训练 + SDK2** 部署闭环。二者同属官方开源，但中间件与仿真器代际不同，见 [unitree_ros](./unitree-ros.md)。

## 局限与风险

- **观测与动作配置不可跨仓混用**：本仓任务注册、观测顺序与动作缩放和 `unitree_rl_gym` / `unitree_rl_lab` 不通用，切勿跨仓直接套用 checkpoint。
- **C++ 与驱动依赖链敏感**：真机部署依赖 `cyclonedds`、`unitree_sdk2` 以及导出 `policy.onnx` + `policy.onnx.data` 配套；仿真（`--network=lo`）与真机（`--network=enp5s0`）网口参数和 `192.168.123.222` 网段须严格核对。
- **物理安全前提**：上机前务必让机器人在悬吊状态下进入零力矩模式并按 `L2 + R2` 确认进入 `debug mode`；本轮仅核查源码入口，未实际运行 GPU 训练或真机测试。

## 源码运行时序图

官方实现 [unitreerobotics/unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)（短 SHA `1425b15`）：基于 mjlab（MuJoCo 后端）的训练与部署仓，训练入口 `scripts/train.py`（tyro 命令行）。任务在 `src/tasks/velocity/__init__.py` 中经 `register_mjlab_task` 注册，环境与 PPO 配置来自 `config/g1/env_cfgs.py` 与 `rl_cfg.py`；`VelocityOnPolicyRunner.save()` 在每次保存 checkpoint 时额外导出 `policy.onnx`（供 `deploy/` 真机部署）并写入元数据。仿真环境、包装器与 PPO 循环由外部包 mjlab 与 rsl_rl 提供，未读。

入口：`python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096`

```mermaid
sequenceDiagram
    autonumber
    actor U as 用户
    participant TR as train.py<br/>main() / TrainConfig
    participant REG as register_mjlab_task<br/>src/tasks/velocity/__init__.py
    participant CFG as env_cfgs.py / rl_cfg.py<br/>config/g1/
    participant LA as launch_training / run_train<br/>scripts/train.py
    participant ENV as ManagerBasedRlEnv<br/>mjlab（外部）
    participant WRAP as RslRlVecEnvWrapper<br/>mjlab.rl（外部）
    participant RUN as VelocityOnPolicyRunner<br/>src/tasks/velocity/rl/runner.py
    participant EXP as 日志目录<br/>logs/rsl_rl/<experiment>/
    U->>TR: python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096
    TR->>REG: import mjlab.tasks、src.tasks（执行注册）
    TR->>TR: tyro.cli 选择任务；TrainConfig.from_task → load_env_cfg / load_rl_cfg
    REG->>CFG: unitree_g1_flat_env_cfg() 与 unitree_g1_ppo_runner_cfg()
    TR->>LA: launch_training(task_id, args)
    LA->>EXP: 生成 {时间戳}_{run_name} 日志目录
    LA->>LA: select_gpus；设置 CUDA_VISIBLE_DEVICES、MUJOCO_GL=egl
    alt 单卡或 CPU
        LA->>LA: run_train(task_id, args, log_dir)
    else 多卡
        LA->>LA: torchrunx.Launcher.run(run_train)
    end
    LA->>ENV: ManagerBasedRlEnv(cfg, device, render_mode)
    LA->>WRAP: RslRlVecEnvWrapper(env, clip_actions)
    LA->>RUN: load_runner_cls(task) → VelocityOnPolicyRunner(env, agent_cfg, log_dir, device)
    LA->>EXP: add_git_repo_to_log；dump env.yaml / agent.yaml（仅 rank 0）
    LA->>RUN: learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)
    Note over RUN,ENV: rollout 与 PPO 更新在 rsl_rl 内部完成（未读源码）；每步经 WRAP 调用环境 step（mjlab 内部未读）
    RUN->>RUN: save(path)（checkpoint 保存时，推断由 rsl_rl 调用）
    RUN->>RUN: super().save → export_policy_to_onnx("policy.onnx")
    RUN->>EXP: attach_metadata_to_onnx(get_base_metadata(env))
    LA->>ENV: env.close()
```

图依据：unitree_rl_mjlab@1425b15，读过 `README.md`（训练与回放命令）、`scripts/train.py`（`main`、`launch_training`、`run_train`）、`src/tasks/velocity/__init__.py`（`register_mjlab_task`）、`src/tasks/velocity/rl/runner.py`（`VelocityOnPolicyRunner.save`）、`src/tasks/velocity/rl/__init__.py`。`src/tasks/velocity/config/g1/env_cfgs.py`、`rl_cfg.py` 仅由注册处引用，内部参数未读；mjlab、rsl_rl 内部未读。`save` 的调用方为推断。未运行代码。

## 关联页面

- [PPO（Proximal Policy Optimization）](../methods/ppo.md) — 速度跟踪与动作模仿任务共用训练框架时，需分别核对奖励、观测及 PPO 更新配置。

- [mjlab](./mjlab.md) — 底层框架（Isaac Lab API + MuJoCo Warp）
- [Unitree](./unitree.md) — 品牌主页
- [unitree_ros](./unitree-ros.md) — 官方 ROS1 / Gazebo URDF 与关节仿真栈（与本书 MuJoCo 主线对照）
- [Unitree G1](./unitree-g1.md) — 主要人形目标机型
- [AMP_mjlab](./amp-mjlab.md) — 同基于 mjlab 的社区 AMP 实现
- [legged_gym](./legged-gym.md) — 同类框架，基于 IsaacGym
- [Locomotion](../tasks/locomotion.md) — 任务方向
- [Query：视频里的舞蹈怎么搬到 G1？](../queries/g1-video-dance-gvhmr-gmr-beyondmimic-multi-motion.md) — 在 unitree_rl_mjlab 上训 G1 舞蹈追踪策略的完整链路与多动作扩展

## 参考来源

- [sources/repos/unitree_rl_mjlab.md](../../sources/repos/unitree_rl_mjlab.md)
- [unitreerobotics/unitree_rl_mjlab GitHub Repo](https://github.com/unitreerobotics/unitree_rl_mjlab)
- 源码核对（2026-11-21）：unitreerobotics/unitree_rl_mjlab@1425b15，读过 README.md、scripts/train.py、src/tasks/velocity/__init__.py、src/tasks/velocity/config/g1/__init__.py、src/tasks/velocity/rl/runner.py、src/tasks/velocity/rl/__init__.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [unitreerobotics/unitree_rl_mjlab 官方仓库](https://github.com/unitreerobotics/unitree_rl_mjlab)
- [mjlab 官方文档](https://mujocolab.github.io/mjlab/index.html)
- [BeyondMimic / whole_body_tracking 动作处理说明](https://github.com/HybridRobotics/whole_body_tracking/blob/main/README.md#motion-preprocessing--registry-setup)
