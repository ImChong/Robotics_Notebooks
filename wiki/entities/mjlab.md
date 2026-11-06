---

type: entity
tags: [repo, framework, mujoco, mujoco-warp, isaac-lab-api, reinforcement-learning, gpu-simulation, nvidia, unitree]
status: complete
updated: 2026-09-21
related:
  - ./mujoco.md
  - ./mujoco-warp.md
  - ./mjviser.md
  - ./nvidia-warp.md
  - ./mujoco-playground.md
  - ../overview/robot-training-stack-layers-technology-map.md
  - ./newton-physics.md
  - ./isaac-gym-isaac-lab.md
  - ./isaac-lab.md
  - ./isaac-sim.md
  - ./legged-gym.md
  - ./mjlab-playground.md
  - ./amp-mjlab.md
  - ./open-beni.md
  - ./smp-g1-mjlab.md
  - ./unitree-rl-mjlab.md
  - ./booster-mjlab.md
  - ./telekinesis-rlbotics.md
  - ./pollen-microduck-rl.md
  - ./paper-pac-man-perceptive-cbf-rl.md
  - ./paper-yahmp.md
  - ./paper-smpc2rl-loco-manipulation.md
  - ./asimov-v1.md
  - ./weights-and-biases.md
  - ./tensorboard.md
  - ../comparisons/wandb-vs-tensorboard.md
  - ../methods/reinforcement-learning.md
original:
  - ../../sources/papers/mjlab_reading_arxiv_2601_22074_mjl1.md
sources:
  - ../../sources/repos/mjlab.md
  - ../../sources/repos/mjviser.md
  - ../../sources/blogs/wechat_embodied_ai_lab_robot_training_stack_layers_2026.md
summary: "mjlab 将 Isaac Lab 的 manager-based 环境 API 与 MuJoCo Warp（GPU 加速物理）融合，是一个不依赖 Isaac Sim 的轻量 RL 框架，也是 AMP_mjlab 和 unitree_rl_mjlab 的底层依赖。"
---

# mjlab (轻量 GPU 加速 RL 框架)

**mjlab** 是由 mujocolab 开发的轻量机器人学习框架，核心设计是将 **Isaac Lab 的 manager-based API**（结构化环境设计）与 [**MuJoCo Warp**](./mujoco-warp.md)（GPU 并行物理）结合，在不依赖 NVIDIA Isaac Sim 的前提下提供与 Isaac Lab 相似的开发体验。

有对应研究论文：Zakka et al. (2026) arXiv:2601.22074。

## 一句话定义

mjlab 是把 Isaac Lab 的 manager-based 环境 API 与 MuJoCo Warp GPU 物理合在一起、单命令安装的轻量机器人 RL 框架。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Isaac Lab | NVIDIA Isaac Lab | 基于 Omniverse 的机器人学习训练框架 |
| API | Application Programming Interface | 应用程序编程接口 |
| MuJoCo | Multi-Joint dynamics with Contact | 接触丰富的刚体物理仿真引擎 |
| GPU | Graphics Processing Unit | 图形处理器，大规模并行仿真训练的算力基础 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |
| legged_gym | Legged Gym | 足式机器人 RL 训练的常用开源框架 |
| Isaac Gym | NVIDIA Isaac Gym | GPU 并行刚体仿真训练环境 |
| SMP | Score-Matching Motion Prior | 可复用的 score-matching 运动先验模块 |
| G1 | Unitree G1 Humanoid | 宇树入门级教育科研人形平台 |

## 为什么重要？

Isaac Lab API 是当前 RL 机器人训练的优秀抽象，但它绑定了 Isaac Sim——部署重、商业授权复杂。mjlab 将这套 API 移植到 MuJoCo Warp 之上：

- 保留熟悉的任务管理器 / 奖励管理器 / 观测管理器接口
- 替换底层为开源、轻量的 MuJoCo Warp
- 安装极简（PyPI 包 + `uv sync`）
- 对 macOS 提供有限评估支持（无 GPU 训练）

策展解读（[训练栈分层地图](../overview/robot-training-stack-layers-technology-map.md)）强调：mjlab 的价值不止于 GPU 吞吐，更在于 **manager-based 模块化**——observation、reward、event、command、curriculum 可组合复用，使多任务、多本体长期迭代时 **环境定义本身成为基础设施**，而非一次性巨型脚本。

## 架构

```
mjlab 架构
├── API 层（Isaac Lab manager-based 设计）
│   ├── SceneManager       # 场景/资产管理
│   ├── ActionManager      # 动作空间
│   ├── ObservationManager # 观测空间
│   └── RewardManager      # 奖励函数组合
└── 物理层
    └── MuJoCo Warp        # GPU 并行刚体仿真
```

## 核心原理

据技术报告（Zakka et al., 2026）：

- **场景管线：** MJCF 实体合成为 MjSpec，CPU 编译为 MjModel，再经 `put_model` / `put_data` 送入 MuJoCo Warp；MjData 带前导 world 维，一个对象存 N 个并行世界，参数默认共享、随机化时按字段扩成 per-world；仿真步以 CUDA graph 捕获重放以省去 CPU 派发开销。
- **step 流水线：** Act → Simulate（decimation 子步）→ Terminate → Reward → Reset → Command → Apply Events → Observe；由 action / termination / reward / curriculum / event / command / observation / metrics 各 manager 编排。
- **设计取舍：** 只支持 MuJoCo Warp 单一后端（跨模拟器可移植为非目标）；单一 Entity 类取代 Isaac Lab 的多个资产类；执行器分层（MJCF 原生、MuJoCo 原生、GPU 上的 PD / DC 电机 / 学习式 MLP 执行器，另有延迟包装）。

## 工程实践

- 训练入口：`uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 4096 --agent.max-iterations 10_000`；on-policy 算法用 RSL-RL，多卡用 torchrunx。
- 配置为实例式 typed dict，经 tyro 自动暴露为命令行参数；创建任务变体是复制并修改现有配置，而非继承新类。
- TorchArray 零拷贝把 Warp 数组当 PyTorch 张量用，奖励 / 观测可写纯 PyTorch。
- 自带三个参考任务：G1 / Go1 速度跟踪、G1 动作模仿（DeepMimic + BeyondMimic 扩展）、YAM 机械臂抬方块。
- 调试：终止管理器检测 NaN/Inf 并保留滚动状态缓冲供回放；域随机化的惯量采用伪惯量参数化以保持物理一致。

## 局限

- 仅面向刚体机器人学习，不做照片级渲染；视觉策略建议先训特权策略再用外部渲染蒸馏（报告原话方向）。
- 不追求跨模拟器可移植；物理能力受限于 MuJoCo Warp（缺口见 [MuJoCo Warp](./mujoco-warp.md)）。
- 技术报告不含吞吐或与 Isaac Lab 的定量对比数据，性能结论需自行实测（本页不给出具体加速倍数）。
- manager 抽象相对单体 step 函数有额外开销，作者认为模块化收益更大（报告自述）。

## 核心能力

| 能力 | 说明 |
|------|------|
| 速度跟踪 | humanoid / 四足速度指令跟踪，flat / rough terrain |
| 动作模仿 | 参考运动数据驱动的 motion imitation |
| 多 GPU 训练 | 分布式 multi-GPU scaling |
| 实验追踪 | 集成 [Weights & Biases](./weights-and-biases.md)（训练期标量本地对照可用 [TensorBoard](./tensorboard.md)） |
| Dummy agent | 零动作 / 随机动作快速 sanity check |

## 与其他仿真框架的定位

| 维度 | mjlab | Isaac Lab | legged_gym | Genesis |
|------|-------|-----------|------------|---------|
| 物理后端 | MuJoCo Warp | Isaac Sim (PhysX) | Isaac Gym | 自研多物理 |
| API 风格 | Isaac Lab（移植） | 原生 Isaac Lab | 简单脚本 | Pythonic |
| 依赖重量 | 轻（pip 安装） | 重（Isaac Sim） | 重（IsaacGym） | 中 |
| 开源授权 | Apache 2.0 | BSD-3 | BSD-3 | Apache 2.0 |
| 上层框架 | AMP_mjlab、[SMP on G1](./smp-g1-mjlab.md)、unitree_rl_mjlab、[booster_mjlab](./booster-mjlab.md)、[asimov-mjlab](https://github.com/asimovinc/asimov-mjlab)（Asimov 官方 fork）、[Microduck RL](./pollen-microduck-rl.md) | robot_lab | legged_gym 生态 | — |
| 引擎层对照 | [Newton Physics](./newton-physics.md) 为通用 Warp 物理引擎；mjlab 专注 **RL 环境 API**，二者可并存选型 | — | — | — |

## 关联页面

- [Open-BENI](./open-beni.md) — BENI 策略训练与浏览器端 MuJoCo + ONNX 演示
- [MuJoCo](./mujoco.md) — 物理内核（mjlab 使用 [MuJoCo Warp](./mujoco-warp.md)）
- [MuJoCo Warp](./mujoco-warp.md) — 实际 GPU 刚体后端；PGS/PLUGIN 缺口与 AD 未通见该页
- [mjviser](./mjviser.md) — 基于 Viser 的 MuJoCo 浏览器查看器；负责可视化与交互，不是训练框架
- [NVIDIA Warp](./nvidia-warp.md) — MJWarp 的 JIT 计算层
- [Newton Physics](./newton-physics.md) — 同生态的底层多求解器引擎（非 RL 框架封装）
- [Isaac Lab](./isaac-lab.md) — API 设计来源；底层依赖 [Isaac Sim](./isaac-sim.md)
- [Isaac Gym / Isaac Sim / Isaac Lab](./isaac-gym-isaac-lab.md) — 三代产品总览
- [legged_gym](./legged-gym.md) — 同类框架，绑定 IsaacGym
- [AMP_mjlab](./amp-mjlab.md) — 以 mjlab 为底层的 AMP 统一策略实现
- [senlanke/mimic（mjlab）](./smp-g1-mjlab.md) — G1 上 SMP / CMoE / AME 移植枢纽
- [unitree-rl-mjlab](./unitree-rl-mjlab.md) — Unitree 官方以 mjlab 为底层的训练框架
- [booster_mjlab](./booster-mjlab.md) — whIRLwind 的 Booster K1 AMP / motion tracking 集成
- [Telekinesis RLbotics](./telekinesis-rlbotics.md) — 跨 Gymnasium/mjlab/Isaac Lab 的 YAML 驱动 PPO + ONNX 部署层
- [Microduck RL](./pollen-microduck-rl.md) — Pollen 桌面双足：BAM XL330 + 共享 61D 观测 + ONNX 热切换
- [YAHMP](./paper-yahmp.md) — G1 全身 GMT 消融与 ONNX 部署（基于 mjlab）
- [SMPC-to-RL](./paper-smpc2rl-loco-manipulation.md) — mjlab + MuJoCo Warp 上 tiled SMPC 采数与稀疏 FastTD3（代码未开源）
- [PAC-MAN](./paper-pac-man-perceptive-cbf-rl.md) — mjlab 上感知感知 CBF-RL 躲避球（全栈开源）
- [Asimov v1](./asimov-v1.md) — 硬件与 MuJoCo 主仓；行走 RL 见 asimov-mjlab（mjlab）与 [isaac_asimov](./isaac-asimov.md)（Isaac Lab）
- [MuJoCo Playground](./mujoco-playground.md) — 上游任务参照；[mjlab_playground](./mjlab-playground.md) 端口
- [训练栈分层地图](../overview/robot-training-stack-layers-technology-map.md) — 任务与训练入口层
- [强化学习](../methods/reinforcement-learning.md) — 框架支持的学习范式

## 源码运行时序图

入口：`uv run train <TaskId>`（`pyproject.toml` 的 `[project.scripts]` 把 `train` 映射到 `mjlab.scripts.train:main`）

```mermaid
sequenceDiagram
    participant CLI as scripts/train.py main
    participant REG as tasks/registry.py
    participant ENV as envs/manager_based_rl_env.py ManagerBasedRlEnv
    participant WRP as rl/vecenv_wrapper.py RslRlVecEnvWrapper
    participant RUN as rl/runner.py MjlabOnPolicyRunner
    participant RSL as rsl_rl OnPolicyRunner 外部依赖

    CLI->>REG: list_tasks / load_env_cfg / load_rl_cfg（tyro 解析 TrainConfig）
    CLI->>CLI: launch_training: select_gpus, 设置 CUDA_VISIBLE_DEVICES
    alt 单卡或 CPU
        CLI->>CLI: run_train(task_id, cfg, log_dir)
    else 多卡
        CLI->>CLI: torchrunx.Launcher.run(run_train)
    end
    CLI->>ENV: ManagerBasedRlEnv(cfg.env, device)
    Note over ENV: 构造 Scene、Simulation 与各 manager
    opt 录像且 rank 0
        CLI->>CLI: VideoRecorder 包裹 env
    end
    CLI->>WRP: RslRlVecEnvWrapper(env, clip_actions)
    CLI->>REG: load_runner_cls(task_id)，缺省用 MjlabOnPolicyRunner
    CLI->>RUN: runner_cls(env, agent_cfg, log_dir, device)
    opt 配置了 resume
        CLI->>RUN: runner.load(checkpoint)
    end
    CLI->>RUN: runner.learn(max_iterations, init_at_random_ep_len=True)
    Note over RUN,RSL: learn 的迭代调度继承自 rsl_rl（未在本仓库内读取，此处为推断）
    loop 每个训练迭代
        RSL->>WRP: step(actions)
        WRP->>ENV: env.step(actions)
        ENV->>ENV: action_manager.process_action
        loop decimation 次物理子步
            ENV->>ENV: action_manager.apply_action / scene.write_data_to_sim
            ENV->>ENV: sim.step() / scene.update
        end
        ENV->>ENV: termination_manager.compute / reward_manager.compute
        ENV->>ENV: event_manager 的 step 与 interval 事件
        opt 环境终止或超时（auto_reset）
            ENV->>ENV: _reset_idx 并写回场景
        end
        ENV->>ENV: sim.forward / command_manager.compute / sim.sense
        ENV->>ENV: observation_manager.compute
        ENV-->>WRP: obs, reward, terminated, truncated, extras
        WRP-->>RSL: 返回 rsl_rl 格式的向量化结果
    end
```

图依据：mujocolab/mjlab@dcdf6ce，读过 `pyproject.toml`、`README.md`、`src/mjlab/scripts/train.py`、`src/mjlab/scripts/play.py`、`src/mjlab/envs/manager_based_rl_env.py`（`step`）、`src/mjlab/rl/vecenv_wrapper.py`（`step`）、`src/mjlab/rl/runner.py`（类与方法签名）、`src/mjlab/tasks/registry.py`（函数签名）；未运行代码。

## 继续阅读

- [mjlab 技术报告通读归档](../../sources/papers/mjlab_reading_arxiv_2601_22074_mjl1.md) — 架构、manager 流水线与软件设计摘录
- [MuJoCo Playground](./mujoco-playground.md) — 单体式环境的对照路线

## 参考来源

- [sources/repos/mjlab.md](../../sources/repos/mjlab.md)
- [具身智能研究室：训练栈分层解读](../../sources/blogs/wechat_embodied_ai_lab_robot_training_stack_layers_2026.md)
- [mujocolab/mjlab GitHub Repo](https://github.com/mujocolab/mjlab)
- [mjviser 来源归档](../../sources/repos/mjviser.md)
- Zakka et al. (2026), arXiv:2601.22074
- 源码核对（2026-11-02）：mujocolab/mjlab@dcdf6ce，读过 `pyproject.toml`、`README.md`、`src/mjlab/scripts/train.py`、`src/mjlab/scripts/play.py`、`src/mjlab/envs/manager_based_rl_env.py`、`src/mjlab/rl/vecenv_wrapper.py`、`src/mjlab/rl/runner.py`、`src/mjlab/tasks/registry.py`；时序图依据以上文件，未运行代码。
