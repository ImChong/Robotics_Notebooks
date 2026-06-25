---

type: entity
tags: [repo, framework, mujoco, mujoco-warp, isaac-lab-api, reinforcement-learning, gpu-simulation, nvidia, unitree]
status: complete
updated: 2026-06-25
related:
  - ./mujoco.md
  - ./mujoco-playground.md
  - ../overview/robot-training-stack-layers-technology-map.md
  - ./newton-physics.md
  - ./isaac-gym-isaac-lab.md
  - ./legged-gym.md
  - ./mjlab-playground.md
  - ./amp-mjlab.md
  - ./smp-g1-mjlab.md
  - ./unitree-rl-mjlab.md
  - ./asimov-v1.md
  - ./weights-and-biases.md
  - ./tensorboard.md
  - ../comparisons/wandb-vs-tensorboard.md
  - ../methods/reinforcement-learning.md
sources:
  - ../../sources/repos/mjlab.md
  - ../../sources/blogs/wechat_embodied_ai_lab_robot_training_stack_layers_2026.md
summary: "mjlab 将 Isaac Lab 的 manager-based 环境 API 与 MuJoCo Warp（GPU 加速物理）融合，是一个不依赖 Isaac Sim 的轻量 RL 框架，也是 AMP_mjlab 和 unitree_rl_mjlab 的底层依赖。"
---

# mjlab (轻量 GPU 加速 RL 框架)

**mjlab** 是由 mujocolab 开发的轻量机器人学习框架，核心设计是将 **Isaac Lab 的 manager-based API**（结构化环境设计）与 **MuJoCo Warp**（GPU 并行物理）结合，在不依赖 NVIDIA Isaac Sim 的前提下提供与 Isaac Lab 相似的开发体验。

有对应研究论文：Zakka et al. (2026) arXiv:2601.22074。

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
| 上层框架 | AMP_mjlab、[SMP on G1](./smp-g1-mjlab.md)、unitree_rl_mjlab、[asimov-mjlab](https://github.com/asimovinc/asimov-mjlab)（Asimov 官方 fork） | robot_lab | legged_gym 生态 | — |
| 引擎层对照 | [Newton Physics](./newton-physics.md) 为通用 Warp 物理引擎；mjlab 专注 **RL 环境 API**，二者可并存选型 | — | — | — |

## 关联页面

- [MuJoCo](./mujoco.md) — 物理内核（mjlab 使用 MuJoCo Warp）
- [Newton Physics](./newton-physics.md) — 同生态的底层 GPU 可微引擎（非 RL 框架封装）
- [Isaac Gym / Isaac Lab](./isaac-gym-isaac-lab.md) — API 设计来源
- [legged_gym](./legged-gym.md) — 同类框架，绑定 IsaacGym
- [AMP_mjlab](./amp-mjlab.md) — 以 mjlab 为底层的 AMP 统一策略实现
- [SMP on G1（mjlab）](./smp-g1-mjlab.md) — G1 上 score-matching 运动先验复现
- [unitree-rl-mjlab](./unitree-rl-mjlab.md) — Unitree 官方以 mjlab 为底层的训练框架
- [Asimov v1](./asimov-v1.md) — 硬件与 MuJoCo 主仓；行走 RL 公开在 asimov-mjlab fork
- [MuJoCo Playground](./mujoco-playground.md) — 上游任务参照；[mjlab_playground](./mjlab-playground.md) 端口
- [训练栈分层地图](../overview/robot-training-stack-layers-technology-map.md) — 任务与训练入口层
- [强化学习](../methods/reinforcement-learning.md) — 框架支持的学习范式

## 参考来源

- [sources/repos/mjlab.md](../../sources/repos/mjlab.md)
- [具身智能研究室：训练栈分层解读](../../sources/blogs/wechat_embodied_ai_lab_robot_training_stack_layers_2026.md)
- [mujocolab/mjlab GitHub Repo](https://github.com/mujocolab/mjlab)
- Zakka et al. (2026), arXiv:2601.22074
