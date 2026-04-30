---
type: entity
tags: [repo, amp, imitation-learning, mjlab, rsl-rl, unitree, humanoid, locomotion, recovery]
status: complete
updated: 2026-05-01
related:
  - ../methods/amp-reward.md
  - ./mjlab.md
  - ./unitree-g1.md
  - ./legged-gym.md
  - ../methods/imitation-learning.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/repos/amp_mjlab.md
summary: "AMP_mjlab 是基于 mjlab + rsl_rl 的 Unitree G1 统一 AMP 策略实现，用单一 actor-critic + 判别器同时覆盖 locomotion 与 fall-recovery，消除模式切换断裂。"
---

# AMP_mjlab (G1 统一 AMP 策略)

**AMP_mjlab** 是一个针对 **Unitree G1** 人形机器人的强化学习训练框架，建立在 **mjlab**（MuJoCo 并行仿真）和 **rsl_rl**（RSL PPO 训练库）之上，核心贡献在于用一个统一策略同时学习正常行走（locomotion）与跌倒恢复（fall-recovery）。

## 为什么重要？

传统做法需要维护独立的 "locomotion 策略" 和 "recovery 策略"，并在运行时检测跌倒再触发切换，模式切换时易产生动作撕裂（behavioral discontinuity）。AMP_mjlab 的统一策略消除了这个切换逻辑，同时 AMP 判别器保证了动作的自然风格。

## 从零理解 AMP_mjlab

下面用**从零理解**的方式讲解这个项目。

### 1. 这个项目一句话是什么？

`AMP_mjlab` 是一个基于 **mjlab + rsl_rl + AMP** 的 Unitree G1 人形机器人强化学习项目。目标是训练一个 policy，让 G1 同时学会：
1. **走路 / 跑步 / 跟踪速度指令**
2. **跌倒后的恢复 / 起身**
3. **动作风格尽量接近参考动作数据**

它不是单独训练“走路策略”和“起身策略”，而是把两种能力放进**同一个 policy** 里训练，减少策略切换带来的不连续问题。([GitHub][1])

### 2. AMP 是什么？先用直觉理解

AMP = **Adversarial Motion Priors**，可以简单理解成：**让机器人一边完成任务，一边模仿人类/参考动作的风格。**

它有两个网络：
- **Policy / Actor-Critic**：负责真正控制机器人。输入机器人状态 + 速度指令，输出关节动作。
- **AMP Discriminator**：像一个“动作裁判”。它会看真实参考动作数据（如走、跑、起身）和 policy 生成的动作，判断“像不像真实动作”。如果像，Policy 就会得到额外奖励。

### 3. 仓库核心结构

```text
AMP_mjlab/
├── scripts/
│   ├── train.py          # 训练入口
│   ├── play.py           # 回放 / 测试 / 导出 ONNX
│   ├── list_envs.py      # 查看可用任务
│   └── csv_to_npz.py     # 动作数据 CSV 转 NPZ
├── src/
│   ├── assets/
│   │   ├── robots/       # G1 机器人模型配置
│   │   └── motions/g1/amp/
│   │       ├── WalkandRun # 走/跑参考动作
│   │       └── Recovery   # 起身/恢复参考动作
│   └── tasks/
│       └── amp_loco/
│           ├── amp_env_cfg.py       # 环境、观测、动作、奖励配置
│           ├── ampmotion_loader.py  # 加载参考动作数据
│           ├── config/g1/
│           │   ├── env_cfgs.py      # G1 Flat / Rough 环境配置
│           │   ├── rl_cfg.py        # PPO + AMP 训练参数
│           └── mdp/
│               ├── observations.py  # 观测项
│               ├── rewards.py       # 奖励项
│               ├── events.py        # reset / 推扰 / 随机化
│               └── terminations.py  # 终止条件
```

### 4. 模型输入与输出

#### 输入 (Actor Observation)
Policy 真正用的输入包含 4 帧历史，单帧约 96 维（384 维合计）：
- 机身角速度、重力方向、速度指令
- 关节位置、关节速度（G1 共 29 个关节）
- 上一次输出动作

#### 输出 (Action)
输出是 29 维的**关节位置控制目标**（`JointPositionActionCfg`），经 scale 后转换为目标位置偏移，由底层 PD 负责跟踪。([GitHub][2])

### 5. 奖励函数与判别器

- **AMP 判别器**：观察 Pelvis、髋、膝、踝、肩、肘、腕等关键部位的相对位置、姿态和速度，判断是否符合参考动作风格。([GitHub][4])
- **奖励组合**：
    - 速度跟踪：`track_anchor_linear_velocity` / `angular_velocity`
    - 恢复奖励：`track_root_height`（鼓励回到正常高度）
    - 惩罚项：身体乱晃、关节加速度、打滑、自碰撞、撞限位等。

### 6. 训练与回放

- **开始训练**：`python scripts/train.py Unitree-G1-AMP-Flat --env.scene.num-envs=4096`
- **收敛特征**：约 20k iterations 附近，recovery 行为可能突然涌现，指标跳变属正常。([GitHub][1])
- **回放测试**：`python scripts/play.py Unitree-G1-AMP-Flat --checkpoint-file logs/rsl_rl/.../model_<iter>.pt`
- **ONNX 导出**：训练和回放默认启用导出。ONNX 输入名为 `obs`，输出名为 `actions`，且内置了 normalizer。([GitHub][8])

### 7. 源码阅读建议顺序

1. `README.md`
2. `src/tasks/amp_loco/config/g1/__init__.py`
3. `src/tasks/amp_loco/config/g1/env_cfgs.py`
4. `src/tasks/amp_loco/amp_env_cfg.py`
5. `src/tasks/amp_loco/config/g1/rl_cfg.py`
6. `scripts/train.py` & `play.py`

## 核心架构

```
AMP_mjlab 统一训练
├── Actor-Critic 网络（单一策略）
│   ├── 速度跟踪奖励（locomotion 目标）
│   └── AMP 风格奖励（自然度）
├── AMP Discriminator
│   ├── 参考数据：WalkandRun clip
│   └── 参考数据：Recovery clip（跌倒恢复）
└── Delayed Termination
    └── 部分 env 在 reset 前给 recovery 窗口
```

**关键机制：Delayed Termination**——不立即 reset 跌倒的 env，而是给策略一个时间窗口尝试自主恢复，迫使策略学习爬起行为。

## 训练特征

- **规模**：4096 并行环境
- **收敛特征**：约 2 万步时 recovery 行为突然涌现，loss 指标跳变属正常
- **任务**：`Unitree-G1-AMP-Rough` / `Unitree-G1-AMP-Flat`
- **部署**：ONNX export，训练与推理 pipeline 一致

## 与 AMP 方法的关系

AMP_mjlab 是 [AMP & HumanX](../methods/amp-reward.md) 方法的一个具体实现，区别在于：

| 维度 | AMP 原论文 | AMP_mjlab |
|------|-----------|-----------|
| 硬件目标 | 角色控制（通用） | Unitree G1（人形） |
| 仿真框架 | IsaacGym / PhysX | mjlab（MuJoCo） |
| 任务范围 | 单一风格 | Locomotion + Recovery 统一 |
| 训练库 | 各异 | rsl_rl (PPO) |

## 关联页面

- [AMP & HumanX 方法](../methods/amp-reward.md) — AMP 方法本体
- [mjlab](./mjlab.md) — 底层仿真框架（Isaac Lab API + MuJoCo Warp）
- [Unitree G1](./unitree-g1.md) — 目标硬件
- [legged_gym](./legged-gym.md) — 同为 rsl_rl + 并行仿真，基于 IsaacGym
- [Imitation Learning](../methods/imitation-learning.md) — AMP 属于模仿学习范式
- [Locomotion](../tasks/locomotion.md) — 任务方向

## 参考来源

- [sources/repos/amp_mjlab.md](../../sources/repos/amp_mjlab.md)
- [ccrpRepo/AMP_mjlab GitHub Repo](https://github.com/ccrpRepo/AMP_mjlab)

[1]: https://github.com/ccrpRepo/AMP_mjlab "GitHub - ccrpRepo/AMP_mjlab · GitHub"
[2]: https://raw.githubusercontent.com/ccrpRepo/AMP_mjlab/main/src/tasks/amp_loco/amp_env_cfg.py "amp_env_cfg.py"
[3]: https://raw.githubusercontent.com/ccrpRepo/AMP_mjlab/main/scripts/csv_to_npz.py "csv_to_npz.py"
[4]: https://raw.githubusercontent.com/ccrpRepo/AMP_mjlab/main/src/tasks/amp_loco/config/g1/rl_cfg.py "rl_cfg.py"
[5]: https://raw.githubusercontent.com/ccrpRepo/AMP_mjlab/main/src/tasks/amp_loco/ampmotion_loader.py "ampmotion_loader.py"
[6]: https://raw.githubusercontent.com/ccrpRepo/AMP_mjlab/main/README_zh.md "README_zh.md"
[7]: https://raw.githubusercontent.com/ccrpRepo/AMP_mjlab/main/scripts/train.py "train.py"
[8]: https://raw.githubusercontent.com/ccrpRepo/AMP_mjlab/main/src/tasks/amp_loco/rl/runner.py "runner.py"
[9]: https://raw.githubusercontent.com/ccrpRepo/AMP_mjlab/main/src/tasks/amp_loco/config/g1/__init__.py "__init__.py"
[10]: https://raw.githubusercontent.com/ccrpRepo/AMP_mjlab/main/src/tasks/amp_loco/config/g1/env_cfgs.py "env_cfgs.py"
