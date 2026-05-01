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
## 操作指南

### 1. 训练启动命令

该项目基于 `rsl_rl` 框架，使用 `mjlab` 进行并行仿真。

```bash
# 基础训练（G1 在平地上练习 AMP 风格行走与起身）
python scripts/train.py Unitree-G1-AMP-Flat --env.scene.num-envs=4096

# 崎岖地形训练（推荐用于鲁棒性测试）
python scripts/train.py Unitree-G1-AMP-Rough --env.scene.num-envs=4096
```

- `--env.scene.num-envs`：指定并行环境数量。在高端显卡上可设置更大（如 8192）以加速收敛。
- 任务列表可通过 `python scripts/list_envs.py --keyword AMP` 查看。

### 2. 训练监控与曲线分析

使用 `tensorboard --logdir logs/rsl_rl` 监控训练过程。

| 曲线名称 | 含义 | 理想走势 |
| :--- | :--- | :--- |
| **`rew_total`** | 总奖励，包含速度跟踪、高度保持和判别器奖励。 | 稳步上升，并在 2w iteration 附近出现**阶跃式跳变**。 |
| **`episode_length`** | 回合平均长度。 | 同样在 2w 步左右显著增加，标志着策略学会了“自我救赎”从而避免被 reset。 |
| **`disc_loss`** | AMP 判别器的损失函数。 | 初始较高，随后稳定在 **0.5 左右**。若过低（趋于 0）说明判别器太强，Policy 无法学习。 |
| **`disc_reward`** | 判别器给予 Policy 的奖励，衡量动作的自然度。 | 随着训练进行逐渐提高并震荡趋稳。 |

**关键判断标准**：如果在 2.5w iteration 后仍未出现 `episode_length` 的跳变，说明 Recovery 行为未被成功诱导，建议检查 Delayed Termination 参数或参考动作数据质量。

### 3. 模型导出

- **自动导出**：训练过程中，系统会周期性在 `logs/` 目录下生成 `model.onnx`。
- **验证并导出**：
  使用 `play.py` 可以可视化训练成果，并确保生成最新的 ONNX 模型：
  ```bash
  python scripts/play.py Unitree-G1-AMP-Flat --checkpoint-file logs/rsl_rl/g1_amp_locomotion/<RUN_DIR>/model_<ITER>.pt
  ```
  回放结束后，目录下会生成对应的 `model.onnx`。

### 4. 部署注意事项

在真机部署（如 Unitree G1）或集成到 `wbc_fsm` 时，需注意以下一致性问题：

- **mjlab 补丁**：必须确保 `mjlab` 安装了 `mjlab_patch/mjlab/managers/observation_manager.py`。该补丁修复了观测历史的排序问题，将默认的 "按项（term）排序" 改为 "按时间（time）排序"。
- **观测历史（History）**：策略期望 4 帧连续观测作为输入。如果部署端缓存的历史帧顺序不一致，会导致动作发疯。
- **动作缩放（Action Scaling）**：ONNX 导出时已包含 Normalizer，但输出的关节目标位置偏移仍需乘以训练配置中的 `action_scale`（通常为 0.25）。
- **FSM 切换**：虽然是统一策略，但在真机部署时，通常仍需通过 `wbc_fsm` 将其封装在 `MJAmpState` 下，以便管理安全急停和模式初速初始化。

## 源码阅读建议顺序
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
