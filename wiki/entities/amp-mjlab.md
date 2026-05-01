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
  - ./wbc-fsm.md
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
# 安装后先确认任务已注册
python scripts/list_envs.py --keyword AMP

# 基础训练（G1 在平地上练习 AMP 风格行走与起身）
python scripts/train.py Unitree-G1-AMP-Flat --env.scene.num-envs=4096

# 崎岖地形训练（推荐用于鲁棒性测试）
python scripts/train.py Unitree-G1-AMP-Rough --env.scene.num-envs=4096

# 多卡机器上可显式选择 GPU；脚本会通过 torchrunx 启动多进程
python scripts/train.py Unitree-G1-AMP-Flat --env.scene.num-envs=8192 --gpu-ids 0 1
```

- `--env.scene.num-envs`：指定并行环境数量。单卡常用 4096；显存和仿真吞吐足够时可提高到 8192。
- `--gpu-ids`：训练脚本会根据该参数设置 `CUDA_VISIBLE_DEVICES`；多卡时用 `torchrunx` 分布式启动。
- `--video`、`--video-interval`、`--video-length`：训练中录制 rollouts，用于排查“奖励上升但动作异常”的情况。
- 任务列表可通过 `python scripts/list_envs.py --keyword AMP` 查看。
- 日志默认写入 `logs/rsl_rl/g1_amp_locomotion/<time_stamp_run>/`，同时保存 `params/env.yaml` 与 `params/agent.yaml` 以便复现实验配置。

### 2. 训练监控与曲线分析

使用 `tensorboard --logdir logs/rsl_rl` 监控训练过程。

| 曲线 / 指标 | 含义 | 好走势 | 异常信号 |
| :--- | :--- | :--- |
| `Train/mean_reward`、`rew_total` 或 episode reward | 平均总奖励，混合了速度跟踪、高度恢复、AMP 风格奖励与惩罚项。 | 前期有噪声但总体上升；约 20k iterations 附近可能出现阶跃式上升。 | 长时间横盘或下降，通常说明动作先验、reset 分布或奖励权重没有形成有效课程。 |
| `Train/mean_episode_length` 或 `episode_length` | 平均回合长度，反映策略是否能避免 bad orientation / bad base height 终止。 | 与总奖励同步上升；在学会 fall-recovery 后会明显变长。 | 2.5w iterations 后仍很短，通常表示 recovery 没被诱导出来。 |
| `Episode/rew_track_anchor_linear_velocity` | 躯干/anchor 线速度跟踪奖励，对应 x/y 方向速度指令。 | 逐步升高并趋稳，说明 locomotion 能跟上指令。 | 高度恢复很好但该项低，可能是策略偏向“站稳/起身”而不会走。 |
| `Episode/rew_track_anchor_angular_velocity` | yaw 角速度跟踪奖励。 | 随着前进速度跟踪一起改善，不应长期接近 0。 | 转向时摔倒或该项长期低，需检查 command range、摩擦随机化和足端打滑。 |
| `Episode/rew_track_root_height` | root 高度恢复奖励，是 recovery 是否站回正常高度的重要信号。 | recovery 涌现阶段通常会明显跳升。 | 高度奖励低且 episode length 短，说明起身过程失败或 reset 太早。 |
| `Episode/rew_is_terminated` | 终止惩罚，触发 bad orientation / bad base height 时会显著拉低总奖励。 | 绝对惩罚占比下降，终止次数减少。 | 惩罚长期很大，说明策略经常摔倒或 delayed reset 窗口不足。 |
| `Episode/rew_joint_acc_l2`、`rew_action_rate_l2` | 关节加速度与动作变化惩罚，衡量动作是否抖动。 | 总体保持较低，随策略收敛更平滑。 | 奖励高但这些惩罚同步变大，部署时容易出现电机发热、冲击或高频震荡。 |
| `Episode/rew_foot_slip`、`rew_self_collisions`、`rew_joint_pos_limits` | 足端打滑、自碰撞、关节限位惩罚。 | 负项逐渐减小，偶发尖峰可以接受。 | 长期大幅负值，说明 gait 不物理、地面接触不稳或姿态解空间被关节限位卡住。 |
| `Loss/value_function`、`Loss/surrogate` | PPO critic value loss 与 policy surrogate loss。 | 有波动但不持续发散；学习率自适应后应保持在可训练区间。 | value loss 持续爆炸或 surrogate 大幅震荡，优先降低学习率、环境数或检查 reward scale。 |
| `Policy/mean_noise_std` | 高斯策略的平均探索标准差。 | 从初始化值逐步下降，但不要过早塌到很低。 | 过早接近最小标准差会导致探索不足，recovery 很难涌现。 |
| AMP 判别器 loss / `disc_loss` / `amp_loss` | 判别器区分参考 motion 与 policy motion 的训练损失。 | 初期快速变化，随后进入对抗平衡；不追求单调下降。 | 判别器过强时 policy 获得不了风格奖励；过弱时动作会“骗过”判别器但不自然。 |
| AMP 风格奖励 / `disc_reward` / `amp_reward` | 判别器给 policy 的动作风格奖励。 | 随训练提高并震荡趋稳，和任务奖励共同上升最好。 | 风格奖励高但速度/高度奖励低，说明只学到 motion prior 外观，任务目标没有完成。 |
| `Metrics/mean_action_acc` | 环境自定义动作加速度指标。 | 越低越适合部署，但不能以牺牲速度跟踪为代价。 | 回放中出现抖腿、膝关节抽动时通常会同步升高。 |
| `Metrics/mean_delay_steps` | delayed termination / recovery 窗口相关指标，反映环境在终止后延迟 reset 的步数。 | recovery 学会后不应持续堆高。 | 长期很高且 episode length 不提升，说明环境给了恢复窗口但策略没恢复成功。 |

**关键判断标准**：README 明确提示约 `2w` iterations 附近可能出现多个指标突变，这是 recovery 行为突然涌现的正常现象。判断训练好坏时不要只看总奖励，至少同时看 `episode_length`、`track_root_height`、速度跟踪奖励、终止惩罚和动作平滑指标；只有“能站起、能跟踪速度、动作不抖、终止减少”同时成立，才适合进入部署验证。

### 3. 模型导出

- **训练自动导出**：AMP runner 覆盖了 `save()`；每次保存 `model_<ITER>.pt` 后，会在同一 run 目录导出 `policy.onnx`，并附加环境 metadata。
- **回放验证并导出**：`play.py` 的 `export_onnx=True` 默认开启，会把指定 checkpoint 导出为 `export/<task>_<checkpoint>.onnx`。
  ```bash
  python scripts/play.py Unitree-G1-AMP-Flat \
    --checkpoint-file logs/rsl_rl/g1_amp_locomotion/<RUN_DIR>/model_<ITER>.pt
  ```
- **导出模型接口**：ONNX 输入名为 `obs`，输出名为 `actions`；batch 维是动态轴；导出 wrapper 已包含 empirical observation normalizer，因此部署端应输入训练同分布的 raw observation，而不是再手动做一遍同样的归一化。
- **选择 checkpoint**：优先选择 `episode_length`、速度跟踪、root height 和动作平滑指标同时稳定后的 checkpoint，不要只选总奖励最高的瞬间点。

### 4. 部署注意事项

在真机部署（如 Unitree G1）或集成到 `wbc_fsm` 时，需注意以下一致性问题：

- **mjlab 补丁**：训练端使用 `history_ordering="time"`；必须确保 `mjlab_patch/mjlab/managers/observation_manager.py` 已覆盖到当前环境。如果不打补丁，则需要从代码里移除 `history_ordering` 配置，否则训练/部署观测顺序会不一致。
- **观测历史（History）**：actor 输入是 4 帧历史，单帧包含 base angular velocity、projected gravity、velocity command、joint position、joint velocity、last action。部署端缓存顺序必须是按时间展开，且 last action 要使用上一控制周期真正下发的动作。
- **归一化边界**：ONNX 已包含 obs normalizer；C++/部署端不要重复套训练时的 normalizer，但要保证传入的物理量单位、关节顺序、符号方向与训练配置一致。
- **动作语义**：policy 输出 `actions`，训练环境中由 `JointPositionActionCfg(scale=0.25, use_default_offset=True)` 转成关节位置目标偏移；部署端必须复现“默认关节位置 + 0.25 * action”的语义，再交给底层 PD / WBC。
- **控制频率**：仿真 `timestep=0.005`、`decimation=4`，策略控制周期约 20 ms（50 Hz）。真机部署时推理频率、状态估计延迟和低层控制频率要与这一节奏匹配。
- **安全壳**：统一策略减少了 locomotion/recovery 的模式切换，但真机仍应放在 `wbc_fsm` 的 `MJAmpState` 之类状态内，外层保留急停、限幅、姿态保护、接触异常处理和低电量/过热保护。
- **上线顺序**：先用 `play.py` 回放目标 checkpoint，再做仿真扰动和 rough terrain 验证；上真机时先限速、限幅、吊挂或保护架测试，逐步开放速度 command 与 recovery 场景。

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
