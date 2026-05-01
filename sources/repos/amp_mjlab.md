# AMP_mjlab

> 来源归档

- **标题：** AMP_mjlab
- **类型：** repo
- **来源：** ccrpRepo（GitHub 个人项目）
- **链接：** https://github.com/ccrpRepo/AMP_mjlab
- **入库日期：** 2026-04-29
- **一句话说明：** 基于 mjlab + rsl_rl 的 Unitree G1 统一 AMP 策略，用单个 actor-critic + 判别器同时学习 locomotion 与 fall-recovery，消除模式切换的行为断裂。
- **沉淀到 wiki：** 是 → [`wiki/entities/amp-mjlab.md`](../../wiki/entities/amp-mjlab.md)

---

## 核心定位

AMP_mjlab 在足式机器人 RL 中解决了一个实际问题：传统做法需要为 "正常行走" 和 "跌倒恢复" 各维护一个独立策略，切换时容易出现动作撕裂。本项目用 AMP (Adversarial Motion Priors) 的判别器奖励驱动单一策略同时覆盖两类行为。

依赖栈：
- **mjlab**：基于 MuJoCo 的并行仿真框架
- **rsl_rl**：ETH RSL 的轻量 RL 训练库（PPO）
- Python 3.11，Linux，GPU

---

## 技术实现要点

### 1. 训练命令与任务

- **主训练命令**：
  ```bash
  python scripts/train.py Unitree-G1-AMP-Flat --env.scene.num-envs=4096
  ```
- **任务列表**：
  - `Unitree-G1-AMP-Flat`：平地任务
  - `Unitree-G1-AMP-Rough`：崎岖地形任务
- **数据准备**：使用 `scripts/csv_to_npz.py` 将 CSV 动作数据转换为判别器所需的 NPZ 格式。

### 2. 核心指标与曲线含义

在 `rsl_rl` 记录的 Tensorboard 中，需重点关注：
- `rew_total`：总奖励。在 20k iterations 左右会出现陡增。
- `episode_length`：回合长度。同样会在 20k 左右跳变，标志着机器人学会了在跌倒后不 reset 而是自主爬起。
- `disc_loss`：判别器损失。应逐渐下降并维持在 0.5 左右（对抗平衡）。
- `disc_reward`：判别器给出的风格奖励。值越高说明动作越接近参考数据。

**走势判断**：2w iteration 的指标跳变（"The Recovery Jump"）是成功的关键信号，不应视为不稳定。

### 3. 统一 AMP 训练

| 组件 | 说明 |
|------|------|
| Actor-Critic | 单一网络，统一处理 locomotion 与 recovery |
| AMP Discriminator | 区分"参考数据动作"与"策略生成动作"，驱动自然度 |
| 动作数据集 | `WalkandRun`（步行/跑步）+ `Recovery`（跌倒恢复）两套 clip |
| Delayed Termination | 部分环境在 reset 前给予 recovery 窗口，逼迫策略学习爬起 |

### 4. 导出与部署

- **自动导出**：`train.py` 和 `play.py` 默认会导出 `model.onnx`。
- **手动验证与导出**：
  ```bash
  python scripts/play.py Unitree-G1-AMP-Flat --checkpoint-file logs/rsl_rl/.../model_<iter>.pt
  ```
- **部署要点**：
  - **补丁安装**：必须应用 `mjlab_patch/` 下的 `observation_manager.py` 修改，确保观测历史按时间顺序（time）而非 term 排序。
  - **FSM 集成**：部署逻辑见 `ccrpRepo/wbc_fsm` 中的 `MJAmp State`。
  - **观测一致性**：输入为 4 帧历史（包含角速度、重力矢量、指令、关节状态等）。

---

## 目录结构

```
AMP_mjlab/
├── src/tasks/amp_loco/
│   ├── config/g1/       # 任务注册与配置
│   └── mdp/             # 奖励、观测、终止条件
├── mjlab_patch/         # 必要的 mjlab 补丁（观测历史顺序修复）
├── motion_data_csv/amp/ # 原始动作数据（CSV 格式）
├── scripts/             # 训练、评估、数据转换工具
└── rsl_rl/              # RL 框架组件
```

---

## 与本项目其他资料的关系

| 资料 | 关系 |
|------|------|
| [amp-reward.md](../../wiki/entities/amp-reward.md) | AMP 方法本体，本 repo 是其 G1 + mjlab 实现 |
| [unitree.md](unitree.md) | G1 是主要硬件目标 |
| [legged_gym.md](legged_gym.md) | 同为 rsl_rl + 并行仿真框架，legged_gym 基于 IsaacGym |
| [robot_lab.md](robot_lab.md) | robot_lab 也有 BeyondMimic/AMP Dance，但基于 IsaacLab |
