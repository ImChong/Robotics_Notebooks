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

### 1. 统一 AMP 训练

| 组件 | 说明 |
|------|------|
| Actor-Critic | 单一网络，统一处理 locomotion 与 recovery |
| AMP Discriminator | 区分"参考数据动作"与"策略生成动作"，驱动自然度 |
| 动作数据集 | `WalkandRun`（步行/跑步）+ `Recovery`（跌倒恢复）两套 clip |
| Delayed Termination | 部分环境在 reset 前给予 recovery 窗口，逼迫策略学习爬起 |

### 2. 训练特征

- 并行规模：4096 个并行环境
- 收敛特征：约 2 万步时 recovery 行为突然涌现，指标会跳变，属于正常学习现象
- 任务：`Unitree-G1-AMP-Rough` / `Unitree-G1-AMP-Flat`

### 3. 部署

- 支持 ONNX export，直接导出用于真实机器人部署
- 训练与推理 pipeline 一致

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
