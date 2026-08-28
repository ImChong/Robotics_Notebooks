# RL Runner 类型谱系（维护者整理）

- **类型**：`personal`（图示/教学表整理，非正式出版物）
- **日期**：2026-08-16
- **原始形态**：用户提供的 Runner 分类表（深色背景教学图）
- **用途**：为 [RL Runner（训练循环编排）](../../wiki/concepts/rl-runner.md) 提供可追溯编译来源；正文以 wiki 页为准，本文件只保留图中原表与读法边界。

## 图示原意

行业对「驱动一次训练/评测循环的编排层」没有统一命名：有的框架叫 `Runner`，有的叫 `Trainer`、`Algorithm`，或拆成 `Collector` + `Learner`。图中按**训练方法 / 数据来源**把常见循环分成十类。

## 原表转写

| Runner 类型 | 常见算法 | 数据来源 | 核心循环 |
|-------------|----------|----------|----------|
| On-policy Runner | PPO, A2C, TRPO | 当前策略新采集的数据 | rollout → GAE → update → 丢弃本轮 rollout |
| Off-policy Runner | SAC, TD3, DDPG, DQN | Replay Buffer | 少量采集 → 随机抽历史 → 多次更新 |
| Offline Runner | CQL, IQL, BCQ, TD3+BC | 固定离线数据集 | 读数据集 → 更新网络，不跑环境 |
| Distillation Runner | Teacher–Student | Teacher 输出或轨迹 | Teacher 推理 → Student 模仿 |
| Imitation Runner | BC, DAgger, GAIL, AIRL | 专家演示 + 可选环境交互 | 采集专家/学习者轨迹 → 模仿或对抗奖励更新 |
| Multi-agent Runner | MAPPO, QMIX, MADDPG | 多智能体环境 | 组织多 agent 观测、动作与共享/独立策略 |
| Self-play Runner | AlphaZero 类、博弈策略 | 当前或历史策略互打 | 选对手 → 对局 → 更新 → 写入策略池 |
| Distributed Runner | IMPALA, Ape-X, Distributed PPO | 多个采样进程 | Actor 并行采样 → Learner 集中更新 |
| Model-based Runner | Dreamer, MuZero, MBPO | 真环境 + 学到的动力学模型 | 采数 → 训世界模型 → 想象 rollout → 更新策略 |
| Evaluation Runner | 适用于所有算法 | 仅环境交互 | 确定性推理 → 统计指标，不更新参数 |

## 读法边界（编译时不要混）

- **Runner ≠ 算法**：PPO 是损失与更新规则；On-policy Runner 是「采一批、算 GAE、更新、扔掉」的调度。同一套 PPO 损失也可以挂在分布式 Actor–Learner 上。
- **Runner ≠ 环境闭环**：环境侧 `obs → act → step → reward` 见 [具身 RL 最小闭环](../../wiki/concepts/embodied-rl-minimal-closed-loop.md)；Runner 是在闭环外包一层「何时采、采完怎么更新、更新后数据是否还能用」。
- **Evaluation Runner 与训练正交**：任何训练 Runner 都需要一条不更新参数的评测循环，不能用训练噪声策略的回报当选 checkpoint 依据。
- **Distributed / Self-play 是拓扑或数据源**，常叠在 on/off-policy 更新规则之上，而不是互斥的第十一种损失函数。

## 对 wiki 的映射

| 要点 | 目标页 |
|------|--------|
| 十类 Runner 定义、核心循环、选型 | `wiki/concepts/rl-runner.md`（本资料升格页） |
| 环境最小闭环（对照：Runner 是其外包编排） | `wiki/concepts/embodied-rl-minimal-closed-loop.md` |
| On-policy 算法与 GAE | `wiki/methods/ppo.md`、`wiki/methods/gae.md` |
| Off-policy Replay Buffer | `wiki/methods/sac.md`、`wiki/comparisons/online-vs-offline-rl.md` |
| Offline / 固定数据集 | `wiki/comparisons/online-vs-offline-rl.md` |
| Teacher–Student 蒸馏 | `wiki/concepts/privileged-training.md`、`wiki/methods/teacher-student-dagger-training.md` |
| BC / DAgger / GAIL / AIRL | `wiki/methods/imitation-learning.md`、`wiki/methods/behavior-cloning.md`、`wiki/methods/dagger.md`、`wiki/methods/inverse-reinforcement-learning.md` |
| 多智能体 | `wiki/methods/marl.md` |
| 自博弈 | `wiki/entities/paper-notebook-robostriker.md`、`wiki/methods/marl.md` |
| 世界模型想象 rollout | `wiki/methods/model-based-rl.md`、`wiki/concepts/latent-imagination.md` |
| 人形训练模块栈中的位置 | `wiki/overview/humanoid-rl-policy-training-five-modules.md` |
