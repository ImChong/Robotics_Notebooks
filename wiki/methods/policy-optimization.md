---
type: method
tags: [rl, policy-optimization, ppo, sac, locomotion]
status: complete
related:
  - ./reinforcement-learning.md
  - ./imitation-learning.md
  - ../tasks/locomotion.md
  - ../concepts/sim2real.md
  - ../formalizations/mdp.md
  - ../formalizations/bellman-equation.md
  - ../queries/rl-hyperparameter-guide.md
summary: "Policy Optimization 汇总 PPO、SAC、TD3 等主流策略更新方法，是机器人 RL 的算法核心。"
---

# Policy Optimization

**策略优化**：通过直接对策略参数做梯度上升或近似优化，使期望累积奖励最大化的一类强化学习方法。

## 一句话定义

不学值函数再推策略，而是**直接优化"在每个状态下该怎么行动"的参数**，用梯度或近似梯度做更新。

## 为什么重要

机器人控制的动作空间通常是**连续高维**的（30+ 自由度的关节力矩），Q-learning 类方法在离散动作空间里很强，但对连续动作空间扩展困难。

Policy Optimization 天然适合连续动作空间：
- 策略直接输出连续动作
- 梯度来自 reward 信号，不需要显式建模状态转移
- PPO、SAC 是人形机器人 RL 训练中使用最广泛的两类算法

## 核心思想

### Policy Gradient 定理

策略梯度的核心等式：

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t \right]$$

其中：
- $\theta$：策略参数
- $G_t$：从时刻 $t$ 起的累积折扣奖励
- $\pi_\theta(a|s)$：参数化策略

直觉：**让"好的动作"概率变高，让"坏的动作"概率变低。**

### Advantage Function

用优势函数 $A(s,a)$ 替代 $G_t$，降低方差：

$$A(s,a) = Q(s,a) - V(s)$$

> "这个动作比平均水平好多少？"——而不是它的绝对回报。估计方法见 [Generalized Advantage Estimation (GAE)](./gae.md)，它通过 λ 权衡偏差与方差。

## 主要分类

| 算法 | 类型 | 特点 | 机器人使用场景 |
|------|------|------|--------------|
| **PPO** | On-policy | 稳定、易调参、clip 约束防止更新过猛 | locomotion 训练主流 |
| **SAC** | Off-policy | 最大熵、样本效率高、自动调温度 | 操作类任务、样本受限场景 |
| **TRPO** | On-policy | 用 KL 约束保证策略更新安全 | 理论意义大，工程不常用 |
| **AWR** | Off-policy | 用 advantage 加权行为克隆 | IL 与 RL 混合场景 |

### PPO（Proximal Policy Optimization）

当前人形 locomotion RL 中使用最广泛的算法：

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]$$

- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$：重要性采样比
- $\epsilon$：clip 范围（通常 0.1～0.2）

PPO 的核心思想：**每次更新不要步子太大**，clip 机制防止策略崩溃。

### SAC（Soft Actor-Critic）

最大熵强化学习代表：

$$J(\pi) = \sum_t \mathbb{E}_{(s_t,a_t)\sim\rho_\pi} [r(s_t,a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))]$$

- $\mathcal{H}$：策略熵
- $\alpha$：温度参数（可自动调节）

SAC 优势：Off-policy，样本效率高；熵正则化鼓励探索；适合精细操作任务。

## 在人形机器人中的典型应用

### 1. Locomotion 训练

最常见的使用方式：
- 在 Isaac Gym / Isaac Lab 中并行启动 4096+ 个环境
- 用 PPO 收集 rollout，更新策略
- 训练数小时获得鲁棒行走策略

代表工作：
- Rudin et al., *Learning to Walk in Minutes* — legged_gym + PPO
- Kumar et al., *RMA* — 适应性策略 + PPO

### 2. 操作任务

SAC 常用于：
- 机械臂抓取
- 灵巧手操作
- 接触丰富的精细任务

### 3. IL 初始化 + RL 微调

先用 BC / AWR 从演示数据初始化策略，再用 PPO / SAC 在线优化，兼顾样本效率和最终性能。

## 常见问题和调参技巧

### Reward Shaping
奖励设计是 Policy Optimization 成功的关键：
- 太稀疏：策略难以收敛
- 太密集：容易出现 reward hacking
- 常见组合：存活奖励 + 速度跟踪 + 姿态惩罚 + 关节力矩惩罚

### 观测空间设计
- 包含足够信息（关节角、速度、IMU、接触状态、指令）
- 不要包含仿真中存在但真实环境没有的信息
- Sim2Real 友好的观测设计

### 超参数建议（PPO）
- `num_envs`：4096+（GPU 并行）
- `n_steps`：24～64 步
- `clip_range`：0.1～0.2
- `learning_rate`：3e-4（可衰减）

## 参考来源

- Schulman et al., *Proximal Policy Optimization Algorithms* (2017) — PPO 原论文
- Haarnoja et al., *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning* (2018) — SAC 原论文
- Rudin et al., *Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning* (2022) — locomotion + PPO 代表
- **ingest 档案：** [sources/papers/policy_optimization.md](../../sources/papers/policy_optimization.md)

## 关联页面

- [Reinforcement Learning](./reinforcement-learning.md)
- [Imitation Learning](./imitation-learning.md)
- [Locomotion](../tasks/locomotion.md)
- [Sim2Real](../concepts/sim2real.md)
- [Formalizations: MDP](../formalizations/mdp.md)
- [Formalizations: Bellman 方程](../formalizations/bellman-equation.md)
- [Formalizations: GAE](../formalizations/gae.md) — PPO 使用 GAE 作为优势估计标准实现
- [Query：RL 超参数调参指南](../queries/rl-hyperparameter-guide.md)

## 推荐继续阅读

- Schulman et al., [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347) — PPO 原论文
- Haarnoja et al., [*Soft Actor-Critic*](https://arxiv.org/abs/1801.01290) — SAC 原论文
- Andrychowicz et al., [*Learning dexterous in-hand manipulation*](https://arxiv.org/abs/1808.00177) — 灵巧手操作 PPO 经典

## 一句话记忆

> Policy Optimization 直接优化策略参数，PPO 和 SAC 是机器人 RL 里最常用的两把刀：PPO 稳定、适合大批量 locomotion 训练；SAC 样本效率高、适合精细操作。
