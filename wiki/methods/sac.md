---
type: method
tags: [rl, policy-optimization, sac, off-policy, maximum-entropy, manipulation]
status: complete
updated: 2026-07-16
summary: "SAC 是连续控制中最主流的 off-policy 最大熵算法：用 Replay Buffer 反复利用经验、双 Q 抑制高估、自动调温度平衡探索与利用，样本效率远高于 PPO，是真机 RL 与精细操作的首选。"
related:
  - ./flashsac.md
  - ./policy-optimization.md
  - ./reinforcement-learning.md
  - ./ppo.md
  - ../comparisons/ppo-vs-sac.md
  - ../comparisons/online-vs-offline-rl.md
  - ../comparisons/model-based-vs-model-free.md
  - ../queries/ppo-vs-sac-for-robots.md
  - ../queries/rl-hyperparameter-guide.md
  - ../tasks/locomotion.md
  - ../concepts/reward-design.md
  - ../formalizations/mdp.md
  - ../formalizations/bellman-equation.md
sources:
  - ../../sources/papers/policy_optimization.md
---

# SAC（Soft Actor-Critic）

**SAC（软演员-评论家）**：在标准 RL 目标上叠加 **最大熵正则项**，让策略在完成任务的同时保持尽量高的随机性，配合 **Replay Buffer 的 off-policy 复用** 与 **双 Q 网络**，在连续控制上取得远高于 PPO 的样本效率，是真实机器人 RL 与精细操作任务最常用的算法。

## 一句话定义

不只追求"奖励最大"，而是追求"在拿到高奖励的前提下行为尽量随机"——用最大熵目标自动平衡探索与利用，并把每条历史经验存进 Buffer 反复学习。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SAC | Soft Actor-Critic | 最大熵框架的 off-policy actor-critic 算法 |
| MaxEnt RL | Maximum Entropy RL | 在奖励外叠加策略熵的强化学习目标 |
| Replay Buffer | Experience Replay Buffer | 存储历史转移、供 off-policy 反复采样的经验池 |
| TD3 | Twin Delayed DDPG | 双 Q + 延迟更新的确定性 off-policy 算法，与 SAC 思路互补 |
| MDP | Markov Decision Process | SAC 求解的标准问题形式 |
| $\alpha$ | Temperature | 平衡奖励与熵的温度系数，可自动调节 |

## 为什么重要

- 真机 RL 的每条交互数据都很**昂贵**，SAC 的 off-policy 复用使其样本效率比 on-policy 的 [PPO](./ppo.md) 高约 10–100 倍，是真实机器人上少样本学习的现实选择。
- **最大熵框架** 鼓励充分探索、防止策略过早坍缩到单一模式，在 contact-rich 的精细操作、灵巧手任务上通常优于确定性方法。
- **温度自动调节** 把"探索强度"这一原本最难调的超参数变成自适应量，显著降低调参负担。
- 是 [Policy Optimization](./policy-optimization.md) 家族中 off-policy 的代表，与 PPO 形成"仿真大规模 vs 真机高效"的互补组合（详见 [PPO vs SAC](../comparisons/ppo-vs-sac.md)）。
- **[FlashSAC](./flashsac.md)**（2026）在 SAC 上引入 scaling 式少更新 + 大网络 + 范数约束，面向高维机器人 sim-to-real，在 G1 盲行走等任务上墙钟可较 PPO 缩短约一个数量级。

## 主要技术路线

### 1. 最大熵目标

SAC 把标准 RL 的累积奖励目标改写为带熵正则的形式：

$$
J(\pi) = \sum_t \mathbb{E}_{(s_t, a_t) \sim \rho_\pi}\left[ r(s_t, a_t) + \alpha\, \mathcal{H}\big(\pi(\cdot|s_t)\big) \right]
$$

- $\mathcal{H}(\pi(\cdot|s_t)) = -\mathbb{E}_{a\sim\pi}[\log \pi(a|s_t)]$ 为策略熵，度量动作分布的随机程度。
- 温度 $\alpha$ 控制"奖励"与"熵"的相对权重：$\alpha$ 越大越鼓励探索，$\alpha\to 0$ 退化为标准 RL。

### 2. 软 Bellman 与双 Q 网络

- Critic 学习 **软 Q 函数**：在 [Bellman 方程](../formalizations/bellman-equation.md) 的目标值中加入下一步的熵项，使价值评估与最大熵目标一致。
- 采用 **Clipped Double-Q**：维护两个 Q 网络 $Q_{\phi_1}, Q_{\phi_2}$，取两者较小值构造目标，抑制 off-policy 自举带来的 Q 值高估。
- 用 **目标网络软更新**（系数 $\tau$）平滑 Bootstrap 目标，稳定训练。

### 3. 演员更新（重参数化）

- 策略最大化"软 Q 减去对数概率"：$\max_\theta \mathbb{E}_{s\sim D,\,a\sim\pi_\theta}\big[\,Q_\phi(s,a) - \alpha\log\pi_\theta(a|s)\,\big]$。
- 用 **重参数化技巧**（$a = \tanh(\mu_\theta(s) + \sigma_\theta(s)\odot\epsilon)$）把随机采样的梯度变为低方差的确定性梯度，$\tanh$ 把动作约束到有界范围。

### 4. 自动温度调节

- 把 $\alpha$ 设为可学习变量，约束策略熵不低于目标熵 $\bar{\mathcal{H}}$（常取 $-\dim(\mathcal{A})$），通过对偶优化自动调节：

$$
\min_\alpha\ \mathbb{E}_{a\sim\pi}\big[-\alpha\big(\log\pi(a|s) + \bar{\mathcal{H}}\big)\big]
$$

- 这一改进（Haarnoja et al., 2018 的第二版）让 SAC 在不同任务间几乎免调温度，是其鲁棒性的关键。

### 5. Off-policy 训练循环

- 每与环境交互一步，把转移 $(s, a, r, s')$ 存入 [Replay Buffer](../comparisons/online-vs-offline-rl.md)，再从中随机采样 minibatch 更新 Q 与策略——**每条经验被反复利用**，这是高样本效率的来源。

## 关键超参数（机器人实践）

| 超参数 | 典型范围 | 作用 |
|--------|----------|------|
| buffer_size | 1M ~ 5M | 经验池容量，越大越稳，受内存限制 |
| batch_size | 256 ~ 1024 | 每次从 Buffer 采样的 minibatch 大小 |
| learning_rate | 3e-4 | 策略与 Q 网络通常同一学习率 |
| 软更新 $\tau$ | 0.005 | 目标网络更新平滑系数 |
| 温度 $\alpha$ | 自动调节 | 目标熵设为 $-\dim(\mathcal{A})$，让 $\alpha$ 自适应 |
| learning_starts | 1k ~ 10k 步 | Buffer 预热步数，先采样后更新 |
| 折扣 $\gamma$ | 0.99 | 长时域信用分配 |

## 与机器人技术的联系

- **何时选 SAC vs PPO**：off-policy 高样本效率 vs on-policy 稳定并行的权衡，详见 [PPO vs SAC](../comparisons/ppo-vs-sac.md) 与 [面向机器人的 PPO/SAC 选型](../queries/ppo-vs-sac-for-robots.md)。
- **两阶段范式**：常见做法是先用 PPO 在仿真大规模预训练，再用 SAC 在真机少量数据上 fine-tune，消除残余 sim2real gap（见 [Locomotion](../tasks/locomotion.md)）。
- **奖励与探索**：SAC 的最大熵框架与 [奖励设计](../concepts/reward-design.md) 紧密耦合——熵正则缓解稀疏/欺骗性奖励下的过早坍缩。
- **算法族定位**：SAC 是 [Policy Optimization](./policy-optimization.md) 家族中 off-policy 的主力，建立在 [强化学习基础](./reinforcement-learning.md) 与 [MDP](../formalizations/mdp.md) 之上。
- **超参数选型**：连续控制 RL 的通用调参思路见 [RL 超参数指南](../queries/rl-hyperparameter-guide.md)。

## 关联页面
- [Policy Optimization（算法族总览）](./policy-optimization.md)
- [Reinforcement Learning（强化学习基础）](./reinforcement-learning.md)
- [FlashSAC（快速稳定 SAC）](./flashsac.md)
- [PPO（近端策略优化）](./ppo.md)
- [PPO vs SAC（对比）](../comparisons/ppo-vs-sac.md)
- [Online vs Offline RL（对比）](../comparisons/online-vs-offline-rl.md)
- [PPO vs SAC for Robots（选型 Query）](../queries/ppo-vs-sac-for-robots.md)
- [Locomotion（任务）](../tasks/locomotion.md)
- [MDP（形式化）](../formalizations/mdp.md)
- [Bellman 方程（形式化）](../formalizations/bellman-equation.md)

## 参考来源
- [Policy Optimization 来源归档（PPO/SAC/TD3/TRPO/Rudin/BRRL）](../../sources/papers/policy_optimization.md)
- Haarnoja, T., et al. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*. <https://arxiv.org/abs/1801.01290>
- Haarnoja, T., et al. (2018). *Soft Actor-Critic Algorithms and Applications*. <https://arxiv.org/abs/1812.05905>
- Fujimoto, S., et al. (2018). *Addressing Function Approximation Error in Actor-Critic Methods (TD3)*. <https://arxiv.org/abs/1802.09477>
