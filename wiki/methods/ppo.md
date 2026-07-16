---
type: method
tags: [rl, policy-optimization, ppo, on-policy, locomotion]
status: complete
updated: 2026-07-16
summary: "PPO 用 clip 代理目标约束策略更新幅度，兼顾稳定性与实现简单，是人形/足式机器人大规模并行 RL 训练的事实标准算法。"
related:
  - ./flashsac.md
  - ./policy-optimization.md
  - ./reinforcement-learning.md
  - ./sac.md
  - ./gae.md
  - ../concepts/neural-feedback-controller.md
  - ../comparisons/ppo-vs-sac.md
  - ../queries/ppo-vs-sac-for-robots.md
  - ../tasks/locomotion.md
  - ../formalizations/mdp.md
sources:
  - ../../sources/papers/policy_optimization.md
---

# PPO（Proximal Policy Optimization）

**PPO（近端策略优化）**：用 **clip 代理目标** 限制每次策略更新中新旧策略概率比的偏离幅度，在保持 TRPO 级别更新稳定性的同时，把实现复杂度降到一阶优化器即可训练，是机器人 RL 中使用最广的 on-policy 算法。

## 一句话定义

每步更新只允许策略"小步走"——把新旧策略的概率比裁剪在 $[1-\varepsilon, 1+\varepsilon]$ 内，避免一次更新走太远把已学到的行为搞崩。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PPO | Proximal Policy Optimization | clip 约束的 on-policy 策略梯度算法 |
| TRPO | Trust Region Policy Optimization | PPO 前身，用 KL 约束限制更新幅度 |
| GAE | Generalized Advantage Estimation | PPO 计算优势函数的标准配套 |
| KL | Kullback–Leibler Divergence | 度量新旧策略分布差异，用于早停 |
| MDP | Markov Decision Process | PPO 求解的标准问题形式 |

## 为什么重要

- 机器人控制是**连续高维**动作空间（30+ 关节力矩），PPO 直接输出连续动作分布，天然适配。
- 相比 TRPO 的二阶 KL 约束，PPO 只需一阶 SGD/Adam + 简单 clip，**工程实现门槛低**，是 [Isaac Gym / Isaac Lab](../entities/isaac-gym-isaac-lab.md)、legged_gym 等仿真栈的默认算法。
- 在 [大规模并行仿真](../tasks/locomotion.md) 下样本利用率虽不如 off-policy，但凭海量并行环境把"低样本效率"换成"墙钟时间短"，成为人形/足式 locomotion 训练的事实标准。
- 高维人形/灵巧手任务上，[FlashSAC](./flashsac.md) 等 scaling 式 off-policy 方法已在墙钟与渐近性能上挑战 PPO 默认地位（项目页 TL;DR：「If you're using PPO, try FlashSAC!」）。

## 主要技术路线

### 1. Clip 代理目标

PPO 的核心是裁剪后的代理目标。记概率比 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$，则：

$$
L^{CLIP}(\theta) = \mathbb{E}_t\left[\min\big(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta),\,1-\varepsilon,\,1+\varepsilon)\,\hat{A}_t\big)\right]
$$

- $\hat{A}_t$ 为优势估计（通常由 [GAE](./gae.md) 给出）；$\varepsilon$ 常取 0.1~0.2。
- 取 $\min$ 使目标成为真实目标的**悲观下界**：优势为正时奖励被 $1+\varepsilon$ 封顶，优势为负时惩罚不被 clip，从而抑制过大的策略跳变。

### 2. 优势估计与价值损失

- 用 [GAE](./gae.md) 在偏差与方差间权衡地估计 $\hat{A}_t$，依赖 critic 价值网络 $V_\phi$。
- 总损失叠加价值回归项与熵正则：$L = L^{CLIP} - c_1\,L^{VF} + c_2\,\mathcal{H}[\pi_\theta]$，熵项鼓励探索、防止策略过早坍缩。

### 3. On-policy 多轮 minibatch 更新

- 每轮 rollout 收集一批轨迹后，对同一批数据做 **多个 epoch 的 minibatch SGD**（典型 3~10 epoch），这是 PPO 相对 vanilla PG 提升样本利用率的关键。
- 可选 **KL 早停**：当新旧策略 KL 超阈值时提前结束本轮更新，作为 clip 之外的二重保险。

### 4. 大规模并行变体与改进

- **Rudin et al. (2022)**：Isaac Gym + PPO，8192 并行环境约 20 分钟训出四足/双足步态，开启大规模并行 RL 范式（见 [locomotion](../tasks/locomotion.md)）。
- **BRRL/BPO（2026）**：将 clip 重新解释为"朝有界 ratio 最优解"的近似优化，给出单调改进的理论保证并在 IsaacLab 人形 locomotion 上报告优于 PPO 的稳定性。

## 关键超参数（机器人实践）

| 超参数 | 典型范围 | 作用 |
|--------|----------|------|
| clip $\varepsilon$ | 0.1 ~ 0.2 | 控制单步更新幅度 |
| GAE $\lambda$ | 0.9 ~ 0.97 | 优势估计偏差/方差权衡 |
| 折扣 $\gamma$ | 0.99 ~ 0.995 | 长时域信用分配 |
| epoch 数 | 3 ~ 10 | 单批数据复用次数 |
| 熵系数 $c_2$ | 0.0 ~ 0.01 | 探索强度 |

## 与机器人技术的联系

- **何时选 PPO vs SAC**：on-policy 与 off-policy 在稳定性与样本利用率上的权衡，详见 [PPO vs SAC](../comparisons/ppo-vs-sac.md) 与 [面向机器人的 PPO/SAC 选型](../queries/ppo-vs-sac-for-robots.md)。
- **课程与奖励**：PPO 训练效果高度依赖 [课程学习](../concepts/curriculum-learning.md) 与 [奖励设计](../concepts/reward-design.md)。
- **算法族定位**：PPO 是 [Policy Optimization](./policy-optimization.md) 家族中 on-policy 的主力，与 [强化学习基础](./reinforcement-learning.md) 一脉相承。
- **直觉层理解**：参数更新在强化哪些状态→动作连接，见 [神经反馈控制器](../concepts/neural-feedback-controller.md)。

## 关联页面
- [Policy Optimization（算法族总览）](./policy-optimization.md)
- [Reinforcement Learning（强化学习基础）](./reinforcement-learning.md)
- [FlashSAC（快速稳定 SAC）](./flashsac.md)
- [SAC（软演员-评论家）](./sac.md)
- [GAE（广义优势估计）](./gae.md)
- [PPO vs SAC（对比）](../comparisons/ppo-vs-sac.md)
- [PPO vs SAC for Robots（选型 Query）](../queries/ppo-vs-sac-for-robots.md)
- [Locomotion（任务）](../tasks/locomotion.md)
- [MDP（形式化）](../formalizations/mdp.md)

## 参考来源
- [Policy Optimization 来源归档（PPO/SAC/TD3/TRPO/Rudin/BRRL）](../../sources/papers/policy_optimization.md)
- Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms*. <https://arxiv.org/abs/1707.06347>
- Schulman, J., et al. (2015). *Trust Region Policy Optimization*. <https://arxiv.org/abs/1502.05477>
- Rudin, N., et al. (2022). *Learning to Walk in Minutes Using Massively Parallel Deep RL*. <https://arxiv.org/abs/2109.11978>
