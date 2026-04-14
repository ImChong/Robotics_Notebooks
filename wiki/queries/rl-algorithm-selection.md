---
type: query
tags: [rl, locomotion, policy-optimization, humanoid]
status: complete
---

> **Query 产物**：本页由以下问题触发：「在足式/人形机器人里，PPO / SAC / TD3 怎么选？」
> 综合来源：[Reinforcement Learning](../methods/reinforcement-learning.md)、[Policy Optimization](../methods/policy-optimization.md)、[Locomotion](../tasks/locomotion.md)、[Sim2Real](../concepts/sim2real.md)

# RL 算法选型指南：足式机器人中的 PPO / SAC / TD3

## 核心结论（先看这里）

| 场景 | 首选 | 备选 |
|------|------|------|
| 仿真 locomotion（legged_gym / Isaac Lab） | **PPO** | AWR |
| 真实机器人少样本学习 | **SAC** | TD3 |
| 确定性连续控制、操作任务 | **TD3** | SAC |
| 模仿人类运动风格 | **AMP（PPO 变体）** | SAC+GAIL |
| 需要最高渐近性能 | **SAC** | TD3 |
| 调参时间有限，要稳定收敛 | **PPO** | AWR |

---

## 三大算法对比

### PPO（Proximal Policy Optimization）

**类型**：On-policy，Policy Gradient，随机策略

**原理**：用 clip 约束限制每次策略更新幅度，防止策略崩溃。

$$L^{CLIP} = \mathbb{E}_t \left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

**适合足式机器人的原因**：
- 大规模并行仿真（8192+ envs）下 on-policy 收集非常高效
- 训练稳定，超参数不敏感
- legged_gym / Isaac Lab 的标准选择
- ETH ANYmal / Unitree 系列几乎全用 PPO

**缺点**：
- 样本效率低（on-policy，数据不复用）
- 不适合真实机器人少样本场景

**关键超参数**：
- `clip_range`: 0.2（标准）
- `n_steps`: 24（Isaac Lab 默认）
- `num_envs`: 4096~8192（足式任务常用）
- `entropy_coef`: 0.01（适度探索）

---

### SAC（Soft Actor-Critic）

**类型**：Off-policy，Actor-Critic，最大熵框架，随机策略

**原理**：最大化奖励 + 策略熵，自动在探索和利用间平衡。

$$J(\pi) = \sum_t \mathbb{E}\left[r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))\right]$$

**适合足式机器人的原因**：
- 样本效率高（off-policy，经验回放）
- 熵正则化天然鼓励探索，不易陷入局部最优
- 适合真实机器人少量数据学习（RMA 第二阶段等）
- 自动温度调节：不需要手调 entropy coefficient

**缺点**：
- 超参数（replay buffer、batch size、温度 α）比 PPO 多
- 大规模并行仿真优势不如 PPO（off-policy 的并行化收益递减）
- 随机策略在某些精细操作任务上不如 TD3

**关键超参数**：
- `buffer_size`: 1M（足够大）
- `batch_size`: 256
- `learning_rate`: 3e-4
- `tau`: 0.005（软更新）

---

### TD3（Twin Delayed Deep Deterministic Policy Gradient）

**类型**：Off-policy，Actor-Critic，确定性策略

**原理**：在 DDPG 基础上用双 Critic 减少 Q 值过估计，延迟策略更新。

$$\tilde{a} = \pi_{\theta'}(s') + \text{clip}(\epsilon, -c, c), \quad y = r + \gamma \min_{i=1,2} Q_{\theta_i'}(s', \tilde{a})$$

**适合足式机器人的原因**：
- 确定性策略在精细操作、低噪声环境下表现好
- 双 Critic 减少 Q 值过估计，训练更稳定
- 样本效率接近 SAC

**缺点**：
- 确定性策略探索能力弱（需要额外噪声）
- 在接触丰富、地形复杂的任务上不如 SAC
- 超参数调节仍复杂

---

## 足式机器人 RL 实践要点

### 1. 大规模并行仿真 → 用 PPO

Isaac Lab / legged_gym 上 8000+ 并行环境，on-policy 的 PPO 每秒可收集数十万步数据，off-policy 的 SAC 样本效率优势消失。

### 2. 真实机器人 / 样本受限 → 用 SAC

真实机器人每步数据成本极高，SAC 的经验回放可以反复利用每个样本，样本效率比 PPO 高 10-100x。

### 3. 运动风格（自然步态）→ 用 AMP

AMP（Adversarial Motion Priors）= PPO + 判别器 reward。用 MoCap 数据训练判别器，判别器 reward 驱动 RL 策略产生自然步态。仍用 PPO 作为底层优化器。

### 4. 调试阶段 → 先用 PPO

PPO 更容易调试：reward 曲线平滑，超参数不敏感，失败原因更容易定位。SAC 调参更复杂，适合验证好 reward 设计后再切换。

---

## 常见错误

| 错误 | 正确做法 |
|------|---------|
| 仿真里用 SAC，速度极慢 | 仿真大规模并行用 PPO |
| PPO + 真实机器人，样本耗尽 | 切换到 SAC，用经验回放 |
| 直接跑 TD3，策略不收敛 | 先跑 PPO/SAC 验证 reward 设计 |
| 不调 entropy，SAC 探索不足 | 用自动温度调节（auto entropy tuning） |

---

## 关联页面

- [Policy Optimization](../methods/policy-optimization.md) — PPO/SAC/TD3 算法详细说明
- [Reinforcement Learning](../methods/reinforcement-learning.md) — RL 方法全景
- [Reward Design](../concepts/reward-design.md) — reward 设计是算法选型之后的核心问题
- [Sim2Real](../concepts/sim2real.md) — 仿真训练后如何迁移到真实机器人
- [Locomotion](../tasks/locomotion.md) — 足式 locomotion 任务定义与挑战

## 参考来源

- Schulman et al., *Proximal Policy Optimization Algorithms* (2017) — PPO 原论文
- Haarnoja et al., *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning* (2018) — SAC 原论文
- Fujimoto et al., *Addressing Function Approximation Error in Actor-Critic Methods* (TD3, 2018)
- Peng et al., *AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control* (2021)
