---
type: formalization
tags: [gae, rl, advantage-function, policy-gradient, ppo, variance-reduction]
status: complete
related:
  - ./bellman-equation.md
  - ../methods/policy-optimization.md
  - ../methods/reinforcement-learning.md
sources:
  - ../../sources/papers/policy_optimization.md
---

# GAE（广义优势估计）

**GAE（Generalized Advantage Estimation）** 是估计策略梯度中优势函数 $A(s,a)$ 的标准方法，通过参数 $\lambda \in [0,1]$ 在**偏差（bias）和方差（variance）**之间平滑权衡。

## 一句话定义

> GAE = 多步 TD 误差的指数加权平均。$\lambda$ 接近 1 时低偏差高方差，接近 0 时高偏差低方差。

## 为什么重要

策略梯度的关键公式：

$$\nabla_\theta J(\theta) = \mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a \mid s) \cdot A^\pi(s, a) \right]$$

$A^\pi(s,a)$ 的估计质量直接决定策略梯度的有效性。估计差 → 梯度噪声大 → 训练不稳定。

GAE 是 PPO 中优势估计的标准实现，几乎所有主流人形机器人 RL 框架（Isaac Lab、legged\_gym）都使用 GAE。

## 形式化推导

### TD 残差

定义 TD 残差（时序差分误差）：

$$\delta_t^V = R_t + \gamma V(s_{t+1}) - V(s_t)$$

直觉：$\delta_t^V$ 是"一步看到的 actual vs 预期的价值差"——TD error。

### GAE 定义

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V$$

展开为递推形式（实际计算用这个）：

$$\hat{A}_t = \delta_t + (\gamma \lambda) \delta_{t+1} + (\gamma \lambda)^2 \delta_{t+2} + \cdots$$

或等价的反向递推：

$$\hat{A}_t = \delta_t + \gamma \lambda \hat{A}_{t+1}$$

这是 GAE 的实际代码实现：从 episode 末尾往前计算。

### 特殊情况

| $\lambda$ 取值 | 退化为 | 特性 |
|---------------|-------|------|
| $\lambda = 0$ | 单步 TD 优势：$\hat{A}_t = \delta_t = R_t + \gamma V(s_{t+1}) - V(s_t)$ | 高偏差，低方差 |
| $\lambda = 1$ | Monte Carlo 优势：$\hat{A}_t = \sum_{l=0}^\infty \gamma^l R_{t+l} - V(s_t)$ | 低偏差，高方差 |
| $\lambda \in (0,1)$ | λ-return 加权混合 | 偏差-方差折中 |

## Bias-Variance 权衡直觉

```
λ → 0                           λ → 1
高偏差 ←────────────────────→ 低偏差
低方差 ←────────────────────→ 高方差
训练稳定                        训练噪声大
需要好的 V(s) 估计              不太依赖 V(s)
```

**为什么有 bias？** $\lambda < 1$ 时，GAE 依赖值函数 $V(s)$。如果 $V(s)$ 估计不准，GAE 就有偏差。

**为什么有 variance？** $\lambda = 1$ 时，使用真实回报，每条轨迹的随机性都完整保留，方差大。

## 在 PPO 中的应用

```python
# 伪代码：PPO 中的 GAE 计算
advantages = []
gae = 0
for t in reversed(range(T)):
    delta = rewards[t] + gamma * values[t+1] - values[t]
    gae = delta + gamma * lambda_ * gae
    advantages.insert(0, gae)
returns = [adv + val for adv, val in zip(advantages, values)]
```

PPO 推荐超参数：$\lambda = 0.95$，$\gamma = 0.99$（legged\_gym 默认值）。

## 与 TD(λ) 的关系

GAE 和 TD(λ) 的数学结构相同——都是指数加权多步 TD 残差。区别：
- **TD(λ)**：用于值函数学习（价值估计）
- **GAE**：用于策略梯度（优势估计）

两者都通过 $\lambda$ 参数控制 bootstrapping 程度。

## 在人形机器人 RL 中的常见问题

### 问题：步长 mismatch

legged\_gym 通常以 50Hz 运行控制，但 episode 长 20s → T = 1000 步。反向递推需要完整 episode，内存占用大。

解决方案：截断 rollout 长度（通常 24-64 步），加 terminal value bootstrap：
$$\hat{A}_{T-1} = \delta_{T-1} + \gamma \lambda V(s_T)$$

### 问题：V(s) 训练不充分时 GAE 偏差大

人形机器人早期训练时 critic（$V$ 网络）不准，$\lambda$ 不宜过小（否则偏差大训练慢）。实践中 $\lambda = 0.95 \sim 0.99$。

## 参考来源

- Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation* (2016) — GAE 原始论文
- Schulman et al., *Proximal Policy Optimization Algorithms* (2017) — GAE 在 PPO 中的标准应用
- Sutton & Barto, *Reinforcement Learning: An Introduction* Ch.12 — TD(λ) 理论背景

## 关联页面

- [Bellman Equation](./bellman-equation.md) — GAE 基于 TD 残差，TD 残差来自 Bellman 方程的一步展开
- [Policy Optimization](../methods/policy-optimization.md) — PPO 使用 GAE 作为优势估计的标准实现
- [Reinforcement Learning](../methods/reinforcement-learning.md) — GAE 是 policy gradient 方法的核心组件
