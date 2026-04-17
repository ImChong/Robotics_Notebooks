---
type: formalization
tags: [rl, bellman, value-function, dynamic-programming]
status: complete
related:
  - ./mdp.md
  - ../methods/reinforcement-learning.md
  - ../methods/policy-optimization.md
sources:
  - ../../sources/papers/policy_optimization.md
---

# Bellman 方程

**Bellman 方程**：值函数的递归关系，揭示了"未来奖励"与"当前决策"之间的数学联系，是几乎所有强化学习算法的理论基础。

## 一句话定义

>"从现在往前看，期望累积奖励 = 当前一步的奖励 + 折扣后的未来期望奖励"——这个等式叫 Bellman 方程。

## 为什么重要

Bellman 方程是 RL 的核心。没有它，就没有 Q-learning、TD-learning、Actor-Critic ——所有 RL 算法都是对 Bellman 方程的不同近似方式。

它的意义在于：**把一个无限时域的累积问题，分解成一个"当前一步 + 递归未来"的递归结构**，从而可以用动态规划求解。

## 标准 Bellman 方程（策略 $\pi$ 下）

对于状态值函数 $V^\pi$：

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^\pi(s') \right]$$

等价形式（展开期望）：

$$V^\pi(s) = \sum_a \pi(a|s) \, Q^\pi(s,a)$$

对于动作值函数 $Q^\pi$：

$$Q^\pi(s,a) = \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') \, Q^\pi(s',a') \right]$$

## Bellman 最优方程

最优策略 $\pi^*$ 对应的值函数满足：

$$V^*(s) = \max_a \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^*(s') \right]$$

$$Q^*(s,a) = \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma \max_{a'} Q^*(s',a') \right]$$

注意 $\max$ 替代了策略 $\pi$ 的期望——这是"最优"的核心。

## 直观理解

```
时刻 t:  V(s_t) = E[ R(s_t,a_t) + γ·V(s_{t+1}) ]
         ↑          ↑        ↑         ↑
       现在      当前奖励  折扣因子   未来价值
```

把这个等式想象成：从 $s_t$ 出发的总价值 = 马上能拿到的 + 打折后的"从下个状态继续"。

## 主要求解方法

### 1. Value Iteration

直接用 Bellman 最优方程做迭代：

$$V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V_k(s') \right]$$

收敛到 $V^*$ 后，用 $\arg\max$ 提取 $\pi^*$。

适用场景：$P$ 和 $R$ 已知（model-based）。

### 2. Policy Iteration

两步交替：

```
1. Policy Evaluation: 给定 π，计算 V^π（解线性方程组）
2. Policy Improvement: 用 V^π 更新 π（ greedy 选取最大化 Q 的动作）
3. 重复直到 π 不变
```

每轮迭代都比 Value Iteration 收敛更快（策略比值函数更稳定）。

### 3. Q-Learning（无模型）

不知道 $P$ 和 $R$ 时，通过采样交互来近似 Bellman 最优方程：

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ R + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]$$

这就是 TD(0) 的 off-policy 版本，Q-learning 的核心更新。

### 4. TD Learning（时序差分）

介于 DP（需要完整模型）和 MC（需要完整 episode）之间：

$$V(s) \leftarrow V(s) + \alpha \left[ R + \gamma V(s') - V(s) \right]$$

用一步采样的"TD 目标"来更新当前估计，不需要模型，也不需要等 episode 结束。

## 在人形机器人 RL 中的意义

人形机器人 RL 训练的本质：

```
构建 MDP → 与仿真器交互采样 → 用 TD-learning / Q-learning 近似 Bellman 最优方程
→ 得到最优策略 π^*
```

| 步骤 | 具体操作 |
|------|---------|
| 建模 | 定义 $S, A, R$（$P$ 由仿真器提供）|
| 求解 | 用 PPO/SAC 等算法近似解 Bellman 最优方程 |
| 提取策略 | 收敛后 $\pi(a|s)$ 即为控制策略 |

## 常见坑

### 折扣因子 $\gamma$ 的影响

- $\gamma$ 太小（0.9）→ 策略短视，只在乎眼前奖励，步态不稳
- $\gamma$ 太大（0.999）→ 策略偏保守，训练慢；但太大会让不稳定的 MDP 发散
- 人形 locomotion 通常 $\gamma = 0.99 \sim 0.995$

### Bootstrapping

TD 方法用 $V(s')$ 来更新 $V(s)$，这叫 bootstrapping。好处是 sample efficiency 高；坏处是如果 $V(s')$ 估计不准，误差会传播。

### 高维状态空间的近似误差

DP 需要遍历整个状态空间——对人形机器人不可能（状态空间连续、高维）。

解决方案：用函数近似器（神经网络）代替查表，即 Deep RL 的核心。

## 参考来源

- Bellman, *Dynamic Programming* (1957) — Bellman 方程原始提出
- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.) — RL 中 Bellman 方程的标准教材
- Bertsekas, *Dynamic Programming and Optimal Control* — 动态规划理论权威参考

## 关联页面

- [MDP](./mdp.md) — Bellman 方程是 MDP 的求解理论
- [Reinforcement Learning](../methods/reinforcement-learning.md) — 所有 RL 算法都是对 Bellman 方程的近似
- [Optimal Control](../concepts/optimal-control.md) — Bellman 最优方程和 Pontryagin 极大值原理是 OC 的两个理论分支

## 推荐继续阅读

- [Sutton & Barto — Chapter 4: Dynamic Programming](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
- [Sutton & Barto — Chapter 6: TD Learning](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
- [Deep RL Course — Bellman Equation Basics](https://stable-baselines.readthedocs.io/en/master/)
