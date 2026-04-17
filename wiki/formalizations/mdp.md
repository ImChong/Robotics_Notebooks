---
type: formalization
tags: [rl, mdp, markov, decision-process]
status: complete
related:
  - ./bellman-equation.md
  - ../methods/reinforcement-learning.md
  - ../methods/policy-optimization.md
sources:
  - ../../sources/papers/policy_optimization.md
---

# Markov Decision Process (MDP)

**马尔可夫决策过程**：在离散时间步中，智能体根据当前状态选择动作，环境根据转移概率回应新状态和奖励的数学框架，是强化学习的理论基础。

## 一句话定义

>"智能体在不确定环境中做决策——每一步选动作，获得奖励，下一步到达哪个状态由转移概率决定"——这一过程的数学形式化。

## 为什么重要

MDP 是 RL 的根基。RL 中所有算法——Policy Gradient、SARSA、Q-learning——本质上都是在解 MDP。

不懂 MDP，就不懂为什么 RL 要"最大化累积折扣奖励"，也不知道为什么值函数、策略、模型这些概念会出现。

## 核心组成

### 五元组

$$M = (S, A, P, R, \gamma)$$

| 符号 | 含义 | 人形机器人例子 |
|------|------|----------------|
| $S$ | 状态空间 | 关节角度、角速度、IMU、接触状态 |
| $A$ | 动作空间 | 各关节力矩/位置指令 |
| $P(s'|s,a)$ | 转移概率 | 给定当前状态和动作，下一状态是什么 |
| $R(s,a,s')$ | 奖励函数 | 走得稳给正奖励、摔倒给负奖励 |
| $\gamma \in [0,1)$ | 折扣因子 | 近期奖励重要还是远期奖励重要 |

### 轨迹

一个完整的交互序列（episode）：

$$s_0 \xrightarrow{a_0} s_1 \xrightarrow{a_1} s_2 \xrightarrow{a_2} \cdots \xrightarrow{a_{T-1}} s_T$$

累积折扣奖励：

$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k}$$

### 策略 $\pi$

策略 $\pi(a|s)$ 是在状态 $s$ 下选择动作 $a$ 的概率分布：

$$\pi: S \rightarrow \text{Prob}(A)$$

- **确定性策略**：$a = \pi(s)$，输出单一动作
- **随机策略**：$\pi(a|s)$，输出动作分布（更适合部分可观测或高维连续动作空间）

### 值函数

**状态值函数** $V^\pi(s)$：从状态 $s$ 开始，按策略 $\pi$ 行事的期望累积折扣奖励

$$V^\pi(s) = \mathbb{E}_\pi\left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k} \middle| s_t = s \right]$$

**动作值函数** $Q^\pi(s,a)$：从状态 $s$ 出发，执行动作 $a$ 后按策略 $\pi$ 行事的期望累积折扣奖励

$$Q^\pi(s,a) = \mathbb{E}_\pi\left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k} \middle| s_t = s, a_t = a \right]$$

两者关系：

$$V^\pi(s) = \sum_a \pi(a|s) \, Q^\pi(s,a)$$

### 最优性

最优策略 $\pi^*$ 满足：

$$V^{\pi^*}(s) \geq V^\pi(s), \quad \forall s, \pi$$

等价地：

$$Q^{\pi^*}(s,a) \geq Q^\pi(s,a), \quad \forall s,a$$

最优值函数记为 $V^*$ 和 $Q^*$。

## Bellman 方程

值函数满足递归关系（Bellman 方程）：

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^\pi(s') \right]$$

最优值函数（Bellman 最优方程）：

$$V^*(s) = \max_a \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^*(s') \right]$$

这是所有 RL 算法的起点——解这个方程，就得到了最优策略。

## 人形机器人中的 MDP

| MDP 元素 | 人形 locomotion 具体化 |
|----------|------------------------|
| 状态 $s$ | $(q, \dot{q}, r, \dot{r}, c, \omega)$ — 关节位置/速度、基座位置/速度、质心位置、角速度 |
| 动作 $a$ | 关节力矩指令或位置指令 |
| 转移 $P$ | 物理仿真器（MuJoCo/IsaacGym）给出 |
| 奖励 $R$ | 行走速度 + 稳定性 + 能耗 + 摔倒惩罚的加权组合 |
| 折扣 $\gamma$ | 0.99–0.995（步态任务偏短期）|

典型的 RL 训练就是通过与仿真器交互，隐式地解这个 MDP 的 Bellman 最优方程。

## 和 RL 的关系

```
MDP（数学框架）
    ↓ 精确解（已知模型）→ 动态规划 → Value Iteration / Policy Iteration
    ↓ 近似解（模型未知）→ RL → Q-learning / Policy Gradient / Actor-Critic
```

所有 RL 方法都是对 Bellman 方程的近似求解。

## 参考来源

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.) — MDP 与 RL 标准教材，Chapter 3
- Puterman, *Markov Decision Processes: Discrete Stochastic Dynamic Programming* — MDP 理论权威参考
- Bellman, *Dynamic Programming* (1957) — MDP 理论基础

## 关联页面

- [Reinforcement Learning](../methods/reinforcement-learning.md) — MDP 是 RL 的理论根基
- [Optimal Control](../concepts/optimal-control.md) — OCP 和 MDP 的关系：OCP 是确定性的，RL 是随机性的；两者都是"最优决策"问题
- [Reward Design](../concepts/reward-design.md) — MDP 中奖励函数的设计直接决定学到的策略
- [POMDP](./pomdp.md) — MDP 加观测模型的扩展，真实机器人系统的更准确框架

## 推荐继续阅读

- [Spinning Up — Part 1: Key Concepts of RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
- [Sutton & Barto — Chapter 3: Finite MDPs](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
