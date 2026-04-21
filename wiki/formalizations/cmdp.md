---
type: formalization
tags: [rl, safety, control, optimization, math]
status: complete
updated: 2026-04-20
related:
  - ../concepts/safety-filter.md
  - ../methods/reinforcement-learning.md
  - ../methods/safe-rl.md
  - ./mdp.md
  - ./bellman-equation.md
sources:
  - ../../sources/papers/privileged_training.md
summary: "Constrained MDP（CMDP）在标准 MDP 基础上增加了显式约束项，要求在满足约束代价阈值的前提下最大化累积奖励。"
---

# Constrained MDP (CMDP)

**CMDP（Constrained Markov Decision Process）** 是一种强化学习的形式化框架，旨在解决在某些预定义约束（如安全性、能量消耗、物理限制）下寻找最优策略的问题。

## 数学定义

一个 CMDP 通常由一个七元组 $(S, A, P, R, C, \hat{c}, \gamma)$ 定义：

- $S, A, P, R, \gamma$：与标准 MDP 相同，分别代表状态空间、动作空间、转移概率、奖励函数和折扣因子。
- $C = \{c_1, ..., c_k\}$：**约束代价函数（Cost Functions）**，$c_i: S \times A \to \mathbb{R}$ 描述了每个动作带来的“成本”（如碰撞风险）。
- $\hat{c} = \{\hat{c}_1, ..., \hat{c}_k\}$：**约束阈值（Cost Thresholds）**。

### 优化目标

CMDP 的目标是找到一个策略 $\pi$，使得在期望累积成本满足阈值的前提下，最大化累积奖励：

$$
\begin{aligned}
\max_{\pi} \quad & E_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) \right] \\
\text{s.t.} \quad & E_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t c_i(s_t, a_t) \right] \leq \hat{c}_i, \quad \forall i
\end{aligned}
$$

## 为什么重要

在机器人应用中，单纯的 MDP 往往无法保证安全性：
- **奖励工程的局限**：如果仅在奖励函数中加入惩罚项（Reward Shaping），通常很难找到平衡权重，导致机器人要么太激进（撞坏），要么太保守（不动）。
- **硬性安全要求**：在真实硬件部署中，某些约束（如温度上限、关节限位、不跌倒）是必须严格遵守的。

CMDP 提供了一种**数学上完备**的方式来处理这些冲突。

## 常用求解方法

1. **拉格朗日乘子法（Lagrangian Methods）**：
   将约束问题转化为无约束问题：
   $$ L(\pi, \lambda) = J_R(\pi) - \sum \lambda_i (J_{c_i}(\pi) - \hat{c}_i) $$
   交替更新策略参数 $\theta$ 和拉格朗日乘子 $\lambda$。
2. **约束策略优化（CPO, Constrained Policy Optimization）**：
   在每一步更新中，直接求解一个带线性约束的二次规划问题，确保策略更新始终留在安全集内。
3. **安全层（Safety Layers）**：
   在策略输出后添加一个投影层，强制将违反约束的动作映射回安全区域。

## 关联页面
- [Safety Filter](../concepts/safety-filter.md)
- [Reinforcement Learning](../methods/reinforcement-learning.md)
- [Safe RL](../methods/safe-rl.md)
- [MDP 形式化](./mdp.md)
- [Bellman 方程](./bellman-equation.md)

## 参考来源
- Altman, E. (1999). *Constrained Markov Decision Processes*.
- Achiam et al. (2017). *Constrained Policy Optimization*.
