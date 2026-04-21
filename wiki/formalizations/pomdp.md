---
type: formalization
tags: [rl, math, decision-making, perception, uncertainty]
status: complete
updated: 2026-04-21
related:
  - ./mdp.md
  - ../concepts/state-estimation.md
sources:
  - ../../sources/papers/rl_foundation_models.md
summary: "部分可观测马尔可夫决策过程（POMDP）描述了机器人无法获取完整状态信息，只能通过嘈杂观测进行概率推理的决策框架，是状态估计与鲁棒控制的理论基石。"
---

# Partially Observable MDP (POMDP)

在真实的机器人应用中，我们永远无法获取完美的、全知全能的状态 $s$。传感器噪声、视觉遮挡和未知的物理参数使得系统处于**部分可观测 (Partial Observability)** 状态。

## 数学定义

POMDP 由一个六元组 $(S, A, T, R, \Omega, O)$ 描述：
- $S, A, T, R$：与标准 [MDP](./mdp.md) 相同。
- $\Omega$：观测空间（Observation Space）。
- $O(o|s', a)$：观测模型，描述在执行动作 $a$ 到达 $s'$ 后，观察到 $o$ 的概率。

## 核心机制：Belief State

由于 $s$ 不可见，智能体维护一个**信念状态 (Belief State)** $b_t$，它是关于当前可能状态的概率分布：
$$ b_t(s) = P(s_t = s | o_{1:t}, a_{1:t-1}) $$

## 在机器人中的应用

1. **状态估计**：[EKF](../concepts/state-estimation.md) 本质上是在连续空间中维护一个高斯分布的 Belief State。
2. **主动感知 (Active Perception)**：机器人通过移动摄像头来减少 $b_t$ 的方差。

## 关联页面
- [MDP 形式化](./mdp.md)
- [State Estimation (状态估计)](../concepts/state-estimation.md)

## 参考来源
- Kaelbling, L. P., et al. (1998). *Planning and learning in partially observable stochastic domains*.
