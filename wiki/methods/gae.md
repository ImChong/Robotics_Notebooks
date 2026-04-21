---
type: method
tags: [rl, policy-optimization, math, optimization]
status: complete
updated: 2026-04-21
related:
  - ./policy-optimization.md
  - ./reinforcement-learning.md
sources:
  - ../../sources/papers/policy_optimization.md
summary: "广义优势估计（GAE）通过引入衰减因子 λ 在偏差与方差之间进行权衡，是目前 PPO 等主流 Policy Gradient 算法中计算优势函数的标准方法。"
---

# Generalized Advantage Estimation (GAE)

**GAE** 解决了强化学习中一个核心痛点：如何准确估计一个动作比平均水平“好多少”（即优势函数 $A(s, a)$），同时保持低方差。

## 主要技术路线

GAE 通过对不同时间跨度的 TD 误差进行加权平均来计算优势：
$$ \hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V $$
其中 $\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是单步 TD 残差。

- **$\lambda = 0$**：退化为单步 TD，低方差但高偏差（依赖 Value 网络准确度）。
- **$\lambda = 1$**：退化为蒙特卡洛（MC）采样，无偏差但极高方差。

## 为什么重要

在 [PPO](./policy-optimization.md) 中使用 GAE 可以显著稳定训练过程。其优势估计的准确性直接决定了 [Bellman 方程](../formalizations/bellman-equation.md) 迭代中的梯度平滑度，使模型在面对长时域任务时更容易收敛。

## 关联页面
- [Reinforcement Learning](./reinforcement-learning.md)
- [Policy Optimization](./policy-optimization.md)

## 参考来源
- Schulman, J., et al. (2015). *High-Dimensional Continuous Control Using Generalized Advantage Estimation*.
