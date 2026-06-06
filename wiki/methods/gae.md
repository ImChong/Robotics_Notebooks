---
type: method
tags: [rl, policy-optimization, math, optimization]
status: complete
updated: 2026-05-29
related:
  - ./policy-optimization.md
  - ./reinforcement-learning.md
sources:
  - ../../sources/papers/policy_optimization.md
  - ../../sources/papers/intentional_streaming_rl.md
related:
  - ./intentional-updates-streaming-rl.md
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

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PPO | Proximal Policy Optimization | 人形/足式 locomotion 中最常用的 on-policy 策略梯度算法 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |

## 为什么重要

在 [PPO](./policy-optimization.md) 中使用 GAE 可以显著稳定训练过程。其优势估计的准确性直接决定了 [Bellman 方程](../formalizations/bellman-equation.md) 迭代中的梯度平滑度，使模型在面对长时域任务时更容易收敛。

**与流式 RL 的交叉：** [Intentional Updates（流式 RL）](./intentional-updates-streaming-rl.md) 在 **TD($\lambda$) + eligibility traces** 设定下，把「意图更新」写成对 **近期多状态预测折扣 RMS 变化** 与 $|\delta_t|$ 成比例——trace 几何必须与 GAE 的多步信用分配一致，否则 naive 用 $\mathbf{z}_t$ 范数归一化会导致 trace 变长时更新反而缩小。读 GAE 时若关心 **batch=1、无 replay** 的在线设定，应连同 intentional TD($\lambda$) 一并理解。

## 关联页面
- [Reinforcement Learning](./reinforcement-learning.md)
- [Policy Optimization](./policy-optimization.md)

## 参考来源
- Schulman, J., et al. (2015). *High-Dimensional Continuous Control Using Generalized Advantage Estimation*.
- [sources/papers/intentional_streaming_rl.md](../../sources/papers/intentional_streaming_rl.md) — intentional TD($\lambda$) 与 GAE trace 几何
