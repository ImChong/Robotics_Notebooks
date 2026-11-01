---
type: method
tags: [rl, policy-optimization, td3, off-policy, deterministic-policy, continuous-control]
status: complete
updated: 2026-10-29
summary: "TD3 是确定性策略的 off-policy actor-critic 算法：用双 Q 取小抑制高估、延迟更新策略、目标策略平滑三招修正 DDPG，是 SAC 之外连续控制的常用基线。"
related:
  - ./sac.md
  - ./ppo.md
  - ./reinforcement-learning.md
  - ./model-based-rl.md
  - ../comparisons/ppo-vs-sac.md
  - ../comparisons/model-based-vs-model-free.md
  - ../formalizations/mdp.md
  - ../formalizations/bellman-equation.md
  - ./mbpo.md
  - ../entities/paper-notebook-fasttd3-simple-fast-and-capable-reinforcement-le.md
sources:
  - ../../sources/blogs/wechat_senlanke_supermimic_model_based_bptt_review_2026-10-29.md
  - ../../sources/papers/policy_optimization.md
---

# TD3（Twin Delayed DDPG）

**TD3** 是 DDPG 的改进版：确定性策略 + Replay Buffer 的 off-policy actor-critic，用三项修正缓解 Q 值高估造成的训练不稳。

## 一句话定义

学一个确定性 Actor 和**两个** Critic，目标值取两个 Q 的较小者，Actor 比 Critic 更新得更慢，并给目标动作加噪声平滑。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| TD3 | Twin Delayed Deep Deterministic Policy Gradient | 双 Q + 延迟更新 + 目标平滑的 DDPG 改进 |
| DDPG | Deep Deterministic Policy Gradient | 确定性策略 off-policy actor-critic |
| Q | Action-Value Function | 状态–动作价值函数 |

## 为什么重要

- 与 [SAC](./sac.md) 一起是 Model-Free off-policy 连续控制的两个常用基线，[PPO](./ppo.md) 之外的主要选择。
- [senlanke 复习文章](../../sources/blogs/wechat_senlanke_supermimic_model_based_bptt_review_2026-10-29.md)将 PPO、SAC、TD3 并列为 Model-Free 的典型算法。

## 主要技术路线

### 核心机制（Fujimoto et al., 2018）

| 修正 | 做法 | 解决的问题 |
|------|------|-----------|
| Clipped Double Q | 两个 Critic，目标取 min | Q 值过高估计 |
| Delayed Policy Update | Critic 更新多次再更新 Actor 与目标网络 | 策略追逐不稳定的 Q |
| Target Policy Smoothing | 目标动作加截断噪声 | Q 对动作的尖峰过拟合 |

### 在本库中的位置

- 与 [SAC](./sac.md) 同属 off-policy 连续控制：TD3 是确定性策略 + 外加探索噪声，SAC 是最大熵随机策略。
- 属于 Model-Free 一侧；[senlanke 文章](../../sources/blogs/wechat_senlanke_supermimic_model_based_bptt_review_2026-10-29.md)将 PPO、SAC、TD3 并列为 Model-Free 典型算法，与 PETS、MBPO、Dreamer 等 Model-Based 算法对照（见 [Model-Based RL](./model-based-rl.md)）。
- 近期工程延伸：[FastTD3](../entities/paper-notebook-fasttd3-simple-fast-and-capable-reinforcement-le.md)。

## 关联页面

- [Model-Free vs Model-Based（对比）](../comparisons/model-based-vs-model-free.md)
- [MDP（形式化）](../formalizations/mdp.md)
- [Bellman 方程（形式化）](../formalizations/bellman-equation.md)
- [MBPO](./mbpo.md) — 同样以 off-policy 算法（SAC）为策略更新器的 Model-Based 对照

## 参考来源

- [Policy Optimization 来源归档（PPO/SAC/TD3/TRPO/Rudin/BRRL）](../../sources/papers/policy_optimization.md)
- [senlanke 公众号复习文章归档](../../sources/blogs/wechat_senlanke_supermimic_model_based_bptt_review_2026-10-29.md)

- Fujimoto, S., et al. (2018). *Addressing Function Approximation Error in Actor-Critic Methods*. <https://arxiv.org/abs/1802.09477>（本页机制总结依据该论文，未在本次 ingest 中逐条重读）
