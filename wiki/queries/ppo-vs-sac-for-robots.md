---
type: query
tags: [ppo, sac, rl, policy-optimization, locomotion, manipulation]
status: complete
summary: PPO 与 SAC 在机器人 RL 中的实践对比，覆盖 on-policy vs off-policy 核心差异、适用场景与调参要点。
sources:
  - ../../sources/papers/policy_optimization.md
related:
  - ../methods/policy-optimization.md
  - ../methods/reinforcement-learning.md
  - ../tasks/locomotion.md
  - ../formalizations/gae.md
---

# PPO vs SAC：机器人 RL 实践对比

> **Query 产物**：本页由以下问题触发：「机器人 RL 用 PPO 还是 SAC？有什么实践区别？」
> 综合来源：[Policy Optimization](../methods/policy-optimization.md)、[Reinforcement Learning](../methods/reinforcement-learning.md)、[Locomotion](../tasks/locomotion.md)、[GAE](../formalizations/gae.md)

## TL;DR 决策规则

| 场景 | 推荐算法 | 理由 |
|------|---------|------|
| 仿真大规模并行训练（legged_gym / Isaac Lab） | **PPO** | On-policy 与并行采样天然契合，数据量巨大时单样本复用劣势不再是主要瓶颈 |
| 真实机器人 / 样本成本高 | **SAC** | Off-policy 经验回放复用每条数据，样本效率比 PPO 高 10–100x |
| 首次实验 / 快速验证 reward 设计 | **PPO** | 超参数少，收敛曲线平滑，更容易定位问题 |
| 操作任务 / 精细控制 | **SAC** | 最大熵框架探索更充分，连续动作空间表现好 |
| 需要运动风格（MoCap 参考） | **PPO + AMP** | AMP 判别器 reward 驱动自然步态，底层优化器仍用 PPO |

---

## 核心差异对比表

| 维度 | PPO | SAC |
|------|-----|-----|
| **策略类型** | On-policy（随机策略） | Off-policy（随机策略，最大熵） |
| **样本效率** | 低（数据用完即丢，不重复利用） | 高（经验回放，每条数据可多次采样） |
| **训练稳定性** | 高（clip 约束防止策略崩溃） | 中高（双 Critic 稳定 Q 值估计） |
| **超参数敏感度** | 低（`clip_range`、`n_steps` 不敏感） | 中（`buffer_size`、`batch_size`、温度 α 需要调） |
| **大规模并行仿真** | 优（并行采集 on-policy 数据，利用率 100%） | 劣（off-policy 并行化收益递减，replay buffer 会饱和） |
| **真实机器人** | 劣（数据利用率低，样本不足） | 优（经验回放大幅降低所需交互次数） |
| **探索能力** | 中（通过熵正则化，但 clip 限制更新幅度） | 高（熵最大化目标天然鼓励探索） |
| **典型应用** | locomotion 仿真训练、AMP 步态 | 操作任务、真实机器人微调、少样本场景 |

---

## 算法原理简述

### PPO

PPO 用 clip 操作约束每次更新的策略比率，防止一次更新过大导致策略崩溃：

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta),\ 1-\varepsilon,\ 1+\varepsilon)\hat{A}_t\right)\right]$$

其中 $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_{\text{old}}}(a_t|s_t)$，$\hat{A}_t$ 由 GAE 估计。

**在机器人 locomotion 中的标准配置：**
- `num_envs`: 4096–8192
- `n_steps`: 24（每个环境每轮收集步数）
- `clip_range`: 0.2
- `entropy_coef`: 0.01
- `learning_rate`: 1e-3（legged_gym 默认，比 RL 其他场景偏大）

### SAC

SAC 最大化累积奖励和策略熵的加权和，温度参数 α 自动调节探索-利用平衡：

$$J(\pi) = \sum_t \mathbb{E}_{(s_t, a_t)\sim\rho_\pi}\left[r(s_t, a_t) + \alpha\,\mathcal{H}(\pi(\cdot|s_t))\right]$$

用双 Critic 取最小值防止 Q 值过估计，软更新（Polyak averaging）保持目标网络稳定。

**在机器人操作中的标准配置：**
- `buffer_size`: 1M
- `batch_size`: 256
- `learning_rate`: 3e-4（actor 和 critic 同）
- `tau`: 0.005（软更新系数）
- 自动熵调节：`target_entropy = -dim(A)`

---

## 实践决策流程

```
机器人 RL 算法选型
│
├─ 使用仿真 + 大规模并行（>1000 envs）？
│   └─ 是 → PPO（legged_gym / Isaac Lab 标准选择）
│
├─ 真实机器人直接训练 / Fine-tuning？
│   └─ 是 → SAC（样本效率关键）
│
├─ 需要自然步态 / MoCap 风格约束？
│   └─ 是 → PPO + AMP（底层仍是 PPO，加判别器 reward）
│
├─ 操作任务 / 连续精细控制？
│   ├─ 仿真训练 → SAC 或 PPO 均可，SAC 探索更充分
│   └─ 真机直接学习 → SAC
│
└─ 不确定 reward 是否设计合理？
    └─ 先用 PPO 验证（曲线平滑，问题更容易定位）
```

---

## 常见错误与规避

| 错误 | 问题 | 正确做法 |
|------|------|---------|
| 仿真里用 SAC，训练极慢 | Off-policy 在大量并行 envs 下 replay buffer 更新跟不上数据产生速度 | 仿真大规模并行一律用 PPO |
| PPO + 真实机器人，样本耗尽 | On-policy 每条数据用完即丢，真实机器人数据成本极高 | 换 SAC，利用经验回放 |
| SAC 不调 α，探索不足 | 固定 α 往往导致熵过低，策略收敛到局部最优 | 开启自动熵调节（auto entropy tuning） |
| 调 PPO 时改 clip_range | clip_range 对 PPO 不是最敏感参数 | 优先调 `learning_rate`、`n_steps`、`entropy_coef` |
| 直接在真机上跑 PPO | 样本量不足，on-policy 无法在真机上有效训练 | 先仿真用 PPO 训练，真机仅做 sim2real 微调（用 SAC） |

---

## 一句话记忆

> 「仿真大并行 → PPO；真机少样本 → SAC；先验证 reward → PPO；精细探索 → SAC。」

---

## 参考来源

- [Policy Optimization 源文档](../../sources/papers/policy_optimization.md)
- Schulman et al., *Proximal Policy Optimization Algorithms* (2017)
- Haarnoja et al., *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor* (2018)
- Peng et al., *AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control* (2021)
- Rudin et al., *Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning* (2022)

## 关联页面

- [Policy Optimization](../methods/policy-optimization.md) — PPO / SAC / TD3 算法详细推导
- [Reinforcement Learning](../methods/reinforcement-learning.md) — RL 方法全景与核心概念
- [Locomotion](../tasks/locomotion.md) — 足式任务中 PPO 的典型应用
- [GAE](../formalizations/gae.md) — PPO 优势估计基础：Generalized Advantage Estimation
- [RL 算法选型指南](./rl-algorithm-selection.md) — 扩展版选型指南，含 TD3 对比
- [PPO vs SAC 对比页](../comparisons/ppo-vs-sac.md) — 系统性对比：9 个维度 + 伪代码 + 超参数表
