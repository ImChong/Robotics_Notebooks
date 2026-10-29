---
type: method
tags: [rl, model-based, mbpo, sample-efficiency, sac, world-model]
status: complete
updated: 2026-10-29
summary: "MBPO 用真实数据训练集成动力学模型，只从真实状态出发展开短期虚拟轨迹，混入 Replay Buffer 并用 SAC 更新策略，以短 rollout 限制模型误差累积。"
related:
  - ./model-based-rl.md
  - ../formalizations/mdp.md
  - ./sac.md
  - ./ppo.md
  - ./td3.md
  - ../concepts/bptt.md
  - ../concepts/latent-imagination.md
  - ../comparisons/model-based-vs-model-free.md
  - ../entities/paper-dreamer-latent-imagination.md
  - ../entities/paper-pets-probabilistic-dynamics-mpc.md
  - ../entities/paper-supermimic.md
sources:
  - ../../sources/blogs/wechat_senlanke_supermimic_model_based_bptt_review_2026-10-29.md
  - ../../sources/papers/model_based_rl.md
---

# MBPO（Model-Based Policy Optimization）

**MBPO**（Janner et al., 2019）是「学习型动力学模型 + 无模型策略优化」的典型组合：模型只负责生成**短期**虚拟数据，策略仍由 [SAC](./sac.md) 更新。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MBPO | Model-Based Policy Optimization | 短期模型 rollout + SAC 的 Model-Based RL |
| SAC | Soft Actor-Critic | MBPO 使用的 off-policy 策略更新器 |
| MBRL | Model-Based Reinforcement Learning | 显式学习/使用环境模型的 RL |

## 为什么重要

- 用学习型模型换更高样本效率，同时靠短 rollout 控制模型误差，是 Model-Based RL 里「价值函数路线」的代表（对照 Dreamer 的「梯度穿模型」路线）。

## 主要技术路线

### 流程

按 [senlanke 复习文章](../../sources/blogs/wechat_senlanke_supermimic_model_based_bptt_review_2026-10-29.md) §3.3：

1. 收集真实轨迹，存入 Replay Buffer；
2. 用真实轨迹训练 Dynamics Model；
3. 从 Replay Buffer 采样初始状态；
4. 用当前 Policy + Dynamics Model 生成短期虚拟轨迹；
5. 用虚拟轨迹通过 SAC 更新 Actor 和 Critic；
6. 再收集真实轨迹、更新模型，循环。

模型无需收敛后才训策略，二者交替更新，新数据持续修正模型误差。

### 为什么用短 rollout

模型误差会在多步预测中累积；MBPO 用较短预测轨迹并持续用真实交互修正模型来限制它（同文 §3.4、§7.2）。

### 与相邻方法的区别

| 方法 | 策略如何利用模型 |
|------|------------------|
| MBPO | 虚拟轨迹喂给 SAC，**不**让梯度穿过模型（价值函数路线） |
| [Dreamer](../entities/paper-dreamer-latent-imagination.md) | 梯度可经 World Model 反传回 Actor（见 [BPTT](../concepts/bptt.md)） |
| [PETS](../entities/paper-pets-probabilistic-dynamics-mpc.md) | 用模型规划（MPC），不训练 Actor |
| [SuperMimic](../entities/paper-supermimic.md) | 联合训练策略与可微世界模型，短时想象 rollout + 解析梯度 |

更完整的算法对比见 [Model-Based RL](./model-based-rl.md#mbpo-model-based-policy-optimization-janner-et-al-2019)。

## 关联页面

- [MDP（形式化）](../formalizations/mdp.md)
- [Model-Free vs Model-Based（对比）](../comparisons/model-based-vs-model-free.md)
- [TD3](./td3.md) — Model-Free off-policy 对照
- [Latent Imagination](../concepts/latent-imagination.md)

## 参考来源

- [Model-Based RL 来源归档（Dreamer / MBPO / PETS / TD-MPC）](../../sources/papers/model_based_rl.md)
- [senlanke 公众号复习文章归档](../../sources/blogs/wechat_senlanke_supermimic_model_based_bptt_review_2026-10-29.md)

- Janner et al. (2019). *When to Trust Your Model: Model-Based Policy Optimization*. <https://arxiv.org/abs/1906.08253>
