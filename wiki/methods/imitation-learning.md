---
type: method
tags: [il, behavior-cloning, diffusion-policy, sim2real]
status: complete
---

# Imitation Learning (IL)

**模仿学习**：通过专家演示数据，让机器人学会从状态到动作的映射，核心是“抄”。

## 一句话定义

让机器人看人类/专家怎么做，它就模仿着做。

## 为什么重要

- 纯 RL sample efficiency 低，训练慢
- 很多任务难以定义 reward
- 专家演示提供了高质量数据，可以快速初始化策略

## 核心方法

### 1. 行为克隆（Behavior Cloning, BC）

最简单的 IL：把专家数据当监督学习做。

$$\min_\theta \mathbb{E}_{(s,a) \sim D}[-\log \pi_\theta(a|s)]$$

问题：
- 分布偏移（covariate shift）：训练和测试时状态分布不同
- 错误累积：早期的小错误会不断放大

### 2. DAgger（Dataset Aggregation）

迭代式数据收集：

1. 用当前策略收集数据
2. 让专家标注这些数据
3. 合并到训练集
4. 重复

有效缓解 BC 的分布偏移问题。

### 3. GAIL（Generative Adversarial Imitation Learning）

用 GAN 思想：

- 判别器：区分专家数据 vs 策略数据
- 生成器（策略）：试图骗过判别器

让策略在 reward signal 上接近专家，不需要显式 reward。

### 4. 基于重建的方法

先从演示中提取隐表示或技能 latent，再用于控制。

代表：ASE, CALM, Motion Encoder

## 和强化学习的关系

| | 模仿学习 | 强化学习 |
|--|---------|---------|
| 数据来源 | 专家演示 | 环境交互 |
| 样本效率 | 高 | 低 |
| 可超越专家 | 难 | 可以 |
| Reward 设计 | 不需要 | 需要 |
| 适用范围 | 有专家数据的任务 | 任意可定义 reward |

常见组合策略：
- **IL 初始化 + RL 微调**：先用 IL 训一个不错的初始策略，再用 RL 探索超越专家
- **IL + RL 混合**：如 GAIL 本身就是 IL 和 RL 的混合

## 在人形机器人中的应用

典型 pipeline：

```
专家演示（MoCap/遥控）→ 动作重定向（Retarget）→ 模仿学习训练 → Sim2Real部署
```

代表工作：
- DeepMimic：BC + RL 改进
- MimicKit：j提炼 encoder-decoder 框架
- ASE：对抗技能嵌入
- CALM：latent 方向控制

## 常见问题

- **Retarget 误差**：MoCap 动作不一定适配机器人身体结构
- **分布偏移**：训练分布和真实部署差异
- **技能组合**：如何把多个独立技能串成复杂长序列

## 参考来源

- Ross et al., *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* — DAgger 原论文
- Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion* — 生成式 IL 代表工作
- [sources/papers/imitation_learning.md](../../sources/papers/imitation_learning.md) — DAgger / ACT / Diffusion ingest 摘要
- [sources/papers/locomotion_rl.md](../../sources/papers/locomotion_rl.md) — ASE 与 locomotion 技能学习补充
- [Imitation Learning 论文导航](../../references/papers/imitation-learning.md) — 论文集合

## 关联页面

- [Reinforcement Learning](./reinforcement-learning.md)
- [Whole-Body Control](../concepts/whole-body-control.md)
- [Locomotion](../tasks/locomotion.md)
- [Sim2Real](../concepts/sim2real.md)
- [Foundation Policy（基础策略模型）](../concepts/foundation-policy.md)
- [RL vs Imitation Learning](../comparisons/rl-vs-il.md)（两大策略学习路线的系统性对比）
- [Motion Retargeting](../concepts/motion-retargeting.md) — MoCap 数据需经过 Motion Retargeting 才能作为 IL 的参考轨迹

## 推荐继续阅读

- [Imitation Learning 论文导航](../../references/papers/imitation-learning.md)
- [Diffusion Policy (Blog)](https://diffusion-policy.cs.columbia.edu/)（当前 IL 方向最活跃的生成式路线）
- Ross et al., *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*（DAgger 原论文）
- Peng et al., *AMP: Adversarial Motion Priors for Style-Preserving Physics-Based Humanoid Motion Synthesis*（IL + RL 融合路线）
