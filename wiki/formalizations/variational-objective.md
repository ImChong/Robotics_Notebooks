---
type: formalization
tags: [math, machine-learning, reinforcement-learning, world-models, information-theory]
status: complete
updated: 2026-04-21
related:
  - ../concepts/latent-imagination.md
  - ../methods/model-based-rl.md
sources:
  - ../../sources/papers/rl_foundation_models.md
summary: "变分目标函数（Variational Objective）通过最小化证据下界（ELBO），实现了对复杂高维状态分布的有效近似，是世界模型中 RSSM 训练的核心数学基础。"
---

# Variational Objective (变分目标函数)

在构建具身智能的世界模型（World Models）时，我们面临的核心数学挑战是如何从高维、嘈杂的观测（图像）中提取紧凑的、具有预测性的隐变量表示。**变分目标函数 (Variational Objective)**，特别是其核心 **ELBO (Evidence Lower Bound)**，为这一过程提供了坚实的概率论基础。

## 数学定义：ELBO

假设观测为 $x$，隐变量为 $z$。我们希望最大化对数似然 $\log p(x)$，但这通常是不可计算的。通过引入近似后验分布 $q_\phi(z|x)$，我们可以推导出证据下界：

$$ \log p(x) \geq \mathbb{E}_{q_\phi(z|x)} [ \log p_\theta(x|z) ] - D_{KL} (q_\phi(z|x) \| p(z)) $$

其中：
- **重建项 (Reconstruction Term)**：期望模型能从隐变量 $z$ 中完美还原观测 $x$。
- **正则项 (KL Divergence)**：强制近似后验分布不要偏离先验分布 $p(z)$ 太远。

## 在 RSSM / Dreamer 中的形式化

在时序预测模型中，变分目标函数扩展为：

$$ \mathcal{L}(\theta, \phi) = \sum_{t=1}^T \left( \underbrace{\mathbb{E}_{q_\phi} [ \log p_\theta(o_t | \hat{s}_t) ]}_{\text{观测重建}} + \underbrace{\mathbb{E}_{q_\phi} [ \log p_\theta(r_t | \hat{s}_t) ]}_{\text{奖励预测}} - \beta \underbrace{D_{KL} (q_\phi(s_t | s_{t-1}, a_{t-1}, o_t) \| p_\theta(s_t | s_{t-1}, a_{t-1}))}_{\text{跨时域一致性}} \right) $$

### 核心机制：
- **后验 (Posterior)**：看到了真实观测 $o_t$ 后的状态估计。
- **先验 (Prior)**：仅根据历史和动作进行的“盲目”预测。
- **KL 散度最小化**：强制“梦境（先验）”尽可能地逼近“现实（后验）”。这就是为什么智能体在梦中（潜空间展开）训练出的策略能迁移到现实的原因。

## 信息论视角

变分目标函数本质上是在执行 **信息瓶颈 (Information Bottleneck)** 原则：压缩掉与预测未来无关的背景干扰（如环境光影、墙壁颜色），只保留对任务有用的物理规律。

## 关联页面
- [Latent Imagination (潜空间想象)](../concepts/latent-imagination.md)
- [Model-Based RL](../methods/model-based-rl.md)
- [Action Tokenization](./vla-tokenization.md)

## 参考来源
- Kingma, D. P., & Welling, M. (2013). *Auto-Encoding Variational Bayes*.
- Hafner, D., et al. (2019). *Dream to Control*.
