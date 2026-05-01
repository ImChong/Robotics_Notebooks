---
type: formalization
description: 通过学习数据的底层分布来生成新样本的一类概率模型，包括 GANs, VAEs, Normalizing Flows 和 Diffusion Models。
---

# 生成式模型基础 (Generative Foundations)

> **一句话定义**: 通过学习数据的底层分布来生成新样本的一类概率模型，包括 GANs, VAEs, Normalizing Flows 和 Diffusion Models。

## 数学表达

生成式模型的目标是学习或近似真实数据分布 $p_{data}(\mathbf{x})$。

对于潜变量模型（如 VAE），其变分下界（ELBO）为：
$$\mathcal{L}(\theta, \phi; \mathbf{x}) = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) || p(\mathbf{z}))$$

对于扩散模型，其逆向过程可建模为：
$$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t), \Sigma_\theta(\mathbf{x}_t, t))$$

## 核心模型类别

### 1. 变分自编码器 (VAEs)
通过变分推断学习潜空间（Latent Space）分布。
- **核心**: 证据下界 (ELBO) 的最大化。
- **优点**: 潜空间连续且可解释，适合表征学习。

### 2. 生成对抗网络 (GANs)
通过博弈论的方法，生成器与判别器相互对抗。
- **核心**: 极小极大博弈 (Minimax Game)。
- **应用**: 高质量图像生成、Sim2Real 中的视觉迁移。

### 3. 扩散模型 (Diffusion Models)
通过向数据添加噪声并学习逆向去噪过程来生成数据。
- **核心**: 迭代去噪分数匹配 (Iterative Denoising Score Matching)。
- **应用**: [Diffusion Policy](../methods/diffusion-policy.md) 已成为目前机器人操作（Manipulation）领域的主流方法。

## 在机器人中的应用

- **轨迹生成 (Trajectory Generation)**: 利用扩散模型生成平滑且多样的动作序列。
- **环境建模 (World Models)**: 利用 VAE 或 GANs 学习环境的潜空间表征，在潜空间内进行规划（如 Dreamer）。
- **触觉模拟 (Tactile Simulation)**: 生成逼真触觉传感器数据。

## 关联页面
- [Diffusion Policy](../methods/diffusion-policy.md)
- [潜空间想象 (Latent Imagination)](../concepts/latent-imagination.md)
- [生成式世界模型](../methods/generative-world-models.md)

## 参考来源
- [Understanding Deep Learning (Prince, 2023)](../../sources/books/udl_book.md)

## 推荐继续阅读
- [Generative Deep Learning (David Foster)](https://www.oreilly.com/library/view/generative-deep-learning/9781098134174/)
