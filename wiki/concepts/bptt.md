---
type: concept
tags: [backpropagation, differentiable-model, model-based-rl, world-model, gradient]
status: complete
updated: 2026-10-29
summary: "BPTT（通过时间反向传播）是把多个时间步的计算图展开后用链式法则回传梯度；在可微 Model-Based RL 中使未来奖励的梯度穿过动力学模型回到 Actor，但依赖模型梯度（而非仅预测值）准确。"
related:
  - ./backpropagation.md
  - ./differentiable-simulation.md
  - ./latent-imagination.md
  - ../methods/model-based-rl.md
  - ../methods/mbpo.md
  - ../methods/ppo.md
  - ../entities/paper-dreamer-latent-imagination.md
  - ../entities/paper-planet-latent-dynamics.md
  - ../entities/paper-supermimic.md
sources:
  - ../../sources/blogs/wechat_senlanke_supermimic_model_based_bptt_review_2026-10-29.md
---

# BPTT（Backpropagation Through Time）

## 一句话定义

BPTT 就是普通[反向传播](./backpropagation.md)作用在**按时间展开**的计算图上：未来累计奖励的梯度沿多个时间步的链路回传到当前动作和 Actor 参数。它只是计算跨时间梯度的方法，本身不创造梯度关系。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BPTT | Backpropagation Through Time | 沿展开的时间步链式回传梯度 |
| RSSM | Recurrent State-Space Model | PlaNet/Dreamer 的可微 latent 动力学模型 |
| MLP | Multi-Layer Perceptron | 多层感知机 |
| GRU | Gated Recurrent Unit | 门控循环单元 |

## 为什么重要

决定 Model-Based RL 里「梯度能不能穿过动力学模型」：这是 Dreamer / SuperMimic 一类方法与 PPO 的根本区别。

## 在 Model-Based RL 中

据 [senlanke 复习文章](../../sources/blogs/wechat_senlanke_supermimic_model_based_bptt_review_2026-10-29.md) §4–5：

- 一步：Actor 参数 → 动作 → 下一状态（经 Dynamics Model）→ Reward，三段梯度用链式法则相乘；训练 Policy 时可冻结模型参数，但梯度仍要经过模型传回 Actor。
- 多步：预测 H 步时，第二步的奖励也依赖第一步动作；各时间步共享 Actor 参数，梯度需累加；预测范围外用 Critic 估计。
- 判据不是「目标里有未来奖励」，而是「梯度是否沿时间穿过动力学模型」：

| 方法 | 目标含未来奖励 | 梯度穿过动力学跨时间传播 |
|------|----------------|--------------------------|
| PPO | 是 | 否 |
| 可微 Model-Based RL（如初代 Dreamer） | 是 | 是 |

- 模型可微：MLP、GRU、RSSM 等通常可自动微分；普通物理仿真器、含接触切换的动力学不一定提供梯度。

## 风险

- 步数长时导数连乘，易梯度消失/爆炸。
- **预测值准≠梯度准**：状态 MSE 小不保证对动作的导数准确，文中给出误差为 0 而梯度方向相反的反例，梯度错则 Actor 会朝错误方向优化。
- 这也是 PPO 在大规模并行仿真（如 [Isaac Lab](../entities/isaac-lab.md)）里仍是主流的原因之一：它既不需要可微模型，也不需要准确的动力学预测。

## 关联页面

- 可微仿真路线：[Differentiable Simulation](./differentiable-simulation.md)；同类思路的人形案例（短时想象 rollout 上的解析梯度）：[SuperMimic](../entities/paper-supermimic.md)。
- BPTT 最初广泛用于 RNN 训练（文中表述）。

## 参考来源

- [senlanke 公众号复习文章归档](../../sources/blogs/wechat_senlanke_supermimic_model_based_bptt_review_2026-10-29.md)
