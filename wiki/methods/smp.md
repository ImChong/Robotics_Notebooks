---
type: method
tags: [score-matching, diffusion, generative-prior]
status: complete
updated: 2026-04-28
sources:
  - ../../sources/papers/smp.md
summary: "SMP 使用得分匹配 (Score-Matching) 训练运动先验，为 RL 训练提供更稳定的梯度导向。"
---

# SMP: 得分匹配运动先验

**Score-Matching Motion Priors (SMP)** 将生成式模型（如 Diffusion/Score-based models）的最新进展引入了物理控制。

## 原理
- 学习动作数据集的**梯度场 (Score Field)**：$\nabla_s \log p(s)$。
- 在 RL 训练中，SMP 不再仅仅给出一个标量奖励，而是给出一个**梯度方向**。
- 策略学习可以直接被“推”向概率密度更高的动作区域。

## 优势
- 比传统的 GAN 奖励（AMP）更稳定，避免了判别器的塌陷问题。
- 为高维动作空间提供了更丰富的指导信息。

## 参考来源
- [sources/papers/smp.md](../../sources/papers/smp.md)
