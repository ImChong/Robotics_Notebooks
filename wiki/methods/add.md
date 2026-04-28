---
type: method
tags: [gan, discriminator, artifacts]
status: complete
updated: 2026-04-28
sources:
  - ../../sources/papers/add.md
summary: "ADD (Adversarial Differential Discriminator) 通过微分判别器结构消除对抗模仿学习中的运动伪影。"
---

# ADD: 对抗性微分判别器

**ADD** 是对 AMP 架构的进一步优化。

## 痛点：对抗伪影
AMP 有时会生成“飘忽”或“滑步”的动作，因为判别器对静态位姿的绝对位置不够敏感，或者在时序平滑性上判断不足。

## 微分机制
ADD 的判别器不直接看状态 $，而是看状态的**时空差分** $\Delta s$。这强制生成器必须在动作的“变化率”上与参考数据保持一致，从而消除了大部分视觉上的不自然伪影。

## 参考来源
- [sources/papers/add.md](../../sources/papers/add.md)
