---
type: method
tags: [score-matching, diffusion, generative-prior, humanoid, unitree-g1]
status: complete
updated: 2026-04-28
related:
  - ../entities/mimickit.md
  - ../entities/protomotions.md
  - ./amp-reward.md
  - ./ase.md
  - ../entities/unitree-g1.md
sources:
  - ../../sources/papers/smp.md
summary: "SMP (Score-Matching Motion Priors) 通过预训练扩散模型作为“冻结奖励器”，实现了高效、可组合且无需原始数据的运动模仿学习。"
---

# SMP: 基于得分匹配的可复用运动先验

**SMP** 代表了从对抗模仿学习（如 [AMP](./amp-reward.md)）向生成式先验引导学习的范式演进。它将复杂的运动分布建模为一个连续的得分场（Score Field），并以此指导 RL 策略。

## 核心技术路线

### 1. 冻结的扩散模型作为奖励
与 AMP 不同，SMP 不需要判别器与策略共同训练。
- **预训练**：在动作数据集上预训练一个扩散模型。
- **冻结奖励**：在 RL 阶段，扩散模型被冻结，不再需要原始数据集。

### 2. SDS (Score Distillation Sampling)
SMP 借鉴了文本转图像领域的 SDS 技术：
- 策略生成的动作片段被添加噪声，然后输入扩散模型。
- 扩散模型预测噪声，其预测值与实际添加噪声的差异被转化为奖励：
  $$r_{SMP} = -\mathbb{E}_{t, \epsilon} [ \| \epsilon - \epsilon_\theta(x_t; t) \|^2 ]$$

### 3. ESM (Ensemble Score-Matching)
通过在多个噪声水平上聚合评估结果，降低奖励方差。

## 主要技术路线
| 阶段 | 关键技术 | 说明 |
|------|---------|------|
| **先验建模** | Diffusion / Score-based Model | 学习专家动作的概率密度梯度场 |
| **策略优化** | Score Distillation | 将生成模型的得分作为 RL 的外部奖励信号 |
| **初始化** | GSI (Generative State Initialization) | 利用生成模型代替传统 RSI 数据集采样 |

## 关联页面
- [[protomotions]] — 提供大规模并行训练支持。
- [概率流形式化](../formalizations/probability-flow.md)
- [AMP](./amp-reward.md) — 传统的判别器路线。
- [Unitree G1](../entities/unitree-g1.md) — SMP 已在此硬件上完成真机验证。
- [Diffusion Policy](./diffusion-policy.md) — 同样基于扩散模型，但 SMP 侧重于作为先验奖励。
- [Sim2Real](../concepts/sim2real.md) — SMP 提供的结构化先验增强了迁移鲁棒性。

## 参考来源
- [sources/papers/smp.md](../../sources/papers/smp.md)
- Mu et al., *SMP: Reusable Score-Matching Motion Priors for Physics-Based Character Control*, 2026.
