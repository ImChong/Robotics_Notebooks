---
type: method
tags: [gan, discriminator, artifacts, xbpeng]
status: complete
updated: 2026-04-28
related:
  - ../entities/protomotions.md
  - ./amp-reward.md
  - ./smp.md
  - ../entities/mimickit.md
sources:
  - ../../sources/papers/add.md
summary: "ADD (Adversarial Differential Discriminator) 通过微分判别器结构消除对抗模仿学习中的运动伪影，提升动作的物理真实感。"
---

# ADD: 对抗性微分判别器

**Adversarial Differential Discriminator (ADD)** 是对 [[amp-reward]] 架构的重要改进，旨在解决生成动作中的物理不一致性。

## 技术背景
在对抗模仿学习中，判别器通常直接观察代理的状态（如关节位置和速度）。然而，这种直接观察往往会导致判别器在处理静态姿态时过于宽松，从而产生如“滑步”（Foot Sliding）或“关节漂移”等运动伪影。

## 核心机制：微分判别
ADD 的核心思想是将判别器的输入从绝对状态 $ 转变为状态的**时空差分** $\Delta s$。
- **时间差分**：捕捉动作的瞬时变化率，强制生成器在速度和加速度层面与参考数据对齐。
- **空间差分**：捕捉身体各部位之间的相对位移，确保肢体协调性。

## 主要技术路线
| 阶段 | 关键技术 | 目的 |
|------|---------|------|
| **特征提取** | Differential Encoding | 提取动作的差分特征，过滤低频位置噪声 |
| **对抗训练** | [接触动力学](../concepts/contact-dynamics.md) Minimax Optimization | 策略与微分判别器进行对抗，学习高频动作细节 |
| **正则化** | Gradient Penalty | 确保判别器梯度的稳定性，防止训练坍缩 |

## 优势与改进
- **消除伪影**：显著减少了物理模拟中常见的不自然抖动。
- **细节增强**：能够捕捉到人类动作中微小的、具有表现力的动态特征。

## 关联页面
- [[protomotions]] — 提供大规模并行训练支持。
- [[amp-reward]] — ADD 的基础框架。
- [[smp]] — 另一种通过得分匹配解决稳定性的方案。
- [[mimickit]] — ADD 的官方实现框架（集成于 [[mimickit]] 与 [[protomotions]]）。

## 参考来源
- [sources/papers/add.md](../../sources/papers/add.md)
- Peng et al., *ADD: Adversarial Differential Discriminator for Physics-Based Character Control*, SIGGRAPH 2024.
