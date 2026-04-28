---
type: method
tags: [hierarchical-control, embedding, gan, xbpeng]
status: complete
updated: 2026-04-28
related:
  - ./amp-reward.md
  - ../entities/mimickit.md
  - ./smp.md
sources:
  - ../../sources/papers/ase.md
summary: "ASE (Adversarial Skill Embeddings) 通过对抗学习在潜空间中压缩动作风格，实现层次化控制与复杂任务组合。"
---

# ASE: 对抗性技能嵌入

**ASE** 将生成对抗思想与层次化强化学习相结合，旨在从大规模无标注运动数据中学习通用的技能表示。

## 核心架构：两阶段学习
1. **技能发现 (Skill Discovery)**：
   - 训练一个低层策略（Low-level Policy），其输入除了状态 $ 还有潜变量 $。
   - 潜变量 $ 被映射到特定的动作风格。
   - 判别器确保生成的动作分布与参考数据集一致。
2. **任务适配 (Task Adaptation)**：
   - 低层策略被冻结。
   - 训练高层策略（High-level Policy）在潜空间 $ 中进行搜索，以完成特定任务。

## 主要技术路线
| 阶段 | 关键技术 | 目标 |
|------|---------|------|
| **预训练** | Adversarial Information Bottleneck | 学习解耦且稠密的技能潜空间 |
| **潜空间建模** | [SE(3) 表示](../formalizations/se3-representation.md) Embedding | 将技能约束在超球面上，便于高层搜索 |
| **下游训练** | Skill Chaining | 组合多个基本技能完成长程任务 |

## 关联页面
- [[amp-reward]] — ASE 沿用了 AMP 的判别器结构。
- [[mimickit]] — 核心集成框架。
- [[smp]] — 下一代生成式先验。

## 参考来源
- [sources/papers/ase.md](../../sources/papers/ase.md)
