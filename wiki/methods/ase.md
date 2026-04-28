---
type: method
tags: [hierarchical-control, embedding, gan]
status: complete
updated: 2026-04-28
related:
  - ./amp-reward.md
  - ../entities/mimickit.md
sources:
  - ../../sources/papers/ase.md
summary: "ASE (Adversarial Skill Embeddings) 通过对抗学习在潜空间中压缩动作风格，实现层次化控制。"
---

# ASE: 对抗性技能嵌入

**ASE** 结合了 AMP 的风格学习和层次化控制思想。

## 架构：两阶段训练
1. **预训练 (Pre-training)**：
   - 低层策略 (Low-level Policy) 接收潜编码 $ 并生成动作。
   - 判别器确保生成的动作符合数据集风格。
   - 结果：一个结构化的潜空间，不同的 $ 对应不同的基本运动技能。
2. **下游任务 (Downstream Tasks)**：
   - 训练高层策略 (High-level Policy) 来输出 $，从而驱动低层策略完成特定任务（如足球、格斗）。

## 核心价值
- **组合性**：高层任务不需要再从零学习关节控制，只需在技能空间中进行导航。

## 参考来源
- [sources/papers/ase.md](../../sources/papers/ase.md)
