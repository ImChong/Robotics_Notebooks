---
type: formalization
tags: [foundation-policy, machine-learning, math, alignment, multi-task]
status: complete
updated: 2026-04-21
related:
  - ../methods/vla.md
  - ../concepts/embodied-scaling-laws.md
  - ./vla-tokenization.md
sources:
  - ../../sources/papers/rl_foundation_models.md
summary: "基础策略对齐（Foundation Policy Alignment）形式化描述了如何通过对比学习或交叉熵优化，将多源、异构的机器人演示数据映射到统一的语义-动作隐空间，实现跨形态的知识共享。"
---

# Foundation Policy Alignment (基础策略对齐)

在具身基础模型（Foundation Policy）中，**对齐 (Alignment)** 是指将来自不同机器人形态（如四足、双足、机械臂）、不同传感器配置和不同任务目标的异构数据，映射到一个统一的、可迁移的数学表示空间的过程。

## 形式化目标

假设我们有 $N$ 种不同的机器人形态 $\{M_1, M_2, \dots, M_N\}$。每种形态都有其特有的观测空间 $\mathcal{O}_i$ 和动作空间 $\mathcal{A}_i$。

对齐的目标是学习一组映射函数 $E_{obs}, E_{lang}, P_{act}$，使得：

1. **共享表示空间**：观测 $o \in \mathcal{O}_i$ 和指令 $l$ 被映射到同一个隐向量 $z \in \mathcal{Z}$。
   $$ z = \text{Attention}(E_{obs}(o), E_{lang}(l)) $$
2. **跨形态迁移**：隐向量 $z$ 包含了任务的“物理真理”，可以被解码为任何形态的动作。
   $$ a_i = P_{act}(z, \text{ID}_i) $$

## 核心损失函数

### 1. 跨模态对比学习 (Contrastive Alignment)
通过 InfoNCE 损失，拉近“当前观测 + 指令”与“专家采取的动作”在隐空间中的距离：
$$ \mathcal{L}_{align} = -\log \frac{\exp(\text{sim}(z, a^+))}{\sum \exp(\text{sim}(z, a^-))} $$

### 2. 统一交叉熵 (Unified Cross-Entropy)
如果采用了 [Action Tokenization](./vla-tokenization.md)，对齐问题转为对目标动作 Token 的极大似然估计：
$$ \mathcal{L}_{MLE} = -\sum_{(o,l,a) \in \mathcal{D}} \log P(a | o, l) $$

## 为什么这一形式化很重要

- **知识涌现**：通过对齐，模型在形态 A 上学到的“识别红色物体”的语义知识，可以自动迁移到形态 B 的操作中。
- **解决异构性**：它允许我们同时使用真实数据（感知真实，但数量少）和仿真数据（感知有偏，但动作精准）进行互补训练。

## 关键挑战

- **动作幅值对齐**：不同机器人电机的扭矩范围完全不同，通常需要归一化到 $[-1, 1]$。
- **参考系对齐**：必须统一坐标系（如以机器人本体为中心的相对坐标），否则空间语义会发生混乱。

## 关联页面
- [VLA (Vision-Language-Action Models)](../methods/vla.md)
- [具身规模法则 (Scaling Laws)](../concepts/embodied-scaling-laws.md)
- [动作分词 (Tokenization)](./vla-tokenization.md)

## 参考来源
- Reed, S., et al. (2022). *A Generalist Agent (Gato)*.
- [Google DeepMind Blog on Foundation Policies](https://deepmind.google/research/publications/rt-2/).
