---
type: concept
summary: "Transformer 用自注意力替代循环与卷积，凭可并行与长程依赖成为现代序列建模与机器人基础策略（VLA、ACT、扩散策略骨干）的通用架构底座。"
description: Transformer 架构的核心机制（缩放点积注意力、多头注意力、位置编码）及其在机器人具身学习中的角色。
updated: 2026-07-11
related:
  - ./deep-learning-foundations.md
  - ./humanoid-policy-network-architecture.md
  - ../methods/bc-with-transformer.md
  - ../methods/robotics-transformer-rt-series.md
  - ../methods/action-chunking.md
  - ../entities/llms-from-scratch-raschka.md
sources:
  - ../../sources/papers/attention_is_all_you_need.md
  - ../../sources/books/udl_book.md
  - ../../sources/repos/rasbt_llms_from_scratch.md
---

# Transformer

> **一句话定义**：完全基于 **自注意力（self-attention）** 的序列建模架构，去掉循环与卷积，使整序列可并行计算且任意两 token 间路径长度为 $O(1)$，天然擅长长程依赖。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MHA | Multi-Head Attention | 多个注意力头在不同子空间并行建模 |
| QKV | Query / Key / Value | 注意力的三组投影向量 |
| PE | Positional Encoding | 注入序列顺序信息 |
| VLA | Vision-Language-Action | 以 Transformer 为骨干的多模态机器人策略 |
| ACT | Action Chunking Transformer | 预测动作块的序列模型架构 |

## 为什么重要

- Transformer 是现代 **基础模型** 的通用骨干：从 NLP（BERT/GPT）到视觉（ViT），再到机器人 [VLA](../methods/robotics-transformer-rt-series.md) 与 [动作分块（ACT）](../methods/action-chunking.md)，几乎统一了"序列输入 → 序列输出"的建模范式。
- 相比 RNN，**整序列可并行**，训练吞吐高；相比 CNN，**任意位置直接交互**，更易捕捉长程依赖——这两点正契合机器人需要融合长历史观测/语言指令/多关节动作序列的需求。

## 核心架构与机制

### 1. 缩放点积注意力（Scaled Dot-Product Attention）

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

- $Q, K, V$ 分别是 query、key、value 投影；$\sqrt{d_k}$ 缩放防止点积过大导致 softmax 饱和、梯度消失。
- 注意力权重衡量"每个位置应从其他位置取多少信息"。

### 2. 多头注意力（Multi-Head Attention）

将 $Q/K/V$ 投影到 $h$ 个子空间各自做注意力，再拼接回投影。多头让模型在不同表示子空间**联合关注不同位置/不同语义**。

### 3. 位置编码与残差结构

- 自注意力对顺序不敏感，需注入 **位置编码（PE）** 提供序列位置信息。
- 每层 = 多头注意力 + 前馈网络，外加 **残差连接 + LayerNorm**（与 [深度学习基础](./deep-learning-foundations.md) 中的残差思想一致），使深层堆叠可训练。

## 与机器人技术的联系

- **策略网络骨干**：[人形策略网络架构](./humanoid-policy-network-architecture.md) 常以 Transformer 编码本体感受/视觉/历史动作序列。
- **模仿学习**：BC with Transformer 与 [动作分块（ACT）](../methods/action-chunking.md) 用注意力把演示序列映射为动作块。
- **VLA / 通用策略**：[Robotics Transformer（RT 系列）](../methods/robotics-transformer-rt-series.md) 将视觉-语言-动作统一为注意力建模。

## 关联页面
- [反向传播算法](./backpropagation.md)
- [深度学习基础](./deep-learning-foundations.md)
- [人形策略网络架构](./humanoid-policy-network-architecture.md)
- [BC with Transformer](../methods/bc-with-transformer.md)
- [Robotics Transformer（RT 系列）](../methods/robotics-transformer-rt-series.md)
- [动作分块（Action Chunking）](../methods/action-chunking.md)
- [LLMs-from-scratch（Raschka）](../entities/llms-from-scratch-raschka.md)

## 参考来源
- [Attention Is All You Need 来源归档（arXiv:1706.03762）](../../sources/papers/attention_is_all_you_need.md)
- [Understanding Deep Learning (Prince, 2023)](../../sources/books/udl_book.md)
- [rasbt/LLMs-from-scratch 仓库归档](../../sources/repos/rasbt_llms_from_scratch.md)
- Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS. <https://arxiv.org/abs/1706.03762>

## 推荐继续阅读
- [LLMs-from-scratch（Raschka 实体页）](../entities/llms-from-scratch-raschka.md) — 纯 PyTorch 实现 MHA/GPT 的系统教程
