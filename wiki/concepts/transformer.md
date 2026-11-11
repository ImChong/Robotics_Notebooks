---
type: concept
status: complete
summary: "Transformer 用自注意力替代循环与卷积，凭可并行与长程依赖成为现代序列建模与机器人基础策略（VLA、ACT、扩散策略骨干）的通用架构底座。"
description: Transformer 架构的核心机制（缩放点积注意力、多头注意力、位置编码）及其在机器人具身学习中的角色。
updated: 2026-11-11
related:
  - ../entities/paper-attention-is-all-you-need.md
  - ../entities/transformer-cv-curriculum.md
  - ./deep-learning-foundations.md
  - ./humanoid-policy-network-architecture.md
  - ../overview/ai-architecture-map.md
  - ./mamba.md
  - ../methods/bc-with-transformer.md
  - ../methods/robotics-transformer-rt-series.md
  - ../methods/action-chunking.md
  - ../entities/llms-from-scratch-raschka.md
  - ../entities/zju-foundations-of-llms.md
  - ../entities/andrej-karpathy.md
  - ../entities/paper-effective-degree.md
original:
  - ../../sources/papers/attention_is_all_you_need.md
sources:
  - ../../sources/blogs/alexwang_associative_memory.md
  - ../../sources/repos/keawang_associative_memory_blog.md
  - ../../sources/papers/attention_is_all_you_need.md
  - ../../sources/books/udl_book.md
  - ../../sources/repos/rasbt_llms_from_scratch.md
  - ../../sources/courses/karpathy_zero_to_hero_youtube.md
---

# Transformer

## 一句话定义

完全基于 **自注意力（self-attention）** 的序列建模架构，去掉循环与卷积，使整序列可并行计算且任意两 token 间路径长度为 $O(1)$，天然擅长长程依赖（Vaswani et al., 2017）。

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

## 核心原理

以下对照原论文 [Attention Is All You Need](../entities/paper-attention-is-all-you-need.md)（arXiv:1706.03762v7）§3–§4 核实。

### 1. 缩放点积注意力（Scaled Dot-Product Attention）

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

- $Q, K, V$ 分别是 query、key、value 投影；原文用 $\sqrt{d_k}$ 缩放：若 q、k 分量独立、零均值单位方差，点积方差为 $d_k$，作者**猜测**大 $d_k$ 时点积过大会把 softmax 推入梯度极小区（脚注 4，非严格证明）。
- 注意力权重衡量"每个位置应从其他位置取多少信息"。

### 2. 多头注意力（Multi-Head Attention）

将 $Q/K/V$ 用学习到的线性映射投影 $h$ 次，各自做注意力，再 `Concat` 后乘 $W^O$。原文取 $h=8$、$d_k=d_v=d_{model}/h=64$，总计算量与单头全维度相近；多头让模型在不同表示子空间**联合关注不同位置**，单头的加权平均会抑制这一点。详见 [多头注意力](./multi-head-attention.md)。

### 3. 位置编码与残差结构

- 模型无循环无卷积，需把位置信息加到嵌入上。原文用正弦/余弦：$PE_{(pos,2i)}=\sin(pos/10000^{2i/d_{model}})$，奇数维取余弦；与可学习位置嵌入对比结果几乎相同（Table 3 行 E）。
- 每层 = 多头注意力 + 逐位置前馈网络 $\max(0,xW_1+b_1)W_2+b_2$（$d_{model}=512$，$d_{ff}=2048$），子层输出为 $\mathrm{LayerNorm}(x+\mathrm{Sublayer}(x))$（原版 Post-LN；与 [深度学习基础](./deep-learning-foundations.md) 中的残差思想一致）。编码器、解码器各 6 层；解码器在自注意力上加因果掩码，并多一个对编码器输出的交叉注意力。

### 4. 为什么用自注意力（原文 Table 1）

| 层类型 | 每层复杂度 | 顺序操作 | 最大路径长度 |
|--------|-----------|----------|--------------|
| 自注意力 | $O(n^2 d)$ | $O(1)$ | $O(1)$ |
| 循环 | $O(n d^2)$ | $O(n)$ | $O(n)$ |
| 卷积 | $O(k n d^2)$ | $O(1)$ | $O(\log_k n)$ |
| 受限自注意力（邻域 r） | $O(r n d)$ | $O(1)$ | $O(n/r)$ |

$n<d$ 时自注意力层比循环层更省；$n$ 很大时需受限自注意力（原文列为未来工作）。

## 联想记忆视角：参数记忆与上下文记忆

[Alex Wang 教程](../../sources/blogs/alexwang_associative_memory.md)把查询理解为“按键的相似度加权取值”：外积线性层累加键值，MLP 用非线性门控减少检索干扰，自注意力则从上下文构造键值。MLP 的键值在参数中，自注意力的键值随输入变化。

工程上可据此区分两类问题：修改参数中的知识，需要训练或编辑权重；补充当前任务信息，可以增加上下文，但仍受计算、显存和检索质量限制。该区分是理解框架，不是两种机制完全独立的保证。

**适用边界：** 教程的容量估计假设随机键值、用已存储键查询，并以干扰与信号等大为阈值；不能把结果当作训练后模型的事实记忆上限。文中的共享输入投影 SwiGLU 实验也不等于任意实际 SwiGLU 层。

**复现入口：** [公开实验仓库归档](../../sources/repos/keawang_associative_memory_blog.md)；README 提供 `uv run crosstalk.py`。本次核对源码入口，未运行数值实验。

## 工程实践

- **原文基线配置：** base 为 6 层、$d_{model}=512$、8 头、65M 参数，8×P100 训练约 12 小时（10 万步）；big 为 $d_{model}=1024$、16 头、213M 参数，约 3.5 天（30 万步）。优化器 Adam（$\beta_2=0.98$），学习率线性 warmup 4000 步后按步数平方根倒数衰减，label smoothing 0.1。
- **消融启示（Table 3）：** 头数过少（单头 −0.9 BLEU）或过多都会掉点；缩小 $d_k$ 会伤害质量；模型越大、dropout 越有帮助。
- **迁移到机器人（本站其他页汇总，非原文结论）：** 动作块、观测 token 通常很短，$O(n^2)$ 一般可接受；很多后续实现改用 Pre-LN 以稳定训练（推测，非本文内容）。

## 与机器人技术的联系

- **策略网络骨干**：[人形策略网络架构](./humanoid-policy-network-architecture.md) 常以 Transformer 编码本体感受/视觉/历史动作序列。
- **模仿学习**：BC with Transformer 与 [动作分块（ACT）](../methods/action-chunking.md) 用注意力把演示序列映射为动作块。
- **VLA / 通用策略**：[Robotics Transformer（RT 系列）](../methods/robotics-transformer-rt-series.md) 将视觉-语言-动作统一为注意力建模。

## 局限与风险

- **$O(n^2)$ 代价：** 自注意力每层 $O(n^2 d)$，长序列/高分辨率下显存与算力先崩；原文只提出受限邻域注意力作为方向，未实验。
- **原文验证范围窄：** 仅验证机器翻译与句法分析；作者自己把扩展到其他模态、图像音频视频列为未来工作。机器翻译 BLEU 优势不能当作动作任务上的先验优势（推测）。
- **位置外推未证实：** 选正弦编码是“可能”外推到更长序列，原文没有做长度外推实验。
- **联想记忆视角的适用边界**见上节，不等于训练后模型的事实记忆容量。

## 关联页面
- [AI 架构地图](../overview/ai-architecture-map.md)
- [Mamba](./mamba.md)
- [反向传播算法](./backpropagation.md)
- [深度学习基础](./deep-learning-foundations.md)
- [人形策略网络架构](./humanoid-policy-network-architecture.md)
- [BC with Transformer](../methods/bc-with-transformer.md)
- [Robotics Transformer（RT 系列）](../methods/robotics-transformer-rt-series.md)
- [动作分块（Action Chunking）](../methods/action-chunking.md)
- [《大模型基础》（浙江大学 DAILY Lab）](../entities/zju-foundations-of-llms.md) — 系统整理语言模型、架构、Prompt、PEFT、模型编辑和 RAG 六章教材
- [LLMs-from-scratch（Raschka）](../entities/llms-from-scratch-raschka.md)
- [Andrej Karpathy](../entities/andrej-karpathy.md)
- [Effective Degree](../entities/paper-effective-degree.md) — ViT / BERT 上用路径多项式正则提升泛化（ICML 2026）

## 参考来源
- [Alex Wang：联想记忆与 Transformer](../../sources/blogs/alexwang_associative_memory.md)
- [配套数值实验代码](../../sources/repos/keawang_associative_memory_blog.md)
- [Attention Is All You Need（论文实体）](../entities/paper-attention-is-all-you-need.md)
- [Attention Is All You Need 来源归档（arXiv:1706.03762）](../../sources/papers/attention_is_all_you_need.md)
- [Understanding Deep Learning (Prince, 2023)](../../sources/books/udl_book.md)
- [rasbt/LLMs-from-scratch 仓库归档](../../sources/repos/rasbt_llms_from_scratch.md)
- [Zero to Hero 播放列表（#7 GPT）](../../sources/courses/karpathy_zero_to_hero_youtube.md)
- Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS. <https://arxiv.org/abs/1706.03762>

## 推荐继续阅读
- [LLMs-from-scratch（Raschka 实体页）](../entities/llms-from-scratch-raschka.md) — 纯 PyTorch 实现 MHA/GPT 的系统教程
- [原论文 PDF（arXiv:1706.03762）](https://arxiv.org/pdf/1706.03762)
- [Karpathy Zero to Hero #7 Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) — 视频驱动从零实现
