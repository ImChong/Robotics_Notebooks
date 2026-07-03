---
type: formalization
tags: [vla, machine-learning, robotics, math, tokenization]
status: complete
updated: 2026-07-03
related:
  - ../methods/vla.md
  - ../methods/behavior-cloning.md
  - ../formalizations/behavior-cloning-loss.md
  - ../entities/paper-mint-vla.md
sources:
  - ../../sources/papers/perception.md
  - ../../sources/papers/mint_rss_2026.md
summary: "动作分词（Action Tokenization）是将机器人的高维连续动作空间映射为有限离散 Token 序列的过程，是使大语言模型架构能够直接预测物理动作的关键桥梁。"
---

# Action Tokenization (动作分词)

在具身智能大模型（VLA）中，**动作分词 (Action Tokenization)** 是连接符号推理（语言模型）与物理执行（机器人控制）的数学枢纽。它解决了 LLM 架构本质上是离散序列预测器，而机器人动作本质上是连续向量这一根本矛盾。

## 数学定义

假设机器人的原始动作空间为 $\mathcal{A} \subset \mathbb{R}^d$（例如 $d=7$ 的关节力矩或末端位姿增量）。动作分词器由一对函数组成：

1. **编码器 (Encoder) $E$**：将连续动作 $a \in \mathcal{A}$ 映射为离散索引 $k \in \{1, \dots, K\}$。
   $$ k = E(a) $$
2. **解码器 (Decoder) $D$**：将离散索引 $k$ 还原为连续动作近似值 $\hat{a}$。
   $$ \hat{a} = D(k) $$

### 1. 标量分箱 (Scalar Binning)
最简单的形式是对动作的每个维度进行独立量化。
- 将 $[-1, 1]$ 范围均匀切分为 $256$ 个 bin。
- 每个维度对应一个 Token，一个 $d$ 维动作变为由 $d$ 个符号组成的子序列。
- **代表作**：RT-1, RT-2。

### 2. 向量量化 (Vector Quantization, VQ)
利用 VQ-VAE 的思想，在隐空间进行聚类。
- 维护一个可学习的 Codebook $\mathcal{C} = \{e_1, \dots, e_K\}$。
- 编码：寻找欧氏距离最近的码向量索引：$k = \arg\min_i \| z(a) - e_i \|_2$。
- **优点**：能捕捉动作维度间的相关性，Token 序列更短。
- **代表作**：Octo, π₀ (部分组件)。

### 3. 频域多尺度解耦（Spectral / Intent–Execution）
在动作块上施加 **DCT** 等频域变换，用 **逐尺度频域重建损失** 约束：最粗尺度 token 必须解释 **低频全局结构（意图）**，更细尺度专攻 **高频残差（执行）**。与纯时域 VQ 或均匀多尺度残差不同，监督信号直接对齐 **意图–执行** 语义。
- **代表作**：[MINT](../entities/paper-mint-vla.md) 的 **SDAT**（RSS 2026）。

## 损失函数与量化误差

分词质量通常由**重建损失 (Reconstruction Loss)** 衡量：
$$ \mathcal{L}_{recon} = \| a - D(E(a)) \|^2 $$

在 VLA 训练中，为了保证精度，通常需要在 Token 数量（计算开销）与量化分辨率（控制精度）之间进行折中。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作多模态基础策略方向 |
| LLM | Large Language Model | 大语言模型，常作高层任务/语言接口 |
| RT-2 | Robotics Transformer 2 | 将 web 规模 VLM 能力迁移到机器人控制的代表工作 |
| VAE | Variational Autoencoder | 变分自编码器，学习隐变量生成表示 |

## 为什么重要

1. **统一架构**：将动作视为一种特殊的“方言”，使 LLM 可以像预测下一个单词一样预测下一步动作。
2. **多模态对齐**：在共享的 Embedding 空间中，视觉 Token、文本 Token 和动作 Token 可以进行深度跨模态注意力计算。
3. **数据效率**：离散化后的动作分布通常比连续回归更容易被模型建模，尤其是在处理多模态分布（Multimodal Distributions）时。

## 关联页面
- [VLA (Vision-Language-Action Models)](../methods/vla.md)
- [MINT（SDAT 频域意图分词）](../entities/paper-mint-vla.md)
- [Behavior Cloning Loss](./behavior-cloning-loss.md)
- [Cross-modal Attention](./cross-modal-attention.md)

## 参考来源
- Brohan, A., et al. (2022). *RT-1: Robotics Transformer*.
- Padalkar, A., et al. (2023). *Open X-Embodiment: Robotic Learning at Scale*.
- [MINT 论文摘录（arXiv:2602.08602）](../../sources/papers/mint_rss_2026.md)
