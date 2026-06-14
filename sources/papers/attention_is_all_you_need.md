# Attention Is All You Need（arXiv:1706.03762）

> 论文来源归档（ingest）

- **标题：** Attention Is All You Need
- **作者：** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin（Google Brain / Google Research）
- **类型：** paper / deep-learning / sequence-modeling / architecture
- **arXiv：** <https://arxiv.org/abs/1706.03762> · PDF：<https://arxiv.org/pdf/1706.03762.pdf>
- **会议：** NeurIPS 2017
- **入库日期：** 2026-06-14
- **一句话说明：** 提出 **Transformer** 架构，完全用 **自注意力（self-attention）** 替代循环与卷积，凭 **可并行 + 长程依赖** 成为现代序列建模与多模态/机器人基础策略（VLA、ACT、扩散策略骨干）的通用底座。

## 核心摘录（面向 wiki 编译）

### 1) 缩放点积注意力（Scaled Dot-Product Attention）

- **要点：** $\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$；用 $\sqrt{d_k}$ 缩放避免点积过大导致 softmax 饱和、梯度消失。
- **对 wiki 的映射：** [`wiki/concepts/transformer.md`](../../wiki/concepts/transformer.md)

### 2) 多头注意力（Multi-Head Attention）

- **要点：** 将 Q/K/V 投影到 $h$ 个子空间并行做注意力再拼接，使模型在不同表示子空间联合关注不同位置信息。
- **对 wiki 的映射：** [`wiki/concepts/transformer.md`](../../wiki/concepts/transformer.md)

### 3) 位置编码与无循环结构

- **要点：** 自注意力本身对序列顺序不敏感，需注入 **位置编码（positional encoding）**；去掉循环后整序列可并行计算，训练吞吐远超 RNN，且任意两 token 间路径长度为 $O(1)$，利于长程依赖。
- **对 wiki 的映射：** [`wiki/concepts/transformer.md`](../../wiki/concepts/transformer.md)

### 4) Encoder–Decoder 堆叠与残差/LayerNorm

- **要点：** 每层由多头注意力 + 前馈网络组成，配 **残差连接 + LayerNorm**；这一 block 结构后被 BERT/GPT/ViT 及机器人策略网络（如 humanoid-policy-network-architecture、bc-with-transformer）广泛复用。
- **对 wiki 的映射：** [`wiki/concepts/transformer.md`](../../wiki/concepts/transformer.md)

### 5) 对机器人/具身学习的迁移

- **要点：** Transformer 作为序列建模骨干支撑 **action chunking（ACT）**、**VLA**、**Robotics Transformer（RT 系列）** 等，把"观测/语言/历史动作序列 → 动作序列"统一为注意力建模。
- **对 wiki 的映射：** [`wiki/concepts/transformer.md`](../../wiki/concepts/transformer.md)

## 相关资料索引

| 资料 | 关系 |
|------|------|
| [Understanding Deep Learning (Prince, 2023)](../books/udl_book.md) | 教材中 Transformer 章节的统一数学视角 |
| [BERT](https://arxiv.org/abs/1810.04805) | encoder-only 预训练，自注意力下游迁移 |
| [GPT / language models](https://arxiv.org/abs/2005.14165) | decoder-only 自回归生成的代表 |
| [ViT](https://arxiv.org/abs/2010.11929) | 将 Transformer 引入视觉，影响机器人感知骨干 |

## 当前提炼状态

- [x] 注意力机制与架构要点摘录及 wiki 映射
- [x] 机器人/具身策略迁移交叉引用
