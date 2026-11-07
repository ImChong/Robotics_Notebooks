# Attention Is All You Need（arXiv:1706.03762）

> 论文来源归档（ingest）

- **标题：** Attention Is All You Need
- **作者：** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin（Google Brain / Google Research）
- **类型：** paper / deep-learning / sequence-modeling / architecture
- **arXiv：** <https://arxiv.org/abs/1706.03762> · PDF：<https://arxiv.org/pdf/1706.03762.pdf>
- **会议：** NeurIPS 2017
- **入库日期：** 2026-11-07（首次入库 2026-06-14）
- **版本：** arXiv:1706.03762v7（2023-08-02 修订版）；NeurIPS 2017
- **核对范围：** 2026-11-07 下载 arXiv v7 PDF（15 页）并通读全文：摘要、§1 引言、§2 背景、§3 模型结构（编码器/解码器堆叠、缩放点积注意力、多头注意力、三种注意力用法、FFN、嵌入与 softmax、位置编码）、§4 为何自注意力（Table 1）、§5 训练（数据、硬件、优化器、正则）、§6 结果（Table 2 翻译、Table 3 消融、Table 4 句法分析）、§7 结论、附录注意力可视化图 3–5；参考文献仅浏览。未核对 tensor2tensor 源码（见 repos 归档）。
- **通读：** 已完成
- **深读更新：** 2026-09-21
- **一句话说明：** 提出 **Transformer** 架构，完全用 **自注意力（self-attention）** 替代循环与卷积，凭 **可并行 + 长程依赖** 成为现代序列建模与多模态/机器人基础策略（VLA、ACT、扩散策略骨干）的通用底座。

## 核心摘录（面向 wiki 编译）

### 1) 缩放点积注意力（Scaled Dot-Product Attention）

- **要点：** $\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$；用 $\sqrt{d_k}$ 缩放避免点积过大导致 softmax 饱和、梯度消失。
- **对 wiki 的映射：** [`wiki/entities/paper-attention-is-all-you-need.md`](../../wiki/entities/paper-attention-is-all-you-need.md)、[`wiki/concepts/transformer.md`](../../wiki/concepts/transformer.md)

### 2) 多头注意力（Multi-Head Attention）

- **要点：** 将 Q/K/V 投影到 $h$ 个子空间并行做注意力再拼接，使模型在不同表示子空间联合关注不同位置信息。
- **对 wiki 的映射：** [`wiki/entities/paper-attention-is-all-you-need.md`](../../wiki/entities/paper-attention-is-all-you-need.md)、[`wiki/concepts/transformer.md`](../../wiki/concepts/transformer.md)

### 3) 位置编码与无循环结构

- **要点：** 自注意力本身对序列顺序不敏感，需注入 **位置编码（positional encoding）**；去掉循环后整序列可并行计算，训练吞吐远超 RNN，且任意两 token 间路径长度为 $O(1)$，利于长程依赖。
- **对 wiki 的映射：** [`wiki/entities/paper-attention-is-all-you-need.md`](../../wiki/entities/paper-attention-is-all-you-need.md)、[`wiki/concepts/transformer.md`](../../wiki/concepts/transformer.md)

### 4) Encoder–Decoder 堆叠与残差/LayerNorm

- **要点：** 每层由多头注意力 + 前馈网络组成，配 **残差连接 + LayerNorm**；这一 block 结构后被 BERT/GPT/ViT 及机器人策略网络（如 humanoid-policy-network-architecture、bc-with-transformer）广泛复用。
- **对 wiki 的映射：** [`wiki/entities/paper-attention-is-all-you-need.md`](../../wiki/entities/paper-attention-is-all-you-need.md)、[`wiki/concepts/transformer.md`](../../wiki/concepts/transformer.md)

### 5) 对机器人/具身学习的迁移

- **要点：** Transformer 作为序列建模骨干支撑 **action chunking（ACT）**、**VLA**、**Robotics Transformer（RT 系列）** 等，把"观测/语言/历史动作序列 → 动作序列"统一为注意力建模。
- **对 wiki 的映射：** [`wiki/entities/paper-attention-is-all-you-need.md`](../../wiki/entities/paper-attention-is-all-you-need.md)、[`wiki/concepts/transformer.md`](../../wiki/concepts/transformer.md)

## 相关资料索引

| 资料 | 关系 |
|------|------|
| [Understanding Deep Learning (Prince, 2023)](../books/udl_book.md) | 教材中 Transformer 章节的统一数学视角 |
| [BERT](https://arxiv.org/abs/1810.04805) | encoder-only 预训练，自注意力下游迁移 |
| [GPT / language models](https://arxiv.org/abs/2005.14165) | decoder-only 自回归生成的代表 |
| [ViT](https://arxiv.org/abs/2010.11929) | 将 Transformer 引入视觉，影响机器人感知骨干 |

### 6) 实验规模与 SOTA 数值（深读）

- **要点：** WMT14 EN→DE **28.4 BLEU**（big，单模型）；EN→FR **41.8 BLEU**；8×GPU 训练约 3.5 天，成本远低于当时 ensemble SOTA。Constituency parsing 上亦超专用 parser。
- **对 wiki 的映射：** [`wiki/entities/paper-attention-is-all-you-need.md`](../../wiki/entities/paper-attention-is-all-you-need.md)

### 7) 官方代码与复现锚点

- **要点：** Google 官方 [tensor2tensor](https://github.com/tensorflow/tensor2tensor)；模块边界（MHA、FFN、PE、beam decode）为现代 HF/PyTorch 实现的语义对照基准。
- **对 wiki 的映射：** [`sources/repos/tensor2tensor.md`](../repos/tensor2tensor.md)、[`wiki/entities/paper-attention-is-all-you-need.md`](../../wiki/entities/paper-attention-is-all-you-need.md)

## 开源边界（步骤 2.5）

| 状态 | 说明 |
|------|------|
| **已开源** | 官方 tensor2tensor；Apache-2.0 |
| **维护** | 新工程多迁移至 JAX/HF；T2T 仍可作为论文级模块对照 |

## 当前提炼状态

- [x] 注意力机制与架构要点摘录及 wiki 映射
- [x] 机器人/具身策略迁移交叉引用
- [x] 深读实体页与官方代码归档（2026-09-21）

## 通读核对结论（2026-11-07，对照 arXiv v7 全文）

- **结构：** 编码器与解码器各 N=6 层；子层输出为 `LayerNorm(x + Sublayer(x))`（Post-LN）；$d_{model}=512$，$d_{ff}=2048$，$h=8$，$d_k=d_v=64$；解码器自注意力加因果掩码（softmax 输入置 −∞），并有对编码器输出的交叉注意力（§3.2.3）；嵌入与 pre-softmax 线性层共享权重，嵌入乘 $\sqrt{d_{model}}$。
- **注意力缩放动机（脚注 4）：** 设 q、k 各分量独立、均值 0、方差 1，则 $q\cdot k$ 方差为 $d_k$；原文表述为“我们猜测”大 $d_k$ 时点积幅度变大使 softmax 落入梯度极小区，并非严格证明。
- **位置编码：** 正弦/余弦，波长从 $2\pi$ 到 $10000\cdot2\pi$ 成几何级数；与可学习位置嵌入对比“结果几乎相同”（Table 3 行 E）；选正弦是因为可能外推到更长序列。**原文没有做“去掉位置编码”的消融。**
- **Table 1：** 自注意力每层 $O(n^2 d)$、顺序操作 $O(1)$、最大路径 $O(1)$；循环 $O(nd^2)$、$O(n)$、$O(n)$；卷积 $O(knd^2)$、$O(1)$、$O(\log_k n)$；受限自注意力 $O(rnd)$、$O(1)$、$O(n/r)$。自注意力在 $n<d$ 时比循环层更省。
- **训练：** 8×P100；base 每步约 0.4 s、训练 100K 步约 12 小时；big 每步 1.0 s、300K 步约 3.5 天。WMT14 EN-DE 约 450 万句对、BPE 共享词表约 37000；EN-FR 3600 万句、32000 词片；Adam β2=0.98，warmup 4000 步，label smoothing 0.1，Pdrop=0.1（EN-FR big 用 0.1，EN-DE big 用 0.3，见 Table 3）。
- **结果：** EN-DE big 28.4 BLEU，超过此前最好结果（含 ensemble）2.0 以上，训练 3.5 天；base 在 Table 2 测试集为 EN-DE 27.3、EN-FR 38.1（base 开发集 BLEU 25.8 见 Table 3）。EN-FR big：Table 2 为 41.8，正文 §6.1 却写 41.0（原文自身不一致，引用时以 Table 2 并注明）。Table 2 训练成本（FLOPs）EN-DE big 为 2.3×10^19，EN-FR 为 2.3×10^19 量级（以原表为准）。
- **消融（Table 3，EN-DE 开发集 newstest2013）：** 单头比最优设置低 0.9 BLEU，头数过多也下降（行 A）；减小 $d_k$ 伤害质量（行 B）；模型越大越好、dropout 有帮助（行 C/D）。
- **句法分析（Table 4）：** 4 层 Transformer，WSJ 23 F1：仅 WSJ 91.3，半监督 92.7；在 WSJ-only 设置优于 BerkeleyParser，仅略逊于 RNN Grammar 的生成式 93.3。
- **结论中的局限与展望：** 作者计划扩展到文本以外的模态、研究局部/受限注意力处理图像音频视频，并让生成“更少顺序化”。
