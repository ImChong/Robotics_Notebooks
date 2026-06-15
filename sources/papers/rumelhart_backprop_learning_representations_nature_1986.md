# Learning representations by back-propagating errors（Nature, 1986）

> 论文来源归档（ingest）

- **标题：** Learning representations by back-propagating errors
- **作者：** David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams
- **类型：** paper / deep-learning / optimization / representation-learning
- **期刊：** Nature 323, 533–536 (1986)
- **DOI：** <https://doi.org/10.1038/323533a0>
- **扩展版：** Rumelhart, Hinton & Williams (1986), *Parallel Distributed Processing* Vol. 1, Ch. 8 (MIT Press)
- **入库日期：** 2026-06-15
- **一句话说明：** 提出 **反向传播（back-propagation）** 训练程序：在多层 **neurone-like** 网络中反复按误差信号调整连接权重，使 **隐藏层** 自动学到任务域的重要特征，从而突破单层感知机的线性可分局限。

## 核心摘录（面向 wiki 编译）

### 1) 多层网络与隐藏单元表征

- **要点：** 网络含输入、输出与 **内部隐藏单元（hidden units）**；训练目标是最小化实际输出与期望输出之差。权重更新后，隐藏单元会编码任务域中的 **重要特征**，任务规律由这些单元间的交互捕获——这是 back-propagation 相对早期 **perceptron-convergence** 等简单方法的关键能力：**能创造有用的新特征（useful new features）**。
- **对 wiki 的映射：** [`wiki/concepts/backpropagation.md`](../../wiki/concepts/backpropagation.md)

### 2) 误差反向传播与权重调整

- **要点：** 学习程序 **反复（repeatedly）** 沿连接反向传播误差信号，调整各层权重直至输出逼近目标。与只能训练单层、无法有效利用隐藏层的感知机收敛规则不同，该方法使 **深度堆叠的非线性单元** 可通过梯度信息端到端训练。
- **对 wiki 的映射：** [`wiki/concepts/backpropagation.md`](../../wiki/concepts/backpropagation.md)、[`wiki/concepts/deep-learning-foundations.md`](../../wiki/concepts/deep-learning-foundations.md)

### 3) 历史脉络与先行工作

- **要点：** 摘要明确将本方法与 **Rosenblatt 感知机**、**Minsky & Papert** 对单层局限的批评对照；并引用 **Le Cun (1985)** 在 Cognitiva 的独立推导。更完整的数学与实验见 PDP Vol. 1 同名章节。
- **对 wiki 的映射：** [`wiki/concepts/backpropagation.md`](../../wiki/concepts/backpropagation.md)

### 4) 对现代深度学习的地位

- **要点：** 该文使 **链式法则 + 分层计算图** 成为训练深度网络的实用标准，直接支撑今日 CNN、策略网络、VLA 等 **端到端可微训练**；现代框架中的 autograd（如 PyTorch）是其工程化推广，而非另一套独立算法。
- **对 wiki 的映射：** [`wiki/concepts/backpropagation.md`](../../wiki/concepts/backpropagation.md)、[`wiki/concepts/deep-learning-foundations.md`](../../wiki/concepts/deep-learning-foundations.md)

## 相关资料索引

| 资料 | 关系 |
|------|------|
| [Understanding Deep Learning (Prince, 2023)](../books/udl_book.md) | 第 7 章 *Gradients and Initialization* 从现代视角统一讲解反向传播与初始化 |
| [Deep Learning Book Ch. 6](https://www.deeplearningbook.org/contents/mlp.html) | Goodfellow 等对反向传播与计算图的教材化表述 |
| Le Cun (1985), Cognitiva | 与 Rumelhart 等并行的反向传播推导 |
| Werbos (1974) PhD thesis | 更早提出将反向传播用于神经网络训练 |

## 当前提炼状态

- [x] 一手摘要与 wiki 映射
- [ ] 待补：PDP Vol. 1 中 XOR、泛化实验的定量摘录（若需扩展「表征学习」小节）
