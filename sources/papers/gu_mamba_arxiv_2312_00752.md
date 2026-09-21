# Mamba：选择性状态空间的线性时间序列模型（arXiv:2312.00752）

> 论文来源归档（ingest）

- **标题：** Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- **作者：** Albert Gu, Tri Dao
- **类型：** paper / sequence-modeling / ssm / mamba
- **arXiv：** <https://arxiv.org/abs/2312.00752> · PDF：<https://arxiv.org/pdf/2312.00752.pdf>
- **官方代码：** <https://github.com/state-spaces/mamba>
- **入库日期：** 2026-09-21
- **一句话说明：** 让 SSM 的离散参数 **依赖当前输入**（选择性），再用硬件感知的并行扫描实现训练，推理保持逐步 \(O(1)\) 状态更新。

## 核心摘录（面向 wiki 编译）

### 1) 选择性：该记的记、该忘的忘

- **要点：** 线性时不变 S4 对所有 token 用同一 \(A,B\)；Mamba 让 \(\bar B, C\) 等随 \(x_t\) 变化，从而在语言与基因组等数据上过滤无关 token。没有选择性，SSM 很难做离散内容寻址。
- **对 wiki 的映射：** [`wiki/concepts/mamba.md`](../../wiki/concepts/mamba.md)、[`wiki/concepts/state-space-model-ssm.md`](../../wiki/concepts/state-space-model-ssm.md)

### 2) 扫描核：并行训练、常状态推理

- **要点：** 选择性破坏了纯卷积视图，必须用 **并行前缀扫描** 才能在 GPU 上训练。推理仍是逐步更新隐状态，复杂度近线性、无需 KV cache。
- **对 wiki 的映射：** [`wiki/concepts/mamba.md`](../../wiki/concepts/mamba.md)

### 3) 不是免费的 Transformer 替代

- **要点：** 语言与长序列上 Mamba 有竞争力；视觉需设计扫描顺序（Vim/VMamba）。机器人部署还受 CUDA/Triton 核与生态成熟度约束，混合块（注意力+SSM+卷积）更常见。
- **对 wiki 的映射：** [`wiki/comparisons/rnn-cnn-transformer-mamba.md`](../../wiki/comparisons/rnn-cnn-transformer-mamba.md)、[`wiki/overview/ai-architecture-map.md`](../../wiki/overview/ai-architecture-map.md)

## 开源状态（步骤 2.5）

- `state-spaces/mamba` **已开源**，含官方 CUDA 扫描实现。无单独机构项目页；以 GitHub README 为准。

## 当前提炼状态

- [x] 要点摘录与 wiki 映射
