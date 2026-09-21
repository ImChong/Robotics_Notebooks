# TCN：通用卷积序列模型的实证评估（arXiv:1803.01271）

> 论文来源归档（ingest）

- **标题：** An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling
- **作者：** Shaojie Bai, J. Zico Kolter, Vladlen Koltun
- **类型：** paper / sequence-modeling / convolution / tcn
- **arXiv：** <https://arxiv.org/abs/1803.01271> · PDF：<https://arxiv.org/pdf/1803.01271.pdf>
- **官方代码：** <https://github.com/locuslab/TCN>
- **入库日期：** 2026-09-21
- **一句话说明：** 把 **因果卷积 + 膨胀 + 残差** 组成 Temporal Convolutional Network，在多项序列基准上系统对比 LSTM/GRU，发现简单 TCN 常常 **更准、可并行、梯度更稳**。

## 核心摘录（面向 wiki 编译）

### 1) 因果 + 膨胀 = 可并行的长感受野

- **要点：** 因果卷积保证 \(t\) 只看 \(\le t\)；膨胀卷积指数扩大感受野；残差堆叠稳定深度。训练可像 CNN 一样并行，不再逐步展开 RNN。
- **对 wiki 的映射：** [`wiki/concepts/temporal-convolutional-network.md`](../../wiki/concepts/temporal-convolutional-network.md)

### 2) 对「序列默认用 RNN」的反证

- **要点：** 在作者选定的多项基准上，通用 TCN 优于通用 LSTM/GRU。结论不是「卷积永远赢」，而是 **先问清感受野、并行度与状态需求，再选函数族**。
- **对 wiki 的映射：** [`wiki/comparisons/rnn-cnn-transformer-mamba.md`](../../wiki/comparisons/rnn-cnn-transformer-mamba.md)、[`wiki/overview/ai-architecture-map.md`](../../wiki/overview/ai-architecture-map.md)

### 3) 机器人时序的适用边界

- **要点：** TCN 适合固定感受野的历史堆叠（IMU 窗口、触觉短历史）。需要不定长记忆或跨模态 token 对齐时，仍让位 Transformer/Mamba。
- **对 wiki 的映射：** [`wiki/concepts/humanoid-policy-network-architecture.md`](../../wiki/concepts/humanoid-policy-network-architecture.md)

## 开源状态（步骤 2.5）

- `locuslab/TCN` **已开源**（PyTorch 参考实现）。

## 当前提炼状态

- [x] 要点摘录与 wiki 映射
