# Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling（arXiv:1412.3555）

> 论文来源归档（ingest）

- **标题：** Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling
- **作者：** Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, Yoshua Bengio
- **类型：** paper / sequence-modeling / empirical-comparison
- **arXiv：** <https://arxiv.org/abs/1412.3555> · PDF：<https://arxiv.org/pdf/1412.3555.pdf>
- **入库日期：** 2026-09-21
- **一句话说明：** 在 **polyphonic music modeling** 与 **speech signal modeling** 上系统对比 **tanh RNN / LSTM / GRU**，结论：**门控单元显著优于 tanh RNN，GRU 与 LSTM 性能相当**。

## 核心摘录（面向 wiki 编译）

### 1) 实验任务

- **Music modeling：** JSB Chorales 等 polyphonic 序列，评价负对数似然。
- **Speech modeling：** TIMIT 等语音帧序列建模。
- **对 wiki 的映射：** [`wiki/concepts/gru.md`](../../wiki/concepts/gru.md) §GRU vs LSTM

### 2) 主要结论

- **门控 > 传统 RNN：** LSTM 与 GRU 均大幅优于 tanh 单元。
- **GRU ≈ LSTM：** 在两项任务上 **GRU 与 LSTM 可比**；GRU 参数更少、结构更简单，常作默认 gated RNN 选型。
- **对 wiki 的映射：** [`wiki/concepts/gru.md`](../../wiki/concepts/gru.md)、[`wiki/comparisons/rnn-cnn-transformer-mamba.md`](../../wiki/comparisons/rnn-cnn-transformer-mamba.md)

### 3) 与 Cho et al. (2014) 关系

- **GRU 定义** 出自 Cho et al. EMNLP 2014；本论文是 **GRU 与 LSTM 的首批系统实证对照** 之一。
- **对 wiki 的映射：** [`cho_rnn_encoder_decoder_arxiv_1406_1078.md`](cho_rnn_encoder_decoder_arxiv_1406_1078.md)

## 当前提炼状态

- [x] 双任务对比结论摘录
- [x] 映射到 GRU 概念页
