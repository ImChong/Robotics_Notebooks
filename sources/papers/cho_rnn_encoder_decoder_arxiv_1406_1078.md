# Learning Phrase Representations using RNN Encoder–Decoder（arXiv:1406.1078）

> 论文来源归档（ingest）

- **标题：** Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation
- **作者：** Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio
- **类型：** paper / sequence-modeling / machine-translation / gru-origin
- **arXiv：** <https://arxiv.org/abs/1406.1078> · PDF：<https://arxiv.org/pdf/1406.1078.pdf>
- **会议：** EMNLP 2014
- **入库日期：** 2026-09-21
- **一句话说明：** 提出 **RNN Encoder–Decoder** 与 **GRU（Gated Recurrent Unit）** 门控循环单元，用固定长度向量编码变长短语并在统计机器翻译 log-linear 系统中作为额外特征显著提升 BLEU。

## 核心摘录（面向 wiki 编译）

### 1) RNN Encoder–Decoder 框架

- **要点：** 编码器 RNN 将源序列压成固定维向量，解码器 RNN 再生成目标序列；联合训练最大化 $P(\mathbf{y}\mid\mathbf{x})$。
- **对 wiki 的映射：** [`wiki/concepts/gru.md`](../../wiki/concepts/gru.md)

### 2) GRU 门控机制（方法原创点）

- **要点：** 相对 LSTM 用 **更少门（reset $r_t$ + update $z_t$）** 控制信息流入/保留，缓解 vanilla RNN 梯度消失；更新式可写为
  $$h_t = (1-z_t)\odot \tilde h_t + z_t \odot h_{t-1}$$
  其中候选状态 $\tilde h_t$ 受 reset 门调制上一隐状态。
- **对 wiki 的映射：** [`wiki/concepts/gru.md`](../../wiki/concepts/gru.md)

### 3) 机器翻译实证

- **要点：** 在 WMT'14 En→Fr 等设置上，Encoder–Decoder 条件短语概率作为 log-linear 模型额外特征，相对纯 n-gram 基线 **BLEU 提升**；定性分析显示学到语义/句法短语表示。
- **对 wiki 的映射：** [`wiki/concepts/gru.md`](../../wiki/concepts/gru.md)

## 当前提炼状态

- [x] GRU 原创出处与 Encoder–Decoder 框架摘录
- [x] 映射到 `wiki/concepts/gru.md`
