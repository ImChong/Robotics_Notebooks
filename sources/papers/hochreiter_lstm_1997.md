# LSTM：长短期记忆（Neural Computation, 1997）

> 论文来源归档（ingest）

- **标题：** Long Short-Term Memory
- **作者：** Sepp Hochreiter, Jürgen Schmidhuber
- **类型：** paper / sequence-modeling / rnn / architecture
- **期刊：** Neural Computation 9(8):1735–1780 (1997)
- **DOI：** <https://doi.org/10.1162/neco.1997.9.8.1735>
- **入库日期：** 2026-09-21
- **一句话说明：** 用 **恒等误差传送带（CEC）** 与输入/输出/遗忘门，缓解 RNN 训练中的梯度消失，使网络能学习跨越很长时滞的依赖。

## 核心摘录（面向 wiki 编译）

### 1) 为什么朴素 RNN 记不住长程

- **要点：** 误差在时间上反复乘以雅可比，特征值 \(<1\) 则指数衰减、\(>1\) 则爆炸。长时滞监督信号到不了早期时间步。
- **对 wiki 的映射：** [`wiki/concepts/recurrent-neural-network.md`](../../wiki/concepts/recurrent-neural-network.md)

### 2) 恒等误差传送带 + 门控

- **要点：** 记忆单元状态沿时间近似 **乘 1** 传递；输入门保护写入、输出门保护读出，后续实践普遍加上遗忘门。这把「记什么 / 忘什么」变成可学习开关，而不是靠深层非线性硬记。
- **对 wiki 的映射：** [`wiki/concepts/recurrent-neural-network.md`](../../wiki/concepts/recurrent-neural-network.md)、[`wiki/overview/ai-architecture-map.md`](../../wiki/overview/ai-architecture-map.md)

### 3) 对机器人时序策略的读法

- **要点：** 腿式历史观测、接触事件、部分可观测控制仍常用 LSTM/GRU **短历史编码器**；长上下文与多模态则交给 Transformer/Mamba。不要把 1997 年的语言模型实验直接当成机载延迟结论。
- **对 wiki 的映射：** [`wiki/concepts/humanoid-policy-network-architecture.md`](../../wiki/concepts/humanoid-policy-network-architecture.md)

## 开源状态（步骤 2.5）

- 经典期刊论文，**无单一现代官方仓**；LSTM 单元已是框架内置算子。

## 当前提炼状态

- [x] 要点摘录与 wiki 映射
