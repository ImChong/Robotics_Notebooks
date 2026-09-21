---
type: concept
tags: [gru, rnn, lstm, sequence-modeling, deep-learning, pytorch, recurrent]
status: complete
updated: 2026-09-21
summary: "GRU（门控循环单元）用 reset/update 两门在 RNN 隐状态中选择性写入与保留信息，参数少于 LSTM；Cho et al. (2014) 提出，Chung et al. (2014) 证实在序列建模上与 LSTM 相当，机器人栈常用作历史观测/深度 latent 的轻量时序编码。"
related:
  - ./transformer.md
  - ./state-space-model-ssm.md
  - ./humanoid-policy-observation-inputs.md
  - ./deep-learning-foundations.md
  - ../comparisons/rnn-cnn-transformer-mamba.md
  - ../entities/pytorch.md
  - ../methods/model-based-rl.md
sources:
  - ../../sources/papers/cho_rnn_encoder_decoder_arxiv_1406_1078.md
  - ../../sources/papers/chung_gated_rnn_arxiv_1412_3555.md
  - ../../sources/sites/pytorch_nn_gru_docs.md
---

# GRU（Gated Recurrent Unit，门控循环单元）

**GRU** 是一种 **门控循环神经网络（gated RNN）** 单元：用 **reset 门 $r_t$** 与 **update 门 $z_t$** 控制新信息与旧隐状态 $h_{t-1}$ 的混合，缓解 vanilla RNN 的梯度消失，又比 LSTM **少一个 cell 状态与门**，在相似精度下更省参数与算力。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GRU | Gated Recurrent Unit | 两门控 RNN 单元（reset + update） |
| RNN | Recurrent Neural Network | 带隐状态递推的序列模型 |
| LSTM | Long Short-Term Memory | 三门 + cell 的经典门控 RNN |
| EMNLP | Empirical Methods in Natural Language Processing | Cho et al. 2014 发表会议 |
| PyTorch | PyTorch | 常用 `torch.nn.GRU` 实现 |
| BLEU | Bilingual Evaluation Understudy | 机器翻译自动评价指标（Cho 文实证） |

## 为什么重要

- **机器人时序编码默认选项之一：** 部分可观测策略、深度导航 latent、足球运球蒸馏等场景用 **CNN/MLP 提特征 + GRU 融历史**（见 [人形策略观测输入](./humanoid-policy-observation-inputs.md)），在 Transformer 普及前是 **低延迟、小模型** 的主线。
- **相对 LSTM 的工程优势：** [Chung et al. (2014)](../../sources/papers/chung_gated_rnn_arxiv_1412_3555.md) 在 music / speech 建模上显示 **GRU 与 LSTM 可比**；更少参数 → 更易在机载 GPU/Orin 上跑 50 Hz 闭环。
- **与 Transformer 的分工：** 短–中程历史、小 batch 在线推理仍常见 GRU；长上下文与大规模预训练更多走 [Transformer](./transformer.md) / SSM（见 [RNN vs CNN vs Transformer vs Mamba](../comparisons/rnn-cnn-transformer-mamba.md)）。

## 核心机制

### 1. 更新方程（与 PyTorch 文档对齐）

$$
\begin{aligned}
r_t &= \sigma(W_{ir}x_t + W_{hr}h_{t-1} + b_r) \\
z_t &= \sigma(W_{iz}x_t + W_{hz}h_{t-1} + b_z) \\
n_t &= \tanh(W_{in}x_t + b_{in} + r_t \odot (W_{hn}h_{t-1}+b_{hn})) \\
h_t &= (1-z_t)\odot n_t + z_t \odot h_{t-1}
\end{aligned}
$$

- **$z_t$（update）：** 控制「保留旧状态 vs 采纳候选 $n_t$」的比例。
- **$r_t$（reset）：** 决定候选状态在多大程度上 **忽略** $h_{t-1}$（捕捉短程依赖 vs 长程记忆的分工）。

### 2. 历史：Cho et al. (2014) Encoder–Decoder

[Kyunghyun Cho et al.](../../sources/papers/cho_rnn_encoder_decoder_arxiv_1406_1078.md) 在 **RNN Encoder–Decoder** 机器翻译框架中 **首次提出 GRU**，用门控 RNN 编码变长短语、解码目标序列，并作为 SMT log-linear 模型的短语特征 **提升 BLEU**。

### 3. GRU vs LSTM（Chung et al. 2014 实证）

| 维度 | GRU | LSTM |
|------|-----|------|
| 门数量 | 2（reset, update） | 3（input, forget, output）+ cell |
| 状态 | 仅 $h_t$ | $h_t$ + $c_t$ |
| 任务结论 | music / speech 上与 LSTM **相当** | 同左 |
| 选型倾向 | 参数/延迟敏感、中小序列 | 极长依赖或历史最佳实践代码栈 |

## 工程实践

| 维度 | 记录 |
|------|------|
| **PyTorch API** | `torch.nn.GRU(input_size, hidden_size, num_layers=..., batch_first=True, bidirectional=...)` |
| **张量形状** | 默认 `(L, N, H_in)`；机器人 batch 推理常设 `batch_first=True` → `(N, L, H_in)` |
| **实现差异** | PyTorch 在 **$W_{hn}h_{t-1}$ 之后** 做 $r_t\odot$，与原论文部分框架不同（见 [官方文档 Note](../../sources/sites/pytorch_nn_gru_docs.md)） |
| **初始化** | 权重 $\mathcal{U}(-1/\sqrt{H}, 1/\sqrt{H})$ |
| **机器人用法** | 堆在视觉编码器后融 $H$ 帧 proprio/深度；MBRL 世界模型用 GRU 预测 $h_t$（见 [model-based RL](../methods/model-based-rl.md)） |
| **导出** | 部署时常 **截断 unroll** 或转 ONNX/TorchScript；注意 hidden $h_0$ 热启动 |

## 局限与风险

- **训练并行差：** 与所有 RNN 一样时间维递推，难像 Transformer 整序列并行；长序列梯度仍可能衰减（门控仅缓解）。
- **长程 vs Transformer：** 极长历史（语言指令、百步动作块）通常 Transformer/Mamba 更优。
- **框架细节：** reset 乘法顺序、双向拼接方式影响与论文数值对齐；跨框架迁移需对照公式。
- **隐藏状态可解释性弱：** $h_t$ 是分布式表示，调试不如显式 memory token 直观。

## 关联页面

- [Transformer](./transformer.md) — 长序列与并行训练的主流替代
- [RNN / LSTM / GRU 总览](./recurrent-neural-network.md)
- [AI 架构地图](../overview/ai-architecture-map.md)
- [RNN vs CNN vs Transformer vs Mamba](../comparisons/rnn-cnn-transformer-mamba.md) — 骨干选型
- [人形策略观测输入](./humanoid-policy-observation-inputs.md) — GRU 隐状态在策略中的角色
- [PyTorch](../entities/pytorch.md) — `nn.GRU` 所属框架
- [Model-Based RL](../methods/model-based-rl.md) — 集成 RNN 世界模型

## 参考来源

- [Cho et al. (2014) RNN Encoder–Decoder / GRU 原创](../../sources/papers/cho_rnn_encoder_decoder_arxiv_1406_1078.md)
- [Chung et al. (2014) GRU vs LSTM 实证](../../sources/papers/chung_gated_rnn_arxiv_1412_3555.md)
- [PyTorch torch.nn.GRU 官方文档](../../sources/sites/pytorch_nn_gru_docs.md)

## 推荐继续阅读

- [arXiv:1406.1078](https://arxiv.org/abs/1406.1078) — GRU 原论文 PDF
- [arXiv:1412.3555](https://arxiv.org/abs/1412.3555) — 门控 RNN 对比实验
- [PyTorch GRU API](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html)
