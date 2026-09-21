---
type: concept
tags: [tcn, convolution, sequence-modeling, architecture]
status: complete
updated: 2026-09-21
summary: "时间卷积网络用因果膨胀卷积在序列上做可并行的固定感受野建模，是 RNN 的卷积替代而非 Transformer 的替代；适合窗口化 IMU/触觉历史。"
related:
  - ./convolutional-neural-network.md
  - ./recurrent-neural-network.md
  - ./transformer.md
  - ../comparisons/rnn-cnn-transformer-mamba.md
  - ../overview/ai-architecture-map.md
sources:
  - ../../sources/papers/bai_tcn_arxiv_1803_01271.md
  - ../../sources/papers/ai_architecture_foundations.md
---

# TCN（Temporal Convolutional Network，时间卷积网络）

**TCN**：在时间轴上使用 **因果卷积**（不看未来）、**膨胀卷积**（指数扩大感受野）和 **残差块** 的一维卷积网，把序列建模写成可并行的 CNN。

## 一句话定义

给序列加一组「只看过去、感受野可控」的卷积滤波器，而不是维护递推隐状态。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| TCN | Temporal Convolutional Network | 因果膨胀卷积序列骨干 |
| Causal | Causal convolution | 输出 \(t\) 仅依赖 \(\le t\) 的输入 |
| Dilation | Dilated convolution | 隔点采样以指数扩大感受野 |
| RF | Receptive Field | 有效历史长度 |
| IMU | Inertial Measurement Unit | 典型的窗口化 TCN 输入 |

## 为什么重要

- [Bai, Kolter & Koltun, 2018](../../sources/papers/bai_tcn_arxiv_1803_01271.md) 在多项基准上表明：通用 TCN 常常比通用 LSTM/GRU **更准、更好训、可并行**。
- 机器人很多「时序」其实是 **固定窗口**（0.5–2 s 本体历史），与 TCN 的归纳偏置对齐。
- 它提醒选型不要默认 RNN：先问感受野是否固定。

## 核心原理

层 \(l\) 使用膨胀 \(d=2^{l}\) 的因果卷积，感受野随深度指数增长。残差连接稳定深堆叠。整段序列可在训练时并行卷积；推理可缓存边界激活做流式更新。

```mermaid
flowchart LR
  seq["序列 x_1..x_T"] --> c1["因果膨胀 d=1"]
  c1 --> c2["因果膨胀 d=2"]
  c2 --> c3["因果膨胀 d=4"]
  c3 --> y["逐步输出"]
```

## 工程实践

| 项 | 建议 |
|----|------|
| 感受野 | 按控制周期反推需要覆盖的物理时间，再设层数与核宽 |
| 因果 | 训练若泄漏未来，上真机会「会看答案」 |
| 对比基线 | 同一窗口的 MLP-on-stack 往往已经很强，先比再上 TCN |
| 长上下文 | 窗口不够就换 [Transformer](./transformer.md) / [Mamba](./mamba.md)，不要无限加膨胀 |

## 局限与风险

- **固定感受野**：超出设计窗口的依赖被硬切断。
- **不是注意力**：不能按内容动态跳读遥远 token。
- 膨胀过大造成网格伪影（checkerboard）；核与膨胀要配对。

## 关联页面

- [CNN](./convolutional-neural-network.md)
- [RNN / LSTM / GRU](./recurrent-neural-network.md)
- [RNN vs CNN vs Transformer vs Mamba](../comparisons/rnn-cnn-transformer-mamba.md)
- [AI 架构地图](../overview/ai-architecture-map.md)

## 参考来源

- [Bai et al. TCN（arXiv:1803.01271）](../../sources/papers/bai_tcn_arxiv_1803_01271.md)
- [AI 架构地图一手论文簇](../../sources/papers/ai_architecture_foundations.md)

## 推荐继续阅读

- 官方实现：<https://github.com/locuslab/TCN>
