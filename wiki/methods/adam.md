---
type: method
tags: [deep-learning, optimization, adam, adaptive, training]
status: complete
updated: 2026-06-27
summary: "Adam 结合梯度一阶矩与平方梯度二阶矩的偏差校正估计，为每个参数提供自适应步长，是深度学习与机器人 RL 最常用的默认优化器之一。"
related:
  - ./adamw.md
  - ./rmsprop.md
  - ./sgd.md
  - ./sgd-momentum.md
  - ../concepts/backpropagation.md
  - ../methods/ppo.md
  - ../comparisons/deep-learning-optimizers.md
  - ../entities/pytorch.md
sources:
  - ../../sources/papers/deep_learning_optimizers.md
  - ../../sources/books/udl_book.md
  - ../../sources/repos/pytorch-official.md
---

# Adam（Adaptive Moment Estimation）

**Adam**：对 mini-batch 梯度 $g_t$ 同时估计 **一阶矩** $m_t$（动量）与 **二阶矩** $v_t$（[RMSProp](./rmsprop.md) 式方差），经 **偏差校正** 后做 per-parameter 自适应更新；是深度网络与 [PPO](./ppo.md) 等机器人 RL 训练的事实默认优化器（预训练大模型见 [AdamW](./adamw.md)）。

## 一句话定义

> 每个参数各算各的步长——用梯度的均值定方向、用梯度平方的均值定步子大小，并对冷启动做偏差修正。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Adam | Adaptive Moment Estimation | Kingma & Ba 2015 |
| EMA | Exponential Moving Average | $m_t, v_t$ 的指数滑动平均 |
| SGD | Stochastic Gradient Descent | 非自适应基线 |
| RL | Reinforcement Learning | PPO/SAC 等常用 Adam |
| LR | Learning Rate | 全局 $\eta$，默认常 1e-3 ~ 3e-4 |

## 主要技术路线

### 1. 更新规则

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

偏差校正：$\hat{m}_t = m_t / (1-\beta_1^t)$，$\hat{v}_t = v_t / (1-\beta_2^t)$

$$
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

默认 $\beta_1=0.9$，$\beta_2=0.999$，$\epsilon=10^{-8}$。

### 2. 为何成为默认

- **少调参**：同一组 $(\eta, \beta_1, \beta_2)$ 在 CNN、MLP、小 Transformer 上往往「够用」。
- **异构参数尺度**：嵌入层、归一化层、输出头梯度尺度差异大，per-parameter 缩放减轻手工分层 LR 负担。
- **与 RL 兼容**：on-policy 多 epoch 更新时，Adam 的平滑性有助于稳定策略网络。

### 3. 已知局限

- **泛化**：部分视觉任务上 SGD + Momentum 泛化优于 Adam（尖锐/平坦极小值讨论）。
- **权重衰减语义**：在 Adam 中把 L2 正则并入梯度 **不等于** 标准 weight decay；应用 [AdamW](./adamw.md) 修正。
- **Transformer 预训练**：大模型微调与预训练 **首选 AdamW** 而非原生 Adam。

## 优势与局限

| 优势 | 局限 |
|------|------|
| 默认超参鲁棒，工程友好 | 部分任务泛化弱于调优 SGD |
| RL / IL 代码库广泛默认 | L2 与衰减耦合，需 AdamW |
| 实现成熟（PyTorch/TF/JAX） | 二阶矩增加内存（与参数同量级） |

## 在机器人中的典型应用

- **PPO / SAC 策略-价值网络**：Isaac Lab、legged_gym 等默认 `torch.optim.Adam`。
- **模仿学习 / VLA 微调**：除大模型预训练外，中小规模策略头常用 Adam。
- **扩散策略 / ACT**：动作头与编码器联合训练的典型选择。

## 关联页面

- [AdamW](./adamw.md)
- [RMSProp](./rmsprop.md)
- [SGD](./sgd.md)
- [SGD Momentum](./sgd-momentum.md)
- [PPO](./ppo.md)
- [反向传播](../concepts/backpropagation.md)
- [Deep Learning Optimizers 对比](../comparisons/deep-learning-optimizers.md)
- [PyTorch](../entities/pytorch.md)

## 参考来源

- [Deep Learning Optimizers 论文摘录](../../sources/papers/deep_learning_optimizers.md) — Kingma & Ba (2015)
- [Understanding Deep Learning (Prince, 2023)](../../sources/books/udl_book.md)
- [PyTorch 官方站点与文档索引](../../sources/repos/pytorch-official.md)

## 推荐继续阅读

- [Kingma & Ba, Adam: A Method for Stochastic Optimization (ICLR 2015)](https://arxiv.org/abs/1412.6980)
- [PyTorch torch.optim.Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)
