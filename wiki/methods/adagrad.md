---
type: method
tags: [deep-learning, optimization, adagrad, adaptive, training]
status: complete
updated: 2026-06-27
summary: "Adagrad 按参数维度累积历史梯度平方并做 per-parameter 学习率缩放，适合稀疏梯度，但学习率单调衰减可能导致后期停滞。"
related:
  - ./rmsprop.md
  - ./adadelta.md
  - ./adam.md
  - ./sgd.md
  - ../comparisons/deep-learning-optimizers.md
sources:
  - ../../sources/papers/deep_learning_optimizers.md
  - ../../sources/books/udl_book.md
---

# Adagrad（Adaptive Gradient）

**Adagrad**：为每个参数维护 **累积梯度平方** $G_t$，用 $\eta / \sqrt{G_t + \epsilon}$ 缩放各维步长——梯度历史大的参数获得更小有效学习率，稀疏特征上尤为有效；属于 [深度学习基础](../concepts/deep-learning-foundations.md) 自适应优化路线。

## 一句话定义

> 给每个权重单独调学习率——经常收到大梯度的参数自动走小步，稀疏参数保留较大步长。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Adagrad | Adaptive Gradient Algorithm | Duchi et al. 2011 提出 |
| SGD | Stochastic Gradient Descent | 非自适应基线 |
| LR | Learning Rate | 全局缩放 $\eta$ |
| NLP | Natural Language Processing | 稀疏词嵌入的典型应用场景 |
| EMA | Exponential Moving Average | Adagrad 用累积和而非 EMA |

## 主要技术路线

### 1. 更新规则

$$
G_t = G_{t-1} + g_t \odot g_t, \qquad
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t} + \epsilon} \odot g_t
$$

$\odot$ 为逐元素乘；$G_t$ 单调递增，故有效学习率 **只减不增**。

### 2. 设计动机

- **稀疏数据**（如 NLP 词袋）：多数参数梯度为 0，少数活跃参数需要较大步长；Adagrad 自动实现 per-parameter 缩放。
- **在线凸优化**：Duchi et al. 给出遗憾界，适合流式学习。

### 3. 主要缺陷

$G_t$ 持续累积导致有效学习率过早趋近 0，**长程深度网络训练** 中常在中途停滞。这直接催生了 [RMSProp](./rmsprop.md)（指数衰减累积）与 [Adam](./adam.md)。

## 优势与局限

| 优势 | 局限 |
|------|------|
| 稀疏特征上表现优异 | 学习率单调衰减，深度网络易早停 |
| 无需手动 per-layer LR | 非平稳目标（RL）效果通常较差 |
| 理论在线凸优化保证成熟 | 已被 RMSProp/Adam 在多数视觉任务超越 |

## 在机器人中的典型应用

- **稀疏奖励 / 稀疏特征工程**：手工特征维度差异大时可尝试 Adagrad。
- **词表级嵌入**（若用离散 token 接口）：稀疏梯度场景仍有参考价值。
- **现代默认选择**：端到端视觉-运动策略训练 **更常用 Adam/AdamW**；Adagrad 多作历史基线或教学对照。

## 关联页面

- [RMSProp](./rmsprop.md)
- [Adadelta](./adadelta.md)
- [Adam](./adam.md)
- [SGD](./sgd.md)
- [Deep Learning Optimizers 对比](../comparisons/deep-learning-optimizers.md)

## 参考来源

- [Deep Learning Optimizers 论文摘录](../../sources/papers/deep_learning_optimizers.md) — Duchi et al. (2011)
- [Understanding Deep Learning (Prince, 2023)](../../sources/books/udl_book.md)

## 推荐继续阅读

- [Duchi et al., Adaptive Subgradient Methods (JMLR 2011)](https://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
- [PyTorch torch.optim.Adagrad](https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html)
