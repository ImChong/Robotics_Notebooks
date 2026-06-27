---
type: method
tags: [deep-learning, optimization, sgd, training]
status: complete
updated: 2026-06-27
summary: "SGD 用 mini-batch 随机梯度迭代更新参数，是神经网络训练最基础的一阶优化器，也是理解动量与自适应方法的原点。"
related:
  - ./sgd-momentum.md
  - ./adam.md
  - ../concepts/backpropagation.md
  - ../concepts/deep-learning-foundations.md
  - ../comparisons/deep-learning-optimizers.md
  - ../entities/pytorch.md
sources:
  - ../../sources/papers/deep_learning_optimizers.md
  - ../../sources/books/udl_book.md
---

# SGD（Stochastic Gradient Descent）

**SGD（随机梯度下降）**：用训练集 mini-batch 上估计的梯度 $\hat{g}_t$ 替代全数据精确梯度，按 $\theta_{t+1} = \theta_t - \eta \hat{g}_t$ 迭代更新网络参数；是 [反向传播](../concepts/backpropagation.md) 之后最常用的参数更新规则。

## 一句话定义

> 每次只看一小批样本的梯度来更新权重——用噪声换速度，用大量迭代换收敛。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SGD | Stochastic Gradient Descent | 用随机 mini-batch 梯度做迭代优化 |
| GD | Gradient Descent | 全批量梯度下降，SGD 的低方差特例 |
| MB-SGD | Mini-batch SGD | 现代深度学习默认形态，batch 大小 $B$ |
| BP | Backpropagation | 计算 $\hat{g}_t$ 的算法，与 SGD 概念分离 |
| LR | Learning Rate | 全局步长 $\eta$，SGD 最关键超参 |

## 主要技术路线

### 1. 更新规则

给定损失 $L(\theta)$，从 batch $\mathcal{B}_t$ 采样后：

$$
\hat{g}_t = \frac{1}{|\mathcal{B}_t|}\sum_{(x,y)\in\mathcal{B}_t} \nabla_\theta \ell(f_\theta(x), y), \qquad
\theta_{t+1} = \theta_t - \eta_t \hat{g}_t
$$

Robbins & Monro (1951) 证明在适当衰减的 $\eta_t$ 下，含噪梯度迭代可收敛到驻点。

### 2. Mini-batch 的权衡

| batch 大小 | 梯度方差 | 每步算力 | 典型场景 |
|-----------|---------|---------|---------|
| 小（32–128） | 高，有正则化效应 | 低 | 小模型、在线学习 |
| 大（1k–32k） | 低，方向更准 | 高 | 大模型预训练、需配合 LR 缩放 |

Bottou (2010) 总结：**学习率调度**（warmup、cosine decay、step decay）往往比换复杂优化器更能决定 SGD 成败。

### 3. 与全批量 GD 的关系

全批量 GD 每步用整个数据集梯度，方向准确但昂贵；SGD 引入梯度噪声，在非凸深度网络中常能 **逃离尖锐极小值**，泛化有时更好，但收敛轨迹更抖。

## 优势与局限

| 优势 | 局限 |
|------|------|
| 概念最简单，易分析与调试 | 病态条件数时 zig-zag，收敛慢 |
| 内存占用低（只需当前 batch 梯度） | 需仔细调 $\eta$ 与 schedule |
| 大 batch 训练时配合 LR 线性缩放规则成熟 | 无 per-parameter 自适应，稀疏/异构参数尺度难兼顾 |
| 动量、Adam 等均可视为 SGD 扩展 | 纯 SGD 在 Transformer 等大模型上通常不如 AdamW |

## 在机器人中的典型应用

- **RL 策略网络**：PPO 等对同一 rollout batch 做多轮 mini-batch SGD；常与 [Adam](./adam.md) 联用。
- **视觉预训练 backbone**：ResNet 时代 ImageNet 分类常用 SGD + Momentum；现代 ViT 多转 [AdamW](./adamw.md)。
- **小批量模仿学习**：数据有限时小 batch SGD 的噪声有时缓解过拟合，但需监控验证损失震荡。

## 关联页面

- [SGD Momentum](./sgd-momentum.md)
- [Adam](./adam.md)
- [反向传播](../concepts/backpropagation.md)
- [深度学习基础](../concepts/deep-learning-foundations.md)
- [Deep Learning Optimizers 对比](../comparisons/deep-learning-optimizers.md)
- [PyTorch](../entities/pytorch.md)
- [PPO](./ppo.md)

## 参考来源

- [Deep Learning Optimizers 论文摘录](../../sources/papers/deep_learning_optimizers.md) — Robbins & Monro (1951)、Bottou (2010)
- [Understanding Deep Learning (Prince, 2023)](../../sources/books/udl_book.md) — 第 6 章 Training Models

## 推荐继续阅读

- [Bottou, Stochastic Gradient Descent Tricks (2010)](https://leon.bottou.org/publications/pdf/tricks-2010.pdf)
- [PyTorch torch.optim.SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)
