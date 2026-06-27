---
type: method
tags: [deep-learning, optimization, adadelta, adaptive, training]
status: complete
updated: 2026-06-27
summary: "Adadelta 用梯度更新量与参数更新量的 RMS 之比自适应缩放步长，无需手动全局学习率，是 Adagrad 的改进变体。"
related:
  - ./adagrad.md
  - ./rmsprop.md
  - ./adam.md
  - ./sgd.md
  - ../comparisons/deep-learning-optimizers.md
sources:
  - ../../sources/papers/deep_learning_optimizers.md
  - ../../sources/books/udl_book.md
---

# Adadelta（Adaptive Delta）

**Adadelta**：同时维护 **梯度平方的 EMA** 与 **参数更新量的 EMA**，用二者 RMS 之比确定每步缩放因子，**取消显式全局学习率 $\eta$**，在 [Adagrad](./adagrad.md) 框架下缓解学习率消失；与 [深度学习基础](../concepts/deep-learning-foundations.md) 训练模型章节中的自适应方法并列。

## 一句话定义

> 用「过去参数挪了多远」对比「当前梯度有多大」来决定这一步走多远——自带步长尺度，不用手调学习率。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Adadelta | Adaptive Learning Rate Method | Zeiler 2012 提出 |
| RMS | Root Mean Square | 对历史量做均方根归一化 |
| EMA | Exponential Moving Average | 衰减窗口，超参 $\rho$ 典型 0.95 |
| SGD | Stochastic Gradient Descent | 需手动 LR 的对照 |
| CNN | Convolutional Neural Network | 论文主要验证场景 |

## 主要技术路线

### 1. 更新规则（概念形式）

维护 $E[g^2]_t$（梯度平方 EMA）与 $E[\Delta\theta^2]_t$（更新量平方 EMA）：

$$
\Delta\theta_t = - \frac{\sqrt{E[\Delta\theta^2]_{t-1} + \epsilon}}{\sqrt{E[g^2]_t + \epsilon}} \odot g_t, \qquad
\theta_{t+1} = \theta_{t-1} + \Delta\theta_t
$$

步长由 **历史更新幅度** 自动标定，无需外部 $\eta$。

### 2. 设计动机

- 继承 Adagrad 的 per-parameter 自适应思想。
- 用更新量 RMS 作为「单位尺度」，避免 $G_t$ 无限增长。
- Zeiler (2012) 在 MNIST/CIFAR 上报告与 SGD + 精细调参相当的表现。

### 3. 与 RMSProp / Adam 的关系

| 方法 | 全局 LR | 一阶动量 | 二阶归一化 |
|------|---------|---------|-----------|
| RMSProp | 需要 $\eta$ | 无 | $g^2$ EMA |
| Adadelta | **不需要** | 无 | $g^2$ 与 $\Delta\theta^2$ 之比 |
| Adam | 需要 $\eta$ | 有 | $g^2$ EMA + 偏差校正 |

## 优势与局限

| 优势 | 局限 |
|------|------|
| 省去全局学习率搜索 | 超参 $\rho$、$\epsilon$ 仍影响收敛 |
| 训练初期更稳健 | 大模型时代极少作为默认 |
| 概念上优雅地闭合自适应尺度 | 机器人 RL 中几乎不见使用 |

## 在机器人中的典型应用

- **历史 CNN 感知模块**：早期视觉预训练实验中有使用记录。
- **教学对照**：理解「无 LR 自适应」与 RMSProp/Adam 的设计差异。
- **现代实践**：端到端策略学习 **优先 AdamW**；Adadelta 主要保留为知识图谱节点与基线参考。

## 关联页面

- [Adagrad](./adagrad.md)
- [RMSProp](./rmsprop.md)
- [Adam](./adam.md)
- [SGD](./sgd.md)
- [Deep Learning Optimizers 对比](../comparisons/deep-learning-optimizers.md)

## 参考来源

- [Deep Learning Optimizers 论文摘录](../../sources/papers/deep_learning_optimizers.md) — Zeiler (2012)
- [Understanding Deep Learning (Prince, 2023)](../../sources/books/udl_book.md)

## 推荐继续阅读

- [Zeiler, ADADELTA: An Adaptive Learning Rate Method (arXiv:1212.5701)](https://arxiv.org/abs/1212.5701)
- [PyTorch torch.optim.Adadelta](https://pytorch.org/docs/stable/generated/torch.optim.Adadelta.html)
