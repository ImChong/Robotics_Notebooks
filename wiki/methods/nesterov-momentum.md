---
type: method
tags: [deep-learning, optimization, nesterov, momentum, sgd, training]
status: complete
updated: 2026-06-27
summary: "Nesterov 动量在更新前先沿动量方向前瞻一步再取梯度，在凸优化中达到更优收敛率，是经典动量的理论加强版。"
related:
  - ./sgd-momentum.md
  - ./sgd.md
  - ./adam.md
  - ../comparisons/deep-learning-optimizers.md
sources:
  - ../../sources/papers/deep_learning_optimizers.md
  - ../../sources/books/udl_book.md
---

# Nesterov Momentum（Nesterov 加速梯度）

**Nesterov Accelerated Gradient（NAG）**：在施加动量更新 **之前**，先沿当前速度方向做一步「前瞻」，在 **前瞻点** 计算梯度。相比经典 [SGD Momentum](./sgd-momentum.md)，能更早感知即将到来的曲率变化，在凸优化中达到 $O(1/k^2)$ 收敛率；与 [深度学习基础](../concepts/deep-learning-foundations.md) 中的优化训练章节衔接。

## 一句话定义

> 先「往前看一眼」再决定往哪走——用前瞻梯度修正动量，比标准动量更善于在弯曲曲面上刹车。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| NAG | Nesterov Accelerated Gradient | Nesterov 1983 提出的加速梯度法 |
| SGD | Stochastic Gradient Descent | 底层仍为随机梯度估计 |
| HB | Heavy-ball Method | 经典动量，无前瞻 |
| LR | Learning Rate | 步长 $\eta$ |
| $\mu$ | Momentum Coefficient | 动量系数 |

## 主要技术路线

### 1. 更新规则（常见等价写法）

先计算前瞻梯度 $g_t = \nabla f(\theta_t - \mu v_t)$，再更新：

$$
v_{t+1} = \mu v_t + g_t, \qquad \theta_{t+1} = \theta_t - \eta v_{t+1}
$$

或合并为「先动量半步、再梯度修正」的两步形式。PyTorch `SGD(nesterov=True)` 在 `momentum>0` 时启用此变体。

### 2. 与经典动量的直觉差异

| | 经典 Momentum | Nesterov |
|---|--------------|----------|
| 梯度取点 | 当前 $\theta_t$ | 前瞻 $\theta_t - \mu v_t$ |
| 凸理论收敛率 | $O(1/k)$ | $O(1/k^2)$ |
| 非凸深度网络 | 广泛使用 | 有时略优，差异因任务而异 |

### 3. 实践要点

Sutskever et al. (2013) 在 RNN 上报告 Nesterov 与经典动量互有胜负；调参时通常 **固定 $\mu=0.9$**，主要扫学习率。深度非凸情形下理论优势不一定转化为稳定增益。

## 优势与局限

| 优势 | 局限 |
|------|------|
| 凸优化理论最优一阶收敛率之一 | 非凸深度网络增益不稳定 |
| 实现成本与经典动量相同 | 仍需手动学习率调度 |
| 在部分 RNN/CNN 任务上收敛更快 | 大模型时代多被 Adam/AdamW 取代 |

## 在机器人中的典型应用

- **早期序列模型训练**：RNN 策略或时序编码器曾用 Nesterov SGD。
- **小规模网络原型**：调参预算有限时，Nesterov 是低成本尝试项。
- **与 Adam 对比基线**：论文消融中作为「一阶加速 SGD」对照组。

## 关联页面

- [SGD Momentum](./sgd-momentum.md)
- [SGD](./sgd.md)
- [Adam](./adam.md)
- [Deep Learning Optimizers 对比](../comparisons/deep-learning-optimizers.md)

## 参考来源

- [Deep Learning Optimizers 论文摘录](../../sources/papers/deep_learning_optimizers.md) — Nesterov (1983)、Sutskever et al. (2013)
- [Understanding Deep Learning (Prince, 2023)](../../sources/books/udl_book.md)

## 推荐继续阅读

- [Nesterov, A method for unconstrained convex minimization (1983)](https://doi.org/10.1016/0273-0979(83)90028-3)
- [PyTorch SGD nesterov 参数说明](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)
