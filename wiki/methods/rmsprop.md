---
type: method
tags: [deep-learning, optimization, rmsprop, adaptive, training]
status: complete
updated: 2026-06-27
summary: "RMSProp 用梯度平方的指数滑动平均归一化步长，缓解 Adagrad 学习率持续衰减问题，是 Adam 二阶矩估计的直接前驱。"
related:
  - ./adagrad.md
  - ./adam.md
  - ./adadelta.md
  - ./sgd.md
  - ../comparisons/deep-learning-optimizers.md
sources:
  - ../../sources/papers/deep_learning_optimizers.md
  - ../../sources/books/udl_book.md
---

# RMSProp（Root Mean Square Propagation）

**RMSProp**：对梯度平方 $g_t^2$ 做 **指数滑动平均（EMA）** 得到 $v_t$，用 $g_t / \sqrt{v_t + \epsilon}$ 归一化更新方向，使 per-parameter 学习率在非平稳训练中 **可增可减**，克服 [Adagrad](./adagrad.md) 的单调衰减缺陷；是 [深度学习基础](../concepts/deep-learning-foundations.md) 中自适应步长的重要一环。

## 一句话定义

> 只记住「最近梯度有多大」——用滑动窗口替代 Adagrad 的永久累积，让步长不会越训越小。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RMSProp | Root Mean Square Propagation | Hinton 课程讲义中提出 |
| EMA | Exponential Moving Average | 衰减系数 $\beta$，典型 0.9 或 0.99 |
| SGD | Stochastic Gradient Descent | 非自适应对照 |
| Adam | Adaptive Moment Estimation | 在 RMSProp 基础上加一阶动量 |
| LR | Learning Rate | 全局步长 $\eta$ |

## 主要技术路线

### 1. 更新规则

$$
v_t = \beta v_{t-1} + (1-\beta) g_t^2, \qquad
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon} \odot g_t
$$

$\beta$ 控制历史窗口长度；$\beta$ 接近 1 时更平滑。

### 2. 与 Adagrad 的对比

| | Adagrad | RMSProp |
|---|---------|---------|
| 二阶统计 | 累积和 $\sum g^2$ | EMA of $g^2$ |
| 有效 LR 趋势 | 单调递减 | 可随梯度尺度回升 |
| 深度 CNN | 易停滞 | RNN/CNN 上更实用 |

### 3. 与 Adam 的关系

[Adam](./adam.md) 可视为 **RMSProp + 一阶动量 + 偏差校正** 的组合；现代实践几乎总是直接用 Adam/AdamW，但理解 RMSProp 有助于读懂 Adam 的二阶矩分支。

## 优势与局限

| 优势 | 局限 |
|------|------|
| 解决 Adagrad 学习率消失 | 无官方唯一论文，各实现细节略有差异 |
| RNN 等非平稳目标上历史表现良好 | 已被 Adam 在多数任务取代 |
| 实现简单、内存为一阶 | 仍依赖全局 $\eta$ 与 $\beta$ 调参 |

## 在机器人中的典型应用

- **RNN / 时序策略**：早期 recurrent policy 训练常用 RMSProp。
- **教学与消融**：作为 Adam 二阶矩分支的「去动量版」对照。
- **现代默认**：机器人 RL 与 VLA 训练 **首选 Adam/AdamW**；RMSProp 多见于遗留代码或特殊 RNN 任务。

## 关联页面

- [Adagrad](./adagrad.md)
- [Adam](./adam.md)
- [Adadelta](./adadelta.md)
- [SGD](./sgd.md)
- [Deep Learning Optimizers 对比](../comparisons/deep-learning-optimizers.md)

## 参考来源

- [Deep Learning Optimizers 论文摘录](../../sources/papers/deep_learning_optimizers.md) — Tieleman & Hinton (2012)
- [Understanding Deep Learning (Prince, 2023)](../../sources/books/udl_book.md)

## 推荐继续阅读

- [Tieleman & Hinton, RMSProp lecture slides (CSC321)](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
- [PyTorch torch.optim.RMSprop](https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html)
