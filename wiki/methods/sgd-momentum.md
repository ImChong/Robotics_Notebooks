---
type: method
tags: [deep-learning, optimization, momentum, sgd, training]
status: complete
updated: 2026-06-27
summary: "SGD Momentum 在梯度方向外叠加历史速度，平滑更新轨迹并加速沿一致方向的收敛，是深度网络训练的经典一阶加速技巧。"
related:
  - ./sgd.md
  - ./nesterov-momentum.md
  - ./adam.md
  - ../concepts/deep-learning-foundations.md
  - ../comparisons/deep-learning-optimizers.md
sources:
  - ../../sources/papers/deep_learning_optimizers.md
  - ../../sources/books/udl_book.md
---

# SGD Momentum（动量随机梯度下降）

**SGD Momentum**：在 [SGD](./sgd.md) 基础上维护速度向量 $v_t$，将当前梯度与历史速度加权叠加，使参数更新在 **一致梯度方向** 上加速、在 **振荡方向** 上相互抵消。

## 一句话定义

> 给梯度下降加上「惯性」——沿稳定下降方向越滚越快，在来回震荡的方向上被阻尼掉。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SGD | Stochastic Gradient Descent | 基础随机梯度下降 |
| HB | Heavy-ball Method | Polyak 1964 提出的动量方法 |
| EMA | Exponential Moving Average | 动量可视为梯度的指数滑动平均 |
| LR | Learning Rate | 步长 $\eta$ |
| $\mu$ | Momentum Coefficient | 动量系数，典型 0.9 |

## 主要技术路线

### 1. 更新规则（PyTorch 常用形式）

$$
v_{t+1} = \mu v_t + g_t, \qquad \theta_{t+1} = \theta_t - \eta v_{t+1}
$$

其中 $g_t$ 为当前 mini-batch 梯度，$\mu \in [0,1)$ 为动量系数。等价地，速度是历史梯度的指数加权平均。

### 2. 为何有效

- **条件数大的损失曲面**：纯 SGD 在狭长山谷中 zig-zag；动量沿谷底方向累积速度，减少横向振荡。
- **随机梯度噪声**：动量低通滤波噪声，使轨迹更平滑。
- Sutskever et al. (2013) 在 LSTM 语言模型上证明：合适初始化 + $\mu=0.9$ 的动量可将收敛步数缩短一个数量级。

### 3. 与 Nesterov 的区别

经典动量用 **当前位置** 的梯度；[Nesterov Momentum](./nesterov-momentum.md) 在 **前瞻位置** 评估梯度，凸优化理论收敛率更优。实践中二者常互换试验。

## 优势与局限

| 优势 | 局限 |
|------|------|
| 显著加速 CNN/ResNet 等视觉模型训练 | 超参 $\mu$ 与 $\eta$ 需联合调节 |
| 实现极简，几乎无额外内存（一阶动量） | 对 Transformer 等大模型通常不如 AdamW |
| 与 SGD 学习率调度成熟配套 | 动量过大时可能 overshoot 极小值 |

## 在机器人中的典型应用

- **视觉 backbone 预训练**：ResNet、MobileNet 等 ImageNet 训练的经典配方 SGD + Momentum 0.9。
- **RL 早期实践**：部分 locomotion 工作用 Momentum SGD 训策略网络，现多被 Adam 取代。
- **Sim2Real 微调**：小学习率 + 动量可在预训练权重附近平稳微调感知模块。

## 关联页面

- [SGD](./sgd.md)
- [Nesterov Momentum](./nesterov-momentum.md)
- [Adam](./adam.md)
- [深度学习基础](../concepts/deep-learning-foundations.md)
- [Deep Learning Optimizers 对比](../comparisons/deep-learning-optimizers.md)

## 参考来源

- [Deep Learning Optimizers 论文摘录](../../sources/papers/deep_learning_optimizers.md) — Polyak (1964)、Sutskever et al. (2013)
- [Understanding Deep Learning (Prince, 2023)](../../sources/books/udl_book.md)

## 推荐继续阅读

- [Sutskever et al., On the Importance of Initialization and Momentum in Deep Learning (ICML 2013)](https://proceedings.mlr.press/v28/sutskever13.html)
- [PyTorch torch.optim.SGD (momentum 参数)](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)
