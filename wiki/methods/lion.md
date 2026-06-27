---
type: method
tags: [deep-learning, optimization, lion, adaptive, training]
status: complete
updated: 2026-06-27
summary: "Lion 是符号搜索发现的轻量优化器，仅用一阶动量并以 sign 函数更新参数，内存省、计算快，在部分视觉与语言任务上可与 AdamW 竞争。"
related:
  - ./adamw.md
  - ./adam.md
  - ./sgd-momentum.md
  - ../comparisons/deep-learning-optimizers.md
  - ../entities/pytorch.md
sources:
  - ../../sources/papers/deep_learning_optimizers.md
---

# Lion（EvoLved Sign Momentum）

**Lion**：Chen et al. (2023) 通过 **符号搜索（symbolic search）** 自动发现的优化算法；仅维护 **一阶动量** $m_t$，更新方向取 $\mathrm{sign}(m_t)$，**不维护二阶矩**，从而比 [AdamW](./adamw.md) 更省内存与算力；可作为 [深度学习基础](../concepts/deep-learning-foundations.md) 优化训练的新兴备选。

## 一句话定义

> 动量只记方向、更新只看正负号——用 sign 步长砍掉二阶统计，换更轻的优化器。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Lion | EvoLved Sign Momentum | Google Brain 符号搜索发现 |
| AdamW | Adam with decoupled Weight decay | 主要对照基线 |
| EMA | Exponential Moving Average | 动量 $\beta_1$，典型 0.9 |
| WD | Weight Decay | 解耦权重衰减，语义同 AdamW |
| LR | Learning Rate | 通常比 Adam 小 3–10 倍 |

## 主要技术路线

### 1. 更新规则

$$
c_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$

$$
m_t = \beta_2 m_{t-1} + (1-\beta_2) g_t, \qquad
\theta_{t+1} = \theta_t - \eta \big( \mathrm{sign}(c_t) + \lambda \theta_t \big)
$$

（具体实现中 $\beta_1, \beta_2$ 角色与论文一致；PyTorch 第三方实现见 `lion-pytorch`。）

### 2. 设计特点

| 特性 | Lion | AdamW |
|------|------|-------|
| 状态变量 | 仅 $m_t$（与 $\theta$ 同形） | $m_t + v_t$ |
| 更新幅度 | 固定 $\pm \eta$（sign） | 自适应连续步长 |
| 内存 | **约减半**（无 $v_t$） | 二阶矩等量额外内存 |
| 发现方式 | 自动符号搜索 | 手工组合 RMSProp + Momentum |

### 3. 实践要点

- 学习率通常 **小于 AdamW**（论文建议约为 Adam 的 1/3–1/10）。
- weight decay 同样 **解耦** 施加。
- 在 ImageNet、JFT、语言建模等任务上报告与 AdamW **可比或更优** 的样本效率，但 **并非全面替代**；新任务仍需验证。

## 优势与局限

| 优势 | 局限 |
|------|------|
| 内存与通信量更低，利于大模型 sharding | 生态成熟度不及 AdamW |
| sign 更新对量化友好 | 机器人 RL 领域实证仍少 |
| 自动发现，有研究新鲜度 | 非凸景观下理论保证弱于经典方法 |

## 在机器人中的典型应用

- **大视觉 backbone 预训练**：若内存瓶颈显著，可试验 Lion 替代 AdamW。
- **边端微调**：sign 更新与低比特训练结合是潜在方向（待更多实证）。
- **当前默认**：机器人 RL / VLA **仍以 AdamW 为主**；Lion 作为新兴备选写入知识库供选型参考。

## 关联页面

- [AdamW](./adamw.md)
- [Adam](./adam.md)
- [SGD Momentum](./sgd-momentum.md)
- [Deep Learning Optimizers 对比](../comparisons/deep-learning-optimizers.md)
- [PyTorch](../entities/pytorch.md)

## 参考来源

- [Deep Learning Optimizers 论文摘录](../../sources/papers/deep_learning_optimizers.md) — Chen et al. (2023)

## 推荐继续阅读

- [Chen et al., Symbolic Discovery of Optimization Algorithms (arXiv:2302.06675)](https://arxiv.org/abs/2302.06675)
- [Lion 官方实现 (Google Research)](https://github.com/google/automl/tree/master/lion)
