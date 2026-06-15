---
type: concept
summary: "反向传播是在分层可微计算图上高效应用链式法则、逐层回传损失梯度以训练神经网络权重的核心算法，是现代机器人端到端感知与策略学习的优化底座。"
description: 反向传播（Backpropagation）算法：链式法则、前向/反向两趟、与自动微分及机器人策略训练的关系。
updated: 2026-06-15
related:
  - ./deep-learning-foundations.md
  - ./transformer.md
  - ../entities/pytorch.md
  - ../methods/reinforcement-learning.md
sources:
  - ../../sources/papers/rumelhart_backprop_learning_representations_nature_1986.md
  - ../../sources/books/udl_book.md
---

# 反向传播算法 (Backpropagation)

> **一句话定义**：在分层 **可微** 计算图上，用 **链式法则（chain rule）** 从损失函数出发 **自顶向下** 高效计算每个参数梯度的算法；一次前向求值 + 一次反向传梯度，即可端到端训练含隐藏层的深度网络。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BP | Backpropagation | 反向传播，沿计算图回传 $\partial L / \partial \cdot$ |
| SGD | Stochastic Gradient Descent | 用 mini-batch 梯度更新参数的优化器，常与 BP 联用但概念不同 |
| AD | Automatic Differentiation | 自动微分；现代框架对 BP 的工程实现 |
| MLP | Multilayer Perceptron | 含隐藏层的全连接网络，BP 的经典应用场景 |
| RL | Reinforcement Learning | 策略梯度等方法仍依赖 BP 计算网络参数梯度 |

## 为什么重要

- **突破单层感知机局限**：没有 BP，隐藏层无法获得有效学习信号；[Rumelhart et al. (1986)](../../sources/papers/rumelhart_backprop_learning_representations_nature_1986.md) 使多层网络能自动学到任务相关 **中间表征**，奠定今日深度学习的训练范式。
- **机器人端到端学习的通用底座**：视觉编码器、策略 MLP/Transformer、扩散动作头、执行器网络等，凡是用梯度下降训练的模块，底层都依赖 BP（或其在计算图上的等价形式）。
- **与自动微分一体两面**：在 [PyTorch](../entities/pytorch.md) 等框架中，`loss.backward()` 触发的正是反向模式自动微分；理解 BP 有助于诊断 **梯度消失/爆炸**、检查计算图断点、解释 Sim2Real 中哪些模块可微、哪些必须黑盒。

## 核心机制

### 1. 计算图与前向传播

将网络写成复合函数 $L = \ell(f_L(\cdots f_1(\mathbf{x}; \theta_1) \cdots; \theta_L))$。**前向传播**按拓扑序逐层计算各中间激活与最终损失 $L$。

### 2. 链式法则与反向传播

对任意参数 $\theta_i$，有

$$
\frac{\partial L}{\partial \theta_i} = \frac{\partial L}{\partial \mathbf{z}_i} \cdot \frac{\partial \mathbf{z}_i}{\partial \theta_i}
$$

其中 $\mathbf{z}_i$ 为第 $i$ 层输出。**反向传播**从输出层开始，把 $\partial L / \partial \mathbf{z}$ 逐层 **往回乘局部雅可比**，复用已算出的上游梯度，避免对每个参数重复做指数级的前向展开——这是相对「数值差分」或朴素链式法则的 **$O(1)$ 次前向 + $O(1)$ 次反向** 效率来源。

### 3. 与优化器的关系

BP **只负责算梯度**；**SGD / Adam** 等优化器用 $\theta \leftarrow \theta - \eta \nabla_\theta L$ 更新参数。二者常连用，但不应混为一谈。

### 4. 表征学习与隐藏层

[Rumelhart et al.](../../sources/papers/rumelhart_backprop_learning_representations_nature_1986.md) 的核心洞见：误差信号传到 **隐藏单元** 后，它们会编码输入中的 **规律性特征**（如 XOR 中的非线性可分结构），而非仅做线性投影——这使「深度」在优化意义上可行。

## 常见误区或局限

| 误区 | 澄清 |
|------|------|
| 「BP = 深度学习」 | BP 是 **训练算法**；架构（CNN、Transformer）、数据与正则化同样决定效果 |
| 「BP 只能用于监督学习」 | **策略梯度、扩散模型、模仿学习** 等仍对可微部分做 BP；不可微环节（仿真步进、硬量化）需 surrogate 或停梯度 |
| 「层数越深越好」 | 深层 sigmoid/tanh 时代易 **梯度消失**；残差连接、ReLU、归一化与初始化（见 [深度学习基础](./deep-learning-foundations.md)）缓解优化困难 |
| 「现代框架不用 BP」 | Autograd **就是** 反向模式 AD 对 BP 的实现；`detach()` / `torch.no_grad()` 是在计算图上 **切断** 反向路径 |

## 与机器人技术的联系

- **感知栈**：[视觉骨干](./vision-backbones.md)、[目标检测](../methods/object-detection.md) 的 ImageNet 预训练权重，皆由 BP + SGD 在分类损失上习得表征。
- **策略与模仿**：[强化学习](../methods/reinforcement-learning.md) 中策略网络 $\pi_\theta(a|s)$ 的 $\nabla_\theta J$ 经 **策略梯度定理** 仍化为对 $\theta$ 的反向传播；[Transformer](./transformer.md) / ACT 动作块预测同理。
- **何时刻意避免完整 BP**：[LWD](../methods/lwd.md) 等对 flow-matching **多步生成头** 用 **Adjoint Matching** 等局部回归，就是为绕开对整条生成轨迹做不稳定的全链反向传播。

## 关联页面

- [深度学习基础](./deep-learning-foundations.md)
- [Transformer](./transformer.md)
- [视觉骨干](./vision-backbones.md)
- [强化学习](../methods/reinforcement-learning.md)
- [PyTorch](../entities/pytorch.md)
- [LWD（QAM 与反向传播取舍）](../methods/lwd.md)

## 参考来源

- [Learning representations by back-propagating errors（Nature, 1986）](../../sources/papers/rumelhart_backprop_learning_representations_nature_1986.md)
- [Understanding Deep Learning (Prince, 2023) — 第 7 章 Gradients and Initialization](../../sources/books/udl_book.md)
- Rumelhart, D. E., Hinton, G. E. & Williams, R. J. (1986). *Learning representations by back-propagating errors*. Nature 323, 533–536. <https://doi.org/10.1038/323533a0>

## 推荐继续阅读

- Goodfellow, I., Bengio, Y. & Courville, A. — [Deep Learning Book, Ch. 6: Deep Feedforward Networks](https://www.deeplearningbook.org/contents/mlp.html)
- Andrej Karpathy — [micrograd](https://github.com/karpathy/micrograd)（极简 autograd 实现，适合手推 BP）
- [Andrej Karpathy](../entities/andrej-karpathy.md) — *Neural Networks: Zero to Hero* 系列
