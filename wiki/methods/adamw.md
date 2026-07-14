---
type: method
tags: [deep-learning, optimization, adamw, weight-decay, training, transformer]
status: complete
updated: 2026-06-27
summary: "AdamW 将权重衰减从 Adam 的梯度自适应更新中解耦，按参数幅度独立收缩权重，是 Transformer 预训练与 VLA 大模型微调的事实标准优化器。"
related:
  - ./adam.md
  - ./lion.md
  - ./muon.md
  - ../concepts/transformer.md
  - ../methods/vla.md
  - ../comparisons/deep-learning-optimizers.md
  - ../entities/pytorch.md
sources:
  - ../../sources/papers/deep_learning_optimizers.md
  - ../../sources/books/udl_book.md
  - ../../sources/repos/pytorch-official.md
---

# AdamW（Adam with Decoupled Weight Decay）

**AdamW**：在 [Adam](./adam.md) 更新之后（或之前，实现细节各异），**独立**施加权重衰减 $\theta \leftarrow \theta - \eta \lambda \theta$，而非将 L2 惩罚项并入梯度 $g_t$。Loshchilov & Hutter (2019) 证明后者在自适应优化器下 **正则化语义错误**，AdamW 修正后成为 BERT、GPT、ViT 及 [VLA](./vla.md) 预训练的标准配方。

## 一句话定义

> 自适应步长和权重衰减分开算——别让 L2 正则被 Adam 的分母「稀释」，该缩的权重每一轮都按比例缩。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| AdamW | Adam with decoupled Weight decay | Loshchilov & Hutter 2019 |
| Adam | Adaptive Moment Estimation | 解耦前的基线 |
| WD | Weight Decay | 系数 $\lambda$，典型 0.01~0.1 |
| LR | Learning Rate | 常与 warmup + cosine 联用 |
| VLA | Vision-Language-Action | 大模型微调常用 AdamW |

## 主要技术路线

### 1. 解耦更新的含义

**错误做法（Adam + L2）**：令 $\tilde{g}_t = g_t + \lambda \theta_t$，再喂入 Adam；自适应分母 $\sqrt{\hat{v}_t}$ 会 **按参数尺度缩放正则强度**。

**AdamW 做法**：

$$
\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)
$$

或等价地：先 Adam 步，再 $\theta \leftarrow (1 - \eta\lambda)\theta$。正则强度与自适应项 **解耦**。

### 2. 为何对大模型关键

- Transformer 参数多、嵌入与注意力尺度异构；正确 weight decay 抑制过拟合、稳定预训练。
- 与 **学习率 warmup**、**cosine decay**、**梯度裁剪** 组成现代预训练「标准套件」。
- [Lion](./lion.md) 等新优化器常以 AdamW 为对照基线。

### 3. 典型超参（起点，非万能）

| 超参 | 常见范围 | 备注 |
|------|---------|------|
| $\eta$ | 1e-4 ~ 3e-4 | 随 batch 与模型规模缩放 |
| $\lambda$ | 0.01 ~ 0.1 | 与 $\eta$ 联合调节有效收缩 |
| $\beta_1, \beta_2$ | 0.9, 0.999 | 同 Adam |

## 优势与局限

| 优势 | 局限 |
|------|------|
| 大模型预训练/微调事实标准 | 仍非所有视觉任务最优（部分仍偏好 SGD） |
| 正则语义正确、复现性好 | 需配合 LR schedule，调参维度不低 |
| PyTorch `AdamW` 为默认推荐 | 内存与 Adam 相同（二阶矩） |

## 在机器人中的典型应用

- **VLA / 扩散策略预训练**：π0、OpenVLA 类模型微调默认 AdamW。
- **Transformer 动作头**：与 [Transformer](../concepts/transformer.md) 编码器联合训练。
- **从仿真 RL 到大模型**：仿真中小网络可用 Adam；部署前视觉-语言 backbone 微调几乎总是 AdamW。

## 关联页面

- [Adam](./adam.md)
- [Lion](./lion.md)
- [Transformer](../concepts/transformer.md)
- [VLA](./vla.md)
- [Deep Learning Optimizers 对比](../comparisons/deep-learning-optimizers.md)
- [PyTorch](../entities/pytorch.md)

## 参考来源

- [Deep Learning Optimizers 论文摘录](../../sources/papers/deep_learning_optimizers.md) — Loshchilov & Hutter (2019)
- [Understanding Deep Learning (Prince, 2023)](../../sources/books/udl_book.md)
- [PyTorch 官方站点与文档索引](../../sources/repos/pytorch-official.md)

## 推荐继续阅读

- [Loshchilov & Hutter, Decoupled Weight Decay Regularization (ICLR 2019)](https://arxiv.org/abs/1711.05101)
- [PyTorch torch.optim.AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
