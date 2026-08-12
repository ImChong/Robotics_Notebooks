---
type: entity
tags:
  - cnn
  - lenet
  - image-classification
  - deep-learning
status: complete
updated: 2026-08-12
summary: "LeNet-5 是早期卷积分类网络（约 1998）：卷积+池化+全连接完成手写数字识别，奠定现代 CNN 分层特征提取范式。"
related:
  - ../concepts/convolutional-neural-network.md
  - ./dataset-mnist.md
  - ../entities/transformer-cv-curriculum.md
sources:
  - ../../sources/courses/transformer_cv_applications_syllabus.md
---

# LeNet-5

## 一句话定义

**LeNet-5** 用交替的卷积与下采样层提取局部特征，再经全连接完成分类，是深度学习时代之前即验证「可学习卷积特征」可行的经典小网络。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| LeNet | LeNet-5 | 经典 CNN 分类结构 |
| CNN | Convolutional Neural Network | 卷积网络族 |
| MNIST | MNIST digits | 原论文主战场数据集 |
| FC | Fully Connected | 末端分类全连接层 |
| Pool | Pooling / Subsampling | 空间下采样 |

## 为什么重要

- 课程 2.2.1 的历史起点：后续 AlexNet/VGG/ResNet 都可看作其深度与宽度扩展。
- 教学实现成本低，适合在 [MNIST](./dataset-mnist.md) 上打通训练循环。

## 核心原理

典型数据流：输入 → Conv → Pool → Conv → Pool → FC → FC → 输出类别。卷积提供局部感受野与权值共享；池化降维并增强微小平移鲁棒。

```mermaid
flowchart LR
  X["32x32 输入"] --> C1["Conv"] --> P1["Pool"] --> C2["Conv"] --> P2["Pool"] --> FC["FC 头"] --> Y["类别"]
```

## 工程实践

| 项 | 建议 |
|----|------|
| 复现 | PyTorch/torchvision 或手写两层 conv |
| 数据 | MNIST/Fashion-MNIST；注意输入归一化 |
| 迁移 | 勿直接用于机器人 RGB——容量与分辨率都不够 |

## 局限与风险

容量极小，无法拟合 ImageNet 级任务；仅作结构教学与单元测试网络。

## 关联页面

- [CNN](../concepts/convolutional-neural-network.md)
- [MNIST](./dataset-mnist.md)
- [AlexNet](./alexnet.md)
- [Transformer CV 课程策展](../entities/transformer-cv-curriculum.md)

## 参考来源

- [Transformer 视觉应用课程大纲](../../sources/courses/transformer_cv_applications_syllabus.md)

## 推荐继续阅读

- [LeCun et al. Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
