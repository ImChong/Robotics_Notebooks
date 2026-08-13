---
type: concept
tags:
  - cnn
  - convolution
  - computer-vision
  - deep-learning
  - backbone
  - perception
status: complete
updated: 2026-08-12
summary: "卷积神经网络用局部卷积核与权值共享提取层次化视觉特征，是检测/分割与机器人机载感知长期默认骨干，也是理解 ViT 替代路径的对照基线。"
related:
  - ./vision-backbones.md
  - ./vision-transformer.md
  - ../comparisons/cnn-vs-vit-backbones.md
  - ./deep-learning-foundations.md
  - ../entities/transformer-cv-curriculum.md
  - ../queries/robot-perception-stack-selection-loop.md
sources:
  - ../../sources/courses/transformer_cv_applications_syllabus.md
---

# 卷积神经网络（CNN）

## 一句话定义

**CNN** 用可学习的 **局部卷积核** 在空间上滑窗提取特征，借助 **权值共享** 与 **层次化感受野** 建模图像的平移局部结构，是经典视觉与机器人感知骨干的默认族。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CNN | Convolutional Neural Network | 卷积视觉骨干网络族 |
| Conv | Convolution | 局部核滑窗线性滤波+激活 |
| RF | Receptive Field | 神经元可见的输入区域大小 |
| BN | Batch Normalization | 稳定深层卷积训练的归一化 |
| FPN | Feature Pyramid Network | 多尺度特征金字塔，检测常用 |

## 为什么重要

- **归纳偏置强**：局部性与平移等变在中小数据上样本效率高，机器人自采数据场景仍常见 ResNet/CSP 骨干。
- **工程成熟**：量化、TensorRT、边缘部署与 [目标检测](../methods/object-detection.md) 头（YOLO/FPN）工具链完备。
- **对照 ViT**：理解卷积原理是读懂 [CNN vs ViT](../comparisons/cnn-vs-vit-backbones.md) 与课程第 1 章的前提。

## 核心原理

### 卷积运算直觉

对输入特征图 $X$，核 $W$ 在位置 $(i,j)$ 输出为邻域加权和再加偏置并经非线性（ReLU 等）。堆叠多层后感受野近似随深度线性扩大；池化/步幅控制分辨率。

### 典型堆叠

`Conv → BN → ReLU →（可选 Pool）` 重复；现代骨干加入残差（[ResNet](../entities/paper-resnet-deep-residual-learning.md)）、深度可分离卷积或 CSP 等。

```mermaid
flowchart LR
  IMG["图像"] --> C1["浅层 Conv<br/>边缘/纹理"]
  C1 --> C2["中层 Conv<br/>部件"]
  C2 --> C3["深层 Conv<br/>语义"]
  C3 --> HEAD["分类/检测/分割头"]
```

## 工程实践

| 项 | 建议 |
|----|------|
| 机器人骨干 | ResNet-18/50、YOLO 系 CSP；优先测延迟而非只看 ImageNet top-1 |
| 输入分辨率 | 检测常用 640；机载可降到 416/320 换 FPS |
| 预训练 | ImageNet 初始化再微调域数据；域差大时考虑自监督骨干 |
| 调试 | 看中间特征图是否「糊掉」；BN 统计在小 batch 下改 GN/SyncBN |

## 局限与风险

- **远距离依赖**需很深堆叠，不如自注意力直接；大图全局关系弱。
- **固定核尺寸**对尺度变化依赖多尺度/金字塔设计。
- 勿把「CNN 过时」当作选型结论——边缘实时闭环里 CNN 仍常胜出。

## 关联页面

- [Vision Backbones](./vision-backbones.md)
- [CNN vs ViT Backbones](../comparisons/cnn-vs-vit-backbones.md)
- [ResNet](../entities/paper-resnet-deep-residual-learning.md)
- [Transformer CV 课程策展](../entities/transformer-cv-curriculum.md)
- [机器人视觉感知栈选型闭环知识链](../queries/robot-perception-stack-selection-loop.md) — CNN 骨干是②层 2D 检测/分割选型的算力–精度基线

## 参考来源

- [Transformer 视觉应用课程大纲](../../sources/courses/transformer_cv_applications_syllabus.md)

## 推荐继续阅读

- [Goodfellow et al. Deep Learning Book — Conv Nets](https://www.deeplearningbook.org/contents/convnets.html)
- [CS231n Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/)
