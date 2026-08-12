---
type: entity
tags:
  - vit
  - cvt
  - transformer
  - convolution
  - image-classification
status: complete
updated: 2026-08-12
summary: "CvT（Convolutional vision Transformer）把卷积引入 Transformer 的 token 嵌入与投影，兼得 CNN 局部偏置与注意力全局建模。"
related:
  - ../concepts/vision-transformer.md
  - ./tnt.md
  - ../comparisons/cnn-vs-vit-backbones.md
  - ../entities/transformer-cv-curriculum.md
sources:
  - ../../sources/courses/transformer_cv_applications_syllabus.md
---

# CvT（Convolutional Vision Transformer）

## 一句话定义

**CvT** 在视觉 Transformer 中用 **卷积 token 嵌入** 与 **卷积投影 Q/K/V**，把 CNN 的局部/下采样归纳偏置注入注意力骨干。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CvT | Convolutional vision Transformer | 卷积增强的 ViT 变体 |
| CTE | Convolutional Token Embedding | 卷积生成并下采样 token |
| CP | Convolutional Projection | 用卷积实现 QKV 投影 |
| ViT | Vision Transformer | 对照纯注意力骨干 |
| Biases | Inductive Biases | 局部性与步幅下采样 |

## 为什么重要

- 课程 2.3.3：CNN–Transformer 混合设计的代表，呼应第 1 章对比。
- 对中等数据规模往往比朴素 ViT 更稳，启发后续 MobileViT/卷积相对注意力等。

## 核心原理

分层阶段：每阶段卷积嵌入降低分辨率、增加通道；块内用深度可分离卷积生成 Q/K/V，再做多头注意力与 MLP。

```mermaid
flowchart LR
  IMG --> E1["Conv Token Embed"] --> T1["Transformer 阶段1"]
  T1 --> E2["Conv 下采样嵌入"] --> T2["阶段2/3"] --> HEAD["分类"]
```

## 工程实践

| 项 | 建议 |
|----|------|
| 预训练 | 关注官方 ImageNet 配置与位置编码处理 |
| 对照实验 | 同算力下对比 ViT-S / ResNet-50 |
| 迁移检测 | 分层特征比单尺度 ViT 更易接 FPN |

## 局限与风险

「有卷积」不等于一定更快；实现与算子融合影响大。机器人部署仍需实测 TensorRT 延迟。

## 关联页面

- [ViT](../concepts/vision-transformer.md)
- [TNT](./tnt.md)
- [CNN vs ViT](../comparisons/cnn-vs-vit-backbones.md)
- [Transformer CV 课程策展](../entities/transformer-cv-curriculum.md)

## 参考来源

- [Transformer 视觉应用课程大纲](../../sources/courses/transformer_cv_applications_syllabus.md)

## 推荐继续阅读

- [CvT (ICCV 2021) arXiv:2103.15808](https://arxiv.org/abs/2103.15808)
