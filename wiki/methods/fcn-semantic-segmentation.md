---
type: method
tags:
  - segmentation
  - computer-vision
  - cnn
  - deep-learning
status: complete
updated: 2026-08-12
summary: "FCN 将分类网全连接改为卷积，实现任意尺寸输入的端到端像素级语义分割，是深度语义分割的起点。"
related:
  - ../concepts/image-segmentation-taxonomy.md
  - ../entities/transformer-cv-curriculum.md
  - ../entities/dataset-ade20k.md
sources:
  - ../../sources/courses/transformer_cv_applications_syllabus.md
---

# FCN 全卷积网络

## 一句话定义

FCN 将分类网全连接改为卷积，实现任意尺寸输入的端到端像素级语义分割，是深度语义分割的起点。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| FCN | Fully Convolutional Network | 全卷积语义分割 |
| mIoU | mean IoU | 语义分割指标 |
| Upsample | Upsampling / Deconvolution | 恢复空间分辨率 |
| Skip | Skip Connection | 融合深浅层特征 |
| SemSeg | Semantic Segmentation | 像素分类任务 |

## 为什么重要

- 课程 4.2 的 CNN 分割主线节点，为理解 SETR/SegFormer 提供对照。
- 机器人侧仍大量使用 U-Net/Mask R-CNN 类结构做缺陷、可抓取与语义图层。

## 主要技术路线

| 路线 | 机制 | 说明 |
|------|------|------|
| FCN-32s/16s/8s | 不同跨层融合步长 | 精度与细节权衡 |
| 全卷积改造 | FC→1×1 Conv | 任意尺寸输入 |
| 转置卷积上采样 | 可学习上采样 | 恢复分辨率 |

## 核心原理

把分类骨干改为全卷积，用上采样（转置卷积等）恢复分辨率，并以跨层融合改进细节。输出与输入同空间尺寸的类得分图。

```mermaid
flowchart LR
  IMG["图像"] --> ENC["编码/骨干"] --> HEAD["分割头"] --> MASK["像素/实例输出"]
```

## 工程实践

| 项 | 建议 |
|----|------|
| 数据 | ADE20K/Cityscapes/自有掩码；注意类别映射 |
| 损失 | 交叉熵 / Dice / 辅助头按论文配置 |
| 部署 | 测全分辨率延迟；机载可降输入尺寸 |

## 局限与风险

CNN 分割受感受野与多尺度模块设计约束；开放集与提示分割需看 SAM/SEEM。标注噪声会直接体现在边界指标上。

## 关联页面

- [分割任务分类](../concepts/image-segmentation-taxonomy.md)
- [SETR](../entities/setr.md)
- [SegFormer](../entities/segformer.md)
- [Transformer CV 课程策展](../entities/transformer-cv-curriculum.md)

## 参考来源

- [Transformer 视觉应用课程大纲](../../sources/courses/transformer_cv_applications_syllabus.md)

## 推荐继续阅读

- [原论文](https://arxiv.org/abs/1411.4038)
