---
type: method
tags:
  - segmentation
  - computer-vision
  - cnn
  - deep-learning
status: complete
updated: 2026-08-12
summary: "PSPNet 通过金字塔池化模块聚合多尺度全局上下文，显著提升场景解析类语义分割精度。"
related:
  - ../concepts/image-segmentation-taxonomy.md
  - ../entities/transformer-cv-curriculum.md
  - ../entities/dataset-ade20k.md
  - ../queries/robot-perception-stack-selection-loop.md
sources:
  - ../../sources/courses/transformer_cv_applications_syllabus.md
---

# PSPNet

## 一句话定义

PSPNet 通过金字塔池化模块聚合多尺度全局上下文，显著提升场景解析类语义分割精度。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PSP | Pyramid Scene Parsing | 金字塔场景解析 |
| PPM | Pyramid Pooling Module | 多尺度全局池化融合 |
| mIoU | mean IoU | 分割指标 |
| Aux | Auxiliary Loss | 深层辅助监督 |
| ADE | ADE20K | 常用场景解析基准 |

## 为什么重要

- 课程 4.2 的 CNN 分割主线节点，为理解 SETR/SegFormer 提供对照。
- 机器人侧仍大量使用 U-Net/Mask R-CNN 类结构做缺陷、可抓取与语义图层。

## 主要技术路线

| 路线 | 机制 | 说明 |
|------|------|------|
| PPM | 多尺度全局池化融合 | 场景上下文 |
| 辅助损失 | 深层旁路监督 | 稳定训练 |
| 骨干替换 | ResNet 等 | 精度–算力档位 |

## 核心原理

在骨干特征上做不同尺度的全局平均池化，上采样后拼接，再卷积预测，显式注入全局场景先验。

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
- [机器人视觉感知栈选型闭环知识链](../queries/robot-perception-stack-selection-loop.md) — 金字塔池化聚合全局上下文，②层场景解析类分割选型

## 参考来源

- [Transformer 视觉应用课程大纲](../../sources/courses/transformer_cv_applications_syllabus.md)

## 推荐继续阅读

- [原论文](https://arxiv.org/abs/1612.01105)
