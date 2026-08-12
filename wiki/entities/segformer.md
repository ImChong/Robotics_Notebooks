---
type: entity
tags:
  - segmentation
  - segformer
  - transformer
  - semantic-segmentation
  - efficient
status: complete
updated: 2026-08-12
summary: "SegFormer 用分层 Transformer 编码器 + 轻量 MLP 解码器做高效语义分割，无需位置编码与复杂解码，精度–效率均衡出色。"
related:
  - ./setr.md
  - ../concepts/image-segmentation-taxonomy.md
  - ../concepts/vision-transformer.md
  - ../entities/transformer-cv-curriculum.md
sources:
  - ../../sources/courses/transformer_cv_applications_syllabus.md
---

# SegFormer

## 一句话定义

**SegFormer** 结合 **分层高效 Transformer 编码器** 与 **极简 MLP 解码器**，在无pe、无重型解码头的情况下达到强语义分割精度与良好推理效率。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SegFormer | SegFormer | 分层 Transformer+MLP 分割 |
| MiT | Mix Transformer | 其分层编码器族 |
| MLP | Multi-Layer Perceptron | 轻量解码融合 |
| mIoU | mean IoU | 指标 |
| ADE20K | ADE20K | 常用基准 |

## 为什么重要

- 课程 4.3.2：相对 SETR 更「能部署」的 Transformer 分割基线。
- 机器人语义图层可优先试 B0–B2 小配置测延迟。

## 核心原理

编码器产出多尺度特征；解码器上采样对齐后拼接，经 MLP 预测类别。重叠 patch 合并与高效自注意力降低复杂度。

```mermaid
flowchart LR
  IMG --> MIT["MiT 分层编码"] --> MLP["MLP 解码器"] --> OUT["语义图"]
```

## 工程实践

| 项 | 建议 |
|----|------|
| 型号 | B0 边缘；B5 冲精度 |
| 框架 | MMSegmentation 官方配置 |
| 迁移 | Cityscapes/自有域 fine-tune |

## 局限与风险

仍是闭集语义；开放词分割需接语言侧或 SEEM/SAM 管线。极端高分辨率需瓦片推理。

## 关联页面

- [SETR](./setr.md)
- [分割任务分类](../concepts/image-segmentation-taxonomy.md)
- [Cityscapes](./dataset-cityscapes.md)
- [Transformer CV 课程策展](../entities/transformer-cv-curriculum.md)

## 参考来源

- [Transformer 视觉应用课程大纲](../../sources/courses/transformer_cv_applications_syllabus.md)

## 推荐继续阅读

- [SegFormer (NeurIPS 2021) arXiv:2105.15203](https://arxiv.org/abs/2105.15203)
