---
type: entity
tags:
  - segmentation
  - setr
  - transformer
  - semantic-segmentation
  - vit
status: complete
updated: 2026-08-12
summary: "SETR 将语义分割视为序列到序列：用 ViT 编码图像 patch，再经不同解码器上采样为密集类图，证明纯 Transformer 可做分割。"
related:
  - ../concepts/vision-transformer.md
  - ../concepts/image-segmentation-taxonomy.md
  - ./segformer.md
  - ../entities/transformer-cv-curriculum.md
  - ../queries/robot-perception-stack-selection-loop.md
sources:
  - ../../sources/courses/transformer_cv_applications_syllabus.md
---

# SETR（SEgmentation TRansformer）

## 一句话定义

**SETR** 以 **ViT 编码器** 提取全局 patch 表示，再用渐进上采样或多级聚合解码器输出语义分割图，是 Transformer 进入密集预测的早期代表。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SETR | SEgmentation TRansformer | ViT 语义分割方法 |
| ViT | Vision Transformer | 编码器骨干 |
| mIoU | mean IoU | 评测指标 |
| MLA | Multi-Level Aggregation | 多级特征聚合解码变体 |
| ADE20K | ADE20K | 课程作业主数据 |

## 为什么重要

- 课程 4.3.1 与作业 4（ADE20K 训练）指定模型。
- 说明「分类 ViT」可通过解码器扩展到像素任务。

## 核心原理

图像 → patch 嵌入 → Transformer 编码 → 选一层或多层特征 → 上采样解码为 HxW 类图。全局自注意力提供大感受野，无需深层 CNN 堆叠。

```mermaid
flowchart LR
  IMG --> VIT["ViT Encoder"] --> DEC["上采样解码器"] --> SEG["语义图"]
```

## 工程实践

| 项 | 建议 |
|----|------|
| 预训练 | 使用 ImageNet 预训练 ViT 权重 |
| 数据 | ADE20K；注意 crop 尺寸与滑动推理 |
| 显存 | 高分辨率需梯度检查点或更小 ViT |

## 局限与风险

计算重、细节边界可能弱于强多尺度 CNN；后续 [SegFormer](./segformer.md) 更高效。作业复现需对齐官方配置。

## 关联页面

- [SegFormer](./segformer.md)
- [ADE20K](./dataset-ade20k.md)
- [ViT](../concepts/vision-transformer.md)
- [Transformer CV 课程策展](../entities/transformer-cv-curriculum.md)
- [机器人视觉感知栈选型闭环知识链](../queries/robot-perception-stack-selection-loop.md) — 纯 ViT 分割路线，②层选型中与 CNN 分割器对照

## 参考来源

- [Transformer 视觉应用课程大纲](../../sources/courses/transformer_cv_applications_syllabus.md)

## 推荐继续阅读

- [SETR (CVPR 2021) arXiv:2012.15840](https://arxiv.org/abs/2012.15840)
