---
type: method
tags: [vision, learning, representation]
status: draft
updated: 2026-11-26
summary: "从图像或视频中学习可用于识别、定位或下游任务的特征表示。"
sources:
  - ../../sources/courses/vision_transformer_primary_sources.md
related:
  - ../concepts/vision-backbones.md
---

# 视觉表征学习

## 一句话定义

从图像或视频中学习可用于识别、定位或下游任务的特征表示。

## 英文缩写速查

| 缩写 | 英文全称 | 说明 |
|---|---|---|
| AI | Artificial Intelligence | 人工智能 |

## 关联概念

[相关概念](../concepts/vision-backbones.md) 介绍该方法的背景。

## 主要技术路线

1. **输入与骨干**：明确图像或视频输入、分辨率与编码器；CNN、视觉 Transformer 等结构具有不同的计算和数据需求。
2. **学习信号**：可使用标注监督、自监督预训练或跨模态对齐；不同目标决定表示保留的信息。
3. **下游验证**：在分类、检测、分割或机器人任务上报告数据划分与指标；特征质量不能仅由预训练损失断定。

[目标检测](object-detection.md) 与 [语义分割](fcn-semantic-segmentation.md) 是不同的下游视觉任务。

## 局限与后续核查

本页用于跨项目导航；具体数据、实现、评测和授权应以原始资料为准。

## 参考来源

- [资料归档](../../sources/courses/vision_transformer_primary_sources.md)
