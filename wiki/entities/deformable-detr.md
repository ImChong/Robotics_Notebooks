---
type: entity
tags:
  - object-detection
  - deformable-detr
  - transformer
  - deformable-attention
  - computer-vision
status: complete
updated: 2026-08-12
summary: "Deformable DETR 用多尺度可变形注意力加速收敛并提升小目标，成为 DETR 族实用化的关键改进。"
related:
  - ./detr.md
  - ../methods/object-detection.md
  - ./rf-detr.md
  - ../entities/transformer-cv-curriculum.md
sources:
  - ../../sources/courses/transformer_cv_applications_syllabus.md
---

# Deformable DETR

## 一句话定义

**Deformable DETR** 将 DETR 中的密集注意力替换为 **多尺度可变形注意力**：每个 query 只采样少量关键采样点，显著加快收敛并改善小目标检测。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| D-DETR | Deformable DETR | 可变形注意力版 DETR |
| MSDA | Multi-Scale Deformable Attention | 多尺度稀疏采样注意力 |
| Query | Object Query | 检测槽位 |
| FPN | Feature Pyramid（多尺度特征） | 多分辨率输入特征 |
| AP | Average Precision | 检测精度 |

## 为什么重要

- 课程 3.3.2：解决原版 DETR 收敛慢/小目标弱的工程关键。
- 许多实时 DETR 变体的注意力设计源头之一。

## 核心原理

在特征图上学习采样偏移，只聚合少数点的值；结合多尺度特征图，使 decoder/encoder 复杂度与精度更可用。

```mermaid
flowchart LR
  F["多尺度特征"] --> MSDA["可变形注意力采样"]
  Q["Queries"] --> MSDA --> DEC["Decoder 输出框"]
```

## 工程实践

| 项 | 建议 |
|----|------|
| 训练 | 比 DETR 更短的 epoch 即可接近可用精度 |
| 代码 | 官方实现 / MMDetection `deformable-detr` |
| 迁移 | 无人机/VisDrone 等小目标场景优先于原版 DETR |

## 局限与风险

实现与算子依赖比标准 MHA 复杂；部署需确认自定义 CUDA/ONNX 支持。

## 关联页面

- [DETR](./detr.md)
- [RF-DETR](./rf-detr.md)
- [检测指标](../concepts/object-detection-metrics.md)
- [Transformer CV 课程策展](../entities/transformer-cv-curriculum.md)

## 参考来源

- [Transformer 视觉应用课程大纲](../../sources/courses/transformer_cv_applications_syllabus.md)

## 推荐继续阅读

- [Deformable DETR (ICLR 2021) arXiv:2010.04159](https://arxiv.org/abs/2010.04159)
