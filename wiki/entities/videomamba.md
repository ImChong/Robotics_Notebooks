---
type: entity
tags:
  - mamba
  - ssm
  - computer-vision
  - deep-learning
  - backbone
status: complete
updated: 2026-08-12
summary: "VideoMamba 用状态空间模型做高效视频理解，在长视频上相对 Transformer 降低注意力平方复杂度。"
related:
  - ../concepts/state-space-model-ssm.md
  - ../comparisons/rnn-cnn-transformer-mamba.md
  - ../entities/transformer-cv-curriculum.md
sources:
  - ../../sources/courses/transformer_cv_applications_syllabus.md
---

# VideoMamba

## 一句话定义

VideoMamba 用状态空间模型做高效视频理解，在长视频上相对 Transformer 降低注意力平方复杂度。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VideoMamba | VideoMamba | 视频理解 Mamba |
| SSM | State Space Model | 时序建模 |
| CLIP | CLIP 特征 | 常用视觉前端 |
| K400 | Kinetics-400 | 动作识别基准 |
| FLOPs | Floating Point Ops | 算力指标 |

## 为什么重要

- 课程第 7 章 Mamba 视觉谱系节点；对照 Transformer 骨干的效率–精度权衡。
- 为机器人侧长序列感知（视频、扫描图、触觉历史）提供可选结构。

## 核心原理

以 SSM/Mamba 块替代或混合自注意力：将 2D/视频特征展成扫描序列，经选择性状态更新后再映射回空间特征，接分类或密集预测头。

```mermaid
flowchart LR
  X["视觉特征"] --> SCAN["扫描顺序"] --> SSM["Mamba/SSM 块"] --> Y["任务头"]
```

## 工程实践

| 项 | 建议 |
|----|------|
| 复现 | 跟随官方仓库与扫描核版本 |
| 对照 | 同 FLOPs 对比 DeiT/Swín/SegFormer |
| 作业 | 医学线关注 U-Mamba/SegMamba 与 BraTS 协议 |

## 局限与风险

生态与预训练权重少于 ViT；自定义核影响移植。任务论文数字需在统一数据协议下解读。

## 关联页面

- [SSM](../concepts/state-space-model-ssm.md)
- [架构对比](../comparisons/rnn-cnn-transformer-mamba.md)
- [Transformer CV 课程策展](../entities/transformer-cv-curriculum.md)

## 参考来源

- [Transformer 视觉应用课程大纲](../../sources/courses/transformer_cv_applications_syllabus.md)

## 推荐继续阅读

- [相关论文](https://arxiv.org/abs/2403.06977)
