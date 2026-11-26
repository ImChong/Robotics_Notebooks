---
type: method
tags: [3d, vision, reconstruction]
status: draft
updated: 2026-11-26
summary: "从图像、视频或深度观测恢复可用于测量、渲染或仿真的三维场景表示。"
sources:
  - ../../sources/blogs/worldlabs_spark_2_0_streaming_3dgs.md
related:
  - ../concepts/embodied-perception-six-spatial-representations.md
---

# 视觉三维重建流程

## 一句话定义

从图像、视频或深度观测恢复可用于测量、渲染或仿真的三维场景表示。

## 英文缩写速查

| 缩写 | 英文全称 | 说明 |
|---|---|---|
| AI | Artificial Intelligence | 人工智能 |

## 关联概念

[相关概念](../concepts/embodied-perception-six-spatial-representations.md) 介绍该方法的背景。

## 主要技术路线

1. **采集与几何**：检查视角覆盖、相机位姿、标定与尺度；单目和多视角输入的约束不同。
2. **场景表示**：按用途选择网格、点云、隐式场或高斯表示；渲染质量与几何精度不能互相替代。
3. **复核和导出**：检查遮挡、动态物体、尺度与坐标一致性，再验证下游渲染、导航或仿真需求。

[视觉表征学习](visual-representation-learning.md) 可提供输入特征，重建仍需单独验证几何。

## 局限与后续核查

本页用于跨项目导航；具体数据、实现、评测和授权应以原始资料为准。

## 参考来源

- [资料归档](../../sources/blogs/worldlabs_spark_2_0_streaming_3dgs.md)
