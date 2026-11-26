---
type: method
tags: [video, generation, evaluation]
status: draft
updated: 2026-11-26
summary: "利用生成模型或编辑工具产生、变换和检查时序视觉内容。"
sources:
  - ../../sources/blogs/qwen_image_2_1_blog.md
related:
  - ../concepts/generative-vision-pretraining.md
---

# 视频生成与编辑流程

## 一句话定义

利用生成模型或编辑工具产生、变换和检查时序视觉内容。

## 英文缩写速查

| 缩写 | 英文全称 | 说明 |
|---|---|---|
| AI | Artificial Intelligence | 人工智能 |

## 关联概念

[相关概念](../concepts/generative-vision-pretraining.md) 介绍该方法的背景。

## 主要技术路线

1. **定义输入与条件**：区分文本、图像、参考视频、运动或声音条件，并记录时长及分辨率。
2. **生成与编辑**：配置模型、采样与约束；视频续写、镜头变换和动作控制具有不同接口。
3. **检查输出**：核对时间一致性、对象身份、语义和物理合理性；视觉连贯并不保证动力学真实。

[生成式世界模型](generative-world-models.md) 对照动作条件预测；一般视频生成不必具备可交互世界模型。

## 局限与后续核查

本页用于跨项目导航；具体数据、实现、评测和授权应以原始资料为准。

## 参考来源

- [资料归档](../../sources/blogs/qwen_image_2_1_blog.md)
