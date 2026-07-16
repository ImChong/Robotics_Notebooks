---
type: entity
tags: [google-deepmind, vla, gemini, embodied-ai, product]
title: Gemini Robotics
summary: "Gemini Robotics 是 Google DeepMind 基于 Gemini 多模态栈发布的机器人视觉–语言–动作与具身推理模型族（含 ER / 1.5 等迭代），强调泛化、交互与自然语言指令。"
updated: 2026-07-16
related:
  - ../methods/vla.md
  - ../methods/robotics-transformer-rt-series.md
  - ./perceptron-egocentric.md
sources:
  - ../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md
---

# Gemini Robotics

## 一句话定义

**Gemini Robotics**：面向物理交互的 Gemini 系列机器人模型，通常包含 **VLA 式策略骨干**与强调空间 / 任务推理的 **Embodied Reasoning（ER）** 变体；后续迭代（如公开资料中的 1.5）在长程任务分解与跨本体动作迁移等方向扩展能力叙事。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作多模态基础策略方向 |

## 阅读建议

- 能力边界与数值应以 **官方博客 + 技术报告 PDF** 为准，并与开源通用策略（如 [Octo](../methods/octo-model.md)）区分评测设定。
- 概念上与 [VLA](../methods/vla.md)、[Robotics Transformer](../methods/robotics-transformer-rt-series.md) 同属「多模态大模型接入控制」谱系。
- **自动标注对照：** Macrodata **WGO-Bench** 以 **Gemini 3.5 Flash + Gemini Robotics ER-1.6** 构建机器人子任务分段管线；Perceptron Egocentric（2026-07 博客）在同一基准报告更高 **semantic end-to-end F1** 与更低标注成本——见 [Perceptron Egocentric](./perceptron-egocentric.md) 与 [Auto-labeling Pipelines](../methods/auto-labeling-pipelines.md)。

## 关联页面

- [VLA](../methods/vla.md)
- [Foundation Policy](../concepts/foundation-policy.md)
- [Perceptron Egocentric（WGO 自动标注对照）](./perceptron-egocentric.md)

## 参考来源

- Google DeepMind Blog — Gemini Robotics: https://deepmind.google/blog/gemini-robotics-brings-ai-into-the-physical-world/
- Gemini Robotics 1.5: https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/
- 技术报告 PDF: https://storage.googleapis.com/deepmind-media/gemini-robotics/Gemini-Robotics-1.5-Tech-Report.pdf
- [ted_xiao_embodied_three_eras_primary_refs.md](../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md)
