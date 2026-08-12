---
type: entity
tags:
  - multimodal
  - vlm
  - vision-language
  - deep-learning
  - foundation-model
status: complete
updated: 2026-08-12
summary: "LISA 让多模态 LLM 通过特殊分割 token 调用分割解码器，实现推理式分割（reasoning segmentation）：由复杂语言描述直接得到掩码。"
related:
  - ../concepts/multimodality-basics.md
  - ../overview/multimodal-llm-development.md
  - ../entities/transformer-cv-curriculum.md
sources:
  - ../../sources/courses/transformer_cv_applications_syllabus.md
---

# LISA

## 一句话定义

LISA 让多模态 LLM 通过特殊分割 token 调用分割解码器，实现推理式分割（reasoning segmentation）：由复杂语言描述直接得到掩码。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| LISA | Large Language Instructed Segmentation Assistant | 推理分割助理 |
| SEG | SEG token | 触发分割的特殊词元 |
| VLM | Vision-Language Model | 多模态底座 |
| SAM | Segment Anything | 可接分割解码 |
| RES | Reasoning Segmentation | 推理分割任务 |

## 为什么重要

- 课程第 5–6 章多模态主线节点；与机器人 VLM/VLA 选型直接相关。
- 理解其输入输出接口，才能正确接到检测、分割或策略模块。

## 核心原理

对话式 LLM 在需要接地时输出 SEG token，其隐状态经解码器生成掩码，统一推理与分割。

```mermaid
flowchart LR
  IMG["视觉输入"] --> ENC["视觉编码/桥接"] --> LLM["语言侧/头"] --> OUT["文本/掩码/分数"]
  TXT["文本/指令"] --> LLM
```

## 工程实践

| 项 | 建议 |
|----|------|
| 权重 | 优先官方或 Hugging Face 发布 |
| 微调 | 指令数据质量优先；可用 LoRA |
| 机器人 | 明确延迟预算；重模型可云端核验 |

## 局限与风险

幻觉、错误 grounding、许可与安全过滤必须单独评估；开源状态以项目页为准，部署前核查权重协议。

## 关联页面

- [多模态基础](../concepts/multimodality-basics.md)
- [多模态 LLM 路线](../overview/multimodal-llm-development.md)
- [BLIP-2](./paper-blip2.md)
- [Transformer CV 课程策展](../entities/transformer-cv-curriculum.md)

## 参考来源

- [Transformer 视觉应用课程大纲](../../sources/courses/transformer_cv_applications_syllabus.md)

## 推荐继续阅读

- [论文 / 项目](https://arxiv.org/abs/2308.00692)
