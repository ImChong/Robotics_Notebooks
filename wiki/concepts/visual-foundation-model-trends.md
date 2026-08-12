---
type: concept
tags:
  - foundation-model
  - sam
  - vlm
  - open-vocabulary
  - computer-vision
  - trends
status: complete
updated: 2026-08-12
summary: "视觉基础模型五大趋势：单模态→多模态、小模型→基础模型、专用→通用、闭集→开集、纯感知→感知+推理；指导机器人感知栈演进判断。"
related:
  - ./generative-vision-pretraining.md
  - ../entities/paper-segment-anything.md
  - ../entities/seem.md
  - ../overview/multimodal-llm-development.md
  - ../entities/transformer-cv-curriculum.md
sources:
  - ../../sources/courses/transformer_cv_applications_syllabus.md
---

# 视觉基础模型发展趋势

## 一句话定义

当代视觉基础模型正沿 **多模态化、规模化、通用接口化、开放词汇化与推理化** 五条趋势演进，从单一任务小模型走向可提示、可组合的通用感知层。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VFM | Visual Foundation Model | 视觉基础模型 |
| OV | Open-Vocabulary | 开放词汇识别/分割 |
| SAM | Segment Anything | 可提示分割基座 |
| VLM | Vision-Language Model | 视觉–语言基础模型 |
| MLLM | Multimodal LLM | 多模态大模型 |

## 为什么重要

- 课程 8.2 的归纳框架，帮助判断新论文落在哪条轴上。
- 机器人选型：专用检测器 vs 通用 SAM/VLM 核验 vs 端到端 VLA。

## 核心原理（五条轴）

1. **单模态 → 多模态**：纯视觉 → CLIP/LLaVA 等。  
2. **小模型 → 基础模型**：ImageNet 专训 → 大规模预训练可迁移。  
3. **专用 → 通用**：检测/分割分家 → 统一提示接口（SAM/SEEM）。  
4. **闭集 → 开集**：固定 K 类 → 文本开放词。  
5. **感知 → 感知+推理**：输出框/掩码 → 带语言推理的接地（LISA 等）。

```mermaid
flowchart LR
  A["专用闭集小模型"] --> B["可提示基础模型"]
  B --> C["多模态推理与具身"]
```

## 工程实践

| 项 | 建议 |
|----|------|
| 落地组合 | YOLO/DETR 做闭环速度；SAM/VLM 做开放集核验 |
| 数据 | 基础模型仍需域数据校准 |
| 评估 | 除 mAP/mIoU 外测指令成功率 |

## 局限与风险

趋势叙事不等于取消专用模型；实时安全关键仍常要闭集 specialized 检测器。许可、幻觉与成本是部署硬约束。

## 关联页面

- [生成式视觉预训练](./generative-vision-pretraining.md)
- [SAM](../entities/paper-segment-anything.md)
- [SEEM](../entities/seem.md)
- [多模态 LLM 路线](../overview/multimodal-llm-development.md)
- [Transformer CV 课程策展](../entities/transformer-cv-curriculum.md)

## 参考来源

- [Transformer 视觉应用课程大纲](../../sources/courses/transformer_cv_applications_syllabus.md)

## 推荐继续阅读

- [Segment Anything](https://arxiv.org/abs/2304.02643)
