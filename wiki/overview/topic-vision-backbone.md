---
type: overview
tags: [topic, topic-vision-backbone, cnn, vit, perception, detection]
status: complete
updated: 2026-07-05
summary: "视觉感知骨干专题汇总：CNN/ViT 骨干、检测/分割头与策略输入的衔接，覆盖 ResNet/YOLO 选型与生成式视觉预训练对机器人表征的影响。"
---

# 视觉感知骨干（专题汇总）

> **图谱专题视图**：本页是知识图谱「👁️ 视觉骨干 (Vision Backbone)」专题的统一入口；在 [图谱专题视图](../../docs/graph.html?topic=vision-backbone) 筛选时，本节点为汇总锚点。

## 一句话定义

**视觉感知骨干专题** 关注机器人策略与 VLA **上游的视觉表征**：从 CNN/ViT 骨干到检测/分割头，再到 **policy 可用的特征接口**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CNN | Convolutional Neural Network | 卷积视觉骨干 |
| ViT | Vision Transformer | Transformer 视觉骨干 |
| ResNet | Residual Network | 经典 CNN 骨干族 |
| YOLO | You Only Look Once | 单阶段目标检测代表 |
| VPR | Visual Representation for Policy | 面向策略的视觉表征设计 |

## 为什么重要

- **VLA / 感知 loco 都依赖视觉特征质量**：骨干选型影响样本效率与泛化。
- **检测 ≠ 策略输入**：需 explicit 设计「骨干 → 任务头 → 策略」衔接。
- **V24 专题**：本库把分散的 backbone/detection 页收成图谱视图。

## 本专题覆盖什么

| 层次 | 典型问题 | 站内入口 |
|------|----------|----------|
| 概念 | 视觉骨干总览 | [Vision Backbones](../concepts/vision-backbones.md) |
| 概念 | ViT 机制入门 | [Vision Transformer](../concepts/vision-transformer.md) |
| 对比 | CNN vs ViT | [CNN vs ViT Backbones](../comparisons/cnn-vs-vit-backbones.md) |
| 概念 | 策略侧表征 | [Visual Representation for Policy](../concepts/visual-representation-for-policy.md) |
| 方法 | 目标检测 | [Object Detection](../methods/object-detection.md) |
| 概念 | 生成式视觉预训练 | [Generative Vision Pretraining](../concepts/generative-vision-pretraining.md) |

## 与其他专题的关系

- **[VLA](./topic-vla.md)**：多模态策略消费视觉骨干特征。
- **[状态估计](./topic-state-estimation.md)**：VIO 与检测/feature 共享视觉栈。
- **[Sim2Real](./topic-sim2real.md)**：视觉域差距与随机化。

## 关联页面

- [Object Detection Model Selection](../queries/object-detection-model-selection.md)
- [Perception Backbone Selection](../queries/perception-backbone-selection.md)
- [3D Spatial VQA](../concepts/3d-spatial-vqa.md)

## 参考来源

- 本库归纳自 [Vision Backbones](../concepts/vision-backbones.md)、[Visual Representation for Policy](../concepts/visual-representation-for-policy.md)
- 图谱专题定义：[docs/topic-filters.js](../../docs/topic-filters.js)（`vision-backbone` 命中规则）
