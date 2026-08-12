---
type: entity
tags:
  - dataset
  - computer-vision
  - benchmark
status: complete
updated: 2026-08-12
summary: "32×32 彩色小图分类基准（10/100 类）；验证 CNN/ViT 小模型与正则技巧的常用中小规模集。"
related:
  - ../entities/transformer-cv-curriculum.md
  - ../concepts/vision-backbones.md
  - ../methods/object-detection.md
sources:
  - ../../sources/courses/transformer_cv_applications_syllabus.md
---

# CIFAR-10 / CIFAR-100

## 一句话定义

**CIFAR-10 / CIFAR-100**：32×32 彩色小图分类基准（10/100 类）；验证 CNN/ViT 小模型与正则技巧的常用中小规模集。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CIFAR | Canadian Institute for Advanced Research | 小尺寸彩色分类基准族 |
| CV | Computer Vision | 计算机视觉任务语境 |
| SOTA | State of the Art | 排行榜对照参考 |
| VLM | Vision-Language Model | 多模态模型预训练/评测相关 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 规模 | CIFAR-10：60k 张 32×32；CIFAR-100：100 细类。
| 模态 | RGB 图像；无实例掩码。
| 许可证 | 学术研究广泛使用的公开基准。
| 适配形态 | 算法原型；分辨率远低于机器人相机。
| 重定向就绪度 | 不适用；可直接分类训练。

## 为什么重要

- 课程大纲将其列为对应章节的标准数据入口，便于对齐论文数字与作业配置。
- 机器人感知选型时，需分清 **预训练基准** 与 **部署域数据**：本集多属前者。
- 与同章模型页交叉：先定数据协议，再谈骨干与损失。

## 核心原理

数据以监督学习标注（类别/框/掩码/图文对）组织；训练时按任务采样 batch，评测遵循官方 split 与指标（分类 accuracy、检测 mAP、分割 mIoU、检索 Recall 等）。规模与噪声水平决定可支撑的模型体量——JFT/WIT 类偏 scaling 论证，MNIST/CIFAR 偏教学冒烟。

```mermaid
flowchart LR
  RAW["原始采集/网页"] --> ANN["标注或弱标签"]
  ANN --> SPLIT["train/val/test"]
  SPLIT --> TRAIN["模型训练"]
  TRAIN --> MET["官方指标评测"]
```

## 工程实践

| 项 | 建议 |
|----|------|
| 下载 | 走官方或镜像；注意许可与注册 |
| 机器人迁移 | ImageNet/COCO 预训练 → 自有域精调；勿只报源域分 |
| 缓存 | 使用 TFDS/torchvision/HuggingFace datasets 统一接口 |
| 质控 | 检查坏图、空框、类别映射是否与配置一致 |

## 局限与风险

- 与机器人相机分布常有域差（模糊、运动、鱼眼、室内纹理）。
- 部分集（如 JFT）**不可公开复现**，只能当文献对照。
- 许可与人脸/版权图需在产品化前法务审阅。

## 关联页面

- [Transformer CV 课程策展](../entities/transformer-cv-curriculum.md)
- [Vision Backbones](../concepts/vision-backbones.md)
- [Object Detection](../methods/object-detection.md)

## 参考来源

- [Transformer 视觉应用课程大纲](../../sources/courses/transformer_cv_applications_syllabus.md)

## 推荐继续阅读

- [CIFAR-10 / CIFAR-100 官方入口](https://www.cs.toronto.edu/~kriz/cifar.html)
