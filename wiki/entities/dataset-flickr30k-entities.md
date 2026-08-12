---
type: entity
tags:
  - dataset
  - computer-vision
  - benchmark
status: complete
updated: 2026-08-12
summary: "Flickr30K 的短语级图文对齐扩展：把描述中的实体短语与图像区域框关联，服务定位/接地评测。"
related:
  - ../entities/transformer-cv-curriculum.md
  - ../concepts/vision-backbones.md
  - ../methods/object-detection.md
sources:
  - ../../sources/courses/transformer_cv_applications_syllabus.md
---

# Flickr30K Entities

## 一句话定义

**Flickr30K Entities**：Flickr30K 的短语级图文对齐扩展：把描述中的实体短语与图像区域框关联，服务定位/接地评测。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Flickr30K | Flickr30K Entities | 短语–区域对齐图文集 |
| CV | Computer Vision | 计算机视觉任务语境 |
| SOTA | State of the Art | 排行榜对照参考 |
| VLM | Vision-Language Model | 多模态模型预训练/评测相关 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 规模 | 约 31k 图像，每图多句描述 + 实体框。
| 模态 | RGB + 文本短语 + 区域框。
| 许可证 | Flickr 图像版权复杂；研究使用需遵守原协议。
| 适配形态 | VLM grounding 评测；非机器人轨迹数据。
| 重定向就绪度 | 不适用；区域–短语对可直接训练 grounding。

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

- [Flickr30K Entities 官方入口](https://github.com/BryanPlummer/flickr30k_entities)
