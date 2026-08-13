---
type: method
tags:
  - segmentation
  - computer-vision
  - cnn
  - deep-learning
status: complete
updated: 2026-08-12
summary: "U-Net 以对称编码器–解码器与跳跃连接融合多尺度特征，在医学分割上极成功，并广泛迁移到机器人与工业缺陷分割。"
related:
  - ../concepts/image-segmentation-taxonomy.md
  - ../entities/transformer-cv-curriculum.md
  - ../entities/dataset-ade20k.md
  - ../queries/robot-perception-stack-selection-loop.md
sources:
  - ../../sources/courses/transformer_cv_applications_syllabus.md
---

# U-Net

## 一句话定义

U-Net 以对称编码器–解码器与跳跃连接融合多尺度特征，在医学分割上极成功，并广泛迁移到机器人与工业缺陷分割。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| U-Net | U-Net | U 形编解码分割网 |
| Skip | Skip Connection | 拷贝裁剪拼接浅层特征 |
| Encoder | Encoder / Contracting path | 下采样提取语义 |
| Decoder | Decoder / Expanding path | 上采样恢复细节 |
| Dice | Dice Loss/Coefficient | 医学分割常用损失/指标 |

## 为什么重要

- 课程 4.2 的 CNN 分割主线节点，为理解 SETR/SegFormer 提供对照。
- 机器人侧仍大量使用 U-Net/Mask R-CNN 类结构做缺陷、可抓取与语义图层。

## 主要技术路线

| 路线 | 机制 | 说明 |
|------|------|------|
| 经典 U-Net | 对称编解码 + skip | 医学分割默认 |
| U-Net++ / 注意力 U-Net | 密集 skip / 门控 | 边界增强 |
| 3D U-Net | 体数据卷积 | 医学体积分割 |

## 核心原理

左侧收缩路径提语义，右侧扩张路径上采样；每级 skip 拼接同分辨率编码器特征，保留边界细节。

```mermaid
flowchart LR
  IMG["图像"] --> ENC["编码/骨干"] --> HEAD["分割头"] --> MASK["像素/实例输出"]
```

## 工程实践

| 项 | 建议 |
|----|------|
| 数据 | ADE20K/Cityscapes/自有掩码；注意类别映射 |
| 损失 | 交叉熵 / Dice / 辅助头按论文配置 |
| 部署 | 测全分辨率延迟；机载可降输入尺寸 |

## 局限与风险

CNN 分割受感受野与多尺度模块设计约束；开放集与提示分割需看 SAM/SEEM。标注噪声会直接体现在边界指标上。

## 关联页面

- [分割任务分类](../concepts/image-segmentation-taxonomy.md)
- [SETR](../entities/setr.md)
- [SegFormer](../entities/segformer.md)
- [Transformer CV 课程策展](../entities/transformer-cv-curriculum.md)
- [机器人视觉感知栈选型闭环知识链](../queries/robot-perception-stack-selection-loop.md) — 跳跃连接多尺度融合，②层工业/医学缺陷分割迁移的常用底子

## 参考来源

- [Transformer 视觉应用课程大纲](../../sources/courses/transformer_cv_applications_syllabus.md)

## 推荐继续阅读

- [原论文](https://arxiv.org/abs/1505.04597)
