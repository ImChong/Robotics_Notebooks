---
type: entity
tags: [paper, computer-vision, object-detection, real-time, perception, robotics, ultralytics]
status: draft
updated: 2026-11-05
project_id: yolo27
project: https://docs.ultralytics.com/models/yolo27/
related:
  - ./paper-yolo26-unified-realtime-e2e-vision.md
  - ./ultralytics.md
  - ./rf-detr.md
  - ../methods/object-detection.md
sources:
  - ../../sources/papers/yolo27-alphaxiv-overview.md
summary: "YOLO27 预发布综述：小模型采用双尺度 CNN，大模型使用 query 解码；官方模型和权重尚未发布。"
---

# YOLO27: An Overview of Dual-Scale Architecture and Query-Based Real-Time Computer Vision（YOLO27：双尺度架构与基于查询的实时计算机视觉综述）

## 一句话定义

YOLO27 是 Ultralytics 预告的实时视觉模型族；Ranjan Sapkota 的综述整理了其按模型规模选择检测架构的设计和初步结果。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CNN | Convolutional Neural Network | 卷积神经网络 |
| NMS | Non-Maximum Suppression | 非极大值抑制，用于检测去重 |
| mAP | mean Average Precision | 平均精度均值 |
| FPN | Feature Pyramid Network | 多尺度特征金字塔 |

## 流程总览

按论文检测架构绘制，模型具体实现尚未公开。

```mermaid
flowchart TB
  image[输入图像] --> scale{模型规模}
  scale -->|n / s| cnn[轻量 CNN 与特征融合]
  cnn --> maps[细尺度与粗尺度特征]
  maps --> dense[一对多检测头]
  dense --> nms[NMS 去重]
  maps --> unique[可选一对一检测头]
  scale -->|m / l| backbone[骨干网络与多尺度特征融合]
  backbone --> decoder[对象查询与 Transformer 解码]
  decoder --> boxes[类别与边界框]
```

训练阶段的前景对齐分支用于缩小密集监督与一对一预测的差距，推理时移除。论文未给出独立消融，不能仅凭模型间精度差归因。

## 源码运行时序图

不适用：截至 2026-11-05，官方预览注明模型实现、配置和权重尚未发布；公开的通用 Ultralytics 仓库不等于 YOLO27 可运行实现。

## 实验与评测

论文 PDF 报告 COCO 640 输入下，n 至 l 为 **42.3–60.4 mAP**、**0.62–2.32 ms**；条件为 RTX PRO 6000、TensorRT 11、FP16，检测使用 NMS-free 路径。800 输入的 l 报告 61.2 mAP。结果是预览数据，作者声明未独立验证。

官方页面在核对日列出的 640 输入结果为 **42.6–60.5 mAP、0.44–1.86 ms**；其 m/l 骨干描述与论文不同。两者属于不同时间的预览，具体原因未核实，不应拼接为同一组实验。

## 结论

这篇综述的价值在于梳理预告中的双架构设计，目前无法独立复现。

- 小模型默认带 NMS，复现论文评测时需显式区分推理头。
- 小目标收益与训练分支效果需要受控消融确认。
- 设备选型应在模型发布后测量目标硬件上的精度、时延与内存；GPU 模型时延不等于机器人完整感知到控制时延。
- 官方与论文预览存在差异，比较时应记录来源及版本。

## 局限与待核查

本文为草稿：PDF 已核对摘要、架构、评测、复现和声明等部分，尚需逐页通读其余章节。实际模型与权重发布后需重新核验实现与基准。

## 关联页面

- [YOLO26](./paper-yolo26-unified-realtime-e2e-vision.md)
- [Ultralytics](./ultralytics.md)
- [RF-DETR](./rf-detr.md)
- [目标检测](../methods/object-detection.md)

## 参考来源

- [论文来源归档](../../sources/papers/yolo27-alphaxiv-overview.md)
- [Ultralytics YOLO27 官方预览](https://docs.ultralytics.com/models/yolo27/)

## 推荐继续阅读

- [论文 PDF](https://www.alphaxiv.org/abs/2609.yolo27-dual-scale-query-based-vision.pdf)
