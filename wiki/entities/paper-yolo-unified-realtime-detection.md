---
type: entity
tags:
  - paper
  - computer-vision
  - object-detection
  - real-time
  - perception
  - robotics
status: complete
updated: 2026-06-06
arxiv: "1506.02640"
venue: "CVPR 2016"
code: https://github.com/pjreddie/darknet
related:
  - ../methods/object-detection.md
  - ../concepts/vision-backbones.md
  - ./paper-resnet-deep-residual-learning.md
  - ./booster-robocup-demo.md
  - ../tasks/humanoid-soccer.md
sources:
  - ../../sources/papers/yolo_arxiv_1506_02640.md
  - ../../sources/papers/vision_backbone_detection_classics.md
summary: "YOLO v1（arXiv:1506.02640）将目标检测重构为单次 CNN 回归，在 Pascal VOC 上达 63.4 mAP @ 45 FPS，开启端到端实时检测范式并深刻影响机器人机载感知栈。"
---

# YOLO v1（You Only Look Once）

**YOLO**（You Only Look Once）是 Joseph Redmon 等提出的 **统一实时目标检测** 方法（arXiv:1506.02640，CVPR 2016）。它将检测从「区域提议 + 分类器」的多阶段管线 **压缩为单次前向回归**：整图经单一卷积网络直接输出网格单元上的 **边界框坐标 + 置信度 + 类别概率**，在 Titan X 上实现 **45 FPS**，是后续 **YOLO 系列** 与大量机器人实时感知系统的起点。

## 一句话定义

**把检测当成从像素到框坐标的端到端回归，一次看完整张图就输出所有目标，用速度换部分定位精度，但显著减少背景误报。**

## 为什么重要

- **范式转换：** 首次在 **精度可接受** 的前提下把检测做到 **真·实时**（>30 FPS），证明 **单网络端到端优化** 可行。
- **全局上下文：** 与滑动窗口 / R-CNN 局部 patch 不同，YOLO 训练与推理时 **看到全图**，背景假阳性约为 Fast R-CNN 的 **三分之一**。
- **机器人落地广泛：** 从 RoboCup [Booster 演示](./booster-robocup-demo.md) 的 YOLOv8，到人形足球 [寻球模块](../tasks/humanoid-soccer.md)，**YOLO 谱系** 仍是机载边缘设备的主流选择之一。

## 核心结构

| 模块 | 作用 |
|------|------|
| **S×S 网格** | VOC 设定 **S=7**；物体中心落入的格子负责检测该物体 |
| **每格预测** | **B=2** 个框（5 维：x,y,w,h,conf）+ **C=20** 条件类概率 → **7×7×30** 张量 |
| **骨干网络** | 24 conv + 2 FC，受 GoogLeNet 启发；ImageNet 预训练后 **448×448** 微调 |
| **损失设计** | $\lambda_{\text{coord}}=5$ 加重框回归；$\lambda_{\text{noobj}}=0.5$ 抑制空格子置信度；**responsible predictor** 按 IOU 分配责任 |
| **Fast YOLO** | 9 层卷积轻量版：**155 FPS**，mAP 52.7% |

### 检测流水线

```mermaid
flowchart LR
  img[输入图像] --> resize["Resize 448×448"]
  resize --> cnn["24-layer CNN\n(ImageNet 预训练)"]
  cnn --> tensor["7×7×30 预测张量"]
  tensor --> thresh[置信度阈值]
  thresh --> nms[可选 NMS]
  nms --> det[最终检测框]
```

## 方法栈

见上文 **核心结构** 与检测流水线；网络层数、损失权重与 responsible predictor 机制以原文 §2 为准（[参考来源](#参考来源)）。

## 实验与评测

- Pascal VOC 2007：**YOLO 63.4 mAP @ 45 FPS**；**Fast YOLO 52.7 mAP @ 155 FPS**（Titan X）。
- VOC 2012 leaderboard：与 SOTA 两阶段方法对比见论文 Table 3；**Fast R-CNN + YOLO** 组合 **70.7 mAP**。
- 跨域：在 **artwork** 上泛化优于 DPM / R-CNN。

## 与其他工作对比

- 相对 **Fast R-CNN**：YOLO **定位错误更多**，但 **背景误报约少 3×**；组合可 **+3.2 mAP**。
- 相对 **Faster R-CNN**：精度略低，但 **21 FPS（VGG-16 版）** vs 7 FPS，真·实时 vs 准实时。
- 相对 **DPM / 滑动窗口**：**单网络端到端**，无需分模块调参。

## 常见误区或局限

- **误区：「YOLO 精度一直不如两阶段。」** v1 时代确实 **定位错误占主导**，但后续 YOLOv3–v8 已大幅缩小差距；选型应看 **延迟预算** 与 **目标尺度**。
- **局限（v1）：** 每格仅 **2 框 + 1 类**，对 **小目标与密集目标** 吃力；框回归用 sum-squared error，大小框误差权重不优。
- **工程注意：** 网格设计带来 **强空间先验**——物体必须落在某格中心附近才被该格负责；后续版本引入 **anchor / 多尺度** 缓解。

## 关联页面

- [目标检测（方法）](../methods/object-detection.md)
- [视觉骨干（概念）](../concepts/vision-backbones.md)
- [ResNet（论文实体）](./paper-resnet-deep-residual-learning.md)
- [Booster RoboCup Demo](./booster-robocup-demo.md)
- [Humanoid Soccer（任务）](../tasks/humanoid-soccer.md)

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| YOLO | You Only Look Once | 单次前向回归的实时目标检测范式 |
| mAP | mean Average Precision | 检测任务标准精度指标 |
| IOU | Intersection over Union | 预测框与真值框重叠率 |
| NMS | Non-Maximum Suppression | 抑制重叠冗余检测框的后处理 |
| FPS | Frames Per Second | 每秒处理帧数，实时性指标 |
| VOC | Pascal Visual Object Classes | 经典目标检测基准数据集 |
| CNN | Convolutional Neural Network | 卷积神经网络，处理图像/深度感知 |

## 参考来源

- [YOLO v1 论文摘录（arXiv:1506.02640）](../../sources/papers/yolo_arxiv_1506_02640.md)
- [经典视觉骨干与检测文献簇](../../sources/papers/vision_backbone_detection_classics.md)

## 推荐继续阅读

- 论文 PDF：<https://arxiv.org/pdf/1506.02640.pdf>
- 项目页：<http://pjreddie.com/yolo/>
- Darknet：<https://github.com/pjreddie/darknet>
- [Fast R-CNN](https://arxiv.org/abs/1504.08083)（误差分析对照）
- [Faster R-CNN](https://arxiv.org/abs/1506.01497)（两阶段准实时路线）
