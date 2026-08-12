# Transformer 架构及其在计算机视觉中的应用（课程大纲）

> 来源归档（ingest）

- **类型：** course
- **来源：** 课程大纲截图整理（Transformer 基础 → 分类 / 检测 / 分割 → 多模态 VLM → Mamba → 视觉基础模型）
- **收录日期：** 2026-08-12
- **一句话说明：** 八章系统课：从 CNN vs Transformer 与注意力基础，经 ViT/检测/分割与多模态大模型（CLIP→LLaVA），再到 Mamba 视觉变体与 SAM/SEEM 等视觉基础模型趋势；每节映射本库独立 wiki 详情节点。

## 为什么值得保留

- 把 **感知骨干选型**（CNN/ViT）、**经典检测分割管线**、**VLM/MLLM** 与 **SSM/Mamba** 串成一条可对照学习链，直接服务机器人上游视觉与具身多模态。
- 与本库 [视觉感知骨干知识链](../../wiki/overview/hub-vision-backbone.md)、[CNN vs ViT](../../wiki/comparisons/cnn-vs-vit-backbones.md)、[Object Detection](../../wiki/methods/object-detection.md) 及 SAM/BLIP-2 实体页形成互补：本大纲补齐经典 CNN、DETR 族、分割 Transformer、MLLM 下游与 Mamba 视觉节点。
- 可作为「截图技术点 → 独立 wiki 详情节点」覆盖验收清单。

## 章节大纲（8 章 + 实践作业）

### 第 1 章 Transformer 基础知识介绍

| 节 | 主题 |
|----|------|
| 1.1.1 | 经典 CNN 结构与卷积原理 |
| 1.1.2 | CNN 与 Transformer 网络结构差异对比 |
| 1.2.1 | 早期 Attention：SENet、SE-ResNet、DANet |
| 1.2.2 | Attention 与 Multi-Head Attention 详解 |
| 1.3.1 | Transformer 网络结构详解 |
| 1.4 | 章节小结 |
| **作业 1** | 多头注意力模块实现与调试 |

### 第 2 章 Transformer 在图像分类中的应用

| 节 | 主题 |
|----|------|
| 2.1.1 | 常用数据集：MNIST、CIFAR、ImageNet |
| 2.1.2 | 大规模数据集：ImageNet-21K、JFT-300M |
| 2.2 | CNN 分类：LeNet5、AlexNet、VGGNet、ResNet |
| 2.3 | Transformer 分类：ViT、TNT、CvT |
| 2.4 | 章节小结 |
| **作业 2** | 基于 TNT 的图像分类任务 |

### 第 3 章 Transformer 在目标检测中的应用

| 节 | 主题 |
|----|------|
| 3.1.1 | 数据集：COCO、Objects365 |
| 3.1.2 | 目标检测评价指标 |
| 3.2.1 | 两阶段：R-CNN、Fast R-CNN、Faster R-CNN |
| 3.2.2 | 单阶段：YOLO、RetinaNet |
| 3.3 | Transformer 检测：DETR、Deformable DETR |
| 3.4 | 章节小结 |
| **作业 3** | VisDrone 上 DETR 训练 |

### 第 4 章 Transformer 在图像分割中的应用

| 节 | 主题 |
|----|------|
| 4.1.1 | 语义 / 实例 / 全景分割任务 |
| 4.1.2 | 数据集：PASCAL VOC、ADE20K、Cityscapes、Mapillary |
| 4.2 | CNN 分割：FCN、U-Net、SegNet、PSPNet、Mask R-CNN |
| 4.3 | Transformer 分割：SETR、SegFormer |
| 4.4 | 章节小结 |
| **作业 4** | ADE20K 上 SETR 训练 |

### 第 5–6 章 Transformer 在多模态任务中的应用

| 节 | 主题 |
|----|------|
| 5.1.1 | 多模态基本概念 |
| 5.1.2 | 数据集：Flickr30K Entities、VaTeX、WIT |
| 5.1.3 | 多模态 LLM 发展路线 |
| 5.2 | CLIP、BLIP、BLIP-2 |
| 6.1 | LLaVA、MiniGPT-4、InstructBLIP |
| 6.2 | VLM 下游：LISA、Sa2VA、SIDA |
| **作业 5** | `llava_instruct_150k.json` 微调 LLaVA |

### 第 7 章 新型网络架构 — Mamba

| 节 | 主题 |
|----|------|
| 7.1.1 | RNN / CNN / Transformer 结构优劣 |
| 7.1.2 | 状态空间模型（SSM） |
| 7.2 | Vision Mamba（Vim）、Visual Mamba（VMamba） |
| 7.3 | MambaIR、RS-Mamba、ChangeMamba、VideoMamba、U-Mamba |
| **作业 6** | BraTS 2023 GLI 上 SegMamba 训练 |

### 第 8 章 视觉基础模型与发展趋势

| 节 | 主题 |
|----|------|
| 8.1 | SAM、SAM 2、SEEM |
| 8.2 | 趋势：单模态→多模态；小模型→基础模型；专用→通用；闭集→开集；感知→感知+推理 |
| **作业 7** | COD10K 上微调 SAM |

## 对 wiki 的映射

策展总表见 [transformer-cv-curriculum](../../wiki/entities/transformer-cv-curriculum.md)。

| 课程节点（摘要） | wiki 详情页 |
|------------------|-------------|
| CNN / 卷积 | [convolutional-neural-network](../../wiki/concepts/convolutional-neural-network.md) |
| CNN vs Transformer | [cnn-vs-vit-backbones](../../wiki/comparisons/cnn-vs-vit-backbones.md) |
| SENet / DANet | [channel-spatial-attention](../../wiki/methods/channel-spatial-attention.md) |
| Multi-Head Attention | [multi-head-attention](../../wiki/concepts/multi-head-attention.md) |
| Transformer 基础 | [transformer](../../wiki/concepts/transformer.md) |
| ImageNet 族 | [dataset-imagenet](../../wiki/entities/dataset-imagenet.md) |
| ViT / TNT / CvT | [vision-transformer](../../wiki/concepts/vision-transformer.md)、[tnt](../../wiki/entities/tnt.md)、[cvt](../../wiki/entities/cvt.md) |
| DETR 族 | [detr](../../wiki/entities/detr.md)、[deformable-detr](../../wiki/entities/deformable-detr.md) |
| 分割任务分类 | [image-segmentation-taxonomy](../../wiki/concepts/image-segmentation-taxonomy.md) |
| CLIP / LLaVA / SAM | [clip](../../wiki/entities/clip.md)、[llava](../../wiki/entities/llava.md)、[paper-segment-anything](../../wiki/entities/paper-segment-anything.md) |
| SSM / Mamba 视觉 | [state-space-model-ssm](../../wiki/concepts/state-space-model-ssm.md)、[vision-mamba-vim](../../wiki/entities/vision-mamba-vim.md)、[vmamba](../../wiki/entities/vmamba.md) |
| 视觉基础模型趋势 | [visual-foundation-model-trends](../../wiki/concepts/visual-foundation-model-trends.md) |
| 多模态 LLM 路线 | [multimodal-llm-development](../../wiki/overview/multimodal-llm-development.md) |
| 策展总表 | [transformer-cv-curriculum](../../wiki/entities/transformer-cv-curriculum.md) |

## 当前提炼状态

- [x] 大纲章节与作业清单
- [x] wiki 独立节点全覆盖映射（见策展 hub）
- [ ] 各作业数据集（VisDrone / BraTS / COD10K）单独深读（可选后续）
