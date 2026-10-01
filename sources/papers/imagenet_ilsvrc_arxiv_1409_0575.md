# ImageNet Large Scale Visual Recognition Challenge（IJCV 2015 / arXiv:1409.0575）

> 论文来源归档（ingest · 数据集 / benchmark 一手资料）

- **标题：** ImageNet Large Scale Visual Recognition Challenge
- **作者：** Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, Li Fei-Fei（* equal contribution；Stanford / Michigan / MIT / UNC 等）
- **类型：** paper / dataset / benchmark / object-recognition
- **期刊：** International Journal of Computer Vision, 2015
- **arXiv：** <https://arxiv.org/abs/1409.0575> · PDF：<https://arxiv.org/pdf/1409.0575.pdf>
- **竞赛入口：** <http://image-net.org/challenges/LSVRC/>
- **入库日期：** 2026-10-01
- **一句话说明：** 定义 **ILSVRC** 公开训练集 + 隐藏测试标注的 **年度竞赛机制**，将 ImageNet 子集规范为 **1000 类、百万级图像** 的分类 / 检测 / 定位 benchmark，并系统回顾 **2010–2014** 算法突破与人类误差对照。

## 核心摘录（面向 wiki 编译）

### 1) 与 PASCAL VOC 的尺度跃迁

- **要点：** ILSVRC **2010** 训练集约 **1,461,406** 张图、**1000** 类，相对 PASCAL VOC 2010（约 **1.97 万** 图、**20** 类）数量级跃升；延续 **(1) 公开数据集 + (2) 年度竞赛/workshop** 双组件模式。
- **对 wiki 的映射：** [`wiki/entities/dataset-imagenet.md`](../../wiki/entities/dataset-imagenet.md)

### 2) 两类标注与任务

- **要点：** **图像级** 二值 presence 标签（「图中有车」）；**物体级** 紧致 **bounding box + 类名**（检测/定位）。测试集标注长期对参赛者隐藏，由 **evaluation server** 统一评测。
- **对 wiki 的映射：** [`wiki/methods/object-detection.md`](../../wiki/methods/object-detection.md)

### 3) ImageNet 全库 vs ILSVRC-1K

- **要点：** **ImageNet 本体**（Deng et al., 2009）截至 **2014-08** 约 **21,841 synsets、14,197,122** 张人工验证全分辨率图；**ILSVRC** 使用其中 **子集** 训练算法，并用 ImageNet 采集协议扩展测试图。**ImageNet-21K** 在 wiki 语境常指更广 synset 覆盖（与 torchvision/timm 的 21k 类预训练词表对齐，版本需看具体发布）。
- **对 wiki 的映射：** [`wiki/entities/dataset-imagenet.md`](../../wiki/entities/dataset-imagenet.md)「数据集速查」

### 4)  crowdsourcing 与评测现实

- **要点：** 百万级标注依赖 **Su et al. / Deng et al.** 等众包与清洗流程；部分类（如成串香蕉）边界难标，**完美手工标注不可行**，需调整评测准则。
- **对 wiki 的映射：** 同上「局限与风险」

### 5) 历史影响（AlexNet 时代）

- **要点：** 论文汇总 **2010–2014** 分类/检测 SOTA 曲线；**2012 AlexNet**、**2014 VGG/GoogLeNet**、**2015 ResNet** 等里程碑均在此 benchmark 上确立 **ImageNet 预训练 → 下游 COCO 检测** 的工业习惯。
- **对 wiki 的映射：** [`wiki/entities/alexnet.md`](../../wiki/entities/alexnet.md)、[`wiki/entities/paper-resnet-deep-residual-learning.md`](../../wiki/entities/paper-resnet-deep-residual-learning.md)、[`wiki/concepts/vision-backbones.md`](../../wiki/concepts/vision-backbones.md)

## 相关资料索引

| 资料 | 关系 |
|------|------|
| [ImageNet CVPR 2009](imagenet_hierarchical_database_cvpr_2009.md) | 全库层次本体定义 |
| [image-net.org](../sites/image-net-org.md) | 下载与隐私/过滤政策更新 |
| [PASCAL VOC](https://host.robots.ox.ac.uk/pascal/VOC/) | ILSVRC 前身 benchmark |

## 当前提炼状态

- [x] 要点摘录与 wiki 映射
- [x] 规模数字与任务定义对齐 arXiv 原文
