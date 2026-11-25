# 视觉 Transformer 理论与实践（Vision Transformer Theory and Practice）一手资料索引

- **类型：** course / primary-source bibliography
- **收录日期：** 2026-11-24
- **课程：** 深蓝学院 [官方课程页](https://www.shenlanxueyuan.com/course/922)（搜索结果与截图标题匹配；正文抓取失败，期次待确认）
- **核对范围：** 用户提供的两张课程截图、已有八章大纲与 wiki 映射、部分论文官方摘要/会议页、部分作者项目和代码发布页；SegMamba v4 已核对方法、实验与结论。
- **通读：** 未完成

本表按截图八章组织，不把课程框图的相邻节点视为论文之间的严格继承关系。论文链接指向原始出版方或 arXiv；代码链接为作者/机构入口（未逐仓检查执行入口及许可证，不能等同于完整复现）。本轮为来源登记，未核实全文的条目不得标记为已通读。复用[已有课程策展](../../wiki/entities/transformer-cv-curriculum.md)，不新建重复实体。

## 第 1 章：基础与注意力

| 主题 | 原始论文或官方文档 |
|---|---|
| CNN / LeNet | [Gradient-Based Learning Applied to Document Recognition](https://doi.org/10.1109/5.726791) |
| SENet / SE-ResNet | [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)（SE-ResNet 是将 SE 模块用于 ResNet 的变体） |
| DANet | [Dual Attention Network for Scene Segmentation](https://openaccess.thecvf.com/content_CVPR_2019/html/Fu_Dual_Attention_Network_for_Scene_Segmentation_CVPR_2019_paper.html) |
| Transformer / MHA | [Attention Is All You Need](https://arxiv.org/abs/1706.03762)；[PyTorch MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) |

## 第 2 章：分类

| 模型 | 原始论文 | 作者代码入口 |
|---|---|---|
| LeNet-5 | [1998](https://doi.org/10.1109/5.726791) | — |
| AlexNet | [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) | — |
| VGG | [1409.1556](https://arxiv.org/abs/1409.1556) | — |
| ResNet | [1512.03385](https://arxiv.org/abs/1512.03385) | [作者仓库](https://github.com/KaimingHe/deep-residual-networks) |
| ViT | [2010.11929](https://arxiv.org/abs/2010.11929) | [Google Research](https://github.com/google-research/vision_transformer) |
| TNT | [2103.00112](https://arxiv.org/abs/2103.00112) | [Huawei CV-Backbones](https://github.com/huawei-noah/CV-Backbones)（TNT 子目录） |
| CvT | [2103.15808](https://arxiv.org/abs/2103.15808) | [Microsoft](https://github.com/microsoft/CvT) |

分类数据集： [MNIST](https://yann.lecun.com/exdb/mnist/)、[CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html)、[ImageNet / ImageNet-21K](https://www.image-net.org/)、[JFT-300M 原始研究](https://arxiv.org/abs/1707.02968)。JFT-300M 的论文入口不代表完整数据可公开下载。

## 第 3 章：检测

| 模型 | 原始论文 | 作者代码入口 |
|---|---|---|
| R-CNN | [1311.2524](https://arxiv.org/abs/1311.2524) | [rcnn](https://github.com/rbgirshick/rcnn) |
| Fast R-CNN | [1504.08083](https://arxiv.org/abs/1504.08083) | [fast-rcnn](https://github.com/rbgirshick/fast-rcnn) |
| Faster R-CNN | [1506.01497](https://arxiv.org/abs/1506.01497) | [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn) |
| YOLO | [1506.02640](https://arxiv.org/abs/1506.02640) | [Darknet](https://github.com/pjreddie/darknet) |
| RetinaNet | [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) | — |
| DETR | [2005.12872](https://arxiv.org/abs/2005.12872) | [facebookresearch/detr](https://github.com/facebookresearch/detr) |
| Deformable DETR | [2010.04159](https://arxiv.org/abs/2010.04159) | [fundamentalvision](https://github.com/fundamentalvision/Deformable-DETR) |

数据与评测： [COCO](https://cocodataset.org/)、[COCO API](https://github.com/cocodataset/cocoapi)、[Objects365](https://www.objects365.org/)、[VisDrone 官方数据发布](https://github.com/VisDrone/VisDrone-Dataset)。检测作业需明确 DET 数据划分，不把视频追踪划分与单帧检测混用。

## 第 4 章：分割

| 模型 | 原始论文 | 作者代码入口 |
|---|---|---|
| FCN | [1411.4038](https://arxiv.org/abs/1411.4038) | [shelhamer/fcn.berkeleyvision.org](https://github.com/shelhamer/fcn.berkeleyvision.org) |
| U-Net | [1505.04597](https://arxiv.org/abs/1505.04597) | [作者项目](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) |
| SegNet | [1511.00561](https://arxiv.org/abs/1511.00561) | [SegNet](https://github.com/alexgkendall/caffe-segnet) |
| PSPNet | [1612.01105](https://arxiv.org/abs/1612.01105) | [PSPNet](https://github.com/hszhao/PSPNet) |
| Mask R-CNN | [1703.06870](https://arxiv.org/abs/1703.06870) | [Detectron](https://github.com/facebookresearch/Detectron) |
| SETR | [2012.15840](https://arxiv.org/abs/2012.15840) | [fudan-zvg/SETR](https://github.com/fudan-zvg/SETR) |
| SegFormer | [2105.15203](https://arxiv.org/abs/2105.15203) | [NVlabs/SegFormer](https://github.com/NVlabs/SegFormer) |
| 全景分割定义 | [Panoptic Segmentation](https://arxiv.org/abs/1801.00868) | — |

数据：[PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)、[ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/)、[Cityscapes](https://www.cityscapes-dataset.com/)、[Mapillary Vistas](https://www.mapillary.com/dataset/vistas)。

## 第 5–6 章：多模态与下游

| 模型 | 原始论文 | 作者代码或项目 |
|---|---|---|
| CLIP | [2103.00020](https://arxiv.org/abs/2103.00020) | [OpenAI CLIP](https://github.com/openai/CLIP) |
| BLIP | [2201.12086](https://arxiv.org/abs/2201.12086) | [Salesforce BLIP](https://github.com/salesforce/BLIP) |
| BLIP-2 | [2301.12597](https://arxiv.org/abs/2301.12597) | [LAVIS](https://github.com/salesforce/LAVIS) |
| LLaVA | [2304.08485](https://arxiv.org/abs/2304.08485) | [LLaVA](https://github.com/haotian-liu/LLaVA) |
| MiniGPT-4 | [2304.10592](https://arxiv.org/abs/2304.10592) | [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) |
| InstructBLIP | [2305.06500](https://arxiv.org/abs/2305.06500) | [LAVIS](https://github.com/salesforce/LAVIS) |
| LISA | [2308.00692](https://arxiv.org/abs/2308.00692) | [LISA](https://github.com/JIA-Lab-research/LISA) |
| Sa2VA | [2501.04001](https://arxiv.org/abs/2501.04001) | [项目](https://lxtgh.github.io/project/sa2va/)、[代码](https://github.com/bytedance/Sa2VA) |
| SIDA | [2412.04292](https://arxiv.org/abs/2412.04292) | [项目](https://hzlsaber.github.io/projects/SIDA/)、[代码](https://github.com/hzlsaber/SIDA) |

数据：[Flickr30K Entities 论文](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Plummer_Flickr30k_Entities_Collecting_ICCV_2015_paper.html)、[VaTeX](https://eric-xw.github.io/vatex-website/)、[WIT](https://github.com/google-research-datasets/wit)、[LLaVA 官方仓库中的 instruction 数据入口](https://github.com/haotian-liu/LLaVA)。SIDA 的任务是图像伪造检测、区域定位与解释，不是通用机器人分割能力的证明。

## 第 7 章：Mamba 与视觉状态空间模型

| 模型 | 原始论文 | 作者代码入口 |
|---|---|---|
| SSM / S4 | [2111.00396](https://arxiv.org/abs/2111.00396) | [state-spaces/s4](https://github.com/state-spaces/s4) |
| Mamba | [2312.00752](https://arxiv.org/abs/2312.00752) | [state-spaces/mamba](https://github.com/state-spaces/mamba) |
| Vim | [2401.09417](https://arxiv.org/abs/2401.09417) | [hustvl/Vim](https://github.com/hustvl/Vim) |
| VMamba | [2401.10166](https://arxiv.org/abs/2401.10166) | [MzeroMiko/VMamba](https://github.com/MzeroMiko/VMamba) |
| MambaIR | [2402.15648](https://arxiv.org/abs/2402.15648) | [csguoh/MambaIR](https://github.com/csguoh/MambaIR) |
| RS-Mamba（密集预测） | [2404.02668](https://arxiv.org/abs/2404.02668) | [NJU-LHRS](https://github.com/NJU-LHRS/Official_Remote_Sensing_Mamba) |
| ChangeMamba | [2404.03425](https://arxiv.org/abs/2404.03425) | — |
| VideoMamba | [2403.06977](https://arxiv.org/abs/2403.06977) | [OpenGVLab/VideoMamba](https://github.com/OpenGVLab/VideoMamba) |
| U-Mamba | [2401.04722](https://arxiv.org/abs/2401.04722) | [bowang-lab/U-Mamba](https://github.com/bowang-lab/U-Mamba) |
| SegMamba（作业 6） | [2401.13560v4](https://arxiv.org/abs/2401.13560v4) | [ge-xing/SegMamba](https://github.com/ge-xing/SegMamba) |

数据：[BraTS 2023 Adult Glioma 官方挑战](https://www.synapse.org/Synapse:syn51156910/wiki/622351)。需遵守注册和数据使用条款；不要将作者代码公开误解为医疗数据无条件开放。

## 第 8 章：视觉基础模型

| 模型 | 原始论文 | 作者代码或项目 |
|---|---|---|
| SAM | [2304.02643](https://arxiv.org/abs/2304.02643) | [官方项目](https://segment-anything.com/)、[代码](https://github.com/facebookresearch/segment-anything) |
| SAM 2 | [2408.00714](https://arxiv.org/abs/2408.00714) | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) |
| SEEM | [2304.06718](https://arxiv.org/abs/2304.06718) | [UX-Decoder/Segment-Everything-Everywhere-All-At-Once](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once) |

作业 7：[COD10K 原论文](https://openaccess.thecvf.com/content_CVPR_2020/html/Fan_Camouflaged_Object_Detection_CVPR_2020_paper.html)、[作者项目/数据入口](https://dengpingfan.github.io/pages/COD.html)。

## 查重与已发现的身份错误

- 既有 [RS-Mamba](../../wiki/entities/rs-mamba.md) 推荐阅读指向 2403.19654，实际是遥感分类 **RSMamba**，与截图的密集预测 RS-Mamba (2404.02668) 不同。
- 既有 [ChangeMamba](../../wiki/entities/changemamba.md) 推荐阅读指向 2404.03493，实际是脉冲神经网络参数研究；正确的遥感变化检测论文是 2404.03425。
- 既有 [Deformable DETR 正式论文节点](../../wiki/entities/paper-2010-04159-deformable-detr.md) 应与历史课程页统一身份，不再新增第三个节点。
- 作业 6 不能以 U-Mamba 代替 SegMamba；SegMamba 的独立论文归档见 [SegMamba](../papers/segmamba_arxiv_2401_13560.md)。

## 后续知识层验收（未完成）

- 逐篇全文、版本、源码入口与调用链核对；将已有课程级实体升级为唯一论文实体，处理历史路径与入链；不得批量新增重复论文节点。
- 已核对纵深路线：[模型架构 Stage 1/2/4](../../roadmap/depth-model-architecture.md)、[VLA](../../roadmap/depth-vla.md)，后续按实际完成节点挂载。
- 补 Mermaid 方法流与实际代码时序图；运行 make ci-preflight 和 report_health.py 后更新 PR。当前只登记来源，不宣称完成或满分。
