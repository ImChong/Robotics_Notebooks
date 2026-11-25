# 视觉 Transformer 课程节点身份与健康度验收清单

- **核对日期：** 2026-11-24
- **范围：** 两张课程截图中的模型项目；仓库树、已有课程策展、相关节点的 frontmatter 与 `scripts/original_reading.py` 核对。
- **状态：** 未通过全量验收。已有路径不等于独立论文身份，status: complete 也不等于 3/3；此表不代替 report_health.py 输出。

## 缺少 original 字段的既有详情或方法页

以下页面经读取确认缺少原文通读声明，必须补原文归档、真实通读和相应引用，不直接改状态冒充满分。

| 项目 | 已有页面 |
|---|---|
| LeNet-5 | [lenet5](../../wiki/entities/lenet5.md) |
| AlexNet | [alexnet](../../wiki/entities/alexnet.md) |
| VGG | [vggnet](../../wiki/entities/vggnet.md) |
| ResNet | [paper-resnet-deep-residual-learning](../../wiki/entities/paper-resnet-deep-residual-learning.md) |
| TNT | [tnt](../../wiki/entities/tnt.md) |
| CvT | [cvt](../../wiki/entities/cvt.md) |
| R-CNN / Fast / Faster | [rcnn-family](../../wiki/methods/rcnn-family.md)（共享方法页，尚需逐篇身份核查） |
| YOLO | [paper-yolo-unified-realtime-detection](../../wiki/entities/paper-yolo-unified-realtime-detection.md) |
| RetinaNet | [retinanet](../../wiki/entities/retinanet.md) |
| DETR | [detr](../../wiki/entities/detr.md) |
| FCN | [fcn-semantic-segmentation](../../wiki/methods/fcn-semantic-segmentation.md) |
| SegNet | [segnet](../../wiki/methods/segnet.md) |
| PSPNet | [pspnet](../../wiki/methods/pspnet.md) |
| Mask R-CNN | [mask-rcnn](../../wiki/methods/mask-rcnn.md) |
| SETR | [setr](../../wiki/entities/setr.md) |
| SegFormer | [segformer](../../wiki/entities/segformer.md) |
| CLIP | [paper-clip](../../wiki/entities/paper-clip.md) |
| BLIP | [blip](../../wiki/entities/blip.md) |
| BLIP-2 | [paper-blip2](../../wiki/entities/paper-blip2.md) |
| LLaVA | [paper-llava](../../wiki/entities/paper-llava.md) |
| MiniGPT-4 | [minigpt4](../../wiki/entities/minigpt4.md) |
| InstructBLIP | [instructblip](../../wiki/entities/instructblip.md) |
| LISA | [lisa](../../wiki/entities/lisa.md) |
| Sa2VA | [sa2va](../../wiki/entities/sa2va.md) |
| Vim | [vision-mamba-vim](../../wiki/entities/vision-mamba-vim.md) |
| VMamba | [vmamba](../../wiki/entities/vmamba.md) |
| MambaIR | [mambair](../../wiki/entities/mambair.md) |
| RS-Mamba | [rs-mamba](../../wiki/entities/rs-mamba.md)（原论文编号应为 2404.02668） |
| ChangeMamba | [changemamba](../../wiki/entities/changemamba.md)（原论文编号应为 2404.03425） |
| VideoMamba | [videomamba](../../wiki/entities/videomamba.md) |
| U-Mamba | [u-mamba](../../wiki/entities/u-mamba.md) |
| SAM 2 | [paper-sam2](../../wiki/entities/paper-sam2.md) |
| SEEM | [seem](../../wiki/entities/seem.md) |

## 已有 original 字段但仍需验收的项目

| 项目 | 页面 | 待核对 |
|---|---|---|
| ViT | [vision-transformer](../../wiki/concepts/vision-transformer.md) | 已有概念页，需核对是否已有独立论文身份；原文归档与实际分数 |
| Mamba | [mamba](../../wiki/concepts/mamba.md) | 同上 |
| U-Net | [unet](../../wiki/methods/unet.md) | 原文归档与独立论文身份 |
| SIDA | [sida](../../wiki/entities/sida.md) | 论文与源码归档的通读状态 |
| SAM | [paper-segment-anything](../../wiki/entities/paper-segment-anything.md) | 原文通读状态及实际分数 |
| Deformable DETR | [正式论文](../../wiki/entities/paper-2010-04159-deformable-detr.md) / [旧详情](../../wiki/entities/deformable-detr.md) | 确认两页指向同一研究成果；迁移独有内容、全库入链与旧 ID alias 后才能删除旧页 |

## 本轮完成的独立节点

- [SegMamba](../../wiki/entities/paper-segmamba.md)：此前仅在作业中以 U-Mamba 近邻替代；现有独立 arXiv 身份、论文归档、方法流程及实际源码时序图，已挂课程与模型架构路线。实际健康度等待 CI 脚本报告。

## 其他验收项

- SENet / SE-ResNet / DANet 在课程中由同一注意力方法页承载；还需按论文身份检查独立节点。SE-ResNet 不是默认独立论文。
- 原始 Transformer 的理论概念与论文实体区分；既有论文节点需核对原文与源码。
- 课程中的数据集、作业、评价指标须按各自类型查重，不把作业误建为论文。
- 逐篇按 arXiv / DOI / 规范标题 / 项目和代码身份查重，必要时迁移历史别名，不仅检查文件名。
- 每个涉及的 wiki 页面以 `python3 scripts/report_health.py` 实测，且 `make ci-preflight` 与 GitHub CI 通过，才算最终验收。

原始资料入口见 [八章一手资料索引](./vision_transformer_primary_sources.md)。
