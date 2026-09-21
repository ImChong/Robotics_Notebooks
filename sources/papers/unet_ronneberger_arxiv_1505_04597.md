# U-Net：生物医学图像分割的卷积网络（arXiv:1505.04597）

> 论文来源归档（ingest）

- **标题：** U-Net: Convolutional Networks for Biomedical Image Segmentation
- **作者：** Olaf Ronneberger, Philipp Fischer, Thomas Brox（University of Freiburg）
- **类型：** paper / computer-vision / segmentation / architecture
- **arXiv：** <https://arxiv.org/abs/1505.04597> · PDF：<https://arxiv.org/pdf/1505.04597.pdf>
- **会议：** MICCAI 2015
- **项目页：** <https://lmb.informatik.uni-freiburg.de/people/ronneberger/u-net/>
- **入库日期：** 2026-09-21
- **一句话说明：** 对称 **编码–解码 U 形** 卷积网，用 **跳跃连接** 把高分辨率浅层特征直接送到解码器，在极少标注下做像素级分割。

## 核心摘录（面向 wiki 编译）

### 1) 收缩路径 + 扩张路径

- **要点：** 左侧反复卷积与下采样捕获上下文；右侧上采样恢复定位。整网呈 U 形，一次前向即可对任意尺寸输入做密集预测（重叠-tile 推理）。
- **对 wiki 的映射：** [`wiki/concepts/unet.md`](../../wiki/concepts/unet.md)

### 2) 跳跃连接保细节

- **要点：** 把编码器同尺度特征图 **裁剪后拼接** 到解码器，避免单纯上采样丢失边界。这对医学细胞轮廓与后来的扩散降噪 U-Net 都是同一几何动机。
- **对 wiki 的映射：** [`wiki/concepts/unet.md`](../../wiki/concepts/unet.md)、[`wiki/concepts/diffusion-model.md`](../../wiki/concepts/diffusion-model.md)

### 3) 弹性形变增广与少样本

- **要点：** 用弹性形变在极少标注图上训练；ISBI 细胞追踪挑战上以大优势获胜。机器人侧的语义分割/占用栅格常复用同一 U 形而不是复用医学增广配方。
- **对 wiki 的映射：** [`wiki/concepts/unet.md`](../../wiki/concepts/unet.md)、[`wiki/overview/ai-architecture-map.md`](../../wiki/overview/ai-architecture-map.md)

## 开源状态（步骤 2.5）

- 项目页提供 Caffe 参考实现与示例。**已开源**。
- 现代训练多用 PyTorch/TF 再实现；扩散库中的 U-Net 是结构后裔，不是原仓直接部署。

## 当前提炼状态

- [x] 要点摘录与 wiki 映射
