# ResNet：深度残差学习（arXiv:1512.03385）

> 论文来源归档（ingest）

- **标题：** Deep Residual Learning for Image Recognition
- **作者：** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun（Microsoft Research）
- **类型：** paper / computer-vision / backbone / classification
- **arXiv：** <https://arxiv.org/abs/1512.03385> · PDF：<https://arxiv.org/pdf/1512.03385.pdf>
- **会议：** CVPR 2016（Best Paper）
- **官方实现（Caffe）：** <https://github.com/KaimingHe/deep-residual-networks>
- **入库日期：** 2026-06-06
- **一句话说明：** 用 **残差映射 + 恒等捷径连接** 缓解极深 CNN 的 **退化（degradation）** 问题，使 **152 层** 网络在 ImageNet 上可训练并取得 SOTA，成为后续视觉骨干与机器人感知栈的默认底座之一。

## 核心摘录（面向 wiki 编译）

### 1) 退化问题与残差 reformulation

- **要点：** 单纯堆叠层数时，更深 **plain net** 在 ImageNet / CIFAR-10 上出现 **训练误差反而升高**（非过拟合）；理论上更深网络应至少不差于浅层（恒等映射构造解存在），说明 **优化困难** 是瓶颈。
- **对 wiki 的映射：** [`wiki/entities/paper-resnet-deep-residual-learning.md`](../../wiki/entities/paper-resnet-deep-residual-learning.md)、[`wiki/concepts/vision-backbones.md`](../../wiki/concepts/vision-backbones.md)

### 2) 残差块与捷径连接

- **要点：** 令堆叠层学习 $\mathcal{F}(\mathbf{x}) := \mathcal{H}(\mathbf{x}) - \mathbf{x}$，输出 $\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$；**恒等捷径** 不增参数与 FLOPs；维度不匹配时用 **1×1 投影捷径**（option B）。
- **对 wiki 的映射：** 同上；[`wiki/concepts/deep-learning-foundations.md`](../../wiki/concepts/deep-learning-foundations.md)

### 3) Bottleneck 与深度缩放

- **要点：** ResNet-50/101/152 用 **1×1 → 3×3 → 1×1** bottleneck 块控制算力；ResNet-152 在 ImageNet val 上 top-5 **4.49%**（单模型），集成 **3.57%** test（ILSVRC 2015 分类第一）。
- **对 wiki 的映射：** [`wiki/concepts/vision-backbones.md`](../../wiki/concepts/vision-backbones.md)

### 4) 检测 / 分割迁移

- **要点：** 极深表征在 **COCO 检测** 上相对基线 **28% 相对提升**；同一骨干支撑 ILSVRC & COCO 2015 **检测、定位、分割** 多项第一，说明残差学习是 **通用视觉表征** 而非仅分类技巧。
- **对 wiki 的映射：** [`wiki/methods/object-detection.md`](../../wiki/methods/object-detection.md)

## 相关资料索引

| 资料 | 关系 |
|------|------|
| [Highway Networks](https://arxiv.org/abs/1505.00387)（Srivastava et al., 2015） | 同期 **门控捷径**；ResNet 用 **无参恒等捷径** 始终传递信息 |
| [VGG](https://arxiv.org/abs/1409.1556) | Plain 基线设计哲学来源 |
| [Batch Normalization](https://arxiv.org/abs/1502.03167) | 训练极深网络的标准配套 |
| [GoogLeNet / Inception](https://arxiv.org/abs/1409.4842) | 多尺度卷积与模块堆叠对照 |
| [Faster R-CNN](https://arxiv.org/abs/1506.01497) | 检测管线中常用 **ResNet-101** 作特征骨干 |

## 当前提炼状态

- [x] 要点摘录与 wiki 映射
- [x] 关联概念/方法页交叉引用
