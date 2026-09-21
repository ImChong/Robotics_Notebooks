# LeCun et al.：面向文档识别的梯度学习（Proc. IEEE, 1998）

> 论文来源归档（ingest）

- **标题：** Gradient-Based Learning Applied to Document Recognition
- **作者：** Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner
- **类型：** paper / computer-vision / cnn / architecture
- **期刊：** Proceedings of the IEEE 86(11):2278–2324 (1998)
- **DOI：** <https://doi.org/10.1109/5.726791>
- **入库日期：** 2026-09-21
- **一句话说明：** 用可学习卷积、下采样与全连接组成 **LeNet-5**，在手写字符上证明 **局部感受野 + 权值共享** 可由梯度端到端训练，是现代 CNN 的工程原型。

## 核心摘录（面向 wiki 编译）

### 1) 卷积与权值共享

- **要点：** 同一核在空间滑窗，参数量不随图幅线性爆炸；平移局部结构被直接写成归纳偏置。这比把像素拉平进 MLP 更适合图像。
- **对 wiki 的映射：** [`wiki/concepts/convolutional-neural-network.md`](../../wiki/concepts/convolutional-neural-network.md)、[`wiki/entities/lenet5.md`](../../wiki/entities/lenet5.md)

### 2) 分层特征：边缘 → 部件 → 字符

- **要点：** 浅层响应边缘与笔画，深层组合为字符部件。机器人视觉骨干仍按同一层次读特征（纹理 / 部件 / 语义）。
- **对 wiki 的映射：** [`wiki/concepts/vision-backbones.md`](../../wiki/concepts/vision-backbones.md)、[`wiki/overview/ai-architecture-map.md`](../../wiki/overview/ai-architecture-map.md)

### 3) 与后续残差 / 检测栈的关系

- **要点：** AlexNet/VGG/ResNet 是深度与正则化的扩展，不是换掉卷积几何。YOLO/FPN 检测头仍吃 CNN 特征图。
- **对 wiki 的映射：** [`wiki/entities/paper-resnet-deep-residual-learning.md`](../../wiki/entities/paper-resnet-deep-residual-learning.md)、[`wiki/methods/object-detection.md`](../../wiki/methods/object-detection.md)

## 开源状态（步骤 2.5）

- 1998 期刊长文，**无单一现代官方仓**；LeNet 示例遍布框架教程。实体页 [lenet5](../../wiki/entities/lenet5.md) 已覆盖教学实现。

## 当前提炼状态

- [x] 要点摘录与 wiki 映射
