# ResNet 论文通读（视觉骨干视角）

> 来源归档（原文通读 · 支撑 `wiki/concepts/vision-backbones.md`）

- **标题：** Deep Residual Learning for Image Recognition
- **作者：** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun（Microsoft Research）
- **类型：** paper / computer-vision / backbone
- **原文：** <https://arxiv.org/abs/1512.03385>（2026-11 通过 arXiv PDF v1 获取全文）
- **一句话说明：** 用残差映射 F(x)+x 与无参恒等捷径解决极深网络的退化问题，152 层 ResNet 在 ImageNet 分类与检测迁移上取得当时最佳，成为后续视觉骨干的默认底座。
- **核对范围：** 通读正文 §1–§4（引言、相关工作、§3.1–3.4 方法与实现、§4.1 ImageNet、§4.2 CIFAR-10 与层响应分析、§4.3 检测）及附录 A（检测基线）、B（竞赛检测改进）、C（ImageNet 定位）；参考文献表与各 Table 的逐格数字仅核对了正文引用的部分。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **退化问题（§1，图 1）：** 网络加深到一定程度后精度饱和并迅速下降，且训练误差也升高，不是过拟合；存在"新增层取恒等映射"的构造解，说明是优化器难以找到好解。BN 与规范初始化已使数十层网络能起步收敛，退化是另一回事。
- **残差学习（§3.1，式 1–2）：** 让堆叠层拟合 F(x):=H(x)−x，输出 y=F(x,{Wi})+x；若恒等最优，只需把残差推向零。维度不一致时用线性投影 Ws。作者明确说"最优映射更接近恒等"是预条件假设，并在图 7 用层响应标准差佐证（ResNet 响应更小，越深越小）。
- **架构（§3.3，表 1，图 3）：** plain 网络遵循 VGG 式规则（同尺寸同滤波器数；尺寸减半则滤波器翻倍；stride-2 卷积下采样；全局平均池化 + 1000 路全连接）。34 层为 3.6 B FLOPs，仅为 VGG-19（19.6 B）的 18%。shortcut 选项：A 零填充、B 仅升维时投影、C 全投影；A/B/C 差别很小，故不必全用投影。
- **Bottleneck（§4.1，图 5）：** 1×1 降维、3×3、1×1 恢复；恒等捷径对 bottleneck 尤其重要，换成投影会使时间复杂度与模型大小翻倍。ResNet-50 为 3.8 B FLOPs，ResNet-152 为 11.3 B，仍低于 VGG-16/19（15.3 / 19.6 B）。
- **实现（§3.4）：** 短边随机缩放到 [256,480]、224×224 随机裁剪 + 翻转；卷积后、激活前加 BN；SGD，batch 256，学习率 0.1 平台期除以 10，权重衰减 1e-4，动量 0.9，不使用 dropout。
- **ImageNet 结果（§4.1）：** 34 层 plain 比 18 层 plain 误差更高；34 层 ResNet 比 18 层 ResNet 好 2.8%，比 34 层 plain top-1 低 3.5%；18 层时两者精度相近但 ResNet 收敛更快。作者认为 plain 网络的困难不是梯度消失（BN 保证信号方差，梯度范数正常），推测可能是指数级低的收敛速度（原文称为 conjecture）。ResNet-152 单模型 top-5 验证误差 4.49%，六模型集成测试集 3.57%，ILSVRC 2015 分类第一。
- **CIFAR-10（§4.2，表 6）：** 20/32/44/56/110 层 ResNet 随深度变好；110 层 6.43% 测试误差；1202 层训练误差 <0.1% 但测试误差 7.93% 差于 110 层，作者归因于对小数据集过大（19.4M 参数）而过拟合，且未用 maxout/dropout 等强正则。
- **检测迁移（§4.3，表 7–8）：** 同一 Faster R-CNN 下把 VGG-16 换成 ResNet-101：VOC07 73.2→76.4、VOC12 70.4→73.8 mAP；COCO mAP@[.5,.95] 21.2→27.2（+6.0，相对 28%），mAP@.5 41.5→48.4。BN 统计量在检测微调时冻结（附录 A，主要为省显存）。
- **竞赛改进（附录 B）：** 在 ResNet-101 基线上，box refinement、全局上下文、多尺度测试、集成逐步叠加；ImageNet DET 集成 62.1% mAP，领先第二名 8.5 个点。
- **定位（附录 C）：** 采用 per-class RPN + RoI-centric R-CNN；单模型 top-5 定位误差 10.6%，集成测试集 9.0%，比 ILSVRC14 相对降 64%。
- **论文自述的开放点：** 深度对残差的假设（多层非线性层能逼近残差函数）仍是开放问题（脚注 2）；plain 网络优化困难的原因留待未来研究；超深网络需更强正则。

## 对 wiki 的映射

- [`wiki/concepts/vision-backbones.md`](../../wiki/concepts/vision-backbones.md)
- [`wiki/entities/paper-resnet-deep-residual-learning.md`](../../wiki/entities/paper-resnet-deep-residual-learning.md)

## 参考来源

- 论文：<https://arxiv.org/abs/1512.03385>
