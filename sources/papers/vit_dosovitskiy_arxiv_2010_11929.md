# Vision Transformer：16×16 图像块即词元（arXiv:2010.11929）

> 论文来源归档（ingest）

- **标题：** An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
- **作者：** Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby（Google Research / Brain）
- **类型：** paper / computer-vision / transformer / backbone
- **arXiv：** <https://arxiv.org/abs/2010.11929> · PDF：<https://arxiv.org/pdf/2010.11929.pdf>
- **会议：** ICLR 2021
- **官方代码：** <https://github.com/google-research/vision_transformer>
- **入库日期：** 2026-09-21
- **一句话说明：** 把图像切成固定 patch 当作 token，用纯 Transformer 编码器做大规模图像识别，证明 **弱归纳偏置 + 大数据** 可达到或超过 CNN。

## 核心摘录（面向 wiki 编译）

### 1) 图像块 = 词元

- **要点：** 标准 ViT 将 \(H\times W\) 图分成 \(P\times P\) patch，线性嵌入后加位置编码，送入与 NLP 同构的编码器；分类用 class token。
- **对 wiki 的映射：** [`wiki/concepts/vision-transformer.md`](../../wiki/concepts/vision-transformer.md)

### 2) 数据规模决定能否打败 CNN

- **要点：** 在中小数据上 ViT 不如 ResNet 等强归纳偏置模型；在 JFT 级大规模预训练后迁移 ImageNet，精度与计算效率超过同类 CNN。
- **对 wiki 的映射：** [`wiki/comparisons/cnn-vs-vit-backbones.md`](../../wiki/comparisons/cnn-vs-vit-backbones.md)、[`wiki/concepts/vision-backbones.md`](../../wiki/concepts/vision-backbones.md)

### 3) 对机器人视觉塔的后果

- **要点：** DINOv2 / SigLIP / VLA 视觉编码器默认走 ViT 族，是因为 **与语言 Transformer 同构**，不是因为机载检测一定更快。实时闭环仍常选 CNN。
- **对 wiki 的映射：** [`wiki/overview/hub-vision-backbone.md`](../../wiki/overview/hub-vision-backbone.md)、[`wiki/methods/vla.md`](../../wiki/methods/vla.md)、[`wiki/overview/ai-architecture-map.md`](../../wiki/overview/ai-architecture-map.md)

## 开源状态（步骤 2.5）

- 官方仓 `google-research/vision_transformer` **已开源**（Apache-2.0），含 JAX 实现与预训练权重说明。

## 当前提炼状态

- [x] 要点摘录与 wiki 映射
