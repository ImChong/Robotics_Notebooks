# ViT 论文通读（Transformer 视觉应用课程视角）

> 来源归档（原文通读 · 支撑 `wiki/entities/transformer-cv-curriculum.md`）

- **标题：** An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
- **作者：** Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai 等（Google Research, Brain Team）
- **类型：** paper / computer-vision / transformer
- **原文：** <https://arxiv.org/abs/2010.11929>（2026-11 通过 arXiv PDF v2 获取全文，ICLR 2021）
- **一句话说明：** 把图像切成 patch 当词元，用标准 Transformer 编码器做分类；大规模预训练下可追平或超过 CNN。是课程第 2 章 Transformer 分类（ViT/TNT/CvT）的共同起点。
- **核对范围：** 通读正文 §1–§5（方法 §3、实验 §4.1–§4.6、结论）及附录 A（MSA 公式）、B（训练与微调细节、掩码 patch 自监督）、D.2（形状缩放）、D.3（head 与 class token）、D.5（实测算力）；附录 C 详细结果表及 D.1、D.4、D.6–D.10 的图表只浏览，未逐项核对。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **方法（§3.1，式 1–4）：** 图像 x∈R^{H×W×C} 切成 N=HW/P² 个 P×P patch，展平后经线性投影 E 映射到 D 维，前置可学习 [class] 嵌入，加可学习 1D 位置嵌入；编码器为交替的 MSA 与 MLP（GELU）块，每块前 LayerNorm、后残差。
- **归纳偏置（§3.1）：** 仅 MLP 层局部且平移等变，自注意力为全局；2D 结构只在切 patch 与微调时插值位置嵌入两处引入。
- **混合结构：** 可把 CNN 特征图当作 patch 序列输入 Transformer。
- **变体（表 1）：** ViT-B 12 层/768/86M，ViT-L 24 层/1024/307M，ViT-H 32 层/1280/632M；序列长度与 patch 边长平方成反比。
- **微调（§3.2）：** 换成零初始化的 D×K 线性头；更高分辨率时保持 patch 大小，对预训练位置嵌入做 2D 插值。
- **主结果（§4.2，表 2）：** JFT-300M 预训练的 ViT-H/14 在 ImageNet 88.55%、CIFAR-100 94.55%、VTAB 77.63%；预训练算力 2.5k TPUv3-core-days，BiT-L 为 9.9k、Noisy Student 为 12.3k。
- **数据需求（§4.3）：** 仅 ImageNet 预训练时 ViT-L 不如 ViT-B 且逊于 BiT；ImageNet-21k 时相近；JFT-300M 时大模型优势显现。小数据上 ViT 比同算力 ResNet 更易过拟合，作者归因于缺少卷积归纳偏置。
- **缩放（§4.4）：** 同等性能下 ViT 约比 ResNet 少用 2–4 倍算力；混合模型仅在小算力下略优；ViT 在所试范围内未饱和。
- **分析（§4.5）：** 位置嵌入自行学到 2D 距离结构；部分头在低层即全局注意，另一些保持局部。
- **自监督（§4.6，附录 B.1.2）：** 掩码 patch 预测，ViT-B/16 达 79.9% ImageNet，比从头训练高 2%，仍比有监督预训练低 4%。
- **结论与展望：** 待解决检测、分割等其他视觉任务，以及自监督预训练。
- **附录 D.2：** 形状缩放中加深收益最大，加宽收益最小；缩小 patch 带来稳健提升。

## 对 wiki 的映射

- [`wiki/entities/transformer-cv-curriculum.md`](../../wiki/entities/transformer-cv-curriculum.md)
- [`wiki/concepts/vision-transformer.md`](../../wiki/concepts/vision-transformer.md)

## 参考来源

- 论文：<https://arxiv.org/abs/2010.11929>
