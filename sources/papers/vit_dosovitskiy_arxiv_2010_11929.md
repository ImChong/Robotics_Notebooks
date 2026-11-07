# Vision Transformer：16×16 图像块即词元（arXiv:2010.11929）

> 论文来源归档（ingest）

- **标题：** An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
- **作者：** Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby（Google Research / Brain）
- **类型：** paper / computer-vision / transformer / backbone
- **arXiv：** <https://arxiv.org/abs/2010.11929> · PDF：<https://arxiv.org/pdf/2010.11929.pdf>
- **会议：** ICLR 2021
- **官方代码：** <https://github.com/google-research/vision_transformer>
- **入库日期：** 2026-11-07（首次入库 2026-09-21）
- **版本：** arXiv:2010.11929（ICLR 2021 版；本次 PDF 页眉为 Published as a conference paper at ICLR 2021）
- **核对范围：** 2026-11-07 下载 PDF（22 页）并通读：摘要、§1 引言、§2 相关工作、§3 方法（ViT、归纳偏置、混合结构、微调与高分辨率）、§4 实验（设置、Table 1–2 与 SOTA 对比、预训练数据需求 Fig. 3–4、缩放研究 Fig. 5、检查 ViT 内部 Fig. 6–7、自监督）、§5 结论；附录读了 A（多头自注意力）、B.1/B.1.1/B.1.2（训练、微调、掩码补丁自监督细节）、D.4 位置嵌入与轴向注意力部分，其余附录图表（D.1–D.3、D.5–D.7 的具体数值与 Table 6 逐模型结果）只浏览；参考文献仅浏览。未运行官方代码。
- **通读：** 已完成
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

## 通读核对结论（2026-11-07，对照论文全文）

- **结构（式 1–4）：** $x\in\mathbb R^{H\times W\times C}$ 重排为 $N=HW/P^2$ 个展平补丁 $x_p\in\mathbb R^{N\times(P^2C)}$，经可训练线性投影 $E$ 映射到恒定维度 $D$；前置可学习 [class] 嵌入（$z_0^0=x_{class}$），其编码器输出 $z_L^0$ 经 LayerNorm 作为图像表示 $y$；加**可学习 1D 位置嵌入**（作者未观察到 2D 感知位置嵌入有显著增益，附录 D.4）。编码器由交替的多头自注意力与 MLP（两层、GELU）组成，**LayerNorm 在每个块之前、残差在每个块之后**（Pre-LN）：$z'_\ell=\mathrm{MSA}(\mathrm{LN}(z_{\ell-1}))+z_{\ell-1}$，$z_\ell=\mathrm{MLP}(\mathrm{LN}(z'_\ell))+z'_\ell$。预训练头为单隐层 MLP，微调时换成零初始化的 $D\times K$ 线性层。
- **归纳偏置：** 只有 MLP 层是局部且平移等变的，自注意力是全局的；二维邻域结构只在切补丁和微调时插值位置嵌入时使用。更高分辨率微调保持补丁大小，序列变长，对预训练位置嵌入做 2D 插值。
- **模型（Table 1）：** Base 12 层/D=768/MLP 3072/12 头/86M；Large 24/1024/4096/16/307M；Huge 32/1280/5120/16/632M；ViT-L/16 表示 Large、补丁 16×16；序列长度与补丁边长平方成反比。混合模型用 CNN 特征图作补丁序列。
- **主结果（Table 2，JFT-300M 预训练）：** ViT-H/14：ImageNet 88.55、ReaL 90.72、CIFAR-100 94.55、VTAB(19) 77.63；ViT-L/16：87.76；BiT-L(ResNet152x4)：87.54；Noisy Student(EfficientNet-L2)：88.4/88.5。预训练算力 TPUv3-core-days：H/14 约 2.5k、L/16 约 0.68k、BiT-L 约 9.9k、Noisy Student 约 12.3k；ViT-L/16 在 ImageNet-21k 上预训练约需 8 核 TPUv3 约 30 天。原文提醒预训练效率还受训练计划、优化器、权重衰减等影响，故另做受控缩放研究。
- **数据需求（§4.3）：** ImageNet 预训练时 ViT-Large 不如 ViT-Base，且 ViT 低于 BiT ResNet；ImageNet-21k 时二者相近；只有 JFT-300M 才显出大模型优势。在 9M/30M/90M/300M 的 JFT 子集上，ViT 在小子集上比算力相近的 ResNet 更易过拟合，90M+ 才反超；原文结论“大规模训练胜过归纳偏置”。
- **缩放（§4.4，Fig. 5）：** ViT 在性能–算力权衡上约比 ResNet 省 2–4 倍预训练算力；混合模型在小算力下略优，大模型时差距消失；在试验范围内 ViT 未饱和。
- **检查内部（§4.5）：** 位置嵌入学出了 2D 拓扑（近处补丁相似、同行同列相似）；部分注意力头在最低层已关注大范围，另一些头在低层保持局部（混合模型里这种局部头不明显），注意力距离随深度增加。
- **自监督（§4.6、B.1.2）：** 掩码补丁预测（腐蚀 50% 补丁嵌入，预测 3-bit 平均色，共 512 色），ViT-B/16 达 79.9% ImageNet，比从头训练高约 2%，仍比有监督预训练低约 4%。
- **训练细节（§4.1、附录 B）：** 所有模型 Adam（β1=0.9，β2=0.999），batch 4096，权重衰减 0.1，线性 warmup 后线性衰减；微调用 SGD+动量，batch 512，ImageNet 微调分辨率 512（L/16）/518（H/14）并做 Polyak 平均 0.9999。
- **原文自述局限与展望：** 尚未验证检测与分割；自监督与大规模有监督预训练仍有明显差距；继续放大模型预计继续提升。
- **原文没有的内容：** 机器人/VLA 视觉塔、DINOv2/SigLIP、冻结编码器只训分类头（参数占比约 0.03%）等说法来自博客或本站其他页，并非本文结论。
