# DiT：用 Transformer 做可扩展扩散骨干（arXiv:2212.09748）

> 论文来源归档（ingest）

- **标题：** Scalable Diffusion Models with Transformers
- **作者：** William Peebles, Saining Xie
- **类型：** paper / generative-model / diffusion / transformer
- **arXiv：** <https://arxiv.org/abs/2212.09748> · PDF：<https://arxiv.org/pdf/2212.09748.pdf>
- **会议：** ICCV 2023
- **项目页：** <https://www.wpeebles.com/DiT> — [`sources/sites/dit-wpeebles-com.md`](../sites/dit-wpeebles-com.md)
- **官方代码：** <https://github.com/facebookresearch/DiT> — [`sources/repos/facebookresearch-dit.md`](../repos/facebookresearch-dit.md)
- **入库日期：** 2026-11-07（首次入库 2026-09-21（项目页 / 仓库归档深化：2026-09-28））
- **版本：** arXiv:2212.09748v2（2023-03-02）；ICCV 2023
- **核对范围：** 2026-11-07 下载 arXiv v2 PDF（25 页）并通读全文：摘要、§1 引言、§2 相关工作、§3 Diffusion Transformers（DDPM 预备、classifier-free guidance、LDM、patchify、四种 DiT block、模型规模、解码器）、§4 实验设置、§5 实验（block 设计、规模扫描、Gflops 相关性、SOTA 对比 Table 2/3、采样算力 vs 模型算力）、§6 结论，以及附录 A–D（实现细节、样本、额外缩放结果、VAE 解码器消融、Table 4–6）；参考文献仅浏览。未运行官方代码；样本图只读图注。
- **通读：** 已完成
- **一句话说明：** 在隐空间扩散里用 **patch 化 Transformer** 替换 U-Net，展示清晰的 **Gflops–FID scaling**，成为后续文生图与机器人 flow/DiT 动作头的骨干模板。

## 核心摘录（面向 wiki 编译）

### 1) 扩散骨干可以不是 U-Net

- **要点：** 潜张量被切成 patch token，经 AdaLN 等调制注入时间步/类别条件，Transformer 预测噪声。U 形跳跃不再是必要条件。
- **对 wiki 的映射：** [`wiki/concepts/diffusion-transformer.md`](../../wiki/concepts/diffusion-transformer.md)、[`wiki/concepts/diffusion-model.md`](../../wiki/concepts/diffusion-model.md)

### 2) 前向算力与样本质量可预测缩放

- **要点：** 在控制变量下，更高 Gflops 的 DiT 对应更好 FID。这给「加大动作头」提供了图像域先例；机器人侧仍须另测延迟。
- **对 wiki 的映射：** [`wiki/concepts/diffusion-transformer.md`](../../wiki/concepts/diffusion-transformer.md)、[`wiki/overview/ai-architecture-map.md`](../../wiki/overview/ai-architecture-map.md)

### 3) 机器人 VLA 为何爱 DiT

- **要点：** 动作块是短序列，天然适合 Transformer token；与 VLM 骨干同族，便于共享工具链。Xiaomi / π / GR00T 等公开 VLA 的连续动作头大量写 **DiT + flow matching**。
- **对 wiki 的映射：** [`wiki/methods/vla.md`](../../wiki/methods/vla.md)、[`wiki/methods/diffusion-policy.md`](../../wiki/methods/diffusion-policy.md)

## 开源状态（步骤 2.5，2026-09-28 再核）

| 资源 | 状态 |
|------|------|
| [wpeebles.com/DiT](https://www.wpeebles.com/DiT) | 链到 GitHub + 可视化；非空壳 |
| [facebookresearch/DiT](https://github.com/facebookresearch/DiT) | **已开源**（`sample.py` / `train.py` / XL/2 权重直链） |
| HF Demo | [wpeebles/DiT Space](https://huggingface.co/spaces/wpeebles/DiT) |

## 当前提炼状态

- [x] 要点摘录与 wiki 映射
- [x] 升格深度实体：[`wiki/entities/paper-dit-scalable-diffusion-transformers.md`](../../wiki/entities/paper-dit-scalable-diffusion-transformers.md)

## 通读核对结论（2026-11-07，对照 arXiv v2 全文）

- **框架：** 在 LDM 中用冻结的 Stable Diffusion 卷积 VAE（下采样 8×，256×256×3 → 32×32×4 latent）；DiT 只替换去噪骨干。扩散超参沿用 ADM（$t_{max}=1000$，线性方差表 $10^{-4}\to2\times10^{-2}$，学习方差 $\Sigma_\theta$，$L_{simple}$ 训练 $\epsilon_\theta$，完整 $L$ 训练 $\Sigma_\theta$）。
- **Patchify：** latent 切为 $p\times p$ patch，token 数 $T=(I/p)^2$，加 sine-cosine 位置编码；$p$ 减半 token 约 ×4、Gflops 至少 ×4，参数量基本不变；设计空间取 $p=2,4,8$。
- **四种 block：** in-context（t、c 作两个额外 token，约无额外 Gflops）、cross-attention（约 +15% Gflops）、adaLN（由 t 与 c 嵌入之和回归 γ、β，Gflops 最少，对所有 token 施加同一函数）、adaLN-Zero（另回归 α 并零初始化，使 block 初始为恒等）。XL/2 在 400K 步，adaLN-Zero 的 FID 约为 in-context 的一半，且优于 adaLN、cross-attention；此后所有模型用 adaLN-Zero。Table 4 无引导 FID-50K（400K 步）：in-context 35.24、cross-attention 26.14、adaLN 25.21、adaLN-Zero 19.47。
- **规模（Table 1）：** S/B/L/XL 为 12/12/24/28 层，隐藏维 384/768/1024/1152，6/12/16/16 头；Gflops（I=32，p=4）1.4/5.6/19.7/29.1；12 个模型覆盖 0.3–118.6 Gflops。
- **训练：** AdamW，学习率 1e-4 常数，无权重衰减，batch 256，仅水平翻转；不需要 warmup/正则，未见 loss 尖峰；EMA 衰减 0.9999；JAX + TPU v3-256，XL/2 约 5.7 it/s。
- **Gflops 与 FID：** 400K 步时 Gflops 与 FID-50K 相关系数 −0.93（Fig. 8）；参数量不能唯一决定质量（保持模型大小减小 patch，参数量不增反略减，FID 仍显著改善）；DiT-S/2 与 DiT-B/4 Gflops 相近、FID 相近。较大模型训练算力利用更高效；加大采样步数不能补足模型算力（L/2 用 1000 步 80.7 Tflops，FID-10K 25.9，劣于 XL/2 用 128 步 15.2 Tflops 的 23.7）。
- **SOTA：** ImageNet 256×256，DiT-XL/2 训练 7M 步，cfg=1.5 时 FID-50K 2.27（sFID 4.60，IS 278.24，Precision 0.83，Recall 0.57），优于 LDM-4-G 3.60 与 StyleGAN-XL 2.30；cfg=1.25 时 3.22；无引导 9.62；2.35M 步时 2.55。ImageNet 512×512，XL/2 训练 3M 步，cfg=1.5 时 3.04（前最好 ADM-G,ADM-U 3.85）；XL/2 512 为 524.6 Gflops（1024 token）。DiT-XL/2 118.6 Gflops，对比 LDM-4 约 104（Table 6）、ADM 1120、ADM-U 742。
- **注意：引导只用于前三个通道**（附录 A）：cfg 报告值仅对 latent 前 3 通道施加；四通道 1.375 约等价三通道 1.5（FID 2.20 vs 2.27）。cfg 与采样步数：FID-50K 用 250 步 DDPM。
- **VAE 解码器消融（Table 5，附录 D）：** original 2.46、ft-MSE 2.30、ft-EMA 2.27；最终表用 ft-EMA。参数与 Gflops 不含 84M 参数的 VAE。
- **原文没有的内容：** 无文本条件、无机器人实验；摘要/总结中的“VLA 动作头常用 DiT + flow matching”不是本文结论，须以 VLA 论文自身核实（见 wiki 页标注）。官方 `sample.py` 的 cfg 默认值是仓库实现细节，不在论文里。
