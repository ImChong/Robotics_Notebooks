# SENet 与 DANet：通道与空间注意力经典原论文簇

> 来源归档（ingest · 原文通读）

- **论文 1：** *Squeeze-and-Excitation Networks*（Jie Hu, Li Shen, Gang Sun, CVPR 2018 / arXiv:[1709.01507](https://arxiv.org/abs/1709.01507)）
- **论文 2：** *Dual Attention Network for Scene Segmentation*（Jun Fu et al., CVPR 2019 / arXiv:[1809.02983](https://arxiv.org/abs/1809.02983)）
- **通读：** 已完成
- **核对范围：** 已通读 `arXiv:1709.01507v4`（67,825 字符，Squeeze global avg pooling + Excitation bottleneck MLP + channel-wise scale 重标定）与 `arXiv:1809.02983v4`（39,198 字符，Position Attention Module 空间非局部自注意 + Channel Attention Module 通道亲和矩阵 + Cityscapes 75.74% mIoU 等消融）。

## 核心摘要

- SENet 提出仅增加极少参数量即插即用的 SE Block，通过 Squeeze–Excitation 重标定通道响应；DANet 在扩张 FCN 上并联位置注意（PAM）与通道注意（CAM）以显式捕获长程空间与通道依赖。

## 对 wiki 的映射

- [`wiki/methods/channel-spatial-attention.md`](../../wiki/methods/channel-spatial-attention.md)
