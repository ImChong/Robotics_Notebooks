# DiT 项目页（wpeebles.com/DiT）

> 来源归档（site）

- **标题：** Scalable Diffusion Models with Transformers（DiT）
- **类型：** site（论文项目页 + 可视化）
- **URL：** <https://www.wpeebles.com/DiT>
- **论文：** [arXiv:2212.09748](https://arxiv.org/abs/2212.09748) — [`sources/papers/peebles_dit_arxiv_2212_09748.md`](../papers/peebles_dit_arxiv_2212_09748.md)
- **会议：** ICCV 2023（Oral）
- **作者：** William Peebles（UC Berkeley）、Saining Xie（NYU）
- **入库日期：** 2026-09-28
- **一句话说明：** 官方项目页：用 **ViT 式 patch Transformer + AdaLN** 替换 LDM 的 U-Net，展示 **Gflops–FID** 缩放与 ImageNet 256/512 SOTA 样本。

## 开源核查（步骤 2.5，2026-09-28）

| 资源 | 状态 | 说明 |
|------|------|------|
| 训练 / 采样代码 | **已开源** | [facebookresearch/DiT](https://github.com/facebookresearch/DiT)（`train.py` / `sample.py` / `models.py`） |
| 预训练权重 | **已发布** | README 直链 `dl.fbaipublicfiles.com/DiT/models/`（256/512 XL/2） |
| 在线 Demo | **已发布** | [HF Spaces wpeebles/DiT](https://huggingface.co/spaces/wpeebles/DiT)、Colab `run_DiT.ipynb` |
| 项目页 Code 按钮 | **链到 GitHub** | 页脚与 README 互指，非「待发布」 |

**判定：已开源可复现。** 图像生成最短路径为 `sample.py`；全量 ImageNet 训练需自备数据与 DDP 环境（见仓库 `environment.yml`）。

## 公开要点（编译自项目页，2026-09-28）

### 核心主张

- 在 **隐空间 LDM** 框架内，用 **patch 化 Transformer** 作去噪骨干（DiT），结构接近标准 ViT。
- **条件注入：** 实验多种 block 设计后，**AdaLN 调制 + 残差前 scale/shift**（初始化接近恒等）效果最好。
- **缩放轴：** (1) 模型规模 DiT-S/B/L/XL（约 33M–675M 参数，0.4–119 Gflops）；(2) **patch 大小** 2/4/8 — patch 越小 token 越多、Gflops 越高，**算力（Gflops）比参数量更能预测 FID**。

### 代表性结果（class-conditional ImageNet，CFG）

| 设置 | FID-50K（项目页 / 论文） | 备注 |
|------|--------------------------|------|
| DiT-XL/2 @ 256×256 | **2.27** | 优于此前 LDM 等扩散基线 |
| DiT-XL/2 @ 512×512 | **3.04** | 相对 ADM-U 等 U-Net 扩散更省 Gflops |
| XL/8 vs XL/2 | XL/8 FID 差 | 参数量略多但 Gflops 远低，说明 **算力** 是关键 |

### 可视化内容

- 固定噪声下 **scaling 对比**（12 个配置 @ 400K steps）
- **隐空间 slerp**（DDIM）与 **类别 embedding 插值**（犬种、海洋、鸟类等）

## 关联资料

- 论文归档：[`sources/papers/peebles_dit_arxiv_2212_09748.md`](../papers/peebles_dit_arxiv_2212_09748.md)
- 代码归档：[`sources/repos/facebookresearch-dit.md`](../repos/facebookresearch-dit.md)
- Wiki 实体：[`wiki/entities/paper-dit-scalable-diffusion-transformers.md`](../../wiki/entities/paper-dit-scalable-diffusion-transformers.md)
