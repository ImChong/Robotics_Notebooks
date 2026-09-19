# GaussianGPT: Towards Autoregressive 3D Gaussian Scene Generation

> 来源归档（ingest）

- **标题：** GaussianGPT: Towards Autoregressive 3D Gaussian Scene Generation
- **类型：** paper
- **arXiv：** 2603.26661
- **出处：** ECCV 2026 Oral
- **项目页：** <https://nicolasvonluetzow.github.io/GaussianGPT/>
- **论文：** <https://arxiv.org/abs/2603.26661>
- **代码：** <https://github.com/nicolasvonluetzow/GaussianGPT>
- **Hugging Face Papers：** <https://huggingface.co/papers/2603.26661>
- **入库日期：** 2026-09-19
- **最后更新：** 2026-09-19
- **一句话说明：** 用稀疏 VQ-VAE 将 3D Gaussian 场景离散化为 token，GPT 式 transformer 自回归生成完整场景，并天然支持 completion 与 outpainting。

## 核心论文摘录

### 1) 问题与动机

- 近期 3D 生成多依赖 diffusion / flow-matching  holistic  refine；本文探索 **完全自回归** 路线，直接 next-token 预测 3D Gaussian 原语。
- **机构：** 慕尼黑工业大学（TU Munich）；作者 Nicolas von Lützow、Barbara Rössle、Katharina Schmid、Matthias Nießner。

### 2) 方法要点

- **压缩：** 稀疏 3D 卷积自编码器 + lookup-free vector quantization，将 per-voxel Gaussian 压成离散 latent grid。
- **序列化：** 固定 xyz 遍历顺序，位置 token 与特征 token 交错写入 1D 序列。
- **生成：** 带 3D rotary positional embedding 的因果 transformer；推理时可调 temperature / top-k / top-p。
- **任务统一：** 无条件生成、部分场景 conditioning 的 completion、重复 outpainting 扩展超出训练 horizon 的大场景。

### 3) 数据与工程

- 训练数据含 **PhotoShape**（物体级）、**ASE**、**3D-FRONT**（场景级）；Gaussian 由 Voxel-GS（L3DG 简化 Scaffold-GS）预处理。
- 两阶段训练：先 VQ-VAE，再 tokenize 数据集，最后训 GPT；配置由 Hydra 管理。

### 4) 开源与复现

- **开放程度：** **已开源** — 训练/推理代码、scene-level 与 PhotoShape checkpoint 均已发布（2026-09-19 项目页与 GitHub README 核查）。
- [`sources/repos/nicolasvonluetzow-gaussiangpt.md`](../repos/nicolasvonluetzow-gaussiangpt.md)
- [`sources/sites/gaussiangpt-project.md`](../sites/gaussiangpt-project.md)

## 对 wiki 的映射

- 实体页：[`wiki/entities/paper-sa-2603-26661-gaussiangpt.md`](../../wiki/entities/paper-sa-2603-26661-gaussiangpt.md)
- 方法交叉：[`wiki/methods/generative-world-models.md`](../../wiki/methods/generative-world-models.md)
