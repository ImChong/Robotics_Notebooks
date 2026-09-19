# GaussianGPT 项目页

- **URL：** <https://nicolasvonluetzow.github.io/GaussianGPT/>
- **论文：** *GaussianGPT: Towards Autoregressive 3D Gaussian Scene Generation*（[arXiv:2603.26661](https://arxiv.org/abs/2603.26661)）
- **出处：** ECCV 2026 Oral
- **机构：** 慕尼黑工业大学（TU Munich）
- **入库日期：** 2026-09-19
- **代码：** <https://github.com/nicolasvonluetzow/GaussianGPT>
- **Hugging Face Papers：** <https://huggingface.co/papers/2603.26661>
- **关联 source：** [`sources/papers/gaussiangpt_arxiv_2603_26661.md`](../papers/gaussiangpt_arxiv_2603_26661.md)
- **关联 wiki：** [`wiki/entities/paper-sa-2603-26661-gaussiangpt.md`](../../wiki/entities/paper-sa-2603-26661-gaussiangpt.md)

## 开源核查（步骤 2.5）

| 状态 | 说明 |
|------|------|
| **已开源** | 项目页链至 GitHub；README 写明 2026-06-19 发布训练/推理代码，2026-07-01 发布 scene-level VQ-VAE 与 GPT checkpoint，2026-07-12 发布 PhotoShape 物体级 checkpoint。权重托管于 `kaldir.vc.cit.tum.de/gaussiangpt`。 |

## 方法要点（项目页摘录）

1. **3D Gaussian 压缩：** 将 Gaussian 原语投影到稀疏 3D 体素网格，经稀疏 3D 卷积编码器压缩为低维 latent grid，lookup-free quantization 离散化为 codebook 索引；对称解码器用渲染、占用与 codebook 熵损失端到端训练。
2. **自回归生成：** 量化 latent grid 按固定 xyz 顺序序列化为 1D token（位置 token 与特征 token 交错）；带 3D RoPE 的 GPT 式因果 transformer 做 next-token 预测。
3. **能力：** 无条件生成、以部分场景为 prompt 的 completion、重复 outpainting 扩展大场景；同一模型与采样机制，温度可控。

## 交叉链接

- 论文归档：[gaussiangpt_arxiv_2603_26661.md](../papers/gaussiangpt_arxiv_2603_26661.md)
- 代码归档：[nicolasvonluetzow-gaussiangpt.md](../repos/nicolasvonluetzow-gaussiangpt.md)
