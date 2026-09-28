# DiT（facebookresearch/DiT）

> 来源归档（repo）

- **标题：** Scalable Diffusion Models with Transformers — Official PyTorch Implementation
- **类型：** repo / generative-model / diffusion / transformer / imagenet
- **来源：** Meta FAIR（facebookresearch）
- **链接：** <https://github.com/facebookresearch/DiT>
- **论文：** [arXiv:2212.09748](https://arxiv.org/abs/2212.09748) — [`sources/papers/peebles_dit_arxiv_2212_09748.md`](../papers/peebles_dit_arxiv_2212_09748.md)
- **项目页：** <https://www.wpeebles.com/DiT> — [`sources/sites/dit-wpeebles-com.md`](../sites/dit-wpeebles-com.md)
- **入库日期：** 2026-09-28
- **一句话说明：** ImageNet 类条件 DiT 的官方实现：`models.py` 定义、`sample.py` 一键采样（权重自动下载）、`train.py` DDP 训练。
- **沉淀到 wiki：** [`wiki/entities/paper-dit-scalable-diffusion-transformers.md`](../../wiki/entities/paper-dit-scalable-diffusion-transformers.md)

---

## 开源状态（步骤 2.5）

| 项 | 状态（2026-09-28 复核） |
|----|-------------------------|
| 推理 | **已开源**：`sample.py`（256/512、CFG scale、步数、seed） |
| 训练 | **已开源**：`train.py`（PyTorch DDP；需 ImageNet 与 VAE  latent） |
| 权重 | **已发布**：`DiT-XL-2-256x256.pt` / `DiT-XL-2-512x512.pt`（README 表 + 直链） |
| 环境 | `environment.yml`（Conda；CPU-only 可删 CUDA 依赖） |
| 生态 | Hugging Face `diffusers` DiT pipeline（README 外链） |

**结论：** **已开源可运行。** 最短复现：`conda env create -f environment.yml` → `python sample.py --image-size 512 --seed 1`。

## 仓库入口（对齐时序图）

| 路径 | 角色 |
|------|------|
| `models.py` | DiT 模型定义（S/B/L/XL × patch 2/4/8） |
| `sample.py` | 预训练或 `--ckpt` 自定义权重采样 |
| `train.py` | ImageNet 类条件训练（EMA、日志） |
| `run_DiT.ipynb` | Colab 采样示例 |
| `environment.yml` | 依赖与 PyTorch/CUDA 版本 |

## 关联资料

- 论文：[`sources/papers/peebles_dit_arxiv_2212_09748.md`](../papers/peebles_dit_arxiv_2212_09748.md)
- 项目页：[`sources/sites/dit-wpeebles-com.md`](../sites/dit-wpeebles-com.md)
- Wiki：[wiki/entities/paper-dit-scalable-diffusion-transformers.md](../../wiki/entities/paper-dit-scalable-diffusion-transformers.md)
