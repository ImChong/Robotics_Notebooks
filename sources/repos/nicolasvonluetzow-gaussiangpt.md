# GaussianGPT（nicolasvonluetzow/GaussianGPT）

- **URL：** <https://github.com/nicolasvonluetzow/GaussianGPT>
- **项目页：** <https://nicolasvonluetzow.github.io/GaussianGPT/>
- **关联论文：** [gaussiangpt_arxiv_2603_26661.md](../papers/gaussiangpt_arxiv_2603_26661.md)
- **权重：** <https://kaldir.vc.cit.tum.de/gaussiangpt/>（scene-level 与 PhotoShape 物体级 VQ-VAE + GPT 成对发布）

## 一句话说明

ECCV 2026 Oral：稀疏 VQ-VAE 将 3D Gaussian 场景离散化为 token 流，GPT 式 transformer 自回归生成、补全与 outpainting 的完整训练/推理实现。

## 运行时入口（README 对齐）

| 阶段 | 脚本 | 作用 |
|------|------|------|
| VQ-VAE 训练 | `train_ae.py` | 稀疏 3D CNN + 向量量化；`gsplat` 重渲染监督 |
| 数据集 token 化 | `tokenize_dataset.py` | 冻结编码器写出 per-scene token 流 |
| GPT 训练 | `train_gpt.py` | 3D RoPE 因果 transformer 建模 token 序列 |
| 单块/物体采样 | `generate_chunks.py` | 无条件或 sequence-prefix completion |
| 空间补全 | `complete_chunks.py` | 按空间范围 prompt 的 inpainting/outpainting |
| 大场景拼接 | `generate_scene.py` + `decode_scene.py` | 多块 tile 后解码为可渲染 Gaussian |

Hydra 配置位于 `conf/`；依赖含 MinkowskiEngine、gsplat、pytorch3d、Flash Attention。

## 交叉链接

- [paper-sa-2603-26661-gaussiangpt 论文实体](../../wiki/entities/paper-sa-2603-26661-gaussiangpt.md)
- [项目页归档](../sites/gaussiangpt-project.md)
