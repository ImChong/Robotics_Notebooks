# PointDiT（google-research/pointdit）

> 来源归档（repo）

- **标题：** PointDiT — Pixel-Space Diffusion for Monocular Geometry Estimation
- **类型：** repo / monocular-geometry / pixel-space-diffusion / icml2026
- **来源：** Google Research
- **链接：** <https://github.com/google-research/pointdit>
- **论文：** [arXiv:2607.02515](https://arxiv.org/abs/2607.02515) — [`sources/papers/pointdit_arxiv_2607_02515.md`](../papers/pointdit_arxiv_2607_02515.md)
- **项目页：** <https://haofeixu.github.io/pointdit/> — [`sources/sites/pointdit-github-io.md`](../sites/pointdit-github-io.md)
- **权重：** <https://huggingface.co/haofeixu/pointdit>
- **入库日期：** 2026-09-05
- **一句话说明：** 官方可运行仓：demo / 零样本评测 / 两阶段训练脚本齐全；checkpoint 不含门控 DINOv3。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-pointdit.md`](../../wiki/entities/paper-pointdit.md)

---

## 开源状态（步骤 2.5）

| 项 | 状态（2026-09-05 复核） |
|----|-------------------|
| 训练 / 推理代码 | **已开源**：`main.py` + `scripts/demo_*.sh` / `eval_*.sh` / `train_stage1_256_*.sh` / `train_stage2_512_*.sh` |
| 权重 | [haofeixu/pointdit](https://huggingface.co/haofeixu/pointdit) 六个 `*-nodinov3-*.pth`（B/L/H × 256/512） |
| DINOv3 | **门控、不随仓**：需申请 [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)，放到 `pretrained/dinov3/` 或设 `DINOV3_WEIGHTS_DIR` |
| 许可证 | Apache-2.0（`LICENSE`）；第三方见 `third_party/THIRD_PARTY_NOTICES` |
| 环境 | Python 3.12、PyTorch 2.7.0、CUDA 12.8（README） |
| 数据集 | [`DATASETS.md`](https://github.com/google-research/pointdit/blob/main/DATASETS.md)：12 个训练源 + 7 个零样本评测；点云不落盘，RGB-D + 内参在线反投影 |

**结论：** **已开源可运行**。最短路径是 demo：HF 权重 + DINOv3 门控权重 → `bash scripts/demo_l_512.sh`。找不到 encoder 时启动会直接报错，不会用随机初始化凑合。

## 仓库入口（对齐时序图）

| 路径 | 角色 |
|------|------|
| `main.py` | 训练 / 评测 / 野外图推理总入口 |
| `model.py` / `denoiser.py` | ViT 去噪器与点图 patch 化 |
| `engine.py` | Euler 采样、速度换算、评测循环 |
| `loss.py` | flow matching + 相对点损失 |
| `dataloader/` | RGB-D → 点图；`configs/res512mix.yaml` 为 Stage-2 混合 |
| `scripts/demo_l_512.sh` | 默认 3 步、PointDiT-L/16、输出 `generation/` 下深度 PNG + PLY |
| `scripts/eval_{b,l,h}_{256,512}.sh` | 复现论文表；脚本头注释写应对数字 |
| `MODELS.md` | checkpoint 文件名与 sha256 前缀 |

## 关联资料

- 论文：[`sources/papers/pointdit_arxiv_2607_02515.md`](../papers/pointdit_arxiv_2607_02515.md)
- 项目页：[`sources/sites/pointdit-github-io.md`](../sites/pointdit-github-io.md)
- Wiki：[wiki/entities/paper-pointdit.md](../../wiki/entities/paper-pointdit.md)
