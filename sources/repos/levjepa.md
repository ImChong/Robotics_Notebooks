# MLO-lab/LeVJEPA

> 来源归档

- **标题：** LeVJEPA（官方实现）
- **类型：** repo
- **组织：** MLO-lab（DKFZ / Goethe 机器学习与优化组）
- **代码：** <https://github.com/MLO-lab/LeVJEPA>
- **License：** MIT（`module.py` 除外，**CC BY-NC 4.0**，改编自 [facebookresearch/jepa](https://github.com/facebookresearch/jepa)）
- **论文：** <https://arxiv.org/abs/2608.27395>
- **项目页：** <https://levjepa.github.io/>
- **权重：** [`galilai-group/LeVJEPA-VideoMix-Large`](https://huggingface.co/galilai-group/LeVJEPA-VideoMix-Large)（CC BY-NC 4.0）
- **入库日期：** 2026-09-04
- **一句话说明：** LeJEPA 目标的视频预训练官方仓：Hydra/`main.py` 预训练、Walking Tours 下载与 Lance 编码、特征可视化 notebook，以及 HF 推理入口。

## 开源核查（2026-09-04）

| 项 | 状态 |
|----|------|
| 代码 | **已开源** · 主体 MIT |
| 预训练权重 | ViT-L/16 VideoMix-Large 公开（HF） |
| 默认数据 | Walking Tours（HF 只给 YouTube URL，需自下） |
| Demo | `notebooks/feature_visualization.ipynb`、`notebooks/training_workshop.ipynb` |
| 备注 | 发布权重是 encoder 的 **EMA 副本**（只作评测 checkpoint，不进目标） |

## 入口速查（对齐 README）

| 路径 / 命令 | 作用 |
|-------------|------|
| `uv sync` | 训练环境（Linux CUDA 12.8 wheels） |
| `uv sync --extra data` | 加 decord / yt-dlp，仅建数据集需要 |
| `uv sync --extra notebook` | 可视化 / workshop notebook |
| `bash scripts/download_walking_tours.sh` | 下载 10 条 Walking Tours（~25 GB） |
| `uv run python scripts/build_lance_walking_tours.py --workers 16` | 编 15 fps Lance 库（~33 GB） |
| `sbatch slurm/train_walking_tours_vitb.slurm` | 2×8 GPU 默认 ViT-B 预训练 |
| `uv run python main.py trainer.devices=1 ...` | 单卡冒烟（有效 batch 已变，不是论文配方） |
| `AutoModel.from_pretrained("galilai-group/LeVJEPA-VideoMix-Large", trust_remote_code=True)` | 冻结特征提取 |
| `notebooks/feature_visualization.ipynb` | patch PCA / query-patch 余弦图 |

**推理注意（HF 卡）：** 输入 `(B, C, T, H, W)`，ImageNet normalize，默认 `attn_mode=block_causal`——改成 full attention **不会报错但特征变差**。单图沿时间轴重复 16 帧。训练时 ~7.5 fps，16 帧约 2 秒。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [LeVJEPA](../../wiki/entities/paper-levjepa.md) | 实体归纳：无启发式视频预训练 |
| [V-JEPA 2](../../wiki/entities/paper-vjepa2.md) | 同数据重训基线；EMA + predictor 路线 |
| [WCM](../../wiki/entities/paper-wcm-world-critic-model.md) | 同 SIGReg / LeJEPA 目标，落点在 VLA critic |
| [world-model-physics-fidelity-outputs](../../wiki/overview/world-model-physics-fidelity-outputs.md) | 因果表征是 latent 规划前置，本文无 AC |

## 对 wiki 的映射

- 论文：[`sources/papers/levjepa_arxiv_2608_27395.md`](../papers/levjepa_arxiv_2608_27395.md)
- 项目页：[`sources/sites/levjepa-github-io.md`](../sites/levjepa-github-io.md)
- 沉淀 **[`wiki/entities/paper-levjepa.md`](../../wiki/entities/paper-levjepa.md)**
