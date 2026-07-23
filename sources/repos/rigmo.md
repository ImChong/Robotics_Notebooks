# RigMo（haoz19/RigMo）

> 来源归档

- **标题：** RigMo — Unifying Rig and Motion Learning for Generative Animation（训练代码）
- **类型：** repo
- **组织 / 作者：** Hao Zhang 等（Snap / UIUC 等）；GitHub `haoz19`
- **代码：** <https://github.com/haoz19/RigMo>
- **项目页：** <https://haoz19.github.io/RigMo-page/>
- **论文：** <https://arxiv.org/abs/2601.06378>
- **数据集：** <https://huggingface.co/datasets/haoz19/RigMo-data>（gated）
- **License：** CC BY-NC 4.0（非商业研究）
- **入库日期：** 2026-07-23
- **一句话说明：** 自包含的 **RigMo-VAE** 训练管线（含 temporal-attention 变体）；基于 Step1X-3D geometry 框架；**不含** 论文中的 Motion-DiT 生成阶段。
- **沉淀到 wiki：** [RigMo（实体页）](../../wiki/entities/rigmo.md)

## 开源状态（2026-07-23 核查）

| 项 | 结论 |
|----|------|
| 仓库 | 公开；Python 3.10 / PyTorch 2.5 |
| 可运行内容 | **是（VAE）** — `train.py` + `scripts/train_single_node.sh` / SLURM；`configs/rigmo_vae_temporal*.yaml` |
| 未发布 | **Motion-DiT**（README 明示） |
| 数据 | HF gated ~18,985 序列 / ~534k 帧；需 `huggingface-cli` + zstd 解压至 `data/rigmo_data/` |
| 关键文件 | `step1x3d_geometry/models/autoencoders/mesh_motion_vae.py`、`systems/mesh_motion_autoencoder.py`、`datamodules/mesh_motion.py` |

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [RigMo 论文归档](../papers/rigmo_arxiv_2601_06378.md) | 方法与评测来源 |
| [Diffusion-based Motion Generation](../../wiki/methods/diffusion-motion-generation.md) | 结构感知潜空间 → 下游 DiT（论文有、仓无） |
| [Blender / DCC](../../wiki/entities/blender.md) | 对照：RigMo 产出可动画 mesh 资产，非 Blender 插件 |

## 为何值得保留

- 官方可训练入口；复现与二次开发的导航锚点。
- 明确 **部分开源** 边界，避免读者以为完整 Motion-DiT 管线已发布。
