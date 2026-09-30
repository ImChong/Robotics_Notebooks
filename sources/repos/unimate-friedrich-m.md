# UniMate（Friedrich-M/UniMate）

> 来源归档

- **标题：** UniMate — [SIGGRAPH Asia 2026] One Unified Model to Animate Diverse Skeletons
- **类型：** repo
- **链接：** https://github.com/Friedrich-M/UniMate
- **许可证：** 见仓库 `LICENSE`
- **入库日期：** 2026-09-30
- **代码：** **已开源**（训练、推理、数据处理、配置与脚本）
- **权重 / 数据：** [Hugging Face UniMate](https://huggingface.co/Linzhan/UniMate)、[UniML3D 集合](https://huggingface.co/collections/Linzhan/unimate)
- **一句话说明：** **TADiT** flow-matching 实现：`unimate/` 含 denoiser（graph/full × adaln/cross_attn）、flow transport、数据集 mixture 与在线拓扑增广；`data_process/` 从 raw rig 到训练特征与 GLB/FBX 动画导出。
- **为什么值得保留：** 跨拓扑骨骼 **text-to-motion** 的可复现官方入口；与人体 SMPL 系 T2M 及机器人 **GMR** 上游形成对照。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-unimate.md`](../../wiki/entities/paper-unimate.md)

## README 要点（归纳）

- **环境：** conda `python=3.10` + `requirements.txt`（含 PyTorch / Accelerate / 渲染依赖）。
- **训练：** `accelerate launch -m unimate.training.train --config configs/uniml3d_60frames_graph_adaln.json`；输出 `outputs/<exp>/`（`config.json`、`dataset_stats.npy`、`checkpoints/`）。
- **推理：** `python -m unimate.inference.sample --exp_dir ... --test_cases_json ...`；可选 `motion_inbetweening` / `motion_expansion` / `motion_editing` 模块。
- **文本：** 默认 `google/flan-t5-base`；可 `unimate.tools.precompute_text_emb` 缓存 caption/joint 嵌入。
- **动画导出：** `scripts/run_animate_motion.sh` 将 `.npy` motion 特征驱动 rig → GLB/FBX。

## 目录结构（运行时）

| 路径 | 作用 |
|------|------|
| `unimate/training/train.py` | Accelerate 训练入口 |
| `unimate/inference/sample.py` | 文本条件采样主 CLI |
| `unimate/models/denoiser/` | TADiT 变体（graph_adaln、full_cross_attn 等） |
| `unimate/dataset/mixture/` | 多源混合、增广、collate |
| `configs/*.json` | 8 组论文对比配置（数据源 × attention × text_cond） |
| `data_process/` | UniML3D 构建五阶段 |

## 关联资料

- 项目页：[`sources/sites/unimate-linzhanmou.md`](../sites/unimate-linzhanmou.md)
- 论文归档：[`sources/papers/unimate_arxiv_2609_05415.md`](../papers/unimate_arxiv_2609_05415.md)
