# ReViV（lvsean/reviv4d）

> 来源归档（repo）

- **标题：** ReViV: Reconstructing the Viewer and the View in 4D from Monocular Egocentric Video
- **类型：** repo / paper implementation
- **项目页：** <https://reviv4d.github.io/>
- **代码：** <https://github.com/lvsean/reviv4d>
- **论文：** <https://arxiv.org/abs/2607.17790>（ECCV 2026）
- **机构：** ETH Zürich · Delft University of Technology · Microsoft
- **入库日期：** 2026-09-07
- **一句话说明：** 基于 [4M](https://github.com/apple/ml-4m) any-to-any 框架扩展 [EgoM2P](https://github.com/ligengen/EgoM2P)，从单目 egocentric RGB 一次前向重建 depth / camera / gaze / body / hands；提供 `demo_infer.py`、`demo_hand.py`、`demo_vis.py` 与两套预训练权重。

## 开源状态（步骤 2.5，截至 2026-09-07）

- **已开源：** 官方 GitHub [lvsean/reviv4d](https://github.com/lvsean/reviv4d)；项目页互链。
- **许可：** 代码 Apache 2.0；模型权重 Sample Code License（**非商用**）。
- **权重：** `metric_depth/`（512×512 metric depth）与 `reviv_500b/`（256 relative depth）自 [polybox](https://polybox.ethz.ch/index.php/s/LHz64M2YnRo3CpL) 下载；需与对应 detokenizer + `norm_stats/` 成套使用。
- **训练：** 不附带数据集；需按 `README_DATA.md` 处理 clip 与 WebDataset shard；tokenizer 训练见 `run_training_vqvae.py`，主模型 `run_training_reviv.py`。
- **推理依赖：** Cosmos video tokenizer（HF gated）；`conda env create -f environment.yaml` → `reviv` 环境（Python 3.12, CUDA 12.4）。

## 核心入口（README）

| 脚本 | 作用 |
|------|------|
| `demo_infer.py` | RGB 2 s clip → depth / camera / gaze / body（`--targets` 可选子集） |
| `demo_hand.py` | RGB → 左右手 `[60,21,3]` 相机系关节 |
| `demo_vis.py` | viser 交互 3D：点云 + 相机 + 骨架 + 注视射线 |
| `demo_vis_hand.py` | 手部关节重投影叠加视频 |
| `cosmos_tokenizer/download_cosmos_tokenizer.py` | 下载 NVIDIA Cosmos tokenizer |

## 对 wiki 的映射

- [ReViV 论文实体页](../../wiki/entities/paper-reviv4d.md)
- [EgoM2P 策展索引](../../wiki/entities/paper-sa-2506-07886-egom2p-egocentric-multimodal-multitask-pretraini.md)
- [Macrodata egocentric hand action](../../wiki/methods/macrodata-egocentric-hand-action.md) — Dyn-HaMR / HaMeR 工程对照

## 当前提炼状态

- [x] 项目页 / GitHub / arXiv 入口与开源结论
- [x] 推理脚本与 checkpoint 套装说明
- [x] 升格 ReViV 深度论文实体页
