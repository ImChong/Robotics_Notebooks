# SOMA（edualvarado/SOMA）

- **标题**: SOMA: From Surface Observations to Muscle Anatomy
- **链接**: [https://github.com/edualvarado/SOMA](https://github.com/edualvarado/SOMA)
- **类型**: repo / training + dataset-tooling + evaluation
- **作者**: Eduardo Alvarado, Emily Kim, Gerrit Nolte, Friedemann Runte, Mario Botsch, Marc Habermann, Christian Theobalt（MPI-INF / TU Dortmund）
- **项目页**: [https://vcai.mpi-inf.mpg.de/projects/SOMA/](https://vcai.mpi-inf.mpg.de/projects/SOMA/)
- **论文**: arXiv:2606.09246 — [`sources/papers/soma_arxiv_2606_09246.md`](../papers/soma_arxiv_2606_09246.md)
- **数据集**: [https://gvv-assets.mpi-inf.mpg.de/soma](https://gvv-assets.mpi-inf.mpg.de/soma)
- **许可**: MIT
- **入库日期**: 2026-09-09
- **摘要**: 官方仓托管 **SOMA 全管线源码**（Suit marker 处理 → 规范模型 → LBS 注册 → Blender 残余/拉普拉斯工具 → U-Net 训练与生物力学评测）及 **SKIM 数据集构建脚本**（`Residuals-Python/`）；被试数据与 checkpoint **独立下载**。

> **命名消歧：** 非 NVIDIA [`soma_retargeter.md`](./soma_retargeter.md) / [`nvlabs-soma-x.md`](./nvlabs-soma-x.md) 生态。

## 开源状态（截至 2026-09-09）

| 项 | 状态 |
|----|------|
| **仓库** | `edualvarado/SOMA`（项目页 GitHub 按钮；默认分支 `master`） |
| **数据集 SKIM** | **已发布** — `gvv-assets.mpi-inf.mpg.de/soma`；S1–S5 自包含训练张量 |
| **训练入口** | **有** — `05-Training/01_end_to_end_training.py`（`ARCH`: linear / mlp / unet） |
| **评测** | **有** — `02_validate_training.py`、`03_evaluate_metrics.py`、`06-Evaluation/hit_bio_evaluation.py` |
| **数据预处理复现** | **有** — `01`–`04` 阶段 + `Residuals-Python/`；Blender 脚本需本机 Blender API |
| **Checkpoint** | **不入 git** — 训练产出本地 `runs/`；推理需自行训练或待官方权重链 |
| **结论** | **已开源（可训练 / 可复现管线）**；数据与权重需外站下载 |

## 目录要点

| 路径 | 作用 |
|------|------|
| `01-Suit-Processing/` | ArUco suit 2D 检测、三角化、3D 跟踪 |
| `02-Canonical-Model/` | 规范姿态 marker 模型、UV 标注 |
| `03-Registration/` | Marker LBS 权重（S1–S5 导出 JSON） |
| `04-Blender/` | 残余估计、拉普拉斯稠密形变、肌肉分离与可视化（`bpy`） |
| `05-Training/` | 端到端训练、验证、指标与动画可视化 |
| `06-Evaluation/` | 肌肉/皮肤穿透比、体积稳定性、SMPL 对齐 |
| `Residuals-Python/` | 无 Blender 的 SKIM 残余管线 + Viser viewer |

## 可运行入口（对齐 wiki 时序图）

1. `python -m venv env && pip install -r requirements.txt` + 按平台装 PyTorch
2. 下载 SKIM（S1–S5）→ 设 `PROCESSED_ROOT`（见 `05-Training/01_end_to_end_training.py`）
3. **训练：** `cd 05-Training && python 01_end_to_end_training.py`
4. **验证 / 指标：** `python 02_validate_training.py` → `03_evaluate_metrics.py`
5. **生物力学评测：** `python ../06-Evaluation/hit_bio_evaluation.py`
6. **数据浏览：** `Residuals-Python/` 内 Viser viewer（见该目录 README）

## 对 Wiki 的映射

- **wiki/entities/paper-soma-surface-observations-muscle-anatomy.md**：升格实体；`## 源码运行时序图` 对齐本页入口
- **交叉：** UMA / MAMMA（MPI 多视角人体捕获谱系）、SMPL 对齐评测脚本
