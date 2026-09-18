# NPHM（SimonGiebenhain/NPHM 官方实现）

> 来源归档

- **标题：** Learning Neural Parametric Head Models (NPHM)
- **类型：** repo / training / fitting / evaluation
- **组织：** Simon Giebenhain（TUM / Nießner Lab）
- **代码：** <https://github.com/SimonGiebenhain/NPHM>
- **License：** GitHub 标注 **Other**（根目录未见 SPDX LICENSE 文件）
- **论文：** <https://arxiv.org/abs/2212.02761>
- **项目页：** <https://simongiebenhain.github.io/NPHM/>
- **入库日期：** 2026-09-18
- **一句话说明：** CVPR 2023 官方仓：两阶段训练 identity SDF + expression deformation、点云拟合推理、Chamfer/F-Score 评测；预训练与 demo 数据在 Google Drive。
- **沉淀到 wiki：** [NPHM（论文实体）](../../wiki/entities/paper-nphm.md)

---

## README 归纳（安装 → 训练 → 推理）

1. **环境：** conda `nphm` + Python 3.9；`pip install -e .`；GPU PyTorch（README 示例 1.13 / CUDA 11.6）。
2. **路径：** 编辑 `src/NPHM/env_paths.py`（数据集、checkpoint、输出）；可多机 `git rm --cached env_paths.py`。
3. **数据准备：** `sample_surface.py` + `sample_deformation_field.py` → 监督缓存约 **320GB**。
4. **Stage 1 身份几何：** `python scripts/training/train.py -cfg_file scripts/configs/nphm.yaml -local -exp_name ...`（NPM 用 `npm.yaml` 且无 `-local`）。
5. **Stage 2 表情形变：** `python scripts/training/train_corresp.py -cfg_file scripts/configs/nphm_def.yaml -exp_name ... -mode compress`。
6. **预训练：** [Google Drive](https://drive.google.com/drive/folders/1dajUVhnYgRxbmX9CpAXDw702YYb0VHm9?usp=sharing) → `env_paths.EXPERIMENT_DIR/`。
7. **拟合：** `python scripts/fitting/fitting_pointclouds.py -cfg_file scripts/configs/fitting_nphm.yaml ...`（`-demo` / `-sample` / `-resolution`）。
8. **评测：** `python scripts/evaluation/eval.py --results_dir FITTING_DIR/...`。

---

## 目录导航

| 路径 | 作用 |
|------|------|
| `scripts/training/` | Stage 1/2 训练入口 |
| `scripts/fitting/` | 点云拟合与随机采样 |
| `scripts/data_processing/` | 表面/形变场监督采样 |
| `scripts/evaluation/` | Chamfer / F-Score 等指标 |
| `scripts/configs/` | `nphm.yaml`、`nphm_def.yaml`、`fitting_nphm.yaml` 等 |
| `dataset/README.md` | 数据集说明与申请流程 |
| `src/NPHM/` | 核心模型与 `env_paths.py` |

---

## 开源边界

| 产物 | 状态 |
|------|------|
| 代码 + 配置 + 脚本 | **已开源** |
| 预训练权重 | **Google Drive 已发布** |
| Demo 数据 | **Google Drive 已发布** |
| 全量 5200+ 扫描 | **Form 申请** |
| MonoNPHM（单目跟踪） | **独立仓** [SimonGiebenhain/MonoNPHM](https://github.com/SimonGiebenhain/MonoNPHM) |

---

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [SHELLS](../../wiki/entities/paper-shells-layered-surface-sampling.md) | 多视角前馈固定拓扑人头；NPHM 为 **神经隐式 + 可拟合** 参数化路线 |
| [DynHair](../../wiki/entities/paper-dynhair.md) | 同 Nießner 系动态头发化身；NPHM 偏 **静态几何/表情** 基座 |
| Telepresence | 完整人头 morphable model 上游资产 |
