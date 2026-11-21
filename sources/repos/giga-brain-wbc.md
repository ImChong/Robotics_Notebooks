# open-gigaai/giga-brain-wbc（GigaBrain-WBC-0.5 官方仓库）

> 来源归档（ingest · arXiv:2608.18234 配套官方实现）

- **标题：** GigaBrain-WBC-0.5 Official Repository
- **类型：** repo
- **链接：** <https://github.com/open-gigaai/giga-brain-wbc>
- **项目页：** <https://shepherd1226.github.io/gigabrain-wbc-0.5/>（[`gigabrain-wbc-0-5-github-io.md`](../sites/gigabrain-wbc-0-5-github-io.md)）
- **配套论文：** [GigaBrain-WBC-0.5（arXiv:2608.18234）](../papers/gigabrain_wbc_0_5_arxiv_2608_18234.md)
- **数据：** `open-gigaai/GigaBrain-WBC-0.5-TerrainData`（Hugging Face `bundles/data.tar` 459 MB）
- **许可证：** Apache-2.0
- **入库日期：** 2026-11-21
- **通读：** 已完成
- **核对范围：** 2026-11-21 通读 `README.md`、完整 530 文件目录树、`docs/training.md`、`giga_brain_wbc/train.py`、`giga_brain_wbc/learning/trainer/ppo_trainer_aux_loss.py`、`giga_brain_wbc/learning/modules/actor_critic_modules.py`、`giga_brain_wbc/export.py` 与 `evaluation/run.py`。
- **一句话说明：** GigaBrain-WBC-0.5 官方开源实现（530 文件），提供 Unitree G1（29 DoF）与 Maker L01 的 Isaac Lab 训练、ONNX 编码器/解码器导出、MuJoCo 同步评测与真机/PICO VR 部署。

## 仓库核心结构（2026-11-21 核查）

| 路径 | 职责 |
|------|------|
| `giga_brain_wbc/train.py` | Hydra 入口：构建 `ManagerBasedRLEnv` 封装、`Actor` / `Critic`、`TRLAuxLossPPOTrainer`（经 HuggingFace `accelerate` + `trl` 驱动分布式训练） |
| `giga_brain_wbc/learning/trainer/ppo_trainer_aux_loss.py` | `TRLAuxLossPPOTrainer`：在标准 PPO loss 上叠加 `UniversalTokenModule` 的重构与 latent 对齐辅助损失 |
| `giga_brain_wbc/learning/modules/` | `actor_critic_modules.py`、`transformer_decoder.py`、`universal_token_modules.py` |
| `giga_brain_wbc/envs/` | `tracking.py`、`terrain.py`、`motion_terrain.py`、`recovery/`（collision/gate/reset/sampling） |
| `giga_brain_wbc/scripts/` | `download_release.py`、`convert_bones_csv_to_motion_lib.py`、`check_data.py` |
| `giga_brain_wbc/export.py` | 将训练权重拆分为 `*_encoder.onnx` 与 `*_decoder.onnx`（cached Transformer，`g1_dyn`） |
| `evaluation/run.py` | MuJoCo 同步评测闭环，加载 `_encoder.onnx` + `_decoder.onnx` 对 `standard` / `terrain` / `fall` 等基准集评分 |
| `deploy/unitree_g1/` & `deploy/maker_l01/` | 实机控制与 PICO teleop + GMR 重定向适配 |

## 两阶段训练配方（`docs/training.md`）

1. 下载地形数据 `bundles/data.tar`（`download_release.py --training-data`），自 Bones-SEED CSV 转存平地动作并 `check_data.py` 校验。
2. 阶段 1：`exp=recovery`；阶段 2：从阶段 1 checkpoint 续训 `exp=terrain_recovery`（`tracking` 3072 envs，`terrain` 512 envs，推荐 24× RTX 5090）。

## 对 wiki 的映射

- [`wiki/entities/paper-gigabrain-wbc-0-5.md`](../../wiki/entities/paper-gigabrain-wbc-0-5.md)
