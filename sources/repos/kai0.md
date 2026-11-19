# kai0（OpenDriveLab / χ₀）

> 来源归档（ingest · 仓库源码核对）

- **标题：** kai0（χ₀）
- **类型：** repo
- **链接：** <https://github.com/OpenDriveLab/kai0>
- **许可证：** Apache-2.0（另含 Gemma LICENSE；HF/ModelScope 数据与权重为 CC-BY-NC-SA-4.0）
- **项目页：** <https://mmlab.hk/research/kai0>
- **论文：** <https://arxiv.org/abs/2602.09021>
- **数据：** <https://huggingface.co/datasets/OpenDriveLab-org/Kai0>（CC-BY-NC-SA-4.0）· ModelScope `OpenDriveLab/Kai0`
- **权重：** <https://huggingface.co/OpenDriveLab-org/Kai0>（每任务 best model）· ModelScope 镜像
- **入库日期：** 2026-08-05
- **最后核对：** 2026-11-19
- **核对范围：** 通读克隆仓库 `README.md`、训练与服务入口 `scripts/train.py` / `scripts/serve_policy.py`、`src/openpi/training/config.py`（Task A/B/C normal 与 AWBC 配置段）、`src/openpi/models/pi0.py`、`src/openpi/models_pytorch/pi0_pytorch.py`（`AdvantageEstimator`）、`model_arithmetic/arithmetic.py`、`stage_advantage/README.md` 与 `annotation/eval.py`、`discretize_advantage.py`、以及 `train_deploy_alignment/` 下 DAgger 采集（`agilex_openpi_dagger_collect.py`）与推理客户端（`agilex_inference_openpi_temporal_smoothing.py` `StreamActionBuffer`）；未在真机或 GPU 上执行训练推理。
- **通读：** 已完成
- **一句话说明：** 基于 [openpi](./openpi.md) 的 χ₀ 官方实现：全参微调 π₀/π₀.₅、Model Arithmetic 权重合并、Stage Advantage / AWBC、Heuristic DAgger + 时空增强 + temporal chunk-wise smoothing / RTC 推理；配套下载脚本与硬件 3D 打印文件。
- **开源状态：** **已开源（可运行）** — README Update（2026-02-10…15）将 MA / SA / TDA 标为 Released；`scripts/download_dataset.py` / `download_checkpoints.py` 拉取 HF 资产；子模块含 openpi 训练与 `serve_policy.py`（PyTorch 版 checkpoint 合并在 README 标明未充分测试）。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-kai0.md`](../../wiki/entities/paper-kai0.md)

## 目录入口与源码核对（对齐源码运行时序图）

| 路径 | 实读关键符号与作用 |
|------|------|
| `scripts/download_dataset.py` / `download_checkpoints.py` | 从 HF `OpenDriveLab-org/Kai0` 拉 Task_A/B/C 数据与 best ckpt |
| `scripts/compute_norm_states_fast.py` / `scripts/train.py` | openpi JAX 全参训练：`init_train_state` → `TrainState`；`src/openpi/training/config.py` 定义 `pi05_flatten_fold_normal` 及 `pi05_*_awbc`（Task A/B/C） |
| `model_arithmetic/arithmetic.py` | JAX Orbax/OCDBT checkpoint soup：支持 `average`、`inverse_loss`（`compute_checkpoint_losses`）、`gradient_descent`、`adaptive_gradient_descent` 与 `greedy`（`optimize_weights_greedy`）；CPU 混合参数 + GPU 算 `compute_loss`，并同步混合 `norm_stats.json` |
| `src/openpi/models_pytorch/pi0_pytorch.py` | `AdvantageEstimator(PI0Pytorch)`：在后缀状态 token 终层表示 `suffix_out_full[:, 0, :]` 接 3 层 SiLU+Tanh `value_head`，回归被截断至 `[-1, 1]` 的帧间进度差 `obs_full.progress`（`loss_value_weight=1.0`, `loss_action_weight=0.0`） |
| `stage_advantage/annotation/eval.py` & `discretize_advantage.py` | Step 0 手标 `stage_progress_gt` → Step 1 `scripts/train_pytorch.py` 训估计器 → Step 2 `eval.py`（`SimpleValueEvaluator`）产出 `relative_advantage` / `absolute_value` / `absolute_advantage` → Step 3 `discretize_advantage.py`（支持 `--stage-nums > 1` 分阶段百分位阈值，默认 `--threshold 30` 二值化写入 `task_index` 与 `meta/tasks.jsonl`）→ Step 4 `train.py pi05_*_awbc` |
| `train_deploy_alignment/data_augment/` | `time_scaling.py`、`space_mirroring.py`、`merge_lerobot.py`、`convert_h5_lerobot.py` |
| `train_deploy_alignment/dagger/` | `agilex/agilex_openpi_dagger_collect.py`（`SimpleDAggerCollector` + `CollectOperator`，ROS 键盘切换 DAgger 人工纠偏并存 HDF5）与 `arx/arx_openpi_dagger_collect.py` |
| `scripts/serve_policy.py` & `train_deploy_alignment/inference/` | GPU 端 `WebsocketPolicyServer` 对接机器人 IPC 客户端（如 `agilex_inference_openpi_temporal_smoothing.py` 中的 `StreamActionBuffer.integrate_new_chunk(actions_chunk, max_k, min_m)` 完成丢弃延迟旧步 + 线性重叠平滑，及 `arx_openpi_inference_rtc.py` RTC 模式） |
| `setup/README.md` | Agilex Piper / ARX X5、D435i、夹爪 3D 打印 |

## 算力 / 硬件（README）

| 模式 | 需求 |
|------|------|
| 推理 | >8 GB（例 RTX 4090） |
| 全参微调 | >70 GB（A100 80GB / H100）；文内实验 8×A100 |
| 真机 | Task A/B Agilex Piper；Task C ARX X5；双臂协作布局 |

## 对 wiki 的映射

- [χ₀ / kai0](../../wiki/entities/paper-kai0.md)
- [openpi](./openpi.md) — 底座仓
- [π₀ Policy](../../wiki/methods/pi0-policy.md)
- [DAgger](../../wiki/methods/dagger.md)
