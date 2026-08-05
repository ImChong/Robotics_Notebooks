# kai0（OpenDriveLab / χ₀）

> 来源归档（ingest）

- **标题：** kai0（χ₀）
- **类型：** repo
- **链接：** <https://github.com/OpenDriveLab/kai0>
- **许可证：** Apache-2.0（另含 Gemma LICENSE）
- **项目页：** <https://mmlab.hk/research/kai0>
- **论文：** <https://arxiv.org/abs/2602.09021>
- **数据：** <https://huggingface.co/datasets/OpenDriveLab-org/Kai0>（CC-BY-NC-SA-4.0）· ModelScope `OpenDriveLab/Kai0`
- **权重：** <https://huggingface.co/OpenDriveLab-org/Kai0>（每任务 best model）· ModelScope 镜像
- **入库日期：** 2026-08-05
- **一句话说明：** 基于 [openpi](./openpi.md) 的 χ₀ 官方实现：全参微调 π₀/π₀.₅、Model Arithmetic 权重合并、Stage Advantage / AWBC、Heuristic DAgger + 时空增强 + temporal chunk-wise smoothing / RTC 推理；配套下载脚本与硬件 3D 打印文件。
- **开源状态：** **已开源（可运行）** — README Update（2026-02-10…15）将 MA / SA / TDA 标为 Released；`scripts/download_dataset.py` / `download_checkpoints.py` 拉取 HF 资产；子模块含 openpi 训练与 `serve_policy.py`。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-kai0.md`](../../wiki/entities/paper-kai0.md)

## 目录入口（对齐源码运行时序图）

| 路径 | 作用 |
|------|------|
| `scripts/download_dataset.py` / `download_checkpoints.py` | 拉 Task_A/B/C 数据与 best ckpt |
| `scripts/train.py` / `compute_norm_states_fast.py` | openpi 全参 fine-tune（如 `pi05_flatten_fold_normal`） |
| `model_arithmetic/` | JAX/PyTorch checkpoint soup（average / inverse_loss / GD / greedy…） |
| `stage_advantage/` | advantage 估计器、GT 标注、AWBC |
| `train_deploy_alignment/` | 数据增强、DAgger 采集、推理平滑/RTC |
| `scripts/serve_policy.py` | 策略服务；真机客户端见 `setup/` 与 TDA docs |
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
- [π₀ Policy](../../wiki/methods/π0-policy.md)
- [DAgger](../../wiki/methods/dagger.md)
