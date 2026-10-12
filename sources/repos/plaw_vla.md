# PLaW-VLA GitHub 仓库归档

> 来源归档

- **标题：** RainyRobo/PLaW-VLA
- **类型：** repo
- **链接：** <https://github.com/RainyRobo/PLaW-VLA>
- **代码：** <https://github.com/RainyRobo/PLaW-VLA>
- **论文：** <https://arxiv.org/abs/2610.12285>
- **项目页：** <https://rainyrobo.github.io/PLaW-VLA/>（[归档](../sites/plaw-vla.md)）
- **许可：** Apache-2.0（另含 Gemma 许可文件 `LICENSE_GEMMA.txt`、`LICENSES/`）
- **入库日期：** 2026-10-12（浅克隆核查，tip 2026-10-09）
- **沉淀到 wiki：** [paper-plaw-vla-predictive-latent-world-model-vla.md](../../wiki/entities/paper-plaw-vla-predictive-latent-world-model-vla.md)

## 定位

基于 π0.5 代码路线的 PyTorch 实现：视觉语言、潜在世界模型、动作三专家 MoT，冻结 V-JEPA 2 作为未来表征编码器。

## 目录与入口（浅克隆核对）

| 路径 | 内容 |
|------|------|
| `src/plawvla/models_pytorch/` | `pi0_pytorch.py`、`world_model_pytorch.py` |
| `src/plawvla/models/` | `vjepa2.py`、`gemma.py`、`siglip.py` 等 |
| `src/plawvla/serving/`、`policies/` | 策略服务与各本体策略适配 |
| `scripts/train/` | `run_stage1_world_model_pretraining.sh`、`run_stage2_pretraining.sh`、`run_stage3_finetuning_libero.sh`、`run_stage3_libero_quickstart.sh`、`train_pytorch.py`、`compute_norm_stats.py` |
| `scripts/serve/` | `serve_policy.py`、`serve_policy_robotwin.py` |
| `scripts/data/` | agibotworld、egodex、intern_a1、lerobot、libero、robotwin 数据转换 |
| `examples/` | `libero`、`libero_plus`、`robotwin`、`simple_client` |
| `docs/` | `data.md`、`pretraining.md`、`norm_stats.md`、`remote_inference.md`、`benchmarks.md` |

## 运行方式（README）

- 环境：Linux x86_64、CUDA 12.8 驱动、Python 3.12，`uv sync --python 3.12 --frozen` 并打 transformers 补丁。
- Stage III LIBERO 快速上手：`NUM_GPUS=8 EXP_NAME=libero_finetune bash scripts/train/run_stage3_libero_quickstart.sh`（官方预训练检查点缺失时回退 π0.5 公开权重并初始化世界模型模块）。
- 完整训练：Stage I（100k 步，batch 512）→ Stage II（100k，256）→ Stage III（50k，256，warmup 10k）。
- 评测：`scripts/serve/serve_policy.py --env LIBERO policy:checkpoint ...` 起服务，再 `examples/libero/main.py --task-suite-name libero_spatial` 跑闭环；服务端返回带时间戳的绝对末端目标，客户端插值后转 robosuite OSC 指令。

## 发布状态（2026-10-12）

- 已发布：训练/评测/服务代码；LIBERO EEF 数据集（HF）。
- 待发布：官方 PLaW-VLA 预训练检查点（README 标 Pending）。
