# AutoLab-SAI-SJTU/GSR-ParaVLA

- **标题：** GSR / ParaVLA 官方实现
- **类型：** repo
- **URL：** <https://github.com/AutoLab-SAI-SJTU/GSR-ParaVLA>
- **许可：** MIT
- **配套论文：** [arXiv:2608.02497](https://arxiv.org/abs/2608.02497) — [`sources/papers/gsr_paravla_arxiv_2608_02497.md`](../papers/gsr_paravla_arxiv_2608_02497.md)
- **权重：** <https://huggingface.co/AutoLab-SJTU/GSR>（`manifest.json` 核查：`gated: false`，公开可拉取）
- **入库日期：** 2026-08-15
- **通读：** 已完成
- **核对范围：** `_tree.txt`（240 项）、`README.md`（397 行）、`recipes/{train_paravla,train_vla_adapter_gsr,train_smolvla_gsr,train_pi05_gsr,eval_lerobot_libero_para}.sh`、`lerobot_backend/src/lerobot/policies/{paravla/modeling_paravla.py,smolvla/modeling_smolvla.py,pi05/modeling_pi05.py}`、`lerobot_backend/src/lerobot/scripts/lerobot_train.py`、`lerobot_backend/experiments/gsr/eval_libero_para.py`、`vla_adapter_backend/{vla-scripts/train_gsr.py,prismatic/models/action_heads.py,experiments/libero/{run_libero_para_eval.py,gsr_runtime.py}}`。

## 一句话说明

完整训练与 LIBERO-Para（4,092 episodes）分片评测配方：ParaVLA、VLA-Adapter GSR、SmolVLA GSR、π₀.₅ GSR；配套公开 HF checkpoint。

## 仓库状态（2026-08-15 核查，源码已实读）

| 项 | 内容 |
|----|------|
| 环境与检查 | `scripts/create_envs.sh`（`environments/lerobot.yml`、`environments/vla_adapter.yml`）、`configs/paths.env`、`scripts/smoke_imports.py`、`Makefile`（`make smoke` / `make audit`） |
| LeRobot 训练入口 | `recipes/train_{paravla,smolvla_gsr,pi05_gsr}.sh` → `accelerate launch -m lerobot.scripts.lerobot_train`（含 `ParaVLAPolicy`、`SmolVLAPolicy`、`PI05Policy`） |
| VLA-Adapter 训练入口 | `recipes/train_vla_adapter_gsr.sh` → `torchrun vla-scripts/train_gsr.py`（`GSRRLDSBatchTransform` + `MLPResNetBlock_Pro` `use_semantic_sidecar=True` + gate 参数） |
| LIBERO-Para 评测 | `recipes/eval_lerobot_libero_para.sh` → `eval_libero_para.py` / `_libero_para_runtime.py`；`recipes/eval_vla_adapter_libero_para.sh` → `run_libero_para_eval.py` / `gsr_runtime.py`；最后跑 `scripts/verify_libero_para_coverage.py`（校验 4,092 独立 episode）与 `merge_libero_para.py` |
| 模型实现细节 | `modeling_paravla.py` 的 `ParaVLAPolicy` 代码另支持 `vision_encoder_type=lingbot` 选项；论文与发布配方默认 `facebook/dinov2-large` + `google-t5/t5-large` |

最短复现：按 README 建 conda 环境并校验 `smoke_imports.py` → `STAGE=smoke bash recipes/train_paravla.sh` → `bash recipes/eval_lerobot_libero_para.sh`。

## 与 wiki 的关系

- 实体页：[paper-gsr-paravla](../../wiki/entities/paper-gsr-paravla.md) — 含源码运行时序图。
