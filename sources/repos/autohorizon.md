# AutoHorizon（hatchetProject/AutoHorizon）

- **标题:** AutoHorizon
- **链接:** https://github.com/hatchetProject/AutoHorizon
- **类型:** repo / vla / test-time / execution-horizon
- **许可:** Apache-2.0
- **项目页:** https://hatchetproject.github.io/autohorizon/
- **论文:** https://arxiv.org/abs/2602.21445
- **基座:** [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)（π0.5 PyTorch 移植 + LIBERO eval）
- **最后核查:** 2026-09-20
- **入库日期:** 2026-09-20

## 开源状态（步骤 2.5）

- **已开源：** 仓库含 AutoHorizon **Elastic** 实现、`run/serve/serve_libero_horizon.sh`、`run/eval/eval_libero_horizon.sh` 与多种 replanning baseline。
- **外部依赖：** OpenPI JAX checkpoint → 本地 `convert_jax_model_to_pytorch.py`；LIBERO 环境按 `examples/libero/README.md` 安装；需 patch `transformers==4.53.2`。
- **无官方 Hugging Face**（与用户核查一致）。

## 核心内容摘要

1. **Replanning 策略 CLI：** `--elastic`（AutoHorizon）、`--replan_steps N`、`--random`、`--action_trigger`、`--uncertainty`。
2. **超参入口：** `src/openpi/models_pytorch/pi0_pytorch.py` 中 `attn_step_count`、`hold_thr`、`max_entropy_q`；`pick_horizon_softpointer()` vs `bidir_soft_pointer()`。
3. **评测脚本：** 默认 LIBERO 全 benchmark × Static Oracle / Random / AutoHorizon，各 3 次重复（耗时长，可改脚本）。

## 对 wiki 的映射

- **wiki/entities/paper-autohorizon.md** — 论文实体
- **wiki/concepts/receding-horizon-policy-execution.md** — execution horizon 与滚动执行概念
- **wiki/entities/paper-pi05-open-world-vla.md** — 主评测 backbone
